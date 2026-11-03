"""B2: offline cameras and LeRobot export on the SO-101 bench.

Scripted hands, clock and keys through teleop (the SO-101 expert is TODO(daniel)); every recording is labelled
scripted. No human data, no hardware, no ledger rows, no training result.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from xert.data.cameras import CAMERAS, TOLERANCE, CameraRenderer, camera_model, render_cameras

ROOT = Path(__file__).resolve().parents[1]
SO101_ASSETS = ROOT / "assets/menagerie/robotstudio_so101"


def test_so101_cameras_sit_on_the_gripper_and_keep_the_recorded_fingerprint():
    from xle_workbench.sim import ManipulationEnv
    def hashes():
        return {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in SO101_ASSETS.rglob("*") if path.is_file()}
    before = hashes()
    model, fingerprint = camera_model("so101")
    bench = ManipulationEnv({"robot": "so101"})
    assert fingerprint == bench.xml_sha256 == bench.metadata()["model_sha256"]  # What every recording stores.
    assert model.body(model.camera("wrist").bodyid[0]).name == "gripper" and model.camera("scene").bodyid[0] == 0
    # Menagerie's own wrist_cam stays in the model unrendered; the added cameras change no body, geom or qpos.
    assert {model.camera(i).name for i in range(model.ncam)} == {"scene", "wrist", "wrist_cam"}
    assert (model.nq, model.nbody, model.ngeom) == (bench.model.nq, bench.model.nbody, bench.model.ngeom)
    assert before == hashes()
    assert camera_model()[1] != fingerprint  # Per-robot fingerprints: XLeRobot data never renders on the SO-101.


@pytest.fixture(scope="module")
def gl():
    try:
        CameraRenderer(robot="so101").close()
    except Exception as exc:  # No offscreen GL: skip locally, as tests/test_cameras.py does; CI has osmesa.
        if os.environ.get("CI"):
            raise
        pytest.skip(f"No offscreen GL context: {exc!r}")


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    pytest.importorskip("lerobot")
    from scripts.record_scripted_runtime import ScriptedClock, ScriptedSource
    from test_so101_bench import PROFILE, SCRIPTED, SPEC, calibration
    from xert.retarget.calibrate import CalibratedMapper
    from xert.runtime import teleop
    clock = ScriptedClock()
    neutral = calibration(clock)
    keys = iter([" ", None, None, None, None, " ", None, "\x1b"])  # Start, four quiet ticks, stop, drain, quit.
    root = tmp_path_factory.mktemp("so101") / "session"
    teleop.run_session(ScriptedSource(clock), SPEC, root, lambda backend: CalibratedMapper(SPEC, PROFILE, neutral, backend),
                       provenance=SCRIPTED, keys=lambda: next(keys), clock=clock, sleep=clock.sleep)
    return root


def test_original_so101_recording_renders_exports_and_round_trips(gl, session, tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from xert.data.lerobot_io import export_lerobot
    from xert.runtime.teleop import TASK
    recording = session / "episode_000000"
    recorded = json.loads((recording / "xert_recording.json").read_text())
    assert recorded["embodiment"] == "so101" and recorded["source"]["kind"] == "scripted"
    rendered = render_cameras(recording, tmp_path / "rendered")
    assert rendered["episode_000000"]["frames"] == 5  # Streaming verify: source columns identical, PNGs lossless.
    exported = export_lerobot(recording, tmp_path / "exported", task=TASK, cameras=True)
    assert exported == rendered
    for output in (tmp_path / "rendered", tmp_path / "exported"):
        manifest = json.loads((output / "xert_cameras.json").read_text())
        assert manifest["model_sha256"] == recorded["source"]["bench"]["model_sha256"]
        assert {name: camera["body"] for name, camera in manifest["cameras"].items()} == {"scene": "world", "wrist": "gripper"}
        # P2.3 reconstruction check: grasp site, tray and block position/orientation from the rebuilt scene.
        errors = manifest["reconstruction_max_error"]
        assert set(errors) == {"grasp", "tray", "block_position", "block_orientation"}
        assert max(errors.values()) <= TOLERANCE
        print(output.name, "reconstruction max error", errors)
    original = LeRobotDataset("xert/runtime", root=recording)
    loaded = LeRobotDataset("xert/runtime", root=tmp_path / "exported")
    old, new = original.hf_dataset.data.table, loaded.hf_dataset.data.table
    assert all(new.column(column).equals(old.column(column)) for column in old.column_names)
    assert set(loaded.meta.features) - set(original.meta.features) == {"observation.images.scene",
                                                                     "observation.images.wrist"}
    assert loaded.meta.features["observation.images.wrist"]["dtype"] == "image"  # Lossless PNG, opt-in.
    frame = loaded[4]
    assert frame["observation.images.scene"].shape == (3, 128, 128) and loaded.meta.robot_type == "so101"
    del original, loaded


def test_so101_block_pose_check_refuses_a_scene_that_disagrees(gl, session, tmp_path, monkeypatch):
    render = CameraRenderer.render

    def shifted(self, joints, block):
        images, achieved = render(self, joints, block)
        achieved["block_position"] = achieved["block_position"] + 1e-4  # Ten times the tolerance.
        return images, achieved

    monkeypatch.setattr(CameraRenderer, "render", shifted)
    with pytest.raises(ValueError, match="disagrees"):
        render_cameras(session / "episode_000000", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_frame_pair_import_carries_the_renderer_columns_and_model(session, tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from xert.data.lerobot_io import import_lerobot
    recording = session / "episode_000000"
    record = import_lerobot(recording, tmp_path / "import")
    dataset = LeRobotDataset("xert/runtime", root=recording)
    features = dataset.meta.features
    del dataset
    assert record["source_kind"] == "xert_recording" and record["embodiment"] == "so101"
    assert record["columns"]["environment_state"] == features["observation.environment_state"]["names"]
    assert record["columns"]["state"] == features["observation.state"]["names"]
    assert record["model_sha256"] == camera_model("so101")[1]  # Same scene the renderer would rebuild.
    with np.load(tmp_path / "import/frames.npz") as frames:
        assert frames["environment_state"].shape == (5, len(record["columns"]["environment_state"]))


def test_so101_npz_export_with_cameras_rebuilds_the_bench_scene(gl, session, tmp_path):
    from test_lerobot_export import so101_bench_episode
    from xert.data.lerobot_io import export_lerobot
    _, identifier, _ = so101_bench_episode(tmp_path / "source/dataset")
    (tmp_path / "source/collection.json").write_text(json.dumps({"episode_ids": [identifier]}))
    result = export_lerobot(tmp_path / "source/collection.json", tmp_path / "export", task="hold",
                            cameras=True, robot_type="so101")
    [verification] = result.values()
    assert verification["frames"] == 3
    manifest = json.loads((tmp_path / "export/xert_cameras.json").read_text())
    assert max(manifest["reconstruction_max_error"].values()) <= TOLERANCE
    assert manifest["cameras"]["wrist"]["body"] == CAMERAS["so101"]["wrist"]["body"] == "gripper"

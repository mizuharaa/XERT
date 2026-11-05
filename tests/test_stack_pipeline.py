"""Slice H plumbing: the stack task through rollouts, live/offline rendering, recording, quality and the ledger.

Scripted hands, scripted clock, constant-action policy stubs and temporary ledgers only: no human data, no
trained policy and no number. Tiny fixtures (3-frame recordings, 64 px renders, one seed).
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import mujoco
import numpy as np
import pytest

from xert.backends.stack import StackBackend
from xert.eval import ledger
from xert.spec import load_spec
from xert.tasks import environment_names

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "xert/spec/xlerobot_left.yaml"
PROFILE = ROOT / "configs/teleop/xlerobot_bench.json"
SCRIPTED = {"kind": "scripted", "human_demonstrations": False, "generator": "tests/test_stack_pipeline.py"}
PERIOD = 1 / 25


class ConstantPolicy:
    """Holds one normalized action every tick (the gate harness's policy interface)."""
    def __init__(self, action):
        self.action, self.inputs = np.asarray(action, dtype=np.float32), []

    def reset(self):
        self.inputs.clear()

    def act(self, observation):
        self.inputs.append(observation)
        return self.action


def timed(monkeypatch):
    """Record (sim time before, after) of every StackBackend step taken through the harnesses."""
    times, send = [], StackBackend.send_targets

    def wrapped(self, q, t_cmd=None):
        before = self.data.time
        send(self, q, t_cmd)
        times.append((before, self.data.time, t_cmd))
    monkeypatch.setattr(StackBackend, "send_targets", wrapped)
    return times


def hold_reset_action(spec):
    from xle_workbench.sim import ROBOTS
    low, high = spec.position_limits.T
    return 2 * (np.asarray(ROBOTS[spec.name]["reset_pose"]) - low) / (high - low) - 1


def test_gate_rollout_runs_a_full_stack_episode_with_a_constant_policy(monkeypatch):
    from xert.eval import gate_experiment as gx
    spec = load_spec(SPEC)
    times = timed(monkeypatch)
    names = [f"{joint}.pos" for joint in spec.joint_names] + environment_names("stack")
    policy = ConstantPolicy(hold_reset_action(spec))
    trial = gx.rollout(policy, spec, 10000, names, "stack")
    assert trial == {"seed": 10000, "success": False, "steps": 600, "failure_reason": "timeout_no_lift"}
    assert len(times) == 600 and all(t is None for _, _, t in times)  # Policy rollouts pass t_cmd=None.
    np.testing.assert_allclose([after - before for before, after, _ in times], PERIOD, rtol=0, atol=1e-9)
    assert len(policy.inputs) == 600 and policy.inputs[0].shape == (len(names),)
    base = names.index("base_block.x")
    assert policy.inputs[0][base] != 0 and policy.inputs[-1][names.index("time_fraction")] == np.float32(599 / 600)


def test_image_act_evaluate_rolls_out_stack_with_live_renders(monkeypatch, tmp_path):
    pytest.importorskip("lerobot")
    import torch
    from lerobot.policies.act import modeling_act
    from xert.data.cameras import CAMERAS, camera_model
    from xert.eval import image_act
    spec = load_spec(SPEC)
    times = timed(monkeypatch)
    image = {"dtype": "image", "shape": (64, 64, 3), "names": ["height", "width", "channels"]}
    features = {"observation.images.scene": image, "observation.images.wrist": image,
                "observation.state": {"dtype": "float32", "shape": (6,), "names": [f"{j}.pos" for j in spec.joint_names]},
                "observation.environment_state": {"dtype": "float32", "shape": (21,), "names": environment_names("stack")}}
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / image_act.METADATA).write_text(json.dumps({
        "robot": "xlerobot_left", "task_id": "stack", "seed": 0, "bench_seeds": [0], "source_kinds": ["scripted"],
        "action_units": "radians", "git_dirty": False, "dataset_sha256": "0" * 64, "features": features,
        "camera": {"size": 64, "model_sha256": camera_model("xlerobot_left", "stack")[1],
                   "cameras": CAMERAS["xlerobot_left"]}}))
    from xle_workbench.sim import ROBOTS
    hold, seen = torch.tensor([ROBOTS["xlerobot_left"]["reset_pose"]], dtype=torch.float32), []

    class Stub:  # Constant-action policy: no weights, no training.
        config = None

        def eval(self):
            return self

        def reset(self):
            pass

        def select_action(self, frame):
            seen.append({key: tuple(value.shape) for key, value in frame.items()})
            return hold

    monkeypatch.setattr(modeling_act.ACTPolicy, "from_pretrained", lambda *args, **kwargs: Stub())
    monkeypatch.setattr("lerobot.policies.factory.make_pre_post_processors",
                        lambda *args, **kwargs: (lambda frame: frame, lambda action: action))
    try:
        report = image_act.evaluate(checkpoint, [10000], tmp_path / "report.json")
    except Exception as exc:
        if "GL" in repr(exc) and not os.environ.get("CI"):
            pytest.skip(f"No offscreen GL context: {exc!r}")
        raise
    assert report["task_id"] == "stack" and report["n"] == 1 and report["successes"] == 0
    assert report["trials"] == [{"seed": 10000, "success": False, "steps": 600, "failure_reason": "timeout_no_lift"}]
    assert report["ledger"].startswith("no rows: training data is ['scripted']") and ledger.read_rows() == []
    np.testing.assert_allclose([after - before for before, after, _ in times], PERIOD, rtol=0, atol=1e-9)
    assert len(times) == len(seen) == 600 and seen[0] == {
        "observation.images.scene": (3, 64, 64), "observation.images.wrist": (3, 64, 64),
        "observation.state": (6,), "observation.environment_state": (21,)}


def test_live_and_offline_renders_are_pixel_equal_with_base_block_displaced():
    """Review H-03: live image-ACT frames and render-cameras' offline rebuild set base_block the same way."""
    pytest.importorskip("torch")
    from xert.data.cameras import CameraRenderer, _columns, row_state, scene_objects
    from xert.eval import image_act
    backend = StackBackend(load_spec(SPEC), 0)
    base = backend.model.joint("base_block_joint").qposadr[0]
    default = backend.data.qpos[base:base + 7].copy()
    backend.data.qpos[base:base + 3] += [.012, -.006, 0]  # Away from its seeded pose.
    # Recordings store float32; float32-exact joints make the live float64 state the recorded one bit for bit.
    backend.data.qpos[backend.qids] = backend.data.qpos[backend.qids].astype(np.float32)
    mujoco.mj_forward(backend.model, backend.data)
    try:
        renderer = CameraRenderer(64, "xlerobot_left", "stack")
    except Exception as exc:
        if os.environ.get("CI"):
            raise
        pytest.skip(f"No offscreen GL context: {exc!r}")
    try:
        # MSAA off for this comparison only: with MuJoCo's default 4 offscreen samples, re-rendering one identical
        # state on the Windows GPU differed in 3-4 of 40 frames (pick_place and stack alike; 0 of 40 without MSAA),
        # which is driver noise, not a path difference. The path under test is the same with either setting.
        renderer.renderer.close()
        renderer.model.vis.quality.offsamples = 0
        renderer.renderer = mujoco.Renderer(renderer.model, 64, 64)
        image = {"dtype": "image", "shape": (64, 64, 3)}
        live = image_act.live_frame(backend, renderer, {"observation.images.scene": image,
                                                        "observation.images.wrist": image})
        joints = [f"{name}.pos" for name in backend.spec.joint_names]
        recorded = {"observation.state": {"names": joints}, "observation.environment_state": {"names": backend.object_names}}
        frame = {"observation.state": backend.read_state().q.astype(np.float32),
                 "observation.environment_state": np.asarray(backend.object_state(), dtype=np.float32)}
        state = row_state(frame, _columns(recorded, renderer.joints, renderer.bodies))
        offline, achieved = renderer.render(state["joints"], scene_objects(state, renderer.bodies))
        for key, pixels in offline.items():
            np.testing.assert_array_equal(live[key].mul(255).round().byte().permute(1, 2, 0).numpy(), pixels)
        np.testing.assert_allclose(achieved["base_block_position"], backend.data.body("base_block").xpos, atol=1e-6)
        state["base_block"] = default
        undisplaced, _ = renderer.render(state["joints"], scene_objects(state, renderer.bodies))
        assert not np.array_equal(undisplaced["observation.images.scene"], offline["observation.images.scene"])
    finally:
        renderer.close()


# ---------------------------------------------------------------- Recordings (LeRobot environment).

def teleop_session(root, task_id, frames=3):
    from scripts.record_scripted_runtime import ScriptedClock, ScriptedSource, scripted_frame
    from xert.backends.base import task_backend
    from xert.retarget.calibrate import CalibratedMapper, create_calibration
    from xert.runtime import teleop
    clock = ScriptedClock()
    neutral = create_calibration(SPEC, PROFILE, scripted_frame(0, clock()), task_backend(load_spec(SPEC), 0, task_id))
    keys = iter([" ", *[None] * (frames - 1), " ", None, "\x1b"])
    return teleop.run_session(ScriptedSource(clock), SPEC, root, lambda b: CalibratedMapper(SPEC, PROFILE, neutral, b),
                              provenance=SCRIPTED, keys=lambda: next(keys), clock=clock, sleep=clock.sleep,
                              task_id=task_id)


@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    pytest.importorskip("lerobot")
    root = tmp_path_factory.mktemp("stack")
    return {task: teleop_session(root / task, task)[0] for task in ("stack", "pick_place")}


def strip_task_id(path, manifest="xert_recording.json"):
    data = json.loads((path / manifest).read_text())
    del data["task_id"]
    (path / manifest).write_text(json.dumps(data))


def test_stack_teleop_recording_round_trips_with_success_and_both_blocks(sessions):
    from xert.data.cameras import camera_model
    from xert.data.recorder import load_runtime_dataset, verify_recording
    from xert.runtime import teleop
    recording = sessions["stack"]
    assert verify_recording(recording)["frames"] == 3
    manifest = json.loads((recording / "xert_recording.json").read_text())
    assert manifest["task_id"] == "stack" and manifest["task"] == teleop.TASK_TEXT["stack"]
    assert manifest["source"]["bench"]["model_sha256"] == camera_model("xlerobot_left", "stack")[1]
    assert manifest["source"]["bench"]["success_rule"]["settle_ticks"] == 12
    journal = [json.loads(line)["runtime"] for line in (recording / "xert_frames.jsonl").read_text().splitlines()]
    assert {frame["task_id"] for frame in journal} == {"stack"}
    dataset = load_runtime_dataset(recording)
    assert dataset.features["observation.environment_state"]["names"] == environment_names("stack")
    assert dataset.features["next.success"]["dtype"] == "bool" and not any(dataset[i]["next.success"].item() for i in range(3))
    assert dataset[0]["task"] == teleop.TASK_TEXT["stack"]
    del dataset


def gl_render(*args, **kwargs):
    from xert.data.cameras import render_cameras
    try:
        return render_cameras(*args, **kwargs)
    except Exception as exc:
        if "GL" in repr(exc) and not os.environ.get("CI"):
            pytest.skip(f"No offscreen GL context: {exc!r}")
        raise


def test_render_cameras_and_export_take_the_stack_scene_from_the_recording(sessions, tmp_path):
    from xert.data.lerobot_io import export_lerobot
    from xert.runtime import teleop
    gl_render(sessions["stack"], tmp_path / "rendered", size=64)
    manifest = json.loads((tmp_path / "rendered/xert_cameras.json").read_text())
    assert manifest["task_id"] == "stack" and "base_block" in manifest["reconstruction"]
    errors = manifest["reconstruction_max_error"]
    assert {"base_block_position", "base_block_orientation"} <= set(errors) and max(errors.values()) <= manifest["tolerance"]
    export_lerobot(sessions["stack"], tmp_path / "export", task=teleop.TASK_TEXT["stack"])
    assert json.loads((tmp_path / "export/xert_recording.json").read_text())["task_id"] == "stack"


def test_legacy_recording_without_task_id_renders_as_pick_place_and_keeps_its_task_text(sessions, tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from xert.runtime import teleop
    legacy = shutil.copytree(sessions["pick_place"], tmp_path / "legacy")
    strip_task_id(legacy)
    gl_render(legacy, tmp_path / "rendered", size=64)
    assert json.loads((tmp_path / "rendered/xert_cameras.json").read_text())["task_id"] == "pick_place"
    assert LeRobotDataset("xert/runtime", root=tmp_path / "rendered")[0]["task"] == teleop.TASK
    # A stack recording stripped of its task_id has a non-bench fingerprint: refused, never guessed as pick_place.
    unknown = shutil.copytree(sessions["stack"], tmp_path / "unknown")
    strip_task_id(unknown)
    with pytest.raises(ValueError, match="not guessed"):
        gl_render(unknown, tmp_path / "refused", size=64)
    assert not (tmp_path / "refused").exists()


def test_legacy_p2_1_export_without_task_id_renders_with_its_task_text(tmp_path):
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from test_lerobot_io import SOURCE, TASK, copy_collection
    from xert.data.lerobot_io import export_lerobot
    ids = json.loads((SOURCE / "collection.json").read_text())["episode_ids"][:1]
    export_lerobot(copy_collection(tmp_path / "source", ids), tmp_path / "export", task=TASK)
    assert json.loads((tmp_path / "export/xert_export.json").read_text())["task_id"] == "pick_place"
    strip_task_id(tmp_path / "export", "xert_export.json")
    gl_render(tmp_path / "export", tmp_path / "rendered", size=64)
    assert json.loads((tmp_path / "rendered/xert_cameras.json").read_text())["task_id"] == "pick_place"
    assert LeRobotDataset("xert/runtime", root=tmp_path / "rendered")[0]["task"] == TASK


def test_quality_refuses_a_session_mixing_task_ids(sessions, tmp_path, monkeypatch):
    from xert.__main__ import main
    mixed = tmp_path / "mixed"
    for index, task in enumerate(("pick_place", "stack")):
        shutil.copytree(sessions[task], mixed / f"episode_{index:06d}")
    monkeypatch.setattr(sys, "argv", ["xert", "quality", "--dataset", str(mixed), "--output", str(tmp_path / "out")])
    with pytest.raises(ValueError, match="mix task_id"):
        main()
    assert not (tmp_path / "out").exists()
    monkeypatch.setattr(sys, "argv", ["xert", "quality", "--dataset", str(sessions["stack"]), "--output",
                                      str(tmp_path / "stack")])
    main()
    assert json.loads((tmp_path / "stack/metrics.json").read_text())["task_id"] == "stack"


# ---------------------------------------------------------------- Ledger (no LeRobot).

def row(**fields):
    return ledger.new_row(git_commit="0" * 40, git_dirty=False, slice="H", experiment="test", backend="mujoco",
                          embodiment="xlerobot_left", metric="success_rate", value=0.0, ci95=None, n=1,
                          seeds={"train": [], "eval": [0]}, config_path="c", artifact_paths=["a"], notes="test row",
                          **fields)


def test_real_ledger_history_reads_and_takes_an_optional_task_id(tmp_path):
    real = ROOT / "results/ledger.jsonl"
    digest = hashlib.sha256(real.read_bytes()).hexdigest()
    history = ledger.read_rows(real)
    assert len(history) == 64 and not any("task_id" in r for r in history)
    copy = tmp_path / "ledger.jsonl"
    shutil.copyfile(real, copy)
    ledger.append_row(row(task_id="stack"), copy)
    latency = row()  # Like runtime/cli.py latency and scripts/measure_ik.py rows: no task_id.
    latency.update(slice="P3.6", metric="latency_p50_ms", value=12.5)
    ledger.append_row(latency, copy)
    rows = ledger.read_rows(copy)
    assert len(rows) == 66 and rows[-2]["task_id"] == "stack" and "task_id" not in rows[-1]
    with pytest.raises(ValueError, match="task_id"):
        ledger.validate_row(row(task_id="stacking"))
    assert hashlib.sha256(real.read_bytes()).hexdigest() == digest  # The real history is never written.


def test_rows_differing_only_in_task_id_both_survive_resume_numbers():
    sys.path.insert(0, str(ROOT))
    from scripts.resume_numbers import eligible
    pick, stack = row(task_id="pick_place"), row(task_id="stack")
    stack["time_utc"] = pick["time_utc"]
    kept, superseded, repeats = eligible([pick, stack])
    assert {r["id"] for r in kept} == {pick["id"], stack["id"]} and not superseded and not repeats

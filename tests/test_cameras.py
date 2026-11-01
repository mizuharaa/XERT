"""P2.3 offline cameras on scripted bench data only: no human data, no ledger rows, no training result."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from xert.data.cameras import CameraRenderer, camera_model

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets/xlerobot"


@pytest.fixture(scope="module")
def renderer():
    try:
        renderer = CameraRenderer()
    except Exception as exc:  # No display/osmesa/EGL: skip locally, never open a visible window.
        if os.environ.get("CI"):
            raise  # CI installs osmesa; a broken GL stack must fail, not skip.
        pytest.skip(f"No offscreen GL context: {exc!r}")
    yield renderer
    renderer.close()


def test_cameras_are_added_in_memory_and_vendored_assets_stay_pinned():
    import mujoco
    from xle_workbench.sim import model_xml
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in ASSETS.iterdir()}
    model, fingerprint = camera_model()
    plain = mujoco.MjModel.from_xml_string(model_xml())
    assert {model.camera(i).name for i in range(model.ncam)} == {"scene", "wrist"} and plain.ncam == 0
    assert model.body(model.camera("wrist").bodyid[0]).name == "Fixed_Jaw" and model.camera("scene").bodyid[0] == 0
    # Same bodies, geoms and qpos layout: the cameras change nothing the physics or a recording sees.
    assert (model.nq, model.nbody, model.ngeom) == (plain.nq, plain.nbody, plain.ngeom)
    after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in ASSETS.iterdir()}
    assert before == after
    source = json.loads((ASSETS / "SOURCE.json").read_text())
    assert all(after[name] == digest for name, digest in source["sha256"].items())
    assert len(fingerprint) == 64


def test_scene_state_is_found_by_name_in_both_dataset_layouts():
    from xert.data.cameras import BLOCK, _columns
    from xle_workbench.sim import JOINTS
    tail = [*BLOCK, "grasp.x", "grasp.y", "grasp.z", "tray.x", "tray.y", "tray.z"]
    export = {"observation.state": {"names": [f"joint_position.{name}" for name in JOINTS] + tail}}
    recording = {"observation.state": {"names": [f"{name}.pos" for name in JOINTS]},
                 "observation.environment_state": {"names": tail}}
    assert _columns(export)["block"][0] == ("observation.state", 6)
    assert _columns(recording)["joints"][5] == ("observation.state", 5)
    assert _columns(recording)["tray"][2] == ("observation.environment_state", 12)
    with pytest.raises(ValueError, match="lacks"):
        _columns({"observation.state": recording["observation.state"]})


def test_render_is_deterministic_for_a_fixed_state(renderer):
    joints, block = np.array([0, .3, .5, .2, 0, .4]), np.array([.24, -.03, .096, 1, 0, 0, 0])
    first, _ = renderer.render(joints, block)
    renderer.render(joints + .2, block + [0, .05, 0, 0, 0, 0, 0])
    again, achieved = renderer.render(joints, block)
    for key, image in first.items():
        assert image.shape == (128, 128, 3) and image.dtype == np.uint8
        np.testing.assert_array_equal(image, again[key])
    np.testing.assert_allclose(achieved["tray"], [.23, .11, .08])
    np.testing.assert_allclose(achieved["block_position"], block[:3])


def test_sites_are_hidden(renderer):
    import mujoco
    images, _ = renderer.render(np.array([0, .3, .5, .2, 0, .4]), np.array([.24, -.03, .096, 1, 0, 0, 0]))
    # The grasp marker sits in front of the wrist camera; MuJoCo's default options would draw it.
    renderer.renderer.update_scene(renderer.data, camera="wrist", scene_option=mujoco.MjvOption())
    assert not np.array_equal(renderer.renderer.render(), images["observation.images.wrist"])


def test_block_orientation_error_treats_q_and_minus_q_as_equal():
    from xert.data.cameras import _error
    q = np.array([.5, .5, .5, .5])
    assert _error("block_orientation", q, -q) == 0 and _error("block_position", q, -q) == 1


@pytest.fixture(scope="module")
def rendered(renderer, tmp_path_factory):
    pytest.importorskip("lerobot")
    from test_gate_experiment import scripted_session
    from xert.data.cameras import render_cameras
    root = tmp_path_factory.mktemp("cameras")
    session = scripted_session(root / "session", [3, 4])
    return session, render_cameras(session, root / "rendered"), root / "rendered"


def test_rendered_session_keeps_every_source_byte_and_round_trips(rendered):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    session, result, output = rendered
    assert set(result) == {"episode_000000", "episode_000001"}
    for name in result:
        source, copy = session / name, output / name
        info = json.loads((source / "meta/info.json").read_text())
        assert not any(f["dtype"] in ("image", "video") for f in info["features"].values())  # Recorder default.
        for path in source.iterdir():  # Provenance sidecars are copied unchanged.
            if path.is_file():
                assert (copy / path.name).read_bytes() == path.read_bytes()
        manifest = json.loads((copy / "xert_cameras.json").read_text())
        assert manifest["source_sha256"]["xert_recording.json"] == hashlib.sha256(
            (source / "xert_recording.json").read_bytes()).hexdigest()
        errors = manifest["reconstruction_max_error"]
        assert set(errors) == {"grasp", "tray", "block_position", "block_orientation"}
        assert max(errors.values()) <= manifest["tolerance"]
        original = LeRobotDataset("xert/runtime", root=source)
        loaded = LeRobotDataset("xert/runtime", root=copy)
        old, new = original.hf_dataset.data.table, loaded.hf_dataset.data.table
        assert all(new.column(column).equals(old.column(column)) for column in old.column_names)
        assert set(loaded.meta.features) - set(original.meta.features) == {"observation.images.scene",
                                                                         "observation.images.wrist"}
        frame = loaded[0]
        assert frame["observation.images.wrist"].shape == (3, 128, 128) and frame["task"] == original[0]["task"]
        del original, loaded


def test_refuses_a_dataset_from_another_model(rendered, tmp_path):
    import shutil
    from xert.data.cameras import render_cameras
    session = rendered[0]
    copy = shutil.copytree(session / "episode_000000", tmp_path / "episode_000000")
    manifest = json.loads((copy / "xert_recording.json").read_text())
    manifest["source"]["bench"]["model_sha256"] = "0" * 64
    (copy / "xert_recording.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="pinned bench model"):
        render_cameras(copy, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_official_act_trains_a_step_on_the_rendered_images(rendered):
    import torch
    from lerobot.configs.types import FeatureType, NormalizationMode
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.utils.feature_utils import dataset_to_policy_features
    torch.manual_seed(0)
    root = rendered[2] / "episode_000001"
    meta = LeRobotDataset("xert/runtime", root=root).meta
    features = dataset_to_policy_features(meta.features)
    chunk = 2
    # Tiny smoke settings (seconds); not an experiment and no success number.
    config = ACTConfig(
        input_features={key: f for key, f in features.items()
                        if key.startswith("observation.images.") or key == "observation.state"},
        output_features={"action": features["action"]}, chunk_size=chunk, n_action_steps=1, device="cpu",
        pretrained_backbone_weights=None, dim_model=32, n_heads=2, dim_feedforward=32, n_encoder_layers=1,
        n_decoder_layers=1, n_vae_encoder_layers=1, latent_dim=4,
        normalization_mapping={key: NormalizationMode.IDENTITY for key in ("VISUAL", "STATE", "ACTION")})
    assert set(config.image_features) == {"observation.images.scene", "observation.images.wrist"}
    assert all(f.type == FeatureType.VISUAL and f.shape == (3, 128, 128) for f in config.image_features.values())
    delta = {"action": [i / meta.fps for i in range(chunk)]}  # Action chunk targets, padded at the episode end.
    dataset = LeRobotDataset("xert/runtime", root=root, delta_timestamps=delta)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
    policy = ACTPolicy(config)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    loss, _ = policy({key: batch[key] for key in (*config.input_features, "action", "action_is_pad")})
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()

"""Slice L (LeRobot env): `xert add-pose` on scripted and fixture bench data.

Tiny fixtures only (the first 3 steps of one real P1.2 episode, a 3-step SO-101 hold, 3-frame scripted teleop
recordings, 64 px renders): no human data, no ledger rows, no result.
"""
import json
import os
from pathlib import Path
import shutil
import sys

import mujoco
import numpy as np
import pytest

pytest.importorskip("lerobot")
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from xert.data import lerobot_io as io
from xert.data.cameras import CameraRenderer, _read_table, camera_model
from xert.data.pose import (COLUMNS, FEATURE, ROOTS, add_pose, base_transform, clip_gripper, gripper_fraction,
                            site_pose)
from xert.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]


def spec_of(robot):
    return load_spec(ROOT / f"xert/spec/{robot}.yaml")


def fk(robot, task_id, q):
    """One 7-vector from joint radians, through the pose helpers (test_pose_fk.py checks those against MuJoCo)."""
    model, _ = camera_model(robot, task_id)
    spec = spec_of(robot)
    qids = [model.joint(name).qposadr[0] for name in spec.joint_names]
    pose, _ = site_pose(model, mujoco.MjData(model), qids, model.site("grasp").id, base_transform(model, robot), q)
    grip = clip_gripper(gripper_fraction(q[spec.joint_names.index(spec.gripper["joint"])], spec.gripper))[0]
    return np.append(pose, grip).astype(np.float32)


def column(table, name):
    return np.asarray(table.column(name).to_pylist())


@pytest.fixture(scope="module")
def exports(tmp_path_factory):
    """Fresh short P2.1 pick-and-place exports of both bench robots."""
    from test_lerobot_export import so101_bench_episode
    from test_lerobot_io import TASK, tiny_collection
    root = tmp_path_factory.mktemp("pose-exports")
    collection, _ = tiny_collection(root / "store-xlerobot_left")  # Corrective labels differ from executed.
    io.export_lerobot(collection, root / "xlerobot_left", task=TASK)
    _, identifier, _ = so101_bench_episode(root / "store-so101/dataset")
    (root / "store-so101/collection.json").write_text(json.dumps({"episode_ids": [identifier]}))
    io.export_lerobot(root / "store-so101/collection.json", root / "so101", task="hold", robot_type="so101")
    return {robot: root / robot for robot in ROOTS}


@pytest.fixture(scope="module")
def posed(exports, tmp_path_factory):
    root = tmp_path_factory.mktemp("posed")
    return {robot: (add_pose(path, root / robot), root / robot) for robot, path in exports.items()}


@pytest.mark.parametrize("robot", ROOTS)
def test_export_gains_two_pose_columns_and_keeps_every_source_column(exports, posed, robot):
    (result, output), source = posed[robot], exports[robot]
    manifest = json.loads((output / "xert_pose.json").read_text())
    assert result[source.name]["frames"] == 3 and manifest["verification"]["grasp_world_max_error_m"] <= 1e-5
    assert manifest["pose"] == "kinematic model pose of the simulated arm" and manifest["hardware"] is False
    assert manifest["root_body"] == ROOTS[robot][0] and manifest["site"] == "grasp" and manifest["task_id"] == "pick_place"
    assert manifest["source_sha256"] == io._hashes(source) and manifest["model_sha256"] == camera_model(robot)[1]
    old, new = _read_table(source), _read_table(output)
    # Every source column, the AUTOMATIC ones (timestamp, indices) included, byte-equal.
    assert all(new.column(name).equals(old.column(name)) for name in old.column_names)
    assert set(new.column_names) - set(old.column_names) == set(COLUMNS)
    for path in source.iterdir():
        if path.is_file():
            assert (output / path.name).read_bytes() == path.read_bytes()
    loaded, original = LeRobotDataset("xert/runtime", root=output), LeRobotDataset("xert/runtime", root=source)
    info = json.loads((output / "meta/info.json").read_text())
    for key in COLUMNS:  # Literal contract, not the module's own constant.
        assert info["features"][key] == {"dtype": "float32", "shape": [7],
                                         "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]}
        assert str(new.schema.field(key).type.value_type) == "float"  # Arrow float32 in the Parquet.
        assert loaded.features[key] == FEATURE and loaded[0][key].shape == (7,)
        assert str(loaded[0][key].dtype) == "torch.float32"
    assert loaded[0]["task"] == original[0]["task"]
    del loaded, original
    # Recorded world grasp.x/y/z, rebuilt from the stored base-frame xyz through the fixed base transform.
    state_names = json.loads((source / "meta/info.json").read_text())["features"]["observation.state"]["names"]
    recorded = column(old, "observation.state")[:, [state_names.index(f"grasp.{axis}") for axis in "xyz"]]
    model = camera_model(robot)[0]
    position, rotation = base_transform(model, robot)
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, rotation)
    world = column(new, "observation.state.cartesian")[:, :3] @ matrix.reshape(3, 3).T + position
    assert np.max(np.abs(world - recorded)) <= 1e-5
    joints = column(old, "observation.state")[:, :6].astype(np.float32)
    np.testing.assert_array_equal(column(new, "observation.state.cartesian").astype(np.float32),
                                  np.stack([fk(robot, "pick_place", q.astype(np.float64)) for q in joints]))
    low, high = np.asarray(manifest["action"]["action_joint_ranges_radians"]).T
    decoded = low + (column(old, "action").astype(np.float32).astype(np.float64) + 1) / 2 * (high - low)
    np.testing.assert_array_equal(column(new, "action.cartesian").astype(np.float32),
                                  np.stack([fk(robot, "pick_place", q) for q in decoded]))


def test_action_is_decoded_with_the_recorded_ranges_not_the_spec_limits(exports, tmp_path):
    # Both benches record ranges equal to their spec limits, so only changed ranges tell the two apart.
    source = shutil.copytree(exports["so101"], tmp_path / "ranges")
    changed = [[-1., 1.], [-1., 1.], [-1., 1.], [-1., 1.], [-1., 1.], [0., 1.]]
    edit(source / "xert_export.json", lambda m: [episode["simulation"].update(action_joint_ranges_radians=changed)
                                                  for episode in m["source_episodes"]])
    add_pose(source, tmp_path / "out")
    table = _read_table(tmp_path / "out")
    low, high = np.asarray(changed).T
    decoded = low + (column(table, "action").astype(np.float32).astype(np.float64) + 1) / 2 * (high - low)
    np.testing.assert_array_equal(column(table, "action.cartesian").astype(np.float32),
                                  np.stack([fk("so101", "pick_place", q) for q in decoded]))


def test_gripper_beyond_the_spec_interval_is_clipped_and_counted_in_the_dataset(exports, tmp_path, monkeypatch):
    import dataclasses
    from xert import spec as specs
    real = specs.load_spec

    def narrowed(path, **kwargs):  # The so101 hold keeps its gripper at 1.4 rad; "open" moves to 1.0.
        loaded = real(path, **kwargs)
        return dataclasses.replace(loaded, gripper={**loaded.gripper, "open": 1.0})
    monkeypatch.setattr(specs, "load_spec", narrowed)
    add_pose(exports["so101"], tmp_path / "out")
    table, manifest = _read_table(tmp_path / "out"), json.loads((tmp_path / "out/xert_pose.json").read_text())
    for key in COLUMNS:
        gripper = column(table, key)[:, 6]
        assert np.all((gripper >= 0) & (gripper <= 1))
        clips = manifest["gripper"]["clips"][key]
        assert clips["clipped_frames"] == 3 and clips["max_excursion"] == pytest.approx(.4, abs=1e-6)


def test_action_cartesian_is_fk_of_the_decoded_training_labels_not_the_executed_stream(exports, posed):
    from xle_workbench.sim import ManipulationEnv
    from xle_workbench.storage import Store
    source, (_, output) = exports["xlerobot_left"], posed["xlerobot_left"]
    manifest = json.loads((output / "xert_pose.json").read_text())
    table = _read_table(output)
    labels, executed = column(table, "action"), column(table, "xert.executed_action")
    assert not np.array_equal(labels, executed)  # Unequal corrective and executed streams.
    # Store actions are normalized; the recorded ranges are the Store's own and the bench's ctrlrange.
    store = Store(source.parent / "store-xlerobot_left/dataset")
    [identifier] = json.loads((source.parent / "store-xlerobot_left/collection.json").read_text())["episode_ids"]
    ranges = store.load_training_episode(identifier)[1]["simulation"]["action_joint_ranges_radians"]
    assert manifest["action"]["action_joint_ranges_radians"] == ranges
    env = ManipulationEnv({"robot": "xlerobot_left"})
    np.testing.assert_array_equal(env.limits, ranges)
    low, high = np.asarray(ranges).T
    decoded = low + (labels.astype(np.float32).astype(np.float64) + 1) / 2 * (high - low)
    np.testing.assert_allclose(env.normalize(decoded), labels, rtol=0, atol=1e-6)
    cartesian = column(table, "action.cartesian").astype(np.float32)
    np.testing.assert_array_equal(cartesian, np.stack([fk("xlerobot_left", "pick_place", q) for q in decoded]))
    from_executed = low + (executed.astype(np.float32).astype(np.float64) + 1) / 2 * (high - low)
    assert not np.array_equal(cartesian, np.stack([fk("xlerobot_left", "pick_place", q) for q in from_executed]))
    assert manifest["action"]["action_semantics"] == json.loads((source / "xert_export.json").read_text())[
        "action_semantics"]
    assert "no Cartesian column" in manifest["action"]["executed_action"]


# ---------------------------------------------------------------- Teleop recordings: pick_place, stack, drawer.

@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    from test_stack_pipeline import teleop_session
    root = tmp_path_factory.mktemp("pose-sessions")
    return {task: teleop_session(root / task, task)[0] for task in ("pick_place", "stack", "drawer")}


@pytest.mark.parametrize("task_id", ["pick_place", "stack", "drawer"])
def test_recordings_of_every_task_are_accepted_with_radian_actions(sessions, tmp_path, task_id):
    from xert.runtime import teleop
    recording = sessions[task_id]
    export = tmp_path / "export"
    io.export_lerobot(recording, export, task=teleop.TASK_TEXT[task_id])  # P4.1 export: a verified copy.
    result = add_pose(export, tmp_path / "posed")
    manifest = json.loads((tmp_path / "posed/xert_pose.json").read_text())
    assert manifest["task_id"] == task_id and result["export"]["grasp_world_max_error_m"] <= 1e-5
    assert manifest["model_sha256"] == camera_model("xlerobot_left", task_id)[1]
    assert manifest["action"]["decode"] == "none: radians as recorded"
    table = _read_table(tmp_path / "posed")
    expected = np.stack([fk("xlerobot_left", task_id, q.astype(np.float64)) for q in column(table, "action")])
    np.testing.assert_array_equal(column(table, "action.cartesian").astype(np.float32), expected)
    if task_id == "pick_place":  # A session directory of episode_* recordings works too.
        assert set(add_pose(recording.parent, tmp_path / "session")) == {recording.name}
        assert (tmp_path / f"session/{recording.name}/xert_pose.json").exists()


# ---------------------------------------------------------------- Refusals: nothing is written.

def refused(source, output, message, error=ValueError):
    with pytest.raises(error, match=message):
        add_pose(source, output)
    assert not output.exists() or error is FileExistsError


def edit(path, change):
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def test_refusals_of_simulated_inputs(exports, posed, sessions, tmp_path):
    source = exports["xlerobot_left"]
    (tmp_path / "exists").mkdir()
    refused(source, tmp_path / "exists", "exists", FileExistsError)
    assert list((tmp_path / "exists").iterdir()) == []
    refused(posed["xlerobot_left"][1], tmp_path / "out", "already carries Cartesian")
    missing = shutil.copytree(source, tmp_path / "missing")
    edit(missing / "meta/info.json", lambda info: info["features"]["observation.state"]["names"].__setitem__(
        0, "joint_position.renamed"))
    refused(missing, tmp_path / "out", r"lacks \['joint_position.Rotation_L'\]")
    no_grasp = shutil.copytree(source, tmp_path / "no-grasp")
    edit(no_grasp / "meta/info.json", lambda info: info["features"]["observation.state"]["names"].__setitem__(
        info["features"]["observation.state"]["names"].index("grasp.x"), "grasp.renamed"))
    refused(no_grasp, tmp_path / "out", r"lacks \['grasp.x'\]")
    swapped = shutil.copytree(source, tmp_path / "swapped")  # Two joint columns read in each other's place.

    def swap(info):
        names = info["features"]["observation.state"]["names"]
        names[1], names[2] = names[2], names[1]
    edit(swapped / "meta/info.json", swap)
    refused(swapped, tmp_path / "out", "FK grasp position disagrees")
    assert not any(path.name.startswith(".xert-pose-") for path in tmp_path.iterdir())  # Staging removed too.
    other = shutil.copytree(source, tmp_path / "other")
    edit(other / "xert_export.json", lambda m: m["source_episodes"][0]["simulation"].update(model_sha256="0" * 64))
    refused(other, tmp_path / "out", "not recorded on the current xlerobot_left pick_place bench model")
    mixed = shutil.copytree(source, tmp_path / "mixed")

    def second_episode(manifest):
        episode = json.loads(json.dumps(manifest["source_episodes"][0]))
        episode["simulation"]["action_joint_ranges_radians"][0] = [-2, 2]
        manifest["source_episodes"].append(episode)
    edit(mixed / "xert_export.json", second_episode)
    refused(mixed, tmp_path / "out", "mix action_joint_ranges_radians")
    two = tmp_path / "two-robots"
    shutil.copytree(sessions["pick_place"], two / "episode_000000")
    shutil.copytree(sessions["pick_place"], two / "episode_000001")
    edit(two / "episode_000001/xert_recording.json", lambda m: m.update(embodiment="so101"))
    refused(two, tmp_path / "out", "one bench robot")
    tasks = tmp_path / "two-tasks"
    shutil.copytree(sessions["pick_place"], tasks / "episode_000000")
    shutil.copytree(sessions["stack"], tasks / "episode_000001")
    refused(tasks, tmp_path / "out", "one bench task per run")


def test_hardware_imports_are_refused_with_the_reason(tmp_path):
    from test_lerobot_io import hardware_import, synthetic_hardware
    synthetic_hardware(tmp_path / "raw")  # A raw so101_follower dataset: no XERT manifest.
    refused(tmp_path / "raw", tmp_path / "out", "slice L2")
    imported = hardware_import(tmp_path / "hw")  # Path (c) import, then the derived Store.
    refused(imported, tmp_path / "out", "so101_follower dataset has no XERT manifest")
    io.to_store(imported, tmp_path / "store")
    refused(tmp_path / "store", tmp_path / "out", "to-store writes derived_frame_pairs")


# ---------------------------------------------------------------- export-lerobot -> add-pose -> render-cameras.

class NoMsaa(CameraRenderer):
    """MSAA off: re-rendering one state with 4 offscreen samples is not bit-stable on the Windows GPU
    (docs/handoff/H-stack-bench.md); the two renders compared below must be."""
    def __init__(self, *args):
        super().__init__(*args)
        self.renderer.close()
        self.model.vis.quality.offsamples = 0
        self.renderer = mujoco.Renderer(self.model, self.size, self.size)


@pytest.fixture(scope="module")
def chain(exports, posed, tmp_path_factory):
    from xert.data import cameras
    root = tmp_path_factory.mktemp("pose-chain")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cameras, "CameraRenderer", NoMsaa)
        try:
            cameras.render_cameras(exports["xlerobot_left"], root / "plain", size=64)
        except Exception as exc:
            if "GL" in repr(exc) and not os.environ.get("CI"):
                pytest.skip(f"No offscreen GL context: {exc!r}")
            raise
        cameras.render_cameras(posed["xlerobot_left"][1], root / "chained", size=64)
    return root


def test_render_cameras_carries_the_pose_columns_through_byte_equal(posed, chain):
    posed_table, plain, chained = _read_table(posed["xlerobot_left"][1]), _read_table(chain / "plain"), \
        _read_table(chain / "chained")
    assert all(chained.column(name).equals(posed_table.column(name)) for name in posed_table.column_names)
    for key in ("observation.images.scene", "observation.images.wrist"):
        assert chained.column(key).equals(plain.column(key))
    info = json.loads((chain / "chained/meta/info.json").read_text())
    assert all(info["features"][key]["shape"] == [7] and info["features"][key]["dtype"] == "float32" for key in COLUMNS)
    assert (chain / "chained/xert_pose.json").exists()


def test_image_bearing_input_is_refused(chain, tmp_path):
    refused(chain / "plain", tmp_path / "out", "run add-pose before render-cameras")


def test_cli_registers_add_pose(exports, tmp_path, monkeypatch, capsys):
    from xert.__main__ import main
    monkeypatch.setattr(sys, "argv", ["xert", "add-pose", "--dataset", str(exports["so101"]), "--output",
                                      str(tmp_path / "cli")])
    main()
    assert json.loads(capsys.readouterr().out)["so101"]["frames"] == 3

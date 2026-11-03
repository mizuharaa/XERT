"""export-lerobot / import-lerobot / to-store. Scripted and synthetic data only: no hardware, no ledger rows."""
import gc
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from xert.data import lerobot_io as io

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/p1.2/recovery_integrated"
TASK = "Move the block to the tray"


def info_only(root, **info):
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"codebase_version": "v3.0", **info}))
    return root


def test_dispatch_is_by_manifest_and_unknown_sources_are_refused(tmp_path):
    for robot_type in io.HARDWARE:  # 0.6.1 lerobot-record writes "so_follower"; older releases "so101_follower".
        assert io.source_kind(info_only(tmp_path / robot_type, robot_type=robot_type)) == "so101_follower"
    export = info_only(tmp_path / "export", robot_type="so101_follower")
    (export / "xert_export.json").write_text("{}")
    assert io.source_kind(export) == "xert_export"  # The XERT manifest wins over robot_type.
    recording = info_only(tmp_path / "recording", robot_type="so101")
    (recording / "xert_recording.json").write_text("{}")
    assert io.source_kind(recording) == "xert_recording"
    (recording / "xert_export.json").write_text("{}")
    with pytest.raises(ValueError, match="Ambiguous"):
        io.source_kind(recording)
    with pytest.raises(ValueError, match="Unknown source"):
        io.source_kind(info_only(tmp_path / "koch", robot_type="koch_follower"))
    with pytest.raises(ValueError, match="v3.0"):
        io.source_kind(info_only(tmp_path / "old", codebase_version="v2.1", robot_type="so101_follower"))
    with pytest.raises(ValueError, match="Unknown source"):
        io.import_lerobot(tmp_path / "koch", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_to_store_refuses_raw_units_and_non_frame_pair_imports(tmp_path):
    for name, record in {"raw": {"derived": False, "source_kind": "so101_follower", "units": "raw_lerobot"},
                         "store": {"derived": False, "source_kind": "xert_export", "units": "xert_store"},
                         "derived": {"derived": True, "source_kind": "so101_follower", "units": "radians"}}.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "import.json").write_text(json.dumps(record))
        with pytest.raises(ValueError, match="raw_lerobot" if name == "raw" else "frame-pair import"):
            io.to_store(tmp_path / name, tmp_path / f"{name}-out")
        assert not (tmp_path / f"{name}-out").exists()


# ---- LeRobot 0.6.1 environment from here on ----

@pytest.fixture
def lerobot():
    return pytest.importorskip("lerobot")


def copy_collection(root, ids):
    """A collection.json beside a dataset/ Store holding byte copies of real P1.2 episodes."""
    for identifier in ids:
        shutil.copytree(SOURCE / "dataset/episodes" / identifier, root / "dataset/episodes" / identifier)
    (root / "collection.json").write_text(json.dumps({"episode_ids": ids}))
    return root / "collection.json"


def test_a_npz_export_import_restores_both_streams_and_terminal_obs(lerobot, tmp_path):
    from xle_workbench.storage import Store
    ids = json.loads((SOURCE / "collection.json").read_text())["episode_ids"][:2]
    collection = copy_collection(tmp_path / "source", ids)
    result = io.export_lerobot(collection, tmp_path / "export", task=TASK)
    assert result["max_abs_error"] == 0
    record = io.import_lerobot(tmp_path / "export", tmp_path / "import")
    assert record["source_kind"] == "xert_export" and [m["source_id"] for m in record["episodes"]] == ids
    source, restored = Store(tmp_path / "source/dataset"), Store(tmp_path / "import/dataset")
    assert json.loads((tmp_path / "import/collection.json").read_text())["episode_ids"] == \
        [m["id"] for m in record["episodes"]]
    for mapping in record["episodes"]:
        for load in ("load_episode", "load_training_episode"):  # Executed stream, then corrective labels.
            before, old = getattr(source, load)(mapping["source_id"])
            after, new = getattr(restored, load)(mapping["id"])
            assert set(before) == set(after)
            for key in before:  # observations include the terminal one: T+1 rows.
                assert before[key].dtype == after[key].dtype
                np.testing.assert_array_equal(before[key], after[key])
        assert old["training_action_source"] == new["training_action_source"] == "expert_correction"
        assert {k: v for k, v in old.items() if k not in io.DERIVED_KEYS} == \
            {k: v for k, v in new.items() if k not in io.DERIVED_KEYS}
    assert io._hashes(tmp_path / "import/raw") == io._hashes(tmp_path / "export")
    with pytest.raises(FileExistsError):
        io.import_lerobot(tmp_path / "export", tmp_path / "import")


@pytest.fixture
def bench_recording(lerobot, tmp_path):
    from test_gate_experiment import scripted_session
    return scripted_session(tmp_path / "session", [3]) / "episode_000000"


def test_b_runtime_recording_imports_float32_exact_frame_pairs(bench_recording, tmp_path):
    from xert.data.recorder import load_runtime_dataset
    record = io.import_lerobot(bench_recording, tmp_path / "import")
    manifest = json.loads((bench_recording / "xert_recording.json").read_text())
    dataset = load_runtime_dataset(bench_recording)
    expected = {key: np.stack([np.asarray(v) for v in values]) for key, values in dataset.hf_dataset[:].items()
                if key in ("observation.state", "action", "observation.environment_state", "timestamp")}
    names = dataset.features["observation.environment_state"]["names"]
    del dataset
    gc.collect()
    with np.load(tmp_path / "import/frames.npz") as frames:
        frames = dict(frames)
    assert set(frames) == {"episode_index", "frame_index", "timestamp", "state", "action", "environment_state"}
    for ours, theirs in (("state", "observation.state"), ("action", "action"),
                         ("environment_state", "observation.environment_state"), ("timestamp", "timestamp")):
        assert frames[ours].dtype == np.float32
        np.testing.assert_array_equal(frames[ours], expected[theirs])
    np.testing.assert_array_equal(frames["frame_index"], np.arange(3))
    assert record["columns"]["environment_state"] == names and "block.x" in names
    assert record["model_sha256"] == manifest["source"]["bench"]["model_sha256"]
    assert record["units"] == "radians" and "never synthesized" in record["absent"]
    assert "xert.next_observation" in record["not_imported"]
    assert io._hashes(tmp_path / "import/raw") == io._hashes(bench_recording)
    # Derived store: T_e - 1 transitions, normalized by spec limits, reward 0, truncated not terminated.
    from xle_workbench.storage import Store
    from xert.spec import load_spec
    stored = io.to_store(tmp_path / "import", tmp_path / "store")
    arrays, metadata = Store(tmp_path / "store/dataset").load_episode(stored["episode_ids"][0])
    limits = load_spec(ROOT / "xert/spec/xlerobot_left.yaml").position_limits
    # Owner decision: path (b) observations are the joint state, then environment_state in recorded order.
    np.testing.assert_array_equal(arrays["observations"], np.concatenate([frames["state"], frames["environment_state"]], 1))
    assert metadata["observation_names"] == record["columns"]["state"] + names
    assert metadata["observation_contents"] == "joint state + environment_state"
    np.testing.assert_allclose((arrays["actions"] + 1) / 2 * (limits[:, 1] - limits[:, 0]) + limits[:, 0],
                               frames["action"][:-1], atol=1e-6)
    assert not arrays["rewards"].any() and not arrays["terminated"].any() and metadata["derived"] is True


def test_b_recording_export_copies_and_refuses_relabeling(bench_recording, tmp_path):
    from xert.runtime.teleop import TASK as RECORDED
    assert io.export_lerobot(bench_recording, tmp_path / "copy", task=RECORDED) == {"recordings": 1, "copied": True}
    assert io._hashes(tmp_path / "copy") == io._hashes(bench_recording)
    with pytest.raises(ValueError, match="relabelled"):
        io.export_lerobot(bench_recording, tmp_path / "relabel", task="Something else")
    assert not (tmp_path / "relabel").exists()


# TEST FIXTURE: made-up joint-map values, not measured on any arm. Body joints in degrees, gripper 0-100.
FIXTURE_SIGN = [1, -1, 1, -1, 1, 1]
FIXTURE_OFFSET = [2.5, -10.0, 7.25, 0.0, -3.5, 9.0]


def fixture_map(path):
    raw = json.loads(io.JOINT_MAP.read_text())
    raw["status"] = "TEST FIXTURE: made-up values, not measured; never use for hardware"
    for entry, sign, offset in zip(raw["joints"], FIXTURE_SIGN, FIXTURE_OFFSET):
        gripper = entry["units"] == "range_0_100"
        entry.update(sign=sign, units_per_rad=50.0 if gripper else 180 / np.pi, zero_offset=offset,
                     min=0.0 if gripper else -180.0, max=100.0 if gripper else 180.0, max_speed=90.0, max_accel=900.0)
    path.write_text(json.dumps(raw))
    return path


def synthetic_hardware(root, robot_type="so101_follower"):
    """SYNTHETIC so101_follower-format dataset (scripted values): two episodes, different start poses."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.feature_utils import hw_to_dataset_features
    motors = {f"{name}.pos": float for name in
              ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")}
    features = {**hw_to_dataset_features(motors, "observation"), **hw_to_dataset_features(motors, "action")}
    dataset = LeRobotDataset.create(repo_id="synthetic/so101", root=root, robot_type=robot_type, fps=30,
                                    features=features, use_videos=False)
    episodes = []
    # Exact 0.0 is what LeRobot writes when a servo reads its mid position (motors_bus.py:877); elbow_flex has
    # the fixture offset 7.25, where float64 cancellation leaves ~1e-15 instead of 0 on re-encoding.
    for start, length in ((np.float32([0, -20, 0, 5, 0, 10]), 3),(np.float32([12.5, 3, -25, -8, 30, 60]), 4)):
        state = start + np.arange(length, dtype=np.float32)[:, None] * np.float32(0.087890625)  # One STS3215 step.
        action = state + np.float32(0.5)
        for row in range(length):
            dataset.add_frame({"observation.state": state[row], "action": action[row], "task": "synthetic"})
        dataset.save_episode(parallel_encoding=False)
        episodes.append((state, action))
    dataset.finalize()
    del dataset
    gc.collect()
    return episodes


def test_c_hardware_import_reencodes_and_tbd_map_is_refused(lerobot, tmp_path):
    episodes = synthetic_hardware(tmp_path / "hw")
    with pytest.raises(ValueError, match="TBD"):  # The real, unmeasured P6 map.
        io.import_lerobot(tmp_path / "hw", tmp_path / "refused", motor_units="degrees")
    joint_map = fixture_map(tmp_path / "fixture_map.json")
    with pytest.raises(ValueError, match="--motor-units degrees"):  # use_degrees is never recorded: no default.
        io.import_lerobot(tmp_path / "hw", tmp_path / "refused", joint_map=joint_map)
    assert not (tmp_path / "refused").exists()
    record = io.import_lerobot(tmp_path / "hw", tmp_path / "import", joint_map=joint_map, motor_units="degrees")
    assert record["units"] == "radians" and record["joint_map"]["sha256"] == io.sha256(joint_map)
    assert record["motor_units"] == "degrees" and record["use_degrees_assumed"] is True
    with np.load(tmp_path / "import/frames.npz") as frames:
        frames = dict(frames)
    sign, offset = np.asarray(FIXTURE_SIGN, float), np.asarray(FIXTURE_OFFSET, float)
    scale = np.asarray([180 / np.pi] * 5 + [50.0])
    raw_state = np.concatenate([state for state, _ in episodes])
    raw_action = np.concatenate([action for _, action in episodes])
    assert raw_state[0, 2] == 0.0 and offset[2] == 7.25
    for key, raw in (("state", raw_state), ("action", raw_action)):
        assert frames[key].dtype == np.float64
        # Within 1e-9 motor units, far below one servo step (0.088 deg); exact zeros may come back as ~1e-15.
        np.testing.assert_allclose(sign * scale * frames[key] + offset, raw, rtol=0, atol=1e-9)
    # Two independent episodes; LeRobot's timestamps restart at 0.
    np.testing.assert_array_equal(frames["episode_index"], [0, 0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(frames["frame_index"], [0, 1, 2, 0, 1, 2, 3])
    np.testing.assert_allclose(frames["timestamp"], np.array([0, 1, 2, 0, 1, 2, 3]) / 30)
    from xle_workbench.storage import Store
    stored = io.to_store(tmp_path / "import", tmp_path / "store")
    store = Store(tmp_path / "store/dataset")
    lengths = []
    for episode, identifier in enumerate(stored["episode_ids"]):
        arrays, metadata = store.load_episode(identifier)
        rows = frames["episode_index"] == episode
        assert metadata["episode_index"] == episode
        # Successors come from the same episode only: no transition from episode 0's last frame to episode 1.
        np.testing.assert_array_equal(arrays["observations"], frames["state"][rows].astype(np.float32))
        assert metadata["observation_contents"] == "joint state only"  # Path (c) has no environment_state.
        lengths.append(len(arrays["actions"]))
    assert lengths == [2, 3]
    # --raw-units needs no map, keeps native values, and to-store refuses it.
    raw = io.import_lerobot(tmp_path / "hw", tmp_path / "raw-import", raw_units=True)
    with np.load(tmp_path / "raw-import/frames.npz") as frames:
        np.testing.assert_array_equal(frames["state"], raw_state)
    assert raw["units"] == "raw_lerobot"
    with pytest.raises(ValueError, match="raw_lerobot"):
        io.to_store(tmp_path / "raw-import", tmp_path / "raw-store")


def test_c_joint_map_units_must_be_degrees_and_range_0_100(lerobot, tmp_path):
    synthetic_hardware(tmp_path / "hw")
    joint_map = json.loads(fixture_map(tmp_path / "map.json").read_text())
    for units, message in (("range_m100_100", "degrees for the body"), ("radians", "MotorNormMode")):
        joint_map["joints"][1]["units"] = units
        (tmp_path / "map.json").write_text(json.dumps(joint_map))
        with pytest.raises(ValueError, match=message):
            io.import_lerobot(tmp_path / "hw", tmp_path / "out", joint_map=tmp_path / "map.json", motor_units="degrees")
    assert not (tmp_path / "out").exists()


def hardware_import(tmp_path):
    synthetic_hardware(tmp_path / "hw")
    io.import_lerobot(tmp_path / "hw", tmp_path / "import", joint_map=fixture_map(tmp_path / "map.json"),
                      motor_units="degrees")
    return tmp_path / "import"


def test_to_store_refuses_out_of_limit_actions(lerobot, tmp_path):
    synthetic_hardware(tmp_path / "hw")
    joint_map = json.loads(fixture_map(tmp_path / "map.json").read_text())
    joint_map["joints"][5]["zero_offset"] = 200.0  # Gripper 0-100 now maps below its spec limit.
    (tmp_path / "map.json").write_text(json.dumps(joint_map))
    io.import_lerobot(tmp_path / "hw", tmp_path / "import", joint_map=tmp_path / "map.json", motor_units="degrees")
    with pytest.raises(ValueError, match="outside spec position limits"):
        io.to_store(tmp_path / "import", tmp_path / "store")
    assert not (tmp_path / "store").exists()


def test_to_store_limit_slack_is_1e_6_rad(lerobot, tmp_path):
    from xert.spec import load_spec
    root = hardware_import(tmp_path)
    upper = load_spec(ROOT / "xert/spec/so101.yaml").position_limits[0, 1]
    with np.load(root / "frames.npz") as frames:
        frames = dict(frames)
    for excess, accepted in ((0.5e-6, True), (2e-6, False)):
        frames["action"][0, 0] = upper + excess  # A non-final action, so to-store uses it.
        np.savez(root / "frames.npz", **frames)
        output = tmp_path / f"store-{excess}"
        if accepted:
            io.to_store(root, output)
        else:
            with pytest.raises(ValueError, match="outside spec position limits"):
                io.to_store(root, output)


def test_to_store_refuses_a_changed_spec(lerobot, tmp_path):
    root = hardware_import(tmp_path)
    record = json.loads((root / "import.json").read_text())
    record["spec_sha256"] = "0" * 64
    (root / "import.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="current spec differs"):
        io.to_store(root, tmp_path / "store")


def rewrite(root, change):
    """Rewrite a fixture dataset's single data Parquet file in place (tests only)."""
    import pyarrow.parquet as pq
    [path] = sorted(root.glob("data/**/*.parquet"))
    table = pq.read_table(path)
    pq.write_table(change(table), path)


def test_import_sorts_rows_and_refuses_gaps(lerobot, tmp_path):
    import pyarrow as pa
    root = hardware_import(tmp_path)
    shutil.copytree(tmp_path / "hw", tmp_path / "reversed")
    rewrite(tmp_path / "reversed", lambda table: table.take(pa.array(range(table.num_rows - 1, -1, -1))))
    io.import_lerobot(tmp_path / "reversed", tmp_path / "reversed-import", joint_map=tmp_path / "map.json",
                      motor_units="degrees")
    with np.load(root / "frames.npz") as ordered, np.load(tmp_path / "reversed-import/frames.npz") as shuffled:
        assert set(ordered) == set(shuffled)
        for key in ordered:
            np.testing.assert_array_equal(ordered[key], shuffled[key])
    shutil.copytree(tmp_path / "hw", tmp_path / "gap")
    rewrite(tmp_path / "gap", lambda table: table.take(pa.array([0, 2, 3, 4, 5, 6])))  # Episode 0 loses frame 1.
    with pytest.raises(ValueError, match="not contiguous"):
        io.import_lerobot(tmp_path / "gap", tmp_path / "gap-import", raw_units=True)


def tiny_collection(root, steps=3):
    """TEST FIXTURE: the first `steps` transitions of one real P1.2 episode (outcome fields describe the full one)."""
    from xle_workbench.storage import Store
    source = Store(SOURCE / "dataset")
    identifier = json.loads((SOURCE / "collection.json").read_text())["episode_ids"][0]
    executed, metadata = source.load_episode(identifier)
    labels, _ = source.load_training_episode(identifier)
    cut = {"observations": executed["observations"][:steps + 1], "actions": executed["actions"][:steps],
           "rewards": executed["rewards"][:steps], "terminated": executed["terminated"][:steps]}
    record = Store(root / "dataset").save_episode(
        cut, {**{k: v for k, v in metadata.items() if k not in io.DERIVED_KEYS},
              "fixture": f"first {steps} steps of {identifier}"}, corrective_actions=labels["actions"][:steps])
    (root / "collection.json").write_text(json.dumps({"episode_ids": [record["id"]]}))
    return root / "collection.json", record["id"]


def test_camera_keys_agree_across_render_export_recorder_policy_and_hardware(lerobot, tmp_path):
    """docs/CAMERA_KEYS.md: every image feature is observation.images.<name>; sim names are scene/wrist."""
    from lerobot.utils.feature_utils import dataset_to_policy_features, hw_to_dataset_features
    from xert.data.cameras import CAMERAS
    from xert.data.recorder import _features
    from xert.eval.lerobot_act import policy_config
    from xert.spec import load_spec
    from xle_workbench.storage import Store
    rendered = {f"observation.images.{name}" for name in CAMERAS}
    assert rendered == {"observation.images.scene", "observation.images.wrist"}
    recorder = _features(load_spec(ROOT / "xert/spec/so101.yaml"), True, ["block.x"])
    assert not [key for key in recorder if "image" in key]  # The recorder never records images.
    settings = dict(chunk_size=2, dim_model=8, n_heads=1, dim_feedforward=8, n_encoder_layers=1, n_decoder_layers=1,
                    n_vae_encoder_layers=1, latent_dim=2, dropout=0.0, kl_weight=1.0, temporal_ensemble_coeff=None)
    # The P1.4 official ACT reference is state-only: it consumes no camera key (a gap, not a mismatch).
    assert set(policy_config(settings).image_features) == set()
    # Hardware: LeRobot names image features after the camera config key; rollout strips the same prefix.
    hardware = hw_to_dataset_features({"front": (480, 640, 3), "wrist": (480, 640, 3)}, "observation")
    assert {key for key, f in hardware.items() if f["dtype"] == "video"} == \
        {"observation.images.front", "observation.images.wrist"}
    from xert.data.cameras import CameraRenderer
    try:
        CameraRenderer().close()
    except Exception as exc:  # No offscreen GL: skip locally, as tests/test_cameras.py does; CI has osmesa.
        if os.environ.get("CI"):
            raise
        pytest.skip(f"No offscreen GL context: {exc!r}")
    collection, identifier = tiny_collection(tmp_path / "source")
    io.export_lerobot(collection, tmp_path / "export", task=TASK, cameras=True)
    info = json.loads((tmp_path / "export/meta/info.json").read_text())
    exported = {key for key, f in info["features"].items() if f["dtype"] in ("image", "video")}
    visual = {key for key, f in dataset_to_policy_features(info["features"]).items() if f.type.name == "VISUAL"}
    assert exported == visual == rendered  # What an image policy would select is exactly what was rendered.
    record = io.import_lerobot(tmp_path / "export", tmp_path / "import")  # (a) still restores; images stay in raw/.
    for load in ("load_episode", "load_training_episode"):
        before = getattr(Store(tmp_path / "source/dataset"), load)(identifier)[0]
        after = getattr(Store(tmp_path / "import/dataset"), load)(record["episodes"][0]["id"])[0]
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])


def test_a_refuses_a_broken_successor_chain_and_unlabelled_action_streams(lerobot, tmp_path):
    import pyarrow as pa
    collection, _ = tiny_collection(tmp_path / "source")
    io.export_lerobot(collection, tmp_path / "export", task=TASK)
    shutil.copytree(tmp_path / "export", tmp_path / "chain")

    def corrupt(table):  # next_observation[0] no longer equals the state of frame 1.
        column = table.column("xert.next_observation")
        rows = column.to_pylist()
        rows[0][0] += 1.0
        return table.set_column(table.column_names.index("xert.next_observation"), "xert.next_observation",
                                pa.array(rows, type=column.type))
    rewrite(tmp_path / "chain", corrupt)
    with pytest.raises(ValueError, match="successor chain"):
        io.import_lerobot(tmp_path / "chain", tmp_path / "chain-import")
    shutil.copytree(tmp_path / "export", tmp_path / "unlabelled")
    manifest = json.loads((tmp_path / "unlabelled/xert_export.json").read_text())
    episode = manifest["source_episodes"][0]
    del episode["training_action_source"], episode["corrective_sha256"]
    episode["source"] = "test_fixture"  # Not noisy-scripted, so the Store would accept executed actions alone.
    (tmp_path / "unlabelled/xert_export.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="without corrective labels"):
        io.import_lerobot(tmp_path / "unlabelled", tmp_path / "unlabelled-import")
    assert not (tmp_path / "chain-import").exists() and not (tmp_path / "unlabelled-import").exists()

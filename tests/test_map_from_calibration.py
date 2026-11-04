"""`xert map-from-calibration` (slice F1). Every calibration, pose and recording here is a made-up TEST FIXTURE:
scripted numbers chosen so the fit can be checked exactly, never measured on an arm."""
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from xert.backends.so101_hw_backend import TBD, load_joint_map
from xert.data import lerobot_io as io
from xert.data.map_from_calibration import map_from_calibration
from xert.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]
SPEC = load_spec(ROOT / "xert/spec/so101.yaml")
JOINTS = list(SPEC.joint_names)
# FIXTURE truth: motor = A * sim_q + B. Dyadic values, so float32 recordings hold them exactly.
A = np.array([57.25, -57.25, -57.25, 57.25, -57.25, -50.0])
B = np.array([12.5] * 5 + [90.0])
Q0 = np.zeros(6)
Q1 = np.array([1.0, 0.5, -0.5, 1.0, 1.0, 1.0])  # Every spread above its minimum.
Q2 = np.full(6, 0.5)
JITTER = 0.087890625  # Under one STS3215 step (360/4095 deg): a still arm's +-1 step read noise.


@pytest.fixture
def lerobot():
    return pytest.importorskip("lerobot")


def fixture_calibration(path, **ranges):
    """FIXTURE LeRobot calibration (made-up ticks); shoulder_pan range 48..4048, gripper drive_mode 1."""
    cal = {name: {"id": i + 1, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095}
           for i, name in enumerate(JOINTS)}
    cal["shoulder_pan"].update(range_min=48, range_max=4048)
    cal["gripper"].update(drive_mode=1, range_min=2000, range_max=3000)
    for name, (lo, hi) in ranges.items():
        cal[name].update(range_min=lo, range_max=hi)
    path.write_text(json.dumps(cal, indent=4))
    return path


def still(q, joint=None, size=JITTER, a=A, b=B):
    """Three frames of motor = a*q + b; frame 1 offset by `size` on `joint` (default: read noise on every joint)."""
    frames = np.repeat(np.float32(a * q + b)[None], 3, axis=0)
    frames[1, slice(None) if joint is None else joint] += np.float32(size)
    return frames


def roll(x):
    """sim_q at zero except wrist_roll = x."""
    return np.where(np.arange(6) == 4, x, 0.0)


def pose_dataset(root, episodes):
    """SYNTHETIC so_follower lerobot-record layout: one episode per entry of scripted frames."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.feature_utils import hw_to_dataset_features
    motors = {f"{name}.pos": float for name in JOINTS}
    features = {**hw_to_dataset_features(motors, "observation"), **hw_to_dataset_features(motors, "action")}
    dataset = LeRobotDataset.create(repo_id="synthetic/so101-poses", root=root, robot_type="so_follower", fps=30,
                                    features=features, use_videos=False)
    for frames in episodes:
        for row in frames:
            dataset.add_frame({"observation.state": row, "action": row, "task": "hold still"})
        dataset.save_episode(parallel_encoding=False)
    dataset.finalize()
    del dataset
    gc.collect()
    return root


def write_poses(path, poses):
    path.write_text(json.dumps([{"label": f"pose {i}", "dataset": "hw", "episode": episode, "sim_q": list(q)}
                                for i, (episode, q) in enumerate(poses)]))
    return path


def test_two_pose_fit_recovers_sign_scale_offset_and_imports_end_to_end(lerobot, tmp_path):
    pose_dataset(tmp_path / "hw", [still(Q0), still(Q1)])
    calibration = fixture_calibration(tmp_path / "follower_arm.json")
    poses = write_poses(tmp_path / "poses.json", [(0, Q0), (1, Q1)])
    map_from_calibration(calibration, poses, tmp_path / "map.yaml")
    raw = json.loads((tmp_path / "map.yaml").read_text())
    joints = raw["joints"]
    np.testing.assert_allclose([j["sign"] * j["units_per_rad"] for j in joints], A, rtol=0, atol=1e-9)
    np.testing.assert_allclose([j["zero_offset"] for j in joints], B, rtol=0, atol=1e-9)
    assert [j["sign"] for j in joints] == [1, -1, -1, 1, -1, -1]  # Gripper is drive_mode 1 in the fixture.
    assert (joints[0]["min"], joints[0]["max"]) == ((48 - 2048) * 360 / 4095, (4048 - 2048) * 360 / 4095)
    assert all((j["min"], j["max"]) == (-180.0, 180.0) for j in joints[1:5])
    assert (joints[5]["min"], joints[5]["max"]) == (0.0, 100.0)
    assert all(j["max_speed"] == j["max_accel"] == TBD for j in joints)
    provenance = raw["provenance"]
    assert provenance["sim_limits_excess"] == {} and provenance["calibration"]["sha256"] == io.sha256(calibration)
    assert np.isclose(provenance["fit"]["wrist_roll"]["min_sim_spread_rad"], 2 * 360 / 4095 * 2.8412063)
    assert provenance["calibration"]["robot_id"] == "follower_arm"
    assert [p["sim_q"] for p in provenance["poses"]] == [Q0.tolist(), Q1.tolist()]
    assert provenance["poses"][0]["dataset_sha256"] == io._hashes(tmp_path / "hw")
    assert provenance["poses"][0]["median_motor"] == (A * Q0 + B).tolist()  # Median ignores the jittered frame.
    assert all(f["max_abs_residual"] < 1e-9 and "exact by construction" in f["note"]
               for f in provenance["fit"].values())
    # Kinematic tier: import-lerobot and replay-real accept it; motion still needs max_speed/max_accel.
    assert all(load_joint_map(tmp_path / "map.yaml", SPEC, dry_run=False, require="kinematic"))
    with pytest.raises(ValueError, match="for motion"):
        load_joint_map(tmp_path / "map.yaml", SPEC, dry_run=False)
    io.import_lerobot(tmp_path / "hw", tmp_path / "import", joint_map=tmp_path / "map.yaml", motor_units="degrees",
                      calibration=calibration)
    with np.load(tmp_path / "import/frames.npz") as frames:
        np.testing.assert_allclose(frames["state"][frames["frame_index"] != 1], [Q0, Q0, Q1, Q1], rtol=0, atol=1e-9)
    from xert.eval.replay_real import load_import
    assert load_import(tmp_path / "import")[0]["joint_map"]["sha256"] == io.sha256(tmp_path / "map.yaml")
    other = fixture_calibration(tmp_path / "other_arm.json", shoulder_pan=(40, 4040))  # A different robot.
    with pytest.raises(ValueError, match="different calibration"):
        io.import_lerobot(tmp_path / "hw", tmp_path / "refused", joint_map=tmp_path / "map.yaml",
                          motor_units="degrees", calibration=other)


def test_one_pose_leaves_the_line_tbd_and_is_refused(lerobot, tmp_path):
    pose_dataset(tmp_path / "hw", [still(Q0)])
    calibration = fixture_calibration(tmp_path / "follower_arm.json")
    map_from_calibration(calibration, write_poses(tmp_path / "poses.json", [(0, Q0)]), tmp_path / "map.yaml")
    raw = json.loads((tmp_path / "map.yaml").read_text())
    assert all(j[k] == TBD for j in raw["joints"] for k in ("sign", "units_per_rad", "zero_offset"))  # Never assumed.
    assert (raw["joints"][5]["min"], raw["joints"][5]["max"]) == (0.0, 100.0)  # The calibration still fills these.
    assert all("record >= 2 distinct sim_q" in f["method"] for f in raw["provenance"]["fit"].values())
    for require in ("kinematic", "motion"):
        with pytest.raises(ValueError, match="shoulder_pan.*TBD"):
            load_joint_map(tmp_path / "map.yaml", SPEC, dry_run=False, require=require)
    with pytest.raises(ValueError, match="TBD"):
        io.import_lerobot(tmp_path / "hw", tmp_path / "import", joint_map=tmp_path / "map.yaml", motor_units="degrees")


def test_moving_pose_residual_sim_limits_and_existing_output_are_refused(lerobot, tmp_path):
    off_line = still(Q2)
    off_line[:, 0] += np.float32(2.0)  # shoulder_pan 2 units off the A, B line.
    moving = still(Q0, joint=2, size=3 * JITTER)  # elbow_flex: 3 steps > the 2-step still rule.
    one_tick = still(Q0)
    one_tick[:, 4] += np.float32(JITTER)  # wrist_roll reading changes by about one tick ...
    wide = A.copy()
    wide[4] = -70.0  # wrist_roll: 70 deg/rad x 5.585 rad of sim range = 391 deg, more than a full turn.
    pose_dataset(tmp_path / "hw", [still(Q0), still(Q1), off_line, moving, one_tick, still(roll(0.02)),
                                   still(Q0, a=wide), still(Q1, a=wide)])
    calibration = fixture_calibration(tmp_path / "follower_arm.json")
    cases = {"moved": ([(3, Q0), (1, Q1)], calibration),
             "did not move": ([(0, Q0), (4, roll(1.5))], calibration),  # ... while sim_q claims 1.5 rad.
             # 0.02 rad < 2 steps x 2.84 rad / 1.0 = 0.4996 rad; the reading moved 1.1 deg, beyond still noise.
             "below the minimum": ([(0, Q0), (5, roll(0.02))], calibration),
             "calibrated span": ([(6, Q0), (7, Q1)], calibration),
             "residual": ([(0, Q0), (1, Q1), (2, Q2)], calibration),
             # wrist_flex +-550 ticks = +-48.4 deg, but its sim limits map to 12.5 -+ 94.9 deg.
             "outside the calibrated range": ([(0, Q0), (1, Q1)],
                                              fixture_calibration(tmp_path / "narrow.json", wrist_flex=(1500, 2600)))}
    for message, (poses, cal) in cases.items():
        with pytest.raises(ValueError, match=message):
            map_from_calibration(cal, write_poses(tmp_path / "poses.json", poses), tmp_path / "map.yaml")
        assert not (tmp_path / "map.yaml").exists()
    with pytest.raises(FileExistsError):  # The real map exists, so it is never overwritten.
        map_from_calibration(calibration, write_poses(tmp_path / "poses.json", [(0, Q0), (1, Q1)]), io.JOINT_MAP)


def test_full_turn_joint_and_gripper_record_sim_limit_excess(lerobot, tmp_path, capsys):
    a, b = A.copy(), B.copy()
    b[4] = 40.0  # wrist_roll reads +40 deg at sim roll 0 (full-turn calibration 0..4095 = +-180 deg).
    a[5], b[5] = 50.0, 5.0  # Gripper: the sim limit -0.1745 rad maps to -3.7, below LeRobot's clamp at 0.
    pose_dataset(tmp_path / "hw", [still(Q0, a=a, b=b), still(Q1, a=a, b=b)])
    calibration = fixture_calibration(tmp_path / "follower_arm.json")
    map_from_calibration(calibration, write_poses(tmp_path / "poses.json", [(0, Q0), (1, Q1)]), tmp_path / "map.yaml")
    raw = json.loads((tmp_path / "map.yaml").read_text())
    excess = raw["provenance"]["sim_limits_excess"]
    assert set(excess) == {"wrist_roll", "gripper"}
    # Low end cut: from the sim limit up to the sim angle that reads the calibrated end (+180 deg; 0 for gripper).
    assert np.isclose(excess["wrist_roll"]["low_rad"], (180 - 40) / -57.25 + 2.7438473) and excess["wrist_roll"]["high_rad"] == 0
    assert np.isclose(excess["gripper"]["low_rad"], (0 - 5) / 50 + 0.17453) and excess["gripper"]["high_rad"] == 0
    assert (raw["joints"][4]["min"], raw["joints"][4]["max"]) == (-180.0, 180.0)  # Calibrated ends kept.
    assert all(load_joint_map(tmp_path / "map.yaml", SPEC, dry_run=False, require="kinematic"))
    err = capsys.readouterr().err
    assert "warning: wrist_roll" in err and "warning: gripper" in err


def test_never_imports_hardware_modules(lerobot, tmp_path):
    pose_dataset(tmp_path / "hw", [still(Q0), still(Q1)])
    calibration = fixture_calibration(tmp_path / "follower_arm.json")
    poses = write_poses(tmp_path / "poses.json", [(0, Q0), (1, Q1)])
    code = f"""
import sys
sys.argv = ["xert", "map-from-calibration", "--calibration", r"{calibration}", "--poses", r"{poses}",
            "--output", r"{tmp_path / 'map.yaml'}"]
from xert.__main__ import main
main()
bad = [m for m in sys.modules if m == "serial" or m.startswith(("serial.", "lerobot.robots", "lerobot.motors"))]
assert not bad, bad
"""
    child = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120,
                           env={**os.environ, "XERT_LEDGER_PATH": str(tmp_path / "ledger.jsonl")})
    assert child.returncode == 0, child.stderr
    assert (tmp_path / "map.yaml").exists() and not (tmp_path / "ledger.jsonl").exists()

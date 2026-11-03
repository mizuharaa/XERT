"""replay-real (slice D). SYNTHETIC episodes made by simulating the same model: no hardware, temp ledger only."""
import csv
import hashlib
import importlib.util
import json

import mujoco
import numpy as np
import pytest

from xert.eval import replay_real as rr
from xert.eval.ledger import read_rows
from xert.spec import load_spec

FPS = 30
SPEC = load_spec(rr.SPEC)
STARTS = (np.array([0, 0, 0, 0, 0, 0.6]), np.array([0.5, -0.4, 0.6, -0.3, 0.8, 0.5]))
# Measured 2026-11-02 (MuJoCo, Windows): self-replay RMSE was exactly 0.0 on every joint and episode (1e-12 is
# headroom for other platforms); kp x0.3 gave pooled RMSE 0.015-0.035 rad; no reset gave episode-2 RMSE 0.06-0.27 rad.
ZERO_TOL_RAD = 1e-12


def commands(start, frames=20):
    """Smooth goal positions around the start pose, well inside the spec limits."""
    t = np.arange(frames)[:, None] / FPS
    return start + 0.3 * np.sin(2 * np.pi * 1.5 * t + np.arange(6))


def simulate(starts=STARTS, kp_scale=1.0):
    """The "recording": raw MuJoCo, not the backend, at replay_timing's dt; returns [(state, action)]."""
    model = SPEC.load_model()
    model.opt.timestep, substeps = rr.replay_timing(FPS, model.opt.timestep)
    model.actuator_gainprm[:, 0] *= kp_scale
    model.actuator_biasprm[:, 1] *= kp_scale  # Still a position servo, with a deliberately wrong gain.
    qpos = [model.joint(name).qposadr[0] for name in SPEC.joint_names]
    actuators = [model.actuator(name).id for name in SPEC.joint_names]
    episodes = []
    for start in starts:
        action, data = commands(start), mujoco.MjData(model)
        data.qpos[qpos] = data.ctrl[actuators] = start
        mujoco.mj_forward(model, data)
        state = [data.qpos[qpos].copy()]
        for target in action[:-1]:
            data.ctrl[actuators] = target
            mujoco.mj_step(model, data, nstep=substeps)
            state.append(data.qpos[qpos].copy())
        episodes.append((np.array(state), action))
    return episodes


def write_import(root, episodes, **overrides):
    """A SYNTHETIC path (c)-format import (frames.npz + import.json); no raw/ copy, so never ledger-eligible."""
    root.mkdir()
    lengths = [len(state) for state, _ in episodes]
    frame_index = np.concatenate([np.arange(n) for n in lengths])
    np.savez(root / "frames.npz", episode_index=np.repeat(np.arange(len(lengths)), lengths),
             frame_index=frame_index, timestamp=frame_index / FPS,
             state=np.concatenate([s for s, _ in episodes]), action=np.concatenate([a for _, a in episodes]))
    joints = [f"{name}.pos" for name in SPEC.joint_names]
    record = {"version": 1, "source_kind": "so101_follower", "derived": False, "fps": FPS,
              "lerobot_robot_type": "so101_follower", "units": "radians", "embodiment": "so101",
              "spec_sha256": hashlib.sha256(rr.SPEC.read_bytes()).hexdigest(),
              "columns": {"state": joints, "action": joints}, "motor_units": "degrees",
              "joint_map": {"path": "SYNTHETIC", "sha256": hashlib.sha256(b"synthetic map").hexdigest()},
              "status": "SYNTHETIC test fixture, simulated by the same model"}
    record.update(overrides)
    (root / "import.json").write_text(json.dumps({k: v for k, v in record.items() if v is not None}))
    return root


def pooled(result):
    return np.array(list(result["pooled_rmse_rad"].values()))


def per_episode(output):
    with (output / "summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {(r["episode"], r["joint"]): (float(r["rmse_rad"]), int(r["n"])) for r in rows}


def test_timing_is_an_exact_integer_number_of_physics_steps_per_frame():
    assert rr.replay_timing(30, 0.005) == (1 / 210, 7)  # 1/(30*0.005) = 6.67 does not divide.
    assert rr.replay_timing(25, 0.005) == (0.005, 8)
    assert rr.replay_timing(50, 0.005) == (0.005, 4)
    dt, substeps = rr.replay_timing(FPS, 0.005)
    backend = rr.start_episode(SPEC, FPS, dt, STARTS[1])
    assert backend.substeps == substeps and backend.model.opt.timestep == dt
    _, times = rr.run_episode(backend, commands(STARTS[1]))
    np.testing.assert_allclose(np.diff(times), 1 / FPS, rtol=0, atol=1e-12)
    np.testing.assert_allclose(times, np.arange(len(times)) / FPS, rtol=0, atol=1e-12)


def test_self_replay_is_zero_per_episode_with_reset_and_artifacts_are_hashed(tmp_path):
    episodes = simulate()
    dataset = write_import(tmp_path / "import", episodes)
    result = rr.replay_real(dataset, tmp_path / "replay")
    out = tmp_path / "replay"
    assert result["substeps_per_frame"] == 7 and result["timestep_s"] == 1 / 210
    assert np.all(pooled(result) <= ZERO_TOL_RAD)
    summary = per_episode(out)
    for episode in ("0", "1"):  # Two different start poses, each about 0 because each episode is reset.
        assert all(summary[(episode, j)][0] <= ZERO_TOL_RAD and summary[(episode, j)][1] == 19
                   for j in SPEC.joint_names)
    assert summary[("pooled", "gripper")][1] == 38
    manifest = json.loads((out / "replay.json").read_text())
    assert manifest["import_sha256"] == hashlib.sha256((dataset / "import.json").read_bytes()).hexdigest()
    assert manifest["joint_map_sha256"] == json.loads((dataset / "import.json").read_text())["joint_map"]["sha256"]
    assert manifest["timestep_s"] == 1 / 210 and manifest["substeps_per_frame"] == 7
    for name, digest in manifest["artifact_sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    has_plots = importlib.util.find_spec("matplotlib") is not None
    assert {f"plots/{j}.png" for j in SPEC.joint_names} <= set(manifest["artifact_sha256"]) if has_plots \
        else manifest["plots"].startswith("skipped") and not (out / "plots").exists()
    assert {"trace.csv", "summary.csv", "summary.md"} <= set(manifest["artifact_sha256"])
    with (out / "trace.csv").open(newline="") as stream:
        trace = list(csv.reader(stream))
    assert len(trace) == 1 + 40 and all(len(row) == 15 for row in trace)
    values = np.array([[float(x) for x in row[2:]] for row in trace[1:]])  # Plain floats, not np.float64(...).
    np.testing.assert_array_equal(values[:, 1:7], np.concatenate([state for state, _ in episodes]))  # Round trip.
    assert read_rows() == []  # No --ledger, no rows.


def test_actuator_gain_mismatch_gives_clearly_nonzero_rmse(tmp_path):
    dataset = write_import(tmp_path / "import", simulate(kp_scale=0.3))
    assert pooled(rr.replay_real(dataset, tmp_path / "replay")).min() > 1e-3


def test_without_the_per_episode_reset_episode_2_is_wrong(tmp_path):
    """Test-only helper: one backend for both episodes, so episode 2 starts where episode 1 ended."""
    (state0, action0), (state1, action1) = simulate()
    dt, _ = rr.replay_timing(FPS, 0.005)
    backend = rr.start_episode(SPEC, FPS, dt, state0[0])
    first, _ = rr.run_episode(backend, action0)
    carried, _ = rr.run_episode(backend, action1)  # No start_episode: sim state carried across episodes.
    assert np.abs(first - state0).max() <= ZERO_TOL_RAD
    assert np.sqrt(np.mean((carried[1:] - state1[1:]) ** 2, axis=0)).max() > 0.1
    reset, _ = rr.run_episode(rr.start_episode(SPEC, FPS, dt, state1[0]), action1)
    assert np.abs(reset - state1).max() <= ZERO_TOL_RAD


@pytest.mark.parametrize("change, message", [
    ({"units": "raw_lerobot"}, "raw_lerobot"),
    ({"units": None}, "None"),
    ({"joint_map": None}, "joint-map sha256"),
    ({"joint_map": {"path": "x"}}, "joint-map sha256"),
    ({"source_kind": "xert_recording"}, "hardware frame-pair import"),
    ({"derived": True}, "hardware frame-pair import"),
    ({"spec_sha256": "0" * 64}, "spec differs"),
    ({"fps": 29.97}, "resamples"),
])
def test_inputs_outside_the_import_contract_are_refused(tmp_path, change, message):
    dataset = write_import(tmp_path / "import", simulate(starts=STARTS[:1]), **change)
    with pytest.raises(ValueError, match=message):
        rr.replay_real(dataset, tmp_path / "replay")
    assert not (tmp_path / "replay").exists()


def test_actions_within_1e_6_rad_of_a_limit_are_clipped_and_beyond_are_refused(tmp_path):
    for excess, accepted in ((5e-7, True), (2e-6, False)):
        state, action = simulate(starts=STARTS[:1])[0]
        action = action.copy()
        action[3, 5] = SPEC.position_limits[5, 1] + excess  # A gripper command just past its upper limit.
        dataset = write_import(tmp_path / f"import-{excess}", [(state, action)])
        out = tmp_path / f"replay-{excess}"
        if accepted:
            rr.replay_real(dataset, out)
            assert json.loads((out / "replay.json").read_text())["action_max_limit_excess_rad"] == pytest.approx(excess)
        else:
            with pytest.raises(ValueError, match="outside the spec limits"):
                rr.replay_real(dataset, out)
            assert not out.exists()


def test_a_direct_lerobot_dataset_is_refused(tmp_path):
    (tmp_path / "lerobot/meta").mkdir(parents=True)
    (tmp_path / "lerobot/meta/info.json").write_text(json.dumps({"codebase_version": "v3.0", "robot_type": "so_follower"}))
    with pytest.raises(ValueError, match="import-lerobot --motor-units degrees"):
        rr.replay_real(tmp_path / "lerobot", tmp_path / "replay")
    assert not (tmp_path / "replay").exists()


def test_ledger_rows_are_refused_without_a_verified_raw_recording(tmp_path):
    dataset = write_import(tmp_path / "import", simulate(starts=STARTS[:1]))
    with pytest.raises(ValueError, match="Refusing ledger rows"):
        rr.replay_real(dataset, tmp_path / "replay", ledger=True)
    assert read_rows() == [] and not (tmp_path / "replay").exists()


@pytest.mark.xfail(strict=True, raises=NotImplementedError,
                   reason="TODO(daniel): actuator parameter fitting (kp/kv/damping/armature/frictionloss)")
def test_fit_recovers_a_known_gain_and_generalizes_to_a_held_out_episode(tmp_path):
    starts = (*STARTS, np.array([-0.4, 0.3, -0.5, 0.4, -0.6, 0.2]))
    dataset = write_import(tmp_path / "import", simulate(starts=starts, kp_scale=0.5))
    fit = rr.fit_actuator_params(dataset, held_out_episode=2)
    assert fit["held_out_episode"] == 2 and fit["n"] == {"train": 38, "held_out": 19}
    for joint in SPEC.joint_names:
        assert fit["held_out_rmse_rad"][joint] < 0.1 * fit["baseline_rmse_rad"][joint]
        assert fit["params"][joint]["kp"] == pytest.approx(0.5 * 998.22, rel=0.1)


# ---- LeRobot 0.6.1 environment: the real chain, import-lerobot -> replay-real ----

def test_lerobot_import_then_replay_keeps_shas_and_writes_reference_backend_rows(tmp_path):
    pytest.importorskip("lerobot")
    from test_lerobot_io import hardware_import
    dataset = hardware_import(tmp_path)
    rr.replay_real(dataset, tmp_path / "replay", ledger=True)  # Temp ledger (conftest): synthetic data.
    manifest = json.loads((tmp_path / "replay/replay.json").read_text())
    record = json.loads((dataset / "import.json").read_text())
    assert manifest["joint_map_sha256"] == record["joint_map"]["sha256"]
    rows = read_rows()
    assert [r["joint"] for r in rows] == list(SPEC.joint_names)
    for row in rows:
        assert row["backend"] == "mujoco" and row["reference_backend"] == "so101_hw"
        assert row["import_sha256"] == manifest["import_sha256"] and row["n"] == 2 + 3
        assert "not system identification" in row["notes"].lower()

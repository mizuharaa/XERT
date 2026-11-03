"""SO-101 as a second bench arm: names, limits, scene, timing, reach, teleop profile, CLI refusals.

Scripted hands, scripted clock and scripted keys only: no camera, no hardware, no ledger rows.
"""
import ast
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap

import mujoco
import numpy as np
import pytest

from scripts.record_scripted_runtime import ScriptedClock, ScriptedSource, scripted_frame
from xert.backends.bench import BenchBackend
from xert.retarget.calibrate import CalibratedMapper, create_calibration
from xert.runtime import teleop
from xert.sources import MotionFrame
from xert.spec import load_spec
from xle_workbench import pipeline
from xle_workbench.sim import ROBOTS, SO101_ASSETS, ManipulationEnv
from xle_workbench.storage import Store

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "xert/spec/so101.yaml"
PROFILE = ROOT / "configs/teleop/so101_bench.json"
SCRIPTED = {"kind": "scripted", "human_demonstrations": False, "generator": "tests/test_so101_bench.py"}
# LeRobot 0.6.1 lerobot/robots/so_follower/so_follower.py:54-59, used when lerobot is not installed.
PINNED_LEROBOT_MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def lerobot_motors():
    """Motor names from the installed SOFollower source; constructing it needs the Feetech SDK, so read its AST."""
    try:
        from lerobot.robots.so_follower import so_follower
    except ImportError:
        return None
    tree = ast.parse(textwrap.dedent(inspect.getsource(so_follower.SOFollower.__init__)))
    motors = next(k.value for k in ast.walk(tree) if isinstance(k, ast.keyword) and k.arg == "motors")
    return [key.value for key in motors.keys]


def test_joint_names_match_lerobot_so101_follower():
    installed = lerobot_motors()
    expected = PINNED_LEROBOT_MOTORS if installed is None else installed
    print("compared against", "pinned list" if installed is None else "installed lerobot")
    assert ROBOTS["so101"]["joints"] == expected
    assert list(load_spec(SPEC).joint_names) == expected
    assert ManipulationEnv({"robot": "so101"}).metadata()["action_names"] == expected


def test_limits_come_from_the_menagerie_model_and_spec():
    # LeRobot defines no radian limits (per-arm calibration supplies them), so the reference is the
    # vendored Menagerie model: the bench clamps each actuator ctrlrange to its joint range.
    env, spec = ManipulationEnv({"robot": "so101"}), load_spec(SPEC)
    menagerie = mujoco.MjModel.from_xml_path(str(SO101_ASSETS / "so101_xert.xml"))
    for i, name in enumerate(env.joints):
        joint, actuator = menagerie.joint(name), menagerie.actuator(name)
        expected = [max(joint.range[0], actuator.ctrlrange[0]), min(joint.range[1], actuator.ctrlrange[1])]
        assert env.limits[i].tolist() == expected
    np.testing.assert_array_equal(env.limits, spec.position_limits)
    assert np.all((spec.position_limits[:, 0] <= env.robot["reset_pose"]) & (env.robot["reset_pose"] <= spec.position_limits[:, 1]))


def test_scene_composition_leaves_vendored_files_and_records_provenance():
    files = sorted(path for path in SO101_ASSETS.rglob("*") if path.is_file())
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    env = ManipulationEnv({"robot": "so101"})
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    composition = env.metadata()["source"]["composition"]
    assert composition["timestep"] == {"menagerie": .005, "bench": .002} and env.model.opt.timestep == .002
    assert composition["base"] == {"pos": [.42, .04, .08], "quat": [0, 0, 0, 1]}
    # Menagerie's contact solver settings, disclosed because they differ from the XLeRobot bench (100 / 50 / 1).
    assert composition["option"] == {"timestep": .002, "iterations": 10, "ls_iterations": 20, "impratio": 10}
    np.testing.assert_array_equal(env.model.body("base").pos, [.42, .04, .08])
    assert env.metadata()["source"]["robot_xml_sha256"] == before[SO101_ASSETS / "so101_xert.xml"]
    # Block, tray and table are the XLeRobot bench's own bodies, at the same world poses.
    xle = ManipulationEnv()
    for body in ("tray", "block"):
        np.testing.assert_array_equal(env.model.body(body).pos, xle.model.body(body).pos)
    np.testing.assert_array_equal(env.model.geom("table").size, xle.model.geom("table").size)
    assert env.xml_sha256 != xle.xml_sha256 and env.metadata()["model_sha256"] == env.xml_sha256


def test_reset_holds_still_without_touching_the_block():
    env = ManipulationEnv({"robot": "so101"})
    observation = env.reset(0)
    np.testing.assert_array_equal(observation[:6], np.float32(env.robot["reset_pose"]))
    hold = env.normalize(env.robot["reset_pose"]).astype(np.float32)
    for _ in range(25):
        observation, _, _, info = env.step(hold)
        assert not info["object_in_robot_contact"]
    assert np.abs(observation[:6] - env.robot["reset_pose"]).max() < 1e-3  # One second under gravity.
    assert abs(info["object_position"][2] - .096) < 1e-3


def test_so101_bench_step_advances_one_25hz_period():
    backend = BenchBackend(load_spec(SPEC), 0)
    assert backend.env._core.substeps == 20  # round(1/(25 * 0.002))
    state, period = backend.read_state(), 1 / backend.spec.control_hz
    for _ in range(10):
        backend.send_targets(state.q, 0)
        following = backend.read_state()
        assert following.t - state.t == pytest.approx(period, abs=1e-12)
        state = following


def test_reach_measurement_covers_block_region_and_tray():
    sys.path.insert(0, str(ROOT / "scripts"))
    from so101_bench_reach import TOLERANCE, reach_report
    report = reach_report()
    assert len(report["samples"]) == 14 and report["worst_fk_error_m"] < TOLERANCE


def calibration(clock):
    # What `xert calibrate --bench --spec xert/spec/so101.yaml` does: neutral grasp = bench reset pose.
    return create_calibration(SPEC, PROFILE, scripted_frame(0, clock()), BenchBackend(load_spec(SPEC), 0))


def test_bench_profile_reaches_block_and_tray_from_modest_hand_motion():
    # Same check as the P4.1 XLeRobot profile test, over the whole block-spawn region.
    profile = json.loads(PROFILE.read_text())
    bench = BenchBackend(load_spec(SPEC), 0)
    neutral = calibration(ScriptedClock())
    mapper = CalibratedMapper(SPEC, PROFILE, neutral, bench)
    origin = np.asarray(neutral["sites"]["grasp"]["position"])
    np.testing.assert_allclose(origin, bench.data.site("grasp").xpos)
    base, fk = scripted_frame(0, 1.).hands["right"], mujoco.MjData(bench.model)
    (x0, y0), (x1, y1) = bench.env.config["object_position_bounds"]
    targets = [("seed-0 block", bench.data.body("block").xpos.copy())]
    targets += [(f"block corner {x},{y}", np.array([x, y, .096])) for x in (x0, x1) for y in (y0, y1)]
    targets += [("above block", targets[0][1] + [0, 0, .06]), ("tray", bench.model.body("tray").pos + [0, 0, .04])]
    for name, target in targets:
        image_x, image_y, depth = np.asarray(profile["axes"]).T @ (target - origin) / profile["gains"]
        scale = 1 / (1 + depth)
        assert abs(image_x) < .25 and abs(image_y) < .35 and .7 < scale < 1.5
        image = base.image[0] + (base.image - base.image[0]) * scale + [image_x, image_y, 0]
        hand = replace(base, image=image)
        q = mapper.map(MotionFrame(1, 1., 1., {"left": hand, "right": hand}, "scripted", (640, 480)))
        fk.qpos[:] = bench.data.qpos
        fk.qpos[bench.env.qids] = q
        mujoco.mj_forward(bench.model, fk)
        error = np.linalg.norm(fk.site("grasp").xpos - target)
        print(f"{name}: image dx={image_x:+.3f} dy={image_y:+.3f} palm x{scale:.2f} -> IK error {error:.1e} m")
        assert error < 1e-3


def test_teleop_session_records_so101_bench_with_timing_check(tmp_path):
    pytest.importorskip("lerobot")
    from xert.data.recorder import load_runtime_dataset, verify_recording
    clock = ScriptedClock()
    neutral = calibration(clock)
    keys = iter([" ", None, None, " ", None, "\x1b"])  # Start, two quiet ticks, stop, drain, quit.
    kept = teleop.run_session(ScriptedSource(clock), SPEC, tmp_path / "session",
                              lambda backend: CalibratedMapper(SPEC, PROFILE, neutral, backend), provenance=SCRIPTED,
                              keys=lambda: next(keys), clock=clock, sleep=clock.sleep)
    assert len(kept) == 1 and verify_recording(kept[0])["frames"] == 3  # RuntimeRecorder ran its timing check.
    manifest = json.loads((kept[0] / "xert_recording.json").read_text())
    assert manifest["embodiment"] == "so101" and manifest["source"]["bench"]["config"]["robot"] == "so101"
    times = [json.loads(line)["runtime"]["t_sim"] for line in (kept[0] / "xert_frames.jsonl").read_text().splitlines()]
    assert np.diff([manifest["initial_state"]["t"], *times]) == pytest.approx([.04] * 3, abs=1e-12)
    dataset = load_runtime_dataset(kept[0])
    assert dataset.meta.robot_type == "so101" and len(dataset) == 3
    del dataset


def test_every_bench_robot_has_its_own_ledger_note():
    assert set(pipeline.EVALUATION_NOTES) == set(ROBOTS) and "XLeRobot" not in pipeline.EVALUATION_NOTES["so101"]


def test_collect_and_evaluate_refuse_until_the_expert_is_tuned(tmp_path):
    store = Store(tmp_path / "data")
    for run in (lambda: pipeline.collect(store, [0], {"robot": "so101"}),
                lambda: pipeline.evaluate(store, [10000], {"robot": "so101"})):
        with pytest.raises(NotImplementedError, match=r"SO-101 scripted expert not tuned \(TODO\(daniel\)\)"):
            run()
    assert not any(path.is_file() for path in (tmp_path / "data").rglob("*"))


def test_cli_robot_flag_selects_the_bench_and_replay_checks_it(tmp_path, monkeypatch):
    from xle_workbench import cli
    data = str(tmp_path / "data")

    def main(*argv):
        monkeypatch.setattr(sys, "argv", ["xle_workbench", "--data", data, *argv])
        cli.main()

    with pytest.raises(NotImplementedError, match="not tuned"):
        main("collect", "--robot", "so101", "--episodes", "1")
    with pytest.raises(SystemExit):
        main("collect", "--robot", "so101", "--config", '{"robot": "xlerobot_left"}')
    output = tmp_path / "collection.json"
    main("--output", str(output), "collect", "--robot", "xlerobot_left", "--episodes", "1", "--config", '{"horizon": 2}')
    episode = json.loads(output.read_text())["episode_ids"][0]
    assert pipeline.replay(Store(data), episode, robot="xlerobot_left")["matched"]
    with pytest.raises(ValueError, match="recorded on xlerobot_left, not so101"):
        main("replay", episode, "--robot", "so101")

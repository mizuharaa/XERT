"""Slice K plumbing: the drawer task through rendering, rollouts, recording, quality, the harnesses and DAgger's
refusal.

Scripted hands, scripted clock, constant-action policy stubs and temporary outputs only: no human data, no
trained policy and no number. Tiny fixtures (5-frame recordings, 64 px renders, two seeds).
"""
import json
import os
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

from xert.backends.drawer import DrawerBackend
from xert.spec import load_spec
from xert.tasks import environment_names

ROOT = Path(__file__).resolve().parents[1]
ROBOTS = ("xlerobot_left", "so101")
PERIOD = 1 / 25


def spec_of(robot):
    return load_spec(ROOT / f"xert/spec/{robot}.yaml")


def renderer_without_msaa(robot, size=64):
    """CameraRenderer for the drawer with MSAA off: with MuJoCo's default 4 offscreen samples, re-rendering one state
    on the Windows GPU is not bit-stable (docs/handoff/H-stack-bench.md); the path under test is the same."""
    from xert.data.cameras import CameraRenderer
    try:
        renderer = CameraRenderer(size, robot, "drawer")
    except Exception as exc:
        if os.environ.get("CI"):
            raise
        pytest.skip(f"No offscreen GL context: {exc!r}")
    renderer.renderer.close()
    renderer.model.vis.quality.offsamples = 0
    renderer.renderer = mujoco.Renderer(renderer.model, size, size)
    return renderer


@pytest.mark.parametrize("robot", ROBOTS)
def test_two_seeds_rebuild_offline_and_live_to_each_backends_own_scene(robot):
    """Review K-01: with the cabinet fixed per robot, joints + drawer.pos rebuild the whole scene. Two seeds with
    different initial openings: offline and live_frame reconstruct handle and cabinet within 1e-5 of each backend's
    scene, and the pixels equal a direct render of that backend's MjData."""
    pytest.importorskip("torch")
    from xert.data.cameras import TOLERANCE, _columns, row_state, scene_objects
    from xert.eval import image_act
    renderer = renderer_without_msaa(robot)
    image = {"dtype": "image", "shape": (64, 64, 3)}
    try:
        scenes = []
        for seed in (0, 1):
            backend = DrawerBackend(spec_of(robot), seed)
            # Recordings store float32; float32-exact qpos makes the live float64 state the recorded one bit for bit.
            backend.data.qpos[:] = backend.data.qpos.astype(np.float32)
            mujoco.mj_forward(backend.model, backend.data)
            live = image_act.live_frame(backend, renderer, {"observation.images.scene": image,
                                                            "observation.images.wrist": image})
            joints = [f"{name}.pos" for name in backend.spec.joint_names]
            recorded = {"observation.state": {"names": joints},
                        "observation.environment_state": {"names": backend.object_names}}
            frame = {"observation.state": backend.read_state().q.astype(np.float32),
                     "observation.environment_state": np.asarray(backend.object_state(), dtype=np.float32)}
            state = row_state(frame, _columns(recorded, renderer.joints, renderer.bodies, renderer.points,
                                              renderer.scalars))
            offline, achieved = renderer.render(state["joints"], scene_objects(state, renderer.bodies, renderer.scalars))
            assert set(achieved) == {"handle", "grasp", "drawer.pos"}
            for name, actual in (("handle", backend.data.geom("handle").xpos), ("grasp", backend.data.site("grasp").xpos),
                                 ("drawer.pos", backend.data.qpos[backend.task.qpos:backend.task.qpos + 1])):
                assert np.max(np.abs(achieved[name] - actual)) <= TOLERANCE, name
            np.testing.assert_allclose(renderer.data.body("cabinet").xpos, backend.data.body("cabinet").xpos, atol=1e-12)
            direct = mujoco.MjData(renderer.model)  # The backend's own state, posed directly, no named state.
            direct.qpos[:] = backend.data.qpos
            mujoco.mj_forward(renderer.model, direct)
            for key, pixels in offline.items():
                np.testing.assert_array_equal(live[key].mul(255).round().byte().permute(1, 2, 0).numpy(), pixels)
                renderer.renderer.update_scene(direct, camera=renderer.cameras[key.split(".")[-1]].get(
                    "model_camera", key.split(".")[-1]), scene_option=renderer.option)
                np.testing.assert_array_equal(renderer.renderer.render(), pixels)
            scenes.append((state["drawer.pos"][0], achieved["handle"], offline["observation.images.scene"]))
        (open0, handle0, image0), (open1, handle1, image1) = scenes
        assert open0 != open1 and not np.allclose(handle0, handle1) and not np.array_equal(image0, image1)
    finally:
        renderer.close()


def test_drawer_cameras_override_only_the_scene_camera_and_keep_the_pinned_ones():
    from xert.data.cameras import CAMERAS, cameras
    for robot in ROBOTS:
        assert cameras(robot) == cameras(robot, "stack") == CAMERAS[robot]
        drawer = cameras(robot, "drawer")
        assert drawer["wrist"] == CAMERAS[robot]["wrist"] and drawer["scene"] != CAMERAS[robot]["scene"]


# ---------------------------------------------------------------- Harnesses and callers (no LeRobot).

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
    """Record (sim time before, after, t_cmd) of every DrawerBackend step taken through the harnesses."""
    times, send = [], DrawerBackend.send_targets

    def wrapped(self, q, t_cmd=None):
        before = self.data.time
        send(self, q, t_cmd)
        times.append((before, self.data.time, t_cmd))
    monkeypatch.setattr(DrawerBackend, "send_targets", wrapped)
    return times


def hold_reset_action(spec):
    from xle_workbench.sim import ROBOTS as BENCH
    low, high = spec.position_limits.T
    return 2 * (np.asarray(BENCH[spec.name]["reset_pose"]) - low) / (high - low) - 1


def test_gate_rollout_runs_a_full_drawer_episode_with_a_constant_policy(monkeypatch):
    from xert.eval import gate_experiment as gx
    spec = spec_of("so101")
    times = timed(monkeypatch)
    names = [f"{joint}.pos" for joint in spec.joint_names] + environment_names("drawer")
    policy = ConstantPolicy(hold_reset_action(spec))
    trial = gx.rollout(policy, spec, 10000, names, "drawer")
    assert trial == {"seed": 10000, "success": False, "steps": 600, "failure_reason": "timeout"}
    assert len(times) == 600 and all(t is None for _, _, t in times)  # Policy rollouts pass t_cmd=None.
    np.testing.assert_allclose([after - before for before, after, _ in times], PERIOD, rtol=0, atol=1e-9)
    assert len(policy.inputs) == 600 and policy.inputs[0].shape == (len(names),)
    assert policy.inputs[0][names.index("drawer.pos")] > 0
    assert policy.inputs[-1][names.index("time_fraction")] == np.float32(599 / 600)


def test_ledger_and_teleop_cli_accept_drawer(monkeypatch):
    import argparse
    from xert.eval import ledger
    from xert.runtime import cli, teleop
    row = ledger.new_row(git_commit="0" * 40, git_dirty=False, slice="K", experiment="test", backend="mujoco",
                         embodiment="so101", metric="success_rate", value=0.0, ci95=None, n=1,
                         seeds={"train": [], "eval": [0]}, config_path="c", artifact_paths=["a"], notes="test row",
                         task_id="drawer")
    ledger.validate_row(row)
    parser = argparse.ArgumentParser()
    cli.add_commands(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["teleop", "--spec", "s", "--profile", "p", "--calibration", "c", "--output", "o",
                              "--task", "drawer"])
    assert args.task == "drawer" and set(teleop.TASK_TEXT) == set(teleop.SUCCESS_PROVENANCE) >= {"drawer"}


# ---------------------------------------------------------------- Recordings (LeRobot environment).

SPEC = ROOT / "xert/spec/xlerobot_left.yaml"
PROFILE = ROOT / "configs/teleop/xlerobot_bench.json"
SCRIPTED = {"kind": "scripted", "human_demonstrations": False, "generator": "tests/test_drawer_pipeline.py"}
FRAMES = 5


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    """One scripted-hand teleop episode on the drawer (xlerobot_left, bench seed 0)."""
    pytest.importorskip("lerobot")
    from scripts.record_scripted_runtime import ScriptedClock, ScriptedSource, scripted_frame
    from xert.backends.base import task_backend
    from xert.retarget.calibrate import CalibratedMapper, create_calibration
    from xert.runtime import teleop
    clock = ScriptedClock()
    neutral = create_calibration(SPEC, PROFILE, scripted_frame(0, clock()), task_backend(load_spec(SPEC), 0, "drawer"))
    keys = iter([" ", *[None] * (FRAMES - 1), " ", None, "\x1b"])
    [episode] = teleop.run_session(ScriptedSource(clock), SPEC, tmp_path_factory.mktemp("drawer") / "session",
                                   lambda b: CalibratedMapper(SPEC, PROFILE, neutral, b), provenance=SCRIPTED,
                                   keys=lambda: next(keys), clock=clock, sleep=clock.sleep, task_id="drawer")
    return episode


def test_drawer_teleop_recording_round_trips_with_success_and_drawer_state(recording):
    from xert.data.cameras import camera_model
    from xert.data.recorder import load_runtime_dataset, verify_recording
    from xert.runtime import teleop
    from xert.tasks.drawer import initial_opening
    assert verify_recording(recording)["frames"] == FRAMES
    manifest = json.loads((recording / "xert_recording.json").read_text())
    assert manifest["task_id"] == "drawer" and manifest["task"] == teleop.TASK_TEXT["drawer"]
    bench = manifest["source"]["bench"]
    assert bench["model_sha256"] == camera_model("xlerobot_left", "drawer")[1]
    assert bench["success_rule"]["open_m"] == .08 and bench["success_rule"]["settle_ticks"] == 12
    journal = [json.loads(line)["runtime"] for line in (recording / "xert_frames.jsonl").read_text().splitlines()]
    assert {frame["task_id"] for frame in journal} == {"drawer"}
    dataset = load_runtime_dataset(recording)
    assert dataset.features["observation.environment_state"]["names"] == environment_names("drawer")
    assert dataset.features["next.success"]["dtype"] == "bool"
    assert not any(dataset[i]["next.success"].item() for i in range(FRAMES))
    assert dataset[0]["observation.environment_state"][0].item() == np.float32(initial_opening(0))
    assert dataset[0]["task"] == teleop.TASK_TEXT["drawer"]
    del dataset


def gl_render(*args, **kwargs):
    from xert.data.cameras import render_cameras
    try:
        return render_cameras(*args, **kwargs)
    except Exception as exc:
        if "GL" in repr(exc) and not os.environ.get("CI"):
            pytest.skip(f"No offscreen GL context: {exc!r}")
        raise


@pytest.fixture(scope="module")
def rendered(recording, tmp_path_factory):
    output = tmp_path_factory.mktemp("drawer-rendered") / "rendered"
    gl_render(recording, output, size=64)
    return output


def test_render_cameras_export_and_quality_take_the_drawer_from_the_recording(recording, rendered, tmp_path,
                                                                             monkeypatch):
    from xert.__main__ import main
    from xert.data.cameras import cameras
    from xert.data.lerobot_io import export_lerobot
    from xert.runtime import teleop
    manifest = json.loads((rendered / "xert_cameras.json").read_text())
    assert manifest["task_id"] == "drawer" and "drawer.pos" in manifest["reconstruction"]
    errors = manifest["reconstruction_max_error"]
    assert set(errors) == {"handle", "grasp", "drawer.pos"} and max(errors.values()) <= manifest["tolerance"]
    assert {name: {key: camera[key] for key in definition} for name, camera in manifest["cameras"].items()
            for definition in [cameras("xlerobot_left", "drawer")[name]]} == cameras("xlerobot_left", "drawer")
    assert "cabinet front" in manifest["camera_placement"]
    export_lerobot(recording, tmp_path / "export", task=teleop.TASK_TEXT["drawer"])
    assert json.loads((tmp_path / "export/xert_recording.json").read_text())["task_id"] == "drawer"
    monkeypatch.setattr(sys, "argv", ["xert", "quality", "--dataset", str(recording), "--output",
                                      str(tmp_path / "quality")])
    main()
    assert json.loads((tmp_path / "quality/metrics.json").read_text())["task_id"] == "drawer"


def test_image_act_describes_and_evaluates_the_drawer_with_live_renders(rendered, monkeypatch, tmp_path):
    import torch
    from lerobot.policies.act import modeling_act
    from xert.data.cameras import cameras
    from xert.eval import image_act, ledger
    from xle_workbench.sim import ROBOTS as BENCH
    data = image_act.describe([rendered])
    assert data["task_id"] == "drawer" and data["camera"]["cameras"] == cameras("xlerobot_left", "drawer")
    times = timed(monkeypatch)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    features = {key: data["meta"]["features"][key] for key in ("observation.images.scene", "observation.images.wrist",
                                                                "observation.state", "observation.environment_state")}
    (checkpoint / image_act.METADATA).write_text(json.dumps({
        "robot": "xlerobot_left", "task_id": "drawer", "seed": 0, "bench_seeds": data["bench_seeds"],
        "source_kinds": data["source_kinds"], "action_units": data["action_units"], "git_dirty": False,
        "dataset_sha256": data["dataset_sha256"], "features": features, "camera": data["camera"]}))
    hold, seen = torch.tensor([BENCH["xlerobot_left"]["reset_pose"]], dtype=torch.float32), []

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
    report = image_act.evaluate(checkpoint, [10000], tmp_path / "report.json")
    assert report["task_id"] == "drawer" and report["n"] == 1 and report["successes"] == 0
    assert report["trials"] == [{"seed": 10000, "success": False, "steps": 600, "failure_reason": "timeout"}]
    assert report["ledger"].startswith("no rows: training data is ['scripted']") and ledger.read_rows() == []
    np.testing.assert_allclose([after - before for before, after, _ in times], PERIOD, rtol=0, atol=1e-9)
    assert len(times) == len(seen) == 600 and seen[0] == {
        "observation.images.scene": (3, 64, 64), "observation.images.wrist": (3, 64, 64),
        "observation.state": (6,), "observation.environment_state": (8,)}

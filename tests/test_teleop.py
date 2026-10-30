"""Scripted hands, scripted clock and scripted keys only: no camera, no human data, no ledger rows."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from scripts.record_scripted_runtime import ScriptedClock, ScriptedSource, scripted_frame
from xert.backends.bench import BenchBackend
from xert.retarget.calibrate import CalibratedMapper, create_calibration
from xert.runtime import teleop
from xert.sources import MotionFrame
from xert.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "xert/spec/xlerobot_left.yaml"
PROFILE = ROOT / "configs/teleop/xlerobot_bench.json"
SCRIPTED = {"kind": "scripted", "human_demonstrations": False, "generator": "tests/test_teleop.py"}


def test_key_mapping():
    assert [teleop.key_action(k, recording=False) for k in (" ", "d", "\x1b", "x", None)] == \
        ["start", None, "quit", None, None]
    assert [teleop.key_action(k, recording=True) for k in (" ", "d", "D", "\x1b", "x", None)] == \
        ["stop", "discard", "discard", "quit", None, None]


def calibration(clock):
    # What `xert calibrate --bench` does: neutral grasp = bench reset pose.
    return create_calibration(SPEC, PROFILE, scripted_frame(0, clock()), BenchBackend(load_spec(SPEC), 0))


def session(tmp_path, keys, make_mapper=None):
    """Keys are exact: every poll consumes one entry and each sequence ends with Escape at idle."""
    clock = ScriptedClock()
    neutral = calibration(clock)
    make_mapper = make_mapper or (lambda backend: CalibratedMapper(SPEC, PROFILE, neutral, backend))
    pending = iter(keys)
    kept = teleop.run_session(ScriptedSource(clock), SPEC, tmp_path / "session", make_mapper, provenance=SCRIPTED,
                              keys=lambda: next(pending), clock=clock, sleep=clock.sleep)
    assert next(pending, "done") == "done"
    return kept, tmp_path / "session"


def test_bench_profile_reaches_block_and_tray_from_modest_hand_motion():
    profile = json.loads(PROFILE.read_text())
    bench = BenchBackend(load_spec(SPEC), 0)
    neutral = calibration(ScriptedClock())
    mapper = CalibratedMapper(SPEC, PROFILE, neutral, bench)
    origin = np.asarray(neutral["sites"]["grasp"]["position"])
    np.testing.assert_allclose(origin, bench.data.site("grasp").xpos)  # No reset-to-neutral sweep.
    base, fk = scripted_frame(0, 1.).hands["right"], mujoco.MjData(bench.model)
    block = bench.data.body("block").xpos.copy()
    for name, target in (("block", block), ("above block", block + [0, 0, .06]),
                         ("tray", bench.model.body("tray").pos + [0, 0, .04])):
        image_x, image_y, depth = np.asarray(profile["axes"]).T @ (target - origin) / profile["gains"]
        scale = 1 / (1 + depth)  # Apparent palm-size change that the depth proxy needs.
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


def test_discard_writes_nothing_and_estop_never_publishes(tmp_path, monkeypatch):
    pytest.importorskip("lerobot")
    from xert.runtime import loop
    runs, real = [], loop.run_loop
    monkeypatch.setattr(loop, "run_loop", lambda *a, **k: runs.append(real(*a, **k)) or runs[-1])
    kept, output = session(tmp_path, [" ", None, "d", None, " ", None, "\x1b"])
    assert kept == []
    staging = [path for path in output.iterdir()]
    assert len(staging) == 1 and staging[0].name.startswith(".xert-recording-")  # Escape: recoverable journal.
    manifest = json.loads((staging[0] / "dataset/xert_recording.json").read_text())
    assert manifest["status"] == "aborted" and manifest["source"]["bench_seed"] == 0  # Discard retries seed 0.
    assert runs[-1][-1]["phase"] == "estop" and runs[-1][-2]["phase"] != "estop"


def test_stop_finalises_episode_that_round_trips(tmp_path):
    pytest.importorskip("lerobot")
    from xert.data.recorder import load_runtime_dataset, verify_recording
    # The Space buffered as episode 0 ends is drained; otherwise episode 1 would record 2 frames.
    kept, output = session(tmp_path, [" ", None, None, " ", " ", None, " ", " ", None, "\x1b"])
    assert [path.name for path in kept] == ["episode_000000", "episode_000001"]
    assert sorted(path.name for path in output.iterdir()) == ["episode_000000", "episode_000001"]
    assert verify_recording(kept[0])["frames"] == 3 and verify_recording(kept[1])["frames"] == 1
    manifest = json.loads((kept[0] / "xert_recording.json").read_text())
    assert manifest["source"]["kind"] == "scripted" and manifest["source"]["bench_seed"] == 0
    assert manifest["task_outcome"] == {"success": False, "failure_reason": None, "ended_by": "operator", "steps": 3}
    dataset = load_runtime_dataset(kept[0])
    assert dataset.features["next.success"]["dtype"] == "bool" and len(dataset) == 3
    assert not any(dataset[i]["next.success"].item() for i in range(3))
    del dataset


def test_bench_contact_success_reaches_next_success(tmp_path):
    pytest.importorskip("lerobot")
    from xert.data.recorder import load_runtime_dataset

    def expert(backend):
        # Scripted IK expert of the bench, still passed through the runtime safety clamps.
        limits = backend.env.limits
        return SimpleNamespace(map=lambda frame: limits[:, 0] + (np.asarray(backend.env.expert_action()) + 1) / 2 *
                               (limits[:, 1] - limits[:, 0]))

    truth, joints = [], []  # Object state straight from MjData and the env, not the native observation.

    def tracked(backend):
        def capture():
            env, data = backend.env, backend.data
            truth.append([*data.qpos[env.block_q:env.block_q + 7], *data.site_xpos[env.site],
                          *backend.model.body("tray").pos, env.steps / env.config["horizon"]])
            joints.append(data.qpos[env.qids].copy())
        real = backend.send_targets
        backend.send_targets = lambda q, t: (real(q, t), capture())
        capture()  # Reset state: what the policy sees before action 0.
        return expert(backend)

    kept, _ = session(tmp_path, [" "] + [None] * 600 + ["\x1b"], tracked)
    manifest = json.loads((kept[0] / "xert_recording.json").read_text())
    assert manifest["task_outcome"]["success"] and manifest["task_outcome"]["ended_by"] == "bench"
    dataset = load_runtime_dataset(kept[0])  # finish() already ran verify_recording over every frame.
    flags = [dataset[i]["next.success"].item() for i in range(len(dataset))]
    assert flags[-1] and not any(flags[:-1]) and len(flags) == manifest["task_outcome"]["steps"]
    feature = dataset.features["observation.environment_state"]
    native = BenchBackend(load_spec(SPEC), 0).env.metadata()["observation_names"]
    assert feature["dtype"] == "float32" and feature["names"] == native[12:]  # P5.3 looks these names up.
    observed = np.stack([dataset[i]["observation.environment_state"].numpy() for i in range(len(dataset))])
    # Frame k holds the state captured before action k, the same instant as its observation.state.
    np.testing.assert_array_equal(observed, np.asarray(truth[:-1], dtype=np.float32))
    np.testing.assert_array_equal(np.stack([dataset[i]["observation.state"].numpy() for i in range(len(dataset))]),
                                  np.asarray(joints[:-1], dtype=np.float32))
    assert np.ptp(observed[:, 2]) > .01  # The block really moves.
    del dataset


def test_cli_checks_inputs_before_camera_and_labels_webcam(tmp_path, monkeypatch):
    from xert.runtime import cli
    from xert.sources import webcam_mediapipe
    model = tmp_path / "model.task"
    model.write_bytes(b"mock")
    manifest = tmp_path / "assets/mediapipe/SOURCE.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"sha256": hashlib.sha256(b"mock").hexdigest()}))
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)
    monkeypatch.setattr(cli.time, "perf_counter", lambda: 100.)
    closed, seen = [], {}

    def camera(*args, **kwargs):
        return SimpleNamespace(read=lambda: replace(scripted_frame(1, 100.), source="webcam"),
                               close=lambda: closed.append(True))

    monkeypatch.setattr(webcam_mediapipe, "WebcamSource", camera)
    path = tmp_path / "calibration.json"
    cli.execute(SimpleNamespace(command="calibrate", bench=True, spec=SPEC, profile=PROFILE, model=model, camera=0,
                                output=path, timeout=1.))
    neutral = json.loads(path.read_text())
    assert neutral["neutral_pose"] == "bench_reset"
    np.testing.assert_allclose(neutral["sites"]["grasp"]["position"],
                               BenchBackend(load_spec(SPEC), 0).data.site("grasp").xpos)
    args = SimpleNamespace(command="teleop", spec=SPEC, profile=PROFILE, model=model, camera=0,
                           calibration=path, output=tmp_path / "out")
    monkeypatch.setattr(webcam_mediapipe, "WebcamSource", lambda *a, **k: pytest.fail("Camera opened"))
    monkeypatch.setattr(teleop.sys, "platform", "win32")  # input checks must not depend on the CI host
    for bad, message in (({"mirrored": True}, "unmirrored"), ({"neutral_pose": "spec_rest"}, "--bench")):
        path.write_text(json.dumps({**neutral, **bad}))
        with pytest.raises(ValueError, match=message):
            cli.execute(args)
    path.write_text(json.dumps(neutral))
    monkeypatch.setattr(teleop.sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="Windows console"):
        cli.execute(args)
    monkeypatch.setattr(teleop.sys, "platform", "win32")
    closed.clear()
    monkeypatch.setattr(webcam_mediapipe, "WebcamSource", camera)
    monkeypatch.setattr(teleop, "run_session", lambda source, spec, output, make_mapper, **kw: seen.update(kw))
    cli.execute(args)
    assert closed and seen["provenance"]["kind"] == "webcam" and seen["provenance"]["mirrored"] is False
    assert seen["provenance"]["calibration_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

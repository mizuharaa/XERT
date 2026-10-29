"""Terminal teleoperation on the pick-and-place bench: one LeRobot v3 dataset per kept episode."""
import hashlib
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from xert.spec import load_spec

TASK = "Pick up the orange block and place it in the green tray"
KEYS = "Space starts/stops an episode, d discards it, Esc is e-stop and quit"


def key_action(key, recording):
    """Pure key map. Escape always means e-stop/quit; other keys depend on the episode state."""
    if key == "\x1b":
        return "quit"
    if key == " ":
        return "stop" if recording else "start"
    if key in ("d", "D") and recording:
        return "discard"
    return None


def read_key():
    if sys.platform == "win32":
        import msvcrt
        return msvcrt.getwch() if msvcrt.kbhit() else None
    return None  # ponytail: Windows console keys only; Ctrl-C still stops through finally elsewhere.


def run_session(source, spec_path, output, make_mapper, *, provenance, spec_bytes=None, keys=read_key,
                clock=time.perf_counter, sleep=time.sleep):
    """Caller owns `source`. Bench seed = kept-episode index, so a discarded attempt retries its seed."""
    from xert.backends.bench import BenchBackend
    from xert.data.recorder import RuntimeRecorder
    from .loop import run_loop
    spec_bytes = Path(spec_path).read_bytes() if spec_bytes is None else spec_bytes
    spec = load_spec(spec_path, contents=spec_bytes)
    output = Path(output)
    output.mkdir(parents=True)  # Refuses an existing path: every session is fresh evidence.
    borrowed = SimpleNamespace(read=source.read, close=lambda: None)  # run_loop closes per episode.
    kept = []
    while True:
        seed = len(kept)
        print(f"Episode {seed} (bench seed {seed}) ready. {KEYS}.", flush=True)
        while (action := key_action(keys(), recording=False)) is None:
            sleep(.05)
        if action == "quit":
            return kept
        backend = BenchBackend(spec, seed)
        mapper = make_mapper(backend)
        recorder = RuntimeRecorder(spec_path, backend.read_state(), output / f"episode_{seed:06d}", task=TASK,
                                   source={**provenance, "bench_seed": seed, "bench": backend.env.metadata()},
                                   spec_bytes=spec_bytes, task_success=True)
        end = []

        def step(record):
            if end:
                return False  # E-stop hold tick after Escape; that recording is already ended.
            record["success"] = bool(backend.info["success"])
            recorder(record)
            if record["success"] or backend.finished:
                end.append("finish")
            elif pressed := key_action(keys(), recording=True):
                end.append(pressed)
            if recorder.count % spec.control_hz == 0 or end:
                info, grasp = backend.info, backend.data.site("grasp").xpos
                print(f"  t={recorder.count / spec.control_hz:5.1f}s grasp={grasp.round(3).tolist()} "
                      f"block={[round(v, 3) for v in info['object_position']]} "
                      f"held={info['object_in_robot_contact']} lifted={info['lifted']}", flush=True)
            return bool(end) and end[0] != "quit"

        print("Recording.", flush=True)
        try:
            run_loop(borrowed, backend, mapper, backend.env.config["horizon"] / spec.control_hz,
                     clock=clock, sleep=sleep, on_step=step, stop_requested=lambda: end == ["quit"])
        except BaseException as exc:
            if not recorder.closed:
                recorder._abort_preserving(exc)
            raise
        result = end[0] if end else "stop"
        while result != "quit" and keys() is not None:
            pass  # Drop keys buffered while the episode ended, so none starts the next one.
        if result == "discard":
            recorder.abort()
            shutil.rmtree(recorder.work)
            print("Discarded; nothing written.", flush=True)
        elif result == "quit":
            recorder.abort()
            print(f"E-stop. Episode not published; journal kept at {recorder.staging}", flush=True)
            return kept
        else:
            info = backend.info
            recorder.finish({"success": info["success"], "failure_reason": info["failure_reason"],
                             "ended_by": "operator" if result == "stop" else "bench", "steps": info["steps"]})
            kept.append(recorder.output)
            print(f"Saved {recorder.output} success={info['success']} reason={info['failure_reason']}", flush=True)


def execute(args):
    from xert.backends.bench import BenchBackend
    from xert.eval.ledger import source_provenance
    from xert.retarget.calibrate import CalibratedMapper, load_profile
    from xert.sources.webcam_mediapipe import WebcamSource
    from .cli import _calibration, _close, _pinned_model
    if sys.platform != "win32":
        raise RuntimeError("Teleop keys use the Windows console; no camera opened")
    if args.output.exists():
        raise ValueError("Output already exists; choose a fresh evidence path")
    spec_bytes, profile_bytes = args.spec.read_bytes(), args.profile.read_bytes()
    spec = load_spec(args.spec, contents=spec_bytes)
    profile = load_profile(args.profile, spec, contents=profile_bytes)
    model = _pinned_model(args.model)
    calibration_bytes, calibration = _calibration(args, model)
    if calibration.get("neutral_pose") != "bench_reset":
        raise ValueError("Teleop needs a bench-reset neutral calibration: xert calibrate --bench")

    def make_mapper(backend):
        return CalibratedMapper(args.spec, args.profile, calibration, backend,
                                spec_bytes=spec_bytes, profile_bytes=profile_bytes)

    make_mapper(BenchBackend(spec, 0))  # Reject a non-bench spec or bad calibration before the camera opens.
    provenance = {"kind": "webcam", "camera": args.camera, "mirrored": False, "model": model,
                  "profile": profile, "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
                  "calibration": calibration, "calibration_sha256": hashlib.sha256(calibration_bytes).hexdigest(),
                  "success_rule": "native bench contact rule (cpp/src/env.cpp Env::step)", **source_provenance()}
    source = WebcamSource(args.model, model["sha256"], camera=args.camera)
    try:
        run_session(source, args.spec, args.output, make_mapper, provenance=provenance, spec_bytes=spec_bytes)
    finally:
        _close(source)

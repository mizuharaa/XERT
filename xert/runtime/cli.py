"""Explicit webcam commands. Merely importing XERT does not access capture devices."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def add_commands(commands):
    for name in ("calibrate", "latency", "teleop"):
        parser = commands.add_parser(name, help="Explicitly open the selected webcam for simulation controls")
        parser.add_argument("--spec", type=Path, required=True)
        parser.add_argument("--profile", type=Path, required=True)
        parser.add_argument("--model", type=Path, default=ROOT / "build/models/hand_landmarker.task")
        parser.add_argument("--camera", type=int, default=0)
        parser.add_argument("--output", type=Path, required=True)
        if name == "calibrate":
            parser.add_argument("--timeout", type=float, default=30)
            parser.add_argument("--bench", action="store_true",
                                help="Neutral pose = pick-and-place bench reset pose (required by teleop)")
        else:
            parser.add_argument("--calibration", type=Path, required=True)
        if name == "latency":
            parser.add_argument("--duration", type=float, default=300)
    parser = commands.add_parser("record-motion", help="Explicitly open the selected webcam and save landmarks, not images")
    parser.add_argument("--model", type=Path, default=ROOT / "build/models/hand_landmarker.task")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=30)


def escape_pressed():
    if sys.platform == "win32":
        import msvcrt
        return msvcrt.kbhit() and msvcrt.getwch() == "\x1b"
    return False  # Ctrl-C terminates through finally on every platform.


def _close(resource):
    primary = sys.exception()
    try:
        resource.close()
    except BaseException as cleanup:
        if primary is None:
            raise
        primary.add_note(f"Cleanup also failed: {type(cleanup).__name__}: {cleanup}")


def _pinned_model(path):
    model = json.loads((ROOT / "assets/mediapipe/SOURCE.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != model["sha256"]:
        raise ValueError("Model checksum mismatch; run python -m scripts.fetch_hand_model")
    return model


def record_motion(args):
    from xert.eval.ledger import source_provenance
    from xert.sources.motion_file import dump_frame
    from xert.sources.webcam_mediapipe import WebcamSource
    if args.output.exists():
        raise ValueError("Output already exists; choose a fresh evidence path")
    if not math.isfinite(args.duration) or args.duration <= 0:
        raise ValueError("Duration must be finite and positive")
    model = _pinned_model(args.model)
    provenance = source_provenance()
    print("Hold the hand(s) still in a neutral pose for the first two seconds, then move. "
          "Escape or Ctrl-C ends the clip early.", flush=True)
    time.sleep(3)
    frames = []
    source = WebcamSource(args.model, model["sha256"], camera=args.camera)
    try:
        end = time.perf_counter() + args.duration
        try:
            while time.perf_counter() < end and not escape_pressed():
                frame = source.read()
                if frame is not None:
                    frames.append(frame)
                time.sleep(.005)
        except KeyboardInterrupt:
            pass  # Ctrl-C ends the clip like Escape; frames already inferred are kept.
    finally:
        _close(source)
    if not frames:
        raise RuntimeError("No landmark frames were inferred; nothing written")
    clip = {"version": 1, "source": "webcam", "camera": args.camera, "mirrored": False, "model": model,
            **provenance, "requested_duration_s": args.duration, "captured": source.captured,
            "inferred": source.inferred, "capture_overwritten": source.overwritten,
            "frames": [dump_frame(frame) for frame in frames]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(clip, stream, allow_nan=False)
    print(f"Saved {len(frames)} landmark frames (no images) to {args.output}")


def _calibration(args, model):
    calibration_bytes = args.calibration.read_bytes()
    calibration = json.loads(calibration_bytes)
    if (calibration.get("source") != "webcam" or calibration.get("camera") != args.camera or
            calibration.get("model_sha256") != model["sha256"] or calibration.get("mirrored") is not False):
        raise ValueError("Live run requires a matching, unmirrored webcam calibration")
    return calibration_bytes, calibration


def execute(args):
    if args.command == "record-motion":
        return record_motion(args)
    if args.command == "teleop":
        from .teleop import execute as teleop
        return teleop(args)
    from xert.backends import MuJoCoBackend
    from xert.backends.bench import BenchBackend
    from xert.eval.ledger import append_row, artifact_path, new_row, source_provenance
    from xert.retarget.calibrate import CalibratedMapper, create_calibration, load_profile, required_hands
    from xert.sources.webcam_mediapipe import WebcamSource
    from xert.spec import load_spec
    from .latency import summarize
    from .loop import run_loop
    if args.output.exists():
        raise ValueError("Output already exists; choose a fresh evidence path")
    interval = args.timeout if args.command == "calibrate" else args.duration
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("Duration/timeout must be finite and positive")
    spec_bytes, profile_bytes = args.spec.read_bytes(), args.profile.read_bytes()
    spec = load_spec(args.spec, contents=spec_bytes)
    profile = load_profile(args.profile, spec, contents=profile_bytes)
    model = _pinned_model(args.model)
    bench = getattr(args, "bench", False)
    backend = BenchBackend(spec, 0) if bench else MuJoCoBackend(spec)
    try:
        if args.command == "calibrate":
            print("Hold the mapped hand(s) in a comfortable neutral pose. Capture begins after a short countdown.", flush=True)
            time.sleep(3)
            source = WebcamSource(args.model, model["sha256"], camera=args.camera)
            try:
                end = time.perf_counter() + args.timeout
                while time.perf_counter() < end:
                    frame = source.read()
                    if (frame is not None and required_hands(profile) <= set(frame.hands) and
                            0 <= time.perf_counter() - frame.t_capture <= .2):
                        calibration = create_calibration(args.spec, args.profile, frame, backend,
                                                         spec_bytes=spec_bytes, profile_bytes=profile_bytes)
                        calibration.update(camera=args.camera, model_sha256=model["sha256"], mirrored=False,
                                           neutral_pose="bench_reset" if bench else "spec_rest")
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        with args.output.open("x", encoding="utf-8") as stream:
                            json.dump(calibration, stream, indent=2, allow_nan=False)
                        return
                    time.sleep(.01)
                raise RuntimeError("No fresh complete neutral hand pose observed before timeout")
            finally:
                _close(source)
        calibration_bytes, calibration = _calibration(args, model)
        mapper = CalibratedMapper(args.spec, args.profile, calibration, backend,
                                  spec_bytes=spec_bytes, profile_bytes=profile_bytes)
        provenance = source_provenance()
        config = {"embodiment": spec.name, "joint_names": list(spec.joint_names), "backend": "mujoco",
                  "source": "webcam", "camera": args.camera, "mirrored": False,
                  "spec": artifact_path(args.spec), "profile": artifact_path(args.profile),
                  "spec_snapshot": artifact_path(args.output / "spec.yaml"),
                  "profile_snapshot": artifact_path(args.output / "profile.json"),
                  "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
                  "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
                  "calibration": artifact_path(args.output / "calibration.json"),
                  "calibration_sha256": hashlib.sha256(calibration_bytes).hexdigest(),
                  "duration_s": args.duration,
                  "model": model, "control_hz": spec.control_hz,
                  "latency_endpoint": "host acquisition return to MuJoCo step completion; excludes sensor/driver buffering",
                  **provenance}
        args.output.mkdir(parents=True)
        (args.output / "spec.yaml").write_bytes(spec_bytes)
        (args.output / "profile.json").write_bytes(profile_bytes)
        with (args.output / "calibration.json").open("xb") as snapshot:
            snapshot.write(calibration_bytes)
        config_path, trace_path = args.output / "config.json", args.output / "trace.jsonl"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        source = WebcamSource(args.model, model["sha256"], camera=args.camera)
        try:
            with trace_path.open("x", encoding="utf-8") as trace:
                def record(row):
                    trace.write(json.dumps(row, allow_nan=False) + "\n")
                    trace.flush()
                # run_loop closes both resources, including exceptions/keyboard interruption.
                records = run_loop(source, backend, mapper, args.duration, on_step=record, stop_requested=escape_pressed)
        finally:
            _close(source)
        if len(records) != math.ceil(args.duration * spec.control_hz):
            raise RuntimeError("Run stopped early; trace retained but no completed-run latency rows appended")
        summary = summarize(records)
        summary.update(captured=source.captured, inferred=source.inferred, capture_overwritten=source.overwritten)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        artifacts = [config_path, trace_path, args.output / "summary.json", args.output / "spec.yaml",
                     args.output / "profile.json", args.output / "calibration.json"]
        artifact_hashes = {artifact_path(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts}
        for percentile in (50, 95):
            row = new_row(**provenance, slice="P3.5", experiment="webcam_host_latency_v1", backend="mujoco",
                          embodiment=spec.name, metric=f"latency_p{percentile}_ms", value=summary[f"p{percentile}_ms"],
                          ci95=None, n=summary["n"], seeds={"train": [], "eval": []},
                          config_path=artifact_path(config_path),
                          artifact_paths=list(artifact_hashes), artifact_sha256=artifact_hashes,
                          notes="Unique fresh valid applied webcam frames only; conditional on successful application. "
                                "Host acquisition-return to simulator step completion; "
                                "sensor exposure/driver latency excluded. No motor, tracking-accuracy or task-success claim.")
            append_row(row)
            print(json.dumps(row))
    finally:
        _close(backend)

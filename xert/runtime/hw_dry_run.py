"""SO-101 dry-run: clip -> calibrated retarget -> SafetyController -> hardware-limit clamp -> print. No hardware.

Writes a trace only. Never appends ledger rows: nothing here is a hardware measurement.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
LABEL = "dry-run, no hardware"


def add_command(commands):
    parser = commands.add_parser("hw-dry-run", help="Print SO-101 motor-space targets for a clip; no bus, no motors")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=ROOT / "configs/teleop/so101.json")


class SimClock:
    """Simulated time: deterministic replay, and host speed cannot cause deadline overruns."""

    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


def execute(args):
    from xert.backends import MuJoCoBackend
    from xert.backends.so101_hw_backend import SO101HardwareBackend
    from xert.retarget.calibrate import CalibratedMapper, create_calibration, load_profile, required_hands
    from xert.sources.motion_file import MotionFileSource, load_clip
    from xert.spec import load_spec
    from .loop import run_loop
    if args.output.exists():
        raise ValueError("Output already exists; choose a fresh output path")
    inputs = {"spec_bytes": args.spec.read_bytes(), "profile_bytes": args.profile.read_bytes()}
    map_bytes, clip_bytes = args.map.read_bytes(), args.clip.read_bytes()
    spec = load_spec(args.spec, contents=inputs["spec_bytes"])
    profile = load_profile(args.profile, spec, contents=inputs["profile_bytes"])
    clip, frames = load_clip(args.clip, contents=clip_bytes)
    hardware = SO101HardwareBackend(spec, args.map, dry_run=True)
    # The mapper's IK needs a kinematic model; this MuJoCo copy is never stepped, only seeded with commanded q.
    kinematics = MuJoCoBackend(spec)
    try:
        # ponytail: the clip calibrates itself, as in P3.6; neutral = first frame showing every mapped hand.
        neutral = next((f for f in frames if required_hands(profile) <= set(f.hands)), None)
        if neutral is None:
            raise ValueError("Clip never shows every mapped hand; cannot calibrate")
        calibration = create_calibration(args.spec, args.profile, neutral, kinematics, **inputs)
        mapper = CalibratedMapper(args.spec, args.profile, calibration, kinematics, **inputs)

        def map_frame(frame):
            kinematics.data.qpos[kinematics.qids] = hardware.read_state().q  # IK seed = last command, not measured
            return mapper.map(frame)

        args.output.mkdir(parents=True)
        config = {"mode": LABEL, "hardware": False, "motor_commands_sent": 0, "ledger_rows": 0,
                  "spec": str(args.spec), "spec_sha256": hashlib.sha256(inputs["spec_bytes"]).hexdigest(),
                  "map": str(args.map), "map_sha256": hashlib.sha256(map_bytes).hexdigest(),
                  "clip": str(args.clip), "clip_sha256": hashlib.sha256(clip_bytes).hexdigest(),
                  "clip_source": clip["source"], "calibration_frame_sequence": neutral.sequence,
                  "clock": "simulated time", "state": "commanded targets echoed back, not measured"}
        (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        clock = SimClock()
        with (args.output / "trace.jsonl").open("x", encoding="utf-8") as trace:
            def on_step(record):
                trace.write(json.dumps({**record, "mode": LABEL, "state_measured": False,
                                        "dry_run_targets": hardware.last_sent}, allow_nan=False) + "\n")

            duration = frames[-1].t_landmarks - frames[0].t_capture + 1 / spec.control_hz
            records = run_loop(MotionFileSource(frames, clock=clock), hardware, SimpleNamespace(map=map_frame),
                               duration, clock=clock, sleep=clock.sleep, on_step=on_step)
    finally:
        kinematics.close()
        hardware.close()
    unmapped = [r["joint"] for r in hardware.last_sent if r["status"] == "unmapped"]
    print(f"[{LABEL}] {len(records)} ticks to {args.output / 'trace.jsonl'}; unmapped joints: {unmapped or 'none'}. "
          "No motor commands sent; no ledger rows appended.")

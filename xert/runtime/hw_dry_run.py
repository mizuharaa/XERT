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
    parser.add_argument("--seed-from", type=Path, help="read.json from `xert hw-read`: start pose instead of rest")


def seed_from(read_bytes, spec, map_path, map_bytes):
    """Start pose from a hw-read read.json, RECOMPUTED from its native readings with this run's validated map.

    The saved radians are never trusted. Refused: a different motor table, valid_for_seeding false, an incompatible
    or TBD map, a map other than the one the read was converted with, a map fitted on another calibration, a
    motor-unit round trip off by more than 1e-9, or a pose outside the spec position limits.
    """
    import numpy as np
    from xert.backends.so101_hw_backend import MOTOR_TABLE, SO101HardwareBackend, compatible_map, to_radians
    read = json.loads(read_bytes)
    if read.get("motor_table") != [{"joint": m, "motor": m, "id": i, "units": u} for m, i, u in MOTOR_TABLE]:
        raise ValueError("Seed read has a different motor table than the SO-101 follower")
    if read.get("valid_for_seeding") is not True:
        raise ValueError(f"Seed read is not valid for seeding: {read.get('invalid_reason')}")
    entries, map_sha = compatible_map(map_path, spec, read["calibration"]["sha256"], contents=map_bytes)
    if "joint_map" in read and read["joint_map"]["sha256"] != map_sha:
        raise ValueError("Seed read was converted with a different joint map than --map")
    native = np.array([read["normalized"][m] for m, _, _ in MOTOR_TABLE], dtype=np.float64)
    seed = to_radians(entries, native)
    back = np.array([SO101HardwareBackend._to_motor(e, q) for e, q in zip(entries, seed)])
    if not np.allclose(back, native, rtol=0, atol=1e-9):
        raise ValueError("Seed radians do not map back to the saved motor readings within 1e-9")
    limits = spec.position_limits
    if np.any(seed < limits[:, 0]) or np.any(seed > limits[:, 1]):
        raise ValueError(f"Seed pose {seed.tolist()} is outside the spec position limits")
    return seed


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
    seed_bytes = None if args.seed_from is None else args.seed_from.read_bytes()
    initial = None if seed_bytes is None else seed_from(seed_bytes, spec, args.map, map_bytes)
    hardware = SO101HardwareBackend(spec, args.map, dry_run=True, initial=initial)
    # The mapper's IK needs a kinematic model; this MuJoCo copy is never stepped, only seeded with commanded q.
    kinematics = MuJoCoBackend(spec)
    if initial is not None:  # The neutral calibration stores this model's grasp pose: start it from the seed too.
        import mujoco
        kinematics.data.qpos[kinematics.qids] = initial
        mujoco.mj_forward(kinematics.model, kinematics.data)
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
                  "clock": "simulated time", "state": "commanded targets echoed back, not measured",
                  "start_pose": "spec rest, assumed" if initial is None else {
                      "source": "hw-read read.json, radians recomputed with --map", "path": str(args.seed_from),
                      "sha256": hashlib.sha256(seed_bytes).hexdigest(), "radians": initial.tolist()}}
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

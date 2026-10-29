"""Measure how closely each simulated robot follows its retarget targets on one recorded landmark clip."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from xert.backends import MuJoCoBackend
from xert.eval.ledger import append_row, artifact_path, new_row, source_provenance, validate_row
from xert.retarget.calibrate import CalibratedMapper, create_calibration, load_profile, required_hands
from xert.runtime.loop import run_loop
from xert.sources.motion_file import MotionFileSource, load_clip
from xert.spec import load_spec

ROBOTS = ("so101", "g1_upper", "leap_hand")
TARGETS = {
    "pose": "target = the calibrated mapper's requested IK site position; includes IK residual for unreachable "
            "targets. Orientation and gripper commands are requested but not scored.",
    "curls": "LEAP maps finger-bend changes to joints, so there is no IK target: target = forward kinematics of the "
             "mapper's requested joint vector at the fingertip sites. This is NOT human fingertip position error.",
}
COMMON = ("Per unique applied frame, measured after its first MuJoCo control step, pooled over sites {sites}. "
          "Neutral = first clip frame showing every mapped hand (excluded). Targets are neutral-relative control "
          "coordinates (image XY, palm-size depth proxy), not metric human motion. Error includes safety limiting, "
          "one-step servo lag and steady-state servo sag under gravity. Simulated-time replay: no latency, "
          "task-success or hardware claim.")


class SimClock:
    """Replay in simulated time: deterministic, and host compute never causes deadline overruns."""

    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


def measure(robot, frames, output):
    spec_path, profile_path = ROOT / f"xert/spec/{robot}.yaml", ROOT / f"configs/teleop/{robot}.json"
    inputs = {"spec_bytes": spec_path.read_bytes(), "profile_bytes": profile_path.read_bytes()}
    spec = load_spec(spec_path, contents=inputs["spec_bytes"])
    backend = MuJoCoBackend(spec)
    try:
        profile = load_profile(profile_path, spec, contents=inputs["profile_bytes"])
        # ponytail: the clip calibrates itself, so neutral and motion share one camera session and framing.
        neutral = next((f for f in frames if required_hands(profile) <= set(f.hands)), None)
        if neutral is None:
            raise ValueError(f"Clip never shows every hand {robot} maps; cannot calibrate")
        calibration = create_calibration(spec_path, profile_path, neutral, backend, **inputs)
        mapper = CalibratedMapper(spec_path, profile_path, calibration, backend, **inputs)
        scratch = mujoco.MjData(backend.model)
        requested = {}

        def map_frame(frame):
            requested["q"] = mapper.map(frame)
            return requested["q"]

        def targets():
            if profile["mode"] == "pose":
                return {s: np.asarray(t[0] if isinstance(t, tuple) else t) for s, t in mapper.targets.items()}
            scratch.qpos[:] = backend.data.qpos
            scratch.qpos[backend.qids] = requested["q"]
            mujoco.mj_kinematics(backend.model, scratch)
            return {s: scratch.site(s).xpos.copy() for s in spec.sites}

        output.mkdir(parents=True)
        (output / "spec.yaml").write_bytes(inputs["spec_bytes"])
        (output / "profile.json").write_bytes(inputs["profile_bytes"])
        (output / "calibration.json").write_text(json.dumps(calibration, indent=2, allow_nan=False), encoding="utf-8")
        errors = {}
        clock = SimClock()
        source = MotionFileSource(frames, clock=clock)
        with (output / "trace.jsonl").open("x", encoding="utf-8") as trace:
            def on_step(record):
                result = None
                if record["latency_sample"] and record["frame_sequence"] > neutral.sequence:
                    wanted = targets()
                    achieved = {s: backend.data.site(s).xpos.copy() for s in wanted}
                    error = {s: float(np.linalg.norm(wanted[s] - achieved[s]) * 1000) for s in wanted}
                    for site, value in error.items():
                        errors.setdefault(site, []).append(value)
                    result = {"target": {s: p.tolist() for s, p in wanted.items()},
                              "achieved": {s: p.tolist() for s, p in achieved.items()}, "error_mm": error,
                              "requested_q": requested["q"].tolist()}
                trace.write(json.dumps({**record, "retarget": result}, allow_nan=False) + "\n")

            duration = frames[-1].t_landmarks - frames[0].t_capture + 1 / spec.control_hz
            records = run_loop(source, backend, SimpleNamespace(map=map_frame), duration,
                               clock=clock, sleep=clock.sleep, on_step=on_step)
    finally:
        backend.close()
    pooled = [e for values in errors.values() for e in values]
    if not pooled:
        raise ValueError(f"No applied post-neutral frames for {robot}; no retarget error can be reported")
    summary = {"embodiment": spec.name, "mode": profile["mode"], "n": len(pooled),
               "frames": len(next(iter(errors.values()))), "p50_mm": float(np.percentile(pooled, 50)),
               "p95_mm": float(np.percentile(pooled, 95)), "max_mm": float(max(pooled)),
               "sites": {s: {"n": len(v), "p50_mm": float(np.percentile(v, 50)), "p95_mm": float(np.percentile(v, 95))}
                         for s, v in errors.items()},
               "calibration_frame_sequence": neutral.sequence, "clip_frames": len(frames),
               "replay_skipped_frames": source.skipped, "ticks": len(records),
               "clamped_ticks": sum(r["clamped"] for r in records), "clock": "simulated time"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robots", nargs="+", choices=ROBOTS, default=list(ROBOTS))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Evidence output already exists; choose a new --output")
    provenance = source_provenance()
    clip_bytes = args.clip.read_bytes()
    clip, frames = load_clip(args.clip, contents=clip_bytes)
    rows = []
    for robot in args.robots:
        directory = args.output / robot
        summary = measure(robot, frames, directory)
        config_path = directory / "config.json"
        config = {"embodiment": summary["embodiment"], "backend": "mujoco", "clip": artifact_path(args.clip),
                  "clip_sha256": hashlib.sha256(clip_bytes).hexdigest(), "clip_source": clip["source"],
                  "replayed_as": frames[0].source, "camera": clip.get("camera"),
                  "model_sha256": clip.get("model", {}).get("sha256"), "clip_git_commit": clip.get("git_commit"),
                  "target": TARGETS[summary["mode"]], **provenance}
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        artifacts = [args.clip, config_path] + [directory / name for name in
                     ("summary.json", "trace.jsonl", "calibration.json", "spec.yaml", "profile.json")]
        hashes = {artifact_path(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts}
        for percentile in (50, 95):
            row = new_row(**provenance, slice="P3.6", experiment="recorded_human_retarget_v1", backend="mujoco",
                          embodiment=summary["embodiment"], metric=f"retarget_ee_err_mm_p{percentile}",
                          value=summary[f"p{percentile}_mm"], ci95=None, n=summary["n"],
                          seeds={"train": [], "eval": []}, config_path=artifact_path(config_path),
                          artifact_paths=list(hashes), artifact_sha256=hashes,
                          notes=TARGETS[summary["mode"]] + " " + COMMON.format(sites=", ".join(summary["sites"])))
            validate_row(row)
            rows.append(row)
        print(json.dumps({k: summary[k] for k in ("embodiment", "n", "p50_mm", "p95_mm")}))
    # Rows are appended only after every robot finished, and only for real webcam recordings.
    if clip["source"] != "webcam":
        print(f"Clip source is {clip['source']!r}, not a webcam recording: artifacts written, no ledger rows appended.")
        return
    for row in rows:
        append_row(row)
        print(json.dumps(row))


if __name__ == "__main__":
    main()

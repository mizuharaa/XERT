"""`xert replay-real`: open-loop replay of a recorded SO-101 run in MuJoCo (PLAN.md D, sysid step 1).

Each episode starts a fresh simulator at that episode's first recorded `state`, then holds every recorded
`action` (goal positions, radians) on the Menagerie position actuators for one recorded frame, and compares the
simulated joint positions with the recorded `state` frame by frame. This is NOT system identification: no
parameter is fitted, and the servo's internal PID differs from MuJoCo's position actuator (see
`fit_actuator_params`, reserved for Daniel).

Alignment: sim[0] = state[0]; sim[t+1] is the result of action[t] held for 1/fps. RMSE uses frames 1..T-1, so
an episode of T frames contributes n = T-1 comparisons per joint (frame 0 is equal by construction).
"""
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import re
import tempfile

import numpy as np

from xert.data.lerobot_export import _publish
from xert.data.lerobot_io import HARDWARE, ROOT, _fresh, _hashes
from xle_workbench.storage import atomic_json, sha256

SPEC = ROOT / "xert/spec/so101.yaml"
# LeRobot writes timestamp = frame_index / fps, so a mismatch means edited or resampled frames. It cannot detect
# control-loop overruns: the recorded timestamp is not a measured time.
TIMESTAMP_TOLERANCE_S = 1e-4
# Owner decision: the same 1e-6 rad slack as to-store; float32-degree round trips land up to ~1.1e-7 rad past a limit.
LIMIT_SLACK_RAD = 1e-6
NOTES = ("Open-loop replay of recorded goal positions (action) on the MuJoCo SO-101 position actuators, each "
         "episode reset to its recorded first state; RMSE against recorded observation.state. Not system "
         "identification: no parameter was fitted, and the servo's internal PID differs from the MuJoCo position "
         "actuator. MuJoCo ran this; the hardware only recorded the reference.")


def load_import(dataset):
    """Slice C's frame-pair import from a hardware recording, in radians through a complete joint map, or refuse."""
    from xert.spec import load_spec
    dataset = Path(dataset)
    if not (dataset / "import.json").exists():
        if (dataset / "meta/info.json").exists():
            raise ValueError("A LeRobot dataset, not an import: run `xert import-lerobot --motor-units degrees` first")
        raise ValueError("No import.json: --dataset must be an `xert import-lerobot` output")
    record = json.loads((dataset / "import.json").read_text("utf-8"))
    if record.get("derived") is not False or record.get("source_kind") != "so101_follower":
        raise ValueError("replay-real takes a hardware frame-pair import (import-lerobot path c) only")
    if record.get("units") != "radians":
        raise ValueError(f"replay-real refuses {record.get('units')!r} values; import through a joint map in radians")
    if not re.fullmatch(r"[0-9a-f]{64}", str((record.get("joint_map") or {}).get("sha256"))):
        raise ValueError("import.json records no joint-map sha256; radians need a complete, recorded joint map")
    spec_bytes = SPEC.read_bytes()
    if record.get("embodiment") != "so101" or sha256(SPEC) != record.get("spec_sha256"):
        raise ValueError("The current so101 spec differs from the one the import was converted with")
    spec = load_spec(SPEC, contents=spec_bytes)
    joints = [f"{name}.pos" for name in spec.joint_names]
    if record["columns"]["state"] != joints or record["columns"]["action"] != joints:
        raise ValueError(f"Import state/action columns must be exactly {joints}")
    fps = record["fps"]
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid fps {fps!r}")
    with np.load(dataset / "frames.npz", allow_pickle=False) as frames:
        frames = dict(frames)
    if not np.allclose(frames["timestamp"], frames["frame_index"] / fps, rtol=0, atol=TIMESTAMP_TOLERANCE_S):
        raise ValueError("Timestamps differ from frame_index / fps; replay never resamples a recording")
    return record, spec, frames


def replay_timing(fps, asset_timestep):
    """(dt, k): k = ceil(1/(fps*asset dt)) physics steps per frame and dt = 1/(fps*k), never above the asset dt.

    30 fps on the 0.005 s asset: 1/(30*0.005) = 6.67 steps does not divide, so k = 7 and dt = 1/210 s.
    """
    k = math.ceil(1 / (fps * asset_timestep) - 1e-9)
    return 1 / (fps * k), k


def start_episode(spec, fps, dt, q0):
    """A new backend and MjData for every episode (review R8): no sim state carries over; qpos = q0, qvel 0."""
    import mujoco
    from xert.backends import MuJoCoBackend
    backend = MuJoCoBackend(replace(spec, control_hz=fps), timestep=dt)  # Checks fps*dt*k == 1 exactly.
    backend.data.qpos[backend.qids] = q0
    backend.data.qvel[:] = 0  # ponytail: recorded velocity is not in the import; the arm is assumed at rest.
    mujoco.mj_forward(backend.model, backend.data)
    return backend


def run_episode(backend, action):
    """(sim joint positions, sim times) for frames 0..T-1; frame t+1 follows action[t] held for one frame."""
    q, times = [backend.read_state().q], [backend.data.time]
    for target in action[:-1]:  # The last action has no recorded successor state.
        backend.send_targets(target, backend.data.time)  # Refuses targets past the limits by >1e-10 rad.
        q.append(backend.read_state().q)
        times.append(backend.data.time)
    return np.array(q), np.array(times)


def within_limits(action, limits):
    """(action clipped to the spec limits, max excess in rad); refuses anything past a limit by more than the slack."""
    excess = float(np.max(np.maximum(action - limits[:, 1], limits[:, 0] - action), initial=0.0))
    if excess > LIMIT_SLACK_RAD:
        raise ValueError(f"Recorded action {excess:.3g} rad outside the spec limits (slack {LIMIT_SLACK_RAD}); refused")
    return np.clip(action, limits[:, 0], limits[:, 1]), max(excess, 0.0)


def _rmse(errors):
    return np.sqrt(np.mean(np.square(errors), axis=0))


def _plots(staging, names, traces, fps):
    """One PNG per joint: real vs sim position over the episodes in order, and the error below."""
    try:
        import matplotlib
    except ImportError:
        return "skipped: matplotlib is not installed in this environment (tables still written)"
    matplotlib.use("Agg")  # No window, ever.
    import matplotlib.pyplot as plt
    (staging / "plots").mkdir()
    for j, name in enumerate(names):
        figure, (position, error) = plt.subplots(2, 1, sharex=True, figsize=(8, 4.5), height_ratios=(2, 1))
        offset = 0.0
        for e, (real, sim, _, _) in enumerate(traces):
            t = offset + np.arange(len(real)) / fps
            position.plot(t, real[:, j], color="#2a78d6", lw=2, label="recorded state" if e == 0 else None)
            position.plot(t, sim[:, j], color="#eb6834", lw=2, ls="--", label="sim" if e == 0 else None)
            error.plot(t, sim[:, j] - real[:, j], color="#52514e", lw=1.5)
            if e:
                for axis in (position, error):
                    axis.axvline(offset, color="#c3c2b7", lw=1)
            offset = t[-1] + 1 / fps
        position.set_ylabel("position (rad)")
        error.set_ylabel("sim - real (rad)")
        error.set_xlabel("time (s), episodes in order, separated by vertical lines")
        position.set_title(f"{name}: open-loop replay vs recorded state")
        position.legend(frameon=False)
        for axis in (position, error):
            axis.grid(color="#e6e5e0", lw=0.8)
            axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        figure.savefig(staging / f"plots/{name}.png", dpi=110)
        plt.close(figure)
    return "written"


def _check_raw(dataset, record):
    """Ledger rows only for a real hardware import whose raw/ copy still matches the recorded source hashes."""
    if record.get("lerobot_robot_type") not in HARDWARE or not (dataset / "raw").is_dir() \
            or _hashes(dataset / "raw") != record.get("source_sha256"):
        raise ValueError("Refusing ledger rows: no verified raw/ hardware recording behind this import")


def replay_real(dataset, output, *, ledger=False):
    """Fresh output: trace.csv, summary.csv/.md, plots/ (when matplotlib exists) and replay.json with every sha256."""
    dataset, output = Path(dataset).absolute(), _fresh(output)
    record, spec, frames = load_import(dataset)
    if ledger:
        _check_raw(dataset, record)
    fps = record["fps"]
    asset_dt = spec.load_model().opt.timestep
    dt, substeps = replay_timing(fps, asset_dt)
    names, traces, max_excess = list(spec.joint_names), [], 0.0
    for episode in np.unique(frames["episode_index"]):
        rows = frames["episode_index"] == episode
        real, action = frames["state"][rows].astype(np.float64), frames["action"][rows].astype(np.float64)
        if len(real) < 2:
            raise ValueError(f"Episode {episode} has one frame; nothing to compare")
        action, excess = within_limits(action, spec.position_limits)
        max_excess = max(max_excess, excess)
        sim, times = run_episode(start_episode(spec, fps, dt, real[0]), action)
        traces.append((real, sim, int(episode), times))
    summary = []
    for real, sim, episode, _ in traces:
        summary += [{"episode": episode, "joint": n, "n": len(real) - 1, "rmse_rad": r}
                    for n, r in zip(names, _rmse(sim[1:] - real[1:]))]
    pooled = np.concatenate([sim[1:] - real[1:] for real, sim, _, _ in traces])
    summary += [{"episode": "pooled", "joint": n, "n": len(pooled), "rmse_rad": r} for n, r in zip(names, _rmse(pooled))]
    for row in summary:
        row["rmse_rad"] = float(row["rmse_rad"])
        row["rmse_deg"] = math.degrees(row["rmse_rad"])  # Derived column.
    import_sha256, frames_sha256 = sha256(dataset / "import.json"), sha256(dataset / "frames.npz")
    with tempfile.TemporaryDirectory(prefix=".xert-replay-", dir=output.parent) as temporary:
        staging = Path(temporary) / "replay"
        staging.mkdir()
        with (staging / "trace.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["episode_index", "frame_index", "sim_time_s", *(f"real_{n}" for n in names),
                             *(f"sim_{n}" for n in names)])
            for real, sim, episode, times in traces:
                for t in range(len(real)):
                    writer.writerow([episode, t, repr(float(times[t])), *map(repr, real[t].tolist()), *map(repr, sim[t].tolist())])
        with (staging / "summary.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["episode", "joint", "n", "rmse_rad", "rmse_deg"])
            writer.writeheader()
            writer.writerows(summary)
        plots = _plots(staging, names, traces, fps)
        lines = ["# Real-vs-sim replay (open loop)", "", NOTES, "",
                 f"- import: `{dataset}` (import.json sha256 `{import_sha256}`)",
                 f"- joint map sha256 `{record['joint_map']['sha256']}`",
                 f"- {fps} fps; dt {dt!r} s = 1/({fps}*{substeps}), {substeps} physics steps per frame "
                 f"(asset dt {asset_dt!r} s); every recorded frame used, none resampled",
                 f"- actions clipped to the spec limits within {LIMIT_SLACK_RAD} rad; max excess {max_excess!r} rad",
                 f"- n = frames compared per joint (frames 1..T-1 of each episode); plots: {plots}", "",
                 "| episode | joint | n | RMSE (rad) | RMSE (deg) |", "|---|---|---|---|---|"]
        lines += [f"| {r['episode']} | {r['joint']} | {r['n']} | {r['rmse_rad']:.6g} | {r['rmse_deg']:.6g} |"
                  for r in summary]
        (staging / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        artifacts = {p.relative_to(staging).as_posix(): sha256(p) for p in sorted(staging.rglob("*")) if p.is_file()}
        manifest = {"version": 1, "command": "replay-real", "dataset": str(dataset), "import_sha256": import_sha256,
                    "frames_sha256": frames_sha256, "joint_map_sha256": record["joint_map"]["sha256"],
                    "spec_sha256": record["spec_sha256"], "model_path": str(spec.model_path),
                    "model_sha256": sha256(spec.model_path), "fps": fps, "timestep_s": dt,
                    "substeps_per_frame": substeps, "asset_timestep_s": asset_dt,
                    "action_limit_slack_rad": LIMIT_SLACK_RAD, "action_max_limit_excess_rad": max_excess,
                    "timing_rule": "k = ceil(1/(fps*asset_dt)), dt = 1/(fps*k); MuJoCoBackend checks 1/fps per frame (1e-9 s)",
                    "alignment": "sim[0] = state[0]; sim[t+1] after action[t] held 1/fps; RMSE over frames 1..T-1",
                    "reset": "fresh MjData per episode at its recorded first state, qvel 0; nothing carried over",
                    "episodes": [episode for _, _, episode, _ in traces], "plots": plots, "notes": NOTES,
                    "summary": summary, "artifact_sha256": artifacts}
        atomic_json(staging / "replay.json", manifest)
        _publish(staging, output)
    if ledger:
        _append_rows(output, record, manifest)
    return {"output": str(output), "timestep_s": dt, "substeps_per_frame": substeps,
            "pooled_rmse_rad": {r["joint"]: r["rmse_rad"] for r in summary if r["episode"] == "pooled"}}


def _append_rows(output, record, manifest):
    from xert.eval.ledger import append_row, artifact_path, new_row, source_provenance
    provenance = source_provenance()
    hashes = {artifact_path(output / name): digest for name, digest in manifest["artifact_sha256"].items()}
    hashes[artifact_path(output / "replay.json")] = sha256(output / "replay.json")
    for row in (r for r in manifest["summary"] if r["episode"] == "pooled"):
        entry = new_row(**provenance, slice="D", experiment="replay_real_v1", backend="mujoco",
                        reference_backend="so101_hw", embodiment="so101", metric="replay_position_rmse_rad",
                        joint=row["joint"], value=row["rmse_rad"], ci95=None, n=row["n"],
                        seeds={"train": [], "eval": []}, config_path=artifact_path(output / "replay.json"),
                        artifact_paths=list(hashes), artifact_sha256=hashes,
                        import_sha256=manifest["import_sha256"], joint_map_sha256=manifest["joint_map_sha256"],
                        notes=f"{NOTES} Pooled over {len(manifest['episodes'])} episodes; per-episode in summary.csv.")
        append_row(entry)
        print(json.dumps(entry))


def fit_actuator_params(dataset, *, held_out_episode=-1):
    """TODO(daniel): fit the MuJoCo SO-101 actuator/joint parameters to a recorded hardware run.

    Inputs:
      dataset: an import that `load_import` accepts (hardware frame pairs, radians, complete joint map).
      held_out_episode: position in the dataset's episode order excluded from fitting (default: the last one).
    Output: {"params": {joint: {"kp", "kv", "damping", "armature", "frictionloss"}},
             "baseline_rmse_rad": {joint: float}   # asset parameters, same episodes
             "train_rmse_rad": {joint: float}, "held_out_rmse_rad": {joint: float},
             "held_out_episode": int, "n": {"train": int, "held_out": int}}
    Invariants:
      - fit per joint: each joint gets its own five parameters;
      - objective: tracking RMSE of `run_episode` against the recorded state, every episode started with
        `start_episode` at its recorded first state, with the same `replay_timing` dt and substeps;
      - the held-out episode never influences the fit and is reported separately;
      - kp > 0 and kv, damping, armature, frictionloss >= 0; kp and the bias term stay a position servo
        (biasprm[1] == -kp), so `MuJoCoBackend` still accepts the model;
      - the asset file is never modified; results are reported, not written back.
    """
    raise NotImplementedError("TODO(daniel): actuator parameter fitting (kp/kv/damping/armature/frictionloss)")


def add_command(commands):
    parser = commands.add_parser("replay-real", help="Open-loop MuJoCo replay of an imported SO-101 hardware run")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="An `xert import-lerobot --motor-units degrees` output (frames.npz + import.json)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", action="store_true",
                        help="Append pooled per-joint RMSE rows; only for a verified hardware import (raw/ present)")


def execute(args):
    print(json.dumps(replay_real(args.dataset, args.output, ledger=args.ledger)))

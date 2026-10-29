"""Per-episode gate metrics from a LeRobot v3 runtime recording.

No thresholds live here: P5.2 pre-registers them from real human-demo distributions.
"""
import csv
import gc
import hashlib
import json
from pathlib import Path

import numpy as np

from xert.data.recorder import TIMES

METRICS = ("frames", "duration_s", "action_jitter", "max_command_gap_s", "max_capture_gap_s",
           "dropped_frame_fraction", "mean_tracking_confidence", "mean_handedness_score",
           "clamp_count", "success")
COMMAND, CAPTURE = TIMES.index("t_command"), TIMES.index("t_capture")
HUMAN_SOURCES = ("webcam", "recorded_human")


def _max_gap(times):
    return float(np.diff(times).max()) if len(times) > 1 else None


def episode_metrics(frames, fps):
    """frames: recorded feature name -> numpy array for ONE episode, in frame order."""
    action = np.asarray(frames["action"], dtype=np.float64)
    times, present = frames["xert.times"], frames["xert.times_present"]
    seen = frames["xert.frame_sequence_present"] & present[:, CAPTURE]
    sequences = np.unique(frames["xert.frame_sequence"][seen])  # Held frames repeat their sequence.
    confidence = frames["xert.tracking_confidence"][frames["xert.tracking_confidence_present"]]
    handedness = frames["xert.handedness_scores"][frames["xert.handedness_present"]]
    second = np.diff(action, n=2, axis=0)
    return {"frames": len(action), "duration_s": len(action) / fps,
            "action_jitter": float(np.sqrt(np.mean(second ** 2))) if len(second) else None,
            "max_command_gap_s": _max_gap(times[:, COMMAND]),
            "max_capture_gap_s": _max_gap(times[seen, CAPTURE]),
            # ponytail: drops before the first / after the last applied frame are unobservable here.
            "dropped_frame_fraction": (1 - len(sequences) / (sequences[-1] - sequences[0] + 1)
                                       if len(sequences) else None),
            "mean_tracking_confidence": float(confidence.mean()) if len(confidence) else None,
            "mean_handedness_score": float(handedness.mean()) if len(handedness) else None,
            "clamp_count": int(frames["xert.clamped"].sum()),
            # Same reduction as LeRobot eval: an episode succeeds if any step reports success.
            "success": bool(frames["next.success"].any()) if "next.success" in frames else None}


def dataset_metrics(root, repo_id="xert/runtime"):
    from xert.data.recorder import load_runtime_dataset
    dataset = load_runtime_dataset(root, repo_id)
    try:
        batch = dataset.hf_dataset[:]
        columns = {key: np.stack([np.asarray(value) for value in values])
                   for key, values in batch.items() if not isinstance(values[0], str)}
        fps = dataset.fps
    finally:
        del dataset
        gc.collect()  # Windows keeps Parquet files locked while the reader lives.
    order = np.lexsort((columns["frame_index"], columns["episode_index"]))
    columns = {key: value[order] for key, value in columns.items()}
    rows = []
    for episode in np.unique(columns["episode_index"]):
        mask = columns["episode_index"] == episode
        rows.append({"episode_index": int(episode),
                     **episode_metrics({key: value[mask] for key, value in columns.items()}, fps)})
    return rows


def distributions(rows):
    """Histogram bin counts per metric over episodes where it is available."""
    result = {}
    for name in METRICS:
        values = [float(row[name]) for row in rows if row[name] is not None]
        counts, edges = np.histogram(values, bins="auto") if values else ([], [])
        result[name] = {"n": len(values), "unavailable": len(rows) - len(values),
                        "counts": list(map(int, counts)), "bin_edges": list(map(float, edges))}
    return result


def verdict(metrics, thresholds):
    """thresholds: {metric: {"min": x, "max": y}}. Returns (passed, reasons); unavailable fails."""
    reasons = []
    for name, bounds in thresholds.items():
        if name not in metrics or not bounds or set(bounds) - {"min", "max"}:
            raise ValueError(f"Unknown gate threshold: {name}={bounds!r}")
        if any(type(bound) not in (int, float) for bound in bounds.values()):
            raise ValueError(f"Gate threshold bounds must be numbers: {name}={bounds!r}")
        value = metrics[name]
        if value is None:
            reasons.append(f"{name} unavailable")
            continue
        if "min" in bounds and value < bounds["min"]:
            reasons.append(f"{name}={value} below min {bounds['min']}")
        if "max" in bounds and value > bounds["max"]:
            reasons.append(f"{name}={value} above max {bounds['max']}")
    return not reasons, reasons


def add_command(commands):
    parser = commands.add_parser("quality", help="Per-episode gate metrics for a runtime recording")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", action="store_true", help="Append median rows; human-source data only")


def execute(args):
    from xert.eval.ledger import append_row, artifact_path, new_row, source_provenance
    if args.output.exists():
        raise ValueError("Output already exists; choose a fresh evidence path")
    manifest_path = args.dataset / "xert_recording.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    kind = manifest["source"]["kind"]
    if args.ledger and kind not in HUMAN_SOURCES:
        raise ValueError(f"Refusing ledger rows for {kind!r} data; only {HUMAN_SOURCES} count as demonstrations")
    rows = dataset_metrics(args.dataset)
    args.output.mkdir(parents=True)
    report = args.output / "metrics.json"
    report.write_text(json.dumps({"dataset": str(args.dataset), "source_kind": kind, "episodes": rows,
                                  "distributions": distributions(rows)}, indent=2, allow_nan=False),
                      encoding="utf-8")
    table = args.output / "episodes.csv"
    with table.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["episode_index", *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    if not args.ledger:
        return
    provenance = source_provenance()
    hashes = {artifact_path(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (report, table)}
    for name in METRICS[1:]:
        values = [row[name] for row in rows if row[name] is not None]
        if not values:
            continue  # Unavailable is not zero; no row.
        metric, value = ("success_rate", float(np.mean(values))) if name == "success" else \
            (f"{name}_median", float(np.median(values)))
        row = new_row(**provenance, slice="P5.1", experiment="gate_metrics_v1", backend=manifest["backend"],
                      embodiment=manifest["embodiment"], metric=metric, value=value, ci95=None, n=len(values),
                      seeds={"train": [], "eval": []}, config_path=artifact_path(manifest_path),
                      artifact_paths=list(hashes), artifact_sha256=hashes,
                      notes=f"Per-episode {name} over {len(values)} of {len(rows)} {kind} episodes in simulation; "
                            "distribution in metrics.json. No thresholds applied, no policy or hardware claim.")
        append_row(row)
        print(json.dumps(row))

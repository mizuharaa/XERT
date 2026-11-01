"""Summarize only unique valid applied frames, never absent-hand or replay timing."""
import numpy as np


def summarize(records, *, require_webcam=True):
    if any(row.get("state_measured", True) is False for row in records):
        raise ValueError("Latency needs measured state; these records echo commanded targets")
    samples = [row for row in records if row["latency_sample"]]
    if not samples:
        raise ValueError("No valid applied motion frames; no latency metric can be reported")
    if require_webcam and any(row["source"] != "webcam" for row in samples):
        raise ValueError("Live webcam latency requires webcam source frames")
    if len({row["frame_sequence"] for row in samples}) != len(samples):
        raise ValueError("Latency samples must be unique frames")
    values = []
    for row in samples:
        times = [row[key] for key in ("t_capture", "t_landmarks", "t_ik", "t_command", "t_applied")]
        if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
            raise ValueError("Latency stage timestamps must be finite and ordered")
        values.append((times[-1] - times[0]) * 1000)
    return {"n": len(values), "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "valid_ticks": sum(row["valid_motion"] for row in records),
            "ticks": len(records), "overruns": sum(row["deadline_overrun"] for row in records)}

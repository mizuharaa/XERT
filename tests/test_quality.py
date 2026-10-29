"""Gate metrics on hand-computable episodes; the CLI check needs the optional LeRobot env."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from xert.data.quality import CAPTURE, COMMAND, episode_metrics, verdict

ROOT = Path(__file__).resolve().parents[1]


def episode(**overrides):
    times = np.zeros((5, 8))
    times[:, COMMAND] = [0, .04, .08, .20, .24]
    times[:, CAPTURE] = [0, 10.0, 10.0, 10.1, 10.3]
    present = np.ones((5, 8), bool)
    present[0, CAPTURE] = False
    frames = {"action": np.stack([[0, 1, 4, 9, 16], np.zeros(5)], axis=1),  # second difference 2 and 0
              "xert.times": times, "xert.times_present": present,
              "xert.frame_sequence": np.array([0, 1, 1, 3, 6]),
              "xert.frame_sequence_present": np.array([False, True, True, True, True]),
              "xert.tracking_confidence": np.zeros(5), "xert.tracking_confidence_present": np.zeros(5, bool),
              "xert.handedness_scores": np.full((5, 2), .9),
              "xert.handedness_present": np.array([[False, False]] + [[True, False]] * 4),
              "xert.clamped": np.array([True, False, True, True, False])}
    return {**frames, **overrides}


def test_jitter_is_rms_of_second_difference():
    assert episode_metrics(episode(), 25)["action_jitter"] == pytest.approx(np.sqrt(2))
    assert episode_metrics(episode(action=np.zeros((2, 2))), 25)["action_jitter"] is None


def test_gaps_and_dropped_frames_use_sequence_and_host_times():
    metrics = episode_metrics(episode(), 25)
    assert metrics["max_command_gap_s"] == pytest.approx(.12)
    assert metrics["max_capture_gap_s"] == pytest.approx(.2)  # Held repeat of sequence 1 is not a new capture.
    assert metrics["dropped_frame_fraction"] == pytest.approx(.5)  # Saw 1, 3, 6 of 1..6.
    absent = episode_metrics(episode(**{"xert.frame_sequence_present": np.zeros(5, bool)}), 25)
    assert absent["dropped_frame_fraction"] is None and absent["max_capture_gap_s"] is None


def test_counts_confidence_duration_and_success():
    metrics = episode_metrics(episode(), 25)
    assert metrics["clamp_count"] == 3 and metrics["frames"] == 5 and metrics["duration_s"] == .2
    assert metrics["mean_tracking_confidence"] is None  # Never substituted by handedness.
    assert metrics["mean_handedness_score"] == pytest.approx(.9)
    assert metrics["success"] is None
    success = np.array([False, False, True, False, False])
    assert episode_metrics(episode(**{"next.success": success}), 25)["success"] is True
    assert episode_metrics(episode(**{"next.success": np.zeros(5, bool)}), 25)["success"] is False


def test_verdict_reasons_and_unknown_threshold():
    metrics = episode_metrics(episode(), 25)
    passed, reasons = verdict(metrics, {"action_jitter": {"max": 1}, "mean_tracking_confidence": {"min": .5},
                                        "clamp_count": {"min": 0, "max": 2}, "frames": {"min": 5}})
    assert not passed
    assert reasons == [f"action_jitter={np.sqrt(2)} above max 1", "mean_tracking_confidence unavailable",
                       "clamp_count=3 above max 2"]
    assert verdict(metrics, {"dropped_frame_fraction": {"max": .5}}) == (True, [])
    with pytest.raises(ValueError, match="Unknown"):
        verdict(metrics, {"jiter": {"max": 1}})


def run_cli(monkeypatch, *args):
    from xert.__main__ import main
    monkeypatch.setattr(sys, "argv", ["xert", "quality", *map(str, args)])
    main()


def test_cli_on_scripted_recording_refuses_ledger(tmp_path, monkeypatch):
    pytest.importorskip("lerobot")
    from scripts.record_scripted_runtime import record_scripted
    from xert.eval.ledger import ledger_path, read_rows
    dataset = tmp_path / "recording"
    records = record_scripted(ROOT / "xert/spec/so101.yaml", ROOT / "configs/teleop/so101.json", dataset, frames=5)
    run_cli(monkeypatch, "--dataset", dataset, "--output", tmp_path / "quality")
    report = json.loads((tmp_path / "quality/metrics.json").read_text())
    [metrics] = report["episodes"]
    action = np.asarray([record["command"] for record in records], np.float32).astype(np.float64)
    assert metrics["action_jitter"] == pytest.approx(np.sqrt(np.mean(np.diff(action, n=2, axis=0) ** 2)))
    assert metrics["clamp_count"] == sum(record["clamped"] for record in records)
    assert metrics["success"] is None and metrics["mean_tracking_confidence"] is None
    assert metrics["dropped_frame_fraction"] == 0 and report["source_kind"] == "scripted"
    assert report["distributions"]["success"] == {"n": 0, "unavailable": 1, "counts": [], "bin_edges": []}
    assert sum(report["distributions"]["action_jitter"]["counts"]) == 1
    with (tmp_path / "quality/episodes.csv").open(newline="") as stream:
        assert [row["clamp_count"] for row in csv.DictReader(stream)] == [str(metrics["clamp_count"])]
    with pytest.raises(ValueError, match="Refusing ledger rows for 'scripted'"):
        run_cli(monkeypatch, "--dataset", dataset, "--output", tmp_path / "refused", "--ledger")
    assert not ledger_path().exists() and not (tmp_path / "refused").exists()
    # Relabelled copy only exercises the human-source path against the temporary test ledger.
    manifest = json.loads((dataset / "xert_recording.json").read_text())
    manifest["source"]["kind"] = "recorded_human"
    (dataset / "xert_recording.json").write_text(json.dumps(manifest))
    run_cli(monkeypatch, "--dataset", dataset, "--output", tmp_path / "human", "--ledger")
    rows = read_rows()
    assert ledger_path() == tmp_path / "results/ledger.jsonl"  # conftest isolation, never the real ledger.
    assert {row["metric"] for row in rows} == {f"{name}_median" for name in (
        "duration_s", "action_jitter", "max_command_gap_s", "max_capture_gap_s", "dropped_frame_fraction",
        "mean_handedness_score", "clamp_count")}
    assert all(row["slice"] == "P5.1" and row["n"] == 1 for row in rows)

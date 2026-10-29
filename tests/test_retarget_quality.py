"""Scripted clips only: scoring checks, never human evidence or ledger rows."""
import json
import sys

import numpy as np

from test_motion_file import scripted_clip
from xert.eval.ledger import ledger_path


def test_scripted_clip_scores_every_robot_but_refuses_ledger(tmp_path, monkeypatch, capsys):
    from scripts.measure_retarget import main
    moves = [(.004 * k, .05 * k) for k in range(1, 9)]
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["measure_retarget", "--clip", str(scripted_clip(tmp_path, moves)),
                                      "--output", str(output)])
    main()
    assert "no ledger rows appended" in capsys.readouterr().out
    assert not ledger_path().exists()
    for robot in ("so101", "g1_upper", "leap_hand"):
        summary = json.loads((output / robot / "summary.json").read_text())
        rows = [json.loads(line) for line in (output / robot / "trace.jsonl").read_text().splitlines()]
        scored = [row for row in rows if row["retarget"] is not None]
        assert {row["source"] for row in scored} == {"scripted"}
        assert min(row["frame_sequence"] for row in scored) > summary["calibration_frame_sequence"]
        # Known fixture: each error is the distance between the stored target and achieved site.
        errors = []
        for row in scored:
            for site, value in row["retarget"]["error_mm"].items():
                target, achieved = (np.asarray(row["retarget"][k][site]) for k in ("target", "achieved"))
                assert np.isclose(value, np.linalg.norm(target - achieved) * 1000)
                errors.append(value)
        assert summary["n"] == len(errors) and summary["frames"] == len(scored)
        assert np.isclose(summary["p95_mm"], np.percentile(errors, 95))

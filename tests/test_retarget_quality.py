"""Scripted clips only: scoring checks, never human evidence or ledger rows."""
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

from test_motion_file import scripted_clip
from xert.eval.ledger import ledger_path
from xert.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]


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
        assert summary["neutral_offset_s"] == 0
        # Known fixture: each error is the distance between the stored target and achieved site.
        errors = []
        for row in scored:
            for site, value in row["retarget"]["error_mm"].items():
                target, achieved = (np.asarray(row["retarget"][k][site]) for k in ("target", "achieved"))
                assert np.isclose(value, np.linalg.norm(target - achieved) * 1000)
                errors.append(value)
        assert summary["n"] == len(errors) and summary["frames"] == len(scored)
        assert np.isclose(summary["p95_mm"], np.percentile(errors, 95)) and summary["p95_mm"] > 0
        assert scored[0]["retarget"]["target"] != scored[-1]["retarget"]["target"]
    # Independent LEAP target: this test's own forward kinematics of the requested joint vector.
    spec = load_spec(ROOT / "xert/spec/leap_hand.yaml")
    model = spec.load_model()
    data = mujoco.MjData(model)
    rows = [json.loads(line) for line in (output / "leap_hand/trace.jsonl").read_text().splitlines()]
    last = [row for row in rows if row["retarget"] is not None][-1]["retarget"]
    data.qpos[[model.joint(n).qposadr[0] for n in spec.joint_names]] = last["requested_q"]
    mujoco.mj_kinematics(model, data)
    for site, target in last["target"].items():
        assert np.allclose(target, data.site(site).xpos)


@pytest.mark.parametrize("pinned_model", [False, True])
def test_relabelled_scripted_clip_is_refused(tmp_path, monkeypatch, capsys, pinned_model):
    from scripts.measure_retarget import main
    path = scripted_clip(tmp_path, [(.01, .1)] * 2)
    clip = json.loads(path.read_text())
    clip["source"] = "webcam"  # forged label; recorder-only fields stay absent
    if pinned_model:
        clip["model"] = json.loads((ROOT / "assets/mediapipe/SOURCE.json").read_text())
    path.write_text(json.dumps(clip))
    monkeypatch.setattr(sys, "argv", ["measure_retarget", "--clip", str(path), "--output", str(tmp_path / "run"),
                                      "--robots", "so101"])
    main()
    assert "no ledger rows appended" in capsys.readouterr().out
    assert not ledger_path().exists()


def test_neutral_more_than_two_seconds_in_is_refused(tmp_path, monkeypatch):
    from scripts.measure_retarget import main
    path = scripted_clip(tmp_path, [(0, 0)] * 64)
    clip = json.loads(path.read_text())
    for f in clip["frames"][:64]:  # hands first appear at frame 64, 2.13 s after the first capture
        f["hands"] = {}
    path.write_text(json.dumps(clip))
    monkeypatch.setattr(sys, "argv", ["measure_retarget", "--clip", str(path), "--output", str(tmp_path / "run"),
                                      "--robots", "so101"])
    with pytest.raises(ValueError, match="2.13 s into the clip"):
        main()

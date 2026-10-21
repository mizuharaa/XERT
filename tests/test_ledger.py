"""Ledger integrity and real evaluation wiring; short runs are not policy evidence."""
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from xert.eval import ledger
from xle_workbench import pipeline
from xle_workbench.storage import Store


def sample_row():
    return ledger.new_row(git_commit="a" * 40, git_dirty=True, slice="P0.2",
                          experiment="storage_test", backend="mujoco", embodiment="xlerobot_left",
                          metric="success_rate", value=0.0, ci95=None, n=1,
                          seeds={"train": [], "eval": [0]}, config_path="test-config.json",
                          artifact_paths=["test-report.json"], notes="Synthetic storage check only.")


def append_in_process(path):
    ledger.append_row(sample_row(), Path(path))


def test_append_only_validation_and_cli(tmp_path):
    path = ledger.ledger_path()
    first, second = sample_row(), sample_row()
    ledger.append_row(first)
    original = path.read_bytes()
    ledger.append_row(second)
    assert path.read_bytes().startswith(original)
    assert ledger.read_rows() == [first, second]
    with pytest.raises(ValueError, match="Duplicate"):
        ledger.append_row(first)
    for invalid in ({"value": float("nan")}, {"backend": "hardware"}, {"ci95": [1, 0]},
                    {"seeds": {"train": [0, 0], "eval": [1]}}, {"git_dirty": "false"}):
        with pytest.raises(ValueError):
            ledger.append_row({**sample_row(), **invalid})
    assert ledger.read_rows() == [first, second]
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    shown = subprocess.check_output([sys.executable, "-m", "xert", "ledger", "show"],
                                    text=True, **options)
    assert [json.loads(line) for line in shown.splitlines()] == [first, second]
    for suffix in (b'{"id":', b'{}\n', b'\n'):
        corrupt = tmp_path / "corrupt.jsonl"
        corrupt.write_bytes(original + suffix)
        with pytest.raises(ValueError, match="line 2"):
            ledger.append_row(sample_row(), corrupt)
        assert corrupt.read_bytes() == original + suffix


def test_writers_in_separate_processes(tmp_path):
    path = tmp_path / "concurrent.jsonl"
    with ProcessPoolExecutor(max_workers=2) as pool:
        list(pool.map(append_in_process, [str(path)] * 6))
    rows = ledger.read_rows(path)
    assert len(rows) == len({row["id"] for row in rows}) == 6


def test_real_evaluation_and_paired_benchmark(tmp_path):
    store = Store(tmp_path / "data")
    result = pipeline.evaluate(store, [20, 21], {"horizon": 2, "object_position": [.24, -.03]})
    first = ledger.read_rows()[0]
    assert first["id"] == result["ledger_id"]
    assert first["ci95"] is None and first["seeds"] == {"train": [], "eval": [20, 21]}
    assert first["value"] == result["success_rate"] and first["n"] == result["trials"]
    assert first["backend"] == "mujoco" and first["embodiment"] == "xlerobot_left"
    assert first["git_commit"] == ledger.source_provenance()["git_commit"]
    config = json.loads(Path(first["config_path"]).read_text("utf-8"))
    assert config["environment"] == result["config"]
    assert all(Path(path).exists() for path in first["artifact_paths"])
    report = pipeline.benchmark(store, [30], [{"horizon": 2}, {"horizon": 2, "delay_steps": 1}], [])
    rows = ledger.read_rows()
    assert len(rows) == 3
    assert {row["experiment"] for row in rows[1:]} == {f"benchmark_{report['id']}"}
    assert [row["id"] for row in rows[1:]] == [item["ledger_id"] for item in report["comparisons"]]
    assert all(row["n"] == 1 and row["seeds"]["eval"] == [30] for row in rows[1:])


def test_ledger_failure_does_not_report_completed_evaluation(tmp_path, monkeypatch):
    def fail(row):
        raise OSError("ledger unavailable")

    monkeypatch.setattr(ledger, "append_row", fail)
    with pytest.raises(OSError, match="ledger unavailable"):
        pipeline.evaluate(Store(tmp_path / "data"), [30], {"horizon": 2})
    assert ledger.read_rows() == []


def test_wilson_boundary_evaluations_are_recordable(tmp_path):
    for count in range(1, 101):
        for successes in range(count + 1):
            low, high = pipeline.wilson(successes, count)
            assert 0 <= low <= successes / count <= high <= 1
        assert pipeline.wilson(0, count)[0] == 0
        assert pipeline.wilson(count, count)[1] == 1
    result = pipeline.evaluate(Store(tmp_path / "data"), list(range(11)), {"horizon": 2})
    row, = ledger.read_rows()
    assert row["id"] == result["ledger_id"]
    assert row["value"] == 0 and row["ci95"][0] == 0

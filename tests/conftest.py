"""Keep software checks out of the portfolio measurement ledger."""
import pytest


@pytest.fixture(autouse=True)
def isolated_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("XERT_LEDGER_PATH", str(tmp_path / "results/ledger.jsonl"))

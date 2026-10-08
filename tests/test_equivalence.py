"""The C++ core must reproduce the original Python simulator exactly, including delay and friction paths."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("equivalence_check", Path(__file__).resolve().parents[1] / "scripts/equivalence_check.py")
equivalence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(equivalence)


@pytest.mark.parametrize("config", [None, {"delay_steps": 3}, {"friction": 0.5}])
def test_cpp_core_matches_python_reference(config):
    rows = equivalence.compare([7], config)
    assert equivalence.identical(rows), rows

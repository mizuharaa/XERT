"""Compare independent scripted rollouts with the unmodified baseline, seeds 0..15.

Run from the checkout: .venv/Scripts/python.exe scripts/equivalence_check.py
Reports every difference; no observation tolerance masks numerical drift.
Exits non-zero unless outcomes, step counts and every observation match exactly.
"""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xle_workbench.sim import ManipulationEnv


def compare(seeds, config=None):
    spec = importlib.util.spec_from_file_location("reference_sim", ROOT / "tests/reference_sim.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    rows = []
    for seed in seeds:
        original, port = reference.ManipulationEnv(config), ManipulationEnv(config)
        left, right = original.reset(seed), port.reset(seed)
        difference = float(np.max(np.abs(left.astype(float) - right.astype(float))))
        outcomes = [{}, {}]
        compared = 1
        while not (original.finished and port.finished):
            both_running = not original.finished and not port.finished
            if not original.finished:
                left, _, _, outcomes[0] = original.step(original.expert_action())
            if not port.finished:
                right, _, _, outcomes[1] = port.step(port.expert_action())
            if both_running:
                difference = max(difference, float(np.max(np.abs(left.astype(float) - right.astype(float)))))
                compared += 1
        rows.append({"seed": seed, "config": config or {},
                     "success_match": outcomes[0]["success"] == outcomes[1]["success"],
                     "failure_reason_match": outcomes[0]["failure_reason"] == outcomes[1]["failure_reason"],
                     "step_count_match": original.steps == port.steps,
                     "reference_success": outcomes[0]["success"], "cpp_success": outcomes[1]["success"],
                     "reference_failure_reason": outcomes[0]["failure_reason"], "cpp_failure_reason": outcomes[1]["failure_reason"],
                     "reference_steps": original.steps, "cpp_steps": port.steps,
                     "compared_observations": compared, "max_observation_difference": difference})
    return rows


def identical(rows):
    return all(r["success_match"] and r["failure_reason_match"] and r["step_count_match"]
               and r["max_observation_difference"] == 0.0 for r in rows)


def main():
    rows = compare(range(16))
    for row in rows:
        print(json.dumps(row, allow_nan=False), flush=True)
    print(json.dumps({"seeds": len(rows), "identical": identical(rows),
                      "max_observation_difference": max(r["max_observation_difference"] for r in rows)}), flush=True)
    return 0 if identical(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

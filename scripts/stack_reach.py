"""Measure that both bench arms reach the stack task's two spawn rectangles at the heights stacking needs.

Run from the checkout: .venv/Scripts/python.exe scripts/stack_reach.py
Same method as scripts/so101_bench_reach.py (slice B): position-only generalized IK (xert.retarget.ik.solve, no new
solver math) from the bench reset pose, then FK, on the composed stack scene of each arm.
Samples, at the four corners of each rectangle: block rectangle at the block centre and 6 / 7.5 cm above it (pick);
base_block rectangle at the stacked block's centre (base top + half a block) and 6 / 7.5 cm above it (place).
Exits non-zero unless every FK error is below 1 mm (the P4.1 bench-profile IK tolerance).
"""
import itertools
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xert.backends.stack import StackBackend
from xert.retarget.ik import solve
from xert.spec import load_spec
from xert.tasks.stack import REST_Z, SPAWN

TOLERANCE = 1e-3
STACKED_Z = REST_Z + .032  # Block centre resting on base_block: one block height (2 x .016 m) higher.


def reach_report(robot):
    backend = StackBackend(load_spec(ROOT / f"xert/spec/{robot}.yaml"), 0)
    samples = []
    for name, z in (("block", REST_Z), ("base_block", STACKED_Z)):
        (x0, y0), (x1, y1) = SPAWN[name]
        for x, y in itertools.product((x0, x1), (y0, y1)):
            samples += [(f"{name} {x:.3f},{y:.3f} +{dz:.3f}", [x, y, z + dz]) for dz in (0, .06, .075)]
    fk, rows = mujoco.MjData(backend.model), []
    for name, target in samples:
        q = solve(backend.spec, backend.model, backend.data, {"grasp": np.asarray(target)}, iterations=200)
        fk.qpos[:] = backend.data.qpos
        fk.qpos[backend.qids] = q
        mujoco.mj_forward(backend.model, fk)
        rows.append({"sample": name, "target": target, "q": q.tolist(),
                     "fk_error_m": float(np.linalg.norm(fk.site("grasp").xpos - target))})
    return {"robot": robot, "ik": "xert.retarget.ik.solve, position-only, 200 iterations", "start": "bench reset pose",
            "tolerance_m": TOLERANCE, "samples": rows, "worst_fk_error_m": max(row["fk_error_m"] for row in rows)}


if __name__ == "__main__":
    reports = [reach_report(robot) for robot in ("xlerobot_left", "so101")]
    print(json.dumps(reports, indent=2))
    raise SystemExit(0 if all(r["worst_fk_error_m"] < TOLERANCE for r in reports) else 1)

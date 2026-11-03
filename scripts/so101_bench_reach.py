"""Measure that the SO-101 bench placement reaches the block-spawn region and the tray.

Run from the checkout: .venv/Scripts/python.exe scripts/so101_bench_reach.py
Position-only generalized IK (xert.retarget.ik.solve, no new solver math) from the bench reset pose, then FK.
Samples: the four corners of the default block-spawn bounds at the block centre and 6 / 7.5 cm above it,
and the tray at 4 / 10 cm above its body origin (the heights the XLeRobot expert uses).
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
from xert.retarget.ik import solve
from xert.spec import load_spec
from xle_workbench.sim import ManipulationEnv

TOLERANCE = 1e-3


def reach_report():
    env = ManipulationEnv({"robot": "so101"})
    spec = load_spec(ROOT / "xert/spec/so101.yaml")
    env.reset(0)
    (x0, y0), (x1, y1) = env.config["object_position_bounds"]
    samples = []
    for x, y in itertools.product((x0, x1), (y0, y1)):
        samples += [(f"block {x:.3f},{y:.3f} +{dz:.3f}", [x, y, .096 + dz]) for dz in (0, .06, .075)]
    tray = env.model.body("tray").pos
    samples += [(f"tray +{dz:.2f}", (tray + [0, 0, dz]).tolist()) for dz in (.04, .10)]
    fk, rows = mujoco.MjData(env.model), []
    for name, target in samples:
        q = solve(spec, env.model, env.data, {"grasp": np.asarray(target)}, iterations=200)
        fk.qpos[:] = env.data.qpos
        fk.qpos[env.qids] = q
        mujoco.mj_forward(env.model, fk)
        rows.append({"sample": name, "target": target, "q": q.tolist(),
                     "fk_error_m": float(np.linalg.norm(fk.site("grasp").xpos - target))})
    return {"base": env.source["composition"]["base"], "ik": "xert.retarget.ik.solve, position-only, 200 iterations",
            "start": "bench reset pose", "tolerance_m": TOLERANCE, "samples": rows,
            "worst_fk_error_m": max(row["fk_error_m"] for row in rows)}


if __name__ == "__main__":
    report = reach_report()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["worst_fk_error_m"] < TOLERANCE else 1)

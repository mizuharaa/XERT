"""Two stack-task measurements the handoff cites (slice H). Prints JSON; no ledger row, nothing written.

Run from the checkout: .venv/Scripts/python.exe scripts/stack_probe.py
1. Wall-clock cost of one StackBackend tick (20 physics substeps + StackTask scoring), holding the reset pose for
   TICKS ticks on seed 0. One run on this machine; it varies between runs and machines.
2. Settling: `block` placed on `base_block` (10 um penetration, or dropped from 3 mm), offset (+2, -1) mm in xy,
   arm holding its reset pose; the tick at which StackTask reports success (12 is the earliest possible).
"""
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xert.backends.stack import StackBackend
from xert.spec import load_spec

TICKS = 50


def tick_cost(robot):
    backend = StackBackend(load_spec(ROOT / f"xert/spec/{robot}.yaml"), 0)
    q, seconds = backend.read_state().q, []
    for _ in range(TICKS):
        start = time.perf_counter()
        backend.send_targets(q, None)
        seconds.append(time.perf_counter() - start)
    return {"ticks": TICKS, "mean_ms": 1e3 * float(np.mean(seconds)), "max_ms": 1e3 * float(np.max(seconds))}


def settle_tick(robot, drop):
    backend = StackBackend(load_spec(ROOT / f"xert/spec/{robot}.yaml"), 1)
    base = backend.model.joint("base_block_joint").qposadr[0]
    block = backend.model.joint("block_joint").qposadr[0]
    xy, z = backend.data.qpos[base:base + 2] + [.002, -.001], backend.data.qpos[base + 2] + .032 + drop
    backend.data.qpos[block:block + 7] = [*xy, z, 1, 0, 0, 0]
    mujoco.mj_forward(backend.model, backend.data)
    q = backend.read_state().q
    while not backend.finished:
        backend.send_targets(q, None)
    return {"drop_m": drop, "success": backend.info["success"], "steps": backend.info["steps"],
            "failure_reason": backend.info["failure_reason"]}


if __name__ == "__main__":
    print(json.dumps({robot: {"tick_cost": tick_cost(robot),
                              "settle": [settle_tick(robot, drop) for drop in (-1e-5, .003)]}
                      for robot in ("xlerobot_left", "so101")}, indent=2))

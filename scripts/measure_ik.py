"""Measure seeded reachable-target IK position errors in simulation and append evidence."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from xert.eval.ledger import append_row, artifact_path, new_row, source_provenance
from xert.retarget.ik import solve
from xert.spec import load_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=ROOT / "xert/spec/so101.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/p3.2/ik_errors.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Evidence output already exists; choose a new --output to preserve ledger artifacts")
    provenance = source_provenance()
    spec = load_spec(args.spec)
    model = spec.load_model()
    state = mujoco.MjData(model)
    qids = [model.joint(n).qposadr[0] for n in spec.joint_names]
    records = []
    for seed in range(32):
        rng = np.random.default_rng(seed)
        target_q = np.clip(spec.rest + rng.uniform(-.25, .25, len(qids)), *spec.position_limits.T)
        state.qpos[qids] = target_q
        mujoco.mj_forward(model, state)
        targets = {s: state.site(s).xpos.copy() for s in spec.sites}
        state.qpos[qids] = spec.rest
        result = solve(spec, model, state, targets, iterations=200)
        state.qpos[qids] = result
        mujoco.mj_forward(model, state)
        error = max(float(np.linalg.norm(state.site(s).xpos - targets[s])) * 1000 for s in spec.sites)
        records.append({"seed": seed, "target_q": target_q.tolist(), "solution_q": result.tolist(), "error_mm": error})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"spec": artifact_path(args.spec), "iterations": 200,
            "damping": .01, "sampling": "rest +/- 0.25 rad, clipped to limits; position only, no collision test",
            "records": records}, stream, indent=2)
        stream.write("\n")
    errors = [r["error_mm"] for r in records]
    for metric, value in (("ik_err_mm_mean", np.mean(errors)), ("ik_err_mm_p95", np.percentile(errors, 95)),
                           ("ik_err_mm_max", max(errors))):
        row = new_row(**provenance, slice="P3.2", experiment="reachable_local_ik_v1", backend="mujoco",
            embodiment=spec.name, metric=metric, value=float(value), ci95=None, n=len(records),
            seeds={"train": [], "eval": list(range(32))}, config_path=artifact_path(args.spec),
            artifact_paths=[artifact_path(args.output)],
            notes="Simulation kinematics only; local reachable position targets, not global reachability, collision avoidance, dynamics or hardware.")
        append_row(row)
        print(json.dumps(row))


if __name__ == "__main__":
    main()

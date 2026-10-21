"""Drive arbitrary specs with the same recorded, neutral-relative motion file in simulation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from xert.backends import MuJoCoBackend
from xert.eval.ledger import append_row, artifact_path, new_row, source_provenance
from xert.retarget.ik import solve
from xert.retarget.mapping import map_positions
from xert.safety import SafetyController
from xert.spec import load_spec


def drive(spec_path, motion):
    spec = load_spec(spec_path)
    frames = motion.get("frames")
    if motion.get("source") != "scripted" or not isinstance(frames, list) or not frames:
        raise ValueError("This acceptance runner requires a nonempty explicitly scripted clip")
    backend = MuJoCoBackend(spec)
    safety = SafetyController(spec, backend.read_state().q)
    neutral = {s: backend.data.site(s).xpos.copy() for s in spec.sites}
    records = []
    try:
        for frame in frames:
            t = frame["t"]
            targets = map_positions(spec, frame["offsets"], neutral)
            desired = solve(spec, backend.model, backend.data, targets)
            command = safety.step(desired, t_now=t, t_capture=t)
            backend.send_targets(command.q, t)
            state = backend.read_state()
            records.append({"t_cmd": t, "t_sim": state.t, "q": state.q.tolist(),
                "qd": state.qd.tolist(), "command": command.q.tolist(), "clamped": command.clamped,
                "events": command.events,
                "targets": {s: p.tolist() for s, p in targets.items()},
                "achieved": {s: backend.data.site(s).xpos.tolist() for s in spec.sites}})
    finally:
        backend.close()
    return {"embodiment": spec.name, "spec": artifact_path(Path(spec_path)), "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", type=Path, nargs="+", required=True)
    parser.add_argument("--motion", type=Path, default=ROOT / "configs/motion/scripted_offsets.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Evidence output already exists; choose a new --output")
    provenance = source_provenance()
    motion = json.loads(args.motion.read_text())
    runs = [drive(path, motion) for path in args.specs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"source": motion["source"], "motion": artifact_path(args.motion), "runs": runs}, stream, indent=2)
        stream.write("\n")
    for run in runs:
        n = len(run["records"])
        row = new_row(**provenance, slice="P3.4", experiment="scripted_cross_embodiment_v1", backend="mujoco",
            embodiment=run["embodiment"], metric="rollout_steps", value=n, ci95=None, n=n,
            seeds={"train": [], "eval": []}, config_path=run["spec"],
            artifact_paths=[artifact_path(args.motion), artifact_path(args.output)],
            notes="Finite simulated rollout from one scripted offset clip; not human demonstrations, task success, tracking-quality or hardware evidence.")
        append_row(row)
        print(json.dumps(row))


if __name__ == "__main__":
    main()

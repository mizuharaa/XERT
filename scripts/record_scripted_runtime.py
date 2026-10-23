"""Record scripted landmarks and a scripted clock; this does not measure live latency."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from xert.backends import MuJoCoBackend
from xert.data.recorder import RuntimeRecorder
from xert.eval.ledger import append_row, artifact_path, new_row, source_provenance
from xert.retarget.calibrate import CalibratedMapper, create_calibration
from xert.runtime.loop import run_loop
from xert.sources import HandFrame, MotionFrame
from xert.spec import load_spec


class ScriptedClock:
    def __init__(self):
        self.now = 1000000.123456789

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


def scripted_frame(sequence, now, shift=0., curl=0.):
    points = np.zeros((21, 3))
    for start, x in ((1, .045), (5, .025), (9, 0), (13, -.02), (17, -.04)):
        for j in range(4):
            points[start + j] = [x, .04 + j * .025, curl * max(0, j - 1) * .02]
    hand = HandFrame(points * 2 + [.5 + shift, .3, 0], points, .9)
    return MotionFrame(sequence, now - .0002, now - .0001, {"left": hand, "right": hand},
                       "scripted", (640, 480))


class ScriptedSource:
    def __init__(self, clock):
        self.clock, self.sequence = clock, -1

    def read(self):
        self.sequence += 1
        if self.sequence == 0:
            return None  # Exercise genuinely absent timestamps, distinct from numeric zero.
        return scripted_frame(self.sequence, self.clock(), shift=.08, curl=.6)

    def close(self):
        pass


def record_scripted(spec_path, profile, output, frames=12):
    spec_bytes, profile_bytes = Path(spec_path).read_bytes(), Path(profile).read_bytes()
    inputs = {"spec_bytes": spec_bytes, "profile_bytes": profile_bytes}
    backend = MuJoCoBackend(load_spec(spec_path, contents=spec_bytes))
    clock = ScriptedClock()
    try:
        calibration = create_calibration(spec_path, profile, scripted_frame(0, clock()), backend, **inputs)
        mapper = CalibratedMapper(spec_path, profile, calibration, backend, **inputs)
        source = {"kind": "scripted", "clock": "scripted, not elapsed real time",
                  "generator": "scripts/record_scripted_runtime.py", "human_demonstrations": False,
                  "calibration": calibration, "profile_sha256": hashlib.sha256(profile_bytes).hexdigest()}
        with RuntimeRecorder(spec_path, backend.read_state(), output,
                             task="Scripted motion recording round trip; task success not evaluated", source=source,
                             spec_bytes=spec_bytes) as recorder:
            records = run_loop(ScriptedSource(clock), backend, mapper, frames / backend.spec.control_hz,
                               clock=clock, sleep=clock.sleep, on_step=recorder)
        return records
    finally:
        backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = source_provenance()
    for robot in ("so101", "g1_upper", "leap_hand"):
        spec = ROOT / f"xert/spec/{robot}.yaml"
        output = args.output / robot
        records = record_scripted(spec, ROOT / f"configs/teleop/{robot}.json", output)
        verification = json.loads((output / "roundtrip.json").read_text())
        files = sorted(path for path in output.rglob("*") if path.is_file())
        row = new_row(**provenance, slice="P2.2", experiment="scripted_runtime_v3_roundtrip_v1",
                      backend="mujoco", embodiment=robot, metric="roundtrip_max_abs_error",
                      value=verification["max_abs_error"], ci95=None, n=len(records),
                      seeds={"train": [], "eval": []}, config_path=artifact_path(spec),
                      artifact_paths=[artifact_path(path) for path in files],
                      artifact_sha256={artifact_path(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
                      notes="Actual LeRobot v3 reload against recorded frames. Scripted landmarks and clock; no human data, live latency, task-success or hardware claim.")
        append_row(row)
        print(json.dumps(row))


if __name__ == "__main__":
    main()

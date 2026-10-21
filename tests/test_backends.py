import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from scripts.drive_embodiments import drive
from xert.backends import MuJoCoBackend
from xert.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]
MOTION = ROOT / "configs/motion/scripted_offsets.json"


@pytest.mark.parametrize("name", ["so101", "g1_upper", "leap_hand"])
def test_same_motion_driver_across_embodiments(name):
    spec_path = ROOT / f"xert/spec/{name}.yaml"
    spec = load_spec(spec_path)
    model = spec.load_model()
    assert not np.any(model.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE))
    motion = json.loads(MOTION.read_text())
    result = drive(spec_path, motion)
    q = np.asarray([f["q"] for f in result["records"]])
    assert len(q) == len(motion["frames"])
    assert np.isfinite(q).all()
    assert np.max(np.abs(q[-1] - spec.rest)) > 1e-6
    zero_motion = json.loads(MOTION.read_text())
    for frame in zero_motion["frames"]:
        frame["offsets"] = {key: [0, 0, 0] for key in frame["offsets"]}
    zero_result = drive(spec_path, zero_motion)
    commands = np.asarray([f["command"] for f in result["records"]])
    zero_commands = np.asarray([f["command"] for f in zero_result["records"]])
    assert np.max(np.abs(commands - zero_commands)) > 1e-5


def test_new_spec_requires_no_controller_edits(tmp_path):
    protected = [p for folder in ("retarget", "safety", "runtime")
                 for p in (ROOT / "xert" / folder).glob("*.py")]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    original = ROOT / "xert/spec/leap_hand.yaml"
    raw = json.loads(original.read_text())
    raw["name"] = "previously_unknown_robot"
    raw["model_path"] = str((original.parent / raw["model_path"]).resolve())
    added = tmp_path / "added.yaml"
    added.write_text(json.dumps(raw))
    motion = json.loads(MOTION.read_text())
    motion["frames"] = motion["frames"][:3]
    assert drive(added, motion)["embodiment"] == "previously_unknown_robot"
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}


def test_backend_rejects_bad_targets_stale_times_and_closed_use():
    backend = MuJoCoBackend(load_spec(ROOT / "xert/spec/so101.yaml"))
    state = backend.read_state()
    with pytest.raises(ValueError):
        backend.send_targets(np.full_like(state.q, 100), 0)
    with pytest.raises(ValueError):
        backend.send_targets(np.full_like(state.q, float("nan")), 0)
    backend.send_targets(state.q, 0)
    with pytest.raises(ValueError, match="timestamps"):
        backend.send_targets(state.q, 0)
    state.q[:] = 100
    assert np.all(backend.read_state().q < 100)
    backend.close()
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        backend.read_state()
    with pytest.raises(RuntimeError, match="closed"):
        backend.send_targets(state.q, 1)


def test_motion_mapping_rejects_missing_landmark():
    motion = json.loads(MOTION.read_text())
    del motion["frames"][0]["offsets"]["right_wrist"]
    with pytest.raises(ValueError, match="Missing"):
        drive(ROOT / "xert/spec/so101.yaml", motion)


def test_backend_rejects_torque_actuators(tmp_path):
    from dataclasses import replace
    path = tmp_path / "torque.xml"
    path.write_text('''<mujoco><worldbody><body><joint name="j" range="-1 1"/>
        <geom size=".1"/><site name="tip"/></body></worldbody>
        <actuator><motor joint="j"/></actuator></mujoco>''')
    original = load_spec(ROOT / "xert/spec/so101.yaml")
    spec = replace(original, model_path=path, joint_names=("j",), rest=np.zeros(1),
                   position_limits=np.array([[-1., 1.]]), velocity_limits=np.ones(1),
                   acceleration_limits=np.ones(1), sites=("tip",))
    with pytest.raises(ValueError, match="position servo"):
        MuJoCoBackend(spec)


def test_sites_match_returned_integrated_state():
    backend = MuJoCoBackend(load_spec(ROOT / "xert/spec/so101.yaml"))
    backend.send_targets(backend.spec.rest + .1, 0)
    reported = backend.data.site("grasp").xpos.copy()
    mujoco.mj_forward(backend.model, backend.data)
    np.testing.assert_allclose(reported, backend.data.site("grasp").xpos, rtol=0, atol=1e-12)
    backend.close()


@pytest.mark.parametrize("fault", ["nan", "clock", "exception"])
def test_engine_fault_latches_closed(monkeypatch, fault):
    backend = MuJoCoBackend(load_spec(ROOT / "xert/spec/so101.yaml"))
    original = mujoco.mj_step

    def broken_step(model, data, **kwargs):
        if fault == "exception":
            raise RuntimeError("Engine failed")
        original(model, data, **kwargs)
        if fault == "nan":
            data.qpos[0] = float("nan")
        else:
            data.time = 0

    monkeypatch.setattr(mujoco, "mj_step", broken_step)
    with pytest.raises(RuntimeError):
        backend.send_targets(backend.spec.rest, 0)
    with pytest.raises(RuntimeError, match="closed"):
        backend.send_targets(backend.spec.rest, 1)
    with pytest.raises(RuntimeError, match="closed"):
        backend.read_state()

from pathlib import Path

import mujoco
import numpy as np
import pytest

from xert.retarget.ik import solve
from xert.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["xlerobot_left", "so101"])
def test_reachable_target_without_state_mutation(name):
    spec = load_spec(ROOT / f"xert/spec/{name}.yaml")
    model = spec.load_model()
    state = mujoco.MjData(model)
    qids = [model.joint(n).qposadr[0] for n in spec.joint_names]
    state.qpos[qids] = spec.rest + 0.1
    mujoco.mj_forward(model, state)
    target = state.site(spec.sites[0]).xpos.copy()
    state.qpos[qids] = spec.rest
    before = state.qpos.copy()
    result = solve(spec, model, state, {spec.sites[0]: target})
    np.testing.assert_array_equal(state.qpos, before)
    state.qpos[qids] = result
    mujoco.mj_forward(model, state)
    np.testing.assert_allclose(state.site(spec.sites[0]).xpos, target, atol=1e-4)


def test_orientation_and_input_validation():
    spec = load_spec(ROOT / "xert/spec/so101.yaml")
    model = spec.load_model()
    state = mujoco.MjData(model)
    qids = [model.joint(n).qposadr[0] for n in spec.joint_names]
    state.qpos[qids] = spec.rest + 0.05
    mujoco.mj_forward(model, state)
    site = state.site(spec.sites[0])
    position = site.xpos.copy()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, site.xmat)
    state.qpos[qids] = spec.rest
    result = solve(spec, model, state, {spec.sites[0]: (position, quat)}, iterations=200)
    state.qpos[qids] = result
    mujoco.mj_forward(model, state)
    np.testing.assert_allclose(site.xpos, position, atol=1e-4)
    actual = np.zeros(4)
    mujoco.mju_mat2Quat(actual, site.xmat)
    assert abs(np.dot(actual, quat)) > 0.99999
    for targets in ({"unknown": [0, 0, 0]}, {spec.sites[0]: [float("nan"), 0, 0]},
                    {spec.sites[0]: (position, [0, 0, 0, 0])}):
        with pytest.raises(ValueError):
            solve(spec, model, state, targets)
    with pytest.raises(ValueError):
        solve(spec, model, state, {spec.sites[0]: position}, damping=0)


def test_pinned_menagerie_assets():
    import hashlib
    import json
    for manifest in (ROOT / "assets/menagerie").glob("*/SOURCE.json"):
        source = json.loads(manifest.read_text())
        assert len(source["commit"]) == 40
        assert "LICENSE" in source["sha256"]
        for name, expected in source["sha256"].items():
            assert hashlib.sha256((manifest.parent / name).read_bytes()).hexdigest() == expected


def test_measurement_preserves_existing_artifact(tmp_path, monkeypatch):
    import sys
    from scripts.measure_ik import main
    path = tmp_path / "existing.json"
    path.write_text("previous evidence")
    monkeypatch.setattr(sys, "argv", ["measure_ik", "--output", str(path)])
    with pytest.raises(SystemExit):
        main()
    assert path.read_text() == "previous evidence"


def test_simultaneous_targets_respect_weights():
    from xle_workbench._reach_core import SiteTarget, solve_ik
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
        <joint type="slide" axis="1 0 0" range="-2 2"/><geom size=".1"/>
        <site name="left" pos="-.5 0 0"/><site name="right" pos=".5 0 0"/>
        </body></worldbody></mujoco>''')
    state = mujoco.MjData(model)
    equal = solve_ik(model, state, [0], [SiteTarget(0, [0, 0, 0], [], 1),
                                       SiteTarget(1, [0, 0, 0], [], 1)], [[-2, 2]], 100, .01)
    weighted = solve_ik(model, state, [0], [SiteTarget(0, [0, 0, 0], [], 9),
                                          SiteTarget(1, [0, 0, 0], [], 1)], [[-2, 2]], 100, .01)
    np.testing.assert_allclose(equal, [0], atol=1e-8)
    np.testing.assert_allclose(weighted, [.4], atol=1e-8)
    assert abs(weighted[0] - .5) < abs(equal[0] - .5)

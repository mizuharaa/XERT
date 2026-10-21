"""Damped least-squares IK backed by the native MuJoCo core."""
import numpy as np

from xle_workbench._reach_core import SiteTarget, solve_ik


def solve(spec, model, state, targets, *, iterations=100, damping=0.01):
    """Return ordered joint positions; targets map site names to position or (position, wxyz)."""
    if not targets or set(targets) - set(spec.sites):
        raise ValueError("Targets must name declared sites")
    native = []
    for name, target in targets.items():
        if isinstance(target, tuple) and len(target) == 2:
            position, quaternion = target
        else:
            position, quaternion = target, []
        native.append(SiteTarget(model.site(name).id, position, quaternion, spec.ik_weights[name]))
    return np.asarray(solve_ik(model, state, [model.joint(n).id for n in spec.joint_names],
                              native, spec.position_limits, iterations, damping))

"""Slice H: block stacking on the bench. Scene composition and the success rule (environment code, not CORE).

The scene is a robot's pick-and-place bench (the same MjSpec render-cameras rebuilds: xle_workbench.sim's
table, tray, block and arm placement) plus a second free block, `base_block`, added in memory. No vendored file
and no sim.py byte changes, so the native Env, its contract hash and every pick-and-place recording are untouched.
Scoring runs in Python on MjData after each 25 Hz control tick (StackTask); the C++ core is not involved.
"""
import hashlib
import itertools
import json

import numpy as np

# Copied from the native pick-and-place bench, so both tasks judge "settled" and "done" the same way.
SETTLE_TICKS = 12  # cpp/src/env.cpp Env::step: success = stable_steps >= 12 consecutive settled ticks.
HORIZON = 600      # xle_workbench/sim.py:145, ManipulationEnv's default horizon (24 s at 25 Hz).
OFF_TABLE_Z = .04  # cpp/src/env.cpp Env::step: block centre z < .04 m is "fell_off_table" (table top is .08 m).
LIFTED_Z = .145    # cpp/src/env.cpp Env::step: lifted |= block centre z > .145 m.
REST_Z = .096      # cpp/src/env.cpp Env::reset_state: block centre on the table (top .08 + half height .016).
# Stack-specific constants (stated in every StackBackend metadata() under "success_rule").
GAP_M = .005        # (c) |block bottom - base_block top| <= 5 mm.
LINEAR_SPEED = .01  # (e) block linear speed below 0.01 m/s ...
ANGULAR_SPEED = .1  # ... and angular speed below 0.1 rad/s.
# (d) block centre within half a block width of the base centre in xy: the block geom's x half-size, 0.008 m.
# Seeded spawn rectangles (xy, metres). block: ManipulationEnv's default object_position_bounds. base_block: a
# disjoint strip 3 cm away in y, so the two 16 mm-square footprints never overlap; scripts/stack_reach.py measures
# that both arms reach every corner at the stacking heights.
SPAWN = {"block": [[.225, -.045], [.255, -.015]], "base_block": [[.225, .015], [.255, .04]]}
BASE_RGBA = [.2, .4, .9, 1]  # Blue, so the camera images tell the two blocks apart.


def stack_spec(robot):
    """(uncompiled MjSpec, model fingerprint, composition) of the stack scene for a bench robot.

    The fingerprint hashes the pick-and-place bench fingerprint plus everything this function adds, so it is
    checkout-independent like the bench's own and changes whenever the composition does.
    """
    import mujoco
    from xert.data.cameras import _bench_spec
    spec, bench_sha256 = _bench_spec(robot)
    block = spec.geom("block_geom")  # base_block copies the bench block's geometry, mass and contact settings.
    base = {"size": block.size.tolist(), "mass": float(block.mass), "condim": int(block.condim),
            "friction": block.friction.tolist(), "rgba": BASE_RGBA}
    body = spec.worldbody.add_body(name="base_block", pos=[*np.mean(SPAWN["base_block"], axis=0), REST_Z])
    body.add_freejoint(name="base_block_joint")
    body.add_geom(name="base_block_geom", type=mujoco.mjtGeom.mjGEOM_BOX, **base)
    composition = {"bench_model_sha256": bench_sha256, "base_block": base, "spawn_xy": SPAWN, "rest_z": REST_Z}
    fingerprint = hashlib.sha256(json.dumps(composition, sort_keys=True).encode()).hexdigest()
    return spec, fingerprint, composition


def stack_scene(robot):
    """(compiled model, fingerprint, composition). Sliding friction 1.0 on every geom, as ManipulationEnv's default
    friction config sets on the bench."""
    spec, fingerprint, composition = stack_spec(robot)
    model = spec.compile()
    model.geom_friction[:, 0] = 1.0
    return model, fingerprint, composition


def spawn(seed):
    """Seeded (block xy, base_block xy), each uniform in its SPAWN rectangle."""
    rng = np.random.default_rng(seed)
    return tuple(rng.uniform(*np.asarray(SPAWN[name])) for name in ("block", "base_block"))


def _z_extent(data, geom, half):
    """(lowest, highest) world z of a box geom's eight corners: correct for a tilted block too."""
    corners = np.array(list(itertools.product(*[(-h, h) for h in half])))
    z = data.geom_xpos[geom][2] + corners @ data.geom_xmat[geom].reshape(3, 3)[2]
    return z.min(), z.max()


class StackTask:
    """Scores one stack episode from MjData after each control tick.

    A tick is `stacked` when all five checks hold; success is SETTLE_TICKS consecutive stacked ticks. A tick with any
    robot contact on `block` is not stacked and resets the count, so the robot touches the block at none of the
    window's ticks (the native rule's `!held`, cpp/src/env.cpp). Like the native rule, contacts are read at control
    ticks, not at the 20 physics substeps between them.
    """
    def __init__(self, model, horizon=HORIZON):
        self.model, self.horizon = model, horizon
        self.block, self.base, self.tray = (model.body(name).id for name in ("block", "base_block", "tray"))
        self.block_geom, self.base_geom = model.geom("block_geom").id, model.geom("base_block_geom").id
        self.block_dof = model.joint("block_joint").dofadr[0]
        self.steps = self.stable_steps = 0
        self.lifted = self.finished = False

    def contacts(self, data):
        """(block touches base_block, robot touches block). Contact geoms resolve to bodies: anything but the world
        (table, floor), the tray and base_block touching `block` is the robot, exactly as env.cpp counts `held`."""
        bodies = self.model.geom_bodyid[np.stack([data.contact.geom1, data.contact.geom2], axis=1)]
        touching = held = False
        for a, b in bodies:
            if self.block in (a, b):
                other = b if a == self.block else a
                touching |= other == self.base
                held |= other not in (0, self.tray, self.base)
        return bool(touching), bool(held)

    def checks(self, data):
        """The five conditions on one state, by name: (a) contact, (b) released, (c) gap, (d) centred, (e) still."""
        touching, held = self.contacts(data)
        half = self.model.geom_size[self.block_geom]
        bottom = _z_extent(data, self.block_geom, half)[0]
        top = _z_extent(data, self.base_geom, self.model.geom_size[self.base_geom])[1]
        offset = np.linalg.norm(data.xpos[self.block][:2] - data.xpos[self.base][:2])
        velocity = data.qvel[self.block_dof:self.block_dof + 6]  # Free joint: linear (world), then angular.
        return {"contact": touching, "released": not held, "gap": bool(abs(bottom - top) <= GAP_M),
                "centred": bool(offset <= half[0]),
                "still": bool(np.linalg.norm(velocity[:3]) < LINEAR_SPEED and np.linalg.norm(velocity[3:]) < ANGULAR_SPEED)}

    def step(self, data):
        """Score the state after one control tick; returns the same info keys as the native bench step."""
        if self.finished:
            raise RuntimeError("episode finished; reset before stepping")
        self.steps += 1
        stacked, (_, held) = all(self.checks(data).values()), self.contacts(data)
        block, base = data.xpos[self.block], data.xpos[self.base]
        self.lifted |= bool(block[2] > LIFTED_Z)
        self.stable_steps = self.stable_steps + 1 if stacked else 0
        success = self.stable_steps >= SETTLE_TICKS
        failure = "fell_off_table" if min(block[2], base[2]) < OFF_TABLE_Z else None
        self.finished = success or failure is not None or self.steps >= self.horizon
        if self.finished and not success and failure is None:
            failure = "timeout_after_lift" if self.lifted else "timeout_no_lift"
        return {"success": success, "failure_reason": failure, "lifted": self.lifted, "steps": self.steps,
                "object_position": block.tolist(), "object_in_robot_contact": held, "stable_steps": self.stable_steps}


SUCCESS_RULE = {
    "rule": "xert/tasks/stack.py StackTask: SETTLE_TICKS consecutive control ticks with (a) block-base_block "
            "contact, (b) no robot contact on block, (c) |block bottom - base top| <= gap_m, (d) xy centre offset "
            "<= half block width, (e) block linear speed < linear_speed and angular speed < angular_speed",
    "settle_ticks": SETTLE_TICKS, "horizon": HORIZON, "gap_m": GAP_M, "centre_offset_m": "block geom x half-size",
    "linear_speed_m_s": LINEAR_SPEED, "angular_speed_rad_s": ANGULAR_SPEED, "off_table_z_m": OFF_TABLE_Z,
    "lifted_z_m": LIFTED_Z,
    "sources": {"settle_ticks, off_table_z_m, lifted_z_m, released": "cpp/src/env.cpp Env::step",
                "horizon": "xle_workbench/sim.py:145"}}

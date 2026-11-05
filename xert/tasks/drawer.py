"""Slice K: drawer opening on the bench, the first articulated object. Scene and success rule (environment code).

The scene is a robot's pick-and-place bench MjSpec (the one render-cameras rebuilds: table, arm placement) with the
free block and the static tray removed and a cabinet added in memory: a box-geom shell fixed to the table and a
`drawer` body on the slide joint `drawer_slide`, carrying a horizontal cylinder `handle`. Built from primitives, so
no asset is vendored; sim.py, cpp/ and assets/ are untouched, and the native Env and its contract hash do not change.
The cabinet pose is FIXED per robot (review K-01) and hashed into the model fingerprint, so the whole scene is
rebuilt from the arm joints plus `drawer.pos`. Scoring runs in Python on MjData after each 25 Hz tick (DrawerTask).
"""
import hashlib
import json

import numpy as np

from xert.tasks.stack import HORIZON, LINEAR_SPEED, SETTLE_TICKS

DRAWER_TRAVEL = .12  # drawer_slide range [0, DRAWER_TRAVEL] metres, 0 = closed.
OPEN_M = .08         # Success needs drawer.pos >= OPEN_M: two thirds of the travel.
SLIDE_SPEED = LINEAR_SPEED  # 0.01 m/s, the stack rule's "still" bound (xert/tasks/stack.py), on the slide joint.
INITIAL_OPEN = (0., .02)    # reset(seed): drawer.pos uniform in this range, metres (the seeded variation).
# Slide joint dynamics: MuJoCo's own defaults, written out so a bench default class cannot change them. Not a
# measured cabinet.
SLIDE = {"damping": 0., "frictionloss": 0., "armature": 0.}
# Geometry, metres. Cabinet frame: origin on the table top at the centre of the front opening, +x out of the front.
SHELL = {"depth": .16, "width": .14, "height": .09, "wall": .006}
CLEARANCE = .004    # Drawer panels to the shell, on every side: the slide joint alone guides the drawer.
DRAWER_DEPTH = .14  # Still 2 cm inside the shell at full travel.
HANDLE = {"radius": .01, "half_length": .035, "standoff": .035}  # Axis along the front, 3.5 cm proud of the face;
# radius sized against both arms' jaws by scripts/drawer_reach.py jaw_sizing() (fits open, gripped closed).
# Per robot: cabinet frame position on the table top (x, y, z) and yaw about world z (radians), chosen so the handle
# stays inside the arm's measured reach over the whole travel (scripts/drawer_reach.py). xlerobot_left: the front
# faces the arm (-x) and the drawer is pulled toward it. so101: the front faces -y and the drawer is pulled sideways
# across the arm's front, because facing the SO-101 (+x) would put the cabinet's back off the table (x < 0.12 m) or
# the open handle over the arm base (x = 0.42 m).
PLACEMENT = {"xlerobot_left": {"pos": [.35, -.03, .08], "yaw": np.pi},
             "so101": {"pos": [.26, .12, .08], "yaw": -np.pi / 2}}
SHELL_RGBA, DRAWER_RGBA, HANDLE_RGBA = [.55, .45, .35, 1], [.75, .6, .4, 1], [.2, .2, .2, 1]


def _box(body, name, centre, half, rgba):
    import mujoco
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(centre), size=list(half), rgba=rgba)


def drawer_spec(robot):
    """(uncompiled MjSpec, model fingerprint, composition) of the drawer scene for a bench robot."""
    import mujoco
    from xert.data.cameras import _bench_spec
    spec, bench_sha256 = _bench_spec(robot)
    for name in ("block", "tray"):  # Not part of this task; a free block would also need its pose recorded.
        spec.delete(spec.body(name))
    place = PLACEMENT[robot]
    d, w, h, t = (SHELL[key] for key in ("depth", "width", "height", "wall"))
    cabinet = spec.worldbody.add_body(name="cabinet", pos=place["pos"],
                                      quat=[np.cos(place["yaw"] / 2), 0, 0, np.sin(place["yaw"] / 2)])
    for name, centre, half in (("bottom", (-d / 2, 0, t / 2), (d / 2, w / 2, t / 2)),
                               ("top", (-d / 2, 0, h - t / 2), (d / 2, w / 2, t / 2)),
                               ("left", (-d / 2, w / 2 - t / 2, h / 2), (d / 2, t / 2, h / 2 - t)),
                               ("right", (-d / 2, -w / 2 + t / 2, h / 2), (d / 2, t / 2, h / 2 - t)),
                               ("back", (-d + t / 2, 0, h / 2), (t / 2, w / 2 - t, h / 2 - t))):
        _box(cabinet, f"cabinet_{name}", centre, half, SHELL_RGBA)
    # Drawer frame at drawer.pos = 0: the cabinet frame. Inner box of the shell, less CLEARANCE on every side.
    iw, ih = w / 2 - t - CLEARANCE, h / 2 - t - CLEARANCE  # Half extents of the drawer's front panel.
    z0 = h / 2
    drawer = cabinet.add_body(name="drawer")
    drawer.add_joint(name="drawer_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0],
                     range=[0, DRAWER_TRAVEL], limited=mujoco.mjtLimited.mjLIMITED_TRUE, **SLIDE)
    side = ih / 2  # Drawer sides and back are half the front panel's height.
    for name, centre, half in (("front", (-t / 2, 0, z0), (t / 2, iw, ih)),
                               ("bottom", (-DRAWER_DEPTH / 2, 0, z0 - ih + t / 2), (DRAWER_DEPTH / 2, iw, t / 2)),
                               ("left", (-DRAWER_DEPTH / 2, iw - t / 2, z0 - ih + side), (DRAWER_DEPTH / 2, t / 2, side)),
                               ("right", (-DRAWER_DEPTH / 2, -iw + t / 2, z0 - ih + side), (DRAWER_DEPTH / 2, t / 2, side)),
                               ("back", (-DRAWER_DEPTH + t / 2, 0, z0 - ih + side), (t / 2, iw, side))):
        _box(drawer, f"drawer_{name}", centre, half, DRAWER_RGBA)
    drawer.add_geom(name="handle", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[HANDLE["standoff"], 0, z0],
                    quat=[np.sqrt(.5), np.sqrt(.5), 0, 0],  # Cylinder axis (local z) turned onto the cabinet's y.
                    size=[HANDLE["radius"], HANDLE["half_length"], 0], rgba=HANDLE_RGBA)
    spec.add_exclude(bodyname1="cabinet", bodyname2="drawer")  # Guided by the joint, never by shell contact.
    composition = {"bench_model_sha256": bench_sha256, "removed": ["block", "tray"], "placement": PLACEMENT[robot],
                   "shell": SHELL, "clearance_m": CLEARANCE, "drawer_depth_m": DRAWER_DEPTH, "handle": HANDLE,
                   "travel_m": DRAWER_TRAVEL, "slide": SLIDE, "density": "MuJoCo default (1000 kg/m^3)",
                   "handle_sizing": "scripts/drawer_reach.py jaw_sizing(); numbers in docs/handoff/K-drawer-bench.md",
                   "reach": "scripts/drawer_reach.py reach_report(), position-only; orientation not measured",
                   "solver": {"timestep": spec.option.timestep, "solver": int(spec.option.solver),
                              "iterations": spec.option.iterations, "cone": int(spec.option.cone),
                              "impratio": spec.option.impratio, "integrator": int(spec.option.integrator)}}
    composition = json.loads(json.dumps(composition, default=float))  # numpy scalars -> JSON numbers.
    fingerprint = hashlib.sha256(json.dumps(composition, sort_keys=True).encode()).hexdigest()
    return spec, fingerprint, composition


def drawer_scene(robot):
    """(compiled model, fingerprint, composition). Sliding friction 1.0 on every geom, as on the bench."""
    spec, fingerprint, composition = drawer_spec(robot)
    model = spec.compile()
    model.geom_friction[:, 0] = 1.0
    return model, fingerprint, composition


def initial_opening(seed):
    """Seeded drawer.pos at reset, uniform in INITIAL_OPEN (metres)."""
    return float(np.random.default_rng(seed).uniform(*INITIAL_OPEN))


class DrawerTask:
    """Scores one drawer episode from MjData after each control tick.

    A tick is `open` when all three checks hold: (a) drawer.pos >= OPEN_M; (b) released: no robot contact on the
    drawer body, whose geoms include the handle (the native rule's `!held`, cpp/src/env.cpp Env::step, as the stack
    rule's (b)); (c) still: slide speed below SLIDE_SPEED. Success is SETTLE_TICKS consecutive open ticks; any other
    tick, e.g. a dip below OPEN_M or a touch, resets the count. Failure is only the horizon. Contacts are read at
    control ticks, not at the 20 physics substeps between them, like the native rule.
    """
    def __init__(self, model, horizon=HORIZON):
        self.model, self.horizon = model, horizon
        self.drawer, self.cabinet = model.body("drawer").id, model.body("cabinet").id
        slide = model.joint("drawer_slide")
        self.qpos, self.dof = slide.qposadr[0], slide.dofadr[0]
        self.handle = model.geom("handle").id
        self.steps = self.stable_steps = 0
        self.finished = False

    def held(self, data):
        """Robot contact on the drawer (panels or handle). Contact geoms resolve to bodies; anything but the world
        (table, floor) and the cabinet is the robot, as env.cpp counts `held`."""
        bodies = self.model.geom_bodyid[np.stack([data.contact.geom1, data.contact.geom2], axis=1)]
        return bool(any((b if a == self.drawer else a) not in (0, self.cabinet)
                        for a, b in bodies if self.drawer in (a, b)))

    def checks(self, data):
        """The three conditions on one state, by name."""
        return {"open": bool(data.qpos[self.qpos] >= OPEN_M), "released": not self.held(data),
                "still": bool(abs(data.qvel[self.dof]) < SLIDE_SPEED)}

    def step(self, data):
        """Score the state after one control tick: the bench info keys (`lifted` is always False, nothing is lifted
        in this task) plus drawer_open_fraction."""
        if self.finished:
            raise RuntimeError("episode finished; reset before stepping")
        self.steps += 1
        checks = self.checks(data)
        self.stable_steps = self.stable_steps + 1 if all(checks.values()) else 0
        success = self.stable_steps >= SETTLE_TICKS
        self.finished = success or self.steps >= self.horizon
        return {"success": success, "failure_reason": "timeout" if self.finished and not success else None,
                "lifted": False, "steps": self.steps, "object_position": data.geom_xpos[self.handle].tolist(),
                "object_in_robot_contact": not checks["released"], "stable_steps": self.stable_steps,
                "drawer_open_fraction": float(data.qpos[self.qpos] / DRAWER_TRAVEL)}


SUCCESS_RULE = {
    "rule": "xert/tasks/drawer.py DrawerTask: SETTLE_TICKS consecutive control ticks with (a) drawer.pos >= open_m, "
            "(b) no robot contact on the drawer body (handle included), (c) |slide speed| < slide_speed_m_s; "
            "failure only at the horizon",
    "settle_ticks": SETTLE_TICKS, "horizon": HORIZON, "open_m": OPEN_M, "travel_m": DRAWER_TRAVEL,
    "slide_speed_m_s": SLIDE_SPEED, "initial_open_m": list(INITIAL_OPEN),
    "sources": {"settle_ticks, released": "cpp/src/env.cpp Env::step", "horizon": "xle_workbench/sim.py:145",
                "slide_speed_m_s": "xert/tasks/stack.py LINEAR_SPEED",
                "open_m": "two thirds of travel, chosen here (slice K)"}}

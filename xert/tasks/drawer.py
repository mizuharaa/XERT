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

"""Bench manipulation tasks. `task_id` names the scene and success rule an episode ran on.

`task_id` is never the LeRobot `task` field: that one stays the natural-language description
(recorder.py `task=`, teleop.TASK, lerobot_export.py `task`).
"""
TASKS = ("pick_place", "stack", "drawer")
POSE = ("x", "y", "z", "qw", "qx", "qy", "qz")  # Free-joint qpos order: world position, then wxyz quaternion.
BODIES = {"pick_place": ("block",), "stack": ("block", "base_block"), "drawer": ()}  # Free bodies a rebuild poses.
# Scalar object state equal to one joint's qpos (slice K), name -> joint. The drawer has no free pose: with its
# cabinet fixed per robot, the slide value is all a scene rebuild needs.
JOINTS = {"pick_place": {}, "stack": {}, "drawer": {"drawer.pos": "drawer_slide"}}
# Recorded world points a rebuilt scene must reproduce, read from MjData by kind (the site/body/geom of that name).
POINTS = {"pick_place": ("grasp", "tray"), "stack": ("grasp", "tray"), "drawer": ("handle", "grasp")}
POINT_KIND = {"grasp": "site", "tray": "body", "handle": "geom"}


def environment_names(task_id):
    """observation.environment_state names, in ManipulationEnv.metadata() order after the 12 joint entries.

    The one list teleop records and render-cameras / image-ACT rebuild scenes from: scalar joint states, free-body
    poses, tracked points, time fraction. For pick_place it equals the native bench observation names [12:]
    (tests/test_stack.py checks that).
    """
    return [*JOINTS[task_id], *(f"{body}.{axis}" for body in BODIES[task_id] for axis in POSE),
            *(f"{point}.{axis}" for point in POINTS[task_id] for axis in "xyz"), "time_fraction"]


def check_task_id(task_id):
    if task_id not in TASKS:
        raise ValueError(f"task_id must be one of {TASKS}, not {task_id!r}")
    return task_id

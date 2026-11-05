"""Bench manipulation tasks. `task_id` names the scene and success rule an episode ran on.

`task_id` is never the LeRobot `task` field: that one stays the natural-language description
(recorder.py `task=`, teleop.TASK, lerobot_export.py `task`).
"""
TASKS = ("pick_place", "stack")
POSE = ("x", "y", "z", "qw", "qx", "qy", "qz")  # Free-joint qpos order: world position, then wxyz quaternion.
BODIES = {"pick_place": ("block",), "stack": ("block", "base_block")}  # Free bodies a scene rebuild must pose.


def environment_names(task_id):
    """observation.environment_state names, in ManipulationEnv.metadata() order after the 12 joint entries.

    The one list teleop records and render-cameras / image-ACT rebuild scenes from. For pick_place it equals the
    native bench observation names [12:] (tests/test_stack.py checks that).
    """
    return [f"{body}.{axis}" for body in BODIES[task_id] for axis in POSE] + \
        ["grasp.x", "grasp.y", "grasp.z", "tray.x", "tray.y", "tray.z", "time_fraction"]


def check_task_id(task_id):
    if task_id not in TASKS:
        raise ValueError(f"task_id must be one of {TASKS}, not {task_id!r}")
    return task_id

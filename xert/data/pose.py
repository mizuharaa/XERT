"""Slice L: Cartesian grasp-site pose columns for end-effector-space VLAs, appended to a simulated bench dataset.

`observation.state.cartesian` and `action.cartesian` are [x, y, z, roll, pitch, yaw, gripper]: the spec's grasp site
in the robot's fixed base frame (metres), extrinsic x-y-z Euler radians, gripper 0 = spec closed .. 1 = spec open
(clipped). Forward kinematics only (mj_kinematics on the recorded bench model): a kinematic model pose of the
simulated arm, never a measurement and never hardware data.
"""
import numpy as np

# Root body of each bench arm (lerobot_export.PROFILES robots); its placement in the world is the base frame.
# xlerobot_left's root carries joint 1 (Rotation_L) itself, so only its reference placement is fixed, never xquat.
ROOTS = {"so101": "base", "xlerobot_left": "Rotation_Pitch"}


def base_transform(model, robot):
    """(position, wxyz quaternion) of the robot's root body in the world: its constant reference placement."""
    root = model.body(ROOTS[robot]).id
    if model.body_parentid[root] != 0:
        raise ValueError(f"{ROOTS[robot]} is not a worldbody child; its placement is not the base frame")
    return model.body_pos[root].copy(), model.body_quat[root].copy()


def in_base(position, matrix, base):
    """A world position and rotation matrix expressed in the base frame (position, rotation matrix)."""
    import mujoco
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, base[1])
    rotation = rotation.reshape(3, 3)
    return rotation.T @ (np.asarray(position) - base[0]), rotation.T @ matrix


def euler_xyz(matrix):
    """(roll, pitch, yaw) with matrix = Rz(yaw) Ry(pitch) Rx(roll). Roll and yaw are not unique at pitch = +-pi/2."""
    return np.array([np.arctan2(matrix[2, 1], matrix[2, 2]), np.arctan2(-matrix[2, 0], np.hypot(matrix[2, 1],
                     matrix[2, 2])), np.arctan2(matrix[1, 0], matrix[0, 0])])


def gripper_fraction(q, gripper):
    """Unclipped (q - closed) / (open - closed) from the spec's gripper block: 0 closed, 1 open."""
    return (q - gripper["closed"]) / (gripper["open"] - gripper["closed"])


def clip_gripper(values):
    """Gripper fractions clipped to [0, 1], with the clip count and the largest excursion beyond [0, 1]."""
    values = np.asarray(values, dtype=np.float64)
    excursion = np.maximum(values - 1, -values)
    return np.clip(values, 0, 1), {"clipped_frames": int(np.sum(excursion > 0)),
                                   "max_excursion": float(max(0., excursion.max(initial=0.)))}


def site_pose(model, data, qids, site, base, q):
    """(base-frame [x, y, z, roll, pitch, yaw], world site position) for the arm at joint radians q."""
    import mujoco
    data.qpos[qids] = q
    mujoco.mj_kinematics(model, data)
    world = data.site_xpos[site].copy()
    position, matrix = in_base(world, data.site_xmat[site].reshape(3, 3), base)
    return np.concatenate([position, euler_xyz(matrix)]), world

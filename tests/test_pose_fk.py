"""Slice L helpers (base env, MuJoCo only): fixed base frame, joint 1, the Euler convention, the gripper fraction.

Forward kinematics of the bench models; no dataset, no ledger row, no measurement of a real arm.
"""
from pathlib import Path

import mujoco
import numpy as np
import pytest

from xert.data.cameras import camera_model
from xert.data.pose import ROOTS, base_transform, clip_gripper, euler_xyz, gripper_fraction, in_base, site_pose
from xert.spec import load_spec
from xle_workbench.sim import ROBOTS as BENCH, SO101_BASE

ROOT = Path(__file__).resolve().parents[1]


def Rx(a):
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def Ry(a):
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def Rz(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def quat(matrix):
    result = np.zeros(4)
    mujoco.mju_mat2Quat(result, np.asarray(matrix, dtype=np.float64).flatten())
    return result


def oracle(euler, sequence="XYZ"):
    result = np.zeros(4)
    mujoco.mju_euler2Quat(result, np.asarray(euler, dtype=np.float64), sequence)
    return result


def same_rotation(a, b):  # q and -q are the same rotation.
    return min(np.max(np.abs(a - b)), np.max(np.abs(a + b)))


def arm(robot, task_id="pick_place"):
    model, _ = camera_model(robot, task_id)
    spec = load_spec(ROOT / f"xert/spec/{robot}.yaml")
    return model, mujoco.MjData(model), [model.joint(name).qposadr[0] for name in spec.joint_names], \
        model.site("grasp").id, spec


@pytest.mark.parametrize("robot", ROOTS)
def test_base_frame_is_the_root_reference_placement_and_never_moves(robot):
    model, data, qids, site, spec = arm(robot)
    base, root = base_transform(model, robot), model.body(ROOTS[robot]).id
    assert model.body_parentid[root] == 0 and spec.sites == ("grasp",)
    if robot == "so101":  # The bench placement of the Menagerie arm (yaw pi about world z).
        np.testing.assert_array_equal(base[0], SO101_BASE["pos"])
        np.testing.assert_array_equal(base[1], SO101_BASE["quat"])
    rng = np.random.default_rng(0)
    for _ in range(5):  # "An episode": the transform is the same whatever the joints do.
        site_pose(model, data, qids, site, base, rng.uniform(*spec.position_limits.T))
        for now, then in zip(base_transform(model, robot), base):
            np.testing.assert_array_equal(now, then)
    for task_id in ("stack", "drawer"):  # Same arm placement in every task scene.
        for now, then in zip(base_transform(camera_model(robot, task_id)[0], robot), base):
            np.testing.assert_array_equal(now, then)
    q = np.zeros(6)
    q[0] = .5
    site_pose(model, data, qids, site, base, q)
    if robot == "xlerobot_left":  # Joint 1 sits on the root body: its LIVE frame turns with Rotation_L.
        assert same_rotation(data.xquat[root], base[1]) > .1
    else:
        assert same_rotation(data.xquat[root], base[1]) < 1e-12


@pytest.mark.parametrize("robot", ROOTS)
def test_joint_1_alone_turns_the_pose_about_its_axis_in_the_base_frame(robot):
    model, data, qids, site, spec = arm(robot)
    base = base_transform(model, robot)
    q = np.asarray(BENCH[robot]["reset_pose"], dtype=np.float64)
    turned = q.copy()
    turned[0] += .4
    first, _ = site_pose(model, data, qids, site, base, q)
    joint = model.joint(spec.joint_names[0]).id
    anchor, rotation = in_base(data.xanchor[joint], np.eye(3), base)
    axis = rotation @ data.xaxis[joint]
    second, _ = site_pose(model, data, qids, site, base, turned)
    assert np.max(np.abs(second - first)) > .01  # Two configurations differing only in joint 1 differ.
    a, b = first[:3] - anchor, second[:3] - anchor
    assert abs(a @ axis - b @ axis) < 1e-12  # Height along the axis is kept,
    np.testing.assert_allclose(np.linalg.norm(a - (a @ axis) * axis), np.linalg.norm(b - (b @ axis) * axis),
                               rtol=0, atol=1e-12)  # and so is the distance from it.
    if robot == "so101":  # shoulder_pan turns about base -z: x, y and yaw change; z, roll, pitch do not.
        np.testing.assert_allclose(axis, [0, 0, -1], atol=1e-12)
        assert np.all(np.abs(second[[0, 1]] - first[[0, 1]]) > 1e-3)
        np.testing.assert_allclose(second[[2, 3, 4]], first[[2, 3, 4]], rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.angle(np.exp(1j * (second[5] - first[5]))), -.4, rtol=0, atol=1e-12)
    else:
        # Measured: Rotation_Pitch's placement turns local y to world up, so Rotation_L is the base -y axis and moves
        # base x and z; in this frame "vertical" is y, not z (an owner decision for DROID-style z-up consumers).
        np.testing.assert_allclose(axis, [0, -1, 0], atol=1e-5)
        assert np.all(np.abs(second[[0, 2]] - first[[0, 2]]) > 1e-3) and abs(second[1] - first[1]) < 1e-5


def test_frame_conversion_of_a_hand_built_pose():
    turn = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])  # 90 degrees about world z.
    base = (np.array([1., 2., 3.]), turn)
    world, matrix = base[0] + Rz(np.pi / 2) @ [.1, .2, .3], Rz(np.pi / 2) @ Rx(.3)
    position, local = in_base(world, matrix, base)
    np.testing.assert_allclose(position, [.1, .2, .3], rtol=0, atol=1e-12)
    np.testing.assert_allclose(euler_xyz(local), [.3, 0, 0], rtol=0, atol=1e-12)


@pytest.mark.parametrize("robot", ROOTS)
def test_euler_reproduces_the_site_quaternion_through_mujocos_extrinsic_xyz(robot):
    model, data, qids, site, spec = arm(robot)
    base, rng, checked = base_transform(model, robot), np.random.default_rng(1), 0
    for _ in range(200):
        pose, _ = site_pose(model, data, qids, site, base, rng.uniform(*spec.position_limits.T))
        if abs(pose[4]) > np.pi / 2 - .05:  # Gimbal lock band: roll and yaw are not unique there.
            continue
        _, matrix = in_base(data.site_xpos[site], data.site_xmat[site].reshape(3, 3), base)
        assert same_rotation(oracle(pose[3:]), quat(matrix)) < 1e-9
        checked += 1
    assert checked > 150
    for _ in range(200):  # Random rotations, not only reachable ones.
        q = rng.normal(size=4)
        matrix = np.zeros(9)
        mujoco.mju_quat2Mat(matrix, q / np.linalg.norm(q))
        euler = euler_xyz(matrix.reshape(3, 3))
        if abs(euler[1]) < np.pi / 2 - .05:
            assert same_rotation(oracle(euler), q / np.linalg.norm(q)) < 1e-9


def test_a_mixed_axis_case_is_extrinsic_and_an_intrinsic_reading_fails():
    roll, pitch, yaw = .3, -.7, 1.1
    matrix = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    np.testing.assert_allclose(euler_xyz(matrix), [roll, pitch, yaw], rtol=0, atol=1e-12)
    assert same_rotation(oracle([roll, pitch, yaw]), quat(matrix)) < 1e-12
    # Intrinsic x-y-z (MuJoCo lowercase) is Rx Ry Rz: the same three angles are a different rotation.
    assert same_rotation(oracle([roll, pitch, yaw], "xyz"), quat(matrix)) > .1
    assert np.max(np.abs(euler_xyz(Rx(roll) @ Ry(pitch) @ Rz(yaw)) - [roll, pitch, yaw])) > .1


def test_gripper_fraction_endpoints_midpoint_and_counted_clipping():
    for robot in ROOTS:
        gripper = load_spec(ROOT / f"xert/spec/{robot}.yaml").gripper
        # DROID's direction: open -> 0, closed -> 1.
        assert gripper_fraction(gripper["open"], gripper) == 0 and gripper_fraction(gripper["closed"], gripper) == 1
        assert gripper_fraction((gripper["closed"] + gripper["open"]) / 2, gripper) == pytest.approx(.5, abs=1e-15)
    spec = load_spec(ROOT / "xert/spec/so101.yaml")
    low, high = spec.position_limits[spec.joint_names.index("gripper")]
    raw = gripper_fraction(np.array([0, .7, 1.4, high, low]), spec.gripper)
    assert raw[3] == pytest.approx(1 - 1.7453292 / 1.4) and raw[3] < -.24  # The spec limit is past "open".
    assert raw[4] == pytest.approx(1 + .17453 / 1.4)  # And the lower limit is past "closed".
    clipped, clips = clip_gripper(raw)
    np.testing.assert_allclose(clipped, [1, .5, 0, 0, 1], rtol=0, atol=1e-15)
    assert clips == {"clipped_frames": 2, "max_excursion": pytest.approx(1.7453292 / 1.4 - 1)}
    assert clip_gripper([0, 1])[1] == {"clipped_frames": 0, "max_excursion": 0.}

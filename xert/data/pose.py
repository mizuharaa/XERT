"""Slice L: Cartesian grasp-site pose columns for end-effector-space VLAs, appended to a simulated bench dataset.

`observation.state.cartesian` and `action.cartesian` are [x, y, z, roll, pitch, yaw, gripper]: the spec's grasp site
in the robot's fixed base frame (metres), extrinsic x-y-z Euler radians, gripper 0 = spec closed .. 1 = spec open
(clipped). Forward kinematics only (mj_kinematics on the recorded bench model): a kinematic model pose of the
simulated arm, never a measurement and never hardware data. Every existing column is copied byte-equal.
"""
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from xert.data.cameras import (AUTOMATIC, TOLERANCE, _columns, _read_table, _recorded_models, _recorded_robot,
                               _recorded_task, camera_model, row_state)
from xert.data.lerobot_export import _publish
from xert.data.lerobot_io import _fresh, _hashes
from xle_workbench.storage import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[2]
# Root body of each bench arm (lerobot_export.PROFILES robots); its placement in the world is the base frame.
# xlerobot_left's root carries joint 1 (Rotation_L) itself, so only its reference placement is fixed, never xquat.
ROOTS = {"so101": "base", "xlerobot_left": "Rotation_Pitch"}
NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
FEATURE = {"dtype": "float32", "shape": (7,), "names": NAMES}
COLUMNS = {"observation.state.cartesian": "observation.state", "action.cartesian": "action"}
EULER = "extrinsic x-y-z radians: R = Rz(yaw) Ry(pitch) Rx(roll) (DROID; MuJoCo mju_euler2Quat sequence 'XYZ')"
HARDWARE = ("add-pose takes simulator recordings and exports only: admission needs the recorded bench model "
            "fingerprint (xert_recording.json source.bench, xert_export.json simulation.model_sha256). Hardware "
            "imports carry none: a so101_follower dataset has no XERT manifest, and to-store writes "
            "derived_frame_pairs episodes with no simulation provenance, which export-lerobot refuses "
            "(hardware=False, simulation_only=True). A hardware pose path is slice L2")


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


def _action_radians(source, names, joints):
    """(decode, action semantics) for the dataset's `action`: radians as recorded (P4.1) or decoded from [-1, 1]."""
    if (source / "xert_export.json").exists():
        manifest = json.loads((source / "xert_export.json").read_text("utf-8"))
        ranges = {json.dumps(episode["simulation"]["action_joint_ranges_radians"])
                  for episode in manifest["source_episodes"]}
        if len(ranges) != 1 or names != joints:
            raise ValueError("Export episodes mix action_joint_ranges_radians or the action names differ; refused")
        low, high = np.asarray(json.loads(ranges.pop()), dtype=np.float64).T
        return (lambda action: low + (action + 1) / 2 * (high - low),
                {"action_semantics": manifest["action_semantics"], "decode": "q = low + (a + 1) / 2 * (high - low), "
                 "action_joint_ranges_radians from xert_export.json", "action_joint_ranges_radians":
                 np.stack([low, high], 1).tolist()})
    manifest = json.loads((source / "xert_recording.json").read_text("utf-8"))
    if names != [f"{name}.pos" for name in joints] or "radians" not in manifest["action_units"]:
        raise ValueError(f"Recording action is not joint radians named {joints}.pos; refused")
    return lambda action: action, {"action_semantics": manifest["action_units"], "decode": "none: radians as recorded"}


def _add_one(source, staging, robot, task_id, model, fingerprint, spec, repo_id):
    import mujoco
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if _recorded_models(source) != {fingerprint}:
        raise ValueError(f"{source} was not recorded on the current {robot} {task_id} bench model; refused")
    meta = LeRobotDatasetMetadata(repo_id, root=source)
    features = {key: value for key, value in meta.features.items() if key not in AUTOMATIC}
    if any(value["dtype"] in ("image", "video") for value in features.values()):
        raise ValueError("Dataset has image or video features: run add-pose before render-cameras")
    if any(key.endswith(".cartesian") for key in features):
        raise ValueError("Dataset already carries Cartesian pose columns")
    joints, gripper = list(spec.joint_names), spec.gripper
    where = _columns(features, joints, bodies=(), points=("grasp",))
    decode, semantics = _action_radians(source, features["action"]["names"], joints)
    data, qids = mujoco.MjData(model), [model.joint(name).qposadr[0] for name in joints]
    [site_name] = spec.sites
    site, base, grip = model.site(site_name).id, base_transform(model, robot), joints.index(gripper["joint"])
    tasks = dict(zip(meta.tasks["task_index"], meta.tasks.index))
    table = _read_table(source)
    dataset = LeRobotDataset.create(repo_id=repo_id, root=staging, robot_type=meta.robot_type, fps=meta.fps,
                                    features={**features, **dict.fromkeys(COLUMNS, FEATURE)}, use_videos=False)
    episodes = table.column("episode_index").to_numpy()
    bounds = [0, *(np.flatnonzero(np.diff(episodes)) + 1).tolist(), len(episodes)]
    written, raw_gripper, error = {key: [] for key in COLUMNS}, {key: [] for key in COLUMNS}, 0.
    try:
        for start, stop in zip(bounds[:-1], bounds[1:]):  # One episode's rows in Python at a time.
            rows = table.slice(start, stop - start)
            columns = {key: rows.column(key).to_pylist() for key in [*features, "task_index"]}
            for row in range(stop - start):
                frame = {key: columns[key][row] if feature["dtype"] == "string" else
                         np.asarray(columns[key][row], dtype=feature["dtype"]).reshape(feature["shape"])
                         for key, feature in features.items()}
                state = row_state(frame, where)
                poses = {}
                for key, q in (("observation.state.cartesian", state["joints"]),
                               ("action.cartesian", decode(frame["action"].astype(np.float64)))):
                    pose, world = site_pose(model, data, qids, site, base, q)
                    if key == "observation.state.cartesian":
                        error = max(error, float(np.max(np.abs(world - state["grasp"]))))
                    raw_gripper[key].append(gripper_fraction(q[grip], gripper))
                    poses[key] = np.append(pose, clip_gripper(raw_gripper[key][-1])[0]).astype(np.float32)
                    written[key].append(poses[key])
                dataset.add_frame({**frame, **poses, "task": tasks[columns["task_index"][row]]})
            dataset.save_episode(parallel_encoding=False)
    finally:
        dataset.finalize()
    del dataset
    if error > TOLERANCE:
        raise ValueError(f"FK grasp position disagrees with the recorded grasp.x/y/z by {error} m")
    output = _read_table(staging)
    if output.num_rows != table.num_rows or not all(output.column(name).equals(table.column(name))
                                                   for name in table.column_names):
        raise ValueError("Pose dataset changed a source column")
    for key, values in written.items():
        if not np.array_equal(np.asarray(output.column(key).to_pylist(), dtype=np.float32), np.stack(values)):
            raise ValueError(f"Stored {key} differs from the computed pose")
    del output
    for path in source.iterdir():  # Provenance sidecars (xert_recording.json, journal, export manifest, roundtrip).
        if path.is_file():
            shutil.copyfile(path, staging / path.name)
    clips = {key: clip_gripper(values)[1] for key, values in raw_gripper.items()}
    verification = {"frames": table.num_rows, "episodes": len(set(episodes.tolist())),
                    "grasp_world_max_error_m": error, "tolerance_m": TOLERANCE, "source_columns_identical": True}
    atomic_json(staging / "xert_pose.json", {
        "version": 1, "backend": "mujoco", "hardware": False, "pose": "kinematic model pose of the simulated arm",
        "robot": robot, "task_id": task_id,
        "features": {key: {**FEATURE, "shape": list(FEATURE["shape"]), "from": f"FK of {stream}"}
                     for key, stream in COLUMNS.items()},
        "action": {**semantics, "executed_action": "xert.executed_action (exports) gets no Cartesian column"},
        "site": site_name, "root_body": ROOTS[robot],
        "base_frame": {"position": base[0].tolist(), "quat_wxyz": base[1].tolist(),
                       "definition": "reference placement of the root body in the world (model.body_pos/body_quat)"},
        "position_units": "metres", "euler": EULER,
        "gripper": {**gripper, "definition": "(q - closed) / (open - closed), clipped to [0, 1]; 0 = spec closed, "
                    "1 = spec open; raw joint columns stay in the dataset", "clips": clips},
        "fk": "mujoco.mj_kinematics on the recorded bench model, qpos = recorded joint radians (float32 -> float64)",
        "model_sha256": fingerprint, "spec_sha256": sha256(ROOT / f"xert/spec/{robot}.yaml"),
        "mujoco_version": mujoco.__version__, "lerobot_version": importlib.metadata.version("lerobot"),
        "source": str(source), "source_sha256": _hashes(source), "verification": verification})
    return {**verification, "clips": clips}


def add_pose(source, output, repo_id="xert/runtime"):
    """A bench recording/export, or a P4.1 session of episode_* recordings, into a fresh output. Never overwrites."""
    from xert.spec import load_spec
    source, output = Path(source).absolute(), _fresh(output)
    recordings = [source] if (source / "meta/info.json").exists() else sorted(source.glob("episode_*"))
    if not recordings or any(not _recorded_models(path) or None in _recorded_models(path) for path in recordings):
        raise ValueError(HARDWARE)
    robots = {_recorded_robot(path) for path in recordings}
    if len(robots) != 1 or not robots <= set(ROOTS):
        raise ValueError(f"Add pose for one bench robot of {sorted(ROOTS)} per run; datasets name {robots}")
    tasks = {_recorded_task(path) for path in recordings}
    if len(tasks) != 1 or None in tasks:
        raise ValueError(f"Add pose for one bench task per run; datasets name task_id {sorted(map(str, tasks))}")
    robot, task_id = robots.pop(), tasks.pop()
    model, fingerprint = camera_model(robot, task_id)  # The arm is the same body chain in every task scene.
    spec = load_spec(ROOT / f"xert/spec/{robot}.yaml")
    with tempfile.TemporaryDirectory(prefix=".xert-pose-", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        if recordings != [source]:
            staging.mkdir()
        results = {path.name: _add_one(path, staging if path == source else staging / path.name, robot, task_id,
                                       model, fingerprint, spec, repo_id) for path in recordings}
        _publish(staging, output)
    return results


def add_command(commands):
    parser = commands.add_parser("add-pose", help="Append Cartesian grasp-site pose columns to a simulated bench "
                                                  "dataset (before render-cameras)")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="A P4.1 recording, a P2.1 export, or a P4.1 session directory of episode_* recordings")
    parser.add_argument("--output", type=Path, required=True)


def execute(args):
    print(json.dumps(add_pose(args.dataset, args.output)))

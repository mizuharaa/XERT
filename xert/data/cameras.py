"""P2.3: render a fixed scene camera and a wrist camera offline into a copy of a bench dataset.

Each frame's scene is rebuilt from its recorded joint positions and block pose (the bench's whole
qpos: the tray is static), so live teleop never renders and never risks a control deadline.
Images are simulator renders, not real camera data.
"""
from functools import cache
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from xert.data.lerobot_export import _publish
from xert.tasks import BODIES, POSE, check_task_id
from xle_workbench.storage import atomic_json, sha256

# Per bench robot (xle_workbench.sim ROBOTS). Scene cameras are in the world frame, wrist cameras in a gripper body
# frame, metres. A "model_camera" entry renders a camera the vendored model already defines.
CAMERAS = {
    # P2.3. Scene: above the far end of the table, facing the arm, aimed between the block start area and the tray.
    # Wrist: Fixed_Jaw frame, where the jaw tips point along local -y (grasp site at y = -0.10).
    "xlerobot_left": {
        "scene": {"body": "world", "pos": [.5, -.15, .35], "target": [.24, .04, .1], "up": [0, 0, 1], "fovy": 45},
        "wrist": {"body": "Fixed_Jaw", "pos": [.05, -.03, 0], "target": [-.01, -.1, 0], "up": [1, 0, 0], "fovy": 75},
    },
    # B2. Scene, by eye: the arm base is at the far end of the table, (0.42, 0.04, 0.08) facing -x
    # (sim.SO101_BASE), so the camera sits past the near end, off to -y, aimed between the block area, tray and arm.
    # Wrist: Menagerie's own wrist_cam on body camera_mount (so101_xert.xml), rendered as it is defined there.
    "so101": {
        "scene": {"body": "world", "pos": [-.02, -.25, .4], "target": [.3, .04, .15], "up": [0, 0, 1], "fovy": 45},
        "wrist": {"model_camera": "wrist_cam"},
    },
}
# Written to xert_cameras.json as "camera_placement".
PLACEMENT = {
    "xlerobot_left": "chosen by eye in simulation; not the real SO-101 wrist camera mount or intrinsics",
    "so101": "scene: chosen by eye in simulation. wrist: Menagerie's wrist_cam, plausibly the 32x32 UVC wrist mount; "
             "mount pose and fovy 61 are unsourced and not measured on Daniel's arm; measure the real lens FOV before "
             "any sim-to-real claim",
}
SIZE = 128  # Square pixels: a 4x4 grid after ACT's stride-32 ResNet-18 (docs/learn/P2.3.md).
BLOCK = [f"block.{axis}" for axis in POSE]


def checks(bodies=("block",)):
    """Recorded values each rebuilt scene must reproduce, read by name independently of the qpos order."""
    result = {"grasp": ["grasp.x", "grasp.y", "grasp.z"], "tray": ["tray.x", "tray.y", "tray.z"]}
    for body in bodies:
        result[f"{body}_position"] = [f"{body}.{axis}" for axis in POSE[:3]]
        result[f"{body}_orientation"] = [f"{body}.{axis}" for axis in POSE[3:]]
    return result


CHECKS = checks()  # The pick-and-place bench.
TOLERANCE = 1e-5  # Metres, or unit-quaternion components; float32 rounding stays far below this.
AUTOMATIC = ("timestamp", "frame_index", "episode_index", "index", "task_index")  # The writer recomputes them.


def _xyaxes(pos, target, up):
    """MuJoCo camera x and y axes in the parent frame; the camera looks along its -z."""
    forward = np.subtract(target, pos) / np.linalg.norm(np.subtract(target, pos))
    right = np.cross(forward, up) / np.linalg.norm(np.cross(forward, up))
    return np.concatenate([right, np.cross(right, forward)])


def _so101_spec():
    """(uncompiled MjSpec, recorded-model fingerprint) of the SO-101 bench, rebuilt step by step like sim.so101_scene().

    That function returns only a compiled model and an XML that does not parse back, and sim.py is not edited here
    because its bytes are hashed into every recording's implementation_sha256. The rebuild is refused unless it
    writes exactly so101_scene()'s XML; the fingerprint is ManipulationEnv's for the SO-101.
    """
    import xml.etree.ElementTree as ET
    import mujoco
    from xle_workbench.sim import SO101_ASSETS, SO101_BASE, SO101_TIMESTEP, model_xml, so101_scene
    _, xml, source = so101_scene()  # Also checks the vendored Menagerie checksums.
    props = ET.Element("mujoco")
    world = ET.SubElement(props, "worldbody")
    for element in ET.fromstring(model_xml()).find("worldbody"):
        if element.get("name") not in ("mount", "Rotation_Pitch"):
            world.append(element)
    spec = mujoco.MjSpec.from_file(str(SO101_ASSETS / "so101_xert.xml"))
    spec.option.timestep = SO101_TIMESTEP
    spec.body("base").pos, spec.body("base").quat = SO101_BASE["pos"], SO101_BASE["quat"]
    for actuator in spec.actuators:
        joint = spec.joint(actuator.target)
        actuator.ctrlrange = [max(actuator.ctrlrange[0], joint.range[0]), min(actuator.ctrlrange[1], joint.range[1])]
    spec.attach(mujoco.MjSpec.from_string(ET.tostring(props, encoding="unicode")), frame=spec.worldbody.add_frame(),
                prefix="")
    if spec.to_xml() != xml:
        raise RuntimeError("Camera scene differs from xle_workbench.sim.so101_scene(); update _so101_spec()")
    return spec, hashlib.sha256(xml.encode() + json.dumps(source, sort_keys=True).encode()).hexdigest()


def _bench_spec(robot):
    """(uncompiled MjSpec, recorded-model fingerprint) of a robot's pick-and-place bench (ManipulationEnv's model)."""
    import mujoco
    from xle_workbench.sim import ASSETS, model_fingerprint, model_xml, verified_source
    if robot == "so101":
        return _so101_spec()
    source = verified_source(ASSETS)  # Vendored files are checksummed and only read.
    xml = model_xml()
    return mujoco.MjSpec.from_string(xml), model_fingerprint(xml, source)


@cache
def bench_fingerprint(robot):
    """The pick-and-place bench model fingerprint that recordings of `robot` carry (source.bench.model_sha256)."""
    return _bench_spec(robot)[1]


def camera_model(robot="xlerobot_left", task_id="pick_place"):
    """(model, recorded-model fingerprint): the robot's bench (or stack) model with both cameras added in memory."""
    import mujoco
    if check_task_id(task_id) == "stack":
        from xert.tasks.stack import stack_spec
        spec, fingerprint, _ = stack_spec(robot)
    else:
        spec, fingerprint = _bench_spec(robot)
    for name, camera in CAMERAS[robot].items():
        if "model_camera" in camera:
            continue
        added = spec.body(camera["body"]).add_camera(name=name, pos=camera["pos"], fovy=camera["fovy"])
        added.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
        added.alt.xyaxes = _xyaxes(camera["pos"], camera["target"], camera["up"])
    return spec.compile(), fingerprint


class CameraRenderer:
    """Offscreen renderer; MuJoCo's default GL on Windows opens a hidden GLFW context, CI uses MUJOCO_GL=osmesa."""
    def __init__(self, size=SIZE, robot="xlerobot_left", task_id="pick_place"):
        import mujoco
        from xle_workbench.sim import ROBOTS
        self.mujoco, self.size, self.robot, self.task_id = mujoco, size, robot, task_id
        self.cameras, self.joints = CAMERAS[robot], ROBOTS[robot]["joints"]
        self.model, self.model_sha256 = camera_model(robot, task_id)
        self.data = mujoco.MjData(self.model)
        self.qids = [self.model.joint(name).qposadr[0] for name in self.joints]
        self.bodies = {body: self.model.joint(f"{body}_joint").qposadr[0] for body in BODIES[task_id]}
        self.object_names = [f"{body}.{axis}" for body in self.bodies for axis in POSE]
        self.checks = checks(self.bodies)
        self.option = mujoco.MjvOption()
        self.option.sitegroup[:] = 0  # Hide the red grasp marker: it is a debugging aid, not part of the scene.
        self.renderer = mujoco.Renderer(self.model, size, size)

    def render(self, joints, objects):
        """Pose the scene from joint radians and named free-body poses {"block.x": ..., "block.qw": ...} (every name in
        object_names); return images and checks. Offline reconstruction and live image-ACT frames both come here."""
        self.data.qpos[self.qids] = joints
        for body, address in self.bodies.items():
            self.data.qpos[address:address + 7] = [objects[f"{body}.{axis}"] for axis in POSE]
        self.mujoco.mj_forward(self.model, self.data)
        images = {}
        for name, camera in self.cameras.items():
            self.renderer.update_scene(self.data, camera=camera.get("model_camera", name), scene_option=self.option)
            images[f"observation.images.{name}"] = self.renderer.render().copy()
        achieved = {"grasp": self.data.site("grasp").xpos.copy(), "tray": self.data.body("tray").xpos.copy()}
        for body in self.bodies:
            achieved[f"{body}_position"] = self.data.body(body).xpos.copy()
            achieved[f"{body}_orientation"] = self.data.body(body).xquat.copy()
        return images, achieved

    def close(self):
        self.renderer.close()


def _recorded_models(root):
    """Model fingerprints a P4.1 recording or a P2.1 export says it was recorded on."""
    if (root / "xert_recording.json").exists():
        bench = json.loads((root / "xert_recording.json").read_text("utf-8"))["source"].get("bench")
        return {bench["model_sha256"]} if bench else set()
    if (root / "xert_export.json").exists():
        manifest = json.loads((root / "xert_export.json").read_text("utf-8"))
        return {episode["simulation"].get("model_sha256") for episode in manifest["source_episodes"]}
    return set()


def _recorded_robot(root):
    """The bench robot a P4.1 recording (embodiment) or a P2.1 export (robot_type) names, else None."""
    for name, key in (("xert_recording.json", "embodiment"), ("xert_export.json", "robot_type")):
        if (root / name).exists():
            return json.loads((root / name).read_text("utf-8")).get(key)
    return None


def _recorded_task(root):
    """task_id of a recording/export: its manifest's, else (legacy data, review H-07) inferred from fingerprints.

    Legacy data is pick_place only when every recorded model fingerprint is the named robot's pick-and-place bench
    model; any other fingerprint is refused, never guessed. None: no bench scene was recorded (e.g. record-motion).
    """
    for name in ("xert_recording.json", "xert_export.json"):
        if (root / name).exists():
            manifest = json.loads((root / name).read_text("utf-8"))
            if "task_id" in manifest:
                return check_task_id(manifest["task_id"])
    models, robot = _recorded_models(root), _recorded_robot(root)
    if not models:
        return None
    if robot in CAMERAS and models == {bench_fingerprint(robot)}:
        return "pick_place"
    raise ValueError(f"{root} names no task_id and was not recorded on the pinned bench model of {robot} "
                     "(pick-and-place); its task is unknown and is not guessed")


def _columns(features, robot_joints, bodies=("block",)):
    """(feature, index) of every state entry the scene needs, by name across both observation features."""
    where = {name: (key, i) for key in ("observation.state", "observation.environment_state")
             for i, name in enumerate(features.get(key, {}).get("names") or [])}
    # P4.1 recordings name joints "<joint>.pos"; P2.1 exports keep the native "joint_position.<joint>".
    joints = [f"{name}.pos" if f"{name}.pos" in where else f"joint_position.{name}" for name in robot_joints]
    wanted = {"joints": joints, **{body: [f"{body}.{axis}" for axis in POSE] for body in bodies}, **checks(bodies)}
    missing = [name for names in wanted.values() for name in names if name not in where]
    if missing:
        raise ValueError(f"Cannot rebuild the bench scene; dataset lacks {missing}")
    return {part: [where[name] for name in names] for part, names in wanted.items()}


def scene_objects(state, bodies):
    """Named free-body poses for CameraRenderer.render from the per-body vectors `_columns` located."""
    return {f"{body}.{axis}": value for body in bodies for axis, value in zip(POSE, state[body])}


def _error(name, achieved, recorded):
    if name.endswith("_orientation"):  # q and -q are the same rotation.
        return float(min(np.max(np.abs(achieved - recorded)), np.max(np.abs(achieved + recorded))))
    return float(np.max(np.abs(achieved - recorded)))


def _read_table(root):
    import pyarrow as pa
    import pyarrow.parquet as pq
    return pa.concat_tables([pq.read_table(path) for path in sorted(root.glob("data/**/*.parquet"))])


def _render_one(source, staging, renderer, repo_id):
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if _recorded_models(source) != {renderer.model_sha256}:
        raise ValueError(f"{source} was not recorded on the pinned bench model; its scene cannot be rebuilt")
    # Metadata + raw Parquet only: LeRobotDataset(source) would build a Hugging Face Arrow cache.
    meta = LeRobotDatasetMetadata(repo_id, root=source)
    features = {key: value for key, value in meta.features.items() if key not in AUTOMATIC}
    if any(value["dtype"] in ("image", "video") for value in features.values()):
        raise ValueError("Dataset already has image or video features")
    tasks = dict(zip(meta.tasks["task_index"], meta.tasks.index))
    table = _read_table(source)
    where = _columns(features, renderer.joints, renderer.bodies)
    images = {f"observation.images.{name}": {"dtype": "image", "shape": (renderer.size, renderer.size, 3),
                                             "names": ["height", "width", "channels"]} for name in renderer.cameras}
    dataset = LeRobotDataset.create(repo_id=repo_id, root=staging, robot_type=meta.robot_type, fps=meta.fps,
                                    features={**features, **images}, use_videos=False)
    episodes = table.column("episode_index").to_numpy()
    bounds = [0, *(np.flatnonzero(np.diff(episodes)) + 1).tolist(), len(episodes)]
    digests, errors = [], dict.fromkeys(renderer.checks, 0.)
    try:
        for start, stop in zip(bounds[:-1], bounds[1:]):  # One episode's rows in Python at a time.
            rows = table.slice(start, stop - start)
            columns = {key: rows.column(key).to_pylist() for key in [*features, "task_index"]}
            for row in range(stop - start):
                frame = {}
                for key, feature in features.items():
                    value = columns[key][row]
                    frame[key] = value if feature["dtype"] == "string" else \
                        np.asarray(value, dtype=feature["dtype"]).reshape(feature["shape"])
                state = {part: np.asarray([frame[key][i] for key, i in entries], dtype=np.float64)
                         for part, entries in where.items()}
                rendered, achieved = renderer.render(state["joints"], scene_objects(state, renderer.bodies))
                for name in renderer.checks:
                    errors[name] = max(errors[name], _error(name, achieved[name], state[name]))
                digests.append({key: hashlib.sha256(image.tobytes()).hexdigest() for key, image in rendered.items()})
                dataset.add_frame({**frame, **rendered, "task": tasks[columns["task_index"][row]]})
            dataset.save_episode(parallel_encoding=False)
    finally:
        dataset.finalize()
    del dataset
    if max(errors.values()) > TOLERANCE:
        raise ValueError(f"Rebuilt scene disagrees with the recorded grasp, tray or block: {errors}")
    verification = _verify(table, digests, staging, renderer.cameras)
    for path in source.iterdir():  # Provenance sidecars (xert_recording.json, journal, roundtrip, export manifest).
        if path.is_file():
            shutil.copyfile(path, staging / path.name)
    manifest = {
        "version": 1, "backend": "mujoco", "hardware": False,
        "images": "offline MuJoCo renders of the simulated bench, not real camera data",
        "task_id": renderer.task_id,
        "reconstruction": f"qpos = recorded joint positions + {' + '.join(renderer.bodies)} pose (float32), then "
                          "mj_forward; tray is static",
        "reconstruction_max_error": errors, "tolerance": TOLERANCE,
        "cameras": {name: {**camera, "feature": f"observation.images.{name}", "height": renderer.size,
                           "width": renderer.size, **_placement(renderer.model, camera)}
                    for name, camera in renderer.cameras.items()},
        "camera_placement": PLACEMENT[renderer.robot],
        "hidden": "all sites (the grasp marker)",
        "image_storage": "LeRobot dtype image: lossless PNG in the data Parquet",
        "model_sha256": renderer.model_sha256, "mujoco_version": renderer.mujoco.__version__,
        "mujoco_gl": os.environ.get("MUJOCO_GL", "default"), "lerobot_version": importlib.metadata.version("lerobot"),
        "source": str(source),
        "source_sha256": {path.relative_to(source).as_posix(): sha256(path)
                          for path in sorted(source.rglob("*")) if path.is_file()},
        "verification": verification}
    atomic_json(staging / "xert_cameras.json", manifest)
    return verification


def _placement(model, camera):
    """A by-eye camera's axes, or a model camera's pose and fovy in its body frame, as compiled."""
    if "model_camera" not in camera:
        return {"xyaxes": _xyaxes(camera["pos"], camera["target"], camera["up"]).tolist()}
    added = model.camera(camera["model_camera"])
    return {"body": model.body(added.bodyid[0]).name, "pos": added.pos.tolist(), "quat": added.quat.tolist(),
            "fovy": float(added.fovy[0])}


def _verify(source_table, digests, root, cameras):
    """Stream the written Parquet: source columns byte-identical, every PNG decodes to the rendered pixels."""
    import pyarrow.parquet as pq
    from PIL import Image
    offset = 0
    for path in sorted(root.glob("data/**/*.parquet")):
        with pq.ParquetFile(path) as parquet:  # Closed before the staging directory is renamed (Windows).
            # Single-threaded: threaded reads left Arrow's mimalloc holding memory per row group.
            for batch in parquet.iter_batches(batch_size=64, use_threads=False):
                expected = source_table.slice(offset, batch.num_rows)
                for name in source_table.column_names:
                    if not batch.column(name).equals(expected.column(name).combine_chunks()):
                        raise ValueError(f"Rendered dataset changed source column {name}")
                for key in (f"observation.images.{name}" for name in cameras):
                    for row, image in enumerate(batch.column(key).field("bytes").to_pylist()):
                        pixels = np.asarray(Image.open(io.BytesIO(image)).convert("RGB"))
                        if hashlib.sha256(pixels.tobytes()).hexdigest() != digests[offset + row][key]:
                            raise ValueError(f"Stored image differs from the render: {key} frame {offset + row}")
                offset += batch.num_rows
    if offset != len(digests) or offset != source_table.num_rows:
        raise ValueError("Rendered dataset frame count differs from source")
    return {"frames": offset, "episodes": len(set(source_table.column("episode_index").to_pylist())),
            "source_columns_identical": True, "images_lossless": True}


def render_cameras(source, output, size=SIZE, repo_id="xert/runtime"):
    """A recording/export, or a P4.1 session of episode_* recordings, into a fresh output. Never overwrites."""
    source, output = Path(source).absolute(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    recordings = [source] if (source / "meta/info.json").exists() else sorted(source.glob("episode_*"))
    if not recordings:
        raise ValueError("Source is neither a LeRobot dataset nor a session of episode_* datasets")
    robots = {_recorded_robot(path) for path in recordings}
    if len(robots) != 1 or not robots <= set(CAMERAS):
        raise ValueError(f"Cameras are defined for one bench robot of {sorted(CAMERAS)} per run; dataset names {robots}")
    tasks = {_recorded_task(path) for path in recordings}
    if len(tasks) != 1 or None in tasks:
        raise ValueError(f"Render one bench task per run; datasets name task_id {sorted(map(str, tasks))}")
    output.parent.mkdir(parents=True, exist_ok=True)
    renderer = CameraRenderer(size, robots.pop(), tasks.pop())
    try:
        with tempfile.TemporaryDirectory(prefix=".xert-cameras-", dir=output.parent) as temporary:
            staging = Path(temporary) / "dataset"
            if recordings != [source]:
                staging.mkdir()
            results = {path.name: _render_one(path, staging if path == source else staging / path.name,
                                              renderer, repo_id) for path in recordings}
            _publish(staging, output)
    finally:
        renderer.close()
    return results


def add_command(commands):
    parser = commands.add_parser("render-cameras", help="Add offline scene + wrist camera images to a bench dataset")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="A P4.1 recording, a P2.1 export, or a P4.1 session directory of episode_* recordings")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=SIZE, help="Square image side in pixels (max 480)")


def execute(args):
    print(json.dumps(render_cameras(args.dataset, args.output, args.size)))

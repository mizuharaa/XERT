"""P2.3: render a fixed scene camera and a wrist camera offline into a copy of a bench dataset.

Each frame's scene is rebuilt from its recorded joint positions and block pose (the bench's whole
qpos: the tray is static), so live teleop never renders and never risks a control deadline.
Images are simulator renders, not real camera data.
"""
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from xert.data.lerobot_export import _publish
from xle_workbench.storage import atomic_json, sha256

# Scene camera: world frame, metres; above the far end of the table, facing the arm, aimed between
# the block start area and the tray.
# Wrist camera: Fixed_Jaw body frame, where the jaw tips point along local -y (grasp site at y = -0.10).
CAMERAS = {
    "scene": {"body": "world", "pos": [.5, -.15, .35], "target": [.24, .04, .1], "up": [0, 0, 1], "fovy": 45},
    "wrist": {"body": "Fixed_Jaw", "pos": [.05, -.03, 0], "target": [-.01, -.1, 0], "up": [1, 0, 0], "fovy": 75},
}
SIZE = 128  # Square pixels: a 4x4 grid after ACT's stride-32 ResNet-18 (docs/learn/P2.3.md).
BLOCK = [f"block.{axis}" for axis in ("x", "y", "z", "qw", "qx", "qy", "qz")]
CHECKS = {"grasp": ["grasp.x", "grasp.y", "grasp.z"], "tray": ["tray.x", "tray.y", "tray.z"]}
TOLERANCE_M = 1e-5  # Float32-rounded state moves the recorded grasp/tray far less than this.
AUTOMATIC = ("timestamp", "frame_index", "episode_index", "index", "task_index")  # The writer recomputes them.


def _xyaxes(pos, target, up):
    """MuJoCo camera x and y axes in the parent frame; the camera looks along its -z."""
    forward = np.subtract(target, pos) / np.linalg.norm(np.subtract(target, pos))
    right = np.cross(forward, up) / np.linalg.norm(np.cross(forward, up))
    return np.concatenate([right, np.cross(right, forward)])


def camera_model():
    """(model, recorded-model fingerprint): the bench model with both cameras added in memory."""
    import mujoco
    from xle_workbench.sim import ASSETS, model_fingerprint, model_xml, verified_source
    source = verified_source(ASSETS)  # Vendored files are checksummed and only read.
    xml = model_xml()
    spec = mujoco.MjSpec.from_string(xml)
    for name, camera in CAMERAS.items():
        added = spec.body(camera["body"]).add_camera(name=name, pos=camera["pos"], fovy=camera["fovy"])
        added.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
        added.alt.xyaxes = _xyaxes(camera["pos"], camera["target"], camera["up"])
    return spec.compile(), model_fingerprint(xml, source)


class CameraRenderer:
    """Offscreen renderer; MuJoCo's default GL on Windows opens a hidden GLFW context, CI uses MUJOCO_GL=osmesa."""
    def __init__(self, size=SIZE):
        import mujoco
        from xle_workbench.sim import JOINTS
        self.mujoco, self.size = mujoco, size
        self.model, self.model_sha256 = camera_model()
        self.data = mujoco.MjData(self.model)
        self.qids = [self.model.joint(name).qposadr[0] for name in JOINTS]
        self.block = self.model.joint("block_joint").qposadr[0]
        self.option = mujoco.MjvOption()
        self.option.sitegroup[:] = 0  # Hide the red grasp marker: it is a debugging aid, not part of the scene.
        self.renderer = mujoco.Renderer(self.model, size, size)

    def render(self, joints, block):
        """Pose the scene from joint radians and the block's [x y z qw qx qy qz]; return images and checks."""
        self.data.qpos[self.qids] = joints
        self.data.qpos[self.block:self.block + 7] = block
        self.mujoco.mj_forward(self.model, self.data)
        images = {}
        for name in CAMERAS:
            self.renderer.update_scene(self.data, camera=name, scene_option=self.option)
            images[f"observation.images.{name}"] = self.renderer.render().copy()
        return images, {"grasp": self.data.site("grasp").xpos.copy(), "tray": self.data.body("tray").xpos.copy()}

    def close(self):
        self.renderer.close()


def _recorded_models(root):
    """Model fingerprints a P4.1 recording or a P2.1 export says it was recorded on."""
    if (root / "xert_recording.json").exists():
        bench = json.loads((root / "xert_recording.json").read_text("utf-8"))["source"].get("bench")
        return {bench["model_sha256"]} if bench else set()
    if (root / "xert_export.json").exists():
        manifest = json.loads((root / "xert_export.json").read_text("utf-8"))
        return {episode["simulation"]["model_sha256"] for episode in manifest["source_episodes"]}
    return set()


def _columns(features):
    """(feature, index) of every state entry the scene needs, by name across both observation features."""
    from xle_workbench.sim import JOINTS
    where = {name: (key, i) for key in ("observation.state", "observation.environment_state")
             for i, name in enumerate(features.get(key, {}).get("names") or [])}
    # P4.1 recordings name joints "<joint>.pos"; P2.1 exports keep the native "joint_position.<joint>".
    joints = [f"{name}.pos" if f"{name}.pos" in where else f"joint_position.{name}" for name in JOINTS]
    wanted = {"joints": joints, "block": BLOCK, **CHECKS}
    missing = [name for names in wanted.values() for name in names if name not in where]
    if missing:
        raise ValueError(f"Cannot rebuild the bench scene; dataset lacks {missing}")
    return {part: [where[name] for name in names] for part, names in wanted.items()}


def _render_one(source, staging, renderer, repo_id):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if _recorded_models(source) != {renderer.model_sha256}:
        raise ValueError(f"{source} was not recorded on the pinned bench model; its scene cannot be rebuilt")
    original = LeRobotDataset(repo_id=repo_id, root=source, video_backend="pyav", download_videos=False)
    try:
        features = {key: value for key, value in original.meta.features.items() if key not in AUTOMATIC}
        if any(value["dtype"] in ("image", "video") for value in features.values()):
            raise ValueError("Dataset already has image or video features")
        table, fps, robot_type = original.hf_dataset.data.table, original.fps, original.meta.robot_type
        tasks = dict(zip(original.meta.tasks["task_index"], original.meta.tasks.index))
    finally:
        del original
        gc.collect()  # Release Arrow memory maps (Windows).
    where = _columns(features)
    columns = {key: table.column(key).to_pylist() for key in [*features, "episode_index", "task_index"]}
    images = {f"observation.images.{name}": {"dtype": "image", "shape": (renderer.size, renderer.size, 3),
                                             "names": ["height", "width", "channels"]} for name in CAMERAS}
    dataset = LeRobotDataset.create(repo_id=repo_id, root=staging, robot_type=robot_type, fps=fps,
                                    features={**features, **images}, use_videos=False, video_backend="pyav")
    digests, errors = [], dict.fromkeys(CHECKS, 0.)
    try:
        for row in range(table.num_rows):
            frame = {}
            for key, feature in features.items():
                value = columns[key][row]
                frame[key] = value if feature["dtype"] == "string" else \
                    np.asarray(value, dtype=feature["dtype"]).reshape(feature["shape"])
            state = {part: np.asarray([frame[key][i] for key, i in entries], dtype=np.float64)
                     for part, entries in where.items()}
            rendered, achieved = renderer.render(state["joints"], state["block"])
            for name in CHECKS:
                errors[name] = max(errors[name], float(np.max(np.abs(achieved[name] - state[name]))))
            digests.append({key: hashlib.sha256(image.tobytes()).hexdigest() for key, image in rendered.items()})
            dataset.add_frame({**frame, **rendered, "task": tasks[columns["task_index"][row]]})
            if row + 1 == table.num_rows or columns["episode_index"][row + 1] != columns["episode_index"][row]:
                dataset.save_episode(parallel_encoding=False)
    finally:
        dataset.finalize()
    del dataset
    gc.collect()
    if max(errors.values()) > TOLERANCE_M:
        raise ValueError(f"Rebuilt scene disagrees with the recorded grasp/tray positions: {errors}")
    verification = _verify(table, digests, staging, repo_id)
    for path in source.iterdir():  # Provenance sidecars (xert_recording.json, journal, roundtrip, export manifest).
        if path.is_file():
            shutil.copyfile(path, staging / path.name)
    manifest = {
        "version": 1, "backend": "mujoco", "hardware": False,
        "images": "offline MuJoCo renders of the simulated bench, not real camera data",
        "reconstruction": "qpos = recorded joint positions + block pose (float32), then mj_forward; tray is static",
        "reconstruction_max_error_m": errors, "tolerance_m": TOLERANCE_M,
        "cameras": {name: {**camera, "feature": f"observation.images.{name}", "height": renderer.size,
                           "width": renderer.size,
                           "xyaxes": _xyaxes(camera["pos"], camera["target"], camera["up"]).tolist()}
                    for name, camera in CAMERAS.items()},
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


def _verify(source_table, digests, root, repo_id):
    """Every source column byte-identical; every image decodes through LeRobotDataset to the rendered pixels."""
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(repo_id=repo_id, root=root, video_backend="pyav", download_videos=False)
    try:
        table = dataset.hf_dataset.data.table
        for name in source_table.column_names:
            if not table.column(name).equals(source_table.column(name)):
                raise ValueError(f"Rendered dataset changed source column {name}")
        if len(dataset) != len(digests):
            raise ValueError("Rendered dataset frame count differs from source")
        for row, expected in enumerate(digests):
            frame = dataset[row]
            for key, digest in expected.items():
                pixels = (frame[key] * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
                if hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest() != digest:
                    raise ValueError(f"Stored image differs from the render: {key} frame {row}")
        return {"frames": len(dataset), "episodes": dataset.num_episodes, "source_columns_identical": True,
                "images_lossless": True}
    finally:
        del dataset
        gc.collect()


def render_cameras(source, output, size=SIZE, repo_id="xert/runtime"):
    """A recording/export, or a P4.1 session of episode_* recordings, into a fresh output. Never overwrites."""
    source, output = Path(source).absolute(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    recordings = [source] if (source / "meta/info.json").exists() else sorted(source.glob("episode_*"))
    if not recordings:
        raise ValueError("Source is neither a LeRobot dataset nor a session of episode_* datasets")
    output.parent.mkdir(parents=True, exist_ok=True)
    renderer = CameraRenderer(size)
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

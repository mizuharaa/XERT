"""Stationary upstream XLeRobot arm benchmark. All observations are simulator state."""
from collections import deque
from copy import deepcopy
import hashlib
import json
from numbers import Real
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "xlerobot"
SO101_ASSETS = ASSETS.parent / "menagerie" / "robotstudio_so101"
JOINTS = ["Rotation_L", "Pitch_L", "Elbow_L", "Wrist_Pitch_L", "Wrist_Roll_L", "Jaw_L"]
CONTROL_HZ = 25
# Bench arms: actuator i drives joints[i]. Physics substeps per control step are derived in the
# native core as 1/(CONTROL_HZ * timestep) and refused unless that is an exact integer.
ROBOTS = {
    "xlerobot_left": {"joints": JOINTS, "grasp_site": "grasp", "reset_pose": [0, 1.0, 1.2, -.6, 0, 1.0],
                      "scripted_expert": True},
    # LeRobot so101_follower motor names. Reset: arm raised, grasp ~7 cm above the block area, gripper at
    # the spec's open value (docs/handoff/B-so101-bench.md). The expert is TODO(daniel): so101_expert.py.
    "so101": {"joints": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
              "grasp_site": "grasp", "reset_pose": [0, -1.5, 1.0, 1.4, 0, 1.4], "scripted_expert": False},
}
# SO-101 base on the table top, turned to face the block area and tray. Measured with the generalized
# IK in scripts/so101_bench_reach.py (results in the scene provenance below and the handoff).
SO101_BASE = {"pos": [.42, .04, .08], "quat": [0, 0, 0, 1]}  # yaw = pi about world z
SO101_TIMESTEP = .002  # Menagerie ships 0.005 s; 0.002 s matches the XLeRobot bench integration step.


def verified_source(asset_dir):
    """Reject changed downloaded assets before compiling collision geometry."""
    source = json.loads((asset_dir / "SOURCE.json").read_text())
    hashes = source["sha256"]
    if not {"LICENSE", "xlerobot.xml"}.issubset(hashes):
        raise ValueError("Asset manifest must include LICENSE and xlerobot.xml")
    for name, expected in hashes.items():
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Asset manifest names must be local filenames")
        if hashlib.sha256((asset_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Pinned model asset changed: {name}")
    return source


def model_fingerprint(xml, source):
    """Moving a checkout preserves identity; changed XML or mesh bytes do not."""
    tree = ET.fromstring(xml)
    tree.find("compiler").set("meshdir", "assets/xlerobot")
    payload = ET.tostring(tree) + json.dumps(source, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def model_xml():
    source = ET.parse(ASSETS / "xlerobot.xml").getroot()
    root = ET.Element("mujoco", model="xlerobot_single_arm_bench")
    ET.SubElement(root, "compiler", angle="radian", meshdir=str(ASSETS))
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast", cone="elliptic")
    ET.SubElement(root, "visual").append(ET.Element("global", offwidth="640", offheight="480"))
    root.append(deepcopy(source.find("default")))
    arm = deepcopy(source.find(".//body[@name='Rotation_Pitch']"))
    arm.set("pos", "0 0 0.20")
    for j in arm.iter("joint"):
        j.set("damping", "0.2")
        j.set("armature", "0.01")
    meshes = {g.get("mesh") for g in arm.iter("geom")}
    asset = ET.SubElement(root, "asset")
    for mesh in source.find("asset"):
        if mesh.tag != "mesh" or mesh.get("name") in meshes:
            asset.append(deepcopy(mesh))
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -0.3 1", dir="0 0 -1")
    ET.SubElement(world, "geom", name="floor", type="plane", size="1 1 .02", rgba=".17 .2 .24 1")
    ET.SubElement(world, "geom", name="mount", type="box", pos="0 0 .10", size=".04 .04 .10", rgba=".3 .32 .35 1")
    world.append(arm)
    jaw = arm.find(".//body[@name='Fixed_Jaw']")
    ET.SubElement(jaw, "site", name="grasp", pos="-.003 -.10 0", size=".003", rgba="1 0 0 1")
    ET.SubElement(world, "geom", name="table", type="box", pos=".35 0 .04", size=".23 .4 .04", rgba=".45 .48 .52 1")
    tray = ET.SubElement(world, "body", name="tray", pos=".23 .11 .08")
    ET.SubElement(tray, "geom", name="tray_floor", type="box", size=".045 .045 .004", pos="0 0 .004", rgba=".2 .6 .5 1")
    for i, (pos, size) in enumerate([("-.049 0 .014", ".004 .053 .014"), (".049 0 .014", ".004 .053 .014"), ("0 -.049 .014", ".045 .004 .014"), ("0 .049 .014", ".045 .004 .014")]):
        ET.SubElement(tray, "geom", name=f"tray_wall_{i}", type="box", pos=pos, size=size, rgba=".2 .6 .5 1")
    block = ET.SubElement(world, "body", name="block", pos="-.03 .25 .096")
    ET.SubElement(block, "freejoint", name="block_joint")
    ET.SubElement(block, "geom", name="block_geom", type="box", size=".008 .008 .016", mass=".015", rgba="1 .55 .12 1", condim="4", friction="1 .005 .0001")
    actuator = ET.SubElement(root, "actuator")
    for name in JOINTS:
        joint = arm.find(f".//joint[@name='{name}']")
        limits = joint.get("range") if name != "Jaw_L" else "-.2 1.4"
        ET.SubElement(actuator, "position", name=name, joint=name, kp="5" if name == "Jaw_L" else "60",
                      kv=".1" if name == "Jaw_L" else "2", ctrlrange=limits, forcerange="-.3 .3" if name == "Jaw_L" else "-3 3")
    return ET.tostring(root, encoding="unicode")


def so101_scene():
    """(compiled model, composed XML, source provenance): the vendored SO-101 plus the XLeRobot bench's own
    light, floor, table, tray and block, composed in memory with MjSpec. No vendored file is written."""
    source = json.loads((SO101_ASSETS / "SOURCE.json").read_text())
    for name, expected in source["sha256"].items():
        if hashlib.sha256((SO101_ASSETS / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Pinned model asset changed: {name}")
    robot_xml = SO101_ASSETS / "so101_xert.xml"  # Derived by scripts/prepare_so101.py; hashed below.
    props = ET.Element("mujoco")
    world = ET.SubElement(props, "worldbody")
    for element in ET.fromstring(model_xml()).find("worldbody"):
        if element.get("name") not in ("mount", "Rotation_Pitch"):  # Drop the XLeRobot arm and its mount.
            world.append(element)
    spec = mujoco.MjSpec.from_file(str(robot_xml))
    menagerie_timestep = spec.option.timestep
    spec.option.timestep = SO101_TIMESTEP
    spec.body("base").pos, spec.body("base").quat = SO101_BASE["pos"], SO101_BASE["quat"]
    clamped = {}
    for actuator in spec.actuators:
        # The bench normalizes actions by ctrlrange, so targets stay inside the joint range (the so101.yaml limits).
        joint = spec.joint(actuator.target)
        low, high = max(actuator.ctrlrange[0], joint.range[0]), min(actuator.ctrlrange[1], joint.range[1])
        if [low, high] != list(actuator.ctrlrange):
            clamped[actuator.name] = {"menagerie": actuator.ctrlrange.tolist(), "bench": [float(low), float(high)]}
            actuator.ctrlrange = [low, high]
    spec.attach(mujoco.MjSpec.from_string(ET.tostring(props, encoding="unicode")), frame=spec.worldbody.add_frame(),
                prefix="")
    model = spec.compile()
    provenance = {**source, "robot_xml": robot_xml.name,
                  "robot_xml_sha256": hashlib.sha256(robot_xml.read_bytes()).hexdigest(),
                  "composition": {"props": "light, floor, table, tray and block from the XLeRobot bench model_xml()",
                                  "base": SO101_BASE, "timestep": {"menagerie": menagerie_timestep, "bench": SO101_TIMESTEP},
                                  "ctrlrange_clamped_to_joint_range": clamped,
                                  "reach_evidence": "scripts/so101_bench_reach.py; docs/handoff/B-so101-bench.md"}}
    return model, spec.to_xml(), provenance


class ManipulationEnv:
    def __init__(self, config=None):
        config = dict(config or {})
        allowed = {"robot", "friction", "delay_steps", "horizon", "object_position", "object_position_bounds"}
        if set(config) - allowed:
            raise ValueError(f"Unknown config: {sorted(set(config) - allowed)}")
        self.config = {"robot": "xlerobot_left", "friction": 1.0, "delay_steps": 0, "horizon": 600,
                       "object_position_bounds": [[.225, -.045], [.255, -.015]], **config}
        for key, lo, hi in [("friction", .1, 2), ("delay_steps", 0, 20), ("horizon", 1, 2000)]:
            val = self.config[key]
            if isinstance(val, bool) or not isinstance(val, Real) or not np.isfinite(val) or not lo <= val <= hi:
                raise ValueError(f"{key} must be finite in [{lo}, {hi}]")
            if key != "friction" and int(val) != val:
                raise ValueError(f"{key} must be an integer")
            self.config[key] = float(val) if key == "friction" else int(val)
        bounds = np.asarray(self.config["object_position_bounds"], dtype=float)
        if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[0] > bounds[1]) or np.any(np.abs(bounds) > .4):
            raise ValueError("object_position_bounds must be finite ordered [low_xy, high_xy] inside +/-0.4 m")
        self.config["object_position_bounds"] = bounds.tolist()
        if "object_position" in self.config:
            pos = np.asarray(self.config["object_position"], dtype=float)
            if pos.shape != (2,) or not np.isfinite(pos).all() or np.any(np.abs(pos) > .4):
                raise ValueError("object_position must be finite [x,y] inside +/-0.4 m")
            self.config["object_position"] = pos.tolist()
        if self.config["robot"] not in ROBOTS:
            raise ValueError(f"robot must be one of {sorted(ROBOTS)}")
        self.robot = ROBOTS[self.config["robot"]]
        self.joints = self.robot["joints"]
        if self.config["robot"] == "xlerobot_left":
            self.source = verified_source(ASSETS)
            xml = model_xml()
            mesh_files = {mesh.get("file") for mesh in ET.fromstring(xml).findall("asset/mesh")}
            if not mesh_files.issubset(self.source["sha256"]):
                raise ValueError("Asset manifest does not cover every model mesh")
            self.model = mujoco.MjModel.from_xml_string(xml)
            self.xml_sha256 = model_fingerprint(xml, self.source)
        else:
            self.model, xml, self.source = so101_scene()
            self.xml_sha256 = hashlib.sha256(xml.encode() + json.dumps(self.source, sort_keys=True).encode()).hexdigest()
        self.model.geom_friction[:, 0] = self.config["friction"]
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.qids = np.array([self.model.joint(n).qposadr[0] for n in self.joints])
        self.dids = np.array([self.model.joint(n).dofadr[0] for n in self.joints])
        self.limits = self.model.actuator_ctrlrange.copy()
        self.site = self.model.site(self.robot["grasp_site"]).id
        self.block = self.model.body("block").id
        self.block_q = self.model.joint("block_joint").qposadr[0]
        self.block_d = self.model.joint("block_joint").dofadr[0]
        root = Path(__file__).resolve().parents[1]
        sources = [Path(__file__).resolve(), *sorted((root / "cpp/include/reach").glob("*.hpp")),
                   *(root / "cpp/src" / name for name in ("env.cpp", "ik.cpp", "bindings.cpp")),
                   Path(__file__).resolve().with_name("so101_expert.py")]
        digest = hashlib.sha256()
        for path in sorted(sources):
            digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
        self.implementation_sha256 = digest.hexdigest()
        from ._reach_core import Env
        self._core = Env(self.model, self.data, self.ik_data, self.config["horizon"], self.joints,
                         self.robot["grasp_site"], self.robot["reset_pose"], CONTROL_HZ, self.robot["scripted_expert"])
        self.reset()

    def metadata(self):
        names = ([f"joint_position.{n}" for n in self.joints] + [f"joint_velocity.{n}" for n in self.joints]
                 + ["block.x", "block.y", "block.z", "block.qw", "block.qx", "block.qy", "block.qz"]
                 + ["grasp.x", "grasp.y", "grasp.z", "tray.x", "tray.y", "tray.z", "time_fraction"])
        return {"observation_dim": len(names), "action_dim": 6, "observation_names": names,
                "action_names": self.joints, "control_hz": CONTROL_HZ, "horizon": self.config["horizon"],
                "action_convention": "absolute joint position targets normalized to [-1,1] by actuator_ctrlrange",
                "action_joint_ranges_radians": self.limits.tolist(), "observation_convention": "privileged simulator state; world XYZ metres, wxyz quaternion, joint radians, rad/s",
                "model_sha256": self.xml_sha256, "source": deepcopy(self.source),
                "implementation_sha256": self.implementation_sha256,
                "config": deepcopy(self.config), "simulation_only": True, "mujoco_version": mujoco.__version__}

    def _obs(self):
        return self._core.observation()

    def _sync_state(self):
        self.steps = self._core.steps
        self.stable_steps = self._core.stable_steps
        self.lifted = self._core.lifted
        self.finished = self._core.finished
        self.initial_block = self._core.initial_block
        self.pending = deque(self._core.pending)

    def normalize(self, target):
        return 2 * (np.asarray(target) - self.limits[:, 0]) / np.diff(self.limits, axis=1)[:, 0] - 1

    def reset(self, seed=0):
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        rng = np.random.default_rng(seed)
        xy = self.config.get("object_position", rng.uniform(*np.asarray(self.config["object_position_bounds"])))
        observation = self._core.reset_state(xy, self.config["delay_steps"])
        self._sync_state()
        return observation

    def step(self, action):
        result = self._core.step(action)
        self._sync_state()
        return result

    def expert_action(self):
        if not self.robot["scripted_expert"]:
            from .so101_expert import so101_expert_action
            return so101_expert_action(self)
        return self._core.expert_action()

    def render(self):
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.18, 0, .17]
        camera.distance = .8
        camera.azimuth = 130
        camera.elevation = -25
        # Context creation/destruction stay on the caller thread. GLFW requires the
        # main thread; HTTP callers render directly on the event-loop/main thread.
        with mujoco.Renderer(self.model, height=480, width=640) as renderer:
            renderer.update_scene(self.data, camera=camera)
            return renderer.render().copy()

    def close(self):
        pass  # Rendering contexts are closed by render(); physics owns no external handle.

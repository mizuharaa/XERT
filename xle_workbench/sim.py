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
JOINTS = ["Rotation_L", "Pitch_L", "Elbow_L", "Wrist_Pitch_L", "Wrist_Roll_L", "Jaw_L"]


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


class ManipulationEnv:
    def __init__(self, config=None):
        config = dict(config or {})
        allowed = {"friction", "delay_steps", "horizon", "object_position", "object_position_bounds"}
        if set(config) - allowed:
            raise ValueError(f"Unknown config: {sorted(set(config) - allowed)}")
        self.config = {"friction": 1.0, "delay_steps": 0, "horizon": 600,
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
        self.source = verified_source(ASSETS)
        xml = model_xml()
        mesh_files = {mesh.get("file") for mesh in ET.fromstring(xml).findall("asset/mesh")}
        if not mesh_files.issubset(self.source["sha256"]):
            raise ValueError("Asset manifest does not cover every model mesh")
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.model.geom_friction[:, 0] = self.config["friction"]
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.qids = np.array([self.model.joint(n).qposadr[0] for n in JOINTS])
        self.dids = np.array([self.model.joint(n).dofadr[0] for n in JOINTS])
        self.limits = self.model.actuator_ctrlrange.copy()
        self.site = self.model.site("grasp").id
        self.block = self.model.body("block").id
        self.block_q = self.model.joint("block_joint").qposadr[0]
        self.block_d = self.model.joint("block_joint").dofadr[0]
        self.xml_sha256 = model_fingerprint(xml, self.source)
        self.implementation_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.reset()

    def metadata(self):
        names = ([f"joint_position.{n}" for n in JOINTS] + [f"joint_velocity.{n}" for n in JOINTS]
                 + ["block.x", "block.y", "block.z", "block.qw", "block.qx", "block.qy", "block.qz"]
                 + ["grasp.x", "grasp.y", "grasp.z", "tray.x", "tray.y", "tray.z", "time_fraction"])
        return {"observation_dim": len(names), "action_dim": 6, "observation_names": names,
                "action_names": JOINTS, "control_hz": 25, "horizon": self.config["horizon"],
                "action_convention": "absolute joint position targets normalized to [-1,1] by actuator_ctrlrange",
                "action_joint_ranges_radians": self.limits.tolist(), "observation_convention": "privileged simulator state; world XYZ metres, wxyz quaternion, joint radians, rad/s",
                "model_sha256": self.xml_sha256, "source": deepcopy(self.source),
                "implementation_sha256": self.implementation_sha256,
                "config": deepcopy(self.config), "simulation_only": True, "mujoco_version": mujoco.__version__}

    def _obs(self):
        return np.concatenate([self.data.qpos[self.qids], self.data.qvel[self.dids],
            self.data.qpos[self.block_q:self.block_q+7], self.data.site_xpos[self.site],
            self.model.body("tray").pos, [self.steps / self.config["horizon"]]]).astype(np.float32)

    def normalize(self, target):
        return 2 * (np.asarray(target) - self.limits[:, 0]) / np.diff(self.limits, axis=1)[:, 0] - 1

    def reset(self, seed=0):
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        mujoco.mj_resetData(self.model, self.data)
        rng = np.random.default_rng(seed)
        xy = self.config.get("object_position", rng.uniform(*np.asarray(self.config["object_position_bounds"])))
        self.data.qpos[self.block_q:self.block_q+3] = [*xy, .096]
        self.data.qpos[self.qids] = [0, 1.0, 1.2, -.6, 0, 1.0]
        self.data.ctrl[:] = self.data.qpos[self.qids]
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.stable_steps = 0
        self.lifted = False
        self.finished = False
        self.initial_block = self.data.xpos[self.block].copy()
        self.pending = deque([self.data.ctrl.copy() for _ in range(self.config["delay_steps"])])
        return self._obs()

    def step(self, action):
        # The recorder stores float32; quantize here so replay drives identical contacts.
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (6,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1 + 1e-6):
            raise ValueError("action must contain six finite normalized joint targets in [-1,1]")
        if self.finished:
            raise RuntimeError("episode finished; reset before stepping")
        self.pending.append(self.limits[:, 0] + (action.clip(-1, 1) + 1) * .5 * np.diff(self.limits, axis=1)[:, 0])
        self.data.ctrl[:] = self.pending.popleft()
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("Nonfinite simulation state")
        # mj_step integrates qpos after computing transforms; refresh before observation/scoring.
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        pos = self.data.xpos[self.block]
        self.lifted |= bool(pos[2] > .145)
        tray = self.model.body("tray").pos
        held = False
        for con in self.data.contact[:self.data.ncon]:
            b1, b2 = self.model.geom_bodyid[con.geom1], self.model.geom_bodyid[con.geom2]
            if self.block in (b1, b2) and (b2 if b1 == self.block else b1) not in (0, self.model.body("tray").id):
                held = True
        settled = bool(self.lifted and np.all(np.abs(pos[:2]-tray[:2]) < .029) and .095 < pos[2] < .115
                       and np.linalg.norm(self.data.qvel[self.block_d:self.block_d+3]) < .025 and not held)
        self.stable_steps = self.stable_steps + 1 if settled else 0
        success = self.stable_steps >= 12
        failure = "fell_off_table" if pos[2] < .04 else None
        self.finished = bool(success or failure or self.steps >= self.config["horizon"])
        if self.finished and not success and not failure:
            failure = "timeout_after_lift" if self.lifted else "timeout_no_lift"
        return self._obs(), float(success), self.finished, {"success": bool(success), "failure_reason": failure,
            "lifted": self.lifted, "steps": self.steps, "object_position": pos.tolist(), "object_in_robot_contact": held}

    def expert_action(self):
        phase = self.steps
        start = self.initial_block.copy()
        tray = self.model.body("tray").pos.copy()
        if phase < 65:
            target, grip = start + [0, 0, .06], 1.0
        elif phase < 140:
            target, grip = start + [0, 0, .06 * (1-(phase-65)/75)], 1.0
        elif phase < 185:
            target, grip = start, -.1
        elif phase < 255:
            target, grip = start + [0, 0, .075 * ((phase-185)/70)], -.1
        elif phase < 340:
            alpha = (phase-255)/85
            target, grip = (1-alpha)*(start + [0, 0, .075]) + alpha*(tray + [0, 0, .10]), -.1
        elif phase < 410:
            target, grip = tray + [0, 0, .10 - .06*(phase-340)/70], -.1
        elif phase < 465:
            target, grip = tray + [0, 0, .04], 1.0
        else:
            target, grip = tray + [0, 0, .10], 1.0
        self.ik_data.qpos[:] = self.data.qpos
        q = self.ik_data.qpos
        q[self.qids[4]] = 0
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for _ in range(35):
            # IK needs transforms and Jacobians, not contact/dynamics solves on hypothetical poses.
            mujoco.mj_kinematics(self.model, self.ik_data)
            mujoco.mj_comPos(self.model, self.ik_data)
            mujoco.mj_jacSite(self.model, self.ik_data, jp, jr, self.site)
            axis = self.ik_data.site_xmat[self.site].reshape(3, 3)[:, 1]
            error = np.r_[target-self.ik_data.site_xpos[self.site], .1*np.cross(axis, [0, 0, 1])]
            jac = np.vstack([jp[:, self.dids[:4]], .1*(np.eye(3)-np.outer(axis, axis)) @ jr[:, self.dids[:4]]])
            delta = jac.T @ np.linalg.solve(jac @ jac.T + .0001*np.eye(6), error)
            q[self.qids[:4]] = np.clip(q[self.qids[:4]] + delta.clip(-.15, .15), self.limits[:4, 0], self.limits[:4, 1])
        targets = q[self.qids].copy()
        targets[-1] = grip
        # Rate limit target changes; this changes servo commands, never the block's pose.
        targets[:5] = self.data.ctrl[:5] + np.clip(targets[:5]-self.data.ctrl[:5], -.05, .05)
        return self.normalize(targets).clip(-1, 1)

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

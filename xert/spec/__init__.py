"""Embodiment specifications in the JSON-compatible subset of YAML."""
from dataclasses import dataclass
import json
from pathlib import Path

import mujoco
import numpy as np


@dataclass(frozen=True)
class EmbodimentSpec:
    name: str
    model_path: Path
    joint_names: tuple[str, ...]
    position_limits: np.ndarray
    velocity_limits: np.ndarray
    acceleration_limits: np.ndarray
    rest: np.ndarray
    sites: tuple[str, ...]
    mapping: dict[str, str]
    ik_weights: dict[str, float]
    control_hz: float
    gripper: dict

    def load_model(self):
        return mujoco.MjModel.from_xml_path(str(self.model_path))


def load_spec(path: str | Path, *, contents: bytes | None = None) -> EmbodimentSpec:
    """Load JSON-form YAML; reject ambiguous YAML syntax rather than guessing."""
    path = Path(path).resolve()
    try:
        raw = json.loads((path.read_bytes() if contents is None else contents).decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Specs use the JSON-compatible YAML subset (quoted keys, no tags)") from exc
    fields = {"name", "model_path", "joint_names", "position_limits", "velocity_limits",
              "acceleration_limits", "rest", "sites", "mapping", "ik_weights", "control_hz", "gripper"}
    if not isinstance(raw, dict) or set(raw) != fields:
        raise ValueError("Spec fields must exactly match the documented schema")
    if not isinstance(raw["name"], str) or not raw["name"]:
        raise ValueError("name must be nonempty")
    names = raw["joint_names"]
    sites = raw["sites"]
    for label, values in (("joint_names", names), ("sites", sites)):
        if (not isinstance(values, list) or not values or
                any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values)):
            raise ValueError(f"{label} must contain unique nonempty names")
    arrays = {}
    for key in ("position_limits", "velocity_limits", "acceleration_limits", "rest"):
        def numeric(value):
            return (all(numeric(v) for v in value) if isinstance(value, list)
                    else type(value) in (int, float))
        if not numeric(raw[key]):
            raise ValueError(f"{key} must contain numbers, not strings or booleans")
        values = np.asarray(raw[key], dtype=float)
        shape = (len(names), 2) if key == "position_limits" else (len(names),)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"{key} must be finite with shape {shape}")
        values.setflags(write=False)
        arrays[key] = values
    limits = arrays["position_limits"]
    if (np.any(limits[:, 0] >= limits[:, 1]) or np.any(arrays["velocity_limits"] <= 0) or
            np.any(arrays["acceleration_limits"] <= 0) or
            np.any(arrays["rest"] < limits[:, 0]) or np.any(arrays["rest"] > limits[:, 1])):
        raise ValueError("Limits must be ordered and positive; rest must lie inside position limits")
    hz = raw["control_hz"]
    if isinstance(hz, bool) or not isinstance(hz, (float, int)) or not np.isfinite(hz) or hz <= 0:
        raise ValueError("control_hz must be finite and positive")
    mapping, weights, gripper = raw["mapping"], raw["ik_weights"], raw["gripper"]
    if (not isinstance(mapping, dict) or not mapping or
            any(not isinstance(k, str) or not k or v not in sites for k, v in mapping.items())):
        raise ValueError("mapping must map source keypoint names to declared sites")
    if (not isinstance(weights, dict) or set(weights) != set(sites) or
            any(isinstance(v, bool) or not isinstance(v, (float, int)) or not np.isfinite(v) or v <= 0
                for v in weights.values())):
        raise ValueError("Each site needs a finite positive IK weight")
    if not isinstance(gripper, dict):
        raise ValueError("gripper must be a mapping")
    if gripper:
        if set(gripper) != {"joint", "open", "closed"} or gripper["joint"] not in names:
            raise ValueError("gripper needs a declared joint, open and closed positions")
        lo, hi = limits[names.index(gripper["joint"])]
        if any(type(gripper[k]) not in (int, float) or not lo <= gripper[k] <= hi
               for k in ("open", "closed")):
            raise ValueError("Gripper positions must lie inside limits")
    if not isinstance(raw["model_path"], str) or not raw["model_path"]:
        raise ValueError("model_path must be nonempty")
    spec = EmbodimentSpec(raw["name"], (path.parent / raw["model_path"]).resolve(), tuple(names),
                          **arrays, sites=tuple(sites), mapping=mapping, ik_weights=weights,
                          control_hz=float(hz), gripper=gripper)
    model = spec.load_model()
    for i, name in enumerate(names):
        try:
            joint = model.joint(name)
        except KeyError as exc:
            raise ValueError(f"Unknown joint: {name}") from exc
        if int(joint.type[0]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            raise ValueError(f"Joint must have one coordinate: {name}")
        if joint.limited[0] and (limits[i, 0] < joint.range[0] - 1e-8 or limits[i, 1] > joint.range[1] + 1e-8):
            raise ValueError(f"Spec exceeds MJCF joint limits: {name}")
        for a in range(model.nu):
            # A unit-gear position actuator's control is a joint position; torque limits are not.
            position = (int(model.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT) and
                        model.actuator_trnid[a, 0] == joint.id and model.actuator_gear[a, 0] == 1 and
                        int(model.actuator_gaintype[a]) == int(mujoco.mjtGain.mjGAIN_FIXED) and
                        int(model.actuator_biastype[a]) == int(mujoco.mjtBias.mjBIAS_AFFINE) and
                        model.actuator_gainprm[a, 0] > 0 and
                        model.actuator_biasprm[a, 1] == -model.actuator_gainprm[a, 0])
            if position and model.actuator_ctrllimited[a] and (
                    limits[i, 0] < model.actuator_ctrlrange[a, 0] - 1e-8 or
                    limits[i, 1] > model.actuator_ctrlrange[a, 1] + 1e-8):
                raise ValueError(f"Spec exceeds position actuator target limits: {name}")
    for name in sites:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) < 0:
            raise ValueError(f"Unknown site: {name}")
    return spec

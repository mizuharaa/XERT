"""Neutral-relative camera controls, deliberately not metric human motion capture."""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .ik import solve


def hand_features(hand):
    world, image = hand.world, hand.image
    forward = world[9] - world[0]
    across = world[5] - world[17]
    forward /= max(np.linalg.norm(forward), 1e-15)
    across -= forward * np.dot(forward, across)
    length = np.linalg.norm(across)
    palm = np.linalg.norm(image[5, :2] - image[17, :2])
    width = np.linalg.norm(world[5] - world[17])
    if length < 1e-6 or palm < 1e-6 or width < 1e-6:
        raise ValueError("Degenerate palm geometry")
    across /= length
    basis = np.column_stack((across, forward, np.cross(across, forward)))
    curls = {}
    for name, ids in {"thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8],
                      "middle": [9, 10, 11, 12], "ring": [13, 14, 15, 16]}.items():
        points = world[[0] + ids]
        bones = np.diff(points, axis=0)
        lengths = np.linalg.norm(bones, axis=1)
        if np.any(lengths < 1e-6):
            raise ValueError("Degenerate finger geometry")
        bones /= lengths[:, None]
        curls[name] = np.arccos(np.clip(np.sum(bones[:-1] * bones[1:], axis=1), -1, 1))
    return {"xy": image[0, :2], "palm": palm, "basis": basis, "curls": curls,
            "pinch": np.linalg.norm(world[4] - world[8]) / width}


def load_profile(path, spec):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("mode") not in ("pose", "curls"):
        raise ValueError("Teleop profile mode must be pose or curls")
    if raw["mode"] == "pose":
        if set(raw) != {"mode", "hands", "axes", "gains", "orientation", "pinch_hand", "pinch_range"}:
            raise ValueError("Unexpected pose profile fields")
        if set(raw["hands"]) != set(spec.mapping) or any(h not in ("left", "right") for h in raw["hands"].values()):
            raise ValueError("Pose hands must map every spec source to left/right")
        axes = np.asarray(raw["axes"], dtype=float)
        gains = np.asarray(raw["gains"], dtype=float)
        if (axes.shape != (3, 3) or not np.isfinite(axes).all() or
                not np.allclose(axes.T @ axes, np.eye(3)) or not np.isclose(np.linalg.det(axes), 1) or
                gains.shape != (3,) or not np.isfinite(gains).all() or np.any(gains <= 0)):
            raise ValueError("Axes must be a proper rotation; gains must be finite positive XYZ scales")
        if type(raw["orientation"]) is not bool:
            raise ValueError("orientation must be boolean")
        pinch = raw["pinch_hand"]
        if pinch is not None and (not spec.gripper or pinch not in raw["hands"].values()):
            raise ValueError("Pinch hand needs a gripper and a mapped hand")
        limits = np.asarray(raw["pinch_range"], dtype=float)
        if limits.shape != (2,) or not np.isfinite(limits).all() or not 0 <= limits[0] < limits[1]:
            raise ValueError("Pinch range must be finite ordered nonnegative bounds")
    else:
        if set(raw) != {"mode", "hand", "joints"} or raw["hand"] not in ("left", "right"):
            raise ValueError("Curl profile needs a hand and joint mapping")
        if not isinstance(raw["joints"], dict) or not raw["joints"] or set(raw["joints"]) - set(spec.joint_names):
            raise ValueError("Curl mapping must name spec joints")
        for config in raw["joints"].values():
            if (not isinstance(config, dict) or set(config) != {"finger", "segment", "gain"} or
                    config["finger"] not in ("thumb", "index", "middle", "ring") or
                    type(config["segment"]) is not int or config["segment"] not in range(3) or
                    type(config["gain"]) not in (int, float) or not np.isfinite(config["gain"])):
                raise ValueError("Invalid curl joint mapping")
    return raw


def required_hands(profile):
    return set(profile["hands"].values()) if profile["mode"] == "pose" else {profile["hand"]}


def create_calibration(spec_path, profile_path, frame, backend):
    profile = load_profile(profile_path, backend.spec)
    if not required_hands(profile) <= set(frame.hands):
        raise ValueError("Neutral calibration requires every mapped hand")
    features = {name: hand_features(frame.hands[name]) for name in required_hands(profile)}
    return {"version": 1, "source": frame.source, "frame_sequence": frame.sequence,
            "t_capture": frame.t_capture, "image_size": list(frame.image_size),
            "spec_sha256": hashlib.sha256(Path(spec_path).read_bytes()).hexdigest(),
            "profile_sha256": hashlib.sha256(Path(profile_path).read_bytes()).hexdigest(),
            "hands": {name: {"xy": f["xy"].tolist(), "palm": float(f["palm"]),
                              "basis": f["basis"].tolist(),
                              "curls": {k: v.tolist() for k, v in f["curls"].items()}}
                      for name, f in features.items()},
            "sites": {name: {"position": backend.data.site(name).xpos.tolist(),
                              "rotation": backend.data.site(name).xmat.reshape(3, 3).tolist()}
                      for name in backend.spec.sites},
            "notes": "Image XY and palm-size depth proxy are control coordinates, not metric wrist tracking."}


class CalibratedMapper:
    def __init__(self, spec_path, profile_path, calibration, backend):
        self.profile = load_profile(profile_path, backend.spec)
        self.spec, self.calibration, self.backend = backend.spec, calibration, backend
        for key, path in (("spec_sha256", spec_path), ("profile_sha256", profile_path)):
            if calibration.get(key) != hashlib.sha256(Path(path).read_bytes()).hexdigest():
                raise ValueError("Calibration spec/profile checksum mismatch")
        if calibration.get("version") != 1 or not required_hands(self.profile) <= set(calibration.get("hands", {})):
            raise ValueError("Calibration is missing required hands/version")
        # Validate persisted geometry by running the same mapping checks before live use.
        for value in calibration["hands"].values():
            if (np.asarray(value["xy"]).shape != (2,) or not np.isfinite(value["xy"]).all() or
                    not np.isfinite(value["palm"]) or value["palm"] <= 0):
                raise ValueError("Invalid neutral hand geometry")
            self._rotation(value["basis"])
            for finger in ("thumb", "index", "middle", "ring"):
                curls = np.asarray(value["curls"][finger])
                if curls.shape != (3,) or not np.isfinite(curls).all():
                    raise ValueError("Invalid neutral curls")
        if set(calibration.get("sites", {})) != set(self.spec.sites):
            raise ValueError("Calibration site set differs from spec")
        for site in calibration["sites"].values():
            if np.asarray(site["position"]).shape != (3,) or not np.isfinite(site["position"]).all():
                raise ValueError("Invalid neutral site position")
            self._rotation(site["rotation"])

    @staticmethod
    def _rotation(value):
        array = np.asarray(value, dtype=float)
        if (array.shape != (3, 3) or not np.isfinite(array).all() or
                not np.allclose(array.T @ array, np.eye(3), atol=1e-6) or
                not np.isclose(np.linalg.det(array), 1)):
            raise ValueError("Invalid calibration rotation")
        return array

    def map(self, frame):
        if tuple(self.calibration["image_size"]) != frame.image_size:
            raise ValueError("Camera image dimensions changed; recalibrate")
        if not required_hands(self.profile) <= set(frame.hands):
            return None
        try:
            features = {name: hand_features(frame.hands[name]) for name in required_hands(self.profile)}
        except ValueError:
            return None
        profile, neutral = self.profile, self.calibration
        if profile["mode"] == "curls":
            desired = self.spec.rest.copy()
            hand = profile["hand"]
            for joint, config in profile["joints"].items():
                finger, segment = config["finger"], config["segment"]
                delta = features[hand]["curls"][finger][segment] - neutral["hands"][hand]["curls"][finger][segment]
                desired[self.spec.joint_names.index(joint)] += config["gain"] * delta
            return desired
        axes, gains = np.asarray(profile["axes"]), np.asarray(profile["gains"])
        targets = {}
        for source, hand in profile["hands"].items():
            site = self.spec.mapping[source]
            f, initial = features[hand], neutral["hands"][hand]
            # ponytail: palm size is a depth CONTROL proxy; use calibrated camera pose for metric tracking.
            control = np.r_[f["xy"] - initial["xy"], initial["palm"] / f["palm"] - 1]
            position = np.asarray(neutral["sites"][site]["position"]) + axes @ (gains * control)
            if profile["orientation"]:
                delta = f["basis"] @ np.asarray(initial["basis"]).T
                rotation = axes @ delta @ axes.T @ np.asarray(neutral["sites"][site]["rotation"])
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, rotation.ravel())
                targets[site] = (position, quat)
            else:
                targets[site] = position
        desired = solve(self.spec, self.backend.model, self.backend.data, targets)
        if profile["pinch_hand"] is not None:
            lo, hi = profile["pinch_range"]
            fraction = np.clip((features[profile["pinch_hand"]]["pinch"] - lo) / (hi - lo), 0, 1)
            gripper = self.spec.gripper
            desired[self.spec.joint_names.index(gripper["joint"])] = gripper["closed"] + fraction * (gripper["open"] - gripper["closed"])
        return desired

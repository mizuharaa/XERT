import json
from pathlib import Path

import numpy as np
import pytest

from xert.spec import load_spec

SPEC = Path(__file__).resolve().parents[1] / "xert/spec/xlerobot_left.yaml"


def test_existing_arm_spec():
    spec = load_spec(SPEC)
    assert spec.name == "xlerobot_left"
    assert spec.joint_names[0] == "Rotation_L"
    assert spec.load_model().site("grasp").id >= 0
    assert np.all(spec.velocity_limits > 0)


@pytest.mark.parametrize("key,value", [
    ("joint_names", ["missing"] * 6), ("sites", ["missing"]),
    ("velocity_limits", [0] * 6), ("acceleration_limits", [float("nan")] * 6),
    ("position_limits", [[2, 1]] * 6), ("position_limits", [[-100, 100]] * 6),
    ("rest", [100] * 6), ("control_hz", 0), ("control_hz", True),
    ("mapping", {"wrist": "missing"}), ("ik_weights", {"grasp": -1}),
    ("gripper", {"joint": "Jaw_L", "open": 100, "closed": 0}),
])
def test_invalid_specs(tmp_path, key, value):
    raw = json.loads(SPEC.read_text())
    raw["model_path"] = str((SPEC.parent / raw["model_path"]).resolve())
    raw[key] = value
    path = tmp_path / "bad.yaml"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_spec(path)


def test_non_json_yaml_is_explicitly_rejected(tmp_path):
    path = tmp_path / "unsupported.yaml"
    path.write_text("name: example\n")
    with pytest.raises(ValueError, match="JSON-compatible"):
        load_spec(path)

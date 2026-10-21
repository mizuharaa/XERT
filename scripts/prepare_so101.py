"""Add a named tool point to the pinned SO101 model and write its specification."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / "assets/menagerie/robotstudio_so101"
tree = ET.parse(folder / "so101.xml")
body = tree.find(".//body[@name='gripper']")
tip = body.find("geom[@name='fixed_jaw_sph_tip1']")
ET.SubElement(body, "site", name="grasp", pos=tip.get("pos"), size=".003")
ET.indent(tree)
tree.write(folder / "so101_xert.xml", encoding="unicode")
model = mujoco.MjModel.from_xml_path(str(folder / "so101_xert.xml"))
names = [model.joint(i).name for i in range(model.njnt)]
limits = np.asarray([model.joint(n).range for n in names]).copy()
for i, name in enumerate(names):
    control = model.actuator(name).ctrlrange
    limits[i] = [max(limits[i, 0], control[0]), min(limits[i, 1], control[1])]
spec = dict(name="so101", model_path="../../assets/menagerie/robotstudio_so101/so101_xert.xml",
    joint_names=names, position_limits=limits.tolist(), velocity_limits=[1.0] * len(names),
    acceleration_limits=[3.0] * len(names), rest=[0.0] * len(names), sites=["grasp"],
    mapping={"right_wrist": "grasp"}, ik_weights={"grasp": 1.0}, control_hz=25,
    gripper={"joint": "gripper", "open": 1.4, "closed": 0.0})
(ROOT / "xert/spec/so101.yaml").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")

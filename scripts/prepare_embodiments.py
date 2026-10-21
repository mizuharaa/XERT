"""Derive fixed G1 upper-body and LEAP tool sites from pinned upstream models."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def write_spec(name, path, mapping):
    model = mujoco.MjModel.from_xml_path(str(path))
    names = [model.joint(i).name for i in range(model.njnt)]
    limits = np.asarray([model.joint(n).range for n in names]).copy()
    for i, joint in enumerate(names):
        acts = np.flatnonzero(model.actuator_trnid[:, 0] == model.joint(joint).id)
        assert len(acts) == 1
        a = acts[0]
        if model.actuator_ctrllimited[a]:
            limits[i] = [max(limits[i, 0], model.actuator_ctrlrange[a, 0]),
                         min(limits[i, 1], model.actuator_ctrlrange[a, 1])]
    sites = list(mapping.values())
    spec = dict(name=name, model_path="../../" + path.relative_to(ROOT).as_posix(), joint_names=names,
        position_limits=limits.tolist(), velocity_limits=[1.0] * len(names),
        acceleration_limits=[3.0] * len(names), rest=np.clip(np.zeros(len(names)), *limits.T).tolist(),
        sites=sites, mapping=mapping, ik_weights={s: 1.0 for s in sites}, control_hz=25, gripper={})
    (ROOT / f"xert/spec/{name}.yaml").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")


def main():
    folder = ROOT / "assets/menagerie/unitree_g1"
    tree = ET.parse(folder / "g1.xml")
    pelvis = tree.find(".//body[@name='pelvis']")
    pelvis.remove(pelvis.find("freejoint"))
    for side in ("left", "right"):
        pelvis.remove(pelvis.find(f"body[@name='{side}_hip_pitch_link']"))
        wrist = tree.find(f".//body[@name='{side}_wrist_yaw_link']")
        ET.SubElement(wrist, "site", name=f"{side}_hand", pos="0 0 0", size=".005")
    joints = {j.get("name") for j in tree.findall(".//worldbody//joint")}
    actuator = tree.find("actuator")
    for child in list(actuator):
        if child.get("joint") not in joints:
            actuator.remove(child)
    keyframes = tree.find("keyframe")
    if keyframes is not None:
        tree.getroot().remove(keyframes)
    ET.indent(tree)
    path = folder / "g1_upper.xml"
    tree.write(path, encoding="unicode")
    write_spec("g1_upper", path, {"left_wrist": "left_hand", "right_wrist": "right_hand"})

    folder = ROOT / "assets/menagerie/leap_hand"
    tree = ET.parse(folder / "right_hand.xml")
    model = mujoco.MjModel.from_xml_path(str(folder / "right_hand.xml"))
    mapping = {}
    for prefix, source in (("if", "index_tip"), ("mf", "middle_tip"), ("rf", "ring_tip"), ("th", "thumb_tip")):
        body = tree.find(f".//body[@name='{prefix}_ds']")
        center = model.geom(f"{prefix}_tip").pos
        site = f"{prefix}_tip_site"
        ET.SubElement(body, "site", name=site, pos=" ".join(map(str, center)), size=".002")
        mapping[source] = site
    ET.indent(tree)
    path = folder / "right_hand_xert.xml"
    tree.write(path, encoding="unicode")
    write_spec("leap_hand", path, mapping)


if __name__ == "__main__":
    main()

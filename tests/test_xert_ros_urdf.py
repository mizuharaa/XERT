"""The ROS package's URDF must describe the same SO-101 joints as the XERT spec (no ROS needed)."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "ros/xert_ros"
VENDORED = "assets/menagerie/robotstudio_so101"
# CMakeLists.txt installs VENDORED/assets as share/xert_ros/meshes.
MESH_SOURCE = VENDORED + "/assets"
MESH_PREFIX = "package://xert_ros/meshes/"


def _robot():
    return ET.parse(PACKAGE / "urdf/so101.urdf").getroot()


def _movable_joints_from_base(robot):
    """Movable joints in kinematic order: depth-first from the root link."""
    joints = robot.findall("joint")  # direct children: <transmission> also has <joint>
    children = {}
    for joint in joints:
        children.setdefault(joint.find("parent").get("link"), []).append(joint)
    child_links = {joint.find("child").get("link") for joint in joints}
    roots = [link.get("name") for link in robot.findall("link") if link.get("name") not in child_links]
    assert roots == ["base_link"]
    ordered, stack = [], [roots[0]]
    while stack:
        for joint in children.get(stack.pop(), []):
            if joint.get("type") != "fixed":
                ordered.append(joint)
            stack.append(joint.find("child").get("link"))
    return ordered


def test_urdf_joints_match_so101_spec():
    spec = json.loads((ROOT / "xert/spec/so101.yaml").read_text())
    joints = _movable_joints_from_base(_robot())
    assert [j.get("name") for j in joints] == spec["joint_names"]
    assert {j.get("type") for j in joints} == {"revolute"}
    for joint, (lower, upper), velocity in zip(joints, spec["position_limits"], spec["velocity_limits"]):
        limit = joint.find("limit")
        assert (float(limit.get("lower")), float(limit.get("upper"))) == (lower, upper), joint.get("name")
        assert float(limit.get("velocity")) == velocity, joint.get("name")


def test_urdf_meshes_resolve_to_vendored_files():
    assert f"../../{VENDORED}" in (PACKAGE / "CMakeLists.txt").read_text()
    filenames = {mesh.get("filename") for mesh in _robot().iter("mesh")}
    assert filenames
    for filename in filenames:
        assert filename.startswith(MESH_PREFIX), filename
        assert (ROOT / MESH_SOURCE / filename.removeprefix(MESH_PREFIX)).is_file(), filename


def test_package_and_licence_files_parse():
    package = ET.parse(PACKAGE / "package.xml").getroot()
    assert package.get("format") == "3"
    assert package.findtext("name") == "xert_ros"
    assert (PACKAGE / "urdf/LICENSE").read_bytes() == (ROOT / VENDORED / "LICENSE").read_bytes()
    source = json.loads((PACKAGE / "urdf/SOURCE.json").read_text())
    assert source["license"] == "Apache-2.0" and len(source["commit"]) == 40

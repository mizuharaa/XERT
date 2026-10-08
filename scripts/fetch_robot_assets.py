"""Fetch only the pinned upstream arm model's required meshes, preserving provenance."""
import hashlib
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET

REVISION = "749abc837d5d771f26aeff961009c290e574b024"
ROOT = Path(__file__).resolve().parents[1] / "assets" / "xlerobot"
BASE = f"https://raw.githubusercontent.com/Vector-Wangel/XLeRobot/{REVISION}/"


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    files = ["LICENSE", "simulation/mujoco/xlerobot.xml"]
    xml = urllib.request.urlopen(BASE + files[1], timeout=30).read()
    tree = ET.fromstring(xml)
    arm = tree.find(".//body[@name='Rotation_Pitch']")
    names = {g.get("mesh") for g in arm.iter("geom") if g.get("mesh")}
    files += ["simulation/mujoco/assets/" + mesh.get("file")
              for mesh in tree.find("asset").findall("mesh") if mesh.get("name") in names]
    hashes = {}
    for name in files:
        data = xml if name == files[1] else urllib.request.urlopen(BASE + name, timeout=30).read()
        dest = ROOT / Path(name).name
        dest.write_bytes(data)
        hashes[dest.name] = hashlib.sha256(data).hexdigest()
        print(dest.name, len(data))
    (ROOT / "SOURCE.json").write_text(json.dumps({"repository": "https://github.com/Vector-Wangel/XLeRobot",
        "revision": REVISION, "license": "Apache-2.0", "sha256": hashes}, indent=2) + "\n")


if __name__ == "__main__":
    main()

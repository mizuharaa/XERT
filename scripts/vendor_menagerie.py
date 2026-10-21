"""Fetch only a selected model's referenced meshes at an immutable upstream commit."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET

COMMIT = "cda5236f98383f6fbaa4820c6b18dba892cd087f"
ROOT = Path(__file__).resolve().parents[1]


def vendor(name, filename):
    base = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{COMMIT}/{name}/"
    dest = ROOT / "assets/menagerie" / name
    hashes = {}

    def fetch(relative):
        path = dest / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(base + relative, timeout=60) as response:
            content = response.read()
        path.write_bytes(content)
        hashes[relative] = hashlib.sha256(content).hexdigest()
        return content

    xml = ET.fromstring(fetch(filename))
    fetch("LICENSE")
    fetch("README.md")
    meshdir = xml.find("compiler").get("meshdir", "")
    for mesh in xml.findall("asset/mesh"):
        fetch((Path(meshdir) / mesh.get("file")).as_posix())
    (dest / "SOURCE.json").write_text(json.dumps({
        "repository": "https://github.com/google-deepmind/mujoco_menagerie",
        "commit": COMMIT, "upstream_directory": name, "sha256": hashes,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["so101", "g1", "leap"])
    args = parser.parse_args()
    name, filename = {"so101": ("robotstudio_so101", "so101.xml"),
                      "g1": ("unitree_g1", "g1.xml"),
                      "leap": ("leap_hand", "right_hand.xml")}[args.model]
    vendor(name, filename)

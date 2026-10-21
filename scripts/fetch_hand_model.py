"""Fetch the versioned Google hand model and verify its committed content hash."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = json.loads((ROOT / "assets/mediapipe/SOURCE.json").read_text())
    target = ROOT / "build/models/hand_landmarker.task"
    payload = target.read_bytes() if target.exists() else urllib.request.urlopen(source["url"], timeout=30).read()
    if hashlib.sha256(payload).hexdigest() != source["sha256"]:
        raise ValueError("Downloaded/existing hand model checksum mismatch")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(payload)
    print(target)


if __name__ == "__main__":
    main()

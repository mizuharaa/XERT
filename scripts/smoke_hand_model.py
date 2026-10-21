"""Real model on a black image: API smoke only, no camera or tracking-accuracy claim."""
import hashlib
import json
from pathlib import Path


def main():
    import mediapipe as mp
    import numpy as np
    root = Path(__file__).resolve().parents[1]
    source = json.loads((root / "assets/mediapipe/SOURCE.json").read_text())
    payload = (root / "build/models/hand_landmarker.task").read_bytes()
    assert hashlib.sha256(payload).hexdigest() == source["sha256"]
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=payload),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2)
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        result = detector.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((480, 640, 3), dtype=np.uint8)), 0)
        assert result.hand_landmarks == []
    print("Black-image model API smoke passed. No camera, human tracking, or latency measurement.")


if __name__ == "__main__":
    main()

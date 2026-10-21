"""Mock camera only: optional dependency import, queue bounds, and worker cleanup."""
import hashlib
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mediapipe")
pytest.importorskip("cv2")

from xert.sources.webcam_mediapipe import WebcamSource


def test_capture_mailbox_drops_old_frames_and_closes(tmp_path):
    model = tmp_path / "model.task"
    model.write_bytes(b"mock")
    gate = threading.Event()
    entered = threading.Event()
    class Capture:
        closed = False
        count = 0
        def isOpened(self): return True
        def read(self):
            time.sleep(.003)
            self.count += 1
            return True, np.full((4, 4, 3), self.count % 255, dtype=np.uint8)
        def release(self): self.closed = True
    class Detector:
        closed = False
        timestamps = []
        def detect_for_video(self, image, timestamp):
            self.timestamps.append(timestamp)
            entered.set()
            assert gate.wait(2)
            return SimpleNamespace(handedness=[], hand_landmarks=[], hand_world_landmarks=[])
        def close(self): self.closed = True
    capture, detector = Capture(), Detector()
    source = WebcamSource(model, hashlib.sha256(b"mock").hexdigest(), capture=capture, detector=detector)
    try:
        assert entered.wait(2)
        time.sleep(.04)
        assert source.overwritten > 0
        assert source.pending is not None
        gate.set()
        deadline = time.perf_counter() + 2
        frame = None
        while frame is None and time.perf_counter() < deadline:
            frame = source.read()
            time.sleep(.005)
        assert frame is not None and frame.hands == {}
        assert frame.t_landmarks >= frame.t_capture
    finally:
        gate.set()
        source.close()
    assert capture.closed and detector.closed
    assert not any(t.is_alive() for t in source.threads)
    assert detector.timestamps == sorted(set(detector.timestamps))

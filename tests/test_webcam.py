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


def test_blocked_capture_retains_handle_until_owning_worker_exits(tmp_path):
    model = tmp_path / "model.task"
    model.write_bytes(b"mock")
    entered, unblock = threading.Event(), threading.Event()
    class Capture:
        closed = False
        in_read = False
        reader = None
        def isOpened(self): return True
        def read(self):
            self.reader = threading.get_ident()
            self.in_read = True
            entered.set()
            assert unblock.wait(2)
            self.in_read = False
            return True, np.zeros((4, 4, 3), dtype=np.uint8)
        def release(self):
            assert not self.in_read
            assert threading.get_ident() == self.reader
            self.closed = True
    class Detector:
        closed = False
        def close(self): self.closed = True
    capture, detector = Capture(), Detector()
    source = WebcamSource(model, hashlib.sha256(b"mock").hexdigest(), capture=capture, detector=detector)
    try:
        assert entered.wait(2)
        with pytest.raises(RuntimeError, match="timeout"):
            source.close(timeout=.02)
        assert not capture.closed and not detector.closed
        assert source.threads[0].is_alive()
    finally:
        unblock.set()
        source.close()
    assert capture.closed and detector.closed


def test_release_failure_is_reported_and_preserves_capture_failure(tmp_path):
    model = tmp_path / "model.task"
    model.write_bytes(b"mock")
    class Capture:
        def isOpened(self): return True
        def read(self): raise ValueError("capture failed first")
        def release(self): raise RuntimeError("release failed second")
    detector = SimpleNamespace(close=lambda: None)
    source = WebcamSource(model, hashlib.sha256(b"mock").hexdigest(), capture=Capture(), detector=detector)
    source.threads[0].join(timeout=2)
    assert not source.threads[0].is_alive()
    with pytest.raises(RuntimeError, match="worker failed") as caught:
        source.read()
    assert isinstance(caught.value.__cause__, ValueError)
    assert any("release failed second" in note for note in source.error.__notes__)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        source.close()
    assert source.closed and not any(t.is_alive() for t in source.threads)
    source.close()  # Cleanup is not retried against already closed native handles.


def test_constructor_primary_failure_survives_both_cleanup_failures(tmp_path):
    model = tmp_path / "model.task"
    model.write_bytes(b"mock")
    attempted = []
    class Capture:
        def isOpened(self): raise ValueError("camera open failed first")
        def release(self):
            attempted.append("capture")
            raise RuntimeError("capture release failed")
    class Detector:
        def close(self):
            attempted.append("detector")
            raise RuntimeError("detector close failed")
    with pytest.raises(ValueError, match="camera open failed first") as caught:
        WebcamSource(model, hashlib.sha256(b"mock").hexdigest(), capture=Capture(), detector=Detector())
    assert attempted == ["capture", "detector"]
    assert any("capture release failed" in note for note in caught.value.__notes__)
    assert any("detector close failed" in note for note in caught.value.__notes__)


@pytest.mark.parametrize("release_fails", [False, True])
def test_native_cleanup_failure_is_reported_without_retrying_handles(tmp_path, release_fails):
    model = tmp_path / "model.task"
    model.write_bytes(b"mock")
    entered, unblock = threading.Event(), threading.Event()
    class Capture:
        def isOpened(self): return True
        def read(self):
            entered.set()
            assert unblock.wait(2)
            return True, np.zeros((4, 4, 3), dtype=np.uint8)
        def release(self):
            if release_fails:
                raise RuntimeError("release failed")
    attempts = []
    def close_detector():
        attempts.append("detector")
        raise RuntimeError("detector close failed")
    detector = SimpleNamespace(close=close_detector)
    source = WebcamSource(model, hashlib.sha256(b"mock").hexdigest(), capture=Capture(), detector=detector)
    try:
        assert entered.wait(2)
        source.stop.set()
    finally:
        unblock.set()
        with pytest.raises(RuntimeError, match="cleanup failed") as caught:
            source.close()
    assert str(caught.value.__cause__) == ("release failed" if release_fails else "detector close failed")
    assert source.closed and not any(t.is_alive() for t in source.threads)
    source.close()
    assert attempts == ["detector"]
    with pytest.raises(RuntimeError, match="closed"):
        source.read()

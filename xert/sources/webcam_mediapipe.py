"""Two bounded mailboxes: drain capture continuously, infer only the latest frame.

OpenCV's read-return timestamp excludes unknown sensor exposure/driver buffering.
No camera is opened until WebcamSource is explicitly constructed.
"""
import hashlib
import math
from pathlib import Path
import threading
import time

from .base import HandFrame, MotionFrame


def decode_result(result):
    hands = {}
    ambiguous = set()
    for categories, image, world in zip(result.handedness, result.hand_landmarks, result.hand_world_landmarks, strict=True):
        if not categories:
            continue
        category = categories[0]
        name = category.category_name.lower()
        if name not in ("left", "right"):
            continue
        if name in hands:
            ambiguous.add(name)
        hands[name] = HandFrame([[p.x, p.y, p.z] for p in image],
                                [[p.x, p.y, p.z] for p in world], category.score)
    return {name: hand for name, hand in hands.items() if name not in ambiguous}


class WebcamSource:
    def __init__(self, model_path, model_sha256, *, camera=0, clock=time.perf_counter,
                 capture=None, detector=None):
        import cv2
        import mediapipe as mp
        payload = Path(model_path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != model_sha256:
            raise ValueError("Hand model checksum mismatch")
        self.clock, self.cv2, self.mp = clock, cv2, mp
        self.condition = threading.Condition()
        self.pending = self.latest = None
        self.error = None
        self.stop = threading.Event()
        self.closed = False
        self.captured = self.inferred = self.overwritten = 0
        self.detector = detector
        self.threads = []
        self.capture = capture if capture is not None else cv2.VideoCapture(camera)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError("Camera could not be opened")
        try:
            if self.detector is None:
                options = mp.tasks.vision.HandLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_buffer=payload),
                    running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2)
                self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
            self.threads = [threading.Thread(target=self._capture, daemon=True),
                            threading.Thread(target=self._infer, daemon=True)]
            for thread in self.threads:
                thread.start()
        except BaseException:
            self.stop.set()
            with self.condition:
                self.condition.notify_all()
            for thread in self.threads:
                if thread.ident is not None:
                    thread.join(timeout=5)
            # Until capture starts, construction still owns the device.
            if not self.threads or self.threads[0].ident is None:
                self.capture.release()
            if self.detector is not None and not any(t.is_alive() for t in self.threads):
                self.detector.close()
            raise

    def _fail(self, error):
        with self.condition:
            self.error = error
            self.stop.set()
            self.condition.notify_all()

    def _capture(self):
        try:
            while not self.stop.is_set():
                ok, pixels = self.capture.read()
                captured = self.clock()
                if self.stop.is_set():
                    break
                if not ok or pixels is None:
                    raise RuntimeError("Camera stopped delivering frames")
                with self.condition:
                    if self.pending is not None:
                        self.overwritten += 1
                    self.captured += 1
                    self.pending = (self.captured, captured, pixels.copy())
                    self.condition.notify_all()
        except BaseException as exc:
            self._fail(exc)
        finally:
            # Never release OpenCV's native handle concurrently with read().
            self.capture.release()

    def _infer(self):
        last_ms = -1
        try:
            while not self.stop.is_set():
                with self.condition:
                    self.condition.wait_for(lambda: self.pending is not None or self.stop.is_set())
                    if self.stop.is_set():
                        break
                    sequence, captured, pixels = self.pending
                    self.pending = None
                # VIDEO inference blocks this worker only. Capture keeps replacing its one slot.
                timestamp_ms = int(captured * 1000)
                if timestamp_ms <= last_ms:
                    continue
                last_ms = timestamp_ms
                rgb = self.cv2.cvtColor(pixels, self.cv2.COLOR_BGR2RGB)
                result = self.detector.detect_for_video(
                    self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
                frame = MotionFrame(sequence, captured, self.clock(), decode_result(result),
                                    "webcam", (pixels.shape[1], pixels.shape[0]))
                with self.condition:
                    self.latest = frame
                    self.inferred += 1
        except BaseException as exc:
            self._fail(exc)

    def read(self):
        with self.condition:
            if self.error is not None:
                raise RuntimeError("Webcam worker failed") from self.error
            frame, self.latest = self.latest, None
            return frame

    def close(self, timeout=5):
        if self.closed:
            return
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Close timeout must be finite and positive")
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        deadline = time.monotonic() + timeout
        for thread in self.threads:
            thread.join(timeout=max(0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in self.threads):
            raise RuntimeError("Camera/inference worker did not stop before timeout; resources remain owned by workers")
        self.detector.close()
        self.closed = True

"""Landmark clips saved without images; webcam recordings replay as recorded_human."""
from dataclasses import replace
import json
from pathlib import Path
import time

from .base import HandFrame, MotionFrame

REPLAY_SOURCE = {"webcam": "recorded_human", "scripted": "scripted"}


def dump_frame(frame):
    return {"sequence": frame.sequence, "t_capture": frame.t_capture, "t_landmarks": frame.t_landmarks,
            "image_size": list(frame.image_size),
            "hands": {name: {"image": hand.image.tolist(), "world": hand.world.tolist(),
                             "handedness_score": hand.handedness_score} for name, hand in frame.hands.items()}}


def load_clip(path, *, contents=None):
    """Return (header, frames); the frame source comes from the recorded provenance."""
    clip = json.loads(Path(path).read_bytes() if contents is None else contents)
    if not isinstance(clip, dict) or clip.get("version") != 1 or clip.get("source") not in REPLAY_SOURCE:
        raise ValueError("Clip needs version 1 and a webcam or scripted source")
    source = REPLAY_SOURCE[clip["source"]]
    frames = [MotionFrame(f["sequence"], f["t_capture"], f["t_landmarks"],
                          {name: HandFrame(h["image"], h["world"], h["handedness_score"])
                           for name, h in f["hands"].items()}, source, tuple(f["image_size"]))
              for f in clip.get("frames", [])]
    if not frames or any(b.sequence <= a.sequence or b.t_capture < a.t_capture or b.t_landmarks < a.t_landmarks
                         for a, b in zip(frames, frames[1:])):
        raise ValueError("Clip frames must be nonempty with increasing sequence and timestamps")
    return clip, frames


class MotionFileSource:
    """Release frames at their recorded relative landmark time on the loop clock, latest only."""

    def __init__(self, frames, *, clock=time.perf_counter):
        self.frames, self.clock = frames, clock
        self.next = 0
        self.offset = None
        self.skipped = 0  # Due frames superseded before a read, like the webcam's latest-value slot.

    def read(self):
        now = self.clock()
        if self.offset is None:
            self.offset = now - self.frames[0].t_capture
        due = None
        # Compare rebased values so run_loop's not-in-the-future check holds exactly.
        while self.next < len(self.frames) and self.frames[self.next].t_landmarks + self.offset <= now:
            self.skipped += due is not None
            due = self.frames[self.next]
            self.next += 1
        if due is None:
            return None
        return replace(due, t_capture=due.t_capture + self.offset, t_landmarks=due.t_landmarks + self.offset)

    def close(self):
        pass

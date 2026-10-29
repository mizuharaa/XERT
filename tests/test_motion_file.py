"""Scripted clips only: replay timing checks, never human evidence."""
import json

import numpy as np
import pytest

from test_runtime import Clock, frame
from xert.sources.motion_file import MotionFileSource, dump_frame, load_clip


def scripted_clip(tmp_path, moves):
    """Neutral for six 30 fps frames, then the given (shift, curl) poses; 5 ms inference delay."""
    poses = [(0, 0)] * 6 + moves
    frames = []
    for i, (shift, curl) in enumerate(poses):
        f = frame(10 + i / 30, i, shift=shift, curl=curl)
        frames.append({**dump_frame(f), "t_landmarks": f.t_capture + .005})
    path = tmp_path / "clip.json"
    path.write_text(json.dumps({"version": 1, "source": "scripted", "frames": frames}))
    return path


def test_replay_keeps_recorded_timing_on_loop_clock(tmp_path):
    _, frames = load_clip(scripted_clip(tmp_path, [(.01, 0)] * 2))
    assert {f.source for f in frames} == {"scripted"}
    clock = Clock()
    source = MotionFileSource(frames, clock=clock)
    assert source.read() is None  # first landmarks finish 5 ms after capture
    clock.sleep(.006)
    first = source.read()
    assert first.sequence == 0 and first.t_landmarks <= clock.now
    assert np.isclose(first.t_capture, 100) and np.isclose(first.t_landmarks, 100.005)
    clock.sleep(.07)  # frames 1 and 2 are both due; only the latest is delivered
    assert source.read().sequence == 2 and source.skipped == 1
    assert source.read() is None
    with pytest.raises(ValueError, match="increasing"):
        load_clip(tmp_path / "x", contents=json.dumps({"version": 1, "source": "scripted",
                                                         "frames": [dump_frame(frame()), dump_frame(frame())]}))
    with pytest.raises(ValueError, match="source"):
        load_clip(tmp_path / "x", contents=json.dumps({"version": 1, "source": "recorded_human", "frames": []}))

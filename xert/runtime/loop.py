"""Pace simulation ticks without confusing nominal time with camera age."""
import time

import numpy as np

from xert.safety import SafetyController


def run_loop(source, backend, mapper, duration, *, clock=time.perf_counter, sleep=time.sleep,
             on_step=None, stop_requested=lambda: False):
    """Own/close source and backend. Return traces; callback sees state AFTER each step.

    A whole missed period aborts the run instead of bursting catch-up commands.
    Latency covers host acquisition return through simulator step completion only.
    """
    records = []
    try:
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("Duration must be finite and positive")
        spec = backend.spec
        safety = SafetyController(spec, backend.read_state().q)
        dt, start = 1 / spec.control_hz, clock()
        frame = desired = None
        ik_done = None
        previous_sequence = -1
        previous_capture = -float("inf")
        applied_sequences = set()
        tick = 0
        while tick * dt < duration:
            deadline = start + tick * dt
            sleep(max(0, deadline - clock()))
            if clock() - deadline >= dt:
                raise RuntimeError("Control deadline missed by a whole period; stopped without catch-up")
            incoming = source.read()
            if incoming is not None:
                now = clock()
                if (incoming.sequence <= previous_sequence or incoming.t_capture < previous_capture or
                        incoming.t_landmarks > now or incoming.t_capture > now):
                    raise ValueError("Source frame sequence/timestamps are not monotonic or are in the future")
                previous_sequence, previous_capture = incoming.sequence, incoming.t_capture
                frame = incoming
                desired = mapper.map(frame)
                ik_done = clock()
            estop = bool(stop_requested())
            command = safety.step(desired, t_now=tick * dt,
                                  t_capture=frame.t_capture if desired is not None else None,
                                  watchdog_now=clock(), estop=estop)
            sent = clock()
            if sent - deadline >= dt:
                raise RuntimeError("Retargeting missed control deadline; no late command sent")
            backend.send_targets(command.q, tick * dt)
            state = backend.read_state()
            applied = clock()
            valid = frame is not None and desired is not None and command.phase == "live"
            sample = valid and frame.sequence not in applied_sequences
            if sample:
                applied_sequences.add(frame.sequence)
            record = {"t_control": tick * dt, "t_sim": float(state.t),
                      "frame_sequence": None if frame is None else frame.sequence,
                      "source": None if frame is None else frame.source,
                      "t_capture": None if frame is None else frame.t_capture,
                      "t_landmarks": None if frame is None else frame.t_landmarks,
                      "t_ik": ik_done, "t_command": sent, "t_applied": applied,
                      "q": state.q.tolist(), "qd": state.qd.tolist(), "command": command.q.tolist(),
                      "clamped": bool(command.clamped), "events": command.events,
                      "phase": command.phase, "valid_motion": bool(valid), "latency_sample": bool(sample),
                      "handedness_scores": {} if frame is None else {name: h.handedness_score for name, h in frame.hands.items()},
                      "tracking_confidence": None, "deadline_overrun": bool(applied > deadline + dt)}
            records.append(record)
            if on_step is not None:
                on_step(record)
            if record["deadline_overrun"]:
                raise RuntimeError("Simulation step exceeded control period; stopped without catch-up")
            if estop:
                break
            tick += 1
        return records
    finally:
        try:
            source.close()
        finally:
            backend.close()

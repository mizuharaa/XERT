"""One simulation episode written directly to LeRobot v3 from runtime callbacks."""
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile

import numpy as np

from xert.backends.base import JointState
from xert.data.lerobot_export import _publish
from xert.spec import load_spec
from xle_workbench.storage import atomic_json

TIMES = ("t_control", "t_sim_before", "t_sim", "t_capture", "t_landmarks", "t_ik", "t_command", "t_applied")
PHASES = {"live", "hold", "return", "rest", "estop"}


def load_runtime_dataset(root, repo_id="xert/runtime"):
    """Return LeRobotDataset with declared float64 features preserved as torch.float64.

    The pinned upstream default transform converts Python float lists to float32,
    even when their Parquet feature is float64. State/action behavior is unchanged.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.io_utils import hf_transform_to_torch
    dataset = LeRobotDataset(repo_id=repo_id, root=root, video_backend="pyav", download_videos=False)
    doubles = {key for key, feature in dataset.features.items() if feature["dtype"] == "float64"}
    def preserve_double(items):
        for key in doubles & items.keys():
            items[key] = [np.asarray(value, dtype=np.float64) for value in items[key]]
        return hf_transform_to_torch(items)
    dataset.hf_dataset.set_transform(preserve_double)
    return dataset


def _features(spec, task_success=False, environment_names=None):
    def vector(names, dtype="float32"):
        return {"dtype": dtype, "shape": (len(names),), "names": names}
    positions = [f"{name}.pos" for name in spec.joint_names]
    velocities = [f"{name}.vel" for name in spec.joint_names]
    features = {"observation.state": vector(positions), "action": vector(positions),
                **({"observation.environment_state": vector(list(environment_names))} if environment_names else {}),
                "xert.observation_velocity": vector(velocities),
                "xert.next_observation": vector(positions), "xert.next_velocity": vector(velocities),
                "xert.times": vector(list(TIMES), "float64"),
                "xert.times_present": vector(list(TIMES), "bool"),
                "xert.handedness_scores": vector(["left", "right"]),
                "xert.handedness_present": vector(["left", "right"], "bool")}
    for name in ("clamped", "valid_motion", "latency_sample", "frame_sequence_present", "tracking_confidence_present"):
        features[f"xert.{name}"] = vector([name], "bool")
    for name in ("clamp_count", "frame_sequence"):
        features[f"xert.{name}"] = vector([name], "int64")
    features["xert.tracking_confidence"] = vector(["tracking_confidence"])
    for name in ("phase", "source", "events_json"):
        features[f"xert.{name}"] = vector([name], "string")
    if task_success:
        features["next.success"] = vector(["success"], "bool")  # LeRobot name: success after this action.
    return features


class RuntimeRecorder:
    """Context-managed on_step callback, initialized with the PRE-action state.

    Successful exit verifies and atomically publishes a fresh dataset. Failures
    retain a private staging directory and flushed journal, never a final dataset.
    ponytail: one episode per transaction; the P4.1 teleop session publishes one dataset per episode.
    task_success=True adds `next.success`; every record then needs an explicit boolean "success".
    environment_state=(names, initial values) adds `observation.environment_state` (LeRobot name); every record
    then needs "environment_state" after its step, stored pre-action on the next frame exactly like observation.state.
    """
    def __init__(self, spec_path, initial_state, output, *, task, source, repo_id="xert/runtime", spec_bytes=None,
                 task_success=False, environment_state=None):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        self.output = Path(output).absolute()
        if self.output.exists() or self.output.is_symlink():
            raise FileExistsError(self.output)
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Recording requires a nonempty task description")
        if not isinstance(source, dict) or source.get("kind") not in ("scripted", "webcam", "recorded_human"):
            raise ValueError("Source provenance must declare scripted, webcam, or recorded_human")
        self.source = json.loads(json.dumps(source, allow_nan=False))
        spec_bytes = Path(spec_path).read_bytes() if spec_bytes is None else spec_bytes
        self.spec = load_spec(spec_path, contents=spec_bytes)
        if self.spec.control_hz != int(self.spec.control_hz):
            raise ValueError("LeRobot requires an integer control rate")
        self.features = _features(self.spec, task_success, environment_state and environment_state[0])
        self.previous = self._state(initial_state.q, initial_state.qd, initial_state.t)
        self.environment = environment_state and self._environment(environment_state[1])
        self.task, self.repo_id = task, repo_id
        self.count, self.closed = 0, False
        self.dataset = self.journal = None
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.work = Path(tempfile.mkdtemp(prefix=".xert-recording-", dir=self.output.parent))
        self.staging = self.work / "dataset"
        try:
            self.dataset = LeRobotDataset.create(repo_id=repo_id, root=self.staging, robot_type=self.spec.name,
                                                fps=int(self.spec.control_hz), features=self.features,
                                                use_videos=False, video_backend="pyav")
            self.manifest = {"version": 1, "status": "recording", "backend": "mujoco", "hardware": False,
                             "embodiment": self.spec.name, "joint_names": list(self.spec.joint_names),
                             "fps": self.spec.control_hz, "task": task, "task_outcome": None,
                             "source": self.source, "lerobot_version": importlib.metadata.version("lerobot"),
                             "state_units": "joint position radians", "velocity_units": "radians per second",
                             "action_units": "absolute joint position targets in radians, after safety clamps",
                             "state_alignment": "observation before action; xert.next_observation after action",
                             "time_names": list(TIMES), "time_units": "float64 seconds",
                             "time_domains": {"t_control": "nominal simulation control clock",
                                              "t_sim_before": "MuJoCo simulation", "t_sim": "MuJoCo simulation",
                                              **{k: "host monotonic clock (source declares real or scripted)" for k in TIMES[3:]}},
                             "missing_values": "Numeric zero is padding only where matching presence mask is false; journal retains null",
                             "timestamp": "LeRobot timestamp is derived frame_index/fps; use xert.times for measured stages",
                             "initial_state": {"q": self.previous.q.tolist(), "qd": self.previous.qd.tolist(), "t": self.previous.t},
                             "spec_sha256": hashlib.sha256(spec_bytes).hexdigest()}
            atomic_json(self.staging / "xert_recording.json", self.manifest)
            self.journal = (self.staging / "xert_frames.jsonl").open("x", encoding="utf-8")
        except BaseException as exc:
            try:
                if self.dataset is not None:
                    self.dataset.finalize()
            except BaseException as cleanup:
                exc.add_note(f"Recording initialization cleanup also failed: {cleanup!r}")
            finally:
                self.dataset = None
                self.closed = True
            exc.add_note(f"Unpublished recording staging retained at {self.staging}")
            raise

    def _vector(self, value):
        result = np.asarray(value, dtype=np.float32)
        if result.shape != self.spec.rest.shape or not np.isfinite(result).all():
            raise ValueError("Recorded joint vectors must be finite and follow spec joint order")
        return result.copy()

    def _environment(self, value):
        result = np.asarray(value, dtype=np.float32)
        if result.shape != self.features["observation.environment_state"]["shape"] or not np.isfinite(result).all():
            raise ValueError("Environment state must be finite and follow its declared names")
        return result

    def _state(self, q, qd, t):
        if not np.isfinite(t) or t < 0:
            raise ValueError("Simulation timestamp must be finite and nonnegative")
        return JointState(self._vector(q), self._vector(qd), float(t))

    def _encode(self, record):
        state = self._state(record["q"], record["qd"], record["t_sim"])
        dt = 1 / self.spec.control_hz
        if (not np.isclose(record["t_control"], self.count * dt, atol=1e-9, rtol=1e-8) or
                not np.isclose(state.t - self.previous.t, dt, atol=1e-9, rtol=1e-8)):
            raise ValueError("Runtime steps must be contiguous and advance simulation by one control period")
        action = self._vector(record["command"])
        if np.any(action < self.spec.position_limits[:, 0] - 1e-6) or np.any(action > self.spec.position_limits[:, 1] + 1e-6):
            raise ValueError("Applied targets exceed spec position limits")
        values = {**record, "t_sim_before": self.previous.t}
        times = [values.get(key) for key in TIMES]
        present = np.asarray([value is not None for value in times], dtype=bool)
        if not present[[0, 1, 2, 6, 7]].all():
            raise ValueError("Control, simulation, command and applied times are required")
        timestamps = np.asarray([0 if value is None else value for value in times], dtype=np.float64)
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps[3:][present[3:]]) < 0):
            raise ValueError("Host stage timestamps must be finite and ordered")
        if present[3] != present[4]:
            raise ValueError("Capture and landmark timestamps must both be present or missing")
        if self.count and timestamps[6] < self.last_command:
            raise ValueError("Command host time must be monotonic")
        if record["source"] not in (None, self.source["kind"]):
            raise ValueError("Frame source conflicts with recording provenance")
        sequence = record["frame_sequence"]
        if sequence is not None and (type(sequence) is not int or sequence < 0):
            raise ValueError("Frame sequence must be nonnegative or missing")
        for key in ("clamped", "valid_motion", "latency_sample"):
            if type(record[key]) is not bool:
                raise ValueError(f"{key} must be an explicit boolean")
        if record["valid_motion"] and (not present[3:6].all() or sequence is None or record["source"] is None):
            raise ValueError("Valid motion requires source identity, sequence and processing timestamps")
        if record["phase"] not in PHASES or not isinstance(record["events"], list):
            raise ValueError("Invalid safety phase/events")
        if any(not isinstance(event, dict) for event in record["events"]):
            raise ValueError("Safety events must be objects")
        events = json.dumps(record["events"], allow_nan=False, separators=(",", ":"))
        scores = record["handedness_scores"]
        confidence = record["tracking_confidence"]
        if not isinstance(scores, dict) or set(scores) - {"left", "right"}:
            raise ValueError("Handedness scores require named hands")
        for value in list(scores.values()) + ([] if confidence is None else [confidence]):
            if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Available confidence scores must be finite probabilities")
        def scalar(value, dtype):
            return np.asarray([value], dtype=dtype)
        payload = {"observation.state": self.previous.q.copy(), "action": action,
                   "xert.observation_velocity": self.previous.qd.copy(),
                   "xert.next_observation": state.q, "xert.next_velocity": state.qd,
                   "xert.times": timestamps, "xert.times_present": present,
                   "xert.handedness_scores": np.asarray([scores.get(h, 0) for h in ("left", "right")], dtype=np.float32),
                   "xert.handedness_present": np.asarray([h in scores for h in ("left", "right")], dtype=bool),
                   "xert.tracking_confidence": scalar(0 if confidence is None else confidence, np.float32),
                   "xert.tracking_confidence_present": scalar(confidence is not None, bool),
                   "xert.frame_sequence": scalar(0 if sequence is None else sequence, np.int64),
                   "xert.frame_sequence_present": scalar(sequence is not None, bool),
                   "xert.clamp_count": scalar(sum(e.get("kind") == "clamp" for e in record["events"]), np.int64),
                   "xert.phase": record["phase"], "xert.source": record["source"] or "",
                   "xert.events_json": events, "task": self.task}
        for key in ("clamped", "valid_motion", "latency_sample"):
            payload[f"xert.{key}"] = scalar(record[key], bool)
        if "next.success" in self.features:
            if type(record["success"]) is not bool:
                raise ValueError("success must be an explicit boolean")
            payload["next.success"] = scalar(record["success"], bool)
        environment = None
        if "observation.environment_state" in self.features:
            environment = self._environment(record["environment_state"])
            payload["observation.environment_state"] = self.environment.copy()
        return payload, state, environment

    def __call__(self, record):
        if self.closed:
            raise RuntimeError("Recorder is closed")
        try:
            payload, state, environment = self._encode(record)
            encoded = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in payload.items()}
            self.journal.write(json.dumps({"frame": encoded, "runtime": record}, allow_nan=False) + "\n")
            self.journal.flush()  # Recoverable journal even before LeRobot saves its episode buffer.
            self.dataset.add_frame(payload)
            self.count += 1
            self.previous, self.environment, self.last_command = state, environment, record["t_command"]
        except BaseException as exc:
            self._abort_preserving(exc)
            raise

    def _abort_preserving(self, original):
        try:
            self.abort()
        except BaseException as cleanup:
            original.add_note(f"Recording cleanup also failed: {cleanup!r}")
        original.add_note(f"Unpublished recording journal retained at {self.staging}")

    def abort(self):
        if self.closed:
            return
        self.closed = True
        if self.journal is not None:
            self.journal.close()
        try:
            if self.dataset is not None:
                self.dataset.finalize()
        finally:
            self.dataset = None
            gc.collect()
            self.manifest.update(status="aborted", frames=self.count, recovery_journal="xert_frames.jsonl")
            atomic_json(self.staging / "xert_recording.json", self.manifest)

    def finish(self, outcome=None):
        if self.closed:
            raise RuntimeError("Recorder is closed")
        try:
            if not self.count:
                raise ValueError("Cannot publish an empty recording")
            self.journal.close()
            self.dataset.save_episode(parallel_encoding=False)
            self.dataset.finalize()
            self.dataset = None
            gc.collect()
            verification = verify_recording(self.staging, self.repo_id)
            self.manifest.update(status="complete", frames=self.count, task_outcome=outcome)
            atomic_json(self.staging / "xert_recording.json", self.manifest)
            atomic_json(self.staging / "roundtrip.json", verification)
            _publish(self.staging, self.output)
            self.closed = True
            try:
                self.work.rmdir()
            except OSError:
                pass  # Published data is complete; an empty staging parent is harmless.
            return verification
        except BaseException as exc:
            self._abort_preserving(exc)
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.finish()
        else:
            self._abort_preserving(exc)
        return False


def verify_recording(root, repo_id="xert/runtime"):
    dataset = load_runtime_dataset(root, repo_id)
    maximum, count = 0., 0
    try:
        with (Path(root) / "xert_frames.jsonl").open(encoding="utf-8") as stream:
            for count, line in enumerate(stream, 1):
                expected = json.loads(line)["frame"]
                actual = dataset[count - 1]
                for key, value in expected.items():
                    if isinstance(value, str):
                        if actual[key] != value:
                            raise ValueError(f"Recorded string mismatch: {key}")
                        continue
                    dtype = dataset.features[key]["dtype"]
                    wanted = np.asarray(value, dtype=dtype)
                    if wanted.shape == (1,):
                        wanted = wanted.reshape(())
                    got = actual[key].numpy()
                    if got.shape != wanted.shape or got.dtype != wanted.dtype or not np.array_equal(got, wanted):
                        raise ValueError(f"Recorded array mismatch: {key}")
                    if dtype.startswith("float"):
                        maximum = max(maximum, float(np.max(np.abs(got - wanted))))
                if actual["episode_index"].item() != 0 or actual["frame_index"].item() != count - 1:
                    raise ValueError("Recorded frame indexing mismatch")
        if len(dataset) != count or dataset.num_episodes != 1:
            raise ValueError("Recorded frame/episode count mismatch")
    finally:
        del dataset
        gc.collect()
    return {"frames": count, "episodes": 1, "max_abs_error": maximum}

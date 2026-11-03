"""`xert export-lerobot`, `xert import-lerobot` and the derived `xert to-store` conversion.

Export only wraps the P2.1 exporter and P2.3 `render-cameras`. Import is dispatched by manifest:
  xert_export.json      -> (a) restore the Store NPZ episodes exactly (both action streams, terminal obs).
  xert_recording.json   -> (b) frame pairs in radians, plus named environment_state and the bench model sha.
  neither, robot_type so101_follower / so_follower -> (c) hardware frame pairs, radians only through a
                           complete P6 joint map, or native units labelled raw_lerobot.
Anything else is refused. Frame pairs carry no rewards, terminal successor or termination flags.
"""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from xert.data.lerobot_export import PROFILES, _publish
from xle_workbench.storage import Store, atomic_json, sha256

ROOT = Path(__file__).resolve().parents[2]
JOINT_MAP = ROOT / "configs/hardware/so101_joint_map.yaml"
# LeRobot 0.6.1 `lerobot-record` writes robot.name, which is "so_follower" for SO101Follower
# (so_follower.py:44,242); older releases wrote the config type "so101_follower".
HARDWARE = ("so101_follower", "so_follower")
ABSENT = "no rewards, terminal successor or termination flags: absent in the source, never synthesized"
# Store metadata keys that save_episode recomputes on restore.
DERIVED_KEYS = {"id", "schema_version", "steps", "observation_dim", "action_dim", "sha256",
                "training_action_source", "corrective_sha256"}


def _fresh(output):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _hashes(root):
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}


def source_kind(root):
    """Which import path a dataset takes, from its manifests alone; unknown or ambiguous sources are refused."""
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text("utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Not a LeRobot v3.0 dataset: {info.get('codebase_version')!r}")
    manifests = [name for name in ("xert_export.json", "xert_recording.json") if (root / name).exists()]
    if len(manifests) > 1:
        raise ValueError(f"Ambiguous source: both {manifests}")
    if manifests:
        return {"xert_export.json": "xert_export", "xert_recording.json": "xert_recording"}[manifests[0]]
    if info.get("robot_type") in HARDWARE:
        return "so101_follower"
    raise ValueError(f"Unknown source (no XERT manifest, robot_type {info.get('robot_type')!r}); refused")


def _columns(root):
    """Every Parquet column as numpy, rows ordered by (episode_index, frame_index); never trusts file order."""
    import pyarrow as pa
    from xert.data.cameras import _read_table
    table = _read_table(root)
    numeric = (pa.types.is_integer, pa.types.is_floating, pa.types.is_boolean)
    columns = {}
    for name in table.column_names:
        column = table.column(name).combine_chunks()
        if (pa.types.is_list(column.type) or pa.types.is_fixed_size_list(column.type)) and any(test(column.type.value_type) for test in numeric):
            columns[name] = column.flatten().to_numpy(zero_copy_only=False).reshape(len(column), -1)
        elif any(test(column.type) for test in numeric):
            columns[name] = column.to_numpy(zero_copy_only=False)
        # Strings and images are not part of any import format; images stay in the raw copy.
    order = np.lexsort((columns["frame_index"], columns["episode_index"]))
    columns = {key: value[order] for key, value in columns.items()}
    for episode in np.unique(columns["episode_index"]):
        frames = columns["frame_index"][columns["episode_index"] == episode]
        if not np.array_equal(frames, np.arange(len(frames))):
            raise ValueError(f"Episode {episode} frame_index is not contiguous from 0")
    return columns


def _names(info, key):
    return info["features"][key]["names"]


def _restore_export(raw, staging):
    """(a) Back into the Store NPZ format: both action streams, rewards, flags and the terminal observation."""
    manifest = json.loads((raw / "xert_export.json").read_text("utf-8"))
    columns = _columns(raw)
    records = manifest["source_episodes"]
    if sorted(np.unique(columns["episode_index"]).tolist()) != list(range(len(records))):
        raise ValueError("Export episodes differ from its manifest")
    store, mapping = Store(staging / "dataset"), []
    for index, record in enumerate(records):
        rows = columns["episode_index"] == index
        state, successor = columns["observation.state"][rows], columns["xert.next_observation"][rows]
        if len(state) != record["steps"] or not np.array_equal(state[1:], successor[:-1]):
            raise ValueError(f"Episode {index}: length or state/successor chain differs from its source")
        arrays = {"observations": np.concatenate([state, successor[-1:]]),
                  "actions": columns["xert.executed_action"][rows],
                  # LeRobot stores shape-(1,) features as scalar columns.
                  "rewards": columns["next.reward"][rows].reshape(-1), "terminated": columns["next.done"][rows].reshape(-1)}
        labels = columns["action"][rows]
        corrective = record.get("training_action_source") == "expert_correction"
        if not corrective and not np.array_equal(labels, arrays["actions"]):
            raise ValueError(f"Episode {index}: action differs from the executed stream without corrective labels")
        saved = store.save_episode(arrays, {k: v for k, v in record.items() if k not in DERIVED_KEYS},
                                   corrective_actions=labels if corrective else None)
        mapping.append({"source_id": record["id"], "id": saved["id"]})
    atomic_json(staging / "collection.json", {"episode_ids": [m["id"] for m in mapping],
                                              "restored_from": "LeRobot v3 xert_export.json (P2.1)"})
    return {"episodes": mapping, "frames": len(columns["frame_index"]), "units": "xert_store",
            "format": "Store NPZ in dataset/ + collection.json; new IDs, arrays identical to the exported source"}


def _recording(raw, info):
    """(b) A P2.2/P4.1 runtime recording: radians already, kept float32 exactly as recorded."""
    manifest = json.loads((raw / "xert_recording.json").read_text("utf-8"))
    joints = [f"{name}.pos" for name in manifest["joint_names"]]
    if manifest.get("status") != "complete" or _names(info, "observation.state") != joints \
            or _names(info, "action") != joints or manifest["state_units"] != "joint position radians":
        raise ValueError("Runtime recording is incomplete or its joint/unit contract differs")
    columns = _columns(raw)
    arrays = {"state": columns["observation.state"], "action": columns["action"]}
    names = {"state": joints, "action": joints}
    if "observation.environment_state" in columns:
        arrays["environment_state"] = columns["observation.environment_state"]
        names["environment_state"] = _names(info, "observation.environment_state")
    bench = manifest["source"].get("bench")
    return columns, arrays, {"units": "radians", "embodiment": manifest["embodiment"],
                             "spec_sha256": manifest["spec_sha256"], "columns": names,
                             "model_sha256": bench["model_sha256"] if bench else None,
                             "state_units": manifest["state_units"], "action_units": manifest["action_units"]}


def _hardware(raw, info, joint_map, raw_units, motor_units):
    """(c) LeRobot so101 follower data: `<motor>.pos`, degrees (use_degrees=True) and gripper RANGE_0_100."""
    from xert.backends.so101_hw_backend import load_joint_map
    from xert.spec import load_spec
    spec_path = ROOT / "xert/spec/so101.yaml"
    spec_bytes = spec_path.read_bytes()
    spec = load_spec(spec_path, contents=spec_bytes)
    motors = [f"{name}.pos" for name in spec.joint_names]  # P6 map: motor name = joint name, spec order.
    if _names(info, "observation.state") != motors or _names(info, "action") != motors:
        raise ValueError(f"Hardware state/action names must be exactly {motors}")
    columns = _columns(raw)
    state, action = columns["observation.state"], columns["action"]
    meta = {"embodiment": spec.name, "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
            "columns": {"state": motors, "action": motors}, "robot_type": info["robot_type"]}
    if raw_units:
        return columns, {"state": state, "action": action}, {**meta, "units": "raw_lerobot",
            "raw_units": "LeRobot motor units as recorded (degrees if use_degrees=True; gripper 0-100); not radians"}
    if motor_units != "degrees":
        raise ValueError("Radians need an explicit --motor-units degrees: LeRobot 0.6.1 does not record use_degrees")
    map_bytes = Path(joint_map).read_bytes()
    entries = load_joint_map(joint_map, spec, dry_run=False, contents=map_bytes)  # Any TBD joint is refused here.
    if [e["units"] for e in entries] != ["degrees"] * 5 + ["range_0_100"]:
        raise ValueError("Joint map units must be degrees for the body joints and range_0_100 for the gripper")
    sign = np.array([e["sign"] for e in entries], dtype=np.float64)
    scale = np.array([e["units_per_rad"] for e in entries], dtype=np.float64)
    offset = np.array([e["zero_offset"] for e in entries], dtype=np.float64)
    # Inverse of the P6 map `motor = sign * units_per_rad * rad + zero_offset`, in float64.
    converted = {key: (motor.astype(np.float64) - offset) / (sign * scale)
                 for key, motor in (("state", state), ("action", action))}
    return columns, converted, {**meta, "units": "radians", "motor_units": motor_units, "use_degrees_assumed": True,
                                "joint_map": {"path": str(Path(joint_map)), "sha256": hashlib.sha256(map_bytes).hexdigest()},
                                "conversion": "rad = (motor - zero_offset) / (sign * units_per_rad), float64"}


def _calibration(path):
    """sha256 and robot id of the follower's LeRobot calibration file (path c provenance for replay-real --ledger).

    LeRobot 0.6.1 saves it as <calibration_dir>/<robot.id>.json (robots/robot.py:49-56) and records the id nowhere
    in the dataset, so the id comes from the file name.
    """
    path = Path(path)
    raw = path.read_bytes()
    calibration = json.loads(raw)
    motors = json.loads((ROOT / "xert/spec/so101.yaml").read_bytes())["joint_names"]
    fields = {"id", "drive_mode", "homing_offset", "range_min", "range_max"}  # motors_bus.py MotorCalibration
    if not isinstance(calibration, dict) or list(calibration) != motors or any(
            not isinstance(v, dict) or set(v) != fields or any(type(x) is not int for x in v.values())
            for v in calibration.values()):
        raise ValueError(f"Not a LeRobot so101 follower calibration file (motors {motors}, fields {sorted(fields)})")
    return {"path": str(path.absolute()), "sha256": hashlib.sha256(raw).hexdigest(), "robot_id": path.stem,
            "robot_id_source": "calibration file name: LeRobot 0.6.1 writes <calibration_dir>/<robot.id>.json"}


def import_lerobot(dataset, output, *, joint_map=JOINT_MAP, raw_units=False, motor_units=None, calibration=None):
    """Fresh output: raw/ (bit-exact copy) + import.json, plus (a) a Store or (b)/(c) frames.npz."""
    dataset, output = Path(dataset).absolute(), _fresh(output)
    kind = source_kind(dataset)
    if raw_units and kind != "so101_follower":
        raise ValueError("--raw-units applies only to hardware (so101_follower) datasets")
    if calibration is not None and kind != "so101_follower":
        raise ValueError("--calibration applies only to hardware (so101_follower) datasets")
    calibration = _calibration(calibration) if calibration is not None else None
    source_sha256 = _hashes(dataset)
    with tempfile.TemporaryDirectory(prefix=".xert-import-", dir=output.parent) as temporary:
        staging = Path(temporary) / "import"
        staging.mkdir()
        raw = staging / "raw"
        shutil.copytree(dataset, raw)
        if _hashes(raw) != source_sha256:
            raise ValueError("Raw copy differs from the source")  # Everything below reads only the copy.
        info = json.loads((raw / "meta/info.json").read_text("utf-8"))
        record = {"version": 1, "source_kind": kind, "derived": False, "source": str(dataset),
                  "source_sha256": source_sha256, "fps": info["fps"], "lerobot_robot_type": info["robot_type"],
                  "raw": "raw/: bit-exact copy of the source dataset (hashes in source_sha256)"}
        if calibration:
            record["calibration"] = calibration
        if kind == "xert_export":
            record.update(_restore_export(raw, staging))
        else:
            columns, arrays, meta = (_recording(raw, info) if kind == "xert_recording"
                                     else _hardware(raw, info, joint_map, raw_units, motor_units))
            np.savez(staging / "frames.npz", episode_index=columns["episode_index"],
                     frame_index=columns["frame_index"], timestamp=columns["timestamp"], **arrays)
            features = set(info["features"]) - {"observation.state", "action", "observation.environment_state",
                                                 "episode_index", "frame_index", "timestamp", "index", "task_index"}
            record.update(meta, absent=ABSENT, frames=len(columns["frame_index"]),
                          episodes=len(np.unique(columns["episode_index"])), not_imported=sorted(features),
                          timestamp="LeRobot timestamp, restarting at 0 for every episode",
                          grouping="consumers group by episode_index and order by frame_index; nothing spans episodes")
        atomic_json(staging / "import.json", record)
        _publish(staging, output)
    return record


def to_store(source, output):
    """Explicit DERIVED Store episodes from a radian frame-pair import, one per source episode.

    Per episode of T_e frames: T_e-1 transitions whose successors exist, actions normalized to [-1,1] by the
    spec position limits (out-of-limit refused), reward 0 (not measured), the last transition truncated
    (terminated stays false), so no transition ever crosses an episode boundary.
    """
    from xert.spec import load_spec
    source, output = Path(source).absolute(), _fresh(output)
    record = json.loads((source / "import.json").read_text("utf-8"))
    if record.get("derived") is not False or record.get("source_kind") not in ("xert_recording", "so101_follower"):
        raise ValueError("to-store takes a frame-pair import from import-lerobot (paths b/c) only")
    if record["units"] != "radians":
        raise ValueError(f"to-store refuses {record['units']!r} values; import in radians through a joint map")
    spec_path = ROOT / f"xert/spec/{record['embodiment']}.yaml"
    spec_bytes = spec_path.read_bytes()
    if hashlib.sha256(spec_bytes).hexdigest() != record["spec_sha256"]:
        raise ValueError("The current spec differs from the one the import was recorded/converted with")
    limits = load_spec(spec_path, contents=spec_bytes).position_limits
    with np.load(source / "frames.npz", allow_pickle=False) as frames:
        frames = dict(frames)
    # Owner decision: observations are the joint state, then environment_state (named columns, recorded order)
    # when the import has it (path b); path (c) hardware imports stay joint-only.
    environment = "environment_state" in frames
    names = record["columns"]["state"] + (record["columns"]["environment_state"] if environment else [])
    with tempfile.TemporaryDirectory(prefix=".xert-to-store-", dir=output.parent) as temporary:
        staging = Path(temporary) / "store"
        store, ids = Store(staging / "dataset"), []
        for episode in np.unique(frames["episode_index"]):
            rows = frames["episode_index"] == episode
            state, action = frames["state"][rows], frames["action"][rows]  # Import already sorted and checked rows.
            if environment:
                state = np.concatenate([state, frames["environment_state"][rows]], axis=1)
            if len(state) < 2:
                raise ValueError(f"Episode {episode} has one frame; a transition needs a successor")
            targets = action[:-1].astype(np.float64)  # The last action has no recorded successor.
            # Same 1e-6 rad slack the recorder allows for float32 targets clamped at a limit.
            if np.any(targets < limits[:, 0] - 1e-6) or np.any(targets > limits[:, 1] + 1e-6):
                raise ValueError(f"Episode {episode}: actions outside spec position limits; refused, not clipped")
            normalized = 2 * (targets - limits[:, 0]) / (limits[:, 1] - limits[:, 0]) - 1
            saved = store.save_episode(
                {"observations": state, "actions": normalized, "rewards": np.zeros(len(normalized)),
                 "terminated": np.zeros(len(normalized), dtype=bool)},
                {"source": "derived_frame_pairs", "derived": True, "source_kind": record["source_kind"],
                 "import": str(source), "import_sha256": sha256(source / "import.json"),
                 "frames_sha256": sha256(source / "frames.npz"), "episode_index": int(episode),
                 "observation_names": names, "action_names": record["columns"]["action"],
                 "observation_contents": "joint state + environment_state" if environment else "joint state only",
                 "action_convention": "absolute joint position targets normalized to [-1,1] by spec position_limits",
                 "rewards": "0 placeholder, not measured",
                 "terminated": "false everywhere: each episode's last transition is truncated, not terminal"})
            ids.append(saved["id"])
        atomic_json(staging / "collection.json", {"episode_ids": ids, "derived": True, "source_import": str(source)})
        _publish(staging, output)
    return {"episodes": len(ids), "episode_ids": ids}


def export_lerobot(episodes, output, *, task, cameras=False, robot_type="xlerobot_left"):
    """NPZ collection.json (Store in its sibling dataset/) -> P2.1 export; runtime recording -> copy.

    --cameras then runs P2.3 render-cameras on that result, so it works exactly where render-cameras works.
    """
    from xert.data.cameras import render_cameras
    from xert.data.lerobot_export import export_episodes
    episodes, output = Path(episodes).absolute(), _fresh(output)
    if episodes.is_file():
        ids = json.loads(episodes.read_text("utf-8"))["episode_ids"]
        store = Store(episodes.parent / "dataset")
        if not cameras:
            return export_episodes(store, ids, output, task=task, robot_type=robot_type)
        with tempfile.TemporaryDirectory(prefix=".xert-export-", dir=output.parent) as temporary:
            plain = Path(temporary) / "export"
            export_episodes(store, ids, plain, task=task, robot_type=robot_type)
            return render_cameras(plain, output)
    recordings = [episodes] if (episodes / "meta/info.json").exists() else sorted(episodes.glob("episode_*"))
    if not recordings or not all((path / "xert_recording.json").exists() for path in recordings):
        raise ValueError("--episodes must be an NPZ collection.json, a runtime recording or a session of them")
    recorded = {json.loads((path / "xert_recording.json").read_text("utf-8"))["task"] for path in recordings}
    if recorded != {task}:
        raise ValueError(f"Recorded task {sorted(recorded)} differs from --task; recordings are not relabelled")
    if cameras:
        return render_cameras(episodes, output)
    with tempfile.TemporaryDirectory(prefix=".xert-export-", dir=output.parent) as temporary:
        staging = Path(temporary) / "export"
        shutil.copytree(episodes, staging)  # Already LeRobot v3: a verified verbatim copy is the export.
        if _hashes(staging) != _hashes(episodes):
            raise ValueError("Copy differs from the source recording")
        _publish(staging, output)
    return {"recordings": len(recordings), "copied": True}


def add_commands(commands):
    parser = commands.add_parser("export-lerobot", help="NPZ collection or runtime recording to LeRobot v3")
    parser.add_argument("--episodes", type=Path, required=True,
                        help="An NPZ collection.json, a runtime recording, or a session of episode_* recordings")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--cameras", action="store_true", help="Add render-cameras images (XLeRobot bench only)")
    parser.add_argument("--robot-type", choices=PROFILES, default="xlerobot_left", help="NPZ export profile")
    parser = commands.add_parser("import-lerobot", help="LeRobot v3 dataset into XERT formats (manifest-dispatched)")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--joint-map", type=Path, default=JOINT_MAP, help="P6 joint map for so101_follower data")
    parser.add_argument("--raw-units", action="store_true", help="Hardware data: keep native units (raw_lerobot)")
    parser.add_argument("--motor-units", choices=["degrees"],
                        help="Required for hardware radians, no default: LeRobot 0.6.1 does not record use_degrees, so "
                             "you assert the arm was recorded with use_degrees=True (body degrees, gripper 0-100)")
    parser.add_argument("--calibration", type=Path,
                        help="Hardware data: the follower's LeRobot calibration <robot.id>.json; records its sha256 "
                             "and the robot id (required later by replay-real --ledger)")
    parser = commands.add_parser("to-store", help="DERIVED Store episodes from a radian frame-pair import")
    parser.add_argument("--frames", type=Path, required=True, help="An import-lerobot output of path (b) or (c)")
    parser.add_argument("--output", type=Path, required=True)


def execute(args):
    if args.command == "export-lerobot":
        result = export_lerobot(args.episodes, args.output, task=args.task, cameras=args.cameras,
                                robot_type=args.robot_type)
    elif args.command == "import-lerobot":
        result = import_lerobot(args.dataset, args.output, joint_map=args.joint_map, raw_units=args.raw_units,
                                motor_units=args.motor_units, calibration=args.calibration)
    else:
        result = to_store(args.frames, args.output)
    print(json.dumps(result))

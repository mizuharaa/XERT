"""Versioned episode records: one NPZ payload and one small JSON manifest.

No pickle, arbitrary client paths, hidden row filtering, or overwrite-on-import.
"""
from __future__ import annotations

import hashlib
import ast
import io
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile
import uuid
import zipfile
import zlib

import numpy as np

SCHEMA_VERSION = 1
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_EXPANDED_BYTES = 128 * 1024 * 1024


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Same-directory replace is atomic; invalid JSON never replaces prior state.
    raw = json.dumps(value, indent=2, allow_nan=False).encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_arrays(arrays: dict) -> dict:
    required = {"observations", "actions", "rewards", "terminated"}
    if set(arrays) != required:
        raise ValueError(f"Episode must contain exactly {sorted(required)}")
    values = {key: np.asarray(value) for key, value in arrays.items()}
    obs, actions = values["observations"], values["actions"]
    if obs.ndim != 2 or actions.ndim != 2:
        raise ValueError("Observations and actions must be rank-2 arrays")
    n = actions.shape[0]
    if not 1 <= n <= 20000 or obs.shape[0] != n + 1:
        raise ValueError("Expected T actions and T+1 observations, 1 <= T <= 20000")
    if not 1 <= obs.shape[1] <= 1024 or not 1 <= actions.shape[1] <= 128:
        raise ValueError("Unsupported observation/action dimensions")
    for name, value in values.items():
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite numeric data")
    if np.any(actions < -1.00001) or np.any(actions > 1.00001):
        raise ValueError("Actions must use normalized absolute targets in [-1,1]")
    if values["rewards"].shape != (n,) or values["terminated"].shape != (n,):
        raise ValueError("Rewards and terminated must have shape (T,)")
    terminal = values["terminated"]
    if not np.isin(terminal, [0, 1]).all() or np.any(terminal[:-1]):
        raise ValueError("Only the final transition may terminate")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = {name: value.astype(np.bool_ if name == "terminated" else np.float32)
                     for name, value in values.items()}
    if any(not np.isfinite(value).all() for value in converted.values()):
        raise ValueError("Episode values must be representable as float32")
    return converted


def decode_npz(raw: bytes) -> dict:
    if len(raw) > MAX_ARCHIVE_BYTES:
        raise ValueError("Episode archive exceeds 32 MiB")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = archive.infolist()
            names = {"observations.npy", "actions.npy", "rewards.npy", "terminated.npy"}
            if (len(members) != 4 or {x.filename for x in members} != names
                    or sum(x.file_size for x in members) > MAX_EXPANDED_BYTES):
                raise ValueError("Unexpected or oversized expanded episode archive")
            arrays = {}
            for member in members:
                if member.flag_bits & 1 or member.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                    raise ValueError("Only unencrypted stored/deflated NPZ members are accepted")
                with archive.open(member) as stream:
                    magic = stream.read(8)
                    if magic[:6] != b"\x93NUMPY" or magic[6:] not in {b"\x01\x00", b"\x02\x00", b"\x03\x00"}:
                        raise ValueError("Unsupported NPY header")
                    width = 2 if magic[6] == 1 else 4
                    length_bytes = stream.read(width)
                    if len(length_bytes) != width:
                        raise ValueError("Truncated NPY header")
                    header_length = struct.unpack("<H" if width == 2 else "<I", length_bytes)[0]
                    if header_length > 1024:
                        raise ValueError("NPY header exceeds 1024 bytes")
                    text = stream.read(header_length)
                    if len(text) != header_length:
                        raise ValueError("Truncated NPY header")
                    header = ast.literal_eval(text.decode("utf-8" if magic[6] == 3 else "latin1"))
                    if not isinstance(header, dict) or set(header) != {"descr", "fortran_order", "shape"}:
                        raise ValueError("Invalid NPY header fields")
                    if not isinstance(header["descr"], str) or type(header["fortran_order"]) is not bool:
                        raise ValueError("Only scalar numeric NPY arrays are accepted")
                    dtype = np.dtype(header["descr"])
                    shape = header["shape"]
                    if (dtype.kind not in "biuf" or not 1 <= dtype.itemsize <= 8
                            or not isinstance(shape, tuple) or not 1 <= len(shape) <= 2
                            or any(type(size) is not int or size < 1 for size in shape)):
                        raise ValueError("Invalid NPY dtype or shape")
                    name = member.filename[:-4]
                    if name in {"observations", "actions"}:
                        if len(shape) != 2 or shape[0] > 20001 or shape[1] > (1024 if name == "observations" else 128):
                            raise ValueError("Oversized observation/action NPY shape")
                    elif len(shape) != 1 or shape[0] > 20000:
                        raise ValueError("Oversized rewards/terminated NPY shape")
                    size = math.prod(shape) * dtype.itemsize
                    if 8 + width + header_length + size != member.file_size or size > MAX_EXPANDED_BYTES:
                        raise ValueError("NPY shape does not match its bounded payload size")
                    payload = stream.read(size + 1)
                    if len(payload) != size:
                        raise ValueError("Truncated or trailing NPY data")
                    arrays[name] = np.frombuffer(payload, dtype=dtype).reshape(shape, order="F" if header["fortran_order"] else "C")
            return validate_arrays(arrays)
    except (OSError, zipfile.BadZipFile, zlib.error, EOFError, SyntaxError, TypeError) as exc:
        raise ValueError("Invalid NPZ episode archive") from exc


def _load_archive(path: Path, expected_hash: str) -> dict:
    # Hash and decode the same bounded bytes, even if another writer replaces the path.
    try:
        with path.open('rb') as stream:
            raw = stream.read(MAX_ARCHIVE_BYTES + 1)
    except OSError as exc:
        raise ValueError('Episode payload is unavailable') from exc
    if len(raw) > MAX_ARCHIVE_BYTES:
        raise ValueError('Episode archive exceeds 32 MiB')
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError('Episode checksum mismatch')
    return decode_npz(raw)


class Store:
    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, kind: str, identifier: str) -> Path:
        if kind not in {"episodes", "jobs", "models", "evaluations"}:
            raise ValueError("Unknown artifact kind")
        if not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise ValueError("Artifact ID must be a 32-character lowercase UUID hex string")
        return self.root / kind / identifier

    def create(self, kind: str) -> tuple[str, Path]:
        identifier = uuid.uuid4().hex
        path = self.path(kind, identifier)
        path.mkdir(parents=True)
        return identifier, path

    def read(self, kind: str, identifier: str) -> dict:
        return json.loads((self.path(kind, identifier) / "metadata.json").read_text("utf-8"))

    def listing(self, kind: str) -> list[dict]:
        if kind not in {"episodes", "jobs", "models", "evaluations"}:
            raise ValueError("Unknown artifact kind")
        results = []
        filename = "pipeline.json" if kind == "models" else "metadata.json"
        for path in sorted((self.root / kind).glob(f"*/{filename}")):
            try:
                results.append(json.loads(path.read_text("utf-8")))
            except (OSError, ValueError) as exc:
                results.append({"id": path.parent.name, "state": "corrupt", "error": str(exc)})
        return results

    def save_episode(self, arrays: dict, metadata: dict, *, corrective_actions=None) -> dict:
        arrays = validate_arrays(arrays)
        corrective = None
        if corrective_actions is not None:
            if metadata.get('source') != 'scripted_simulation_noise':
                raise ValueError('Corrective labels require noisy scripted simulation provenance')
            corrective = validate_arrays({**arrays, 'actions': corrective_actions})
            if corrective['actions'].shape != arrays['actions'].shape:
                raise ValueError('Corrective action dimensions differ from executed actions')
        elif metadata.get('source') == 'scripted_simulation_noise':
            raise ValueError('Noisy scripted simulation requires corrective labels')
        identifier, path = self.create("episodes")
        np.savez_compressed(path / "episode.npz", **arrays)
        result = {**metadata, "id": identifier, "schema_version": SCHEMA_VERSION,
                  "steps": len(arrays["actions"]),
                  "observation_dim": arrays["observations"].shape[1],
                  "action_dim": arrays["actions"].shape[1],
                  "sha256": sha256(path / "episode.npz")}
        if corrective is not None:
            # Reuse the bounded NPZ decoder; duplicated states make alignment checkable.
            np.savez_compressed(path / 'corrective.npz', **corrective)
            result.update(training_action_source='expert_correction',
                          corrective_sha256=sha256(path / 'corrective.npz'))
        atomic_json(path / "metadata.json", result)
        return result

    def load_episode(self, identifier: str) -> tuple[dict, dict]:
        metadata = self.read("episodes", identifier)
        path = self.path("episodes", identifier) / "episode.npz"
        arrays = _load_archive(path, metadata.get('sha256'))
        if arrays["observations"].shape[1] != metadata["observation_dim"]:
            raise ValueError("Observation dimension differs from manifest")
        if arrays["actions"].shape[1] != metadata["action_dim"] or len(arrays["actions"]) != metadata["steps"]:
            raise ValueError("Action dimensions or episode length differ from manifest")
        return arrays, metadata

    def load_training_episode(self, identifier: str) -> tuple[dict, dict]:
        arrays, metadata = self.load_episode(identifier)
        corrective = metadata.get('source') == 'scripted_simulation_noise'
        if not corrective:
            if 'corrective_sha256' in metadata or 'training_action_source' in metadata:
                raise ValueError('Corrective label provenance is inconsistent')
            return arrays, metadata
        path = self.path('episodes', identifier) / 'corrective.npz'
        if metadata.get('training_action_source') != 'expert_correction':
            raise ValueError('Missing, oversized, or corrupt corrective labels')
        try:
            labels = _load_archive(path, metadata.get('corrective_sha256'))
        except ValueError as exc:
            raise ValueError('Missing, oversized, or corrupt corrective labels') from exc
        if (labels['actions'].shape != arrays['actions'].shape
                or any(not np.array_equal(labels[key], arrays[key])
                       for key in ('observations', 'rewards', 'terminated'))):
            raise ValueError('Corrective labels are not aligned with recorded transitions')
        return labels, metadata

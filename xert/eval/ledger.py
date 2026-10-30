"""Append-only JSONL evidence, with an OS lock shared by CLI and API writers."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import uuid

REPO_ROOT = Path(__file__).resolve().parents[2]


def ledger_path() -> Path:
    return Path(os.environ.get("XERT_LEDGER_PATH", REPO_ROOT / "results/ledger.jsonl"))


def artifact_path(path: Path) -> str:
    path = path.resolve()
    return path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else str(path)


def source_provenance() -> dict:
    # Windows child processes must not open a console window.
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}

    def git(*args):
        return subprocess.check_output(["git", "--no-replace-objects", "-C", str(REPO_ROOT), *args],
                                       text=True, **options).strip()

    return {"git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain", "--untracked-files=all"))}


def new_row(**fields) -> dict:
    return {"id": str(uuid.uuid4()), "time_utc": datetime.now(timezone.utc).isoformat(), **fields}


def validate_row(row: dict) -> None:
    required = {"id", "time_utc", "git_commit", "slice", "experiment", "backend", "embodiment",
                "metric", "value", "ci95", "n", "seeds", "config_path", "artifact_paths", "notes",
                "git_dirty"}
    if not isinstance(row, dict) or not required.issubset(row):
        raise ValueError("Ledger row is missing required fields")
    uuid.UUID(row["id"])
    stamp = datetime.fromisoformat(row["time_utc"])
    if stamp.utcoffset() is None or stamp.utcoffset().total_seconds() != 0:
        raise ValueError("Ledger time must be UTC")
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", row["git_commit"]):
        raise ValueError("Ledger git_commit must be a full commit hash")
    if type(row["git_dirty"]) is not bool or row["backend"] not in {"mujoco", "so101_hw"}:
        raise ValueError("Invalid ledger backend or git_dirty flag")
    for name in ("slice", "experiment", "embodiment", "metric", "config_path", "notes"):
        if not isinstance(row[name], str) or not row[name].strip():
            raise ValueError(f"Ledger {name} must be nonempty text")
    if type(row["value"]) not in (int, float) or not math.isfinite(row["value"]):
        raise ValueError("Ledger value must be finite")
    if type(row["n"]) is not int or row["n"] < 1:
        raise ValueError("Ledger n must be a positive integer")
    ci = row["ci95"]
    if ci is not None and (not isinstance(ci, list) or len(ci) != 2
                          or any(type(x) not in (int, float) or not math.isfinite(x) for x in ci)
                          or not ci[0] <= row["value"] <= ci[1]):
        raise ValueError("Ledger ci95 must bound the value, or be null when unavailable")
    if row["metric"] == "success_rate" and (not 0 <= row["value"] <= 1
                                             or ci is not None and not 0 <= ci[0] <= ci[1] <= 1):
        raise ValueError("Success rate and its interval must be within [0, 1]")
    seeds = row["seeds"]
    if not isinstance(seeds, dict) or set(seeds) != {"train", "eval"}:
        raise ValueError("Ledger seeds must contain train and eval lists")
    for values in seeds.values():
        if (not isinstance(values, list) or any(type(x) is not int or x < 0 for x in values)
                or len(values) != len(set(values))):
            raise ValueError("Ledger seeds must be distinct nonnegative integers")
    artifacts = row["artifact_paths"]
    if not isinstance(artifacts, list) or not artifacts or any(not isinstance(x, str) or not x for x in artifacts):
        raise ValueError("Ledger artifact_paths must be a nonempty list of paths")
    json.dumps(row, allow_nan=False)


@contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield stream
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _read(stream) -> list[dict]:
    stream.seek(0)
    rows, ids = [], set()
    for number, raw in enumerate(stream, 1):
        try:
            if not raw.endswith(b"\n"):
                raise ValueError("Incomplete final line; refusing to append or repair history")
            row = json.loads(raw)
            validate_row(row)
            if row["id"] in ids:
                raise ValueError("Duplicate row ID")
            ids.add(row["id"])
            rows.append(row)
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError(f"Invalid ledger line {number}: {exc}") from exc
    return rows


def read_rows(path: Path | None = None) -> list[dict]:
    path = Path(path) if path is not None else ledger_path()
    if not path.exists():
        return []
    with _locked(path) as stream:
        return _read(stream)


def append_row(row: dict, path: Path | None = None) -> None:
    validate_row(row)
    raw = (json.dumps(row, allow_nan=False, sort_keys=True) + "\n").encode("utf-8")
    with _locked(Path(path) if path is not None else ledger_path()) as stream:
        # shortcut: full history scan per append; index it if the local ledger becomes large.
        if any(previous["id"] == row["id"] for previous in _read(stream)):
            raise ValueError("Duplicate ledger row ID")
        stream.seek(0, os.SEEK_END)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())

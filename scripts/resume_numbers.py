"""Print every resume-eligible number from the results ledger, each with its row id.

Run from the repo root: python -m scripts.resume_numbers

Eligibility rule. Everything below is read from results/ledger.jsonl (or
XERT_LEDGER_PATH); no number is typed into this file.
1. Rows are append-only, so a correction is a new row that names the old id:
   in `superseded_id`, or in its `notes` after the word "Corrects" or
   "Supersedes" (any case; the P3.4 correction says "Corrects earlier row IDs:").
   A row named that way is superseded and dropped. A bare id mention is not.
2. Of the rows left, keep the latest `time_utc` among rows that agree on every
   field except id, time_utc, value, ci95, n, notes, config_path and the git_*
   and artifact_* fields. Rows that differ in phase, training seeds, condition,
   chunk_size or any field added later are therefore never merged. Each
   dropped older repeat is printed with the id of the row kept instead.
3. Rows measured from a dirty working tree (`git_dirty: true`) are kept and
   labelled DIRTY. Their source cannot be rebuilt from the commit alone, so
   read the row's notes and its slice note before citing one.
4. Each value prints as stored (Python repr, never rounded or combined), next
   to its backend (mujoco = simulation, so101_hw = hardware), n, ci95 and notes.
"""
from datetime import datetime
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xert.eval.ledger import ledger_path, read_rows

ROW_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
CORRECTION = re.compile(r"\b(?:corrects|supersedes)\b([^.]*)", re.IGNORECASE)
NOT_KEY = {"id", "time_utc", "value", "ci95", "n", "notes", "config_path"}
BASE = ("slice", "experiment", "embodiment", "metric")
BACKEND = {"mujoco": "mujoco (simulation)", "so101_hw": "so101_hw (hardware)"}


def key_fields(row: dict) -> dict:
    return {k: v for k, v in row.items()
            if k not in NOT_KEY and not k.startswith(("git_", "artifact_"))}


def eligible(rows: list[dict]) -> tuple[list[dict], dict[str, list[str]], dict[str, str]]:
    """Return (eligible rows in ledger order, {superseded id: ids naming it}, {older repeat id: kept id})."""
    ids = {row["id"] for row in rows}
    superseded = {}
    for row in rows:
        named = {row.get("superseded_id")}
        for span in CORRECTION.findall(row["notes"]):
            named.update(ROW_ID.findall(span))
        for old in named:
            if old in ids and old != row["id"]:
                superseded.setdefault(old, []).append(row["id"])
    groups = {}
    for row in rows:
        if row["id"] not in superseded:
            groups.setdefault(json.dumps(key_fields(row), sort_keys=True), []).append(row)
    kept, repeats = set(), {}
    for group in groups.values():
        latest = max(group, key=lambda r: datetime.fromisoformat(r["time_utc"]))
        kept.add(latest["id"])
        repeats.update({r["id"]: latest["id"] for r in group if r is not latest})
    return [row for row in rows if row["id"] in kept], superseded, repeats


def main():
    rows = read_rows()
    kept, superseded, repeats = eligible(rows)
    print(f"Ledger: {ledger_path()}")
    print(f"{len(rows)} rows read; {len(superseded)} superseded by a correction; "
          f"{len(repeats)} older repeats dropped; {len(kept)} eligible.")
    for old, by in superseded.items():
        print(f"superseded {old} by {', '.join(by)}")
    for old, new in repeats.items():
        print(f"older repeat {old} dropped; kept {new}")
    for row in sorted(kept, key=lambda r: r["slice"]):  # stable: ledger order within a slice
        label = [row[k] for k in BASE]
        label += [f"{k}={v}" for k, v in key_fields(row).items()
                  if k not in (*BASE, "backend") and v is not None and isinstance(v, (str, int, float))]
        if row["seeds"]["train"]:
            label.append(f"train_seeds={row['seeds']['train']}")
        print()
        print("  ".join(label))
        print(f"  value {row['value']!r}  ci95 {row['ci95']!r}  n {row['n']}  "
              f"backend {BACKEND[row['backend']]}  tree {'DIRTY' if row['git_dirty'] else 'clean'}")
        print(f"  id {row['id']}")
        print(f"  notes: {row['notes']}")


if __name__ == "__main__":
    main()

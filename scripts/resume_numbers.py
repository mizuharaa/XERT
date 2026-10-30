"""Print every resume-eligible number from the results ledger, each with its row id.

Run from the repo root: python -m scripts.resume_numbers

Eligibility rule. Everything below is read from results/ledger.jsonl (or
XERT_LEDGER_PATH); no number is typed into this file.
1. Rows are append-only, so a correction is a new row that names the old id,
   either in `superseded_id` or in its `notes` text (the P3.4 correction lists
   old ids in its notes). A row named by any other row is superseded and dropped.
2. Of the rows left, keep the latest `time_utc` per (slice, experiment,
   embodiment, metric, phase, training seeds). Phase and training seeds are in
   the key because one experiment writes one row per sweep phase or training
   seed; without them, two of three training seeds would silently disappear.
3. Rows measured from a dirty working tree (`git_dirty: true`) are kept and
   labelled DIRTY. Their source cannot be rebuilt from the commit alone, so
   read the row's notes and its slice note before citing one.
4. Each value prints exactly as stored, never rounded or combined, next to its
   backend (mujoco = simulation, so101_hw = hardware), n, ci95 and full notes.
"""
from datetime import datetime
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xert.eval.ledger import ledger_path, read_rows

ROW_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
BACKEND = {"mujoco": "mujoco (simulation)", "so101_hw": "so101_hw (hardware)"}


def eligible(rows: list[dict]) -> tuple[list[dict], dict[str, list[str]]]:
    """Return (eligible rows in ledger order, {superseded id: ids of the rows that name it})."""
    ids = {row["id"] for row in rows}
    superseded = {}
    for row in rows:
        for old in {row.get("superseded_id"), *ROW_ID.findall(row["notes"])}:
            if old in ids and old != row["id"]:
                superseded.setdefault(old, []).append(row["id"])
    latest = {}
    for row in rows:
        if row["id"] in superseded:
            continue
        key = (row["slice"], row["experiment"], row["embodiment"], row["metric"],
               row.get("phase"), tuple(row["seeds"]["train"]))
        if key not in latest or (datetime.fromisoformat(row["time_utc"])
                                 > datetime.fromisoformat(latest[key]["time_utc"])):
            latest[key] = row
    kept = {row["id"] for row in latest.values()}
    return [row for row in rows if row["id"] in kept], superseded


def main():
    rows = read_rows()
    kept, superseded = eligible(rows)
    print(f"Ledger: {ledger_path()}")
    print(f"{len(rows)} rows read; {len(superseded)} superseded by a correction; "
          f"{len(rows) - len(superseded) - len(kept)} older repeats dropped; {len(kept)} eligible.")
    for old, by in superseded.items():
        print(f"superseded {old} by {', '.join(by)}")
    for row in sorted(kept, key=lambda r: r["slice"]):  # stable: ledger order within a slice
        label = [row["slice"], row["experiment"], row["embodiment"], row["metric"]]
        if row.get("phase"):
            label.append(f"phase={row['phase']}")
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

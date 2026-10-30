"""P5.3 harness checks on scripted data, temporary gates and temporary ledgers only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from xert.eval import gate_experiment as gx
from xert.eval.ledger import append_row, ledger_path, new_row, read_rows

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "xert/spec/xlerobot_left.yaml"
NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          check=True, capture_output=True, text=True, **NO_WINDOW).stdout.strip()


def committed_gate(directory, thresholds):
    directory.mkdir(parents=True)
    git(directory, "init", "-q")
    gate = directory / "gate.yaml"
    gate.write_text(json.dumps(thresholds))
    git(directory, "add", "gate.yaml")
    git(directory, "commit", "-q", "-m", "gate")
    return gate


def test_b_is_exactly_passing_and_c_is_same_size_seeded():
    passed = [True, False, True, False, False, True, False]
    split = gx.conditions(passed, train_seed=1, subset_seed=0)
    assert split["A"] == list(range(7)) and split["B"] == [0, 2, 5]
    assert len(split["C"]) == 3 and set(split["C"]) <= set(split["A"]) and len(set(split["C"])) == 3
    assert gx.conditions(passed, 1, 0)["C"] == split["C"]  # Same seeds, same control subset.
    assert len({tuple(gx.conditions(passed, seed, 0)["C"]) for seed in range(10)}) > 1


def test_preregistration_refuses_missing_untracked_and_dirty_gate(tmp_path):
    with pytest.raises(ValueError, match="No gate file"):
        gx.preregistered(tmp_path / "missing/gate.yaml")
    gate = committed_gate(tmp_path / "repo", {"frames": {"min": 4}})
    assert gx.preregistered(gate) == (hashlib.sha256(gate.read_bytes()).hexdigest(), git(gate.parent, "rev-parse", "HEAD"))
    gate.write_text(json.dumps({"frames": {"min": 3}}))
    with pytest.raises(ValueError, match="uncommitted changes"):
        gx.preregistered(gate)
    untracked = gate.parent / "other.yaml"
    untracked.write_text("{}")
    with pytest.raises(ValueError, match="not committed"):
        gx.preregistered(untracked)


def test_run_refuses_without_gate_before_writing(tmp_path):
    config = json.loads((ROOT / "configs/gate_experiment.json").read_text())
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**config, "gate_path": str(tmp_path / "gate.yaml")}))
    with pytest.raises(ValueError, match="No gate file"):
        gx.run(path, [tmp_path], tmp_path / "out", tmp_path / "report.md")
    assert not (tmp_path / "out").exists() and not (tmp_path / "report.md").exists()


def rows(tag, wins):
    for condition, by_seed in wins.items():
        for seed, successes in enumerate(by_seed):
            append_row(new_row(
                git_commit="0" * 40, git_dirty=False, slice="P5.3", experiment=tag, backend="mujoco",
                embodiment="xlerobot_left", metric="success_rate", value=successes / 30, ci95=[0.0, 1.0], n=30,
                seeds={"train": [seed], "eval": list(range(10000, 10030))}, config_path="c", artifact_paths=["a"],
                condition=condition, successes=successes, train_episodes=4, gate_path="configs/gate.yaml",
                gate_sha256="f" * 64, gate_commit="1" * 40, notes="test row"))
    return read_rows()


def test_report_is_generated_from_ledger_rows():
    text = gx.report(rows("p5.3_clear", {"A": [5, 6, 4], "B": [28, 29, 30], "C": [6, 5, 5]}), "p5.3_clear")
    assert "| B: gate-passing demos | 87 / 90 | 0.9667 |" in text and "| A: all demos | 15 / 90 | 0.1667 |" in text
    assert "B - A = +0.8000: the gate helped relative to all demos." in text
    assert "B - C = +0.7889: the gate helped" in text and "not only how many" in text
    assert gx.RULE in text and "f" * 64 in text and "1" * 40 in text
    assert ledger_path().parent.parent != ROOT  # conftest's temporary ledger, never the real one.
    text = gx.report(rows("p5.3_close", {"A": [15, 16, 14], "B": [17, 16, 15], "C": [3, 2, 4]}), "p5.3_close")
    assert "the gate made no measurable difference relative to all demos." in text
    assert "the gate helped relative to a random" in text and "not only how many" not in text
    with pytest.raises(ValueError, match="one row per condition"):
        gx.report([row for row in read_rows() if row["condition"] != "C"], "p5.3_clear")


def scripted_session(root, lengths):
    """Teleop on the bench with scripted hands, clock and keys. Labelled scripted: never a demonstration."""
    from scripts.record_scripted_runtime import ScriptedClock, ScriptedSource, scripted_frame
    from xert.backends.bench import BenchBackend
    from xert.retarget.calibrate import CalibratedMapper, create_calibration
    from xert.runtime import teleop
    from xert.spec import load_spec
    profile, clock = ROOT / "configs/teleop/xlerobot_bench.json", ScriptedClock()
    neutral = create_calibration(SPEC, profile, scripted_frame(0, clock()), BenchBackend(load_spec(SPEC), 0))
    keys = iter([key for n in lengths for key in [" ", *[None] * (n - 1), " ", None]] + ["\x1b"])
    teleop.run_session(ScriptedSource(clock), SPEC, root, lambda backend: CalibratedMapper(SPEC, profile, neutral, backend),
                       provenance={"kind": "scripted", "generator": "tests/test_gate_experiment.py"},
                       keys=lambda: next(keys), clock=clock, sleep=clock.sleep)
    return root


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    pytest.importorskip("lerobot")
    return scripted_session(tmp_path_factory.mktemp("teleop") / "session", [2, 3, 4, 5])


def config_with(tmp_path, gate, tiny=False):
    config = json.loads((ROOT / "configs/gate_experiment.json").read_text())
    config.update(gate_path=str(gate), spec_path=str(SPEC))
    if tiny:  # Seconds of training and one rollout per model; not a measurement.
        config.update(training_seeds=[0, 1], evaluation_seeds=[10000])
        config["training"].update(epochs=1, chunk_size=3, batch_size=4)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def test_scripted_provenance_is_refused(session, tmp_path):
    config = config_with(tmp_path, committed_gate(tmp_path / "repo", {"frames": {"min": 4}}))
    with pytest.raises(ValueError, match="Refusing 'scripted' data"):
        gx.run(config, [session], tmp_path / "out", tmp_path / "report.md")
    assert not (tmp_path / "out").exists() and not ledger_path().exists()


def test_end_to_end_tiny_run(session, tmp_path, monkeypatch):
    # Only the test widens the accepted sources; the data stays labelled scripted in every manifest.
    monkeypatch.setattr(gx, "HUMAN_SOURCES", ("scripted",))
    monkeypatch.setattr(gx, "MIN_TRAIN_SEEDS", 2)
    monkeypatch.setattr(gx, "MIN_EVAL_EPISODES", 1)
    gate = committed_gate(tmp_path / "repo", {"frames": {"min": 4}})
    report = tmp_path / "gate_experiment.md"
    gx.run(config_with(tmp_path, gate, tiny=True), [session], tmp_path / "run", report)
    ledger = read_rows()
    assert ledger_path() == tmp_path / "results/ledger.jsonl"
    assert sorted((row["condition"], row["seeds"]["train"][0], row["train_episodes"]) for row in ledger) == [
        ("A", 0, 4), ("A", 1, 4), ("B", 0, 2), ("B", 1, 2), ("C", 0, 2), ("C", 1, 2)]
    assert {row["gate_sha256"] for row in ledger} == {hashlib.sha256(gate.read_bytes()).hexdigest()}
    assert {row["gate_commit"] for row in ledger} == {git(gate.parent, "rev-parse", "HEAD")}
    assert {row["source_kinds"][0] for row in ledger} == {"scripted"}
    evaluation = json.loads((tmp_path / "run/B/seed_0/evaluation.json").read_text())
    assert [Path(path).name for path in evaluation["episodes"]] == ["episode_000002", "episode_000003"]
    metadata = json.loads((tmp_path / "run/A/seed_0/metadata.json").read_text())
    assert metadata["config"]["observation_names"] == [f"{joint}.pos" for joint in (
        "Rotation_L", "Pitch_L", "Elbow_L", "Wrist_Pitch_L", "Wrist_Roll_L", "Jaw_L")]
    assert report.read_text() == gx.report(ledger, "p5.3_run")

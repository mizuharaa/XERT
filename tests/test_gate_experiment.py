"""P5.3 harness checks on scripted data, temporary repositories and temporary ledgers only."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from xert.eval import gate_experiment as gx
from xert.eval import ledger
from xert.eval.ledger import append_row, ledger_path, new_row, read_rows
from xle_workbench.pipeline import wilson

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "xert/spec/xlerobot_left.yaml"
NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
TINY = {"training_seeds": [0, 1], "evaluation_seeds": [10000]}  # Seconds of work; not a measurement.


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          check=True, capture_output=True, text=True, **NO_WINDOW).stdout.strip()


def make_repo(path, gate=None, **overrides):
    """A temporary repo with a committed experiment config (and gate, if given)."""
    config = json.loads((ROOT / "configs/gate_experiment.json").read_text())
    config.update(gate_path="configs/gate.yaml", spec_path=str(SPEC), **overrides)
    if overrides.get("training_seeds") == TINY["training_seeds"]:
        config["training"].update(epochs=1, chunk_size=3, batch_size=4)
    (path / "configs").mkdir(parents=True)
    git(path, "init", "-q")
    (path / "configs/gate_experiment.json").write_text(json.dumps(config))
    if gate is not None:
        (path / "configs/gate.yaml").write_text(json.dumps(gate))
    git(path, "add", "configs")
    git(path, "commit", "-q", "-m", "preregister")
    return path


@pytest.fixture
def repo(tmp_path, monkeypatch):
    def build(gate=None, **overrides):
        path = make_repo(tmp_path / "repo", gate, **overrides)
        monkeypatch.setattr(ledger, "REPO_ROOT", path)
        return path
    return build


def test_b_is_exactly_passing_and_c_is_same_size_seeded():
    passed = [True, False, True, False, False, True, False]
    split = gx.conditions(passed, train_seed=1, subset_seed=0)
    assert split["A"] == list(range(7)) and split["B"] == [0, 2, 5]
    assert len(split["C"]) == 3 and set(split["C"]) <= set(split["A"]) and len(set(split["C"])) == 3
    assert gx.conditions(passed, 1, 0)["C"] == split["C"]  # Same seeds, same control subset.
    assert len({tuple(gx.conditions(passed, seed, 0)["C"]) for seed in range(10)}) > 1


def test_preregistration_refuses_missing_foreign_untracked_dirty_and_hidden_edits(tmp_path, repo):
    root = repo({"frames": {"min": 4}})
    gate = root / "configs/gate.yaml"
    with pytest.raises(ValueError, match="No missing.yaml"):
        gx.preregistered(root / "missing.yaml")
    foreign = make_repo(tmp_path / "foreign", {"frames": {"min": 4}}) / "configs/gate.yaml"
    with pytest.raises(ValueError, match="outside the repository"):
        gx.preregistered(foreign)
    assert gx.preregistered(gate) == (hashlib.sha256(gate.read_bytes()).hexdigest(), git(root, "rev-parse", "HEAD"),
                                      gate.read_bytes())
    gate.write_text(json.dumps({"frames": {"min": 3}}))
    with pytest.raises(ValueError, match="uncommitted changes"):
        gx.preregistered(gate)
    git(root, "update-index", "--assume-unchanged", "configs/gate.yaml")  # Hides the edit from git status.
    with pytest.raises(ValueError, match="differs from commit"):
        gx.preregistered(gate)
    (root / "other.yaml").write_text("{}")
    with pytest.raises(ValueError, match="not committed"):
        gx.preregistered(root / "other.yaml")


@pytest.mark.parametrize("case, message", [
    ("no gate", "No gate.yaml"), ("dirty config", "uncommitted changes"),
    ("dirty repo", "Repository has uncommitted changes"), ("reused tag", "already has rows tagged p5.3_out")])
def test_run_refuses_before_writing(tmp_path, repo, case, message):
    root = repo(None if case == "no gate" else {"frames": {"min": 4}})
    config = root / "configs/gate_experiment.json"
    if case == "dirty config":
        config.write_text(config.read_text().replace('"control_subset_seed": 0', '"control_subset_seed": 1'))
    if case == "dirty repo":
        (root / "stray.txt").write_text("x")
    if case == "reused tag":
        rows("p5.3_out", {"A": [1] * 3, "B": [1] * 3, "C": [1] * 3})
    with pytest.raises(ValueError, match=message):
        gx.run(config, [tmp_path], tmp_path / "out", tmp_path / "report.md")
    assert not (tmp_path / "out").exists() and not (tmp_path / "report.md").exists()


def rows(tag, wins):
    for condition, by_seed in wins.items():
        for seed, successes in enumerate(by_seed):
            append_row(new_row(
                git_commit="0" * 40, git_dirty=False, slice="P5.3", experiment=tag, backend="mujoco",
                embodiment="xlerobot_left", metric="success_rate", value=successes / 30, ci95=wilson(successes, 30),
                n=30, seeds={"train": [seed], "eval": list(range(10000, 10030))}, config_path="c", artifact_paths=["a"],
                condition=condition, successes=successes, train_episodes=4, train_frames=400, source_kinds=["webcam"],
                gate_path="configs/gate.yaml", gate_sha256="f" * 64, gate_commit="1" * 40,
                experiment_config_sha256="e" * 64, experiment_config_commit="2" * 40, notes="test row"))
    return read_rows()


def test_report_is_generated_from_ledger_rows():
    text = gx.report(rows("p5.3_clear", {"A": [5, 6, 4], "B": [28, 29, 30], "C": [6, 5, 5]}), "p5.3_clear")
    assert "| B: gate-passing demos | 87 / 90 | 0.9667 |" in text and "| A: all demos | 15 / 90 | 0.1667 |" in text
    assert "B - A = +0.8000: the gate helped relative to all demos." in text
    assert "B - C = +0.7889: the gate helped" in text and "not only how many" in text
    assert gx.RULE in text and "f" * 64 in text and "1" * 40 in text and "e" * 64 in text and "| 400 |" in text
    assert ledger_path().parent.parent != ROOT  # conftest's temporary ledger, never the real one.
    text = gx.report(rows("p5.3_close", {"A": [15, 16, 14], "B": [17, 16, 15], "C": [3, 2, 4]}), "p5.3_close")
    assert "the gate made no measurable difference relative to all demos." in text
    assert "the gate helped relative to a random" in text and "not only how many" not in text
    with pytest.raises(ValueError, match="one row per condition"):
        gx.report([row for row in read_rows() if row["condition"] != "C"], "p5.3_clear")
    with pytest.raises(ValueError, match="fewer than 3 training seeds"):
        gx.report(rows("p5.3_two", {"A": [1, 1], "B": [1, 1], "C": [1, 1]}), "p5.3_two")
    for change, message in (({"ci95": [0.0, 1.0]}, "interval disagree"), ({"n": 29}, "too few trials"),
                            ({"source_kinds": ["scripted"]}, "human demonstrations")):
        edited = [{**row, **change} if row["condition"] == "B" else row for row in read_rows()]
        with pytest.raises(ValueError, match=message):
            gx.report(edited, "p5.3_clear")


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


def edit_manifest(path, **source):
    manifest = json.loads((path / "xert_recording.json").read_text())
    manifest["source"].update(source.pop("source", {}))
    manifest.update(source)
    (path / "xert_recording.json").write_text(json.dumps(manifest))


# Each case widens sources to "scripted" only where needed to reach the check; manifests stay honest otherwise.
@pytest.mark.parametrize("case, message", [
    ("scripted", "Refusing 'scripted' data"),
    ("eval overlap", "overlap demonstration bench seeds"),
    ("gate keeps one", "kept 1 of 4"),
    ("gate keeps all", "kept 4 of 4"),
    ("frame source", r"frame sources \['scripted'\] conflict"),
    ("spec sha", "not recorded on this bench spec")])
def test_run_refuses_bad_data_before_writing(session, tmp_path, repo, monkeypatch, case, message):
    gate = {"gate keeps one": {"frames": {"min": 5}}, "gate keeps all": {"frames": {"min": 1}}}.get(
        case, {"frames": {"min": 4}})
    root = repo(gate, **({"evaluation_seeds": list(range(30))} if case == "eval overlap" else {}))
    copy = shutil.copytree(session, tmp_path / "session")
    monkeypatch.setattr(gx, "conditions", lambda *a: pytest.fail("Reached training: the refusal is missing"))
    if case not in ("scripted", "frame source"):
        monkeypatch.setattr(gx, "HUMAN_SOURCES", ("scripted",))
    if case == "frame source":
        edit_manifest(copy / "episode_000000", source={"kind": "recorded_human"})
    if case == "spec sha":
        edit_manifest(copy / "episode_000000", spec_sha256="0" * 64)
    with pytest.raises(ValueError, match=message):
        gx.run(root / "configs/gate_experiment.json", [copy], tmp_path / "out", tmp_path / "report.md")
    assert not (tmp_path / "out").exists() and not ledger_path().exists()


def test_end_to_end_tiny_run(session, tmp_path, repo, monkeypatch):
    # Only the test widens the accepted sources; the data stays labelled scripted in every manifest.
    monkeypatch.setattr(gx, "HUMAN_SOURCES", ("scripted",))
    monkeypatch.setattr(gx, "MIN_TRAIN_SEEDS", 2)
    monkeypatch.setattr(gx, "MIN_EVAL_EPISODES", 1)
    root = repo({"frames": {"min": 4}}, **TINY)
    gate, config = root / "configs/gate.yaml", root / "configs/gate_experiment.json"
    report = tmp_path / "gate_experiment.md"
    gx.run(config, [session], tmp_path / "run", report)
    rows = read_rows()
    assert ledger_path() == tmp_path / "results/ledger.jsonl"
    assert sorted((row["condition"], row["seeds"]["train"][0], row["train_episodes"]) for row in rows) == [
        ("A", 0, 4), ("A", 1, 4), ("B", 0, 2), ("B", 1, 2), ("C", 0, 2), ("C", 1, 2)]
    frames = {f"episode_00000{index}": length for index, length in enumerate([2, 3, 4, 5])}
    for row in rows:  # Episode lengths 2..5 frames: A has 14, B (the 4- and 5-frame demos) 9.
        used = json.loads((tmp_path / f"run/{row['condition']}/seed_{row['seeds']['train'][0]}/evaluation.json")
                          .read_text())["episodes"]
        assert row["train_frames"] == sum(frames[Path(path).name] for path in used)
    head = git(root, "rev-parse", "HEAD")
    assert {row["gate_sha256"] for row in rows} == {hashlib.sha256(gate.read_bytes()).hexdigest()}
    assert {row["experiment_config_sha256"] for row in rows} == {hashlib.sha256(config.read_bytes()).hexdigest()}
    assert {(row["gate_commit"], row["experiment_config_commit"], row["git_commit"]) for row in rows} == {(head,) * 3}
    assert {row["source_kinds"][0] for row in rows} == {"scripted"} and not any(row["git_dirty"] for row in rows)
    evaluation = json.loads((tmp_path / "run/B/seed_0/evaluation.json").read_text())
    assert [Path(path).name for path in evaluation["episodes"]] == ["episode_000002", "episode_000003"]
    metadata = json.loads((tmp_path / "run/A/seed_0/metadata.json").read_text())
    assert metadata["config"]["observation_names"] == [f"{joint}.pos" for joint in (
        "Rotation_L", "Pitch_L", "Elbow_L", "Wrist_Pitch_L", "Wrist_Roll_L", "Jaw_L")]
    assert report.read_text() == gx.report(rows, "p5.3_run")

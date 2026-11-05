"""Slice J: DAgger plumbing on the pick-and-place bench. Scripted seed data, tiny settings, temporary ledgers."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from xert.eval import dagger, ledger
from xle_workbench import learning, pipeline
from xle_workbench.sim import ManipulationEnv
from xle_workbench.storage import Store, sha256


def arrays(steps=3):
    return {"observations": np.zeros((steps + 1, 4), np.float32), "actions": np.full((steps, 6), .25, np.float32),
            "rewards": np.zeros(steps, np.float32), "terminated": np.zeros(steps, bool)}


def test_dagger_source_shares_the_corrective_label_rule(tmp_path):
    store = Store(tmp_path)
    labels = np.full((3, 6), -.5, np.float32)
    record = store.save_episode(arrays(), {"source": "dagger_policy_rollout"}, corrective_actions=labels)
    assert record["training_action_source"] == "expert_correction"
    executed, _ = store.load_episode(record["id"])
    training, _ = store.load_training_episode(record["id"])
    np.testing.assert_array_equal(executed["actions"], arrays()["actions"])  # episode.npz: executed.
    np.testing.assert_array_equal(training["actions"], labels)  # corrective.npz: expert labels.
    for source in ("dagger_policy_rollout", "scripted_simulation_noise"):  # Never saved without labels.
        with pytest.raises(ValueError, match="require corrective labels"):
            store.save_episode(arrays(), {"source": source})
    with pytest.raises(ValueError, match="Corrective labels require"):  # Other sources still refuse labels.
        store.save_episode(arrays(), {"source": "policy_evaluation"}, corrective_actions=labels)


# ---------------------------------------------------------------- xert dagger init / iterate.

ROOT = Path(__file__).resolve().parents[1]
EVAL = [10000, 10001]  # Two held-out seeds: a harness check, not a measurement.
TINY = {"epochs": 1, "hidden_dim": 16}  # Seconds of training; the real settings are TODO(daniel).


@pytest.fixture(scope="module")
def seed_collection(tmp_path_factory):
    """A tiny P1.2-format collection recollected on the current core (scripts/collect_recovery.py needs 100+)."""
    root = tmp_path_factory.mktemp("seed")
    result = pipeline.collect(Store(root / "dataset"), [100, 101, 102], action_noise_std=.005)
    (root / "collection.json").write_text(json.dumps({"episode_ids": result["episode_ids"]}))
    return root


def install_rules(monkeypatch, rollouts=2):
    """Fixture rules (J-03): a bounded mix that differs from the expert, never stop, tiny settings."""
    calls = []

    def mix(policy_action, expert_action, iteration, step):
        action = np.clip(.5 * (policy_action + expert_action), -1, 1)
        calls.append((policy_action.copy(), expert_action.copy(), iteration, step, action.copy()))
        return action

    monkeypatch.setattr(dagger, "dagger_action", mix)
    monkeypatch.setattr(dagger, "should_stop", lambda history: False)
    monkeypatch.setattr(dagger, "dagger_settings", lambda iteration, aggregate: {"training": dict(TINY),
                                                                                 "rollouts": rollouts})
    return calls


def manifest(run):
    return json.loads((run / "manifest.json").read_text())


def aggregate(run):
    return [ref for entry in manifest(run)["iterations"] for ref in entry["episodes"]]


def files(root):
    return {path: sha256(path) for path in Path(root).rglob("*") if path.is_file()}


def assert_aggregate_is_clean(run):
    refs = aggregate(run)
    assert len({ref["id"] for ref in refs}) == len({ref["seed"] for ref in refs}) == len(refs)
    for ref in refs:  # Every committed payload still loads and matches its manifest hashes.
        _, record = Store(run / ref["store"]).load_training_episode(ref["id"])
        assert (record["sha256"], record["corrective_sha256"]) == (ref["sha256"], ref["corrective_sha256"])


def test_init_records_seed_provenance(seed_collection, tmp_path, monkeypatch):
    run = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["xert", "dagger", "init", "--collection", str(seed_collection),
                                      "--output", str(run), "--eval-seeds", *map(str, EVAL)])
    from xert.__main__ import main
    main()
    record = manifest(run)
    env = ManipulationEnv({"robot": "xlerobot_left"})
    assert record["contract_sha256"] == pipeline.contract(env.metadata()) and record["evaluation_seeds"] == EVAL
    [seed] = record["iterations"]
    assert seed["iteration"] == 0 and [ref["seed"] for ref in seed["episodes"]] == [100, 101, 102]
    store = Store(seed_collection / "dataset")
    for ref in seed["episodes"]:
        stored = store.read("episodes", ref["id"])
        assert (ref["sha256"], ref["corrective_sha256"]) == (stored["sha256"], stored["corrective_sha256"])
    reference = json.loads((run / "iteration_000/seed.json").read_text())
    assert reference["collection_sha256"] == sha256(seed_collection / "collection.json")
    assert reference["episodes"] == seed["episodes"]
    with pytest.raises(FileExistsError):
        dagger.init(seed_collection, run, EVAL)
    assert ledger.read_rows() == []


def test_init_refuses_the_pre_slice_b_collection(tmp_path):
    with pytest.raises(ValueError, match=r"473fff47.*scripts\.collect_recovery"):
        dagger.init(ROOT / "results/p1.2/recovery_integrated", tmp_path / "run", EVAL)
    assert not (tmp_path / "run").exists()


def test_init_refuses_evaluation_seeds_inside_the_demonstrations(seed_collection, tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        dagger.init(seed_collection, tmp_path / "run", [101, 10000])
    assert not (tmp_path / "run").exists()


def test_rollout_seeds_skip_held_out_seeds():
    assert dagger.next_seeds([100, 102], {103, 105}, 3) == [104, 106, 107]


def test_iterate_refuses_until_the_rules_exist(seed_collection, tmp_path):
    run = tmp_path / "run"
    dagger.init(seed_collection, run, EVAL)
    before = (run / "manifest.json").read_bytes()
    with pytest.raises(NotImplementedError, match=r"DAgger mixing rule not chosen \(TODO\(daniel\)\)"):
        dagger.iterate(run)
    assert (run / "manifest.json").read_bytes() == before and not (run / "iteration_001").exists()

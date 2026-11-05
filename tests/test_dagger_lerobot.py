"""Slice J (LeRobot env): P2.1 export and slice C import keep the DAgger streams apart. No ledger rows."""
import json

import numpy as np
import pytest

from xert.eval import dagger
from xle_workbench.storage import Store


class ZeroPolicy:  # Stands in for a checkpoint: the streams, not the policy, are under test.
    def reset(self):
        pass

    def act(self, observation):
        return np.zeros(6, np.float32)


def test_export_and_import_keep_corrective_and_executed_streams_apart(tmp_path, monkeypatch):
    pytest.importorskip("lerobot")
    from xert.data import lerobot_io as io
    monkeypatch.setattr(dagger, "dagger_action", lambda policy, expert, iteration, step: .5 * (policy + expert))
    _, backend = dagger._bench(10000)
    source = Store(tmp_path / "source/dataset")
    records = dagger.collect_rollouts(source, backend.env, ZeroPolicy(), [200, 201],
                                      {"iteration": 1, "attempt": 1, "seeds": [200, 201]})
    ids = [record["id"] for record in records]
    (tmp_path / "source/collection.json").write_text(json.dumps({"episode_ids": ids}))
    assert io.export_lerobot(tmp_path / "source/collection.json", tmp_path / "export",
                             task="Move the block to the tray")["max_abs_error"] == 0
    columns = io._columns(tmp_path / "export")
    for index, identifier in enumerate(ids):
        executed, _ = source.load_episode(identifier)
        labels, _ = source.load_training_episode(identifier)
        rows = columns["episode_index"] == index
        np.testing.assert_array_equal(columns["action"][rows], labels["actions"])  # action = corrective.
        np.testing.assert_array_equal(columns["xert.executed_action"][rows], executed["actions"])
        assert not np.array_equal(labels["actions"], executed["actions"])
    imported = io.import_lerobot(tmp_path / "export", tmp_path / "import")
    restored = Store(tmp_path / "import/dataset")
    for mapping in imported["episodes"]:
        for load in ("load_episode", "load_training_episode"):  # Executed stream, then corrective labels.
            before, old = getattr(source, load)(mapping["source_id"])
            after, new = getattr(restored, load)(mapping["id"])
            for key in before:
                np.testing.assert_array_equal(before[key], after[key])
        assert new["source"] == "dagger_policy_rollout" and new["dagger"] == old["dagger"]
        assert new["training_action_source"] == "expert_correction"

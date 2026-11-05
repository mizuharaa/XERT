"""Slice J: DAgger plumbing on the pick-and-place bench. Scripted seed data, tiny settings, temporary ledgers."""
import numpy as np
import pytest

from xle_workbench.storage import Store


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

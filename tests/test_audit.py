"""Regression tests for findings reproduced from the static review."""
import io
import struct
import zipfile

from fastapi.testclient import TestClient
import numpy as np
import pytest
import torch

from xle_workbench import pipeline
from xle_workbench.api import create_app
from xle_workbench.learning import validate_config
from xle_workbench.storage import Store, decode_npz

torch.set_num_threads(1)


def test_corrupt_deflate_and_unknown_training_config():
    buffer = io.BytesIO()
    np.savez_compressed(buffer, observations=np.zeros((2, 26)), actions=np.zeros((1, 6)),
                        rewards=np.zeros(1), terminated=np.zeros(1, dtype=bool))
    raw = bytearray(buffer.getvalue())
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        offset = archive.infolist()[0].header_offset
    name, extra = struct.unpack_from("<HH", raw, offset + 26)
    raw[offset + 30 + name + extra] = 0xFF  # Invalid DEFLATE BTYPE=11.
    with pytest.raises(ValueError, match="archive"):
        decode_npz(bytes(raw))
    with pytest.raises(ValueError, match="Unknown"):
        validate_config({"epoch": 200})


def test_replay_checks_rewards_and_terminals_and_evaluation_checks_states(tmp_path):
    store = Store(tmp_path)
    ids = pipeline.collect(store, [11, 12], {"horizon": 3})["episode_ids"]
    original, metadata = store.load_episode(ids[0])
    for field, replacement in [("rewards", 1), ("terminated", False)]:
        altered = {key: value.copy() for key, value in original.items()}
        altered[field][-1] = replacement
        record = store.save_episode(altered, metadata)
        assert not pipeline.replay(store, record["id"])["matched"], field
    model = pipeline.train_model(store, ids, {"epochs": 1, "hidden_dim": 16})
    xy = original["observations"][0, 12:14].tolist()
    result = pipeline.evaluate(store, [100], {"horizon": 3, "object_position": xy}, model["id"])
    assert result["initial_states_disjoint_from_demonstrations"] is False
    assert result["held_out"] is False
    assert result["success_rate_ci95"] is None


def test_api_rejects_bad_jobs_and_survives_corrupt_model(tmp_path):
    store = Store(tmp_path)
    identifier, path = store.create("models")
    (path / "pipeline.json").write_text("{", encoding="utf-8")
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/models")
        assert response.status_code == 200
        assert response.json() == [{"id": identifier, "state": "corrupt",
                                    "error": response.json()[0]["error"]}]
        for route, body in [
            ("evaluate", {"seeds": [-1]}),
            ("collect", {"seeds": [2**31]}),
            ("evaluate", {"model_id": "bad"}),
            ("train", {"episode_ids": ["a" * 32, "b" * 32], "config": {"hidden_dim": 18}}),
            ("train", {"episode_ids": ["bad", "worse"]}),
        ]:
            assert client.post(f"/jobs/{route}", json=body).status_code == 422, (route, body)
        assert client.get("/jobs").json() == []

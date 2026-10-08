"""Real simulation/API/storage checks; no mocked robot-success outcomes."""
from concurrent.futures import ThreadPoolExecutor
import io
import json
import threading
import time
import struct
import zipfile

from fastapi.testclient import TestClient
import numpy as np
import pytest
import torch

from xle_workbench.api import create_app
from xle_workbench.jobs import Jobs
from xle_workbench import pipeline
from xle_workbench.storage import Store, atomic_json, decode_npz, validate_arrays

torch.set_num_threads(1)


def test_episode_contract_and_corruption(tmp_path):
    store = Store(tmp_path)
    values = dict(observations=np.zeros((3, 2)), actions=np.zeros((2, 1)),
                  rewards=np.zeros(2), terminated=np.array([False, True]))
    record = store.save_episode(values, {"source": "test"})
    loaded, _ = store.load_episode(record["id"])
    assert loaded["observations"].shape == (3, 2)
    for invalid in ["../outside", "A" * 32, "not-an-id"]:
        with pytest.raises(ValueError):
            store.path("episodes", invalid)
    values["observations"][0, 0] = 1e100
    with pytest.raises(ValueError, match="float32"):
        validate_arrays(values)
    values["observations"][0, 0] = 0
    values["actions"] = np.array([[-9223372036854775808], [0]], dtype=np.int64)
    with pytest.raises(ValueError, match="normalized"):
        validate_arrays(values)
    path = store.path("episodes", record["id"]) / "episode.npz"
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        store.load_episode(record["id"])
    raw = io.BytesIO()
    np.savez(raw, arbitrary=np.zeros(1))
    with pytest.raises(ValueError):
        decode_npz(raw.getvalue())
    # Declared NPY allocation must be rejected before any NumPy array allocation.
    header = b"{'descr': '<f4', 'fortran_order': False, 'shape': (1000000000, 26)}\n"
    malicious = io.BytesIO()
    with zipfile.ZipFile(malicious, "w") as archive:
        archive.writestr("observations.npy", b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header)
        for name in ["actions", "rewards", "terminated"]:
            buffer = io.BytesIO()
            np.save(buffer, np.zeros((1, 1) if name == "actions" else 1))
            archive.writestr(f"{name}.npy", buffer.getvalue())
    with pytest.raises(ValueError, match="Oversized"):
        decode_npz(malicious.getvalue())


def test_real_pipeline_replay_train_evaluate(tmp_path):
    store = Store(tmp_path)
    collection = pipeline.collect(store, [11, 12, 13, 14], {"horizon": 12})
    assert collection["episodes"] == 4
    assert pipeline.replay(store, collection["episode_ids"][0])["matched"]
    # A tiny horizon validates wiring, explicitly not pick-place competence.
    model = pipeline.train_model(store, collection["episode_ids"],
                                 {"epochs": 1, "hidden_dim": 16, "batch_size": 16})
    result = pipeline.evaluate(store, [100], {"horizon": 12}, model["id"])
    assert result["trials"] == 1 and result["hardware"] is False
    with pytest.raises(ValueError, match="overlap"):
        pipeline.evaluate(store, [11], {"horizon": 12}, model["id"])
    with pytest.raises(ValueError, match="contract"):
        pipeline.evaluate(store, [100], {"horizon": 13}, model["id"])
    metadata = store.path("models", model["id"]) / "metadata.json"
    metadata.write_text(metadata.read_text("utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="metadata checksum"):
        pipeline.evaluate(store, [100], {"horizon": 12}, model["id"])
    repeated = pipeline.evaluate(store, [20, 21], {"horizon": 2, "object_position": [.24, -.03]})
    assert repeated["unique_initial_states"] == 1
    assert repeated["success_rate_ci95"] is None


def test_api_sessions_import_jobs_and_origins(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/health").json()["hardware_enabled"] is False
        assert client.get("/health", headers={"origin": "https://example.com"}).status_code == 403
        assert client.get("/health", headers={"host": "attacker.test"}).status_code == 400
        created = client.post("/sessions", json={"seed": 123, "config": {"horizon": 6}})
        assert created.status_code == 201, created.text
        identifier = created.json()["id"]
        assert client.post(f"/sessions/{identifier}/step", json={"action": [0]}).status_code == 400
        assert client.post(f"/sessions/{identifier}/step", json={"expert": True, "repeat": 6}).status_code == 200
        saved = client.post(f"/sessions/{identifier}/save").json()
        assert saved["steps"] == 6
        assert client.post(f"/sessions/{identifier}/step", json={"expert": True}).status_code == 400
        assert client.get(f"/episodes/{saved['id']}/replay").json()["matched"]
        payload = client.get(f"/episodes/{saved['id']}/download").content
        imported = client.post("/episodes/import", params={"seed": 123, "config_json": '{"horizon":6}'},
                               content=payload, headers={"content-type": "application/octet-stream"})
        assert imported.status_code == 200, imported.text
        forged = client.post("/episodes/import", params={"seed": 124, "config_json": '{"horizon":6}'}, content=payload)
        assert forged.status_code == 400
        assert client.delete(f"/sessions/{identifier}").status_code == 200
        job = client.post("/jobs/collect", json={"seeds": [30, 31], "config": {"horizon": 3}})
        assert job.status_code == 202, job.text
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            status = client.get(f"/jobs/{job.json()['id']}").json()
            if status["state"] in {"completed", "failed"}:
                break
            time.sleep(.05)
        assert status["state"] == "completed", status
        assert status["result"]["episodes"] == 2


def test_jobs_cancellation_and_restart(tmp_path):
    store = Store(tmp_path)
    identifier, path = store.create("jobs")
    atomic_json(path / "metadata.json", {"id": identifier, "state": "running"})
    queue = Jobs(store)
    assert store.read("jobs", identifier)["state"] == "interrupted"
    started = threading.Event()

    def cancellable(store, progress, cancelled):
        started.set()
        while not cancelled():
            time.sleep(.005)
        raise InterruptedError("cancelled")

    task = queue.submit("test", cancellable)
    assert started.wait(5)
    assert queue.cancel(task["id"])["cancellation_requested"]
    queue.close()
    assert store.read("jobs", task["id"])["state"] == "cancelled"

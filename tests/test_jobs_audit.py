"""Regression tests for job lifecycle and Windows file sharing."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import threading

import pytest

from xle_workbench.jobs import Jobs
from xle_workbench.storage import Store, atomic_json


def test_submission_status_failure_does_not_reserve_slot(tmp_path, monkeypatch):
    jobs = Jobs(Store(tmp_path))
    called = threading.Event()

    def fail_write(record):
        raise OSError("disk full")

    monkeypatch.setattr(jobs, "write", fail_write)
    try:
        with pytest.raises(OSError, match="disk full"):
            jobs.submit("test", lambda *args, **kwargs: called.set())
        assert not jobs.events
        assert not called.is_set()
    finally:
        jobs.close()


def test_final_status_failure_releases_slot_and_logs(tmp_path, monkeypatch, caplog):
    jobs = Jobs(Store(tmp_path))
    original = jobs.write

    def fail_final(record):
        if record["state"] == "completed":
            raise OSError("disk full")
        original(record)

    monkeypatch.setattr(jobs, "write", fail_final)
    record = jobs.submit("test", lambda store, progress, cancelled: {"done": True})
    jobs.pool.shutdown(wait=True)
    assert not jobs.events
    assert jobs.read(record["id"])["state"] == "running"  # Persistent failure is explicit.
    assert "Could not persist job" in caplog.text
    assert "disk full" in caplog.text
    restarted = Jobs(Store(tmp_path))
    try:
        assert restarted.read(record["id"])["state"] == "interrupted"
    finally:
        restarted.close()


def test_progress_failure_does_not_abort_work(tmp_path, monkeypatch, caplog):
    jobs = Jobs(Store(tmp_path))
    original = jobs.write

    def fail_progress(record):
        if record["state"] == "running" and record["progress"] is not None:
            raise OSError("status unavailable")
        original(record)

    monkeypatch.setattr(jobs, "write", fail_progress)

    def work(store, progress, cancelled):
        progress({"completed": 1})
        return {"done": True}

    record = jobs.submit("test", work)
    jobs.pool.shutdown(wait=True)
    assert jobs.read(record["id"])["state"] == "completed"
    assert jobs.read(record["id"])["result"] == {"done": True}
    assert not jobs.events
    assert "status unavailable" in caplog.text


def test_executor_rejection_releases_slot(tmp_path):
    jobs = Jobs(Store(tmp_path))
    jobs.close()
    with pytest.raises(RuntimeError, match="shutdown"):
        jobs.submit("test", lambda *args, **kwargs: {})
    assert not jobs.events
    assert jobs.listing()[0]["state"] == "failed"


def test_failed_traceback_write_preserves_work_error(tmp_path, monkeypatch, caplog):
    jobs = Jobs(Store(tmp_path))
    original = Path.write_text

    def fail_traceback(path, *args, **kwargs):
        if path.name == "traceback.txt":
            raise OSError("traceback disk full")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_traceback)

    def fail_work(store, progress, cancelled):
        raise ValueError("real work error")

    record = jobs.submit("test", fail_work)
    jobs.pool.shutdown(wait=True)
    saved = jobs.read(record["id"])
    assert saved["state"] == "failed"
    assert saved["error"] == "ValueError: real work error"
    assert not jobs.events
    assert "traceback disk full" in caplog.text


@pytest.mark.parametrize("method", ["read", "listing"])
def test_job_readers_share_writer_lock(tmp_path, monkeypatch, method):
    store = Store(tmp_path)
    jobs = Jobs(store)
    identifier, directory = store.create("jobs")
    record = {"id": identifier, "state": "queued"}
    jobs.write(record)
    opened, release, writing = threading.Event(), threading.Event(), threading.Event()
    original = getattr(store, method)

    def hold_reader(*args):
        # On Windows this handle prevents replacement unless writers wait for it.
        with (directory / "metadata.json").open("rb"):
            opened.set()
            assert release.wait(5)
            return original(*args)

    monkeypatch.setattr(store, method, hold_reader)

    def write_status():
        writing.set()
        jobs.write({**record, "state": "running"})

    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            reader = workers.submit(jobs.read, identifier) if method == "read" else workers.submit(jobs.listing)
            try:
                assert opened.wait(5)
                writer = workers.submit(write_status)
                assert writing.wait(5)
                with pytest.raises(TimeoutError):
                    writer.result(timeout=.05)
            finally:
                release.set()
            reader.result(timeout=5)
            writer.result(timeout=5)
        assert jobs.read(identifier)["state"] == "running"
    finally:
        release.set()
        jobs.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows handle-sharing reproduction")
def test_windows_open_reader_blocks_atomic_replace(tmp_path):
    path = tmp_path / "status.json"
    atomic_json(path, {"state": "queued"})
    with path.open("rb"):
        with pytest.raises(PermissionError):
            atomic_json(path, {"state": "running"})
    atomic_json(path, {"state": "completed"})

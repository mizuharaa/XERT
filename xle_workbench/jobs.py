"""One bounded local worker; durable states make interrupted work visible."""
from concurrent.futures import ThreadPoolExecutor
import logging
import threading
import time
import traceback

from .storage import Store, atomic_json

logger = logging.getLogger(__name__)


class Jobs:
    def __init__(self, store: Store):
        self.store = store
        # ponytail: one CPU/GPU job at a time; use a real queue only for multiple users.
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xle-job")
        self.lock = threading.RLock()
        self.events = {}
        for record in store.listing("jobs"):
            if record.get("state") in {"queued", "running"}:
                record.update(state="interrupted", error="Previous process ended; submit a new job")
                self.write(record)

    def write(self, record):
        with self.lock:
            atomic_json(self.store.path("jobs", record["id"]) / "metadata.json", record)

    def read(self, identifier):
        with self.lock:
            return self.store.read("jobs", identifier)

    def listing(self):
        with self.lock:
            return self.store.listing("jobs")

    def _write_best_effort(self, record):
        try:
            self.write(record)
        except Exception:
            # Failed persistence can leave disk status stale; restart marks it interrupted.
            logger.exception("Could not persist job %s status %s", record["id"], record["state"])

    def submit(self, operation, function, **arguments):
        with self.lock:
            if len(self.events) >= 4:
                raise ValueError("Local queue is full (maximum 4 unfinished jobs)")
            identifier, path = self.store.create("jobs")
            event = threading.Event()
            record = {"id": identifier, "operation": operation, "state": "queued",
                      "submitted_at": time.time(), "parameters": arguments, "progress": None}
            self.write(record)
            self.events[identifier] = event

            def progress(value):
                with self.lock:
                    record["progress"] = value
                    self._write_best_effort(record)

            def execute():
                try:
                    if event.is_set():
                        raise InterruptedError("Cancelled before execution")
                    with self.lock:
                        record.update(state="running", started_at=time.time())
                        self._write_best_effort(record)
                    result = function(self.store, progress=progress, cancelled=event.is_set, **arguments)
                    with self.lock:
                        record.update(state="completed", result=result)
                except InterruptedError as exc:
                    with self.lock:
                        record.update(state="cancelled", error=str(exc))
                except Exception as exc:
                    with self.lock:
                        record.update(state="failed", error=f"{type(exc).__name__}: {exc}")
                    try:
                        (path / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    except OSError:
                        logger.exception("Could not persist job %s traceback", identifier)
                finally:
                    with self.lock:
                        record["finished_at"] = time.time()
                        try:
                            self._write_best_effort(record)
                        finally:
                            self.events.pop(identifier, None)

            try:
                self.pool.submit(execute)
            except Exception as exc:
                self.events.pop(identifier, None)
                record.update(state="failed", error=f"Submission failed: {exc}", finished_at=time.time())
                self._write_best_effort(record)
                raise
            return dict(record)

    def cancel(self, identifier):
        with self.lock:
            record = self.read(identifier)
            event = self.events.get(identifier)
            if event:
                event.set()
            return {"id": identifier, "cancellation_requested": event is not None, "state": record["state"]}

    def close(self):
        with self.lock:
            for event in self.events.values():
                event.set()
        self.pool.shutdown(wait=True, cancel_futures=False)

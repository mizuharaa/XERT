"""Optional actual LeRobot v3 integration; no camera or hardware access."""
import gc
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lerobot")
from scripts.record_scripted_runtime import record_scripted
from xert.backends import MuJoCoBackend
from xert.data.recorder import RuntimeRecorder, TIMES, load_runtime_dataset
from xert.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("robot", ["so101", "g1_upper", "leap_hand"])
def test_runtime_real_v3_alignment_masks_precision(robot, tmp_path):
    spec_path = ROOT / f"xert/spec/{robot}.yaml"
    spec = load_spec(spec_path)
    output = tmp_path / robot
    records = record_scripted(spec_path, ROOT / f"configs/teleop/{robot}.json", output, frames=5)
    dataset = load_runtime_dataset(output)
    manifest = json.loads((output / "xert_recording.json").read_text())
    assert manifest["task_outcome"] is None and manifest["source"]["kind"] == "scripted"
    assert dataset.features["action"]["names"] == [f"{joint}.pos" for joint in spec.joint_names]
    assert len(dataset) == len(records) == 5
    previous = np.asarray(manifest["initial_state"]["q"], dtype=np.float32)
    for i, record in enumerate(records):
        frame = dataset[i]
        np.testing.assert_array_equal(frame["observation.state"].numpy(), previous)
        np.testing.assert_array_equal(frame["action"].numpy(), np.asarray(record["command"], np.float32))
        previous = np.asarray(record["q"], np.float32)
        np.testing.assert_array_equal(frame["xert.next_observation"].numpy(), previous)
        assert not frame["xert.tracking_confidence_present"].item()
        assert frame["xert.times"].numpy().dtype == np.float64
        for key in ("t_capture", "t_landmarks", "t_ik", "t_command", "t_applied"):
            index = TIMES.index(key)
            assert frame["xert.times_present"][index].item() == (record[key] is not None)
            if record[key] is not None:
                assert frame["xert.times"][index].item() == record[key]
        assert json.loads(frame["xert.events_json"]) == record["events"]
        assert frame["xert.clamped"].item() == record["clamped"]
    assert not dataset[0]["xert.times_present"][3].item()
    assert dataset[0]["xert.times_present"][0].item() and dataset[0]["xert.times"][0].item() == 0
    delta = dataset[1]["xert.times"][4] - dataset[1]["xert.times"][3]
    assert 0 < delta.item() < .001  # Lost by upstream default float32 transform at this clock magnitude.
    assert any(row["clamped"] for row in records)
    assert not np.array_equal(dataset[0]["observation.state"], dataset[-1]["xert.next_observation"])
    del dataset
    gc.collect()
    with pytest.raises(FileExistsError):
        record_scripted(spec_path, ROOT / f"configs/teleop/{robot}.json", output)


def make_recorder(tmp_path):
    spec_path = ROOT / "xert/spec/so101.yaml"
    backend = MuJoCoBackend(load_spec(spec_path))
    try:
        return RuntimeRecorder(spec_path, backend.read_state(), tmp_path / "final",
                               task="Failure test", source={"kind": "scripted"})
    finally:
        backend.close()


def test_failure_retains_unpublished_journal_and_closes(tmp_path):
    recorder = make_recorder(tmp_path)
    with pytest.raises(KeyError):
        recorder({})
    assert recorder.closed and not recorder.output.exists()
    assert json.loads((recorder.staging / "xert_recording.json").read_text())["status"] == "aborted"
    assert (recorder.staging / "xert_frames.jsonl").exists()
    with pytest.raises(RuntimeError, match="closed"):
        recorder({})


def test_empty_and_body_exception_do_not_publish(tmp_path):
    recorder = make_recorder(tmp_path)
    with pytest.raises(ValueError, match="empty"):
        recorder.finish()
    assert not recorder.output.exists()
    second = make_recorder(tmp_path)
    with pytest.raises(RuntimeError, match="source failed"):
        with second:
            raise RuntimeError("source failed")
    assert second.closed and not second.output.exists()


def test_constructor_keeps_primary_failure_when_cleanup_also_fails(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from xert.data import recorder
    def cleanup_failure():
        raise RuntimeError("secondary cleanup failure")
    def manifest_failure(*args):
        raise ValueError("primary manifest failure")
    monkeypatch.setattr(LeRobotDataset, "create", lambda **kwargs: SimpleNamespace(finalize=cleanup_failure))
    monkeypatch.setattr(recorder, "atomic_json", manifest_failure)
    with pytest.raises(ValueError, match="primary manifest failure") as caught:
        make_recorder(tmp_path)
    assert any("secondary cleanup failure" in note for note in caught.value.__notes__)
    assert any("staging retained" in note for note in caught.value.__notes__)
    assert not (tmp_path / "final").exists()


@pytest.mark.parametrize("failure", ["writer", "publish_race"])
def test_write_and_publish_failures_preserve_evidence(tmp_path, monkeypatch, failure):
    from xert.runtime.loop import run_loop
    from scripts.record_scripted_runtime import ScriptedClock, ScriptedSource
    from types import SimpleNamespace
    spec_path = ROOT / "xert/spec/so101.yaml"
    backend = MuJoCoBackend(load_spec(spec_path))
    recorder = RuntimeRecorder(spec_path, backend.read_state(), tmp_path / "final", task="Failure test",
                               source={"kind": "scripted"})
    clock = ScriptedClock()
    if failure == "writer":
        def fail(_):
            raise OSError("injected disk failure")
        monkeypatch.setattr(recorder.dataset, "add_frame", fail)
    else:
        recorder.output.mkdir()
        (recorder.output / "owned.txt").write_text("preserve")
    with pytest.raises(OSError):
        with recorder:
            run_loop(ScriptedSource(clock), backend, SimpleNamespace(map=lambda _: backend.spec.rest), .08,
                     clock=clock, sleep=clock.sleep, on_step=recorder)
    assert backend.closed and recorder.closed
    assert (recorder.staging / "xert_frames.jsonl").stat().st_size > 0
    if failure == "publish_race":
        assert list(recorder.output.iterdir()) == [recorder.output / "owned.txt"]
        assert (recorder.output / "owned.txt").read_text() == "preserve"
    else:
        assert not recorder.output.exists()

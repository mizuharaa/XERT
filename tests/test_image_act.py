"""G1 plumbing on a tiny scripted rendered recording (64 px, 5 frames, CPU): no ledger rows, no training result.

The pure helper tests run in the base environment; everything that renders or trains needs LeRobot."""
import json
import os

import numpy as np
import pytest

from xert.eval import image_act
from xert.eval import ledger

EVAL_SEEDS = [10000, 10001]


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    pytest.importorskip("lerobot")
    from test_gate_experiment import scripted_session
    from xert.data.cameras import render_cameras
    root = tmp_path_factory.mktemp("image_act")
    session = scripted_session(root / "session", [3, 4])  # Two episodes, 5 frames, labelled scripted.
    try:
        render_cameras(session, root / "rendered", size=64)
    except Exception as exc:  # No offscreen GL context: skip locally, never open a visible window.
        if os.environ.get("CI"):
            raise
        pytest.skip(f"No offscreen GL context: {exc!r}")
    return session, root / "rendered"


def test_decode_keeps_radians_and_maps_normalized_actions_to_the_joint_ranges():
    low, high = np.array([-1.5, 0.0]), np.array([0.5, 3.0])
    radians = np.array([0.25, 2.0])
    np.testing.assert_array_equal(image_act.decode(radians, "radians", low, high), radians)
    np.testing.assert_allclose(image_act.decode(np.array([-1.0, -1.0]), "normalized", low, high), low)
    np.testing.assert_allclose(image_act.decode(np.array([1.0, 1.0]), "normalized", low, high), high)
    np.testing.assert_allclose(image_act.decode(np.zeros(2), "normalized", low, high), (low + high) / 2)
    with pytest.raises(ValueError, match="Unknown action units"):
        image_act.decode(radians, "degrees", low, high)


def test_ledger_rows_need_human_only_data_and_30_seeds():
    assert image_act._ledger_reason(["webcam"], 30) is None
    assert image_act._ledger_reason(["webcam", "recorded_human"], 31) is None
    assert image_act._ledger_reason(["webcam"], 29) == "no rows: 29 seeds < 30"
    assert image_act._ledger_reason([], 30).startswith("no rows: training data is []")
    assert image_act._ledger_reason(["webcam", "scripted"], 30).startswith("no rows: training data is")


def fixture_settings(dataset_meta):
    """Smoke-test shape (tests/test_cameras.py); seconds of CPU, not a choice and not an experiment."""
    from lerobot.configs.types import NormalizationMode
    return {"act": dict(chunk_size=2, n_action_steps=1, device="cpu", pretrained_backbone_weights=None, dim_model=32,
                        n_heads=2, dim_feedforward=32, n_encoder_layers=1, n_decoder_layers=1, n_vae_encoder_layers=1,
                        latent_dim=4, normalization_mapping={key: NormalizationMode.MEAN_STD
                                                             for key in ("VISUAL", "STATE", "ENV", "ACTION")}),
            "epochs": 2, "batch_size": 2, "image_transforms": None, "validation_episodes": [1]}


def fixture_step(policy, batch, optimizer):
    loss, _ = policy(batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def clean_tree(patch):
    """Rows refuse a dirty tree; tests run on a working copy, so they declare it clean (temporary ledger only)."""
    provenance = {**ledger.source_provenance(), "git_dirty": False}
    patch.setattr(ledger, "source_provenance", lambda: provenance)


@pytest.fixture(scope="module")
def checkpoint(rendered, tmp_path_factory):
    import torch
    with pytest.MonkeyPatch.context() as patch:
        clean_tree(patch)
        patch.setattr(image_act, "image_act_settings", fixture_settings)
        patch.setattr(image_act, "make_optimizer", lambda policy: torch.optim.AdamW(policy.parameters(), lr=1e-4))
        patch.setattr(image_act, "training_step", fixture_step)
        patch.setattr(image_act, "select_epoch", lambda history: (history[-1]["epoch"], False))
        return image_act.train([rendered[1]], tmp_path_factory.mktemp("ckpt") / "model", seed=0)


def test_refuses_a_dataset_without_both_camera_images(rendered, tmp_path):
    session, output = rendered
    with pytest.raises(ValueError, match=r"lacks camera images \['observation.images.scene', 'observation.images.wrist'\]"):
        image_act.train([session], tmp_path / "out")
    info = output / "episode_000000/meta/info.json"
    edited = json.loads(info.read_text())
    del edited["features"]["observation.images.wrist"]
    copy = tmp_path / "one_camera"
    (copy / "meta").mkdir(parents=True)
    (copy / "meta/info.json").write_text(json.dumps(edited))
    with pytest.raises(ValueError, match=r"lacks camera images \['observation.images.wrist'\]"):
        image_act.describe([copy])
    assert not (tmp_path / "out").exists()


def test_train_refuses_until_daniel_chooses_the_settings(rendered, tmp_path):
    with pytest.raises(NotImplementedError, match=r"image-ACT settings not chosen \(TODO\(daniel\)\)"):
        image_act.train([rendered[1]], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_checkpoint_records_train_only_normalization_and_provenance(rendered, checkpoint):
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    metadata = json.loads((checkpoint / image_act.METADATA).read_text())
    assert metadata["robot"] == "xlerobot_left" and metadata["source_kinds"] == ["scripted"]
    assert metadata["training_episodes"] == [0] and metadata["validation_episodes"] == [1]
    assert [row["epoch"] for row in metadata["history"]] == [1, 2] and metadata["selected_epoch"] == 2
    assert all(row["validation_l1"] is not None for row in metadata["history"])
    assert len(metadata["dataset_sha256"]) == 64 and metadata["camera"]["size"] == 64
    assert not (checkpoint / "epochs").exists()
    # The normalizer holds statistics of the training episode alone, not of both episodes.
    train = LeRobotDataset(image_act.REPO_ID, root=rendered[1] / "episode_000000")
    state = torch.stack([train[i]["observation.state"] for i in range(len(train))])
    stats = image_act.train_stats([train], ["observation.state", "observation.images.scene"])
    torch.testing.assert_close(stats["observation.state"]["mean"], state.mean(0))
    assert stats["observation.images.scene"]["mean"].shape == (3, 1, 1)
    saved = {name: path for name, path in ((p.name, p) for p in checkpoint.iterdir())}
    assert {"config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"} <= set(saved)
    del train


def test_live_render_features_equal_the_dataset_features(rendered, checkpoint):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from xert.backends.bench import BenchBackend
    from xert.data.cameras import CameraRenderer
    from xert.spec import load_spec
    metadata = json.loads((checkpoint / image_act.METADATA).read_text())
    stored = LeRobotDataset(image_act.REPO_ID, root=rendered[1] / "episode_000000")[0]
    backend = BenchBackend(load_spec(image_act.SPECS / "xlerobot_left.yaml"), EVAL_SEEDS[0])
    renderer = CameraRenderer(64, "xlerobot_left")
    try:
        live = image_act.live_frame(backend, renderer, metadata["features"], backend.env.metadata()["observation_names"])
    finally:
        renderer.close()
    assert set(live) == set(metadata["features"]) >= set(image_act.IMAGE_KEYS)
    for key, value in live.items():
        assert value.shape == stored[key].shape and value.dtype == stored[key].dtype, key
    assert all(0 <= live[key].min() and live[key].max() <= 1 for key in image_act.IMAGE_KEYS)


def test_evaluate_reports_scripted_data_without_ledger_rows(checkpoint, tmp_path):
    report = image_act.evaluate(checkpoint, EVAL_SEEDS, tmp_path / "report.json")
    assert report["n"] == 2 and [t["seed"] for t in report["trials"]] == EVAL_SEEDS
    assert report["ci95"][0] <= report["success_rate"] <= report["ci95"][1]
    assert report["ledger"].startswith("no rows: training data is ['scripted']") and "ledger_row" not in report
    assert json.loads((tmp_path / "report.json").read_text()) == report
    assert ledger.read_rows() == []  # conftest's temporary ledger, and nothing in it.
    assert report["timing_ms_per_tick"]["ticks"] >= 2
    print(json.dumps(report["timing_ms_per_tick"]))
    with pytest.raises(ValueError, match="overlap the training"):
        image_act.evaluate(checkpoint, [0], tmp_path / "again.json")


def test_human_data_on_enough_seeds_writes_one_row(checkpoint, tmp_path, monkeypatch):
    # Harness path only: scripted data is relabelled human and the seed bar lowered, in a temporary ledger.
    monkeypatch.setattr(image_act, "HUMAN_SOURCES", ("scripted",))
    monkeypatch.setattr(image_act, "MIN_EVAL_EPISODES", 1)
    clean_tree(monkeypatch)
    report = image_act.evaluate(checkpoint, EVAL_SEEDS[:1], tmp_path / "report.json")
    [row] = ledger.read_rows()
    assert row["id"] == report["ledger_row"] and row["slice"] == "G1" and row["experiment"] == "image_act_v1"
    assert row["metric"] == "success_rate" and row["n"] == 1 and row["seeds"] == {"train": [0], "eval": EVAL_SEEDS[:1]}
    assert any(path.endswith("report.json") for path in row["artifact_sha256"])
    # The same checkpoint under the same tag again, even on other seeds, is seed-shopping.
    with pytest.raises(ValueError, match="already has image_act_v1 rows for this checkpoint"):
        image_act.evaluate(checkpoint, EVAL_SEEDS[1:], tmp_path / "again.json")
    assert not (tmp_path / "again.json").exists() and len(ledger.read_rows()) == 1


@pytest.mark.parametrize("dirty", ["tree", "checkpoint"])
def test_rows_refuse_a_dirty_tree_or_checkpoint_before_any_rollout(checkpoint, tmp_path, monkeypatch, dirty):
    monkeypatch.setattr(image_act, "HUMAN_SOURCES", ("scripted",))
    monkeypatch.setattr(image_act, "MIN_EVAL_EPISODES", 1)
    clean_tree(monkeypatch)
    metadata = json.loads((checkpoint / image_act.METADATA).read_text())
    if dirty == "tree":
        monkeypatch.setattr(ledger, "source_provenance", lambda: {"git_commit": metadata["git_commit"], "git_dirty": True})
    copy = tmp_path / "checkpoint"  # Metadata only: the refusal must come before the policy is even loaded.
    copy.mkdir()
    (copy / image_act.METADATA).write_text(json.dumps({**metadata, "git_dirty": dirty == "checkpoint"}))
    with pytest.raises(ValueError, match="uncommitted changes"):
        image_act.evaluate(copy, EVAL_SEEDS[:1], tmp_path / "report.json")
    assert not (tmp_path / "report.json").exists() and ledger.read_rows() == []


# ---------------------------------------------------------------- Done for Daniel (strict xfails).
# These are harness checks, not a performance bar: pre-register your success target before the first evaluate.

@pytest.mark.xfail(strict=True, reason="TODO(daniel): image_act_settings must return complete, valid ACT settings")
def test_daniel_settings_build_a_valid_image_act_config(rendered):
    data = image_act.describe([rendered[1]])
    settings = image_act.image_act_settings(data["meta"])
    config = image_act.policy_config(data["meta"]["features"], settings)
    assert set(config.image_features) == set(image_act.IMAGE_KEYS)
    assert settings["epochs"] >= 1 and settings["batch_size"] >= 1
    json.dumps(settings)


@pytest.mark.xfail(strict=True, reason="TODO(daniel): settings, make_optimizer, training_step and select_epoch "
                                       "must train a checkpoint that evaluate turns into a ledger row on 30 seeds")
def test_daniel_training_gives_a_checkpoint_whose_evaluation_writes_rows(rendered, tmp_path, monkeypatch):
    monkeypatch.setattr(image_act, "HUMAN_SOURCES", ("scripted",))  # Harness check only, temporary ledger.
    model = image_act.train([rendered[1]], tmp_path / "model", seed=0)
    seeds = list(range(10000, 10000 + image_act.MIN_EVAL_EPISODES))
    report = image_act.evaluate(model, seeds, tmp_path / "report.json")
    [row] = ledger.read_rows()
    assert row["n"] == report["n"] == 30 and row["id"] == report["ledger_row"]
    assert np.isfinite(row["value"])

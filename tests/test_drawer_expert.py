"""Done for Daniel: the scripted drawer-opening expert (CORE). Strict xfails until xert/tasks/drawer_expert.py exists."""
from pathlib import Path
import sys

import pytest

from xert.backends.drawer import DrawerBackend
from xert.spec import load_spec
from xert.tasks.drawer_expert import drawer_expert_action

ROOT = Path(__file__).resolve().parents[1]
ROBOTS = ("xlerobot_left", "so101")
# Only "not written yet" (the stub's NotImplementedError) or "criterion unmet" (AssertionError) count as the expected
# failure; an expert that crashes any other way (bad shape, out-of-limit targets, ...) fails the test outright.
NOT_DONE = (NotImplementedError, AssertionError)


def expert_rollout(robot, seed):
    backend = DrawerBackend(load_spec(ROOT / f"xert/spec/{robot}.yaml"), seed)
    while not backend.finished:
        backend.send_targets(drawer_expert_action(backend), None)
    return backend.info


@pytest.mark.parametrize("robot", ROBOTS)
@pytest.mark.xfail(strict=True, raises=NOT_DONE, reason="TODO(daniel): scripted drawer success on seeds 0..8 "
                                                     "(drawer_expert.py)")
def test_scripted_drawer_success_on_seeds_0_to_8(robot):
    infos = [expert_rollout(robot, seed) for seed in range(9)]
    assert all(info["success"] for info in infos), [info["failure_reason"] for info in infos]


@pytest.mark.parametrize("robot", ROBOTS)
@pytest.mark.xfail(strict=True, raises=NOT_DONE, reason="TODO(daniel): scripted drawer success on held-out seeds "
                                                     "10000..10008")
def test_scripted_drawer_success_on_held_out_seeds(robot):
    infos = [expert_rollout(robot, seed) for seed in range(10000, 10009)]
    assert all(info["success"] for info in infos), [info["failure_reason"] for info in infos]


def test_scripted_collect_and_evaluate_refuse_drawer(tmp_path, monkeypatch):
    from xle_workbench import cli
    for command in ("collect", "evaluate"):
        monkeypatch.setattr(sys, "argv", ["xle_workbench", "--data", str(tmp_path / "data"), command, "--task", "drawer"])
        with pytest.raises(NotImplementedError, match=r"drawer expert not tuned \(TODO\(daniel\)\)"):
            cli.main()
    assert not any(path.is_file() for path in tmp_path.rglob("*"))

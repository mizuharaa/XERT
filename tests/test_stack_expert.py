"""Done for Daniel: the scripted stacking expert (CORE). Strict xfails until xert/tasks/stack_expert.py exists."""
from pathlib import Path
import sys

import pytest

from xert.backends.stack import StackBackend
from xert.spec import load_spec
from xert.tasks.stack_expert import stack_expert_action

ROOT = Path(__file__).resolve().parents[1]
ROBOTS = ("xlerobot_left", "so101")


def expert_rollout(robot, seed):
    backend = StackBackend(load_spec(ROOT / f"xert/spec/{robot}.yaml"), seed)
    while not backend.finished:
        backend.send_targets(stack_expert_action(backend), None)
    return backend.info


@pytest.mark.parametrize("robot", ROBOTS)
@pytest.mark.xfail(strict=True, reason="TODO(daniel): scripted stack success on seeds 0..8 (stack_expert.py)")
def test_scripted_stack_success_on_seeds_0_to_8(robot):
    infos = [expert_rollout(robot, seed) for seed in range(9)]
    assert all(info["success"] for info in infos), [info["failure_reason"] for info in infos]


@pytest.mark.parametrize("robot", ROBOTS)
@pytest.mark.xfail(strict=True, reason="TODO(daniel): scripted stack success on held-out seeds 10000..10008")
def test_scripted_stack_success_on_held_out_seeds(robot):
    infos = [expert_rollout(robot, seed) for seed in range(10000, 10009)]
    assert all(info["success"] for info in infos), [info["failure_reason"] for info in infos]


def test_scripted_collect_and_evaluate_refuse_stack(tmp_path, monkeypatch):
    from xle_workbench import cli
    for command in ("collect", "evaluate"):
        monkeypatch.setattr(sys, "argv", ["xle_workbench", "--data", str(tmp_path / "data"), command, "--task", "stack"])
        with pytest.raises(NotImplementedError, match=r"stacking expert not tuned \(TODO\(daniel\)\)"):
            cli.main()
    assert not any(path.is_file() for path in tmp_path.rglob("*"))

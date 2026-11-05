"""Scripted stacking expert. CORE: Daniel writes it (waypoints, IK targets, gains, rate limits).

Until both stubs exist, scripted collect/evaluate on the stack task refuse with NOT_TUNED; teleop
(`xert teleop --task stack`) does not need the expert. Done = the strict-xfail tests in tests/test_stack_expert.py
pass on both bench arms (then delete their xfail markers).
"""
NOT_TUNED = "stacking expert not tuned (TODO(daniel)): implement xert/tasks/stack_expert.py"


def stack_expert_targets(step, block, base_block):
    """TODO(daniel): the waypoint schedule for one control step of a stack episode.

    Inputs:
      step: int, control ticks since reset (StackBackend.task.steps), 0 <= step < horizon (600).
      block: (3,) block centre at reset, world metres (the block to pick).
      base_block: (3,) base_block centre at reset, world metres (the block to stack on).
    Returns:
      (target, grip): target is the (3,) grasp-site goal in world metres; grip is the gripper joint target in
      radians inside the spec's gripper range.
    Invariants:
      - pure: depends only on its inputs (no RNG, no backend state), so replays and ports agree;
      - every target keeps the grasp site above the table top (z > 0.08 m);
      - the schedule ends with the gripper open and the grasp site clear of `block`: StackTask only counts ticks
        with no robot contact on `block` (xert/tasks/stack.py), so a held stack never succeeds.
    """
    raise NotImplementedError(NOT_TUNED)


def stack_expert_action(backend):
    """TODO(daniel): absolute joint targets (radians) for one tick of a StackBackend episode.

    Inputs:
      backend: xert.backends.stack.StackBackend. Read backend.task.steps, backend.data (qpos, site "grasp"),
      backend.qids (joint order = backend.spec.joint_names) and backend.limits.
    Returns:
      float array (6,) inside backend.limits, ready for backend.send_targets(q, None).
    Invariants:
      - never mutates backend.data (the physics state StackTask scores);
      - deterministic for a given backend state, so collect and replay agree;
      - joint-space steps bounded by your rate limit (choose and document it).
    """
    raise NotImplementedError(NOT_TUNED)

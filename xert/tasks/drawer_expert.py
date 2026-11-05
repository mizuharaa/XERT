"""Scripted drawer-opening expert. CORE: Daniel writes it (approach, grasp, pull, release; IK targets, gains).

Until both stubs exist, scripted collect/evaluate on the drawer task refuse with NOT_TUNED; teleop
(`xert teleop --task drawer`) does not need the expert. Done = the strict-xfail tests in
tests/test_drawer_expert.py pass on both bench arms (then delete their xfail markers).
"""
NOT_TUNED = "drawer expert not tuned (TODO(daniel)): implement xert/tasks/drawer_expert.py"


def drawer_expert_targets(step, handle, pull_axis):
    """TODO(daniel): the waypoint schedule for one control step of a drawer episode.

    Inputs:
      step: int, control ticks since reset (DrawerBackend.task.steps), 0 <= step < horizon (600).
      handle: (3,) handle centre at reset, world metres (geom "handle"; the drawer starts 0..0.02 m open).
      pull_axis: (3,) unit world vector the drawer opens along (cabinet +x: xert/tasks/drawer.py PLACEMENT yaw).
    Returns:
      (target, grip): target is the (3,) grasp-site goal in world metres; grip is the gripper joint target in
      radians inside the spec's gripper range.
    Invariants:
      - pure: depends only on its inputs (no RNG, no backend state), so replays and ports agree;
      - every target keeps the grasp site above the table top (z > 0.08 m);
      - the pull moves the handle along pull_axis by at least OPEN_M (0.08 m) minus the start opening, and never
        past DRAWER_TRAVEL (0.12 m): the slide limit would take the load;
      - the schedule ends with the gripper open and clear of the drawer: DrawerTask only counts ticks with no robot
        contact on the drawer or its handle (xert/tasks/drawer.py), so a held drawer never succeeds.
    """
    raise NotImplementedError(NOT_TUNED)


def drawer_expert_action(backend):
    """TODO(daniel): absolute joint targets (radians) for one tick of a DrawerBackend episode.

    Inputs:
      backend: xert.backends.drawer.DrawerBackend. Read backend.task.steps, backend.data (qpos, site "grasp",
      geom "handle", joint "drawer_slide"), backend.qids (joint order = backend.spec.joint_names) and
      backend.limits.
    Returns:
      float array (6,) inside backend.limits, ready for backend.send_targets(q, None).
    Invariants:
      - never mutates backend.data (the physics state DrawerTask scores);
      - deterministic for a given backend state, so collect and replay agree;
      - joint-space steps bounded by your rate limit (choose and document it); approach orientation is yours
        too: scripts/drawer_reach.py measures position reach only.
    """
    raise NotImplementedError(NOT_TUNED)

"""SO-101 scripted pick-and-place expert. CORE: Daniel writes it (waypoints, IK targets, gains).

The XLeRobot expert (cpp/src/ik.cpp `expert_action`) is tuned to that arm's kinematics and refuses
other arms. Until both stubs below are implemented, SO-101 collect/evaluate refuse with NOT_TUNED;
teleop on the SO-101 bench does not need the expert. Done = the strict-xfail tests in
tests/test_so101_expert.py pass (then delete their xfail markers).
"""
NOT_TUNED = "SO-101 scripted expert not tuned (TODO(daniel)): implement xle_workbench/so101_expert.py"


def so101_expert_targets(step, initial_block, tray):
    """TODO(daniel): the waypoint schedule for one control step.

    Inputs:
      step: int, control steps since reset (env.steps), 0 <= step < horizon.
      initial_block: (3,) block centre at reset, world metres (env.initial_block).
      tray: (3,) tray body position, world metres (env.model.body("tray").pos).
    Returns:
      (target, grip): target is the (3,) grasp-site goal in world metres; grip is the gripper joint
      target in radians inside the bench gripper range [-0.17453, 1.7453292].
    Invariants:
      - pure: depends only on its inputs (no RNG, no env state), so replays and ports agree;
      - the schedule starts with a pre-grasp hover above initial_block, and the target stays the
        same for every step of that first phase (tests read the phase length that way);
      - every target keeps the grasp site above the table top (z > 0.08 m).
    """
    raise NotImplementedError(NOT_TUNED)


def so101_expert_action(env):
    """TODO(daniel): one normalized action for an SO-101 ManipulationEnv ({"robot": "so101"}).

    Inputs:
      env: the SO-101 bench. Read env.steps, env.initial_block, env.data (qpos, ctrl), env.site,
      env.qids/env.dids (joint order = env.joints), env.limits; env.ik_data is scratch for IK.
    Returns:
      float array (6,) in [-1, 1], normalized by env.limits exactly as env.normalize does.
    Invariants:
      - never mutates env.data (the physics state the native success rule judges);
      - deterministic for a given env state, so collect and replay agree;
      - its targets come from so101_expert_targets(env.steps, env.initial_block, tray).
    The generalized IK (xert.retarget.ik.solve) is available; the targets, gains and per-step
    rate limits that make it work on SO-101 kinematics are the CORE part.
    """
    raise NotImplementedError(NOT_TUNED)

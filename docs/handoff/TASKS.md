# XERT task tracker (SO-101 track)

Plan: `PLAN.md` (local, not committed). Codex approved it on 2026-11-02 after 3 rounds; findings R1–R8 were all accepted.
Owner rules: "Daniel's portfolio rules" in `~/AGENTS.md` and `~/.claude/CLAUDE.md`. CORE logic is left as `TODO(daniel):`.
Each slice has its own handoff in this folder: what changed, why, how to verify, 3 interview questions, and its TODO(daniel) list.

| Slice | What | Branch | Status | Handoff |
| --- | --- | --- | --- | --- |
| R1 | Ubuntu 24.04 CMake + CI (CTest, pytest, equivalence) | done in P0.3 | Already done: `linux.yml` on every push | [P0.3 note](../learn/P0.3.md) |
| E | ROS 2 Jazzy `xert_ros` scaffold, URDF, RViz, launch, QoS README | `feat/e-xert-ros` | Merged; CI builds and tests on Jazzy | [E-xert-ros.md](E-xert-ros.md) |
| C | `xert export-lerobot` / `import-lerobot`, round trips, camera keys | `feat/c-lerobot-io` | Built, in QA | `C-lerobot-io.md` |
| B | SO-101 as a second bench robot, `--robot` CLI, timing contract | `feat/b-so101-bench` | Building | `B-so101-bench.md` |
| D | Real-vs-sim replay: per-joint plots + RMSE (sysid step 1) | not started | Wave 2, after C | `D-replay-real.md` |
| B2 | SO-101 export schema + cameras | not started | Wave 3, after B and C | `B2-so101-export.md` |

## Daniel's TODO(daniel) queue
These fill in as slices land. Each entry gives file:line and the test that defines "done".
- **E, ROS control loop:**
  - `ros/xert_ros/include/xert_ros/xert_node.hpp`: `control_step()`, `on_command()`, `read_state()`. Their contracts are in the header comments; the line numbers are in [E-xert-ros.md](E-xert-ros.md).
  - Done = the two `DISABLED_` gtests in `ros/xert_ros/test/test_xert_node.cpp` are enabled and pass in `ros.yml`.
- **Owner decision:** pick a project licence. The repo has no LICENSE, and `ros/xert_ros/package.xml` says "TODO".

## Waiting on Daniel (hands-on)
P3.5 live webcam runs, P3.6 clip, P4.1 live check, P4.2 demos, P5.2 thresholds, P5.3 run, demo video, SO-101 bring-up (P6.1/P6.2 per `docs/learn/P6.md`).

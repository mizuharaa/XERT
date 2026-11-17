# Slice M: P6.2 software blockers (a)-(c) and a read-only `xert hw-read`

Branch `feat/m-p6-blockers`, from 34f2193. Software only: **no hardware was read and no motor moved**. The bus
path runs only against a fake bus object in the tests. No ledger row. No CORE change: `braking_speed`, the clamp
equations in `SafetyController.step` and the motor-space law in `SO101HardwareBackend.send_targets` are
untouched. This slice moves state between existing components and adds a read-only bus reader. No new
dependency. The plan is section M of `PLAN.md` (Codex review findings M-01..M-06, approved on round 3).

## What changed and why

Before this slice the `SafetyController` and the hardware backend each kept their own idea of where the arm was
(`docs/learn/P6.md`). Three gaps had to close before any SO-101 motor moves.

| Blocker | Commit | Change |
| --- | --- | --- |
| (a) e-stop never reached the backend | `f2a3b0f` | `SO101HardwareBackend.hold(t)` re-sends the backend's own last motor targets at zero velocity (`status: "hold"` rows). `run_loop` calls it on an e-stop tick when the backend has `hold` and something was sent. If the e-stop comes on the first tick, nothing is sent and that one record gets `"emitted": false`. Sim backends have no `hold`; they still get the frozen `command.q`, which is already a hold for a position servo. The strict xfail at `tests/test_so101_hw_backend.py:182` is now a passing test. |
| (b) the controller rate-limited its own output, not what the hardware got | `e0c0dc6` | `SafetyController.observe(applied)` sets `q = applied` and `velocity = (applied - applied_previous) / dt`. `applied_previous` is a new field (initial pose at start), so velocity is between consecutive applied samples, never against `step`'s requested q. No clock. It records an `observe` event when applied differs from requested by more than 1e-12. `run_loop` calls it only when the state has `measured is False`. |
| (c) both started from the assumed rest pose | `9598718`, `f023bb4`, `e906358` | `xert hw-read` reads `Present_Position` once into `read.json`. `SO101HardwareBackend(..., initial=...)` and `hw-dry-run --seed-from <read.json>` start the backend (and so the controller, which takes the backend's first state) from that read. The mapper's MuJoCo copy gets the seed's `qpos` and `mj_forward` before `create_calibration`, so a neutral hand targets the seed pose. |

**`xert hw-read --port <COMx> --calibration <robot.id>.json --output <read.json> [--joint-map <map>]`**
(`xert/runtime/hw_read.py`, parser in `xert/runtime/cli.py`).
- Bus: a bare LeRobot 0.6.1 `FeetechMotorsBus` with the `so_follower.py:50-60` table (`MOTOR_TABLE` in
  `so101_hw_backend.py`: ids 1-6, `sts3215`, degrees for the body, `range_0_100` for the gripper). Calls, in
  order: `connect()`, `sync_read("Present_Position", normalize=False)`, `disconnect(disable_torque=False)`.
  Between the read and the disconnect it applies LeRobot's own `_normalize` to the same ticks (arithmetic, no bus
  I/O). `docs/learn/P6.md` lists what the handshake reads, with line numbers.
- A joint map is checked before the port opens. Two distinct refusals: TBD fields ("TBD joint-map values ...")
  and a complete map whose `(motor, id, units)` differ from `MOTOR_TABLE` ("incompatible"). A map whose
  `provenance.calibration.sha256` names another calibration file is refused (import-lerobot's rule).
- `read.json`: port, UTC time, `bus_calls`, calibration `{path, sha256, robot_id}` (`lerobot_io._calibration`),
  the motor table, `raw_ticks`, `normalized`, `saturated`, `valid_for_seeding`, `invalid_reason`; with a map also
  `joint_map {path, sha256}`, `radians` (never clipped) and `inside_spec_limits`. No ledger row.
- Saturation (M-06): LeRobot clamps range-mode readings to the calibration interval (`motors_bus.py:867-873`),
  so a gripper tick of 2200 against [1000, 2000] reads 100 and its radians look fine. Any raw tick at or past
  either calibration bound is listed in `saturated` and makes `valid_for_seeding` false.
- Conversion: `rad = (motor - zero_offset) / (sign * units_per_rad)` in float64 (`to_radians`), the same formula
  as `import-lerobot`'s `_hardware`; a test runs `_hardware` on the same values and compares exactly.

**`hw-dry-run --seed-from <read.json>`** (`seed_from` in `xert/runtime/hw_dry_run.py`). The saved radians are
never used: radians are recomputed from `normalized` with the run's own `--map`, which must pass the same
compatibility checks, must have the `joint_map.sha256` the read was converted with (when the read has one) and
must satisfy the calibration-sha rule. The recomputed radians must map back to the saved readings within 1e-9.
Refused: `valid_for_seeding: false`, a pose outside the spec position limits, a different motor table.
`config.json` records `start_pose` (path, sha256, radians, or "spec rest, assumed").

## Where the build differed from the plan

1. **One bus read, not two.** The plan says "raw `sync_read(..., normalize=False)` first, then the normalized
   values" and also requires the call list to be exactly connect, sync_read, disconnect. A second `sync_read`
   would be a second acquisition. The reader does one raw read and normalizes those ticks with LeRobot's own
   `_normalize`. The plan's fallback (normalize by hand when `sync_read` has no `normalize` parameter) is not
   built: the installed 0.6.1 `sync_read` has the keyword-only `normalize` (`motors_bus.py:1133`), and the
   lerobot-only test pins that signature and checks that the fake's normalization equals the real `_normalize`.
   The fake bus records `_normalize` as a call too, so the recorded list is connect, sync_read, `_normalize`,
   disconnect(disable_torque=False).
2. **Map compatibility checks.** The plan cites "the importer's checks" for units, motor names and ids. The
   importer (`lerobot_io.py:146-161`) checks the map's units and the dataset's column names, not the map's
   motor names or ids. `compatible_map` checks all three against `MOTOR_TABLE`, as the plan asks.
3. **Flag name.** `hw-dry-run` keeps its existing `--map`; the plan wrote `--joint-map` for it.
4. **`observe` limit slack.** The plan says "the `__init__` checks". `observe` uses the same checks with a
   1e-10 rad slack, the one `MuJoCoBackend.send_targets` uses, because a command clipped exactly to a limit can
   come back through the motor-unit round trip a few ulps outside it. Precautionary: no such case was measured.
5. **Observe events in the trace.** `observe` returns its events and `run_loop` appends them to that tick's
   `events`, so they reach `trace.jsonl`. Like the plan's event, they carry `kind`, `requested`, `applied` and
   no `t`.
6. **Parser location.** The `hw-read` parser is defined in `cli.py`, so `hw_read.py` is imported only when the
   command runs (the plan's import rule).
7. **No-clamp test map.** The no-clamp case uses 1e3 units/s and 1e5 units/s^2. With 1e6 units/s the backend's
   `braking_speed` bisection step leaves 2.1e-12 rad between requested and applied (measured), above the 1e-12
   event threshold. `q` and `qd` still matched exactly in that case.

## How to verify

Measured on this branch at `e906358` (the last code commit), 2026-11-17, Windows, one pytest process at a time:

| Gate | Result |
| --- | --- |
| Base env `pytest -q`, first 24 test files | 137 passed, 42 skipped, 6 xfailed, 103.8 s |
| Base env `pytest -q`, last 24 test files | 158 passed, 32 skipped, 8 xfailed, 8 subtests passed, 64.0 s |
| Base env whole suite (sum) | 295 passed, 74 skipped, 14 xfailed (slice L: 274 / 73 / 15) |
| `ctest --test-dir build -C Release --output-on-failure` | 2/2 passed |
| `scripts/equivalence_check.py` | 16 seeds identical, max observation difference 0.0 |
| `tests/test_hw_read.py` in the LeRobot 0.6.1 env (`build/venv-teleop` of the main checkout) | 13 passed, 10.8 s |
| The lerobot-only signature test alone, same env | 1 passed, 6.2 s |

The delta from slice L is +21 passed, +1 skipped, -1 xfailed: the un-xfailed e-stop test, 2 more backend tests,
2 `run_loop` observe cases, 4 `observe` unit tests and 12 `test_hw_read.py` tests (its 13th, the signature test,
is the extra skip in the base env).

```
.venv/Scripts/python.exe -m pytest -q tests/test_so101_hw_backend.py tests/test_safety.py tests/test_hw_read.py tests/test_runtime.py tests/test_hw_env_check.py
.venv/Scripts/python.exe scripts/equivalence_check.py
<lerobot env>/python.exe -m pytest -q tests/test_hw_read.py   # from this checkout
```

What the tests pin:
- (a) `test_estop_holds_last_hardware_command` (formerly strict xfail): the last two motor targets are equal and
  `motor_velocity` is 0. `test_estop_on_first_tick_emits_no_command`: zero `last_sent` rows, one record with
  `emitted: false`, clean close. `test_hold_before_any_send_raises`.
- (b) `test_controller_tracks_the_applied_hardware_command[clamp|no-clamp]`: after every tick of a `run_loop`,
  `safety.q == backend.q` and `safety.velocity == backend.qd` exactly. `test_observe_velocity_is_between_
  consecutive_applied_samples` (first tick uses the initial pose; no clock touched) and three refusals.
- (c) fake-bus call list, `read.json` fields, no ledger row, motor-set mismatch (port still closed with
  `disable_torque=False`), TBD vs incompatible map (including a complete `range_m100_100` body map), the
  calibration-sha rule, conversion equal to `_hardware`, unclipped out-of-limit radians, the saturated gripper,
  the seed round trip through the CLI, two maps with the same calibration and different `zero_offset`, stale and
  tampered radians, seed refusals, the backend's motor-limit check on the start pose, and the mapper's first
  grasp target equal to the FK of the seed (not of rest).
- Unchanged and green: `test_cli_writes_trace_and_no_ledger_row`, `test_dry_run_never_imports_hardware_modules`,
  `tests/test_hw_env_check.py`, the runtime tests and the equivalence gate.

## Questions an interviewer could ask

1. **Why does the hardware backend hold its own last target instead of the controller's q?** The controller's
   frozen q is where the controller wanted the arm, not where the motor command is. Under a motor-space clamp
   the two differ, so sending the controller's q on e-stop would keep ramping the motor toward it. Holding the
   last sent motor value at zero velocity stops the command where it is.
2. **Why feed the applied command and not the measured position into the controller, and why between
   consecutive applied samples?** The applied command is what the limiter really sent, known exactly and
   without sensor noise or servo lag; feeding measured position would put the servo's tracking lag inside the
   rate limiter, and that is an owner decision for later. Velocity is taken between two applied samples because
   `self.q` on entry already holds `step`'s request: differencing against it gives the clamp error. With
   previous applied 0, requested 0.0048 and applied 0.001 over 0.04 s, that would be -0.095 rad/s instead of
   +0.025 rad/s.
3. **What does the Feetech handshake read and write?** It reads only: a ping per id that returns the model
   number, then the firmware major/minor version registers of every id. `connect()` also sets a host-side packet
   timeout. The only torque write near this path is in `disconnect()`, and it runs only when `disable_torque` is
   true; `hw-read` always passes false.

## Owner decisions

- **Measured-position feedback during motion** (`Present_Position` into the controller): not built.
- **Saturation at the bound.** A tick exactly at `range_min` or `range_max` counts as saturated, for every motor.
  A gripper resting fully closed at its calibration minimum will therefore refuse seeding. Body joints read in
  degrees are not clamped by LeRobot (`motors_bus.py:874-877`), so for them this is a stricter rule than
  LeRobot needs. Keep it, or relax it to "beyond" for the gripper.
- **Opening the bus.** `xert hw-read` is the first XERT command that opens the servo bus. Running it needs
  Daniel's explicit OK and the arm present (TASKS.md stop point).

## TODO(daniel)

None new.

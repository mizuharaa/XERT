# XERT master prompt — pivot from Reach to a cross-embodiment retargeting + data-quality system

Paste this whole file into your coding agent (Claude Code / Codex) at the root of the `XERT` repo.
Work one slice at a time. Start every session with: "Read XERT_MASTER_PROMPT.md, then do slice <ID> only."

---

## 0. Who you are working for, and the rules you never break

You are pairing with Daniel, a Computer Engineering student building XERT as a portfolio project for
robotics internships. He must be able to explain every line in an interview, so you build *with* him,
not *for* him.

**Non-negotiable rules**

1. **No invented numbers. Ever.** Every metric that appears in the README, docs, or anything Daniel
   might put on a resume must come from a script run that appends a row to `results/ledger.jsonl`
   (schema in §6). If a number has not been measured, write `TBD (not measured)`. Never write a
   placeholder like "~95 ms" or "55% → 75%" into any file.
2. **Sim is labelled sim, hardware is labelled hardware.** Every ledger row has `"backend": "mujoco"`
   or `"backend": "so101_hw"`. No sentence anywhere may imply hardware results that came from sim.
3. **Do not break what already works.** Before and after every slice run `pytest -q` and
   `ctest --test-dir build -C Release`. The existing 31 tests, 8 subtests, 2 CTest checks and the
   C++ vs Python equivalence check must stay green. If a slice must change a contract, update the
   test in the same commit and say why in the commit message.
4. **Small commits.** One slice = one branch = one PR-sized change. Conventional commit messages.
5. **Explain as you go.** At the end of every slice, write `docs/learn/<slice-id>.md` (≤ 1 page):
   what changed, why this design, what an interviewer might ask, and the one-sentence answer.
   Use plain language; define every robotics term the first time it appears.
6. **Ask before** adding a heavy dependency, changing the observation/action schema, or touching
   anything that sends commands to real motors.
7. **Hardware safety (Phase 6 only):** never forward simulation actions straight to motors.
   Low speed, low torque limit, human holding the power switch, one joint at a time first.

---

## 1. Where the repo is today (verify this first — do not trust this summary blindly)

Repo: `github.com/mizuharaa/XERT` (renamed from `reach`). README still says "Reach".

- **C++17 core on the MuJoCo C API**, bound with pybind11 (`cpp/src/env.cpp`, `ik.cpp`,
  `bindings.cpp`, `trials.cpp`, `cpp/tests/test_core.cpp`). Damped-least-squares IK expert,
  command-delay queue, 20 physics substeps per control tick, contact-based success logic.
- **Build is Windows-only** (`scripts/build_core.ps1`, VS 2022 Build Tools, CMake ≥ 3.20,
  MuJoCo 3.14.0 wheel DLL).
- **Python package `xle_workbench`**: CLI (`collect`, `train`, `evaluate`, `replay`, `smoke`,
  `export-bench`, `serve`) and a FastAPI HTTP API with sessions, jobs, episodes, models,
  evaluations, benchmark across friction/delay conditions.
- **Robot:** pinned upstream XLeRobot *left arm* kinematic chain (SO-101 geometry) on a
  stationary bench; assets vendored under `assets/xlerobot` with license + checksums.
- **Data:** custom NPZ episodes, simulator-state observations only (no cameras). Not LeRobot format.
- **Models:** behavior-cloning MLP and a state-based ACT-style CVAE transformer.
- **Measured results:** scripted IK expert 9/9 (16/16 after fixes). **Learned policies 0/9 (MLP and
  ACT-style).** This is the most important open problem.
- **Not present:** retargeting from human motion, multiple embodiments, a safety layer, a hardware
  backend, a data-quality gate, LeRobot export, Linux build.

**Slice P0.1 starts by confirming every bullet above against the actual code** and writing the
corrected version to `docs/STATE_OF_REPO.md`.

---

## 2. The target system (what "done" looks like)

XERT = **Cross-Embodiment Retargeting & Learning**. One human motion stream drives several robot
bodies through one middleware, and the recorded teleoperation data is quality-gated before
imitation learning — with a controlled experiment showing whether the gate helps.

```
human motion source ──► retarget (per-robot spec + IK) ──► safety layer ──► backend
 (webcam / keyboard /                                                       ├─ MuJoCo: SO-101, G1 upper body, LEAP hand
  mocap file)                                                               └─ real SO-101 (Phase 6)
                                                                                 │
                         recorder (LeRobot v3) ◄──────────────────────────────────┘
                              │
                     quality gate (pre-registered thresholds)
                              │
             ACT training: all demos vs gated vs random-same-size control
                              │
              held-out evaluation, Wilson 95% CI ──► results/ledger.jsonl
```

### Target layout (add alongside the existing code; do not delete `xle_workbench`)

```
xert/
  spec/          embodiment specs: so101.yaml, g1_upper.yaml, leap_hand.yaml + loader/validator
  sources/       base.py (MotionSource protocol), webcam_mediapipe.py, keyboard.py, motion_file.py
  retarget/      mapping.py (human frame -> robot task targets), ik.py (DLS or mink), calibrate.py
  safety/        limits.py (pos/vel/accel clamps), watchdog.py (stale input -> hold -> rest)
  backends/      base.py (Backend protocol), mujoco_backend.py, so101_hw_backend.py (Phase 6)
  runtime/       loop.py (fixed-rate loop, timestamps at every stage), latency.py
  data/          recorder.py, lerobot_export.py, quality.py (gate metrics + verdict)
  eval/          experiment.py, stats.py (Wilson CI), ledger.py
results/ledger.jsonl
docs/learn/      one explainer per slice
```

### Core interfaces (keep them this small)

```python
class MotionSource(Protocol):
    def read(self) -> MotionFrame | None   # t_capture, landmarks/poses, confidence; None if nothing new

class Backend(Protocol):
    spec: EmbodimentSpec
    def read_state(self) -> JointState      # q, qd, t
    def send_targets(self, q: np.ndarray, t_cmd: float) -> None
    def close(self) -> None
```

An `EmbodimentSpec` YAML holds: model path (MJCF), ordered joint names, position/velocity limits,
end-effector site name(s), retarget mapping (which human keypoint drives which site), IK weights,
control rate, gripper mapping. **Adding a robot must not require changing controller code** — that
is the whole claim of the project, so test it (slice P3.4).

### Tech stack (pin versions in `requirements-lock.txt`)

| Layer | Choice | Notes |
|---|---|---|
| Physics | MuJoCo (keep 3.14.0 already pinned) | MJCF models; C API in C++ core, `mujoco` in Python |
| Robot models | existing XLeRobot left arm (SO-101); MuJoCo Menagerie `unitree_g1`, `leap_hand` | vendor with license + commit pin, same as `assets/xlerobot` |
| IK | start with your C++ DLS IK generalised to any spec; `mink` (MuJoCo differential IK) as the Python fallback/comparison | ask before adding `mink` |
| Human motion | MediaPipe Hands (world landmarks) + OpenCV capture | latest-frame-only capture, timestamp at capture |
| Data format | LeRobotDataset v3.0 (`lerobot >= 0.4`) | SO-101 features: `observation.state` (6), `action` (6), joint order `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` (`.pos`); check names in the installed version, don't hardcode from memory |
| Learning | keep your ACT-style model; add LeRobot's ACT as the reference baseline | same data, same seeds, compare |
| Hardware (P6) | LeRobot SO-101 follower driver (Feetech STS3215 bus) | check the robot type name in the installed lerobot version |
| Stats | Wilson score interval for success rates; ≥ 30 eval episodes per condition, ≥ 3 training seeds | |
| Build | keep Windows script; add CMake presets + Linux CI job | |

---

## 3. Phases and slices

Each slice has: **Goal · Do · Done when · Ledger output**. Never start a slice until the previous
one's "Done when" is true. Estimated effort is for an evening-and-weekend pace.

### Phase 0 — Ground truth and plumbing (≈ 1–2 days)

**P0.1 Repo audit.** Read every file. Write `docs/STATE_OF_REPO.md`: what exists, what is tested,
what the README claims vs what the code does. Rename "Reach" → "XERT" in README title only.
*Done when:* doc exists, tests green, nothing else changed.

**P0.2 Results ledger.** Add `xert/eval/ledger.py` + `results/ledger.jsonl` (schema §6) and a CLI
`xert ledger show`. Make the existing `evaluate` and `benchmark` commands append rows.
*Done when:* re-running the scripted 9-seed eval appends a row reproducing 9/9 with git hash + config.

**P0.3 Cross-platform build.** Add `CMakePresets.json`, a `scripts/build_core.sh`, and a GitHub
Actions job (Ubuntu) that builds the core and runs pytest + ctest. Keep the Windows path working.
*Done when:* CI green on Linux and the Windows script still works.

### Phase 1 — Make learning actually work in sim (≈ 1 week) — **highest priority**

The gate experiment is meaningless while learned policies score 0/9. Fix this first.

**P1.1 Diagnose 0/9.** Run, in order, and record each in `docs/learn/P1.1.md`:
1. Overfit test: train on 1 episode, roll out on that exact seed. Must succeed; if not, the bug is in
   the pipeline (normalisation, action decoding, time obs, chunk execution), not the data.
2. Open-loop replay of the policy's predicted actions vs recorded actions on a training episode
   (plot per-joint error).
3. Check normalisation stats are computed on train split only and applied identically at eval.
4. Check the time/horizon observation the README warns about.
5. Check gripper action (continuous vs open/close threshold).
*Done when:* root cause(s) written down with evidence plots.

**P1.2 More and better demos.** Scale scripted collection to 100–200 episodes and add noise
injection during collection (perturb the expert's executed action, record the expert's *corrective*
label — DART-style) so the policy sees recovery states. Keep the 9-seed and 16-seed sets as held-out.
*Done when:* new collection saved with provenance; dataset stats in ledger.

**P1.3 ACT fixes.** Action chunking with temporal ensembling at inference; chunk size sweep
(e.g. 10/20/50); gripper as a separate binary head if P1.1 pointed there.
*Done when:* learned ACT-style policy success on ≥ 30 held-out seeds is measured (whatever it is),
3 training seeds, Wilson CI, in the ledger. Target is "clearly above 0", not a specific number.

**P1.4 LeRobot ACT baseline.** Export the same data (P2.1 first if needed) and train LeRobot's ACT
on it; evaluate on the same seeds. *Done when:* both models compared in one ledger table.

### Phase 2 — LeRobot v3 data (≈ 2–3 days)

**P2.1 Exporter.** `xert/data/lerobot_export.py`: NPZ episodes → LeRobotDataset v3 with SO-101
feature names, fps, task string, episode index. Round-trip test: export → load with `LeRobotDataset`
→ arrays equal to source within float32 tolerance.

**P2.2 Recorder.** `xert/data/recorder.py` writes v3 directly from the runtime loop, including
per-frame timestamps, source confidence, and a per-step flag when the safety layer clamped anything
(the gate needs these).

**P2.3 Cameras in sim (optional, later).** Render one fixed camera + one wrist camera from MuJoCo into
the dataset so image-conditioned ACT is possible. Ask before doing this; it is expensive.

### Phase 3 — The XERT middleware (≈ 1–2 weeks)

**P3.1 Embodiment spec.** YAML schema + loader + validator (joint names exist in MJCF, limits sane,
sites exist). Write `so101.yaml` from the existing XLeRobot asset. Unit tests for bad specs.

**P3.2 Generalised IK.** Lift the C++ DLS IK out of the task-specific code into a spec-driven
solver (targets = list of (site, position[, orientation], weight)); expose through pybind11.
Keep the old expert working and equivalence tests green. Optional: `mink` comparison in Python.
*Done when:* IK reaches random reachable targets for SO-101 with measured position error
distribution in the ledger.

**P3.3 Safety layer.** Position, velocity and acceleration clamps from the spec; watchdog: stale input
→ hold for `hold_s` → ease to rest over `return_s`; e-stop key. Log every clamp event.
*Done when:* tests inject out-of-range targets and stale streams and assert the clamped output.

**P3.4 Backends + three embodiments.** `Backend` protocol, `MuJoCoBackend`. Vendor Menagerie G1
(upper body, base welded/fixed) and LEAP hand; write their specs. One script drives all three from a
recorded motion file. **Acceptance test:** adding `leap_hand.yaml` requires zero changes under
`xert/retarget`, `xert/safety`, `xert/runtime` (enforce with a test that only loads specs).

**P3.5 Webcam source + runtime loop.** MediaPipe Hands → wrist position/orientation + pinch → SO-101
EE target + gripper; finger curls → LEAP joints; both wrists → G1 hands. Neutral-pose calibration
(`xert calibrate`). Fixed-rate loop with timestamps at capture, landmarks done, IK done, command sent,
sim step applied.
*Done when:* `xert latency --duration 300` writes p50/p95 end-to-end latency per embodiment to the
ledger with `"backend": "mujoco"`.

**P3.6 Retarget quality.** For a recorded human clip, measure target-vs-achieved EE error per robot
(how well each body can follow the human). Ledger rows per embodiment.

### Phase 4 — Human teleop demos in sim (≈ 3–4 days)

**P4.1 Teleop collection UI.** Run the webcam → SO-101 sim loop on the pick-and-place bench; keys to
start/stop/discard an episode; task success auto-detected by the existing contact logic. Record to v3.

**P4.2 Collect.** Daniel records 80–150 real human demos (this is real work; do not synthesise
"human" demos with the scripted expert and do not label scripted data as human).

### Phase 5 — The quality-gate experiment (≈ 1 week) — the headline result

**P5.1 Gate metrics.** Per episode: action jitter (RMS of second difference of joint targets),
timestamp gaps / dropped-frame fraction, mean tracking confidence, safety-clamp count, episode
success, duration. Plot distributions.

**P5.2 Pre-register thresholds.** Write `configs/gate.yaml` with thresholds chosen from the P5.1
distributions *before* any training on gated data. Commit it. Do not tune thresholds after seeing
policy results; if you change them, it is a new experiment with a new ledger tag.

**P5.3 Experiment.** Three training conditions on identical held-out eval seeds:
A = all demos, B = gated demos, C = random subset of A with the same size as B (separates "cleaner
data" from "less data"). ≥ 3 training seeds × ≥ 30 eval episodes per condition. Report success with
Wilson 95% CI and the B-vs-A and B-vs-C differences.
*Done when:* `results/gate_experiment.md` is generated from the ledger (not hand-written) and says
honestly whether the gate helped, hurt, or made no measurable difference.

### Phase 6 — Real SO-101 via XLeRobot (when hardware arrives)

**P6.1 Bring-up.** LeRobot motor setup + calibration. Build a joint map table: sim joint name ↔
motor id, sign, zero offset, limits measured on the real arm. Ask Daniel before any motion.

**P6.2 `SO101HardwareBackend`.** Same `Backend` protocol. Starts read-only (`--dry-run` prints
targets). Then low-speed, low-torque, one joint at a time, human on the power switch.

**P6.3 Sim-to-real tracking.** Same trajectory in sim and on the arm; joint tracking RMS error per
joint, `"backend": "so101_hw"`.

**P6.4 Real teleop + latency.** Webcam → real SO-101; p95 latency to command sent and to measured
motion onset.

**P6.5 Real gate experiment.** Repeat P4–P5 on hardware with real demos and real rollouts
(≥ 20 trials per condition; fewer trials = wider CI, report it).

### Phase 7 — Packaging (≈ 2 days)

README rewritten around the diagram in §2, a 60–90 s demo video (three embodiments + gate result),
`scripts/resume_numbers.py` that prints every resume-eligible number with its ledger row id, and
`docs/learn/INDEX.md` linking all slice explainers for interview prep.

---

## 4. Order of work if time is short

P0.1 → P0.2 → P1.1 → P1.2 → P1.3 → P3.1 → P3.3 → P3.4 → P3.5 → P4 → P5. Everything else is optional.
If only one result can exist before applications, make it **P1.3** (a learned policy that works,
measured) — it turns "0/9" into a real bullet.

## 5. Interview-defensibility checklist (run at the end of each phase)

- Can Daniel draw the data flow from memory?
- Can he explain why DLS IK uses damping and what happens near singularities?
- Can he explain action chunking and why temporal ensembling smooths execution?
- Can he explain why condition C exists in the gate experiment?
- Can he point to the ledger row behind every number in the README?

## 6. Ledger schema (`results/ledger.jsonl`, one JSON object per line, append-only)

```json
{
  "id": "uuid",
  "time_utc": "ISO-8601",
  "git_commit": "sha",
  "slice": "P5.3",
  "experiment": "gate_v1",
  "backend": "mujoco | so101_hw",
  "embodiment": "so101 | g1_upper | leap_hand",
  "metric": "success_rate | latency_p95_ms | tracking_rms_deg | ik_err_mm | ...",
  "value": 0.0,
  "ci95": [0.0, 0.0],
  "n": 30,
  "seeds": {"train": [0, 1, 2], "eval": [10000, 10029]},
  "config_path": "configs/...",
  "artifact_paths": ["data/evaluations/..."],
  "notes": "free text; say what this does NOT show"
}
```

Rows are never edited or deleted; corrections are new rows that reference the old `id`.

---

## 7. First message to send the agent

> Read XERT_MASTER_PROMPT.md fully. Then do slice P0.1 only: audit the repo against §1, write
> docs/STATE_OF_REPO.md, rename the README title to XERT, run pytest and ctest, and stop.
> Tell me what in §1 was wrong.

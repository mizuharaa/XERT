# Data export and runtime review

P2.1 is accepted. P3.5 is a software partial: its tests do not establish live
camera latency, human motion quality or hardware operation.

## Review method

The isolated QA worktree ran the actual Claude Code CLI with repository-scoped
Read, Glob and Grep tools only. Both fixed-commit reviews completed successfully
and reported model `claude-opus-5-5`. Claude neither executed tests nor changed
source. Codex independently inspected findings, ran focused checks and verified
evidence. Raw Claude reports remain local under ignored `build/qa`.

## P2.1: accepted

Claude reviewed implementation `d4cd96f`; Codex also checked evidence commit
`ed5f796`. The real LeRobot tests passed using the isolated LeRobot environment.
An additional failure injection after a partially written export preserved the
original error and removed only the private staging directory. Existing output
directories and source episodes were preserved.

Independent Arrow/NPZ comparison checked every exported transition: pre-action
state, corrective action, executed action, successor state, reward, termination,
episode/frame indices and derived simulation timestamp. This includes the final
successor observation. All published artifact hashes and original source hashes
match the manifest and ledger row
`d56c4f9e-d897-4f97-9574-a60775da2b77`. The measured export is recorded in
[roundtrip.json](../../results/p2.1/legacy_v3/roundtrip.json); it is a data
preservation result, not a learned-policy result.

The legacy profile retains its full simulator state and normalized actions under
the truthful `xlerobot_left` identity. The separate SO-101 fixture tests schema
compatibility only. Corrective-label identity comes from verified provenance,
not from whether the two action arrays happen to differ. Whole-export LeRobot
statistics must not become training normalization without a training-only split.

## P3.5: software partial

Claude reviewed implementation `b668ad7`; Codex reviewed follow-up `30613fb`.
Runtime, safety and mock-camera tests
passed in the isolated vision environment. Independent probes confirmed that
degenerate longitudinal palm geometry is rejected and `xert ledger show` does
not import MediaPipe, OpenCV, LeRobot or PyTorch.

The reviewed loop separates scheduled control time from host-monotonic input
age, preserves original capture timestamps, rejects missed deadlines without
catch-up commands and counts unique valid applied frames only. Empty/no-hand
runs cannot produce a zero-sample latency result. Handedness scores are named as
such; tracking confidence remains unavailable rather than invented.

Claude's proposed left/right label swap relied on the legacy Hands convention.
It was not applied: the pinned
[MediaPipe Tasks source](https://github.com/google-ai-edge/mediapipe/blob/v0.10.35/mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarks_detector_graph.cc#L116-L128)
already uses the changed Tasks label mapping. Actual known-left/known-right
behavior still needs checking with the selected camera.

The follow-up moves device release into its capture worker, preventing concurrent
native read/release. Shutdown waits for a bounded interval and reports blocked
workers without closing resources still in use. A blocking-reader regression
checks that ownership. Calibration is read once; the exact bytes used by the
mapper are hashed and retained in each run's evidence. A source-file mutation
regression checks that snapshot. These fixes received source review and executable
checks by Codex, not a second Claude run.

Palm-size depth is a control proxy coupled to wrist rotation; it can request
unreachable targets before command limiting. This is not metric wrist tracking.
The capture timestamp begins when acquisition returns and excludes unknown
sensor exposure and driver buffering. No camera was opened by QA. Live runs for
each embodiment and their ledger-backed latency measurements remain outstanding.

Focused tests and static reviews are software verification, not portfolio
performance measurements. Full-suite and hosted optional-stack checks belong to
the integration lane and must be reported at their actual commit boundaries.

## P3.5 hardening follow-up

The actual Claude CLI repeated its repository-scoped, read-only review at
`8690df8` and reported model `claude-opus-5-5`. Codex independently reviewed
fixes `1f4f2fb` and `0e00ee6`, integrated together at `8091b16`. No unresolved
software blocker was found in this bounded review. Live-input acceptance remains
partial.

Independent injected-failure probes confirmed that cleanup could replace the
original runtime exception and that capture release failures could escape their
worker without reaching the caller. The fixed loop preserves the primary error,
attaches later cleanup errors and attempts both resource closures. Capture
release failures now reach the shared error channel; detector closure is not
retried after a failed attempt. The isolated vision runtime, safety and
mock-camera suite passed at `1f4f2fb`, including blocked-worker ownership tests.

The calibration follow-up parses and hashes the same captured spec/profile
bytes, retains immutable latency snapshots and records hashes of every reported
artifact. The recorder and scripted recording caller use the same captured-spec
path. Malformed image dimensions and invalid profile values fail before camera
acquisition. CLI cleanup preserves primary errors. The calibration CLI and real
LeRobot recorder tests passed independently at `0e00ee6`, including source-path
mutation and cleanup-failure regressions.

Claude identified conditional sampling as a reporting limitation but suggested
counting rejected stale frames as applied latency. That remedy was rejected:
those targets were not applied. Eligibility remains unique, fresh, valid,
successfully applied frames. Counts and traces must accompany the conditional
percentiles; they do not describe latency for dropped or rejected frames.

Raw Claude output and independent probes remain in ignored `build/qa`. These
follow-up checks produced no camera observations, synthetic latency ledger rows,
human demonstrations or hardware results. Historical measurements retain their
original source commits. Final integration suites and CI are separately owned
by the integration lane.

# XERT interview notes

These notes explain the implemented simulation work. Each slice links its tests
or recorded evidence; a successful simulation check is not a hardware result.

| Slice | Read | Question to practice |
| --- | --- | --- |
| P0.1 | [Repository audit](P0.1.md) | Which parts came from upstream, and what did this project add? |
| P0.2 | [Results ledger](P0.2.md) | How can someone trace a result to its code, configuration and raw episodes? |
| P0.3 | [Cross-platform build](P0.3.md) | How do Python and C++ load the same physics library? |
| P1.1 | [Learning diagnosis](P1.1.md) | Why can a small action-prediction error still produce a failed rollout? |
| P1.2 | [Recovery demonstrations](P1.2.md) | Why must executed noisy actions and corrective training labels be stored separately? |
| P1.3 | [ACT experiment](P1.3.md) | How do overlapping action predictions line up at the current control step? |
| P1.4 | [Official ACT comparison](P1.4.md) | Which differences prevent attributing a result to the architecture alone? |
| P2.1 | [Lossless LeRobot export](P2.1.md) | Why keep the terminal observation and both executed and corrective actions? |
| P2.2 | [Direct runtime recording](P2.2.md) | How do you distinguish a missing timestamp from zero without losing precision? |
| P2.3 | [Offline sim cameras (optional)](P2.3.md) | Why render after recording instead of during teleop? |
| P3.1 | [Embodiment specifications](P3.1.md) | Why must a requested position fit both joint and actuator limits? |
| P3.2 | [Generalized inverse kinematics](P3.2.md) | What does damping do when a robot cannot easily move in the requested direction? |
| P3.3 | [Command safety](P3.3.md) | Why is clipping a position insufficient to limit speed and acceleration? |
| P3.4 | [Shared simulation backend](P3.4.md) | What changes when adding a robot, and which controller code stays shared? |
| P3.5 | [Webcam runtime (partial)](P3.5.md) | Why distinguish host timing, simulation timing, and camera exposure? |
| P3.6 | [Retarget quality (partial)](P3.6.md) | Why score LEAP at forward-kinematics fingertips instead of human fingertips? |
| P4.1 | [Bench teleop collection (partial)](P4.1.md) | Why reuse the native success check instead of writing it again in Python? |
| P5.1 | [Gate metrics (partial)](P5.1.md) | Why pre-register thresholds before training? |
| P5.3 | [Gate experiment harness (partial)](P5.3.md) | Why does the same-size random subset (condition C) exist? |
| P6 | [SO-101 dry-run backend (prep; no hardware)](P6.md) | Why clamp again in the hardware backend when the safety controller already limits speed and acceleration? |
| P7 | [Numbers only from the ledger (software; demo video pending)](P7.md) | Why does the README point to the ledger instead of quoting results? |

Use [the acceptance tracker](../MVP_PROGRESS.md) for current completion status.
The master prompt's human-data quality experiment and hardware phases have
separate acceptance criteria; these notes do not establish those results.

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
| P1.4 | [Official ACT comparison (in progress)](P1.4.md) | Which differences prevent attributing a result to the architecture alone? |
| P2.1 | [Lossless LeRobot export](P2.1.md) | Why keep the terminal observation and both executed and corrective actions? |
| P3.1 | [Embodiment specifications](P3.1.md) | Why must a requested position fit both joint and actuator limits? |
| P3.2 | [Generalized inverse kinematics](P3.2.md) | What does damping do when a robot cannot easily move in the requested direction? |
| P3.3 | [Command safety](P3.3.md) | Why is clipping a position insufficient to limit speed and acceleration? |
| P3.4 | [Shared simulation backend](P3.4.md) | What changes when adding a robot, and which controller code stays shared? |
| P3.5 | [Webcam runtime (partial)](P3.5.md) | Why distinguish host timing, simulation timing, and camera exposure? |

Use [the acceptance tracker](../MVP_PROGRESS.md) for current completion status.
The master prompt's human-data quality experiment and hardware phases have
separate acceptance criteria; these notes do not establish those results.

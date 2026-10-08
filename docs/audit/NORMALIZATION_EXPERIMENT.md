# Normalization ablation: measured improvement in saturation, no task recovery

The two preprocessing changes tested here **did not fix learned pick-and-place**.
They substantially reduced action saturation relative to the original MLP, but
all three variants timed out without lifting the block in all three trials.
The narrow experiment supports treating the original near-constant scaling as
an engineering problem. It does not support declaring it the sole failure cause.

## What ran

Command, from the project root:

```powershell
.\.venv\Scripts\python.exe scripts/normalization_experiment.py
```

- Input: the same 12 saved scripted demonstrations from seeds 100 through 111.
- Split: the original episode-level training/validation split; normalization is
  computed only from training states with action targets.
- Model: two-layer MLP, width 64, 80 epochs, batch 64, learning rate 0.001,
  initialization/data-loader seed 0, CPU with one PyTorch thread.
- Evaluation: nominal friction 1.0, no command delay, 600-step limit at 25 Hz;
  paired seeds 20000, 20001, and 20002.
- Original checkpoint was reused after verifying weights, metadata, configuration,
  and demonstration payload hashes. Its historical learning source hash was not
  recorded during the original training, so that historical identity cannot be
  established retrospectively. Its failed behavior was reproduced by new rollouts.
- Only the two modified variants were trained. No ACT training was performed.

The variants differ in precisely one preprocessing operation:

1. **Original:** standardize each feature using `max(training_std, 1e-4)`.
2. **Floored scale = one:** when the original scale hits `1e-4`, use `1.0`
   instead. Other features retain their original training standard deviations.
3. **Mask floored features:** set normalized values of those same features to
   zero during both training and inference. Other features are unchanged.

Affected features were wrist-roll position, wrist-roll velocity, and three tray
coordinates. The scale-one variant is a sensitivity test, not a claim that
one radian, one radian/second, and one metre are appropriate calibrated scales.
The masking variant deliberately removes information and is not a deployment
recommendation.

## Results

| Variant | Best validation action MAE | Pick-place successes | Saturated action elements, seed 20000 | Peak normalized feature magnitude, seed 20000 |
|---|---:|---:|---:|---:|
| Original checkpoint | 0.00606235 | 0 / 3 | 99.806% | 80,041.98; wrist-roll velocity at 0.08 s |
| Floored scale = one | 0.00626374 | 0 / 3 | 23.889% | 111.19; elbow velocity at 0.48 s |
| Mask floored features | 0.00622103 | 0 / 3 | 20.306% | 122.31; elbow velocity at 0.40 s |

Saturation means a normalized action component had absolute value greater than
0.99. This is measured across all six action components and all control ticks,
not the proportion of episodes that failed. The mask variant's saturation was
47.861% and 63.306% on the other two seeds, so seed 20000 alone would give an
overly favorable impression.

All nine learned rollouts ended with `timeout_no_lift`. Across the scale-one
trials, maximum block height was at most 0.098258 m. Across the masking trials it
was at most 0.096734 m. The initial block-centre height is 0.096 m.

Low supervised action error therefore did not translate into closed-loop task
success. Removing the wrist-roll amplification exposed substantial excursions
in other inputs: elbow velocity reached normalized magnitudes of approximately
100 to 115 in the scale-one trials, compared with training maximum 12.06.
Masking additionally produced quaternion excursions on two seeds. None of these
observations proves that fixing one remaining feature would solve the task.

These seeds were already used in earlier diagnostics. They are development
trials, not a fresh final test set, and three trials provide no broad competence
or hardware confidence claim. Selection by validation MAE is unchanged across
variants; selected epoch was 74 in each case.

## Simulator limitation discovered independently

This experiment intentionally froze the old simulator while another reviewer
investigated a timing defect: after `mj_step`, returned qpos was post-integration
but derived world transforms had not been refreshed before observation/outcome
calculation. That independently measured defect is being corrected separately.

Every result here uses simulator implementation SHA-256:

```text
854b2b0d0a52bb11cb0220d4176ae0bdff6c8262c42f4d2032c5a3a713ee3d89
```

This isolates the normalization comparison under that implementation. It does
not establish performance for a corrected simulator. New demonstrations,
checkpoints, and evaluations must be produced under the corrected contract;
do not silently relabel the old artifacts. The experiment script rejects a
simulator contract that differs from its reference.

## Engineering decision and hand-coding exercise

**Do not ship either ablation as a proven task fix.** The definite design issue
is that a single tiny standard-deviation floor mixes incompatible physical units
and can give enormous gain to barely varying robot signals. The numerical
division is finite and correct, but finite values need not be useful model inputs.

A principled follow-up should explicitly define and persist scales by observation
meaning: joint position scales from documented ranges in radians, velocity scales
from meaningful radian/second limits or measured envelopes, position scales in
metres for the task workspace, and dimensionless treatment of orientation/time.
The chosen units, feature order, scale vectors, and any masks belong in checkpoint
provenance and must be applied identically during training and inference. Avoid
choosing scales from evaluation states.

Distinguish truly fixed context (the tray in this task) from an actuated joint
that happened to remain almost stationary in the demonstrations. Dropping the
latter hides real drift; scaling it sensibly also cannot teach recovery behavior
that the demonstrations never contain. Preserve raw states and monitor departures
from the training distribution instead of clipping evidence away.

To rebuild the next experiment by hand:

1. Print units, train standard deviations, raw ranges, and normalization gains
   for every feature. Explain why `1e-4` has a different meaning for metres and
   radians/second.
2. Implement explicit per-feature physical scales, persist them, and verify that
   saving/loading preserves exactly the same normalized vector.
3. After the simulator timing correction, recollect the dataset and rerun one
   paired MLP comparison before adding architecture changes or more epochs.
4. Inspect the first divergent control ticks and the corresponding observations
   and commands. Separately test observation scaling and coverage of recovery
   states; do not bundle changes and then attribute an improvement to one of them.

## Retained evidence

- Raw report: [NORMALIZATION_EXPERIMENT_MLP.json](NORMALIZATION_EXPERIMENT_MLP.json).
  It contains full training histories, source diffs/hashes, dataset hashes,
  seed-by-seed outcomes and feature excursions.
- Script: [normalization_experiment.py](../../scripts/normalization_experiment.py).
- Snapshot/checkpoint/raw-NPZ directory:
  `<temp>\xle-normalization-ablation-tz80rfpr`.
  It is retained, but Windows temp cleanup may eventually remove it. Each
  variant's exact inference module is included. In particular, the masking
  checkpoint requires that module; its preprocessing must not be assumed from
  the canonical learner's loader.
- Original canonical learning source SHA-256:
  `5a3a99b82da1ad539b808e20ca276ecf77ec0ffdf0ed3014f80912cfe8298cdb`.
  The script confirmed it was unchanged at experiment completion.
- `NORMALIZATION_EXPERIMENT.json` records an earlier interrupted attempt; it
  contains no completed variants. That run was stopped during optimizer import
  to reuse the original checkpoint instead of retraining it. It is retained
  rather than overwritten or presented as a result.

This was an independent local experiment, separate from the static review.

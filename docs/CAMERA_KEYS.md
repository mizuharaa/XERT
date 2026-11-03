# Camera keys: one rule for recording, training and rollout

**Rule.** Every camera image is the LeRobot feature `observation.images.<name>`.
- On hardware, `<name>` is the key of the camera in the LeRobot robot config's `cameras` dict
  (for example `front`, `wrist`).
- In simulation, `render-cameras` uses `scene` and `wrist`.

Nothing renames a key between recording, training and rollout.

## Where each key comes from (read from the code, LeRobot 0.6.1)

| Stage | Code | Keys |
| --- | --- | --- |
| Hardware recording | `lerobot/robots/so_follower/so_follower.py:70-76` turns each `config.cameras` key into an observation feature. `lerobot/utils/feature_utils.py:94` adds the `observation.images.` prefix (`OBS_IMAGES`, `lerobot/utils/constants.py:25`). | `observation.images.<config key>` |
| Hardware rollout | `lerobot/utils/feature_utils.py:134` (`build_dataset_frame`) removes the same prefix to find the camera by its config key. | same |
| Sim cameras | `xert/data/cameras.py:24-27` (`CAMERAS`), written at `:81` and `:140` | `observation.images.scene`, `observation.images.wrist` |
| Sim export | `xert export-lerobot --cameras` runs `render-cameras` on the P2.1 export, so it has the same keys | `observation.images.scene`, `observation.images.wrist` |
| Runtime recorder | `xert/data/recorder.py:38-60` (`_features`) | none: the recorder never records images |
| Official ACT reference (P1.4) | `xert/eval/lerobot_act.py:24-27` (`policy_config`) | none: state and environment state only |
| Image ACT smoke test | `tests/test_cameras.py:151` selects keys that start with `observation.images.` | whatever the dataset has |
| Import | `xert import-lerobot` does not convert images. They stay in the bit-exact `raw/` copy and are listed in `import.json` `not_imported`. | n/a |

## What the test checks

`tests/test_lerobot_io.py::test_camera_keys_agree_across_render_export_recorder_policy_and_hardware` checks five things:
- the `render-cameras` key set is `{scene, wrist}`;
- `export-lerobot --cameras` writes exactly that set;
- LeRobot's `dataset_to_policy_features`, which is how an image policy chooses its inputs, selects exactly that set as VISUAL;
- the recorder and the official ACT reference use no image keys;
- `hw_to_dataset_features` names hardware cameras `observation.images.<config key>`.

## Findings (not fixed here)

- **No key conflict.** Where keys exist, they follow the rule. No existing key was renamed.
- **Gap:** the official ACT reference (`lerobot_act.py:24-27`) and the recorder use no cameras. Image training exists only as the one-step smoke test in `tests/test_cameras.py`.
- **Open (owner decision):** a policy trained on sim `scene`/`wrist` can read real data only if the hardware camera config uses the same keys. A config such as `front` would need a rename at training time, and the rule above forbids that. The hardware camera config keys are not chosen yet (SO-101 cameras are slice B2).

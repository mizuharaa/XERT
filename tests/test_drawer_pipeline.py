"""Slice K plumbing: the drawer task through rendering, rollouts, recording, quality, the harnesses and DAgger's
refusal.

Scripted hands, scripted clock, constant-action policy stubs and temporary outputs only: no human data, no
trained policy and no number. Tiny fixtures (5-frame recordings, 64 px renders, two seeds).
"""
import json
import os
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

from xert.backends.drawer import DrawerBackend
from xert.spec import load_spec
from xert.tasks import environment_names

ROOT = Path(__file__).resolve().parents[1]
ROBOTS = ("xlerobot_left", "so101")
PERIOD = 1 / 25


def spec_of(robot):
    return load_spec(ROOT / f"xert/spec/{robot}.yaml")


def renderer_without_msaa(robot, size=64):
    """CameraRenderer for the drawer with MSAA off: with MuJoCo's default 4 offscreen samples, re-rendering one state
    on the Windows GPU is not bit-stable (docs/handoff/H-stack-bench.md); the path under test is the same."""
    from xert.data.cameras import CameraRenderer
    try:
        renderer = CameraRenderer(size, robot, "drawer")
    except Exception as exc:
        if os.environ.get("CI"):
            raise
        pytest.skip(f"No offscreen GL context: {exc!r}")
    renderer.renderer.close()
    renderer.model.vis.quality.offsamples = 0
    renderer.renderer = mujoco.Renderer(renderer.model, size, size)
    return renderer


@pytest.mark.parametrize("robot", ROBOTS)
def test_two_seeds_rebuild_offline_and_live_to_each_backends_own_scene(robot):
    """Review K-01: with the cabinet fixed per robot, joints + drawer.pos rebuild the whole scene. Two seeds with
    different initial openings: offline and live_frame reconstruct handle and cabinet within 1e-5 of each backend's
    scene, and the pixels equal a direct render of that backend's MjData."""
    pytest.importorskip("torch")
    from xert.data.cameras import TOLERANCE, _columns, row_state, scene_objects
    from xert.eval import image_act
    renderer = renderer_without_msaa(robot)
    image = {"dtype": "image", "shape": (64, 64, 3)}
    try:
        scenes = []
        for seed in (0, 1):
            backend = DrawerBackend(spec_of(robot), seed)
            # Recordings store float32; float32-exact qpos makes the live float64 state the recorded one bit for bit.
            backend.data.qpos[:] = backend.data.qpos.astype(np.float32)
            mujoco.mj_forward(backend.model, backend.data)
            live = image_act.live_frame(backend, renderer, {"observation.images.scene": image,
                                                            "observation.images.wrist": image})
            joints = [f"{name}.pos" for name in backend.spec.joint_names]
            recorded = {"observation.state": {"names": joints},
                        "observation.environment_state": {"names": backend.object_names}}
            frame = {"observation.state": backend.read_state().q.astype(np.float32),
                     "observation.environment_state": np.asarray(backend.object_state(), dtype=np.float32)}
            state = row_state(frame, _columns(recorded, renderer.joints, renderer.bodies, renderer.points,
                                              renderer.scalars))
            offline, achieved = renderer.render(state["joints"], scene_objects(state, renderer.bodies, renderer.scalars))
            assert set(achieved) == {"handle", "grasp", "drawer.pos"}
            for name, actual in (("handle", backend.data.geom("handle").xpos), ("grasp", backend.data.site("grasp").xpos),
                                 ("drawer.pos", backend.data.qpos[backend.task.qpos:backend.task.qpos + 1])):
                assert np.max(np.abs(achieved[name] - actual)) <= TOLERANCE, name
            np.testing.assert_allclose(renderer.data.body("cabinet").xpos, backend.data.body("cabinet").xpos, atol=1e-12)
            direct = mujoco.MjData(renderer.model)  # The backend's own state, posed directly, no named state.
            direct.qpos[:] = backend.data.qpos
            mujoco.mj_forward(renderer.model, direct)
            for key, pixels in offline.items():
                np.testing.assert_array_equal(live[key].mul(255).round().byte().permute(1, 2, 0).numpy(), pixels)
                renderer.renderer.update_scene(direct, camera=renderer.cameras[key.split(".")[-1]].get(
                    "model_camera", key.split(".")[-1]), scene_option=renderer.option)
                np.testing.assert_array_equal(renderer.renderer.render(), pixels)
            scenes.append((state["drawer.pos"][0], achieved["handle"], offline["observation.images.scene"]))
        (open0, handle0, image0), (open1, handle1, image1) = scenes
        assert open0 != open1 and not np.allclose(handle0, handle1) and not np.array_equal(image0, image1)
    finally:
        renderer.close()


def test_drawer_cameras_override_only_the_scene_camera_and_keep_the_pinned_ones():
    from xert.data.cameras import CAMERAS, cameras
    for robot in ROBOTS:
        assert cameras(robot) == cameras(robot, "stack") == CAMERAS[robot]
        drawer = cameras(robot, "drawer")
        assert drawer["wrist"] == CAMERAS[robot]["wrist"] and drawer["scene"] != CAMERAS[robot]["scene"]

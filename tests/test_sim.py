"""Run with python -m unittest discover -s tests -p test_sim.py."""
import unittest
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import mujoco

from xle_workbench.sim import ManipulationEnv, model_fingerprint, verified_source


class SimulationTest(unittest.TestCase):
    def test_step_reports_post_integration_transforms_and_contacts(self):
        env = ManipulationEnv({"horizon": 12})
        self.addCleanup(env.close)
        current = mujoco.MjData(env.model)
        for _ in range(12):
            observation, _, _, info = env.step(env.expert_action())
            current.qpos[:] = env.data.qpos
            current.qvel[:] = env.data.qvel
            current.ctrl[:] = env.data.ctrl
            mujoco.mj_forward(env.model, current)
            np.testing.assert_allclose(env.data.site_xpos[env.site], current.site_xpos[env.site], atol=1e-12, rtol=0)
            np.testing.assert_array_equal(observation[19:22], current.site_xpos[env.site].astype(np.float32))
            np.testing.assert_allclose(info["object_position"], current.xpos[env.block], atol=1e-12, rtol=0)
            self.assertEqual(env.data.ncon, current.ncon)
            for actual, expected in zip(env.data.contact[:env.data.ncon], current.contact[:current.ncon]):
                self.assertEqual((actual.geom1, actual.geom2), (expected.geom1, expected.geom2))
                self.assertAlmostEqual(actual.dist, expected.dist, places=12)

    def test_asset_integrity_and_portable_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {"LICENSE": b"test license", "xlerobot.xml": b"<mujoco/>", "arm.stl": b"mesh"}
            source = {"sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
            for name, data in files.items():
                (root / name).write_bytes(data)
            (root / "SOURCE.json").write_text(json.dumps(source))
            self.assertEqual(verified_source(root), source)
            xml = '<mujoco><compiler meshdir="C:/first"/></mujoco>'
            first = model_fingerprint(xml, source)
            self.assertEqual(first, model_fingerprint(xml.replace("C:/first", "D:/second"), source))
            (root / "arm.stl").write_bytes(b"changed mesh")
            with self.assertRaisesRegex(ValueError, "arm.stl"):
                verified_source(root)
            source["sha256"]["arm.stl"] = hashlib.sha256(b"changed mesh").hexdigest()
            self.assertNotEqual(first, model_fingerprint(xml, source))

    def test_contact_pick_and_release(self):
        env = ManipulationEnv()
        self.addCleanup(env.close)
        obs = env.reset(0)
        self.assertEqual(obs.shape, (26,))
        contact_while_lifted = False
        for _ in range(env.config["horizon"]):
            obs, reward, done, info = env.step(env.expert_action())
            contact_while_lifted |= info["object_in_robot_contact"] and info["object_position"][2] > .145
            self.assertTrue(np.isfinite(obs).all())
            if done:
                break
        self.assertTrue(contact_while_lifted, "must lift through physical finger contacts")
        self.assertTrue(info["success"], info)
        self.assertEqual(reward, 1)
        self.assertFalse(info["object_in_robot_contact"], "must release before success")
        self.assertEqual(env.model.neq, 0, "no attachment equality constraints")
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(6))

    def test_deterministic_replay_and_delayed_actions(self):
        a = ManipulationEnv({"horizon": 30, "delay_steps": 3})
        b = ManipulationEnv({"horizon": 30, "delay_steps": 3})
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        np.testing.assert_array_equal(a.reset(7), b.reset(7))
        for _ in range(30):
            action = a.expert_action()
            left = a.step(action)
            right = b.step(action.astype(np.float32))
            np.testing.assert_array_equal(left[0], right[0])
            self.assertEqual(left[3], right[3])
        self.assertFalse(left[3]["success"])
        self.assertEqual(left[3]["failure_reason"], "timeout_no_lift")

    def test_validation(self):
        for config in [{"friction": float("nan")}, {"friction": 0}, {"delay_steps": 1.5},
                       {"horizon": 2001}, {"horizon": True}, {"object_position": [0, float("inf")]},
                       {"object_position_bounds": [[1, 2], [0, 1]]}, {"unexpected": 1}]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                ManipulationEnv(config)
        env = ManipulationEnv()
        self.addCleanup(env.close)
        for action in [np.zeros(5), [0, 0, 0, 0, 0, float("nan")], np.ones(6)*2]:
            with self.assertRaises(ValueError):
                env.step(action)


if __name__ == "__main__":
    unittest.main()

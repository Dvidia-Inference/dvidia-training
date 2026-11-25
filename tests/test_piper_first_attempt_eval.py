import hashlib
from pathlib import Path
import tempfile
import unittest
import zlib

import numpy as np

from dvidia_training.piper_first_attempt_eval import _observe, _failure_stage
from dvidia_training.piper_first_attempt_replay import read_actions


class IndependentKinematics:
    def fk(self, joints):
        if len(joints) != 6:
            raise AssertionError("FK consumed additional native coordinates")
        return {"tcp_position_world_m": joints[:3], "tcp_quaternion_world_wxyz": [1., 0., 0., 0.]}


class EvaluationContractTests(unittest.TestCase):
    def test_observation_excludes_extra_task_coordinates_and_preserves_age(self):
        q, v = np.arange(15.), np.arange(14.)
        applied = {"joint_targets_rad": [0.] * 6, "gripper_position_m": .03}
        anchor, goal = [.3, -.02, .08], [.35, .08, .08]

        def make_obs():
            return _observe(.84, q, v, IndependentKinematics(), anchor, goal, applied)

        before = make_obs()
        q[8:] = 123
        v[8:] = -654
        self.assertEqual(before, make_obs())
        self.assertEqual(before["initial_object_age_seconds"], .84)
        self.assertEqual(before["initial_object_captured_at_seconds"], 0.)
        self.assertEqual(set(before), {"t_seconds", "joint_position_rad", "joint_velocity_rad_s",
            "gripper_joint_position_m", "gripper_joint_velocity_m_s", "tcp_position_world_m",
            "tcp_quaternion_wxyz", "initial_object_position_world_m", "initial_object_age_seconds",
            "initial_object_captured_at_seconds", "previous_applied", "goal_position_world_m"})
        before["previous_applied"]["joint_targets_rad"][0] = 99.
        before["initial_object_position_world_m"][0] = 99.
        before["goal_position_world_m"][0] = 99.
        self.assertEqual(applied["joint_targets_rad"][0], 0.)
        self.assertEqual(anchor[0], .3)
        self.assertEqual(goal[0], .35)

    def test_independent_failure_stage_preserves_contact_and_lost_retention(self):
        native = dict(unexpected_contact_native_ticks=0, grasp_seen=False, lift_seen=False,
                      retention_lost=False, carry_complete=False)
        score = dict(valid=True, success=False, native_recomputed=native)
        self.assertEqual(_failure_stage(score), "grasp")
        native.update(grasp_seen=True)
        self.assertEqual(_failure_stage(score), "lift")
        native.update(lift_seen=True, retention_lost=True)
        self.assertEqual(_failure_stage(score), "retention")
        native.update(unexpected_contact_native_ticks=1)
        self.assertEqual(_failure_stage(score), "critical_contact")
        score.update(valid=False)
        self.assertEqual(_failure_stage(score), "evidence_invalid")

    def test_action_log_roundtrip_and_hash_size_and_finiteness_checks(self):
        values = np.zeros((1000, 7), dtype="<f8")
        values[:, 0] = np.linspace(.01, .025, 1000)
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)

            def setup(raw):
                body = zlib.compress(raw)
                path = root/"attempt-0000-actions.f64.zlib"
                path.write_bytes(body)
                return dict(path=path.name, codec="zlib", dtype="<f8", shape=[1000, 7],
                            bytes=len(body), sha256=hashlib.sha256(body).hexdigest())

            binding = setup(values.tobytes())
            np.testing.assert_array_equal(read_actions(root, binding), values)
            with self.assertRaisesRegex(ValueError, "digest"):
                read_actions(root, {**binding, "sha256": "0" * 64})
            with self.assertRaisesRegex(ValueError, "receipt directory"):
                read_actions(root, {**binding, "path": "../attempt-0000-actions.f64.zlib"})
            with self.assertRaisesRegex(ValueError, "decoded size"):
                read_actions(root, setup(values.tobytes() + bytes(20)))
            values[0, -1] = float("nan")
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                read_actions(root, setup(values.tobytes()))

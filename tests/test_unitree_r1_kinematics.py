"""Offline pose and oracle checks for the two stand-alone R1-7A arm descriptions."""
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from dvidia_training import unitree_r1_kinematics as kin


class UnitreeKinematicsTests(unittest.TestCase):
    def test_zero_pose_in_each_archived_base_frame(self):
        # Independently recorded from native imports of the pinned side-specific URDFs.
        for side, expected_xyz in (("left", [-.00267454137, -.03412197689, .19292105234]),
                                   ("right", [.00267453800, -.03412193556, .19292059116])):
            with self.subTest(side=side):
                result = kin.pose(side, [0.] * 7)
                transform = np.array(result["base_from_link7"])
                self.assertEqual(result["reference_frame"], "base_link")
                self.assertEqual(result["end_link"], "Link7")
                self.assertEqual(result["joint_names"], [f"J{i}" for i in range(1, 8)])
                self.assertEqual(result["source"]["side"], side)
                np.testing.assert_allclose(transform[:3, 3], expected_xyz, atol=1e-11, rtol=0)
                np.testing.assert_array_equal(transform[3], [0, 0, 0, 1])
                np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-14)
                self.assertAlmostEqual(np.linalg.det(transform[:3, :3]), 1.)

    def test_link7_rotation_remains_visible_when_origin_does_not_move(self):
        for side in ("left", "right"):
            q = [.2, .6, .7, .4, -.3, .2, -.3]
            start = np.array(kin.pose(side, q)["base_from_link7"])
            q[-1] = .4
            end = np.array(kin.pose(side, q)["base_from_link7"])
            np.testing.assert_allclose(start[:3, 3], end[:3, 3], atol=1e-14)
            self.assertAlmostEqual(kin._rotation_error(start[:3, :3], end[:3, :3]), .7)

    def test_side_and_limit_checks(self):
        with self.assertRaises(ValueError):
            kin.pose("r1", [0] * 7)
        q = [0., 0., 0., 0., 0., .7, 0.]
        self.assertEqual(kin.pose("right", q)["joint_positions_rad"], q)
        with self.assertRaisesRegex(ValueError, "J6"):
            kin.pose("left", q)
        for side in ("left", "right"):
            info = kin.pose(side, [0.] * 7)
            for limit_index in (0, 1):
                edge = [joint[limit_index] for joint in info["joint_limits_rad"]]
                self.assertEqual(kin.pose(side, edge)["joint_positions_rad"], edge)
            with self.assertRaisesRegex(ValueError, "J4"):
                kin.pose(side, [0., 0., 0., -.01, 0., 0., 0.])

    def test_refuses_ineligible_vectors(self):
        for bad in ([], [0] * 6, [0] * 8, [True] + [0] * 6, [float("inf")] + [0] * 6,
                    [float("nan")] * 7, ["0"] * 7, np.zeros((7, 1)), 123, None):
            with self.subTest(values=repr(bad)), self.assertRaises(ValueError):
                kin.pose("left", bad)

    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "Optional arm dependencies")
    def test_benchmark_checks_all_link_poses_and_both_signs(self):
        for side in ("left", "right"):
            result = kin.benchmark(side)
            self.assertTrue(result["passed"])
            self.assertEqual(result["summary"]["link_transforms_compared"], 140)
            self.assertEqual(result["native_time_s"], 0.)
            self.assertEqual((result["native_actuators"], result["native_geometries"]), (0, 0))
            rows = {row["case"]: row for row in result["cases"]}
            self.assertEqual(len(rows), 20)
            for index, joint in enumerate(result["joint_names"]):
                minus = np.array(rows[joint + "_minus"]["joint_positions_rad"])
                plus = np.array(rows[joint + "_plus"]["joint_positions_rad"])
                self.assertEqual(np.flatnonzero(plus != minus).tolist(), [index])
                self.assertGreater(plus[index], minus[index])

    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "Optional arm dependencies")
    def test_orientation_only_defect_fails_comparison(self):
        original = kin._transforms

        def reversed_wrist(chain, angles):
            wrong = angles.copy()
            wrong[-1] *= -1
            return original(chain, wrong)

        with patch.object(kin, "_transforms", side_effect=reversed_wrist):
            result = kin.benchmark("left")
        self.assertFalse(result["passed"])
        self.assertLess(result["summary"]["max_link_position_error_m"], 1e-9)
        self.assertGreater(result["summary"]["max_link_orientation_error_rad"], 1.)


if __name__ == "__main__":
    unittest.main()

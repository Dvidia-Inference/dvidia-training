"""Reconstruct the setup experiment from fixed simulated RGB-D without graphics."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from dvidia_training import piper_rgbd_tracker as tracker
from dvidia_training import piper_workspace_calibration as survey


HERE = Path(__file__).resolve().parents[1] / "benchmarks/piper-first-attempt-v01/workspace"
NOMINAL_SHA256 = "1a092b3145d754bbf2e4b584fc90822d878b66330f199547c78425f93ebf19f2"


def saved(profile):
    return json.loads((HERE / f"{profile}.json").read_text())


class WorkspaceCalibrationTests(unittest.TestCase):
    def test_saved_frames_reconstruct_all_six_camera_poses_and_error_comparisons(self):
        comparison = saved("comparison")
        nominal = survey.load_workspace_calibration(HERE / "nominal.json", expected_sha256=NOMINAL_SHA256)
        stale = np.array(nominal["base_from_camera"]).reshape(4, 4)
        for profile in comparison["camera_profiles"]:
            with self.subTest(profile=profile), np.load(HERE / f"{profile}-frame.npz", allow_pickle=False) as frame:
                receipt = saved(profile)
                for name, hash_field in (("rgb", "rgb_sha256"), ("depth", "depth_float32_sha256")):
                    self.assertEqual(sha256(frame[name].tobytes()).hexdigest(), receipt[hash_field])
                observed = survey.observations(frame["rgb"], frame["depth"])
                for marker, value in observed.items():
                    self.assertEqual(value["pixels"], receipt["observations"][marker]["pixels"])
                    np.testing.assert_allclose(value["camera_m"], receipt["observations"][marker]["camera_m"],
                                               atol=1e-12, rtol=0)
                pose, fit_mm, check_mm = survey.solve(observed)
                np.testing.assert_allclose(fit_mm, receipt["fit_error_mm"], atol=1e-7, rtol=0)
                np.testing.assert_allclose(check_mm, receipt["check_error_mm"], atol=1e-7, rtol=0)
                calibration = survey.load_workspace_calibration(HERE / f"{profile}.json",
                                                               camera_profile=profile, duration_s=3)
                _, tracker_pose = tracker._calibration(calibration, (240, 320), 0., 0.)
                np.testing.assert_allclose(tracker_pose, pose, atol=1e-10, rtol=0)
                with self.assertRaisesRegex(ValueError, "observation"):
                    tracker._calibration(calibration, (240, 320), 0., 3.1)
                if profile != "nominal":
                    measured = comparison["shifted_cameras"][profile]
                    self.assertEqual(measured["status"], "accepted")
                    np.testing.assert_allclose(check_mm, measured["held_out_mm"], atol=1e-7, rtol=0)
                    np.testing.assert_allclose(survey.errors_mm(stale, observed, survey.CHECK),
                                               measured["stale_held_out_mm"], atol=1e-7, rtol=0)
                    self.assertGreater(min(measured["stale_held_out_mm"]), survey.MAX_RESIDUAL_M * 1000.)

    def test_withheld_points_do_not_change_fit_but_can_refuse_it(self):
        observed = saved("nominal")["observations"]
        pose, fit, check = survey.solve(observed)
        revised = deepcopy(observed)
        revised["cyan"]["camera_m"][0] += .0005
        other_pose, other_fit, other_check = survey.solve(revised)
        np.testing.assert_array_equal(pose, other_pose)
        self.assertEqual(fit, other_fit)
        self.assertNotEqual(check, other_check)
        revised["cyan"]["camera_m"][0] += .015
        with self.assertRaisesRegex(ValueError, "4 mm"):
            survey.solve(revised)

    def test_missing_marker_and_invalid_depth_refuse(self):
        with np.load(HERE / "nominal-frame.npz", allow_pickle=False) as frame:
            rgb, depth = frame["rgb"], frame["depth"]
        green = (rgb[:, :, 0] <= 45) & (rgb[:, :, 1] >= 100) & (rgb[:, :, 2] <= 45)
        revised = rgb.copy()
        revised[green] = 0
        with self.assertRaisesRegex(ValueError, "support for green"):
            survey.observations(revised, depth)
        depth[green] = np.nan
        with self.assertRaisesRegex(ValueError, "depth at green"):
            survey.observations(rgb, depth)

    def test_true_diameter_accepts_rotated_support_and_rejects_disconnected_points(self):
        # A sampled equilateral triangle has diameter 70 mm and AABB diagonal 93 mm.
        tri = np.array([[0., 0., 1.], [.07, 0., 1.], [.035, .0606218, 1.]])
        self.assertGreater(np.linalg.norm(np.ptp(tri, axis=0)), .075)
        self.assertTrue(survey._compact(tri))
        tri[2, 1] = .09
        self.assertFalse(survey._compact(tri))

    def test_altered_receipt_and_wrong_camera_refuse(self):
        with self.assertRaisesRegex(ValueError, "digest"):
            survey.load_workspace_calibration(HERE / "nominal.json", expected_sha256="f"*64)
        with self.assertRaisesRegex(ValueError, "profile"):
            survey.load_workspace_calibration(HERE / "relocated.json")
        for field, changed in (("fit_error_mm", [0]*4), ("check_error_mm", [0]*2),
                               ("base_from_camera", [0.]*16), ("max_allowed_error_mm", 40)):
            data = saved("nominal")
            data[field] = changed
            with self.subTest(field=field), TemporaryDirectory() as temp:
                path = Path(temp) / "mutated.json"
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    survey.load_workspace_calibration(path)

    def test_collapsed_fit_rejects_unconstrained_pose(self):
        observed = saved("nominal")["observations"]
        for name in survey.FIT:
            observed[name]["camera_m"] = [0., 0., 1.]
        with self.assertRaisesRegex(ValueError, "cannot constrain"):
            survey.solve(observed)

    def test_mounting_faults_and_external_base_reference(self):
        root = HERE / "mounting"
        expected = json.loads((root / "comparison.json").read_text())
        with np.load(root / "reference-frame.npz", allow_pickle=False) as frame:
            reference = survey.observations(frame["rgb"], frame["depth"])
        for case in ("anchor_offset", "check_offset", "shared_offset"):
            with self.subTest(case=case), np.load(root / f"{case}-frame.npz", allow_pickle=False) as frame:
                observed = survey.observations(frame["rgb"], frame["depth"])
                if case == "shared_offset":
                    pose, _, internal = survey.solve(observed)
                    self.assertLess(max(internal), 4.)
                    external = survey.errors_mm(pose, reference, survey.CHECK)
                    self.assertGreater(min(external), 8.)
                    np.testing.assert_allclose(external, expected[case]["independent_base_reference_error_mm"],
                                               atol=1e-7, rtol=0)
                    with self.assertRaisesRegex(ValueError, "fixture"):
                        survey.load_workspace_calibration(root / f"{case}.json")
                else:
                    with self.assertRaises(survey.SurveyMismatch) as failure:
                        survey.solve(observed)
                    np.testing.assert_allclose(failure.exception.check_mm, expected[case]["check_error_mm"],
                                               atol=1e-7, rtol=0)

import copy
import json
import math
import unittest

from dvidia_training.task_feasibility import (
    GRAVITY_M_S2,
    RESULT_SCHEMA,
    SCHEMA,
    TaskFeasibilityError,
    assess,
    template,
)


class TaskFeasibilityTests(unittest.TestCase):
    def test_template_is_synthetic_and_requires_test_lift(self):
        result = assess(template())
        self.assertEqual(result["schema"], RESULT_SCHEMA)
        self.assertEqual(result["status"], "test_lift_required")
        self.assertAlmostEqual(result["computed"]["conservative_required_load_n"],
                               .2 * (GRAVITY_M_S2 + .5) * 1.5)
        self.assertAlmostEqual(result["computed"]["two_finger_friction_capacity_n"], 4.8)
        self.assertIn("not hardware qualification", result["scope"])
        self.assertFalse(result["physical_robot_ready"])
        self.assertEqual(result["observation_origin"], "caller_declared_not_verified")

    def test_ready_only_after_all_checks_and_test_lift(self):
        document = template()
        document["test_lift_passed"] = True
        self.assertEqual(assess(document)["status"], "ready_for_simulated_transport")

    def test_unknown_mass_geometry_calibration_and_reach_require_measurement(self):
        cases = (
            ("mass_interval_kg", None),
            ("geometry_known", False),
            ("calibration_id", None),
            ("reachability", "unknown"),
            ("friction_interval", None),
            ("per_finger_normal_force_budget_n", None),
            ("pose_specific_payload_limit_kg", None),
        )
        for field, value in cases:
            with self.subTest(field=field):
                document = template()
                document[field] = value
                document["test_lift_passed"] = True
                self.assertEqual(assess(document)["status"], "probe_required")

    def test_blocking_conditions_take_precedence_over_unknowns_and_test_lift(self):
        cases = (
            ("observations_fresh", False),
            ("slip_detected", True),
            ("reachability", "unreachable"),
        )
        for field, value in cases:
            with self.subTest(field=field):
                document = template()
                document["mass_interval_kg"] = None
                document[field] = value
                self.assertEqual(assess(document)["status"], "blocked")

    def test_too_heavy_for_calibrated_payload_or_friction_blocks(self):
        payload = template()
        payload["pose_specific_payload_limit_kg"] = .01
        self.assertEqual(assess(payload)["status"], "blocked")

        grip = template()
        grip["per_finger_normal_force_budget_n"] = .1
        self.assertEqual(assess(grip)["status"], "blocked")

    def test_width_fit_and_pose_uncertainty_are_conservative(self):
        too_wide = template()
        too_wide["object_width_interval_m"] = [.061, .065]
        self.assertEqual(assess(too_wide)["status"], "blocked")

        uncertain_fit = template()
        uncertain_fit["object_width_interval_m"] = [.040, .055]
        self.assertEqual(assess(uncertain_fit)["status"], "probe_required")

        too_small = template()
        too_small["object_width_interval_m"] = [.004, .009]
        self.assertEqual(assess(too_small)["status"], "blocked")

        uncertain_closure = template()
        uncertain_closure["object_width_interval_m"] = [.008, .020]
        self.assertEqual(assess(uncertain_closure)["status"], "probe_required")

        uncertain_pose = template()
        uncertain_pose["pose_uncertainty_m"] = .02
        self.assertEqual(assess(uncertain_pose)["status"], "probe_required")

    def test_conservative_interval_uses_upper_mass_and_lower_friction(self):
        document = template()
        result = assess(document)
        self.assertAlmostEqual(result["computed"]["conservative_required_load_n"],
                               document["mass_interval_kg"][1] *
                               (GRAVITY_M_S2 + document["planned_upward_accel_m_s2"]) *
                               document["safety_factor"])
        self.assertAlmostEqual(result["computed"]["two_finger_friction_capacity_n"],
                               2 * document["friction_interval"][0] *
                               document["per_finger_normal_force_budget_n"])

    def test_accepts_json_and_does_not_mutate_input(self):
        document = template()
        original = copy.deepcopy(document)
        self.assertEqual(assess(json.dumps(document))["status"], "test_lift_required")
        self.assertEqual(document, original)

    def test_rejects_unknown_fields_bad_schema_and_malformed_values(self):
        document = template()
        document["physical_robot_ready"] = True
        with self.assertRaises(TaskFeasibilityError):
            assess(document)
        document = template()
        document["schema"] = "future"
        with self.assertRaises(TaskFeasibilityError):
            assess(document)
        document = template()
        document["mass_interval_kg"] = [0.1, math.nan]
        with self.assertRaises(TaskFeasibilityError):
            assess(document)
        document = template()
        document["safety_factor"] = .99
        with self.assertRaises(TaskFeasibilityError):
            assess(document)
        document = template()
        document["planned_upward_accel_m_s2"] = True
        with self.assertRaises(TaskFeasibilityError):
            assess(document)
        document = template()
        document["pose_uncertainty_m"] = 10**10000
        with self.assertRaises(TaskFeasibilityError):
            assess(document)

    def test_json_is_bounded_unique_finite_and_depth_errors_are_normalized(self):
        with self.assertRaises(TaskFeasibilityError):
            assess(b" " * (64 * 1024 + 1))
        with self.assertRaises(TaskFeasibilityError):
            assess('{"schema":"x","schema":"y"}')
        with self.assertRaises(TaskFeasibilityError):
            assess('{"number":NaN}')
        with self.assertRaises(TaskFeasibilityError):
            assess('{"number":1e999}')
        with self.assertRaises(TaskFeasibilityError):
            assess("[" * 2000 + "0" + "]" * 2000)

    def test_exposes_input_schema_for_callers(self):
        self.assertEqual(template()["schema"], SCHEMA)


if __name__ == "__main__":
    unittest.main()

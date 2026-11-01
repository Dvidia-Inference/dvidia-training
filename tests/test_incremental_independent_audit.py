"""Independent fixed32 record-law tests; no simulator or scorer imports."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

_spec = importlib.util.spec_from_file_location('dvidia_incremental_record_audit_tests',
    Path(__file__).resolve().parents[1] / 'benchmarks/piper-incremental-hold-v03/independent_audit.py')
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def correction_fixture():
    n = 1040
    joints = np.full((n, 6), .1, dtype=np.float64)
    joints[1019:, 0] = .101
    arrays = {"t_seconds": np.arange(1, n + 1, dtype=np.float64) * .001,
              "joint_position_rad": joints}
    issued, jaw = [.1] * 6, .012
    actions = [{"t_seconds": 1., "joint_targets_rad": issued.copy(), "gripper_position_m": jaw},
               {"t_seconds": 1.02, "joint_targets_rad": [.099, .1, .1, .1, .1, .1],
                "gripper_position_m": jaw}]
    reference = {"t_seconds": 1., "position_m": [.3, 0., .1], "quaternion_wxyz": [1., 0., 0., 0.],
                 "issued_joint_targets_rad": issued.copy(), "measured_joint_position_rad": issued.copy(),
                 "frozen_jaw_target_m": jaw, "raw_pose_joint_targets_rad": issued.copy(),
                 "fixed_bias_rad": [0.] * 6}

    def record(time, measured, target, before, proposal, accumulated, *, raw, increment,
               accepted, final=False, applied=None):
        return {"t_seconds": time, "final_observation": final, "command_issued": not final,
                "status": "admitted_hold", "admitted": True, "variant": "incremental",
                "candidate": "incremental", "update_law": "fixed increment law",
                "update_gain_per_boundary": 1., "measured_joint_position_rad": measured,
                "joint_targets_rad": None if final else target, "frozen_jaw_target_m": jaw,
                "raw_pose_joint_targets_rad": raw, "fixed_bias_rad": None,
                "pose_error_increment_rad": increment, "correction_before_update_rad": before,
                "proposed_cumulative_correction_rad": proposal,
                "accumulated_correction_rad": accumulated,
                "applied_cumulative_correction_rad": applied,
                "correction_rad": accumulated, "correction_update_accepted": accepted,
                "correction_rate_limited_axes": [], "refused_bounds": None,
                "active_regulation": True,
                **({"preceding_issued_joint_targets_rad": actions[-1]["joint_targets_rad"]} if final else {})}

    records = [
        record(1., issued.copy(), issued.copy(), [0.] * 6, [0.] * 6, [0.] * 6,
               raw=issued.copy(), increment=None, accepted=False, applied=[0.] * 6),
        record(1.02, [.101, .1, .1, .1, .1, .1], actions[1]["joint_targets_rad"],
               [0.] * 6, [-.001, 0., 0., 0., 0., 0.], [-.001, 0., 0., 0., 0., 0.],
               raw=issued.copy(), increment=[-.001, 0., 0., 0., 0., 0.], accepted=True,
               applied=[-.001, 0., 0., 0., 0., 0.]),
        record(1.04, [.101, .1, .1, .1, .1, .1], None,
               [-.001, 0., 0., 0., 0., 0.], None, [-.001, 0., 0., 0., 0., 0.],
               raw=None, increment=None, accepted=False, final=True),
    ]
    policy = {"variant": "incremental", "candidate": "incremental", "admitted": True,
              "frozen_jaw_target_m": jaw,
              "events": [{"type": "evidence_qualified", "t_seconds": 1., "reason": "sampled"}],
              "retention": {"hold_reference": reference, "hold_records": records, "input_refusals": [],
                            "native_joint_control_limits_rad": [[-2., 2.]] * 6,
                            "update_law": "fixed increment law", "update_gain_per_boundary": 1.}}
    return policy, actions, arrays


class NativePredicateTests(unittest.TestCase):
    def test_100_sample_streak_and_strict_force_floor(self):
        n = 240
        arrays = {name: np.zeros((n,) if width == 1 else (n, width), dtype=np.float64)
                  for name, width in audit.FIELDS.items()}
        arrays["t_seconds"] = np.arange(1, n + 1, dtype=np.float64) * .001
        arrays["object_position_m"][:] = [.3, 0., .08]
        arrays["object_minus_tcp_local_m"][:] = [0., 0., -.1]
        arrays["object_position_m"][20:, 2] += .006
        arrays["grasp_contact_counts"][:] = 1.
        arrays["pad_normal_forces_n"][:] = .02
        probe = {"start_time_s": .02, "object_origin_m": [.3, 0., .08],
                 "local_offset_origin_m": [0., 0., -.1]}
        _, streak, qualified, _, _, _ = audit.native_predicate(arrays, probe)
        self.assertEqual((streak[118], bool(qualified[118])), (99, False))
        self.assertEqual((streak[119], bool(qualified[119])), (100, True))
        arrays["pad_normal_forces_n"][60, 0] = .01
        _, streak, qualified, _, _, _ = audit.native_predicate(arrays, probe)
        self.assertEqual(streak[60], 0)
        self.assertFalse(qualified[119])
        self.assertTrue(qualified[160])

    def test_stationarity_is_pose_based_and_quaternion_sign_invariant(self):
        arrays = {"tcp_position_m": np.zeros((3, 3), dtype=np.float64),
                  "tcp_quaternion_wxyz": np.tile([1., 0., 0., 0.], (3, 1))}
        arrays["tcp_quaternion_wxyz"][1:] *= -1.
        self.assertTrue(audit.stationarity(arrays, 0, 2, True, True)["grip_retention_and_stationarity"])
        arrays["tcp_position_m"][1, 0] = .004001
        self.assertFalse(audit.stationarity(arrays, 0, 2, True, True)["tcp_pose_within_bounds"])

    def test_native_endpoint_completion_is_builtin_bool(self):
        n = 5200
        arrays = {name: np.zeros((n,) if width == 1 else (n, width), dtype=np.float64)
                  for name, width in audit.FIELDS.items()}
        arrays["t_seconds"] = np.arange(1, n + 1, dtype=np.float64) * .001
        arrays["object_position_m"][:] = [.3, 0., .08]
        arrays["object_position_m"][20:, 2] += .006
        arrays["tcp_position_m"][:] = [.3, 0., .08]
        arrays["object_minus_tcp_local_m"][:] = 0.
        arrays["tcp_quaternion_wxyz"][:] = [1., 0., 0., 0.]
        arrays["grasp_contact_counts"][:] = 1.
        arrays["pad_normal_forces_n"][:] = .02
        policy = {"probes": [{"probe_index": 0, "start_time_s": .02,
                               "object_origin_m": [.3, 0., .08],
                               "local_offset_origin_m": [0., 0., 0.]}]}
        result = audit.native_success(arrays, policy, {"probe_index": 0, "t_seconds": .1})
        self.assertIs(type(result["complete_5s"]), bool)


class PrefixTests(unittest.TestCase):
    def test_prefix_contains_cutoff_sample_and_excludes_that_command(self):
        cutoff = {"time_s": .04, "reason": "admission", "treatment_boundaries": []}
        arrays = {"t_seconds": np.arange(1, 20001, dtype=np.float64) * .001,
                  "signal": np.arange(20000, dtype=np.float64)}
        actions = [{"t_seconds": i * .02, "command": i} for i in range(1000)]
        result = audit.prefix_identity(arrays, actions, cutoff)
        self.assertEqual(result["native_rows"], 40)
        self.assertEqual(result["commands"], 2)

    def test_invalid_member_keeps_complete_quartet_unknown(self):
        attempts = {variant: {"valid": True} for variant in audit.VARIANTS}
        attempts["incremental"] = {"valid": False, "reason": "partial", "native_rows_available": 12}
        result = audit.paired_prefix("scene-1", attempts)
        self.assertFalse(result["verified"])
        self.assertEqual(result["reason"], "invalid_attempt_prevents_complete_quartet")
        self.assertEqual(result["variants"][3]["native_rows_available"], 12)


class IncrementalRecordAuditTests(unittest.TestCase):
    def test_valid_record_reconstructs_increment_and_final_no_command(self):
        policy, actions, arrays = correction_fixture()
        result = audit.correction_audit(policy, actions, arrays, audit.treatment_boundary(policy), "incremental")
        self.assertTrue(result["first_admitted_target_continuous"])
        self.assertTrue(result["final_observation_no_ik_or_command"])
        self.assertAlmostEqual(result["maximum_abs_pose_error_increment_rad"], .001)
        self.assertAlmostEqual(result["maximum_abs_active_correction_rad"], .001)

    def test_rejects_wrong_increment_first_target_and_hypothetical_final_action(self):
        policy, actions, arrays = correction_fixture()
        policy["retention"]["hold_records"][1]["pose_error_increment_rad"][0] = .001
        with self.assertRaises(AssertionError):
            audit.correction_audit(policy, actions, arrays, audit.treatment_boundary(policy), "incremental")

    def test_total_bound_refusal_preserves_prior_correction_and_issues_measured_hold(self):
        policy, actions, arrays = correction_fixture()
        refusal = policy["retention"]["hold_records"][1]
        refusal.update(
            measured_joint_position_rad=[.101, .1, .1, .1, .1, .1],
            raw_pose_joint_targets_rad=[.132, .1, .1, .1, .1, .1],
            pose_error_increment_rad=[.031, 0., 0., 0., 0., 0.],
            proposed_cumulative_correction_rad=[.031, 0., 0., 0., 0., 0.],
            correction_before_update_rad=[0.] * 6, accumulated_correction_rad=[0.] * 6,
            applied_cumulative_correction_rad=None, correction_rad=None,
            correction_update_accepted=False, refused_bounds="joint_correction_exhausted",
            active_regulation=False, joint_targets_rad=[.101, .1, .1, .1, .1, .1])
        policy["events"].append({"type": "hold_invalidated", "reason": "joint_correction_exhausted",
                                 "t_seconds": 1.02})
        policy["admitted"] = False
        actions[1]["joint_targets_rad"] = refusal["joint_targets_rad"]
        final = policy["retention"]["hold_records"][-1]
        final.update(active_regulation=False, correction_before_update_rad=[0.] * 6,
                     accumulated_correction_rad=[0.] * 6, correction_rad=None,
                     preceding_issued_joint_targets_rad=actions[-1]["joint_targets_rad"])
        result = audit.correction_audit(policy, actions, arrays, audit.treatment_boundary(policy), "incremental")
        self.assertEqual(result["bound_refusal_counts"], {"joint_correction_exhausted": 1})

    def test_measured_joint_margin_refusal_must_precede_pose_solve(self):
        policy, actions, arrays = correction_fixture()
        records = policy["retention"]["hold_records"]
        row = records[1]
        row.update(measured_joint_position_rad=[2., .1, .1, .1, .1, .1],
                   raw_pose_joint_targets_rad=None, pose_error_increment_rad=None,
                   proposed_cumulative_correction_rad=None, applied_cumulative_correction_rad=None,
                   correction_update_accepted=False, refused_bounds="measured_joint_margin_exhausted",
                   active_regulation=False, joint_targets_rad=[2., .1, .1, .1, .1, .1],
                   accumulated_correction_rad=[0.] * 6)
        arrays["joint_position_rad"][1019:, 0] = 2.
        arrays["joint_position_rad"][-1, 0] = .1
        policy["events"].append({"type": "hold_invalidated", "reason": "measured_joint_margin_exhausted",
                                 "t_seconds": 1.02})
        policy["admitted"] = False
        actions[1]["joint_targets_rad"] = row["joint_targets_rad"]
        final = records[-1]
        final.update(measured_joint_position_rad=[.1] * 6,
                     active_regulation=False, correction_before_update_rad=[0.] * 6,
                     accumulated_correction_rad=[0.] * 6, correction_rad=None,
                     preceding_issued_joint_targets_rad=actions[-1]["joint_targets_rad"])
        result = audit.correction_audit(policy, actions, arrays, audit.treatment_boundary(policy), "incremental")
        self.assertEqual(result["bound_refusal_counts"], {"measured_joint_margin_exhausted": 1})

    def test_terminal_fallback_latches_once_and_holds_through_measured_drift(self):
        policy, actions, arrays = correction_fixture()
        refusal = policy["retention"]["hold_records"][1]
        refusal.update(
            measured_joint_position_rad=[.101, .1, .1, .1, .1, .1],
            raw_pose_joint_targets_rad=[.132, .1, .1, .1, .1, .1],
            pose_error_increment_rad=[.031, 0., 0., 0., 0., 0.],
            proposed_cumulative_correction_rad=[.031, 0., 0., 0., 0., 0.],
            correction_before_update_rad=[0.] * 6, accumulated_correction_rad=[0.] * 6,
            applied_cumulative_correction_rad=None, correction_rad=None,
            correction_update_accepted=False, refused_bounds="joint_correction_exhausted",
            active_regulation=False, joint_targets_rad=[.101, .1, .1, .1, .1, .1])
        policy["events"].append({"type": "hold_invalidated", "reason": "joint_correction_exhausted",
                                 "t_seconds": 1.02})
        policy["admitted"] = False
        actions[1]["joint_targets_rad"] = refusal["joint_targets_rad"]
        arrays["joint_position_rad"] = np.concatenate(
            [arrays["joint_position_rad"], np.tile([.102, .1, .1, .1, .1, .1], (20, 1)),
             np.tile([.1] * 6, (20, 1))], axis=0)
        arrays["t_seconds"] = np.arange(1, len(arrays["joint_position_rad"]) + 1) * .001
        arrays["joint_position_rad"][audit.row_at(1.04, arrays["t_seconds"])] = [.102, .1, .1, .1, .1, .1]
        arrays["joint_position_rad"][audit.row_at(1.06, arrays["t_seconds"])] = [.1] * 6
        terminal = dict(refusal)
        terminal.update(t_seconds=1.04, final_observation=False, command_issued=True,
                        measured_joint_position_rad=[.102, .1, .1, .1, .1, .1],
                        raw_pose_joint_targets_rad=None, pose_error_increment_rad=None,
                        proposed_cumulative_correction_rad=None, refused_bounds=None,
                        correction_update_accepted=False, joint_targets_rad=refusal["joint_targets_rad"],
                        preceding_issued_joint_targets_rad=None)
        final = policy["retention"]["hold_records"][-1]
        final.update(t_seconds=1.06, measured_joint_position_rad=[.1] * 6,
                     active_regulation=False, correction_before_update_rad=[0.] * 6,
                     accumulated_correction_rad=[0.] * 6, correction_rad=None,
                     preceding_issued_joint_targets_rad=refusal["joint_targets_rad"])
        policy["retention"]["hold_records"].insert(-1, terminal)
        actions.append({"t_seconds": 1.04, "joint_targets_rad": refusal["joint_targets_rad"],
                        "gripper_position_m": policy["frozen_jaw_target_m"]})
        policy["retention"]["hold_records"][-1]["preceding_issued_joint_targets_rad"] = actions[-1]["joint_targets_rad"]
        audit.correction_audit(policy, actions, arrays, audit.treatment_boundary(policy), "incremental")


class LegacyHoldAuditTests(unittest.TestCase):
    def test_admission_target_matches_each_legacy_variant_contract(self):
        issued, measured = [.1] * 6, [.11] * 6
        for variant, expected in (("measured_joint", measured), ("issued_joint", issued),
                                  ("cartesian", issued)):
            with self.subTest(variant=variant):
                policy = {"variant": variant, "frozen_jaw_target_m": .01,
                          "retention": {"hold_reference": {
                              "issued_joint_targets_rad": issued,
                              "measured_joint_position_rad": measured,
                              "frozen_jaw_target_m": .01},
                              "hold_records": [
                                  {"t_seconds": 1., "command_issued": True,
                                   "joint_targets_rad": expected, "frozen_jaw_target_m": .01},
                                  {"t_seconds": 1.02, "command_issued": False,
                                   "final_observation": True, "raw_pose_joint_targets_rad": None,
                                   "joint_targets_rad": expected}]}}
                actions = [{"t_seconds": 1., "joint_targets_rad": expected,
                            "gripper_position_m": .01}]
                boundary = {"time_s": 1., "reason": "admission"}
                audit.baseline_hold_audit(policy, actions, {}, boundary, variant)

    def test_no_admission_has_no_fabricated_final_observation(self):
        result = audit.correction_audit(
            {"variant": "incremental", "candidate": "incremental", "admitted": False,
             "retention": {"hold_records": [], "hold_reference": None,
                           "input_refusals": [], "native_joint_control_limits_rad": [[-2., 2.]] * 6,
                           "update_law": "fixed increment law", "update_gain_per_boundary": 1.}},
            [], {"t_seconds": np.array([.001])},
            {"time_s": .001, "reason": "pre_admission_bound_refusal",
             "detail": "measured_joint_margin_exhausted"}, "incremental")
        self.assertFalse(result["final_observation_no_ik_or_command"])


class SlewAuditTests(unittest.TestCase):
    def test_requested_to_applied_slew_reconstruction_rejects_tampered_native_control(self):
        n = 20000
        arrays = {"arm_actuator_controls_rad": np.zeros((n, 6), dtype=np.float64),
                  "applied_jaw_target_m": np.zeros(n, dtype=np.float64),
                  "requested_jaw_target_m": np.zeros(n, dtype=np.float64),
                  "object_position_m": np.zeros((n, 3), dtype=np.float64),
                  "joint_position_rad": np.zeros((n, 6), dtype=np.float64),
                  "tcp_position_m": np.zeros((n, 3), dtype=np.float64)}
        action = {"t_seconds": 0., "joint_targets_rad": [0.] * 6, "gripper_position_m": 0.}
        actions = [dict(action, t_seconds=i * .02) for i in range(1000)]
        requested = [dict(action, t_seconds=i * .02, index=i) for i in range(1000)]
        trajectory = []
        for index in range(1001):
            trajectory.append({"command": actions[index] if index < 1000 else None,
                               "issued_joint_targets_rad": actions[index]["joint_targets_rad"] if index < 1000 else None,
                               "issued_gripper_position_m": 0. if index < 1000 else None,
                               "object_position_m": [0., 0., 0.], "joint_position_rad": [0.] * 6,
                               "tcp_position_m": [0., 0., 0.]})
        case = {"config": {"joint_target_rate_rad_s": .12, "jaw_target_rate_m_s": .02}}
        actual = audit.reconstruct_controls(arrays, actions, requested, trajectory, [0.] * 7, case)
        self.assertEqual(actual["native_substeps_reconstructed"], 20000)
        arrays["arm_actuator_controls_rad"][50, 0] = 1e-6
        with self.assertRaises(AssertionError):
            audit.reconstruct_controls(arrays, actions, requested, trajectory, [0.] * 7, case)
        policy, actions, arrays = correction_fixture()
        policy["retention"]["hold_records"][0]["joint_targets_rad"][0] += .001
        actions[0]["joint_targets_rad"][0] += .001
        with self.assertRaises(AssertionError):
            audit.correction_audit(policy, actions, arrays, audit.treatment_boundary(policy), "incremental")
        policy, actions, arrays = correction_fixture()
        policy["retention"]["hold_records"][-1]["command_issued"] = True
        with self.assertRaises(AssertionError):
            audit.correction_audit(policy, actions, arrays, audit.treatment_boundary(policy), "incremental")


if __name__ == "__main__":
    unittest.main()

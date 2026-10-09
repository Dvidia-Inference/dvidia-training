"""Native outcomes, delayed completion claims and complete evidence receipts."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from dvidia_training.motor_benchmark import (digest, inspect, inspect_memory,
    make_protocol, run_trial, score_trace, train)
from dvidia_training.motor_memory import ExpectedContract, MotorMemory
from dvidia_training.motor_metrics import summarize


def candidate():
    memory = MotorMemory(400., 35., 0., 15., .01, .02, 12,
                         ExpectedContract(holding_required=True))
    return {"format": "dvidia.motor-memory-candidate", "schema_version": 1,
        "framework_version": "0.2", "memory": memory.to_dict(),
        "empirical_deceleration_m_s2": 1., "physical_robot_ready": False,
        "task_motion_learned_from_video": False}


def trace_row(index, *, x=.16, v=0., goal=.16, observed=None, usable=True, dropped=False):
    return {"time_s": (index+1)*.01, "position_m": x, "velocity_m_s": v,
        "goal_m": goal, "observed_position_m": x if observed is None else observed,
        "observed_velocity_m_s": v, "sensor_usable": usable, "dropped": dropped,
        "observed_acquired_s": (index+1)*.01, "observed_delivered_s": (index+1)*.01,
        "estimated_position_m": x, "estimate_truth_position_m": x, "estimate_time_s": index*.01,
        "interval_force_peak_n": 0., "interval_impulse_ns": 0., "interval_penetration_m": 0.}


class MotorEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.candidate = candidate()
        self.protocol = make_protocol(self.candidate, 1, 17, development_only=True)

    def test_matching_matrix_and_calibration_are_separate(self):
        full = make_protocol(self.candidate, 2, 17)
        self.assertEqual(len(full["cases"]), 12)
        self.assertEqual(len(self.protocol["cases"]), 6)
        self.assertEqual(set(c["split"] for c in self.protocol["cases"]), {"development"})
        self.assertEqual(full["candidate_sha256"], digest(self.candidate))
        self.assertEqual(full["variants"], ["slow_feedback", "fast_fixed", "adaptive_predictive"])
        for args in ((True, 17), (0, 17), (21, 17), (1, True), (1, -1)):
            with self.assertRaises(ValueError):
                make_protocol(self.candidate, *args)

    def test_success_requires_final_dwell_and_retention(self):
        trace = [trace_row(i) for i in range(10)]
        self.assertTrue(score_trace(trace, self.protocol)["success"])
        trace[-1]["position_m"] = .18
        result = score_trace(trace, self.protocol)
        self.assertFalse(result["success"])
        self.assertTrue(result["completion_revoked"])
        self.assertTrue(result["rest_lost_after_confirmation"])
        trace[-1]["position_m"] = .16
        trace[-1]["dropped"] = True
        self.assertFalse(score_trace(trace, self.protocol)["success"])

    def test_noisy_or_delayed_claim_is_not_ground_truth_completion(self):
        trace = [trace_row(i, x=.14, observed=.16) for i in range(10)]
        result = score_trace(trace, self.protocol)
        self.assertEqual(result["completed_s"], .05)
        self.assertTrue(result["completion_false"])
        self.assertFalse(result["success"])
        self.assertIsNone(result["truth_settled_s"])

    def test_goal_change_resets_dwell_but_preserves_earlier_claim(self):
        trace = [trace_row(i) for i in range(6)]
        trace.extend(trace_row(i, goal=.20) for i in range(6, 10))
        result = score_trace(trace, self.protocol)
        self.assertIsNone(result["completed_s"])
        self.assertTrue(result["completion_revoked"])
        self.assertEqual(result["completion_claims"], [{"time_s": .05, "goal_m": .16, "false": False,
            "truth_error_m": 0., "truth_speed_m_s": 0.}])

    def test_missing_input_and_nonfinite_or_omitted_assessment_are_explicit(self):
        trace = [trace_row(i, usable=False) for i in range(10)]
        self.assertIsNone(score_trace(trace, self.protocol)["completed_s"])
        self.assertAlmostEqual(score_trace(trace, self.protocol)["sensor_unusable_s"], .1)
        trace[3]["time_s"] = .055
        with self.assertRaisesRegex(ValueError, "unordered"):
            score_trace(trace, self.protocol)
        trace[3]["time_s"] = .04
        trace[3]["position_m"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            score_trace(trace, self.protocol)

    def evidence(self, directory):
        p = self.protocol
        records = []
        for i, case in enumerate(p["cases"]):
            for tier, speed in enumerate(p["speed_limits_m_s"]):
                for mode in p["variants"]:
                    record = {"case_id": case["id"], "split": case["split"], "variant": mode,
                        "speed_limit_m_s": speed, "seed": p["seed"]+i*10+tier,
                        "holding": True, "expected_completion": case["expected_completion"],
                        "target_final_m": case.get("updated_goal_m", case["goal_m"]),
                        "valid": False, "error": "deliberate test engine failure", "trace": [],
                        "simulated_seconds": 0., "wall_seconds": .01, "step_count": 0}
                    record.update(score_trace([], p))
                    records.append(record)
        sources = {"motor_memory.py": "a"*64}
        freeze = {"candidate_sha256": digest(self.candidate), "protocol_sha256": digest(p), "source_sha256": sources}
        report = {"format": "dvidia.motor-benchmark", "schema_version": 1, "framework_version": "0.2",
            "physical_robot_ready": False, "benchmark_qualification": False,
            "protocol": p, "protocol_sha256": digest(p), "candidate": self.candidate,
            "candidate_sha256": digest(self.candidate), "source_sha256": sources,
            "freeze": freeze, "records": records, "summary": summarize(records)}
        values = {"protocol.json": p, "memory.json": self.candidate, "sources.json": sources,
                  "freeze.json": freeze, "report.json": report}
        for name, value in values.items():
            (directory/name).write_text(json.dumps(value))
        (directory/"protocol.sha256").write_text(digest(p))
        (directory/"trials.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
        (directory/"report.html").write_text("<p>synthetic test fixture</p>")
        self.manifest(directory)
        return report

    def manifest(self, directory):
        value = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.name != "evidence-manifest.json"}
        (directory/"evidence-manifest.json").write_text(json.dumps(value))

    def test_inspection_recomputes_and_refuses_missing_attempts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = self.evidence(root)
            self.assertEqual(inspect(root)["attempted"], 54)
            records = report["records"][:-1]
            report["records"] = records
            report["summary"] = summarize(records)
            (root/"report.json").write_text(json.dumps(report))
            (root/"trials.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
            self.manifest(root)
            with self.assertRaisesRegex(ValueError, "planned attempts"):
                inspect(root)

    def test_inspection_refuses_changed_scope_paths_and_recomputed_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = self.evidence(root)
            report["physical_robot_ready"] = True
            (root/"report.json").write_text(json.dumps(report))
            self.manifest(root)
            with self.assertRaisesRegex(ValueError, "scope/readiness"):
                inspect(root)
            report["physical_robot_ready"] = False
            report["summary"]["attempted"] = 100
            (root/"report.json").write_text(json.dumps(report))
            self.manifest(root)
            with self.assertRaisesRegex(ValueError, "recomputed"):
                inspect(root)


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "optional native arm extra")
class NativeMotorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.memory_dir = Path(cls.temp.name)/"memory"
        cls.candidate = train(cls.memory_dir, seed=18942)
        cls.protocol = make_protocol(cls.candidate, 1, 18943, development_only=True)
        cls.records = {}
        for case_id, variants in (("dev_mid", ("fast_fixed", "adaptive_predictive")),
                                  ("dev_delay", ("fast_fixed", "adaptive_predictive")),
                                  ("dev_missing", ("adaptive_predictive",))):
            case = next(c for c in cls.protocol["cases"] if c["id"] == case_id)
            for mode in variants:
                cls.records[case_id, mode] = run_trial(cls.protocol, cls.candidate, case, mode, .20, 18943)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_native_calibration_is_refittable_and_keeps_braking_distinct(self):
        self.assertEqual(inspect_memory(self.memory_dir), self.candidate)
        self.assertEqual(self.candidate["memory"]["sample_count"], 8000)
        self.assertGreater(self.candidate["empirical_deceleration_m_s2"], 0)
        self.assertNotEqual(self.candidate["empirical_deceleration_m_s2"], self.candidate["memory"]["max_acceleration_m_s2"])

    def test_native_primitive_continues_physics_to_true_goal_and_rest(self):
        for mode in ("fast_fixed", "adaptive_predictive"):
            r = self.records["dev_mid", mode]
            self.assertTrue(r["valid"], r["error"])
            self.assertEqual(r["step_count"], 4000)
            self.assertEqual(len(r["trace"]), 400)
            self.assertTrue(r["success"])
            self.assertFalse(r["dropped"])
            self.assertLessEqual(r["final_error_m"], .003)
            self.assertLessEqual(r["final_speed_m_s"], .005)

    def test_delayed_estimation_is_measured_at_same_time_as_truth(self):
        r = self.records["dev_delay", "adaptive_predictive"]
        self.assertTrue(r["valid"], r["error"])
        self.assertGreater(r["predictor_rmse_m"], 0)
        self.assertTrue(all(row["estimate_time_s"] < row["time_s"] for row in r["trace"] if row["estimated_position_m"] is not None))
        self.assertEqual({k: r[k] for k in score_trace(r["trace"], self.protocol)}, score_trace(r["trace"], self.protocol))

    def test_sensor_fault_latches_delayed_hold_without_instant_freeze(self):
        r = self.records["dev_missing", "adaptive_predictive"]
        self.assertTrue(r["valid"], r["error"])
        self.assertIsNotNone(r["reaction_s"])
        self.assertAlmostEqual(r["hold_applied_s"]-r["reaction_s"], .02)
        self.assertEqual(r["step_count"], 4000)
        self.assertFalse(r["success"])
        self.assertIsNone(r["completed_s"])
        self.assertGreater(r["sensor_unusable_s"], 0)


if __name__ == "__main__":
    unittest.main()

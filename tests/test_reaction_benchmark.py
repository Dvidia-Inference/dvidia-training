"""Evidence integrity and native reaction/braking behavior, including failures."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dvidia_training.reaction_benchmark import (_hash, inspect, make_protocol,
                                                paired_comparisons, run_trial)
from dvidia_training.reaction_metrics import summarize


class ProtocolEvidenceTests(unittest.TestCase):
    def test_protocol_has_hazard_negative_fault_and_matched_control_cases(self):
        p = make_protocol(2, 41, convergence=True)
        self.assertFalse(p["confirmation"])
        self.assertTrue(p["convergence_enabled"])
        self.assertEqual(len(p["cases"]), 13)
        self.assertEqual(len({c["id"] for c in p["cases"]}), 13)
        self.assertEqual(p["variants"], ["monitor", "disabled_control"])
        self.assertTrue(any(not c["positive"] for c in p["cases"]))
        self.assertEqual({c["group"] for c in p["cases"]}, {"motion", "contact", "grip", "sensor_fault"})
        for args in ((True, 4), (0, 4), (31, 4), (1, True), (1, -1)):
            with self.assertRaises(ValueError):
                make_protocol(*args)

    def evidence(self, directory):
        protocol = make_protocol(1, 3)
        protocol["cases"] = [{"id": "fixture", "group": "motion", "positive": False}]
        record = {"case_id": "fixture", "variant": "monitor", "group": "motion", "seed": 3,
            "valid": True, "expected_detection": False, "detected_s": None,
            "peak_unexpected_force_n": 0., "unexpected_contact_impulse_ns": 0.,
            "minimum_clearance_m": .1, "max_penetration_m": 0.,
            "simulated_seconds": 1., "wall_seconds": .5, "step_count": 1000}
        sources = {"reaction.py": "a"*64}
        records = [record, dict(record, variant="disabled_control")]
        report = {"format": "dvidia.reaction-benchmark", "schema_version": 1,
            "framework_version": "0.1", "benchmark_qualification": False,
            "protocol": protocol, "protocol_sha256": _hash(protocol), "source_sha256": sources,
            "records": records, "summary": summarize(records), "paired_comparisons": paired_comparisons(records),
            "scope": "synthetic", "physical_robot_ready": False}
        values = {"protocol.json": json.dumps(protocol), "protocol.sha256": _hash(protocol),
            "sources.json": json.dumps(sources), "trials.jsonl": "".join(json.dumps(r)+"\n" for r in records),
            "report.json": json.dumps(report), "report.html": "<p>fixture</p>"}
        for name, value in values.items():
            (directory/name).write_text(value)
        self.manifest(directory)
        return report

    def manifest(self, directory):
        manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.name != "evidence-manifest.json"}
        (directory/"evidence-manifest.json").write_text(json.dumps(manifest))

    def test_inspection_checks_bytes_and_recomputed_measurements(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = self.evidence(root)
            self.assertTrue(inspect(root)["consistent"])
            (root/"report.html").write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed evidence"):
                inspect(root)
            self.manifest(root)
            report["summary"]["attempted"] = 90
            (root/"report.json").write_text(json.dumps(report))
            self.manifest(root)
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                inspect(root)

    def test_manifest_cannot_reference_an_external_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.evidence(root)
            manifest = json.loads((root/"evidence-manifest.json").read_text())
            manifest["../outside.json"] = "a"*64
            (root/"evidence-manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "local filenames"):
                inspect(root)

    def test_rewritten_ready_claim_or_omitted_attempt_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = self.evidence(root)
            report["physical_robot_ready"] = True
            (root/"report.json").write_text(json.dumps(report))
            self.manifest(root)
            with self.assertRaisesRegex(ValueError, "scope or readiness"):
                inspect(root)
            report["physical_robot_ready"] = False
            report["records"] = report["records"][:1]
            (root/"trials.jsonl").write_text(json.dumps(report["records"][0])+"\n")
            (root/"report.json").write_text(json.dumps(report))
            self.manifest(root)
            with self.assertRaisesRegex(ValueError, "planned attempts"):
                inspect(root)

    def test_invalid_pair_is_retained_with_unmeasured_deltas(self):
        records = [{"case_id": "a", "seed": 2, "variant": "monitor", "valid": False},
                   {"case_id": "a", "seed": 2, "variant": "disabled_control", "valid": True}]
        pair = paired_comparisons(records)[0]
        self.assertFalse(pair["both_valid"])
        self.assertIsNone(pair["minimum_clearance_m_monitor_minus_control"])


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "optional native arm extra")
class NativeReactionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = make_protocol(1, 48203)
        cls.records = {}
        for case_id, variants in (("crossing", ("monitor", "disabled_control")),
                                  ("slip", ("monitor", "disabled_control")),
                                  ("stale_motion", ("monitor",)),
                                  ("late_occluded_entry", ("monitor",))):
            case = next(c for c in cls.protocol["cases"] if c["id"] == case_id)
            for variant in variants:
                cls.records[case_id, variant] = run_trial(cls.protocol, case, variant, 48203)

    def test_anticipation_changes_native_contact_without_changing_positive_label(self):
        active, control = (self.records["crossing", v] for v in ("monitor", "disabled_control"))
        self.assertTrue(active["valid"], active["error"])
        self.assertTrue(control["valid"], control["error"])
        self.assertEqual(active["expected_detection"], control["expected_detection"])
        self.assertLessEqual(active["peak_unexpected_force_n"], .001)
        self.assertGreater(control["peak_unexpected_force_n"], 10.)
        self.assertIsNone(control["brake_applied_s"])
        self.assertIsNone(control["stopped_s"])

    def test_stop_preserves_motion_and_runs_every_native_tick_to_fixed_horizon(self):
        r = self.records["crossing", "monitor"]
        self.assertAlmostEqual(r["brake_applied_s"]-r["brake_requested_s"], .02)
        self.assertGreater(r["stopped_s"]-r["brake_applied_s"], .05)
        self.assertGreater(r["stopping_distance_m"], .001)
        self.assertGreater(r["request_to_rest_path_m"], r["stopping_distance_m"])
        self.assertGreater(r["max_forward_excursion_m"], r["stopping_displacement_m"])
        self.assertEqual(r["step_count"], 1400)
        self.assertAlmostEqual(r["simulated_seconds"], 1.4)
        self.assertLess(r["trace"][-1]["velocity_m_s"][0], .005)
        braking = next(row for row in r["trace"] if row["braking"])
        self.assertGreater(abs(braking["velocity_m_s"][0]), .005)

    def test_slip_stop_does_not_claim_load_recovery(self):
        r = self.records["slip", "monitor"]
        self.assertTrue(r["valid"], r["error"])
        self.assertIn(r["detected_reason"], ("grip_slip", "grip_lost"))
        self.assertTrue(r["dropped"])
        self.assertIsNotNone(r["stopped_s"])
        self.assertTrue(self.records["slip", "disabled_control"]["dropped"])

    def test_stale_signal_is_visible_as_unusable_without_fabricated_dropout(self):
        r = self.records["stale_motion", "monitor"]
        self.assertEqual(r["detected_reason"], "stale_motion")
        self.assertTrue(r["observation_available"])
        self.assertFalse(r["required_observations_fresh"])
        self.assertEqual(r["observation_blackout_seconds"], 0.)
        self.assertGreater(r["observation_stale_seconds"], .8)

    def test_numerical_validity_does_not_hide_failed_contact_diagnostics(self):
        r = self.records["late_occluded_entry", "monitor"]
        self.assertTrue(r["valid"], r["error"])
        self.assertFalse(r["contact_diagnostic_pass"])
        self.assertGreater(r["max_penetration_m"], .003)
        self.assertEqual(summarize([r])["overall"]["contact_diagnostics"]["exceeded"], 1)

    def test_horizon_and_engine_errors_remain_missing_or_invalid(self):
        p = copy.deepcopy(self.protocol)
        p["fixture_config"]["horizon_s"] = .5
        case = next(c for c in p["cases"] if c["id"] == "slip")
        r = run_trial(p, case, "monitor", 48203)
        self.assertTrue(r["valid"], r["error"])
        self.assertIsNotNone(r["brake_applied_s"])
        self.assertIsNone(r["stopped_s"])
        self.assertIsNone(r["stopping_distance_m"])
        with patch("dvidia_training.reaction_native.NativeReactionFixture", side_effect=RuntimeError("test invalid engine")):
            invalid = run_trial(self.protocol, case, "monitor", 48203)
        self.assertFalse(invalid["valid"])
        self.assertIn("test invalid engine", invalid["error"])
        self.assertEqual(summarize([invalid])["overall"]["detection"]["invalid_positive"], 1)


if __name__ == "__main__":
    unittest.main()

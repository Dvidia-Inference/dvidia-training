"""Publication semantics and bounded rendering for local-verifier evidence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from dvidia_training import local_goal_verifier as verifier
from dvidia_training.local_goal_verifier_report import build_report


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


def fixture(root):
    root.mkdir()
    (root / "scene.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    goal = verifier.GoalContract("green-shoe", "workstation", (.4, -.1), 0.)
    packet = {"schema": verifier.SCHEMA, "units": dict(verifier.UNITS), "sample_period_s": .002,
        "uncertainty_bounds": {key: 0. for key in ("position_m", "yaw_rad", "linear_speed_m_s", "angular_speed_rad_s", "table_support_force_n", "robot_contact_force_n")},
        "history_start_s": .4, "samples": [{"t_s": .4 + (i + 1) * .002, "object_id": "green-shoe", "frame_id": "workstation", "presence": "present", "position_m": [.4, -.1, 0.], "yaw_rad": 0., "linear_speed_m_s": 0., "angular_speed_rad_s": 0., "table_support_force_n": .2943, "robot_contact_force_n": 0.} for i in range(100)]}
    write_json(root / "packet.json", packet)
    write_json(root / "state.json", {"complete": True})
    complete = verifier.verify_goal(packet, contract=goal, now_s=.6)
    stale = verifier.verify_goal(packet, contract=goal, now_s=1.)
    moving_packet = deepcopy(packet)
    moving_packet["samples"][-1]["linear_speed_m_s"] = .2
    moving = verifier.verify_goal(moving_packet, contract=goal, now_s=.6)
    rows, cases = [], []
    specifications = [("native-positive", "native", True, "complete", complete), ("native-negative", "native", False, "incomplete", moving), ("stale-positive", "evidence_fault", True, "unknown", stale), ("forged-negative", "blind_spot", False, "complete", complete)]
    for identifier, cohort, physically_complete, expected, decision in specifications:
        cases.append({"id": identifier, "cohort": cohort, "scenario": identifier, "layout_id": "layout-01", "contract": {"object_id": "green-shoe", "frame_id": "workstation", "goal_xy_m": [.4, -.1], "goal_yaw_rad": 0.}, "now_s": 1. if cohort == "evidence_fault" else .6, "packet_file": "packet.json", "state_file": "state.json", "current_image": "scene.png", "goal_image": "scene.png", "physical_truth": {"present": True, "complete": physically_complete}, "expected_verdict": expected, "note": "A synthetic diagnostic case.", "generation_valid": True})
        rows.append({"case_id": identifier, "decision": decision, "timing_ns": [100000, 200000, 300000]})
    catalog = {"schema": "dvidia.local-goal-cases/v1", "cases": cases}
    result = {"schema": "dvidia.local-goal-results/v1", "status": "completed", "results": rows}
    write_json(root / "cases.json", catalog)
    write_json(root / "results.json", result)
    return catalog, result


class LocalGoalVerifierReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve() / "run"
        self.out = Path(self.temp.name).resolve() / "viewer"
        self.cases, self.results = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def build(self):
        build_report(self.root, self.out)
        return json.loads((self.out / "report.json").read_bytes())

    def test_separate_denominators_expose_forgery_and_correct_abstention(self):
        report = self.build()
        native, fault, forged = report["summaries"]
        self.assertEqual(native["positive_recognition"], {"correct": 1, "total": 1})
        self.assertEqual(native["negative_recognition"], {"correct": 1, "total": 1})
        self.assertEqual(fault["contract_agreement"], {"correct": 1, "total": 1})
        self.assertEqual(fault["positive_recognition"], {"correct": 0, "total": 1})
        self.assertEqual(fault["expected_abstention"], {"correct": 1, "total": 1})
        self.assertEqual(forged["contract_agreement"], {"correct": 1, "total": 1})
        self.assertEqual(forged["false_completion"], {"count": 1, "total": 1})
        self.assertEqual(forged["physical_agreement"]["correct"], 0)
        self.assertEqual(native["planned"], 2)
        self.assertEqual(native["cpu_latency_us"]["n"], 6)
        self.assertEqual(native["cpu_latency_us"]["p95"], 300.)

    def test_renderer_retains_reason_and_uncertainty_without_private_fields(self):
        self.results["results"][0]["decision"]["authorization"] = "private-marker"
        self.results["results"][0]["response"] = "private-marker"
        self.cases["cases"][0]["secret"] = "private-marker"
        write_json(self.root / "cases.json", self.cases)
        write_json(self.root / "results.json", self.results)
        report = self.build()
        self.assertNotIn("private-marker", (self.out / "index.html").read_text())
        positive = report["cases"][0]["decision"]
        self.assertIn("interval", positive["predicates"]["xy_position"])
        self.assertIn("stale_observation", report["cases"][2]["decision"]["reasons"])
        self.assertEqual(positive["window"]["evaluated_sample_count"], 100)
        html = (self.out / "index.html").read_text()
        for text in ("Requires 200 ms observation history", "Zero Jev calls", "Images illustrate scenes", "Consistent fabricated readings", "Physically complete", "Freshness or history fault"):
            self.assertIn(text, html)

    def test_unknown_and_incomplete_decisions_cannot_disappear_from_denominators(self):
        self.results["results"][0]["decision"]["verdict"] = "unknown"
        write_json(self.root / "results.json", self.results)
        native = self.build()["summaries"][0]
        self.assertEqual(native["positive_recognition"], {"correct": 0, "total": 1})
        self.assertEqual(native["contract_agreement"], {"correct": 1, "total": 2})
        self.assertEqual(native["physical_agreement"]["answered"], 1)

    def test_incomplete_or_active_runs_rejected(self):
        for changes in ({"status": "running"}, {"active_trial": "native-positive"}):
            with self.subTest(changes=changes):
                write_json(self.root / "results.json", {**self.results, **changes})
                with self.assertRaises(ValueError): self.build()

    def test_duplicate_missing_and_unplanned_rows_rejected(self):
        mutations = [self.results["results"][:-1], self.results["results"] + self.results["results"][:1], self.results["results"] + [{**self.results["results"][0], "case_id": "extra"}]]
        for rows in mutations:
            with self.subTest(count=len(rows)):
                write_json(self.root / "results.json", {**self.results, "results": rows})
                with self.assertRaises(ValueError): self.build()

    def test_path_traversal_and_symlink_images_rejected(self):
        self.cases["cases"][0]["current_image"] = "../outside.png"
        write_json(self.root / "cases.json", self.cases)
        with self.assertRaises(ValueError): self.build()
        (self.root / "alias.png").symlink_to(self.root / "scene.png")
        self.cases["cases"][0]["current_image"] = "alias.png"
        write_json(self.root / "cases.json", self.cases)
        with self.assertRaises(ValueError): self.build()

    def test_nonfinite_huge_boolean_and_missing_timing_rejected(self):
        for values in ([], [True], [-1], [0], [1.1], [10**500]):
            with self.subTest(values=str(values)[:20]):
                changed = deepcopy(self.results)
                changed["results"][0]["timing_ns"] = values
                write_json(self.root / "results.json", changed)
                with self.assertRaises((ValueError, OverflowError)): self.build()

    def test_free_text_cannot_close_embedded_json_or_inject_markup(self):
        self.cases["cases"][0]["note"] = '</script><script>alert("injection")</script>'
        write_json(self.root / "cases.json", self.cases)
        self.build()
        html = (self.out / "index.html").read_text()
        self.assertNotIn('</script><script>alert', html)
        self.assertIn('\\u003c/script\\u003e', html)
        self.assertNotIn('innerHTML', html)
        self.assertIn('Reference completed placement for layout', html)
        self.assertIn('?.focus()', html)
        self.assertIn("le:'≤'", html)

    def test_fresh_output_and_checksums(self):
        manifest = build_report(self.root, self.out)
        from hashlib import sha256
        for name, evidence in manifest["files"].items():
            raw = (self.out / name).read_bytes()
            self.assertEqual(evidence, {"bytes": len(raw), "sha256": sha256(raw).hexdigest()})
        with self.assertRaises(ValueError): build_report(self.root, self.out)
        with self.assertRaises(ValueError): build_report(self.root, self.root / "nested")


if __name__ == "__main__": unittest.main()

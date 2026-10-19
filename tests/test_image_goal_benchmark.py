"""Failure denominators, abstention and immutable paired evaluation contracts."""
import copy
from pathlib import Path
import tempfile
import unittest

from dvidia_training import image_goal_benchmark as bench


def case(identity, *, present=True, complete=False, expected="incomplete"):
    return {"id": identity, "category": identity, "physical_truth": {"present": present, "complete": complete},
        "measured_inputs": {}, "expected_verdict": {condition: expected for condition in bench.CONDITIONS},
        "expected_presence": {condition: "present" if present else "absent" for condition in bench.CONDITIONS}}


def row(identity, *, verdict="incomplete", presence="present", status="ok", latency=10.):
    return {"case_id": identity, "condition": "rgb", "verdict": verdict, "presence": presence,
            "status": status, "online_total_ms": latency, "latency_ms": {"http": latency},
            "p_present": .9, "p_complete": .9 if verdict == "complete" else .1, "estimated_input_cost_usd": .0001}


def paired(rows):
    return [dict(row, condition=condition) for row in rows for condition in bench.CONDITIONS]


class ScoreTests(unittest.TestCase):
    def test_errors_and_abstention_cannot_improve_reported_scheduled_accuracy(self):
        cases = [case("missing", present=False), case("moving", expected="unknown"),
                 case("good", complete=True, expected="complete"), case("bad")]
        rows = [row("missing", verdict="complete"), row("moving", verdict="unknown"),
                row("good", verdict="complete"), row("bad", status="error", verdict="unknown", latency=20000)]
        result = bench.score(paired(rows), cases)["overall"]
        self.assertEqual(result["physical_false_completion"], {"numerator": 2, "denominator": 6, "rate": 1/3})
        self.assertEqual(result["missing_object_false_presence"]["rate"], 1.)
        self.assertEqual(result["observability_aware_accuracy"]["rate"], .5)
        self.assertEqual(result["completion_coverage"]["rate"], .5)
        self.assertEqual(result["selective_physical_accuracy"]["rate"], .5)
        self.assertEqual(result["ambiguous_abstention"]["rate"], 1.)
        self.assertEqual(result["online_ms_all_attempts"]["max"], 20000)
        self.assertEqual(result["online_ms_successful_requests"]["max"], 10)
        self.assertEqual(result["statuses"], {"ok": 6, "error": 2})

    def test_always_abstain_has_no_selective_accuracy_and_no_success_recall(self):
        cases = [case("good", complete=True, expected="complete"), case("bad")]
        result = bench.score(paired([row(c["id"], verdict="unknown") for c in cases]), cases)["overall"]
        self.assertIsNone(result["selective_physical_accuracy"]["rate"])
        self.assertEqual(result["completion_coverage"]["rate"], 0.)
        self.assertEqual(result["physical_true_completion"]["rate"], 0.)
        self.assertEqual(result["physical_false_completion"]["rate"], 0.)

    def test_no_data_is_unavailable_not_perfect(self):
        result = bench.score([], [])["overall"]
        self.assertIsNone(result["physical_false_completion"]["rate"])
        self.assertIsNone(result["online_ms_all_attempts"]["p50"])
        self.assertIsNone(result["known_usage_estimated_input_cost_usd"])

    def test_duplicate_or_unknown_trials_rejected(self):
        with self.assertRaises(ValueError):
            bench.score([row("one"), row("one")], [case("one")])
        with self.assertRaises(ValueError):
            bench.score([row("two")], [case("one")])

    def test_missing_scheduled_result_and_nonfinite_probability_are_rejected(self):
        with self.assertRaises(ValueError):
            bench.score([row("one")], [case("one")])
        rows = paired([row("one")]); rows[0]["p_complete"] = float("nan")
        with self.assertRaises(ValueError):
            bench.score(rows, [case("one")])

    def test_presence_and_completion_truth_are_consistent(self):
        bad = case("bad", present=False, complete=True)
        with self.assertRaises(ValueError):
            bench.validate_cases({"cases": [bad]})
        bad = case("bad")
        bad["physical_truth"]["complete"] = 0
        with self.assertRaises(ValueError):
            bench.validate_cases({"cases": [bad]})

    def test_artifact_path_cannot_escape_fixtures(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); fixtures = base / "fixtures"; fixtures.mkdir()
            (base / "secret.txt").write_text("not a fixture")
            for name in ("../secret.txt", str(base / "secret.txt")):
                with self.assertRaises(ValueError):
                    bench.relative_file(fixtures, name)
            (fixtures / "link").symlink_to(base / "secret.txt")
            with self.assertRaises(ValueError):
                bench.relative_file(fixtures, "link")


if __name__ == "__main__":
    unittest.main()

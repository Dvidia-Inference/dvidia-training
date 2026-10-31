import contextlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dvidia_training import arm_interface_benchmark as benchmark


class SettingsTests(unittest.TestCase):
    def test_common_fixed_roster_and_timing_contract(self):
        settings = benchmark._settings(list(benchmark.PROFILES), .4, 2)
        self.assertEqual(settings["control_periods_s"], [.01, .02, .04])
        self.assertEqual(settings["physics_timestep_s"], .001)
        self.assertEqual(settings["cases"], ["hold", "arm_perturbation", "gripper_cycle"])
        for period in settings["control_periods_s"]:
            self.assertEqual(round(settings["duration_s"]/period)*round(period/.001), 400)

    def test_invalid_settings_rejected(self):
        for duration in (True, .2, .41, math.nan, math.inf, 3.):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                benchmark._settings(["piper"], duration, 2)
        for profiles in ([], ["unknown"], ["piper", "piper"], "piper"):
            with self.subTest(profiles=profiles), self.assertRaises(ValueError):
                benchmark._settings(profiles, .4, 2)
        for repeats in (True, 0, 4, 1.5):
            with self.subTest(repeats=repeats), self.assertRaises(ValueError):
                benchmark._settings(["piper"], .4, repeats)


class MeasurementTests(unittest.TestCase):
    def test_waveforms_preserve_units_limits_and_common_endpoints(self):
        home, limits = [0., 1., 0.], [[-1., 1.], [.99, 1.01], [0., 255.]]
        amplitude = benchmark._amplitudes(home, limits, [0, 1], [2])
        self.assertAlmostEqual(amplitude[0], .04)
        self.assertAlmostEqual(abs(amplitude[1]), .008)
        self.assertEqual(amplitude[2], 51.)
        for case in benchmark.CASES:
            for step in range(401):
                command = benchmark._command(case, step*.001, .4, home, amplitude, [0, 1], [2])
                self.assertTrue(all(low-1e-12 <= value <= high+1e-12 for value, (low, high) in zip(command, limits)))
            for elapsed in (0., .4):
                command = benchmark._command(case, elapsed, .4, home, amplitude, [0, 1], [2])
                for actual, expected in zip(command, home):
                    self.assertAlmostEqual(actual, expected)

    def test_robotiq_input_coordinate_is_not_misreported_as_radians_or_gap(self):
        gain, bias = .3137255, -100.
        tendon_angle = 51.*gain/100.
        self.assertAlmostEqual(benchmark._servo_equivalent(tendon_angle, gain, 0., bias), 51.)
        self.assertAlmostEqual(benchmark._servo_equivalent(.007, 1000., 0., -1000.), .007)
        for args in ((math.nan, 1., 0., -1.), (0., 0., 0., -1.), (0., 1., 0., 0.)):
            with self.assertRaises(ValueError):
                benchmark._servo_equivalent(*args)

    def test_accuracy_groups_never_mix_arm_and_gripper_units(self):
        rows = [{"tracking_error": {
            "arm_held_target_rad": [3., 4.], "arm_continuous_reference_rad": [0., 2.],
            "gripper_held_target_native": 100., "gripper_continuous_reference_native": 50.,
            "terminal_position_reference_m": .001, "terminal_orientation_reference_rad": .2}}]
        metrics = benchmark._metrics(rows, "native_0_255")
        self.assertEqual(metrics["arm_held_target"]["unit"], "rad")
        self.assertAlmostEqual(metrics["arm_held_target"]["rms"], math.sqrt(12.5))
        self.assertEqual(metrics["gripper_held_target"]["unit"], "native_0_255")
        self.assertEqual(metrics["gripper_held_target"]["rms"], 100.)
        self.assertEqual(metrics["terminal_position_continuous_reference"]["unit"], "m")

    def test_percentiles_are_explicit_and_empty_intervals_do_not_invent_measurements(self):
        timing = benchmark._percentiles([.04, .01, .02, .03])
        self.assertEqual(timing["count"], 4)
        self.assertAlmostEqual(timing["p50"], .025)
        self.assertAlmostEqual(timing["p95"], .0385)
        self.assertAlmostEqual(timing["p99"], .0397)
        self.assertIsNone(benchmark._percentiles([])["p95"])

    def test_failed_nonfinite_state_is_retained_without_invalid_json_numbers(self):
        state = benchmark._safe_state(SimpleNamespace(time=.1, qpos=[math.nan, math.inf, -math.inf], qvel=[0.], ctrl=[1.]))
        self.assertEqual(state["qpos_native"], ["nan", "inf", "-inf"])
        json.dumps(state, allow_nan=False)


class PreservationTests(unittest.TestCase):
    def test_failed_builds_preserve_every_scheduled_attempt_without_retries(self):
        from dvidia_training import reference_sim
        builder = SimpleNamespace(build=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("fixture build failure")))
        with tempfile.TemporaryDirectory() as tmp, patch.object(benchmark, "_source_snapshot", return_value=[]), \
                patch.object(benchmark, "_interfaces", return_value=builder), \
                patch.object(reference_sim, "native_warnings", side_effect=lambda _messages: contextlib.nullcontext()):
            output = Path(tmp)/"study"
            report = benchmark.benchmark("missing-assets", output, profiles=["piper"], duration=.4, repeats=1)
            self.assertEqual(report["summary"]["attempts"], 9)
            self.assertEqual(report["summary"]["failed_attempts"], 9)
            self.assertEqual(len(list(output.glob("attempt-*/started.json"))), 9)
            self.assertEqual(len(list(output.glob("attempt-*/result.json"))), 9)
            self.assertEqual(len(list(output.glob("attempt-*/trace.json"))), 9)
            for attempt in report["attempts"]:
                self.assertFalse(attempt["valid"])
                self.assertEqual(attempt["error"]["stage"], "build")
                self.assertEqual(attempt["trace"]["state_rows"], 0)
            completion = json.loads((output/"completion.json").read_text())
            self.assertEqual(completion["recorded_attempts"], 9)
            self.assertEqual(completion["valid_attempts"], 0)
            with self.assertRaises(ValueError):
                benchmark.benchmark("missing-assets", output, profiles=["piper"])

    def test_invalid_asset_mapping_creates_no_partial_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"study"
            with self.assertRaises(ValueError):
                benchmark.benchmark({"piper": "assets"}, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

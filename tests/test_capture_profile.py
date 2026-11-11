"""Check benchmark units and deadline accounting without native graphics."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('capture_profile',
    Path(__file__).resolve().parents[1]/'scripts/profile-operator-capture.py')
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


class CaptureProfileTests(unittest.TestCase):
    def test_full_interval_deadlines_use_seconds_and_include_only_overruns(self):
        result = profile.summary([.001, .020, .021, .100], .020)
        self.assertEqual(result['count'], 4)
        self.assertEqual(result['deadline_ms'], 20.)
        self.assertEqual(result['deadline_misses'], 2)
        self.assertEqual(result['max_ms'], 100.)
        self.assertAlmostEqual(result['total_seconds'], .142)
        self.assertEqual(profile.summary([]), {'count':0})

    def test_thread_requests_do_not_claim_measured_thread_count(self):
        runtime = profile.numerical_runtime()
        self.assertIn('blas', runtime['build_dependencies'])
        self.assertIsNone(runtime['effective_thread_count'])
        memory = profile.peak_memory()
        if memory['status'] == 'available':
            self.assertGreater(memory['bytes'], 0)


if __name__ == '__main__':
    unittest.main()

"""Benchmark failure retention is part of the publication contract."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from dvidia_training import workstation_benchmark as bench


class WorkstationBenchmarkTests(unittest.TestCase):
    def test_protocol_retains_all_targets_and_negative_control(self):
        plan=bench.protocol(10)
        self.assertEqual(len(plan['attempts']),20)
        self.assertEqual(sum(case['kind']=='tracking' for case in plan['attempts']),16)
        self.assertEqual(sum(case['kind']=='gravity_control' for case in plan['attempts']),1)
        for invalid in (True,0,9,601,10.5):
            with self.subTest(value=invalid),self.assertRaises(ValueError):bench.protocol(invalid)

    def test_all_failed_model_loads_still_produce_complete_report(self):
        try:
            bench.sim._native_imports()
        except ImportError:
            self.skipTest('Optional native runtime is not installed')
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'evidence'
            with patch.object(bench.ws,'build_model',side_effect=ValueError('deliberate missing asset fixture')):
                result=bench.benchmark('fixture',output,stability_seconds=10)
            self.assertEqual(result['attempts'],20)
            self.assertEqual(result['valid_attempts'],0)
            self.assertEqual(result['tracking']['unmeasured_attempts'],16)
            self.assertIsNone(result['tracking']['worst_position_error_m'])
            self.assertIsNone(result['timing']['compute_ms_max'])
            self.assertEqual(result['timing']['intervals'],0)
            attempts=json.loads((output/'attempts.json').read_text())
            self.assertEqual(len(attempts),20)
            self.assertTrue(all(not item['passed'] and item['errors'] for item in attempts))
            self.assertTrue((output/'integrity.json').is_file())


if __name__=='__main__':unittest.main()

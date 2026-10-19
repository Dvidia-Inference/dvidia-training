"""Paired metric denominator and RGB fixture integrity checks."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from dvidia_training.sensor_goal_benchmark import TIMING_STAGES, decode_png, score
from dvidia_training.image_goal_scenes import _png


def row(identifier, cohort, truth, baseline, fused):
    return dict(id=identifier,cohort=cohort,scenario=identifier,layout_id='L0',
        physical_truth={'complete':truth},generation_valid=True,
        baseline={'verdict':baseline},fused={'verdict':fused},
        timing_ms=[{stage:.1 for stage in TIMING_STAGES}])


class SensorGoalBenchmarkTests(unittest.TestCase):
    def test_metrics_keep_false_completions_and_true_completion_losses_visible(self):
        cases=[row('true','native',True,'complete','complete'),
               row('miss','native',True,'unknown','unknown'),
               row('false','primary_forgery',False,'complete','unknown'),
               row('fault','sensor_fault',True,'complete','unknown'),
               row('replay','cross_channel',False,'complete','complete')]
        result=score(cases,cases,repeats=1)
        self.assertEqual(result['overall']['planned'],5)
        self.assertEqual(result['overall']['baseline']['physical_false_completion']['rate'],1.)
        self.assertEqual(result['overall']['fused']['physical_false_completion']['rate'],.5)
        self.assertEqual(result['overall']['paired']['baseline_false_completions'],2)
        self.assertEqual(result['overall']['paired']['prevented_false_completions']['unknown'],1)
        self.assertEqual(result['overall']['paired']['lost_true_completions']['unknown'],1)
        self.assertEqual(result['cohorts']['cross_channel']['fused']['physical_false_completion']['rate'],1.)

    def test_every_case_must_be_present_and_noncomplete_primary_cannot_change(self):
        item=row('case','native',False,'incomplete','incomplete')
        with self.assertRaises(ValueError): score([item],[],repeats=1)
        broken=dict(item,fused={'verdict':'complete'})
        with self.assertRaises(ValueError): score([broken],[broken],repeats=1)

    def test_bounded_png_decoder_checks_crc_dimensions_and_trailing_data(self):
        rgb=np.zeros((480,640,3),dtype=np.uint8);rgb[100:300,280:360]=[30,130,60]
        raw=_png(rgb)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'frame.png';path.write_bytes(raw)
            np.testing.assert_array_equal(decode_png(path),rgb)
            path.write_bytes(raw+b'trailing')
            with self.assertRaises(ValueError): decode_png(path)
            damaged=bytearray(raw);damaged[-15]^=1;path.write_bytes(damaged)
            with self.assertRaises(ValueError): decode_png(path)


if __name__=='__main__': unittest.main()

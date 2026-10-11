"""Offline contract tests for suppression and weight admission; no fake accuracy."""
from pathlib import Path
import tempfile
import unittest

from dvidia_training.observation_models import OpenCVDetector, MODEL_ASSETS, _suppress, _fraction


class DetectorContractTest(unittest.TestCase):
    def test_overlapping_same_class_suppressed_but_distinct_class_retained(self):
        rows = [{'label': 'cup', 'score': .9, 'xyxy': [.1,.1,.5,.5]},
                {'label': 'cup', 'score': .8, 'xyxy': [.11,.11,.51,.51]},
                {'label': 'hand', 'score': .7, 'xyxy': [.1,.1,.5,.5]}]
        self.assertEqual([r['label'] for r in _suppress(rows,.5,30)], ['cup','hand'])

    def test_detections_are_bounded_and_ordered(self):
        rows = [{'label': 'cup', 'score': score, 'xyxy': [0,0,.1,.1]}
                for score in [.1,.8,.4]]
        self.assertEqual(_suppress(rows,.5,1)[0]['score'], .8)

    def test_invalid_thresholds_rejected(self):
        for value in [True, float('nan'), 0, 1, '0.5']:
            with self.assertRaises(ValueError): _fraction(value,'score')

    def test_missing_and_modified_weights_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError): OpenCVDetector(directory)
            asset = MODEL_ASSETS[0]
            with (Path(directory)/asset['name']).open('wb') as output:
                output.truncate(asset['bytes'])
            with self.assertRaisesRegex(ValueError,'SHA-256 mismatch'):
                OpenCVDetector(directory,include_hands=False)


if __name__ == '__main__':
    unittest.main()

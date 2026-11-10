"""Reviewed operator labels and causal feature history, using data fixtures."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from dvidia_training import camera_action_model as model
from dvidia_training import operator_capture as capture
from dvidia_training import operator_export as export
from tests.test_camera_action_capture import STUDENT, fake_estimate
from tests import test_operator_capture as fixtures


class OperatorExportTests(unittest.TestCase):
    setUp = fixtures.OperatorCaptureTests.setUp
    new = fixtures.OperatorCaptureTests.new
    complete = fixtures.OperatorCaptureTests.complete

    def test_reviewed_labels_are_exact_issued_targets_and_keep_source_groups(self):
        recorder, _, _ = self.complete()
        dataset = export.export_episode(recorder.root, reviewed_indices=[0,2,5],
                                        review_reason='Useful reach despite unsuccessful task.')
        self.assertEqual(dataset['features'].shape, (3,269))
        np.testing.assert_array_equal(dataset['targets'], np.tile(
            [*STUDENT['joint_targets_rad'], STUDENT['gripper_position_m']], (3,1)))
        self.assertEqual(dataset['groups'], ['test-start']*3)
        source = dataset['provenance']['episodes'][0]
        self.assertEqual(source['reviewed_indices'], [0,2,5])
        self.assertFalse(source['attempt_strict_success'])
        self.assertEqual(source['source_kind'], 'simulation_scripted_input')
        self.assertEqual(source['label_source'], 'operator_issued')
        self.assertFalse(source['review_authenticated'])
        model._dataset(dataset, blind=False)

    def test_selection_uses_causal_full_prefix_history_before_subsetting(self):
        recorder, _, _ = self.complete()
        full = export.export_episode(recorder.root, reviewed_segments=[[0,10]], review_reason='Reviewed all rows.')
        subset = export.export_episode(recorder.root, reviewed_segments=[[2,4],[7,9]], review_reason='Reviewed reaches.')
        np.testing.assert_array_equal(subset['features'], full['features'][[2,3,7,8]])
        np.testing.assert_array_equal(subset['targets'], full['targets'][[2,3,7,8]])
        blind = export.export_episode(recorder.root, reviewed_indices=[3,7], review_reason='Reviewed.', blind=True)
        np.testing.assert_array_equal(blind['features'][:,model.LIVE_FEATURE_START:],0)

    def test_review_selection_is_explicit_and_no_duplicates_or_bad_indices(self):
        recorder, _, _ = self.complete()
        for selection in ({}, {'reviewed_indices':[]}, {'reviewed_indices':[1,1]},
                          {'reviewed_indices':[True]}, {'reviewed_indices':[10]},
                          {'reviewed_segments':[[0,3],[2,4]]},
                          {'reviewed_segments':[[0,2]],'reviewed_indices':[0]}):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                export.export_episode(recorder.root, review_reason='Reviewed.', **selection)
        with self.assertRaisesRegex(ValueError, 'review reason'):
            export.export_episode(recorder.root, reviewed_indices=[0])

    def test_interrupted_complete_rows_require_review_but_remain_usable(self):
        recorder = self.new(); recorder.open(); recorder.step(STUDENT); recorder.finish('operator_stopped')
        dataset = export.export_episode(recorder.root, reviewed_indices=[0], review_reason='Retain useful first reach.')
        self.assertEqual(dataset['provenance']['episodes'][0]['attempt_status'],'interrupted')
        self.assertEqual(dataset['targets'].shape,(1,7))

    def test_wrong_provenance_cannot_impersonate_operator_supervision(self):
        recorder, _, _ = self.complete()
        recording = capture.load_capture(recorder.root)
        for change in ('teacher', 'student'):
            modified = deepcopy(recording); modified['control_mode'] = change
            with patch.object(capture, 'load_capture', return_value=modified):
                with self.assertRaises(ValueError):
                    export.export_episode(recorder.root, reviewed_indices=[0], review_reason='Reviewed.')

    def test_unsealed_recovery_cannot_be_exported(self):
        recorder = self.new(); recorder.open(); recorder.step(STUDENT)
        recorder._journal.close(); recorder.session.close()
        with self.assertRaises(FileNotFoundError):
            export.export_episode(recorder.root, reviewed_indices=[0], review_reason='Reviewed.')

    def test_native_diagnostics_and_later_actions_do_not_leak_into_features(self):
        recorder, _, _ = self.complete()
        before = export.export_episode(recorder.root, reviewed_indices=[0,1,2], review_reason='Reviewed.')
        other = self.new('other'); other.open()
        for index in range(10):
            action = deepcopy(STUDENT)
            if index >= 3:
                action['joint_targets_rad'][0] += .01
            other.step(action)
        other.finish()
        after = export.export_episode(other.root, reviewed_indices=[0,1,2], review_reason='Reviewed.')
        np.testing.assert_array_equal(before['features'],after['features'])
        np.testing.assert_array_equal(before['targets'],after['targets'])

    def test_dataset_roundtrip_and_exclusive_write(self):
        recorder, _, _ = self.complete()
        dataset = export.export_episode(recorder.root, reviewed_indices=[0,3,5], review_reason='Reviewed reaches.')
        output = Path(self.temp.name)/'export'
        self.assertEqual(export.save_dataset(dataset, output), output/'dataset.json')
        loaded = export.load_dataset(output)
        np.testing.assert_array_equal(loaded['features'], dataset['features'])
        np.testing.assert_array_equal(loaded['targets'], dataset['targets'])
        self.assertEqual(loaded['provenance'], dataset['provenance'])
        with self.assertRaises(FileExistsError):
            export.save_dataset(dataset, output)
        target = output/'targets.npy'
        body = target.read_bytes(); target.write_bytes(body[:-1]+bytes([body[-1]^1]))
        with self.assertRaisesRegex(ValueError, 'byte identity'):
            export.load_dataset(output)

    def test_cli_requires_exact_review_document_and_exports_without_training(self):
        recorder, _, _ = self.complete()
        review = Path(self.temp.name)/'review.json'
        review.write_text(json.dumps(dict(segments=[[0,2]], reason='Reviewed initial reach.')))
        output = Path(self.temp.name)/'cli-export'
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(export.main(['--capture',str(recorder.root), '--review',str(review),
                                          '--output',str(output)]),0)
        receipt = json.loads(stdout.getvalue())
        self.assertEqual(receipt['reviewed_rows'],2)
        self.assertFalse(receipt['training_started'])
        self.assertEqual(export.load_dataset(output)['features'].shape,(2,269))
        review.write_text(json.dumps(dict(indices=[0], reason='Reviewed.', extra=True)))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            export.main(['--capture',str(recorder.root), '--review',str(review), '--output',str(output)])


if __name__ == '__main__':
    unittest.main()

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dvidia_training import camera_action_export as export
from dvidia_training import camera_action_model as m
from tests.test_camera_action_model import context, missing_visual


def fixture(root, *, mode='teacher', group='train-01', anchor_missing=False):
    root.mkdir(exist_ok=True)
    recording = dict(control_mode=mode, goal_position_m=[.35, .08, .08],
        initial_applied_targets=context()['previous_applied'],
        provenance={'source_group': group, 'case_id': group+'-goal0'},
        binding={'calibration_sha256': 'a'*64}, tracker_sha256='b'*64, steps=[])
    for index in range(3):
        c = context(index*.02)
        if index == 0 and anchor_missing:
            c = missing_visual(c)
        elif index == 1:
            c = missing_visual(c)
        before = c['snapshot']
        def save(relative, document):
            path = root/relative
            path.parent.mkdir(exist_ok=True)
            body = m.canonical(document)+b'\n'; path.write_bytes(body)
            return dict(path=relative, bytes=len(body), sha256=sha256(body).hexdigest())
        perception = save(f'perception/{index:04}.json', {'estimate': c['estimate']})
        query_action = dict(joint_targets_rad=[.8, 1.4, -1.3, 0., 0., 0.], gripper_position_m=.01)
        issued_action = dict(joint_targets_rad=[.2, 1.4, -1.3, 0., 0., 0.], gripper_position_m=.03)
        query = save(f'teacher/{index:04}.json', {'action': query_action, 'origin': 'privileged_simulator_teacher'})
        step = dict(index=index, before=before, perception=perception, teacher_query=query,
            command={'issued': issued_action, 'authority': 'teacher' if mode == 'teacher' else 'student'},
            applied={'ramped_targets': c['previous_applied']},
            reference={'action': query_action}, native_truth={'object_position': [9, 9, 9]})
        recording['steps'].append(step)
    (root/'recording.json').write_bytes(m.canonical(recording)+b'\n')
    (root/'integrity.json').write_text('{"fixture":true}\n')
    return recording


class ExportTests(unittest.TestCase):
    def test_export_uses_issued_labels_and_explicit_missing_masks(self):
        with tempfile.TemporaryDirectory() as folder:
            recording = fixture(Path(folder))
            with patch('dvidia_training.camera_action_capture.load_rollout', return_value=recording):
                dataset = export.export_episode(folder)
            self.assertEqual(dataset['features'].shape, (3, m.FEATURE_COUNT))
            self.assertEqual(dataset['targets'][:, 0].tolist(), [.2]*3)
            self.assertEqual(dataset['groups'], ['train-01']*3)
            i = m.FEATURE_NAMES.index('lag0:visual_available')
            self.assertEqual(dataset['features'][:, i].tolist(), [1., 0., 1.])
            self.assertEqual(dataset['provenance']['episodes'][0]['label_source'], 'issued')

    def test_queries_are_labels_not_claimed_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            recording = fixture(Path(folder), mode='mixed')
            with patch('dvidia_training.camera_action_capture.load_rollout', return_value=recording):
                dataset = export.export_episode(folder, label_source='teacher_query')
                with self.assertRaisesRegex(ValueError, 'teacher control'):
                    export.export_episode(folder)
            self.assertEqual(dataset['targets'][:, 0].tolist(), [.8]*3)
            self.assertEqual(recording['steps'][0]['command']['issued']['joint_targets_rad'][0], .2)
            self.assertEqual(recording['steps'][0]['command']['authority'], 'student')
            self.assertIn('separate teacher query', dataset['provenance']['label_definition'])

    def test_full_blind_same_source_identity_and_different_feature_masks(self):
        with tempfile.TemporaryDirectory() as folder:
            recording = fixture(Path(folder))
            with patch('dvidia_training.camera_action_capture.load_rollout', return_value=recording):
                full = export.export_episode(folder)
                blind = export.export_episode(folder, blind=True)
            self.assertEqual(full['provenance']['source_identity_sha256'], blind['provenance']['source_identity_sha256'])
            np.testing.assert_array_equal(full['features'][:, :m.LIVE_FEATURE_START], blind['features'][:, :m.LIVE_FEATURE_START])
            np.testing.assert_array_equal(full['targets'], blind['targets'])
            self.assertTrue(np.all(blind['features'][:, m.LIVE_FEATURE_START:] == 0.))
            masked = export.blind_dataset(full)
            np.testing.assert_array_equal(masked['features'], blind['features'])
            self.assertEqual(masked['provenance'], blind['provenance'])
            self.assertTrue(np.any(full['features'][:, m.LIVE_FEATURE_START:] != 0.))
            with self.assertRaisesRegex(ValueError, 'full-feedback'):
                export.blind_dataset(blind)

    def test_current_labels_truth_reference_and_future_do_not_change_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            recording = fixture(root)
            with patch('dvidia_training.camera_action_capture.load_rollout', return_value=recording):
                before = export.export_episode(root)
            changed = deepcopy(recording)
            for row in changed['steps']:
                row['command']['issued']['joint_targets_rad'][0] = .5
                row['native_truth'] = {'pose': [100]*9}
                row['reference'] = {'action': [99]*7}
                row['after'] = {'timestamp_seconds': 20., 'object': [99]*3}
            (root/'recording.json').write_bytes(m.canonical(changed))
            with patch('dvidia_training.camera_action_capture.load_rollout', return_value=changed):
                after = export.export_episode(root)
            np.testing.assert_array_equal(before['features'], after['features'])
            self.assertFalse(np.array_equal(before['targets'], after['targets']))

    def test_tampered_sidecar_or_changed_recording_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); recording = fixture(root)
            (root/'perception/0001.json').write_text('{}')
            with patch('dvidia_training.camera_action_capture.load_rollout', return_value=recording):
                with self.assertRaisesRegex(ValueError, 'sidecar bytes'):
                    export.export_episode(root)
            (root/'recording.json').write_text('{}')
            with patch('dvidia_training.camera_action_capture.load_rollout', return_value=recording):
                with self.assertRaisesRegex(ValueError, 'changed after'):
                    export.export_episode(root)

    def test_initial_missing_prevents_learning_pair_export(self):
        with tempfile.TemporaryDirectory() as folder:
            recording = fixture(Path(folder), anchor_missing=True)
            with patch('dvidia_training.camera_action_capture.load_rollout', return_value=recording):
                with self.assertRaisesRegex(ValueError, 'initial visual anchor'):
                    export.export_episode(folder)

    def test_combine_rejects_duplicates_and_mixed_feature_contracts(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder)/'a', Path(folder)/'b'
            ra, rb = fixture(a, group='train-a'), fixture(b, group='train-b')
            with patch('dvidia_training.camera_action_capture.load_rollout', return_value=ra):
                left = export.export_episode(a)
                blind = export.export_episode(a, blind=True)
            with patch('dvidia_training.camera_action_capture.load_rollout', return_value=rb):
                right = export.export_episode(b)
            result = export.combine_datasets([left, right])
            self.assertEqual(result['groups'], ['train-a']*3+['train-b']*3)
            self.assertEqual(len(result['provenance']['episodes']), 2)
            with self.assertRaisesRegex(ValueError, 'duplicated'):
                export.combine_datasets([left, left])
            with self.assertRaisesRegex(ValueError, 'feedback masking'):
                export.combine_datasets([left, blind])


if __name__ == '__main__':
    unittest.main()

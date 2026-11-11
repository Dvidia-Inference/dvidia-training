"""Recorder copy and byte-binding regressions; data fixtures, no native physics."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from dvidia_training import action_demonstration as tape
from dvidia_training import camera_action_capture as camera
from dvidia_training import operator_capture as capture
from tests.test_camera_action_capture import FakeSession, fake_estimate, CASE, STUDENT


class OperatorCaptureLatencyTests(unittest.TestCase):
    def setUp(self):
        FakeSession.instances = []; FakeSession.fail_at = None
        self.temp = TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        for target, value in (('_Session', FakeSession), ('_fresh_estimate', fake_estimate),
                ('_native_episode', lambda *args: {'test_only': True}),
                ('_score', lambda *args: dict(valid=False, success=False, reason='fixture_not_physics'))):
            self.enterContext(patch.object(camera, target, value))

    def recorder(self):
        result = capture.OperatorCapture('unused', Path(self.temp.name)/'capture',
            case=CASE, duration_s=.2, source_kind='simulation_scripted_input')
        result.open(); self.addCleanup(result.finish)
        return result

    def test_step_responses_and_inputs_cannot_mutate_siblings_history_or_files(self):
        recorder = self.recorder()
        action = deepcopy(STUDENT)
        event = {'kind': 'test-input', 'path': [1, {'value': 2}]}
        returned = recorder.step(action, input_event=event)
        expected_context = recorder.context()
        expected_step = deepcopy(recorder.steps[0])
        paths = ['decisions/0000.json', 'transitions/0000.json', 'perception/0001.json']
        persisted = {name: (recorder.root/name).read_bytes() for name in paths}

        # Caller-owned inputs stay isolated after the issued command is committed.
        action['joint_targets_rad'][0] = 8.
        event['path'][1]['value'] = 99
        self.assertEqual(recorder.steps[0], expected_step)

        # The top-level return and its context used to be independent copies;
        # mutating either branch must preserve that public behavior.
        returned['snapshot']['signals']['joint_position_rad']['value'][0] = 8.
        returned['snapshot']['camera']['value']['rgb']['path'] = 'frames/changed.png'
        returned['estimate']['object_position_estimate_m'][0] = 8.
        returned['estimate']['signals']['object_position_estimate_m']['value'][0] = 9.
        returned['applied']['ramped_targets']['joint_targets_rad'][0] = 8.
        self.assertEqual(returned['context'], expected_context)
        sibling_snapshot = deepcopy(returned['snapshot'])
        sibling_estimate = deepcopy(returned['estimate'])
        returned['context']['snapshot']['signals']['joint_position_rad']['value'][0] = 7.
        returned['context']['estimate']['object_position_estimate_m'][0] = 7.
        returned['context']['estimate']['signals']['object_position_estimate_m']['value'][0] = 7.
        returned['context']['previous_applied']['joint_targets_rad'][0] = 7.
        returned['context']['goal_position_m'][0] = 7.
        self.assertEqual(returned['snapshot'], sibling_snapshot)
        self.assertEqual(returned['estimate'], sibling_estimate)
        self.assertEqual(recorder.context(), expected_context)

        # A subsequent standalone context is independently owned as well.
        detached = recorder.context()
        detached['snapshot']['camera']['value']['depth']['sha256'] = '0'*64
        detached['estimate']['signals']['object_position_estimate_m']['value'].clear()
        detached['previous_applied']['joint_targets_rad'].clear()
        detached['goal_position_m'].clear()
        self.assertEqual(recorder.context(), expected_context)
        self.assertEqual(recorder.steps[0], expected_step)
        self.assertEqual({name: (recorder.root/name).read_bytes() for name in paths}, persisted)

        recorder.step(deepcopy(STUDENT))
        self.assertEqual(recorder.steps[1]['before'], expected_context['snapshot'])
        report = recorder.finish('operator_stopped')
        self.assertTrue(report['sealed'])
        loaded = capture.load_capture(recorder.root)
        self.assertEqual(loaded['steps'][0], expected_step)
        self.assertEqual(loaded['steps'][1]['before'], expected_context['snapshot'])

    def test_bound_json_decodes_the_same_bytes_it_authenticated(self):
        root = Path(self.temp.name)
        body = b'{"nested":{"values":[1,2,3]}}\n'
        reference = dict(path='transition.json', bytes=len(body), sha256=sha256(body).hexdigest())
        (root/reference['path']).write_bytes(body)
        # A file that changes between reads must not substitute unauthenticated
        # JSON after the first read's hash check. There is no second read now.
        with patch.object(tape, '_ordinary_bytes', side_effect=[body, b'{"forged":true}']) as read:
            self.assertEqual(capture._bound_document(root, reference), {'nested': {'values': [1, 2, 3]}})
            read.assert_called_once_with(root/'transition.json', capture.MAX_DOCUMENT_BYTES)
        for changed in ({**reference, 'bytes': len(body)+1}, {**reference, 'sha256': '0'*64}):
            with self.assertRaisesRegex(ValueError, 'Journal artifact byte identity differs'):
                capture._bound_document(root, changed)
        with self.assertRaisesRegex(ValueError, 'relative path'):
            capture._bound_document(root, {**reference, 'path': '../outside.json'})

    def test_perception_binds_one_read_and_keeps_its_tamper_error(self):
        recorder = self.recorder()
        reference = recorder._perception_ref
        body = (recorder.root/reference['path']).read_bytes()
        with patch.object(tape, '_ordinary_bytes', side_effect=[body, b'{"forged":true}']) as read:
            observed = capture._perception(recorder.root, reference, recorder.snapshot, recorder.header, 0)
            self.assertEqual(observed, tape._decode(body))
            read.assert_called_once_with(recorder.root/reference['path'], capture.MAX_DOCUMENT_BYTES)
        for changed in ({**reference, 'bytes': len(body)+1}, {**reference, 'sha256': '0'*64}):
            with self.assertRaisesRegex(ValueError, 'Perception sidecar byte identity differs'):
                capture._perception(recorder.root, changed, recorder.snapshot, recorder.header, 0)


if __name__ == '__main__':
    unittest.main()

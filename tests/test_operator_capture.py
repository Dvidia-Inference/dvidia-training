"""Operator recording tests use sensor fixtures, never native physics."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from dvidia_training import action_demonstration as tape
from dvidia_training import camera_action_capture as camera
from dvidia_training import operator_capture as capture
from dvidia_training import piper_rgbd_tracker as tracker
from tests.test_camera_action_capture import FakeSession, fake_estimate, CASE, STUDENT


class OperatorCaptureTests(unittest.TestCase):
    def setUp(self):
        FakeSession.instances = []; FakeSession.fail_at = None
        self.temp = TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        for target, value in (('_Session', FakeSession), ('_fresh_estimate', fake_estimate),
                ('_native_episode', lambda *args: {'test_only': True}),
                ('_score', lambda *args: dict(valid=False, success=False, reason='fixture_not_physics'))):
            self.enterContext(patch.object(camera, target, value))

    def new(self, name='capture', **kwargs):
        return capture.OperatorCapture('unused', Path(self.temp.name)/name, case=CASE,
            duration_s=kwargs.pop('duration_s', .2), **kwargs)

    def complete(self, name='capture', **kwargs):
        recorder = self.new(name, **kwargs); opened = recorder.open()
        while not recorder.done:
            recorder.step(deepcopy(STUDENT), input_event={'kind': 'test-key'})
        report = recorder.finish()
        return recorder, opened, report

    def test_external_commands_cadence_and_no_teacher(self):
        recorder, opened, report = self.complete(source_kind='simulation_human_input')
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['camera_pairs'], 6)
        self.assertEqual(report['recorded_complete_intervals'], 10)
        self.assertEqual(report['teacher_queries'], 0)
        self.assertEqual(recorder.session.teacher_calls, 0)
        self.assertTrue(recorder.session.closed)
        self.assertFalse(report['human_authenticated'])
        recording = capture.load_capture(recorder.root)
        for index, step in enumerate(recording['steps']):
            self.assertEqual(step['command']['issued'], STUDENT)
            self.assertEqual(step['command']['authority'], 'simulation_human_input')
            self.assertIsNone(step['teacher_query'])
            self.assertAlmostEqual(step['after']['timestamp_seconds'], (index+1)*.02)
            self.assertEqual(step['before']['camera']['value']['index'], index//2)
            for name in tracker.SIGNALS:
                self.assertEqual(step['before']['signals'][name]['status'], 'missing')
        self.assertFalse((recorder.root/'teacher').exists())
        self.assertFalse(report['strict_success'])
        with self.assertRaises(RuntimeError):
            recorder.step(STUDENT)
        self.assertEqual(recorder.finish(), report)

    def test_context_is_deep_copied_sensor_whitelist_and_events_do_not_advance(self):
        recorder = self.new(); opened = recorder.open()
        self.assertEqual(set(opened['context']), {'snapshot','estimate','previous_applied','goal_position_m'})
        self.assertNotIn('policy_observation', json.dumps(opened['context']))
        self.assertNotIn('native_info', json.dumps(opened['context']))
        recorder.record_event('pause', {'reason':'review'})
        recorder.record_event('resume')
        self.assertEqual(recorder.session.position, 0)
        opened['context']['previous_applied']['joint_targets_rad'][0] = 9.
        self.assertNotEqual(recorder.context()['previous_applied']['joint_targets_rad'][0], 9.)
        returned = recorder.step(STUDENT)
        self.assertEqual(returned['context']['previous_applied'], returned['applied']['ramped_targets'])
        report = recorder.finish('operator_stopped')
        self.assertEqual(report['status'], 'interrupted')
        entries, _ = capture._journal(recorder.root)
        self.assertEqual([e['name'] for e in entries if e['kind']=='event'], ['opened','pause','resume'])
        capture.inspect_capture(recorder.root)

    def test_partial_native_interval_seals_only_committed_prefix_without_retry(self):
        FakeSession.fail_at = 'advance'
        recorder = self.new(); recorder.open()
        recorder.step(STUDENT); recorder.step(STUDENT)
        with self.assertRaisesRegex(RuntimeError, 'partial interval'):
            recorder.step(STUDENT)
        report = recorder.finish()
        self.assertEqual(report['started_intervals'], 3)
        self.assertEqual(report['recorded_complete_intervals'], 2)
        self.assertEqual(report['uncommitted_native_intervals'], 1)
        self.assertEqual(report['status'], 'interrupted')
        self.assertEqual(recorder.session.position, 3)
        self.assertTrue(recorder.session.closed)
        capture.inspect_capture(recorder.root)

    def test_readonly_crash_recovery_does_not_claim_native_completion(self):
        recorder = self.new(); recorder.open(); recorder.step(STUDENT)
        command = dict(issued_at_seconds=.02, requested=deepcopy(STUDENT), issued=deepcopy(STUDENT),
                       authority=recorder.source_kind, operator_correction=False, issue_reason='unchanged')
        decision = dict(index=1, command=command, perception=recorder._perception_ref,
                        input_event=None, snapshot_sha256=tape.digest(recorder.snapshot))
        ref = capture._reference(recorder.root, 'decisions/0001.json', decision)
        recorder._append('intent', index=1, decision=ref)
        recorder._journal.close(); recorder.session.close()
        before = {str(p.relative_to(recorder.root)): p.read_bytes() for p in recorder.root.rglob('*') if p.is_file()}
        recovered = capture.recover_capture(recorder.root)
        self.assertEqual(recovered['report']['recorded_complete_intervals'], 1)
        self.assertEqual(recovered['report']['uncommitted_native_intervals'], 1)
        self.assertFalse(recovered['report']['sealed'])
        self.assertFalse(recovered['report']['strict_success'])
        self.assertEqual(len(recovered['recording']['steps']), 1)
        after = {str(p.relative_to(recorder.root)): p.read_bytes() for p in recorder.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        with self.assertRaises(FileNotFoundError):
            capture.load_capture(recorder.root)

    def test_recovery_discards_incomplete_final_journal_line(self):
        recorder = self.new(); recorder.open(); recorder.step(STUDENT)
        recorder._journal.write(b'{"unfinished":'); recorder._journal.flush()
        recorder._journal.close(); recorder.session.close()
        result = capture.recover_capture(recorder.root)
        self.assertEqual(result['report']['recorded_complete_intervals'], 1)
        self.assertGreater(result['report']['truncated_journal_bytes'], 0)

    def test_transition_without_journal_commit_is_not_recovered(self):
        recorder = self.new(); recorder.open()
        append = recorder._append
        def abrupt_exit(kind, **payload):
            if kind == 'outcome':
                raise SystemExit('simulated process loss before commit')
            return append(kind, **payload)
        with patch.object(recorder, '_append', abrupt_exit), self.assertRaises(SystemExit):
            recorder.step(STUDENT)
        recorder._journal.close(); recorder.session.close()
        self.assertTrue((recorder.root/'transitions/0000.json').exists())
        recovered = capture.recover_capture(recorder.root)
        self.assertEqual(recovered['report']['recorded_complete_intervals'], 0)
        self.assertEqual(recovered['report']['uncommitted_native_intervals'], 1)

    def test_interrupt_after_durable_outcome_rebuilds_native_trace_before_sealing(self):
        recorder = self.new(); recorder.open()
        append = recorder._append; trace = []
        def interrupt_after_outcome(kind, **payload):
            result = append(kind, **payload)
            if kind == 'outcome':
                raise KeyboardInterrupt('after durable outcome, before memory update')
            return result
        def native_episode(session, states, actions):
            trace.append((deepcopy(states), deepcopy(actions)))
            return {'test_only':True}
        with patch.object(recorder, '_append', interrupt_after_outcome), \
                patch.object(camera, '_native_episode', side_effect=native_episode), \
                self.assertRaises(KeyboardInterrupt):
            recorder.step(STUDENT)
        report = recorder.finish()
        self.assertEqual(report['status'], 'interrupted')
        self.assertEqual(report['recorded_complete_intervals'], 1)
        self.assertEqual(report['started_intervals'], 1)
        self.assertEqual(report['uncommitted_native_intervals'], 0)
        self.assertEqual(len(trace[0][0]), 2)
        self.assertEqual(len(trace[0][1]), 1)
        self.assertEqual(trace[0][1][0]['joint_targets_rad'], STUDENT['joint_targets_rad'])
        capture.inspect_capture(recorder.root)

    def test_interrupt_after_durable_intent_rebuilds_started_counter(self):
        recorder = self.new(); recorder.open()
        append = recorder._append
        def interrupt_after_intent(kind, **payload):
            result = append(kind, **payload)
            if kind == 'intent':
                raise KeyboardInterrupt('after durable intent, before started counter')
            return result
        with patch.object(recorder, '_append', interrupt_after_intent), self.assertRaises(KeyboardInterrupt):
            recorder.step(STUDENT)
        report = recorder.finish()
        self.assertEqual(report['started_intervals'],1)
        self.assertEqual(report['recorded_complete_intervals'],0)
        self.assertEqual(report['uncommitted_native_intervals'],1)
        self.assertEqual(recorder.session.position,0)
        capture.inspect_capture(recorder.root)

    def test_interrupt_after_journal_fsync_reconciles_cursor_before_finished_append(self):
        for kind in ('intent','outcome'):
            with self.subTest(kind=kind):
                recorder = self.new(kind); recorder.open()
                journal_fd = recorder._journal.fileno()
                original_fsync = capture.os.fsync
                append = recorder._append; armed = [False]; interrupted = [False]
                def arm_before_append(actual_kind, **payload):
                    armed[0] = actual_kind == kind and not interrupted[0]
                    return append(actual_kind, **payload)
                def interrupt_after_fsync(fd):
                    original_fsync(fd)
                    if fd == journal_fd and armed[0]:
                        armed[0] = False; interrupted[0] = True
                        raise KeyboardInterrupt('journal fsync returned before cursor assignment')
                with patch.object(recorder, '_append', arm_before_append), \
                        patch.object(capture.os, 'fsync', interrupt_after_fsync), \
                        self.assertRaises(KeyboardInterrupt):
                    recorder.step(STUDENT)
                report = recorder.finish()
                self.assertTrue(report['sealed'])
                self.assertEqual(report['started_intervals'],1)
                self.assertEqual(report['recorded_complete_intervals'],int(kind == 'outcome'))
                entries, tail = capture._journal(recorder.root)
                self.assertEqual(tail,0)
                self.assertEqual([entry['sequence'] for entry in entries],list(range(len(entries))))
                self.assertEqual(entries[-1]['kind'],'finished')
                capture.inspect_capture(recorder.root)

    def test_truncated_journal_during_failure_remains_unsealed_for_readonly_recovery(self):
        recorder = self.new(); recorder.open()
        append = recorder._append
        def interrupt_mid_outcome(kind, **payload):
            if kind == 'outcome':
                recorder._journal.write(b'{"partial_outcome":')
                recorder._journal.flush()
                raise KeyboardInterrupt('mid journal record')
            return append(kind, **payload)
        with patch.object(recorder, '_append', interrupt_mid_outcome), self.assertRaises(KeyboardInterrupt):
            recorder.step(STUDENT)
        report = recorder.finish()
        self.assertFalse(report['sealed'])
        self.assertEqual(report['finish_error']['stage'],'finish_reconciliation')
        self.assertTrue(recorder.session.closed)
        self.assertTrue((recorder.root/'finish_error.json').exists())
        self.assertFalse((recorder.root/'integrity.json').exists())
        self.assertFalse((recorder.root/'recording.json').exists())
        recovered = capture.recover_capture(recorder.root)
        self.assertEqual(recovered['report']['recorded_complete_intervals'],0)
        self.assertEqual(recovered['report']['uncommitted_native_intervals'],1)
        self.assertFalse(recovered['report']['strict_success'])

    def test_sidecar_tampering_and_missing_masks_rejected(self):
        recorder, _, _ = self.complete()
        path = recorder.root/'perception/0000.json'
        document = json.loads(path.read_text()); document['estimate']['status'] = 'missing'
        body = tape.canonical(document)+b'\n'; path.write_bytes(body)
        with self.assertRaisesRegex(ValueError, 'byte identity'):
            capture.inspect_capture(recorder.root)
        with self.assertRaisesRegex(ValueError, 'missing masks'):
            capture._perception(recorder.root,
                dict(path='perception/0000.json', bytes=len(body), sha256=sha256(body).hexdigest()),
                recorder.initial, recorder.header, 0)

    def test_missing_live_camera_estimate_is_retained_without_native_truth(self):
        def missing_after_initial(session, snapshot, calibration, negative=None):
            return fake_estimate(session, snapshot, calibration,
                                 'initial_missing' if snapshot['timestamp_seconds'] >= .04 else None)
        with patch.object(camera, '_fresh_estimate', missing_after_initial):
            recorder, _, _ = self.complete()
        recording = capture.load_capture(recorder.root)
        item = capture._read(recorder.root, recording['steps'][2]['perception']['path'])
        self.assertEqual(item['estimate']['status'], 'missing')
        self.assertIsNone(item['estimate']['object_position_estimate_m'])

    def test_validation_and_exclusive_output(self):
        for changes in ({'duration_s':21.}, {'duration_s':.11}, {'source_kind':'teacher'},
                        {'source_kind':'simulation_teleoperation'}):
            with self.assertRaises(ValueError):
                self.new(**changes)
        recorder = self.new(); recorder.open()
        with self.assertRaises(ValueError):
            recorder.step({**STUDENT, 'gripper_position_m':.04})
        with self.assertRaises(ValueError):
            recorder.record_event('oversized', {'data':'x'*4097})
        self.assertEqual(recorder.session.position, 0)
        recorder.finish()
        with self.assertRaises(FileExistsError):
            self.new().open()

    def test_setup_failure_is_sealed_and_closed(self):
        FakeSession.fail_at = 'open'; recorder = self.new()
        with self.assertRaises(RuntimeError):
            recorder.open()
        self.assertEqual(recorder.finish()['status'], 'refused')
        self.assertTrue(recorder.session.closed)
        capture.inspect_capture(recorder.root)

    def test_estimator_initialization_failure_closes_and_seals_refusal(self):
        recorder = self.new()
        with patch.object(camera, '_fresh_estimate', side_effect=RuntimeError('estimator initialization')):
            with self.assertRaisesRegex(RuntimeError, 'estimator initialization'):
                recorder.open()
        self.assertTrue(recorder.session.closed)
        self.assertEqual(recorder.finish()['status'], 'refused')
        self.assertFalse(recorder.finish()['native_evidence_saved'])
        self.assertFalse((recorder.root/'native_episode.json').exists())
        capture.inspect_capture(recorder.root)

    def test_finish_before_open_cleans_up_without_creating_an_attempt(self):
        recorder = self.new(); report = recorder.finish('initialization_failed')
        self.assertTrue(recorder.session.closed)
        self.assertFalse(recorder.root.exists())
        self.assertEqual(report['status'], 'refused')
        self.assertFalse(report['sealed'])


if __name__ == '__main__':
    unittest.main()

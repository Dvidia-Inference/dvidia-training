from copy import deepcopy
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_camera_action_model import context


SCRIPT = Path(__file__).parents[1]/'benchmarks/camera-action-v01/independent_audit.py'
SPEC = importlib.util.spec_from_file_location('camera_action_independent_audit', SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def put(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audit.canonical(document)+b'\n')


def seal(root):
    files = {str(p.relative_to(root)): audit.identity(p) for p in root.rglob('*')
             if p.is_file() and p.name != 'integrity.json'}
    put(root/'integrity.json', {'schema': 'dvidia.camera-action-rollout/v2/integrity', 'files': files})


def episode(root, case):
    root.mkdir(parents=True)
    before = context()['snapshot']; after = context(.02)['snapshot']
    for name in ('rgb', 'depth'):
        path = root/('frames/'+name); path.parent.mkdir(exist_ok=True)
        path.write_bytes(b'fixture bytes '+name.encode())
        reference = {'path': str(path.relative_to(root)), **audit.identity(path)}
        before['camera']['value'][name] = reference
        after['camera']['value'][name] = deepcopy(reference)
    calibration = {'fixture': True}
    estimate = context()['estimate']
    action = context()['previous_applied']
    perception = dict(index=0, snapshot_sha256=audit.digest(before), frame=before['camera']['value'],
        calibration_sha256=audit.digest(calibration), tracker_sha256='a'*64, estimate=estimate)
    query = dict(index=0, snapshot_sha256=audit.digest(before), action=action, origin='privileged_simulator_teacher')
    put(root/'perception/0000.json', perception); put(root/'teacher/0000.json', query)
    step = dict(index=0, before=before, after=after,
        command=dict(authority='teacher', operator_correction=False, requested=action, issued=action),
        perception=dict(path='perception/0000.json', **audit.identity(root/'perception/0000.json')),
        teacher_query=dict(path='teacher/0000.json', **audit.identity(root/'teacher/0000.json')),
        applied={'ramped_targets': action})
    recording = dict(schema='dvidia.camera-action-rollout/v2', control_mode='teacher', initial=before,
        provenance={'source_group': case['group_id'], 'case_id': case['id']},
        goal_position_m=case['goal_position_m'], calibration=calibration,
        binding={'calibration_sha256': audit.digest(calibration)}, tracker_sha256='a'*64, steps=[step])
    transform = np.eye(4).reshape(-1).tolist()
    put(root/'context.json', {'metadata': {'world_from_robot_base': transform,
        'world_from_robot_base_sha256': audit.digest(transform)}})
    put(root/'recording.json', recording)
    put(root/'native_episode.json', {'trajectory': [{'object_position_m': [.3, 0., .08]}]})
    put(root/'native_metadata.json', {'rows': 20})
    arrays = dict(native_tick=np.arange(1, 21), t_seconds=np.arange(1, 21)*.001,
        object_position_m=np.tile([.3, 0., .08], (20, 1)),
        arm_actuator_controls_rad=np.tile(action['joint_targets_rad'], (20, 1)),
        applied_jaw_target_m=np.full(20, .02))
    with (root/'native.npz').open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    summary = dict(case=case, mode='teacher', negative_control=None, status='interrupted',
        recorded_complete_intervals=1, strict_success=False, score={'success': False, 'reason': 'fixture'},
        teacher_queries=1, student_calls=0, camera_pairs=1, initial_anchor_error_m=0.,
        perception={'decisions': 1, 'available': 1, 'missing': 0, 'longest_missing_seconds': 0.})
    put(root/'rollout.json', summary)
    seal(root)
    return summary


class AuditTests(unittest.TestCase):
    def case(self, index=0):
        return {'id': f'dev-{index}', 'group_id': f'dev-{index}', 'goal_position_m': [.35, .08, .08]}

    def test_partial_failure_is_audited_without_becoming_success(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'episode'; case = self.case(); summary = episode(root, case)
            with patch.object(audit.scoring, 'score_feedback_episode', return_value=summary['score']):
                result = audit.audit_episode(root, case)
            self.assertFalse(result['strict_success'])
            self.assertEqual(result['complete_intervals'], 1)
            self.assertEqual(result['native_ticks'], 20)
            self.assertEqual(result['initial_anchor_error_m'], 0.)
            self.assertEqual(result['available_position_error_p95_m'], 0.)

    def test_anchor_error_recomputed_and_cannot_be_claimed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'episode'; case = self.case(); summary = episode(root, case)
            summary['initial_anchor_error_m'] = .1
            put(root/'rollout.json', summary); seal(root)
            with patch.object(audit.scoring, 'score_feedback_episode', return_value=summary['score']):
                with self.assertRaisesRegex(ValueError, 'anchor error'):
                    audit.audit_episode(root, case)

    def test_changed_bytes_and_nonrigid_transform_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'episode'; case = self.case(); summary = episode(root, case)
            (root/'frames/rgb').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                audit.audit_episode(root, case)
            seal(root)
            context_doc = audit.read(root/'context.json')
            context_doc['metadata']['world_from_robot_base'][0] = 2.
            context_doc['metadata']['world_from_robot_base_sha256'] = audit.digest(context_doc['metadata']['world_from_robot_base'])
            put(root/'context.json', context_doc); seal(root)
            with self.assertRaisesRegex(ValueError, 'fixed-base transform'):
                audit.audit_episode(root, case)

    def test_failed_setup_can_pass_consistency_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); case = self.case()
            put(root/'rollout.json', {'case': case, 'mode': 'teacher', 'negative_control': None,
                'status': 'refused', 'recorded_complete_intervals': 0, 'strict_success': False})
            seal(root)
            result = audit.audit_episode(root, case)
            self.assertEqual(result['native_ticks'], 0)
            self.assertFalse(result['strict_success'])

    def test_four_attempt_denominator_and_source_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sources = {'piper_feedback_score.py': audit.identity(Path(audit.scoring.__file__))['sha256']}
            roster, rows = [], []
            for index in range(4):
                case = self.case(index); attempt = {'id': f'diagnostic-{index}', 'case': case}
                directory = root/'episodes/diagnostic'/attempt['id']
                episode(directory, {**case, 'policy_artifact_sha256': audit.digest(sources)})
                roster.append(attempt); rows.append({**attempt, 'strict_success': False})
            protocol = {'source_sha256': sources, 'rosters': {'diagnostic': roster}}
            put(root/'protocol.json', protocol)
            protocol_sha = audit.identity(root/'protocol.json')['sha256']
            put(root/'registration.json', {'source_sha256': sources, 'protocol_sha256': protocol_sha})
            put(root/'stages/diagnostic.json', {'rows': rows, 'model_bindings': {'teacher': audit.digest(sources)}})
            put(root/'stages/diagnostic.integrity.json', {
                'sha256': audit.identity(root/'stages/diagnostic.json')['sha256'], 'protocol_sha256': protocol_sha})
            with patch.object(audit.scoring, 'score_feedback_episode', return_value={'success': False, 'reason': 'fixture'}):
                result = audit.audit(root)
            self.assertTrue(result['passed'], result['errors'])
            self.assertEqual(result['attempts_verified'], 4)
            self.assertEqual(result['strict_successes'], 0)
            self.assertFalse(result['physics_executed'])
            rows.pop(); put(root/'stages/diagnostic.json', {'rows': rows, 'model_bindings': {'teacher': audit.digest(sources)}})
            put(root/'stages/diagnostic.integrity.json', {
                'sha256': audit.identity(root/'stages/diagnostic.json')['sha256'], 'protocol_sha256': protocol_sha})
            self.assertFalse(audit.audit(root)['passed'])


if __name__ == '__main__':
    unittest.main()

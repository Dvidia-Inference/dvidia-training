"""Replay roster selection and original feedback evidence binding checks."""
from copy import deepcopy
import gzip
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dvidia_training import piper_feedback_viewer as viewer


def raw(value):
    return (json.dumps(value, sort_keys=True)+'\n').encode()


def fixtures(root):
    cases = [dict(id=f'confirmation-{start:02}-goal{goal}-{condition}', condition=condition,
                  start_xy=[.286+start*.004, -.031], goal_xy=[.35, .08] if goal == 0 else [.40, -.08],
                  seed=271000+start*2+goal, pulses=[])
             for start in range(6) for goal in (0, 1) for condition in viewer.CONDITIONS]
    protocol = dict(schema=viewer.STUDY_SCHEMA, physical_robot_ready=False,
                    variants=list(viewer.VARIANTS), cases={'confirmation': cases})
    frozen = dict(schema=viewer.STUDY_SCHEMA, protocol_sha256=sha256(raw(protocol)).hexdigest(),
                  selected='rff256', blind='blind256', selected_sha256='1'*64, blind_sha256='2'*64,
                  base_model_sha256='3'*64, runtime_sha256={'native.py': '4'*64})
    rows = [dict(case_id=case['id'], condition=case['condition'], variant=variant,
                 score=dict(success=(i % 3 == 0), valid=True))
            for i, case in enumerate(cases) for variant in viewer.VARIANTS]
    report = dict(schema=viewer.STUDY_SCHEMA, physical_robot_ready=False,
                  protocol_sha256=frozen['protocol_sha256'], frozen_sha256=sha256(raw(frozen)).hexdigest(), rows=rows,
                  variants={variant: dict(scheduled=24, successes=8, valid=24) for variant in viewer.VARIANTS})
    for name, value in [('protocol.json', protocol), ('frozen.json', frozen), ('report.json', report)]:
        (root/name).write_bytes(raw(value))
    return report, protocol, frozen


def write_attempt(root, row, case, frozen):
    prefix = root/'confirmation'/case['id']/row['variant']
    prefix.mkdir(parents=True)
    model = frozen['blind_sha256'] if row['variant'] == 'blind' else frozen['selected_sha256'] if row['variant'] in ('feedback', 'open_jaw') else None
    binding = dict(case=case, variant=row['variant'], collect_labels=False, expert_probability=1.,
                   protocol_sha256=frozen['protocol_sha256'], runtime_sha256=frozen['runtime_sha256'],
                   model_raw_sha256=model, base_model_raw_sha256=frozen['base_model_sha256'],
                   model_sha256=model, base_model_sha256=frozen['base_model_sha256'])
    episode = dict(case_id=case['id'], condition=case['condition'], policy_kind=row['variant'],
                   seed=case['seed'], pulses=case['pulses'], implementation_sha256=frozen['runtime_sha256'],
                   model_sha256=model, base_model_sha256=frozen['base_model_sha256'], config={},
                   compiled_model_sha256='c'*64,
                   trajectory=[dict(grasp_contact_counts={'left': 0, 'right': 0},
                                    pad_normal_forces_n={'left': 0., 'right': 0.},
                                    table_object_contact_count=4, native_actuator_controls=[0.]*7)])
    content = {'started.json': raw(binding), 'episode.json.gz': gzip.compress(raw(episode), mtime=0),
               'native.npz': b'numeric fixture placeholder; not parsed by the viewer',
               'native.json': raw({'pulses': case['pulses']}), 'score.json': raw(row['score'])}
    for name, data in content.items():
        (prefix/name).write_bytes(data)
    receipt = dict(binding=binding, case_id=case['id'], variant=row['variant'],
                   files={name: sha256(data).hexdigest() for name, data in content.items()})
    (prefix/'receipt.json').write_bytes(raw(receipt))
    return prefix, receipt


class FeedbackViewerTests(unittest.TestCase):
    def test_complete_frozen_roster_and_aggregates_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, protocol, frozen = fixtures(root)
            files, actual, _, _ = viewer._study_documents(root)
            self.assertEqual(actual, report)
            self.assertEqual(len(files), 3)
            for mutate in (lambda r: r['rows'].pop(), lambda r: r['rows'].reverse(),
                           lambda r: r['rows'][1].update(variant='shuffled_labels'),
                           lambda r: r['variants']['feedback'].update(successes=24),
                           lambda r: r.update(frozen_sha256='0'*64),
                           lambda r: r.update(physical_robot_ready=True)):
                changed = deepcopy(report)
                mutate(changed)
                (root/'report.json').write_bytes(raw(changed))
                with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                    viewer._study_documents(root)

    def test_selection_retains_all_feedback_and_first_controls_independent_of_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            report, _, _ = fixtures(Path(directory))
            selected = viewer.selected_replays(report)
            identities = [(row['case_id'], row['variant']) for row in selected]
            self.assertEqual(len(selected), 28)
            self.assertEqual(sum(row['variant'] == 'feedback' for row in selected), 24)
            self.assertEqual({row['case_id'] for row in selected if row['variant'] != 'feedback'}, {report['rows'][0]['case_id']})
            changed = deepcopy(report)
            for row in changed['rows']:
                row['score']['success'] = not row['score']['success']
            self.assertEqual(identities, [(row['case_id'], row['variant']) for row in viewer.selected_replays(changed)])

    def test_original_attempt_bytes_and_scores_are_bound_and_native_arrays_remain_external(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, protocol, frozen = fixtures(root)
            row, case = report['rows'][0], protocol['cases']['confirmation'][0]
            prefix, receipt = write_attempt(root, row, case, frozen)
            episode, copies, links = viewer._attempt(root, row, case, frozen, frozen['protocol_sha256'], 'a'*40)
            self.assertEqual(episode['case_id'], case['id'])
            self.assertEqual(len(copies), 5)  # Five small local originals; native.npz stays external.
            self.assertFalse(any(name.endswith('.npz') for name in copies))
            self.assertEqual(links['native_npz_sha256'], receipt['files']['native.npz'])
            self.assertTrue(links['native.npz'].startswith(viewer.NATIVE_DATASET_ROOT+'a'*40+'/confirmation/'))
            (prefix/'native.npz').write_bytes(b'altered native evidence')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                viewer._attempt(root, row, case, frozen, frozen['protocol_sha256'])

    def test_rehashed_score_or_model_swap_cannot_change_the_report_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, protocol, frozen = fixtures(root)
            row, case = report['rows'][0], protocol['cases']['confirmation'][0]
            prefix, receipt = write_attempt(root, row, case, frozen)
            changed = raw({'success': not row['score']['success'], 'valid': True})
            (prefix/'score.json').write_bytes(changed)
            receipt['files']['score.json'] = sha256(changed).hexdigest()
            (prefix/'receipt.json').write_bytes(raw(receipt))
            with self.assertRaisesRegex(ValueError, 'score differs'):
                viewer._attempt(root, row, case, frozen, frozen['protocol_sha256'])
            receipt['binding']['model_raw_sha256'] = frozen['blind_sha256']
            (prefix/'receipt.json').write_bytes(raw(receipt))
            with self.assertRaisesRegex(ValueError, 'model binding'):
                viewer._attempt(root, row, case, frozen, frozen['protocol_sha256'])

    def test_malformed_and_overexpanded_episode_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'gzip'):
            viewer._unpack_episode(b'not gzip')
        old = viewer.MAX_EPISODE
        try:
            viewer.MAX_EPISODE = 8
            with self.assertRaisesRegex(ValueError, 'decompressed bounds'):
                viewer._unpack_episode(gzip.compress(b'123456789'))
        finally:
            viewer.MAX_EPISODE = old

    def test_export_chunks_preserve_every_case_and_small_original_without_copying_native_arrays(self):
        """Synthetic file-layout integration fixture; native geometry/poses mocked."""
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory)/'run', Path(directory)/'viewer'
            root.mkdir()
            report, protocol, frozen = fixtures(root)
            model_files = {'base-model.json': b'{"fixture":"base"}',
                           'candidates/rff256/model.json': b'{"fixture":"feedback"}',
                           'candidates/blind256/model.json': b'{"fixture":"blind"}'}
            frozen['base_model_sha256'] = sha256(model_files['base-model.json']).hexdigest()
            frozen['selected_sha256'] = sha256(model_files['candidates/rff256/model.json']).hexdigest()
            frozen['blind_sha256'] = sha256(model_files['candidates/blind256/model.json']).hexdigest()
            for candidate, prefix in [('rff256', 'selected'), ('blind256', 'blind')]:
                for filename, suffix in [('receipt.json', 'receipt'), ('development.json', 'development')]:
                    name = f'candidates/{candidate}/{filename}'
                    model_files[name] = raw({'fixture': name})
                    frozen[f'{prefix}_{suffix}_sha256'] = sha256(model_files[name]).hexdigest()
            for name, data in model_files.items():
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            parent_raw = raw({'schema': 'dvidia.piper-feedback-training-parent/v1',
                              'path': 'prior-studies/study01', 'fixture': True})
            (root/'training-parent.json').write_bytes(parent_raw)
            frozen['training_parent_sha256'] = sha256(parent_raw).hexdigest()
            (root/'frozen.json').write_bytes(raw(frozen))
            report['frozen_sha256'] = sha256(raw(frozen)).hexdigest()
            (root/'report.json').write_bytes(raw(report))
            case_by_id = {case['id']: case for case in protocol['cases']['confirmation']}
            for row in report['rows']:
                write_attempt(root, row, case_by_id[row['case_id']], frozen)
            rows = [dict(t_seconds=0., joint_position_rad=[0.]*6, gripper_joint_position_m=[.035, -.035],
                         issued_gripper_position_m=None, object_position_m=[.3, 0., .08], tcp_position_m=[.3, 0., .2])]
            poses = np.asarray([[0., 0., 0., 1., 0., 0., 0.]], dtype='<f4').tobytes()
            with (patch('dvidia_training.robot_profiles.verify_assets', return_value={'fixture': True}),
                  patch('dvidia_training.robot_profiles._payload', return_value=(None, {})),
                  patch('dvidia_training.piper_task.build_task_model', return_value=object()),
                  patch.object(viewer.sim, 'model_digest', return_value='c'*64),
                  patch.object(viewer, '_scene', return_value=({'bodies': [{'id': 0, 'name': 'fixture'}]}, b'geometry')),
                  patch.object(viewer, '_poses', side_effect=lambda model, episode, residuals: (poses, deepcopy(rows)))):
                result = viewer.export_viewer(root, 'unused', output)
            self.assertEqual(result['replay_cases'], 28)
            self.assertEqual(result['report_attempts'], 120)
            metadata = json.loads((output/'motion.json').read_bytes())
            self.assertEqual(metadata['training_parent_manifest'], 'training-parent.json')
            self.assertEqual((output/'training-parent.json').read_bytes(), parent_raw)
            self.assertNotIn('motion_binary', metadata)
            self.assertEqual(len(metadata['cases']), 28)
            self.assertEqual(len(list(output.glob('frames-*.json'))), 28)
            self.assertEqual(len(list(output.glob('poses-*.bin'))), 28)
            self.assertFalse(list(output.rglob('*.npz')))
            self.assertEqual(len(list((output/'confirmation').rglob('episode.json.gz'))), 120)
            for case in metadata['cases']:
                self.assertNotIn('rows', case)
                for key in ('rows_file', 'pose_binary'):
                    part = case[key]
                    data = (output/part['href']).read_bytes()
                    self.assertEqual(len(data), part['bytes'])
                    self.assertEqual(sha256(data).hexdigest(), part['sha256'])
            integrity = json.loads((output/'viewer-integrity.json').read_bytes())
            actual = {p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}
            self.assertEqual(set(integrity['files']), actual-{'viewer-integrity.json'})
            self.assertTrue(all(p.stat().st_size <= viewer.MAX_PUBLISHED_FILE for p in output.rglob('*') if p.is_file()))


if __name__ == '__main__':
    unittest.main()

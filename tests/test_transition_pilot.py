"""Grouped split, frozen evaluation and failure denominator checks."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from dvidia_training.episode_tape import CASES, canonical, digest, validate_tape, write_tape
from dvidia_training.transition_pilot import (ALPHAS, CORPUS_KIND, _manifest, _metrics,
    generate_corpus, inspect_pilot, run_pilot)
from tests.test_episode_tape import fixture


def authored_tape(group_index, case):
    tape = fixture()
    tape['configuration'].update(horizon=.06, object_mass=.04+.001*group_index, object_friction=.8+.01*group_index)
    tape['provenance'].update(episode_id=f'reset-{group_index:03d}-{case}',
        source_recording_id=f'recording-{group_index}-{case}', session_id=f'session-{group_index}',
        seed=group_index+10, case=case, scene_sha256=digest({'group': group_index}),
        configuration_sha256=digest(tape['configuration']))
    template = deepcopy(tape['steps'][0])
    before = tape['initial']
    steps = []
    for index in range(3):
        step = deepcopy(template)
        start, end = index*.02, (index+1)*.02
        step.update(index=index, before=deepcopy(before))
        # First nominal/release transition is identical, then only jaw command changes.
        targets = [.02*(index+1)+.001*group_index]*6
        requested = {'joint_targets': targets, 'gripper_width': .04}
        intervention = 'open-jaw' if case == 'open-jaw' else 'release-after-lift' if case == 'release-after-lift' and index >= 1 else 'none'
        issued = deepcopy(requested)
        if intervention != 'none':
            issued['gripper_width'] = .08
        step['command'].update(issued_at_seconds=start, policy_requested=requested, issued=issued, intervention=intervention)
        after = deepcopy(before)
        after['timestamp_seconds'] = end
        for sensor in after['signals'].values():
            sensor.update(captured_at_seconds=end)
        delta = [.005*(target-q) for target, q in zip(targets, before['signals']['joint_position']['value'])]
        after['signals']['joint_position']['value'] = [q+d for q,d in zip(before['signals']['joint_position']['value'],delta)]
        after['signals']['joint_velocity']['value'] = [d/.02 for d in delta]
        old_gap = before['signals']['gripper_width']['value']
        after['signals']['gripper_width']['value'] = old_gap+.1*(issued['gripper_width']-old_gap)
        step['after'] = after
        step['applied'].update(interval_start_seconds=start, interval_end_seconds=end, ramped_joint_targets=targets,
                              ramped_gripper_width=issued['gripper_width'],
                              native_position_controls=targets+[issued['gripper_width']/2]*2)
        result = {'terminated':False,'truncated':index==2,'success':False,'valid':True,'reason':'horizon' if index==2 else 'running'}
        step['result'] = result
        steps.append(step)
        before = after
    tape['steps'] = steps
    tape['outcome'].update(result=deepcopy(steps[-1]['result']), simulation_seconds=.06, control_steps=3)
    return validate_tape(tape)


def corpus(directory, failed=None):
    directory.mkdir()
    (directory/'tapes').mkdir()
    groups = []
    for group_index, split in enumerate(('train','development','test')):
        example = authored_tape(group_index,'nominal')
        group = {'group_id':f'reset-{group_index:03d}','split':split,'seed':group_index+10,
                 'session_id':f'session-{group_index}','reset_lineage_id':f'lineage-{group_index}',
                 'configuration':deepcopy(example['configuration']),
                 'configuration_sha256':example['provenance']['configuration_sha256'],'attempts':[]}
        for case in CASES:
            tape = authored_tape(group_index,case)
            attempt = {'episode_id':tape['provenance']['episode_id'], 'source_recording_id':tape['provenance']['source_recording_id'],
                       'case':case,'status':'completed','path':f'tapes/{tape["provenance"]["episode_id"]}.json',
                       'sha256':None,'bytes':None,'error':None,'recording_elapsed_seconds':.001,
                       'provenance':deepcopy(tape['provenance']), 'outcome':deepcopy(tape['outcome'])}
            if failed == (split,case):
                attempt.update(status='failed',path=None,error='RuntimeError: deliberately failed attempt',provenance=None,outcome=None)
            else:
                receipt = write_tape(directory/attempt['path'],tape)
                attempt.update(sha256=receipt['sha256'],bytes=receipt['bytes'])
            group['attempts'].append(attempt)
        groups.append(group)
    manifest = {'kind':CORPUS_KIND,'schema_version':1,'scope':'simulation-only','physical_robot_ready':False,
                'shoe_skill_ready':False,'status':'completed-with-failed-attempts' if failed else 'completed',
                'protocol':{'revision':'grouped-native-actuator-20ms-v1','cases':list(CASES),
                            'split_unit':'reset group','variation':'predeclared','split_counts':{'train':1,'development':1,'test':1},
                            'source_files_sha256':groups[0]['attempts'][0]['provenance']['source_files_sha256']},
                'groups':groups,'collection_elapsed_seconds':.001}
    path = directory/'manifest.json'
    path.write_bytes(canonical(manifest)+b'\n')
    return path


def rewrite(path, manifest):
    path.write_bytes(canonical(manifest)+b'\n')


class TransitionPilotTests(unittest.TestCase):
    def test_frozen_evaluation_artifacts_metrics_slices_and_matched_branch(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = corpus(root/'corpus')
            report = run_pilot(manifest,root/'pilot')
            self.assertEqual(report['status'],'completed',report.get('error'))
            self.assertEqual(report['counts']['attempts'],9)
            self.assertEqual(report['counts']['rows'],{'train':9,'development':9,'test':9})
            self.assertEqual(report['counts']['simulation_unsuccessful'],9)
            self.assertFalse(report['physical_robot_ready'])
            self.assertFalse(report['shoe_skill_ready'])
            self.assertEqual(report['matched_branch']['status'],'matched')
            self.assertEqual(report['matched_branch']['prediction_delta']['state']['nominal'],report['matched_branch']['prediction_delta']['state']['release'])
            self.assertEqual(set(report['metrics']['test']),{'action','state','persistence','constant_velocity','training_mean'})
            self.assertEqual([trial['alpha'] for trial in report['selection']['action']['trials']],list(ALPHAS))
            self.assertEqual(report['timings']['inference']['action']['measured_calls'],1000)
            self.assertEqual(inspect_pilot(root/'pilot')['integrity'],'verified')
            with self.assertRaisesRegex(ValueError,'fresh'):
                run_pilot(manifest,root/'pilot')

    def test_held_out_tapes_are_not_parsed_before_both_models_and_freeze_exist(self):
        import dvidia_training.transition_pilot as pilot
        original = pilot._load_attempt
        observed = []
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = corpus(root/'corpus')
            def observed_load(corpus_root,group,attempt):
                if group['split'] == 'test':
                    self.assertTrue((root/'pilot'/'freeze.json').is_file())
                    self.assertTrue((root/'pilot'/'action.model.json').is_file())
                    self.assertTrue((root/'pilot'/'state.model.json').is_file())
                    freeze = json.loads((root/'pilot'/'freeze.json').read_text())
                    self.assertFalse(freeze['held_out_transition_targets_read'])
                    observed.append(attempt['episode_id'])
                return original(corpus_root,group,attempt)
            with patch.object(pilot,'_load_attempt',observed_load):
                report = run_pilot(manifest,root/'pilot')
            self.assertEqual(report['status'],'completed',report.get('error'))
            self.assertEqual(len(observed),3)

    def test_failed_recordings_keep_denominators_and_exclusions_are_not_zero_fill(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = corpus(root/'corpus',failed=('test','open-jaw'))
            report = run_pilot(manifest,root/'pilot')
            self.assertEqual(report['status'],'completed',report.get('error'))
            self.assertEqual(report['counts']['attempts'],9)
            self.assertEqual(report['counts']['failed_recordings'],1)
            self.assertEqual(report['counts']['completed_recordings'],8)
            self.assertEqual(report['counts']['rows']['test'],6)
            self.assertEqual(report['counts']['case_attempts']['open-jaw'],3)
            failed = [a for a in report['attempts'] if a['recording_status']=='failed']
            self.assertEqual(len(failed),1)
            self.assertEqual(failed[0]['rows'],0)
            self.assertIsNone(failed[0]['source_sha256'])
            self.assertEqual(inspect_pilot(root/'pilot')['integrity'],'verified')

    def test_missing_after_measurement_is_excluded_with_reason_and_denominator(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = corpus(root/'corpus')
            manifest = json.loads(path.read_text())
            attempt = manifest['groups'][2]['attempts'][0]
            tape_path = root/'corpus'/attempt['path']
            tape = json.loads(tape_path.read_text())
            tape['steps'][-1]['after']['signals']['joint_position'] = {
                'status':'missing','captured_at_seconds':None,'age_seconds':None,'value':None}
            tape_path.unlink()
            receipt = write_tape(tape_path,tape)
            attempt.update(sha256=receipt['sha256'],bytes=receipt['bytes'])
            rewrite(path,manifest)
            report = run_pilot(path,root/'pilot')
            self.assertEqual(report['status'],'completed',report.get('error'))
            self.assertEqual(report['counts']['rows']['test'],8)
            self.assertEqual(report['counts']['attempted_steps']['test'],9)
            self.assertEqual(report['counts']['excluded_steps']['test'],1)
            self.assertIn('missing',report['attempts'][6]['exclusions'][0]['reason'])
            self.assertEqual(inspect_pilot(root/'pilot')['integrity'],'verified')

    def test_shared_settings_or_scene_assets_do_not_imply_duplicate_reset(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = corpus(root/'corpus')
            manifest = json.loads(path.read_text())
            reference = manifest['groups'][0]
            changed = manifest['groups'][2]
            changed['configuration'] = deepcopy(reference['configuration'])
            changed['configuration_sha256'] = reference['configuration_sha256']
            for attempt in changed['attempts']:
                attempt['provenance']['configuration_sha256'] = reference['configuration_sha256']
                attempt['provenance']['scene_sha256'] = reference['attempts'][0]['provenance']['scene_sha256']
            rewrite(path,manifest)
            # The distinct declared seeds/sessions still define distinct groups.
            _manifest(path)

    def test_rehashed_report_cannot_hide_missing_attempts_or_differing_frozen_selection(self):
        for field in ('attempts','selection'):
            with self.subTest(field=field), TemporaryDirectory() as directory:
                root = Path(directory)
                path = corpus(root/'corpus')
                report = run_pilot(path,root/'pilot')
                self.assertEqual(report['status'],'completed',report.get('error'))
                if field == 'attempts':
                    report['attempts'].pop()
                else:
                    report['selection']['action']['alpha'] = 999.
                raw = canonical(report)+b'\n'
                (root/'pilot'/'report.json').write_bytes(raw)
                integrity_path = root/'pilot'/'integrity.json'
                integrity = json.loads(integrity_path.read_text())
                integrity['artifacts']['report.json'].update(sha256=sha256(raw).hexdigest(),bytes=len(raw))
                integrity_path.write_bytes(canonical(integrity)+b'\n')
                with self.assertRaises(ValueError):
                    inspect_pilot(root/'pilot')

    def test_raw_source_hash_mismatch_is_retained_as_error(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = corpus(root/'corpus')
            tape = next((root/'corpus'/'tapes').glob('*nominal.json'))
            raw = tape.read_bytes()
            tape.write_bytes(raw.replace(b'0.04',b'0.05',1))
            report = run_pilot(manifest,root/'pilot')
            self.assertEqual(report['status'],'error')
            self.assertIn('source hash',report['error'])
            self.assertTrue((root/'pilot'/'report.json').is_file())
            self.assertEqual(len(report['attempts']),9)
            self.assertEqual(inspect_pilot(root/'pilot')['status'],'error')

    def test_seed_session_recording_and_hash_aliases_are_rejected(self):
        for alias in ('seed','session_id','reset_lineage_id','source_recording_id','sha256'):
            with self.subTest(alias=alias), TemporaryDirectory() as directory:
                root = Path(directory)
                path = corpus(root/'corpus')
                manifest = json.loads(path.read_text())
                a,b = manifest['groups'][0],manifest['groups'][2]
                if alias == 'configuration':
                    b['configuration'] = deepcopy(a['configuration'])
                    b['configuration_sha256'] = a['configuration_sha256']
                    for attempt in b['attempts']:
                        attempt['provenance']['configuration_sha256'] = a['configuration_sha256']
                elif alias == 'scene':
                    for attempt in b['attempts']:
                        attempt['provenance']['scene_sha256'] = a['attempts'][0]['provenance']['scene_sha256']
                elif alias in ('source_recording_id','sha256'):
                    b['attempts'][0][alias] = a['attempts'][0][alias]
                    if alias == 'source_recording_id':
                        b['attempts'][0]['provenance'][alias] = a['attempts'][0][alias]
                else:
                    b[alias] = a[alias]
                    if alias in ('seed','session_id'):
                        for attempt in b['attempts']:
                            attempt['provenance'][alias] = a[alias]
                rewrite(path,manifest)
                with self.assertRaisesRegex(ValueError,'Duplicate'):
                    _manifest(path)

    def test_paths_symlinks_duplicate_json_nonfinite_and_corrupt_frozen_artifact_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = corpus(root/'corpus')
            original = path.read_bytes()
            manifest = json.loads(original)
            manifest['groups'][0]['attempts'][0]['path'] = '../outside.json'
            rewrite(path,manifest)
            with self.assertRaisesRegex(ValueError,'relative'):
                _manifest(path)
            path.write_bytes(b'{"kind":1,"kind":2}')
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                _manifest(path)
            path.write_bytes(b'{"kind":NaN}')
            with self.assertRaisesRegex(ValueError,'Nonfinite'):
                _manifest(path)
            path.write_bytes(original)
            tape_path = root/'corpus'/json.loads(original)['groups'][0]['attempts'][0]['path']
            real_path = tape_path.with_suffix('.original')
            tape_path.rename(real_path)
            try:
                tape_path.symlink_to(real_path)
            except OSError:
                real_path.rename(tape_path)
            else:
                with self.assertRaisesRegex(ValueError,'regular'):
                    _manifest(path)
                tape_path.unlink()
                real_path.rename(tape_path)
            report = run_pilot(path,root/'pilot')
            self.assertEqual(report['status'],'completed',report.get('error'))
            frozen = root/'pilot'/'state.model.json'
            frozen.write_bytes(frozen.read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'hash'):
                inspect_pilot(root/'pilot')

    def test_joint_and_jaw_metrics_keep_separate_units_and_baseline_errors(self):
        metric = _metrics([{'target_delta':[1.,0.,0.,0.,0.,0.,.003]}],[[0.]*7])
        self.assertAlmostEqual(metric['joint_rmse_rad'],(1/6)**.5)
        self.assertAlmostEqual(metric['jaw_rmse_m'],.003)
        self.assertEqual(metric['joint_max_abs_error_rad'],1.)
        self.assertIsNone(_metrics([],[])['joint_rmse_rad'])

    def test_generator_predeclares_plan_and_preserves_all_attempts_on_missing_runtime(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            def fail(**kwargs):
                self.assertTrue((root/'corpus'/'collection-plan.json').is_file())
                raise RuntimeError('deliberate native recorder failure')
            with patch('dvidia_training.transition_pilot.record_episode',side_effect=fail):
                manifest = generate_corpus(root/'corpus',groups=6)
            self.assertEqual(manifest['protocol']['split_counts'],{'train':4,'development':1,'test':1})
            self.assertEqual(manifest['status'],'completed-with-failed-attempts')
            self.assertEqual(sum(len(g['attempts']) for g in manifest['groups']),18)
            self.assertTrue(all(a['status']=='failed' for g in manifest['groups'] for a in g['attempts']))
            _manifest(root/'corpus'/'corpus.json')
            with self.assertRaisesRegex(ValueError,'fresh'):
                generate_corpus(root/'corpus',groups=6)


if __name__ == '__main__':
    unittest.main()

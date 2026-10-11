"""Failure accounting and artifact boundaries, independent of native task physics."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dvidia_training import reference_task_benchmark as bench


def fixture(root):
    (root/'attempts').mkdir()
    protocol = json.loads(json.dumps(bench.make_protocol(seed=1, nominal=1, stress=0)))
    scenes = protocol['scenes']
    bench._write(root/'protocol.json', protocol)
    rows = []
    for index, control in enumerate(bench.CONTROLS):
        attempt = {'id': 'nominal-00--'+control, 'scene_id': 'nominal-00', 'control': control,
                   'profile': protocol['profile'], 'physical_robot_ready': False,
                   'seed': scenes[0]['seed'], 'config': deepcopy(scenes[0]['config']),
                   'valid': index != 2, 'success': index == 0, 'trajectory': [], 'actions': [], 'arm_tape': []}
        name = 'attempts/'+attempt['id']+'.json'
        bench._write(root/name, attempt)
        rows.append({**bench._summary(attempt), 'result_file': name})
    report = {'schema': bench.SCHEMA, 'physical_robot_ready': False, 'learned_policy': False,
              'profile': protocol['profile'], 'cohort': protocol['cohort'],
              'implementation_sha256': deepcopy(protocol['implementation_sha256']),
              'protocol': protocol, 'protocol_sha256': sha256((root/'protocol.json').read_bytes()).hexdigest(),
              'scenes': scenes, 'attempted': 3, 'valid_attempts': 2, 'attempts': rows,
              'summary': bench.aggregate(rows, scenes)}
    bench._write(root/'report.json', report)
    (root/'report.html').write_text('fixture', encoding='utf-8')
    reseal(root)
    return report


def reseal(root):
    paths = [root/'report.json',root/'report.html',root/'protocol.json',*sorted((root/'attempts').glob('*.json'))]
    manifest = {'schema': 'dvidia.reference-task-integrity/v1', 'files': {
        str(p.relative_to(root)).replace('\\','/'): {'bytes':p.stat().st_size, 'sha256':sha256(p.read_bytes()).hexdigest()} for p in paths}}
    (root/'integrity.json').write_bytes(bench._json_bytes(manifest))


def update(root, name, value):
    (root/name).write_bytes(bench._json_bytes(value))
    reseal(root)


class BundleTests(unittest.TestCase):
    def test_valid_bundle_retains_invalid_control_in_denominator_and_exact_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = fixture(root)
            result = bench.inspect_task_bundle(root)
            self.assertEqual(result['attempted'], 3)
            self.assertEqual(result['valid_attempts'], 2)
            self.assertEqual(result['summary'][2]['failed_or_invalid'], 1)
            self.assertEqual(result['summary'][2]['attempted'], 1)
            loaded, files = bench.load_task_bundle(root)
            self.assertEqual(loaded, report)
            self.assertEqual(files['integrity.json'], (root/'integrity.json').read_bytes())
            self.assertFalse(result['physical_robot_ready'])

    def test_unsealed_edit_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root)
            (root/'report.html').write_text('edited', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'digest/size'):
                bench.load_task_bundle(root)

    def test_resealed_dropped_or_duplicated_attempt_rejected(self):
        for variant in ('dropped', 'duplicate', 'false_success', 'summary'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                report = fixture(root)
                if variant == 'dropped':
                    report['attempts'].pop()
                    report['attempted'] -= 1
                elif variant == 'duplicate':
                    report['attempts'][2] = deepcopy(report['attempts'][0])
                elif variant == 'false_success':
                    report['attempts'][2]['success'] = True
                else:
                    report['summary'][2]['attempted'] = 0
                update(root, 'report.json', report)
                with self.assertRaises(ValueError):
                    bench.load_task_bundle(root)

    def test_resealed_report_disagrees_with_trace(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = fixture(root)
            report['attempts'][0]['success'] = False
            report['summary'] = bench.aggregate(report['attempts'], report['scenes'])
            update(root,'report.json',report)
            with self.assertRaisesRegex(ValueError, 'summary differs'):
                bench.load_task_bundle(root)

    def test_resealed_matching_raw_and_summary_cannot_change_declared_attempt(self):
        # Hashes and raw/summary equality are insufficient when both carry a
        # changed scene or embodiment declaration. No physics is recomputed.
        for field, replacement in (
                ('profile', 'different-robot-profile'), ('physical_robot_ready', True),
                ('seed', 2), ('seed', True), ('id', 'different-attempt'),
                ('control', 'undeclared-control'), ('config', {'mass_kg': .05})):
            with self.subTest(field=field, replacement=replacement), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                report = fixture(root)
                row = report['attempts'][0]
                raw = json.loads((root/row['result_file']).read_bytes())
                raw[field] = deepcopy(replacement)
                row[field] = deepcopy(replacement)
                (root/row['result_file']).write_bytes(bench._json_bytes(raw))
                update(root, 'report.json', report)
                with self.assertRaises(ValueError):
                    bench.load_task_bundle(root)

    def test_report_profile_cohort_and_source_declarations_match_frozen_protocol(self):
        for field, replacement in (
                ('profile', 'different-robot-profile'), ('cohort', 'confirmation'),
                ('implementation_sha256', {'reference_task.py': 'a'*64})):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                report = fixture(root)
                report[field] = replacement
                update(root, 'report.json', report)
                with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                    bench.load_task_bundle(root)

    def test_protocol_type_and_malformed_shared_source_hashes_rejected(self):
        for replacement in ([], None, {'reference_task.py': 'not-a-sha256'}):
            with self.subTest(replacement=replacement), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                report = fixture(root)
                protocol = deepcopy(report['protocol'])
                protocol['implementation_sha256'] = deepcopy(replacement)
                report['implementation_sha256'] = deepcopy(replacement)
                report['protocol'] = protocol
                (root/'protocol.json').write_bytes(bench._json_bytes(protocol))
                report['protocol_sha256'] = sha256((root/'protocol.json').read_bytes()).hexdigest()
                update(root, 'report.json', report)
                with self.assertRaises(ValueError):
                    bench.load_task_bundle(root)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = fixture(root)
            report['protocol'] = []
            (root/'protocol.json').write_bytes(bench._json_bytes([]))
            report['protocol_sha256'] = sha256((root/'protocol.json').read_bytes()).hexdigest()
            update(root, 'report.json', report)
            with self.assertRaisesRegex(ValueError, 'protocol'):
                bench.load_task_bundle(root)

    def test_manifest_path_escape_and_parent_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root)
            manifest = json.loads((root/'integrity.json').read_bytes())
            manifest['files']['../outside.json'] = {'bytes':0,'sha256':sha256(b'').hexdigest()}
            (root/'integrity.json').write_bytes(bench._json_bytes(manifest))
            with self.assertRaisesRegex(ValueError,'allowlist'):
                bench.load_task_bundle(root)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            fixture(root)
            (root/'attempts').rename(root/'original')
            try:
                (root/'attempts').symlink_to(root/'original',target_is_directory=True)
            except OSError:
                self.skipTest('No symlink permission on this platform')
            with self.assertRaisesRegex(ValueError,'symlink'):
                bench.load_task_bundle(root)

    def test_resource_bound_checked_before_read(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root)
            with patch.object(bench, 'MAX_BUNDLE_BYTES', 1), self.assertRaisesRegex(ValueError,'byte bound'):
                bench.load_task_bundle(root)


class ExecutionTests(unittest.TestCase):
    def test_seed_plan_is_reproducible_and_rejects_invalid_counts(self):
        one = bench.make_protocol(seed=420100, nominal=2, stress=1)
        self.assertEqual(one, bench.make_protocol(seed=420100, nominal=2, stress=1))
        self.assertNotEqual(one['scenes'], bench.make_protocol(seed=420101, nominal=2, stress=1)['scenes'])
        self.assertEqual(one['planned_attempts'], 9)
        self.assertEqual([s['config']['friction'] for s in one['scenes']], [.8,.8,.15])
        for values in ({'seed':True}, {'seed':999976}, {'nominal':0}, {'stress':13}, {'cohort':'test'}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                bench.make_protocol(**values)

    def test_runtime_errors_retained_and_dependent_control_is_invalid(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(bench.sim, 'verify_assets',return_value={}), \
                patch('dvidia_training.reference_task.run_episode', side_effect=RuntimeError('fixture failure')) as run:
            output=Path(folder)/'run'
            report=bench.run_benchmark('unused',output,nominal=1,stress=0)
            self.assertEqual(run.call_count,2)  # no invented tape for the dependent control
            self.assertEqual(report['attempted'],3)
            self.assertEqual(report['valid_attempts'],0)
            self.assertEqual([x['success'] for x in report['attempts']],[False]*3)
            self.assertIn('unavailable', report['attempts'][1]['error'])
            self.assertEqual(bench.inspect_task_bundle(output)['attempted'],3)
            with self.assertRaises(FileExistsError):
                bench.run_benchmark('unused',output,nominal=1,stress=0)

    def test_open_jaw_receives_exact_baseline_arm_tape(self):
        tape=[[.1, .2, .3, .4, .5, .6],[-.2,.4,-.6,.8,-1.,1.2]]*500
        seen=[]
        def run(_assets, **kwargs):
            seen.append(kwargs)
            return {'valid':True,'success':False,'trajectory':[], 'arm_tape':deepcopy(tape),'actions':[]}
        with tempfile.TemporaryDirectory() as folder, patch.object(bench.sim,'verify_assets',return_value={}), \
                patch('dvidia_training.reference_task.run_episode',side_effect=run):
            report=bench.run_benchmark('unused',Path(folder)/'run',nominal=1,stress=0)
            self.assertEqual(seen[1]['control'],'replay_open_jaw')
            self.assertEqual(seen[1]['arm_tape'],tape)
            self.assertNotIn('arm_tape',report['attempts'][0])
            self.assertNotIn('trajectory',report['attempts'][0])

    def test_partial_placement_tape_cannot_be_extended_into_a_matched_control(self):
        seen = []
        def run(_assets, **kwargs):
            seen.append(kwargs['control'])
            return {'valid': True, 'success': False, 'trajectory': [], 'actions': [],
                    'arm_tape': [[.1, .2, .3, .4, .5, .6]]}
        with tempfile.TemporaryDirectory() as folder, patch.object(bench.sim, 'verify_assets', return_value={}), \
                patch('dvidia_training.reference_task.run_episode', side_effect=run):
            output = Path(folder)/'run'
            report = bench.run_benchmark('unused', output, nominal=1, stress=0)
            self.assertEqual(seen, ['placement', 'hold'])
            self.assertEqual(report['attempted'], 3)
            self.assertFalse(report['attempts'][1]['valid'])
            self.assertFalse(report['attempts'][1]['success'])
            self.assertIn('unavailable', report['attempts'][1]['error'])
            self.assertEqual(bench.inspect_task_bundle(output)['attempted'], 3)


if __name__ == '__main__':
    unittest.main()

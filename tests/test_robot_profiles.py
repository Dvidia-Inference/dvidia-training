"""Pinned robot identities, native command units and retained CPU smoke evidence."""
from copy import deepcopy
from hashlib import sha256
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from dvidia_training import robot_profiles as profiles


class RobotProfileContractTests(unittest.TestCase):
    def test_catalog_distinguishes_arm_axes_and_native_gripper_units(self):
        piper, so101 = profiles.list_profiles()
        self.assertEqual((piper['arm_axes'], so101['arm_axes']), (6, 5))
        self.assertEqual((piper['gripper_axes'], so101['gripper_axes']), (1, 1))
        self.assertEqual(piper['native_command_units'], ['rad']*6+['m'])
        self.assertEqual(so101['native_command_units'], ['rad']*6)
        self.assertEqual(piper['simulation_asset_license'], 'MIT')
        self.assertEqual(so101['simulation_asset_license'], 'Apache-2.0')
        self.assertIn('not total jaw gap', piper['gripper_command'])
        self.assertIn('must be assessed separately', piper['openness_scope'])
        with self.assertRaises(ValueError):
            profiles._profile('unknown')

    def test_listing_has_no_optional_native_dependency_or_network(self):
        script = "import sys;sys.modules['mujoco']=None;sys.modules['numpy']=None;from dvidia_training.robot_profiles import list_profiles;assert len(list_profiles())==2"
        self.assertEqual(subprocess.run([sys.executable, '-c', script], capture_output=True, text=True).returncode, 0)

    def test_packaged_allowlist_is_fixed_bounded_and_preserves_licenses(self):
        lock = profiles._lock()
        self.assertEqual(lock['commit'], '0059d4335f8156206f63a35662313385f7ad6d74')
        self.assertEqual(lock['schema'], 'dvidia.robot-profiles-lock/v1')
        for item in lock['profiles']:
            names = [record['path'] for record in item['files']]
            self.assertEqual(len(names), len(set(names)))
            self.assertIn(item['folder']+'/LICENSE', names)
            self.assertIn(item['folder']+'/'+item['model'], names)
            self.assertLess(sum(record['bytes'] for record in item['files']), 40000000)
            for record in item['files']:
                self.assertTrue(record['path'].startswith(item['folder']+'/'))
                self.assertNotIn('..', Path(record['path']).parts)
                self.assertEqual(len(record['sha256']), 64)

    def test_fetch_is_explicit_pinned_and_rejects_redirected_or_wrong_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            for label, redirect in (('digest', False), ('redirect', True)):
                response = unittest.mock.MagicMock()
                record = profiles._profile('so101')['files'][0]
                exact = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{profiles._lock()['commit']}/{record['path']}"
                response.__enter__.return_value = response
                response.geturl.return_value = 'https://example.org/assets' if redirect else exact
                response.read.return_value = b'incorrect bytes'
                with patch.object(profiles, 'urlopen', return_value=response) as opening:
                    with self.assertRaises(ValueError):
                        profiles.fetch_assets('so101', Path(temporary)/label)
                self.assertEqual(opening.call_args[0][0].full_url, exact)
                if not redirect:
                    response.read.assert_called_once_with(record['bytes']+1)

    def test_fetch_allows_shared_asset_root_but_never_overwrites_a_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'shared'
            record = {'path': 'agilex_piper/LICENSE', 'bytes': 7, 'sha256': sha256(b'fixture').hexdigest()}
            for name, folder in (('piper', 'agilex_piper'), ('so101', 'robotstudio_so101')):
                item = {'id': name, 'folder': folder, 'files': [{**record, 'path': folder+'/LICENSE'}]}
                response = unittest.mock.MagicMock()
                response.__enter__.return_value = response
                response.geturl.return_value = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{profiles._lock()['commit']}/{folder}/LICENSE"
                response.read.return_value = b'fixture'
                with patch.object(profiles, '_profile', return_value=item), patch.object(profiles, 'urlopen', return_value=response), patch.object(profiles, 'verify_assets', return_value={'profile': name}):
                    self.assertEqual(profiles.fetch_assets(name, root)['profile'], name)
                    with self.assertRaises(FileExistsError):
                        profiles.fetch_assets(name, root)
                self.assertEqual((root/folder/'LICENSE').read_bytes(), b'fixture')

    def test_invalid_benchmark_budgets_reject_before_setup(self):
        for options in ({'duration': True}, {'duration': float('nan')}, {'duration': 10**1000},
                        {'duration': 2.1}, {'duration': .11}, {'repeats': True}, {'repeats': 4},
                        {'timestep': .003}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                profiles.benchmark('piper', 'unused', 'unused-output', **options)

    def test_setup_failure_is_an_attempt_and_does_not_claim_valid_physics(self):
        with patch.object(profiles, 'build_model', side_effect=ValueError('fixture native setup failed')):
            attempt = profiles._attempt('so101', 'unused', 'hold', 0, .4, .001)
        self.assertFalse(attempt['valid'])
        self.assertEqual(attempt['trajectory'], [])
        self.assertEqual(attempt['simulated_seconds'], 0)
        self.assertEqual(attempt['error']['type'], 'ValueError')
        self.assertEqual(attempt['timing']['control_boundaries'], 0)


@unittest.skipUnless(os.environ.get('DVIDIA_ROBOT_PROFILE_ASSETS'), 'Set DVIDIA_ROBOT_PROFILE_ASSETS to pinned model assets')
class NativeRobotProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets = Path(os.environ['DVIDIA_ROBOT_PROFILE_ASSETS'])
        cls.directory = tempfile.TemporaryDirectory()
        cls.results = {}
        for name in ('piper', 'so101'):
            output = Path(cls.directory.name)/name
            cls.results[name] = profiles.benchmark(name, cls.assets, output)
        cls.mj, cls.np = profiles.sim._native_imports()

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_both_native_models_match_topology_actuator_units_and_limits(self):
        for name in ('piper', 'so101'):
            item = profiles._profile(name)
            model = profiles.build_model(name, self.assets)
            self.assertEqual({key: int(getattr(model, key)) for key in ('nq', 'nv', 'nu', 'neq')}, item['expected_dimensions'])
            self.assertTrue(all(model.actuator_ctrllimited))
            self.assertTrue(all(model.actuator_forcelimited))
            for index, units in enumerate(item['native_command_units']):
                joint = model.actuator_trnid[index, 0]
                self.assertEqual(model.jnt_type[joint], self.mj.mjtJoint.mjJNT_SLIDE if units == 'm' else self.mj.mjtJoint.mjJNT_HINGE)
            audit = profiles.audit(name, self.assets)
            self.assertFalse(audit['physical_robot_ready'])
            self.assertFalse(audit['task_qualified'])
            self.assertEqual(audit['dimensions']['nu'], item['expected_dimensions']['nu'])

    def test_native_smoke_retains_all_cases_and_causal_commands_without_task_claim(self):
        for name in ('piper', 'so101'):
            inspected = profiles.inspect_benchmark(Path(self.directory.name)/name)
            report = inspected['report']
            self.assertEqual((inspected['attempts'], inspected['valid_attempts']), (3, 3))
            self.assertFalse(report['task_qualified'])
            self.assertFalse(report['physical_robot_ready'])
            self.assertEqual(report['settings']['cases'], ['hold', 'arm_sweep', 'gripper_cycle'])
            model = profiles.build_model(name, self.assets)
            for attempt in report['attempts']:
                self.assertIsNone(attempt['error'])
                self.assertEqual(attempt['warnings'], [])
                self.assertEqual(len(attempt['trajectory']), 21)
                self.assertEqual(attempt['simulated_seconds'], .4)
                for index, row in enumerate(attempt['trajectory']):
                    self.assertAlmostEqual(row['t_seconds'], index*.02)
                    issued = row['issued_controls_native']
                    if index == 20:
                        self.assertIsNone(issued)
                    else:
                        self.assertEqual(len(issued), model.nu)
                        self.assertTrue(self.np.all(self.np.asarray(issued) >= model.actuator_ctrlrange[:, 0]))
                        self.assertTrue(self.np.all(self.np.asarray(issued) <= model.actuator_ctrlrange[:, 1]))
                self.assertEqual(attempt['timing']['control_boundaries'], 20)

    def test_commanded_arm_and_jaw_cases_change_native_state_against_matched_hold(self):
        for name in ('piper', 'so101'):
            report = profiles.inspect_benchmark(Path(self.directory.name)/name)['report']
            held, arm, jaw = [attempt['trajectory'] for attempt in report['attempts']]
            axes = report['profile']['arm_axes']
            arm_change = max(abs(a-b) for moving, baseline in zip(arm, held)
                             for a, b in zip(moving['qpos_native'][:axes], baseline['qpos_native'][:axes]))
            jaw_change = max(abs(a-b) for moving, baseline in zip(jaw, held)
                             for a, b in zip(moving['qpos_native'][axes:], baseline['qpos_native'][axes:]))
            self.assertGreater(arm_change, 1e-5, f'{name} arm commands must affect actual native positions')
            self.assertGreater(jaw_change, 1e-6, f'{name} jaw commands must affect actual native positions')

    def test_asset_and_receipt_tampering_symlinks_and_oversize_are_rejected(self):
        item, blobs = profiles._payload('so101', self.assets)
        changed = dict(blobs)
        first = item['files'][0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root/item['folder']
            folder.symlink_to(self.assets/item['folder'], target_is_directory=True)
            with self.assertRaises(ValueError):
                profiles.verify_assets('so101', root)
        output = Path(self.directory.name)/'piper'
        report_path = output/'report.json'
        original = report_path.read_bytes()
        try:
            report_path.write_bytes(original+b' ')
            with self.assertRaises(ValueError):
                profiles.inspect_benchmark(output)
        finally:
            report_path.write_bytes(original)
        with patch.object(profiles, '_bounded_file', side_effect=ValueError('oversize')):
            with self.assertRaises(ValueError):
                profiles.inspect_benchmark(output)

    def test_semantic_axes_and_nonfinite_commands_reject_even_after_resealing(self):
        source = Path(self.directory.name)/'so101'
        for mutation in ('axes', 'commands', 'source', 'inventory', 'nonterminal', 'terminal',
                         'redundant', 'gripper', 'quaternion', 'force', 'contact'):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                for path in source.rglob('*'):
                    if path.is_file():
                        target = root/path.relative_to(source)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(path.read_bytes())
                report = json.loads((root/'report.json').read_text())
                if mutation == 'axes':
                    report['profile']['arm_axes'] = 6
                elif mutation == 'source':
                    report['source']['asset_commit'] = '0'*40
                elif mutation == 'inventory':
                    (root/'unexpected.bin').write_bytes(b'unlisted')
                else:
                    if mutation == 'commands':
                        report['attempts'][0]['trajectory'][0]['issued_controls_native'][0] = float('nan')
                    elif mutation == 'nonterminal':
                        report['attempts'][0]['trajectory'][0]['issued_controls_native'] = None
                    elif mutation == 'terminal':
                        report['attempts'][0]['trajectory'][-1]['issued_controls_native'] = [0]*6
                    elif mutation == 'redundant':
                        report['attempts'][0]['trajectory'][0]['arm_joint_position_rad'][0] += .1
                    elif mutation == 'gripper':
                        report['attempts'][0]['trajectory'][0]['gripper_joint_position_native'] = []
                    elif mutation == 'quaternion':
                        report['attempts'][0]['trajectory'][0]['terminal_link_quaternion_wxyz'] = [0]*4
                    elif mutation == 'force':
                        report['attempts'][0]['trajectory'][0]['actuator_force_native'][0] = float('inf')
                    else:
                        report['attempts'][0]['trajectory'][0]['contact_count'] = True
                    (root/'attempt-hold-0.json').write_text(json.dumps(report['attempts'][0]))
                (root/'report.json').write_text(json.dumps(report))
                integrity = json.loads((root/'integrity.json').read_text())
                for record in integrity['files']:
                    content = (root/record['path']).read_bytes()
                    record.update(bytes=len(content), sha256=sha256(content).hexdigest())
                (root/'integrity.json').write_text(json.dumps(integrity))
                with self.assertRaises(ValueError):
                    profiles.inspect_benchmark(root)


if __name__ == '__main__':
    unittest.main()

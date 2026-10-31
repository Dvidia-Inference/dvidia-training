"""Wrong embodiment, changed inputs and unqualified backends fail before motion."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dvidia_training import piper_imitation as imitation
from dvidia_training import arm_skill_install as install
from dvidia_training import arm_interface_cli as cli


class ArmSkillInstallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = [{'features': [0.] * imitation.FEATURE_COUNT,
                 'targets': [0.] * 6 + [.5], 'case_id': 'fixture-train'} for _ in range(16)]
        cls.raw = imitation.canonical(imitation.train({'scene_ids': ['fixture-train'], 'samples': rows},
                                                      hidden_units=8, epochs=1, batch_size=8, seed=7)) + b'\n'

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root/'model.json'
        self.model.write_bytes(self.raw)
        self.output = self.root/'installation'
        self.binding = {**install.arm_interfaces.contract('piper'),
                        'runtime_binding': {'test_fixture': True}}
        self.build = patch.object(install.arm_interfaces, 'build', return_value=(None, None, self.binding))
        self.native = self.build.start()
        self.addCleanup(self.build.stop)

    def make_install(self):
        return install.install_model(self.model, 'piper', 'unused-fixture-assets', self.output)

    def test_check_accepts_existing_policy_but_never_calls_native_or_grants_task_success(self):
        record = install.check_model(self.model, 'piper')
        self.assertEqual(record['model_sha256'], sha256(self.raw).hexdigest())
        self.assertFalse(record['physical_robot_ready'])
        self.assertFalse(record['task_qualified_by_installation'])
        self.assertEqual(record['control_period_s'], .02)
        self.native.assert_not_called()

    def test_same_joint_count_does_not_transfer_policy_or_accept_hardware_backend(self):
        for robot, backend in [('so101', install.INTERFACE), ('ur5e-2f85', install.INTERFACE),
                               ('piper', 'ros2'), ('piper', 'lerobot')]:
            with self.subTest(robot=robot, backend=backend), self.assertRaises(ValueError):
                install.install_model(self.model, robot, 'unused', self.output, interface=backend)
        self.native.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_install_copies_exact_model_and_is_separate_from_execution(self):
        with patch('dvidia_training.piper_learned_skill.run_learned_skill', side_effect=AssertionError('Install moved robot')):
            record = self.make_install()
            inspected = install.inspect_installation(self.output, 'unused-fixture-assets')
        self.assertEqual(inspected, record)
        self.assertEqual((self.output/'model.json').read_bytes(), self.raw)
        self.assertEqual(record['status'], 'installed_simulation_candidate')
        with self.assertRaises(ValueError):
            self.make_install()

    def test_tampered_model_receipt_runtime_and_native_binding_reject(self):
        self.make_install()
        receipt_path = self.output/'installation.json'
        original = receipt_path.read_bytes()
        receipt = json.loads(original)
        for field, replacement in [('status', 'qualified'), ('model_file', '../elsewhere'),
                                   ('runtime_sha256', {}), ('native_binding', {})]:
            changed = deepcopy(receipt)
            changed[field] = replacement
            # Resealing an attacker-changed receipt cannot grant compatibility.
            del changed['installation_sha256']
            changed['installation_sha256'] = sha256(install._canonical(changed)).hexdigest()
            receipt_path.write_bytes(install._canonical(changed))
            with self.subTest(field=field), self.assertRaises(ValueError):
                install.inspect_installation(self.output, 'unused')
        receipt_path.write_bytes(original)
        (self.output/'model.json').write_bytes(self.raw+b' ')
        with self.assertRaises(ValueError):
            install.inspect_installation(self.output, 'unused')

    def test_duplicate_json_keys_and_symlink_models_are_rejected(self):
        self.make_install()
        path = self.output/'installation.json'
        path.write_bytes(b'{"schema":"a","schema":"b"}')
        with self.assertRaises(ValueError):
            install.inspect_installation(self.output, 'unused')
        link = self.root/'linked.json'
        link.symlink_to(self.model)
        with self.assertRaises(ValueError):
            install.check_model(link, 'piper')

    def test_incompatible_native_assets_are_rejected_before_reserving_installation(self):
        self.native.side_effect = ValueError('Asset digest mismatch')
        with self.assertRaisesRegex(ValueError, 'digest'):
            self.make_install()
        self.assertFalse(self.output.exists())

    def test_modified_runtime_refuses_installed_execution(self):
        self.make_install()
        with patch.object(install, '_runtime', return_value={}), \
             patch('dvidia_training.piper_learned_skill.run_learned_skill') as runner:
            with self.assertRaisesRegex(ValueError, 'Runtime changed'):
                install.run_installation(self.output, 'unused', self.root/'attempt')
            runner.assert_not_called()

    def test_run_preserves_failed_attempt_and_dispatches_only_verified_copy(self):
        record = self.make_install()
        output = self.root/'attempt'
        def run(model, assets, result, **kwargs):
            self.assertEqual(model, self.output/'model.json')
            self.assertEqual(kwargs, {'start_x': .309, 'start_y': -.04, 'goal': 1})
            result.mkdir()
            return {'success': False, 'reason': 'dropped', 'model_sha256': sha256(self.raw).hexdigest()}
        with patch('dvidia_training.piper_learned_skill.run_learned_skill', side_effect=run) as runner:
            result = install.run_installation(self.output, 'unused', output, start_x=.309, start_y=-.04, goal=1)
        runner.assert_called_once()
        self.assertFalse(result['success'])
        binding = json.loads((output/'arm-installation-binding.json').read_text())
        self.assertEqual(binding['installation_sha256'], record['installation_sha256'])
        self.assertTrue(binding['run_model_matches_installation'])


if __name__ == '__main__':
    unittest.main()

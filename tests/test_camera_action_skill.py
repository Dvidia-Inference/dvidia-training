"""Data-only camera policy installation cannot grant success or change arms."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dvidia_training import arm_skill_install as dispatch
from dvidia_training import camera_action_model as model
from dvidia_training import camera_action_skill as skill


class CameraSkillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = dict(schema=model.DATASET_SCHEMA, features=[[0.]*model.FEATURE_COUNT]*16,
                    targets=[[0.]*6+[.02]]*16, groups=['training-fixture']*16,
                    provenance={'kind': 'unit-test-only'})
        cls.document = model.train(data)
        cls.raw = model.canonical(cls.document)+b'\n'

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root/'model.json'
        self.source.write_bytes(self.raw)
        self.output = self.root/'installed'
        self.binding = {'test_fixture': True}
        self.native = patch.object(skill.arm_interfaces, 'build', return_value=(None, None, self.binding)).start()
        self.addCleanup(patch.stopall)

    def test_dispatch_checks_camera_schema_without_motion_or_qualification(self):
        result = dispatch.check_model(self.source, 'piper')
        self.assertEqual(result['runner'], skill.RUNNER)
        self.assertFalse(result['task_qualified_by_installation'])
        self.assertFalse(result['physical_robot_ready'])
        self.native.assert_not_called()

    def test_other_arms_interfaces_and_invalid_model_fail_before_motion(self):
        for robot, interface in [('so101', skill.INTERFACE), ('ur5e-2f85', skill.INTERFACE), ('piper', 'ros2')]:
            with self.assertRaises(ValueError):
                dispatch.install_model(self.source, robot, 'unused', self.output, interface=interface)
        self.native.assert_not_called()
        bad = deepcopy(self.document); bad['physical_robot_ready'] = True
        self.source.write_bytes(model.canonical(bad))
        with self.assertRaises(ValueError):
            dispatch.check_model(self.source, 'piper')

    def test_exact_copy_inspection_and_fresh_install(self):
        result = dispatch.install_model(self.source, 'piper', 'unused', self.output)
        self.assertEqual((self.output/'model.json').read_bytes(), self.raw)
        self.assertEqual(dispatch.inspect_installation(self.output, 'unused'), result)
        self.assertEqual(result['status'], 'installed_simulation_candidate')
        with self.assertRaises(ValueError):
            dispatch.install_model(self.source, 'piper', 'unused', self.output)

    def test_tampered_receipt_model_and_runtime_are_rejected(self):
        dispatch.install_model(self.source, 'piper', 'unused', self.output)
        path = self.output/'installation.json'; original = path.read_bytes()
        receipt = json.loads(original)
        for key, value in [('status', 'qualified'), ('runtime_sha256', {}), ('native_binding', {})]:
            bad = deepcopy(receipt); bad[key] = value
            path.write_bytes(skill.canonical(bad))
            with self.assertRaises(ValueError):
                dispatch.inspect_installation(self.output, 'unused')
        path.write_bytes(original)
        with patch.object(skill, 'runtime', return_value={}):
            with self.assertRaisesRegex(ValueError, 'Runtime|runtime'):
                dispatch.inspect_installation(self.output, 'unused')
        (self.output/'model.json').write_bytes(self.raw+b' ')
        with self.assertRaises(ValueError):
            dispatch.inspect_installation(self.output, 'unused')

    def test_run_dispatch_preserves_failure_and_native_rollout_inventory(self):
        dispatch.install_model(self.source, 'piper', 'unused', self.output)
        out = self.root/'attempt'
        def run(assets, output, **kwargs):
            self.assertEqual(kwargs['mode'], 'student')
            self.assertNotIn('teacher', kwargs)
            self.assertEqual(kwargs['case']['object_position_m'], [.30, -.03, .082])
            self.assertEqual(kwargs['case']['policy_artifact_sha256'], sha256(self.raw).hexdigest())
            output.mkdir()
            (output/'integrity.json').write_text('{}')
            (output/'rollout.json').write_text('{"fixture":true}')
            return {'strict_success': False, 'status': 'complete', 'reason': 'failed_task'}
        with patch('dvidia_training.camera_action_capture.run_rollout', side_effect=run):
            result = dispatch.run_installation(self.output, 'unused', out)
        self.assertFalse(result['success'])
        self.assertEqual(sorted(p.name for p in out.iterdir()), ['integrity.json', 'rollout.json'])
        self.assertTrue((self.root/'attempt-installation.json').is_file())
        link = json.loads((self.root/'attempt-installation.json').read_bytes())
        self.assertEqual(link['rollout_sha256'], sha256((out/'rollout.json').read_bytes()).hexdigest())
        self.assertEqual(link['integrity_sha256'], sha256((out/'integrity.json').read_bytes()).hexdigest())
        with patch('dvidia_training.camera_action_capture.run_rollout') as runner:
            with self.assertRaisesRegex(ValueError, 'fresh execution'):
                dispatch.run_installation(self.output, 'unused', out)
            runner.assert_not_called()

    def test_out_of_envelope_refuses_before_execution(self):
        dispatch.install_model(self.source, 'piper', 'unused', self.output)
        with patch('dvidia_training.camera_action_capture.run_rollout') as runner:
            for kwargs in ({'start_x': .284}, {'goal': True}, {'start_y': float('nan')}):
                with self.assertRaises(ValueError):
                    dispatch.run_installation(self.output, 'unused', self.root/'attempt', **kwargs)
            runner.assert_not_called()


if __name__ == '__main__':
    unittest.main()

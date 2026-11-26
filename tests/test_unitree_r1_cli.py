import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dvidia_training import arm_interface_cli


def invoke(*arguments):
    output, errors = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
        code = arm_interface_cli.main(list(arguments))
    return code, json.loads(output.getvalue()) if output.getvalue() else None, errors.getvalue()


class UnitreeReferenceCliTests(unittest.TestCase):
    def test_reference_discovery_keeps_original_native_catalog(self):
        code, native, _ = invoke('list')
        self.assertEqual(code, 0)
        self.assertEqual([row['id'] for row in native], ['piper', 'so101', 'ur5e-2f85'])
        code, extended, _ = invoke('list', '--include-references')
        self.assertEqual(code, 0)
        self.assertEqual(extended[:-1], native)
        family = extended[-1]
        self.assertEqual((family['arm_axes'], family['gripper_axes']), (7, 0))
        self.assertIsNone(family['side'])
        self.assertEqual(family['available_sides'], ['left', 'right'])
        self.assertFalse(family['skill_install_supported'])
        self.assertFalse(family['physical_robot_ready'])
        self.assertEqual(invoke('describe', 'unitree-r1-7a'), (0, family, ''))

    def test_describe_resolves_each_side_to_its_source(self):
        left, right = [invoke('describe', 'unitree-r1-7a', '--side', side)
                       for side in ('left', 'right')]
        self.assertEqual((left[0], right[0]), (0, 0))
        self.assertEqual(left[1]['side'], 'left')
        self.assertEqual(right[1]['side'], 'right')
        self.assertNotEqual(left[1]['source'], right[1]['source'])
        self.assertEqual(invoke('describe', 'piper', '--side', 'right')[0], 2)

    def test_pose_requires_explicit_side_and_seven_coordinates(self):
        for args in (['pose', 'unitree-r1-7a', '--joints', *['0'] * 7],
                     ['pose', 'unitree-r1-7a', '--side', 'right', '--joints', *['0'] * 6],
                     ['benchmark-kinematics', 'unitree-r1-7a']):
            with self.subTest(args=args), self.assertRaises(SystemExit) as error:
                invoke(*args)
            self.assertEqual(error.exception.code, 2)
        with patch('dvidia_training.unitree_r1_kinematics.pose', return_value={'test': True}) as pose:
            code, result, _ = invoke('pose', 'unitree-r1-7a', '--side', 'left',
                                     '--joints', '0', '.1', '.2', '.3', '-.1', '-.2', '-.3')
        self.assertEqual((code, result), (0, {'test': True}))
        pose.assert_called_once_with('left', [0., .1, .2, .3, -.1, -.2, -.3])

    def test_unsuccessful_kinematics_benchmark_returns_receipt_and_failure_code(self):
        receipt = {'passed': False, 'side': 'right'}
        with patch('dvidia_training.unitree_r1_kinematics.benchmark', return_value=receipt):
            self.assertEqual(invoke('benchmark-kinematics', 'unitree-r1-7a', '--side', 'right'),
                             (1, receipt, ''))

    def test_bare_arm_refuses_skill_before_installation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('check', 'install'):
                args = [name, '--robot', 'unitree-r1-7a', '--model', str(root/'piper-policy.json')]
                if name == 'install':
                    args.extend(['--assets', str(root), '--output', str(root/'installed')])
                code, output, error = invoke(*args)
                self.assertEqual((code, output), (2, None))
                self.assertIn('PiPER arm and gripper', error)
            self.assertEqual(list(root.iterdir()), [])

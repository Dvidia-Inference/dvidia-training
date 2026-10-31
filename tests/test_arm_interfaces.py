"""Native actuator semantics and fail-closed command boundaries, without hardware."""
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from dvidia_training import arm_interfaces as arms


def resign(value):
    result = deepcopy(value)
    result.pop('contract_sha256', None)
    result['contract_sha256'] = sha256(json.dumps(result, sort_keys=True, separators=(',', ':'),
                                                 allow_nan=False).encode()).hexdigest()
    return result


class ArmInterfaceDescriptionTests(unittest.TestCase):
    def test_catalog_has_exact_native_semantics_and_no_hardware_claim(self):
        piper, so101, ur = arms.catalog()
        self.assertEqual([item['id'] for item in (piper, so101, ur)], ['piper', 'so101', 'ur5e-2f85'])
        self.assertEqual([item['arm_axes'] for item in (piper, so101, ur)], [6, 5, 6])
        self.assertEqual([len(item['action_names']) for item in (piper, so101, ur)], [7, 6, 7])
        self.assertEqual(piper['action_units'], ['rad'] * 6 + ['m'])
        self.assertEqual(so101['action_units'], ['rad'] * 6)
        self.assertEqual(ur['action_units'], ['rad'] * 6 + ['native_0_255'])
        for item in (piper, so101, ur):
            self.assertEqual(item['interface_id'], 'mujoco-position-v1')
            self.assertFalse(item['physical_robot_ready'])
            self.assertFalse(item['task_qualified'])

    def test_listing_and_description_work_without_optional_native_imports(self):
        script = """
import sys
sys.modules['mujoco'] = None
sys.modules['numpy'] = None
from dvidia_training.arm_interfaces import catalog, contract, validate_contract
assert len(catalog()) == 3
for item in catalog():
    value = contract(item['id'])
    assert validate_contract(value) == value
assert 'dvidia_training.env' not in sys.modules
"""
        completed = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_contract_locks_sources_selected_asset_bytes_and_licenses(self):
        for profile in arms.PROFILES:
            item = arms.contract(profile)
            self.assertIsNone(item['runtime_binding'])
            self.assertEqual(item['asset_lock']['commit'], '0059d4335f8156206f63a35662313385f7ad6d74')
            self.assertEqual(item['source_sha256']['arm_interfaces.py'], sha256(Path(arms.__file__).read_bytes()).hexdigest())
            lock = item['asset_lock']
            self.assertEqual(lock['lock_sha256'], sha256(Path(arms.__file__).with_name(lock['lock_file']).read_bytes()).hexdigest())
            paths = [row['path'] for row in lock['files']]
            self.assertEqual(len(paths), len(set(paths)))
            self.assertTrue(all(not path.startswith('shadow_hand/') for path in paths))
            for folder in lock['simulation_asset_licenses']:
                self.assertIn(folder + '/LICENSE', paths)
            self.assertEqual(item['initial_state']['hidden_settling_steps'], 0)
            self.assertTrue(all(not planned['implemented'] for planned in item['hardware_interfaces']))
            self.assertEqual(arms.validate_contract(item), item)

    def test_contracts_are_fresh_and_do_not_alias_cached_metadata(self):
        original = arms.contract('piper')
        altered = arms.contract('piper')
        altered['action_names'][0] = 'changed'
        altered['asset_lock']['files'][0]['sha256'] = '0' * 64
        altered['initial_state']['ctrl_native'][0] = .2
        self.assertEqual(arms.contract('piper'), original)
        summary = arms.catalog()
        summary[0]['action_units'][0] = 'mm'
        self.assertEqual(arms.catalog()[0]['action_units'][0], 'rad')

    def test_self_consistent_digest_cannot_widen_units_limits_or_source_locks(self):
        original = arms.contract('ur5e-2f85')
        for key, mutate in (
            ('units', lambda value: value['action_units'].__setitem__(-1, 'm')),
            ('limits', lambda value: value['action_limits_native'][-1].__setitem__(1, 256.)),
            ('source', lambda value: value['source_sha256'].__setitem__('arm_interfaces.py', '0' * 64)),
            ('bool', lambda value: value.__setitem__('physical_robot_ready', 0)),
            ('extra', lambda value: value.__setitem__('send_to_hardware', True)),
        ):
            with self.subTest(key=key):
                changed = deepcopy(original)
                mutate(changed)
                with self.assertRaises(ValueError):
                    arms.validate_contract(resign(changed))
        changed = deepcopy(original)
        changed['contract_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            arms.validate_contract(changed)

    def test_bad_profile_and_runtime_metadata_fail_before_native_build(self):
        for value in ('ur5e', '', None, True, ['piper']):
            with self.subTest(profile=value), self.assertRaises(ValueError):
                arms.contract(value)
        item = arms.contract('so101')
        for binding in ({}, {'mujoco_version': 'old'},
                        {'mujoco_version': '3.15.0', 'numpy_version': '2.5.3',
                         'timestep_s': True, 'compiled_model_sha256': '0' * 64}):
            changed = dict(item, runtime_binding=binding)
            with self.assertRaises(ValueError):
                arms.validate_contract(resign(changed))
        for dt in (True, .003, float('nan'), '0.001'):
            with self.subTest(timestep=dt), self.assertRaises(ValueError):
                arms.build('so101', 'unused', timestep=dt)

    def test_contract_describes_independent_coordinates_and_coupled_commands(self):
        piper = arms.contract('piper')
        so = arms.contract('so101')
        ur = arms.contract('ur5e-2f85')
        self.assertEqual(len(piper['joint_names']), 8)
        self.assertEqual(len(piper['action_names']), 7)
        self.assertEqual(piper['gripper_joint_names'], ['joint7', 'joint8'])
        self.assertIn('not total jaw gap', piper['gripper_command'])
        self.assertEqual(so['expected_dimensions']['neq'], 0)
        self.assertIn('not linear jaw gap', so['gripper_command'])
        self.assertEqual(ur['expected_dimensions']['neq'], 3)
        self.assertEqual(ur['actuator_transmissions'][-1], {'type': 'tendon', 'target': 'gripper/split'})
        self.assertEqual(ur['action_limits_native'][-1], [0., 255.])
        self.assertIn('not metres', ur['gripper_command'])
        self.assertIn('not a calibrated', ur['tool_frame_scope'])


@unittest.skipUnless(os.environ.get('DVIDIA_ROBOT_PROFILE_ASSETS') and os.environ.get('DVIDIA_REFERENCE_ASSETS'),
                     'Set DVIDIA_ROBOT_PROFILE_ASSETS and DVIDIA_REFERENCE_ASSETS to pinned native assets')
class NativeArmInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mj, cls.np = arms.sim._native_imports()
        cls.built = {}
        for profile in arms.PROFILES:
            assets = os.environ['DVIDIA_REFERENCE_ASSETS' if profile == 'ur5e-2f85' else 'DVIDIA_ROBOT_PROFILE_ASSETS']
            cls.built[profile] = arms.build(profile, assets)

    def test_all_native_initializations_are_explicit_unstepped_and_bound(self):
        for profile, (model, data, item) in self.built.items():
            with self.subTest(profile=profile):
                self.assertEqual(data.time, 0.)
                self.assertEqual(data.qpos.tolist(), item['initial_state']['qpos_native'])
                self.assertEqual(data.qvel.tolist(), item['initial_state']['qvel_native_per_second'])
                self.assertEqual(data.ctrl.tolist(), item['initial_state']['ctrl_native'])
                self.assertEqual(item['runtime_binding']['mujoco_version'], '3.15.0')
                self.assertEqual(item['runtime_binding']['timestep_s'], .001)
                self.assertEqual(item['runtime_binding']['compiled_model_sha256'], arms.sim.model_digest(model))
                self.assertEqual(arms.validate_contract(item), item)
                self.assertEqual(arms.validate_command(item, model, data.ctrl), data.ctrl.tolist())
                self.assertEqual(int(model.nq), len(item['joint_names']))
                self.assertEqual(int(model.nu), len(item['action_names']))

    def test_joint_and_tendon_native_transmissions_keep_their_units(self):
        for profile, (model, data, item) in self.built.items():
            self.assertEqual(model.jnt_qposadr.tolist(), item['joint_qpos_indices'])
            target = data.ctrl.tolist()
            target[-1] = {'piper': .035, 'so101': 1., 'ur5e-2f85': 255.}[profile]
            self.assertEqual(arms.validate_command(item, model, target), target)
            self.assertEqual(data.time, 0.)
            self.assertEqual(data.ctrl.tolist(), item['initial_state']['ctrl_native'])
        ur_model = self.built['ur5e-2f85'][0]
        self.assertEqual(ur_model.actuator_trntype[-1], self.mj.mjtTrn.mjTRN_TENDON)
        self.assertAlmostEqual(ur_model.actuator_gainprm[-1, 0], .3137255)
        self.assertEqual(ur_model.actuator_biasprm[-1, 1], -100.)

    def test_invalid_vectors_reject_without_control_or_state_mutation(self):
        model, data, item = self.built['piper']
        before = (data.qpos.copy(), data.ctrl.copy(), data.time)
        valid = data.ctrl.tolist()
        invalid = [[], valid + [0.], [valid], '0,1,2,3,4,5,6', {name: 0. for name in item['action_names']},
                   self.np.asarray([valid]), self.np.asarray(valid, dtype=complex)]
        for scalar in (True, self.np.bool_(False), float('nan'), float('inf'), '0', None, 10**1000, 1j):
            invalid.append([scalar] + valid[1:])
        invalid.extend([valid[:-1] + [-.00001], valid[:-1] + [.03500001]])
        for values in invalid:
            with self.subTest(values=repr(values)[:100]), self.assertRaises(ValueError):
                arms.validate_command(item, model, values)
        self.assertTrue(self.np.array_equal(data.qpos, before[0]))
        self.assertTrue(self.np.array_equal(data.ctrl, before[1]))
        self.assertEqual(data.time, before[2])

    def test_each_gripper_enforces_its_own_native_command_range(self):
        for profile, (model, data, item) in self.built.items():
            command = data.ctrl.tolist()
            for value in (item['action_limits_native'][-1][0] - .001,
                          item['action_limits_native'][-1][1] + .001):
                command[-1] = value
                with self.subTest(profile=profile, value=value), self.assertRaises(ValueError):
                    arms.validate_command(item, model, command)

    def test_named_vectors_are_exact_and_reordered_without_aliasing(self):
        model, data, item = self.built['so101']
        mapping = dict(reversed(list(zip(item['action_names'], [.1, .2, .3, .4, .5, .6]))))
        result = arms.named_command(item, model, mapping)
        self.assertEqual(result, [.1, .2, .3, .4, .5, .6])
        result[0] = 0.
        self.assertEqual(mapping['shoulder_pan'], .1)
        self.assertEqual(data.ctrl.tolist(), [0.] * 6)
        for invalid in ({**mapping, 'extra': 0.}, {key: value for key, value in mapping.items() if key != 'gripper'},
                        {**mapping, 'gripper': True}):
            with self.assertRaises(ValueError):
                arms.named_command(item, model, invalid)

    def test_static_cross_profile_foreign_model_and_tampered_contract_reject(self):
        model, data, item = self.built['piper']
        with self.assertRaises(ValueError):
            arms.validate_command(arms.contract('piper'), model, data.ctrl)
        with self.assertRaises(ValueError):
            arms.validate_command(self.built['ur5e-2f85'][2], model, data.ctrl)
        # Even an identical copied model was not bound by this interface build.
        with self.assertRaises(ValueError):
            arms.validate_command(item, deepcopy(model), data.ctrl)
        changed = deepcopy(item)
        changed['action_limits_native'][0][1] = 100.
        with self.assertRaises(ValueError):
            arms.validate_command(resign(changed), model, data.ctrl)

    def test_changed_native_limits_servo_coupling_and_timestep_fail_closed(self):
        model, data, item = self.built['piper']
        for array, index, value in ((model.actuator_ctrlrange, (0, 1), 100.),
                                    (model.actuator_gainprm, (0, 0), 81.),
                                    (model.eq_data, (0, 1), 1.)):
            saved = float(array[index])
            try:
                array[index] = value
                with self.assertRaises(ValueError):
                    arms.validate_command(item, model, data.ctrl)
            finally:
                array[index] = saved
        old_dt = model.opt.timestep
        try:
            model.opt.timestep = .002
            with self.assertRaises(ValueError):
                arms.validate_command(item, model, data.ctrl)
        finally:
            model.opt.timestep = old_dt
        self.assertEqual(arms.validate_command(item, model, data.ctrl), data.ctrl.tolist())

    def test_build_keeps_the_requested_supported_native_step(self):
        for dt in (.0005, .002):
            model, data, item = arms.build('so101', os.environ['DVIDIA_ROBOT_PROFILE_ASSETS'], timestep=dt)
            self.assertEqual(model.opt.timestep, dt)
            self.assertEqual(item['runtime_binding']['timestep_s'], dt)
            self.assertEqual(data.time, 0.)


if __name__ == '__main__':
    unittest.main()

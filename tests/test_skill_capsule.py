"""Portable data integrity and local qualification consistency, not hardware tests."""
from __future__ import annotations

import copy
from hashlib import sha256
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from dvidia_training.skill_capsule import (MAX_CAPSULE_BYTES, SkillCapsuleError, arm_compatibility,
    build_capsule, canonical_sha256, capsule_summary, check_qualification, encode_capsule,
    inspect_capsule, normalize_scene, payload_sha256, qualification_receipt, scene_sha256,
    source_bytes, provenance_for_model)
from dvidia_training.skillspace import load_skillspace


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = {'kind': 'synthetic-test-runtime', 'files_sha256': {'arm_distill.py': 'a'*64},
           'dependencies': {'mujoco': 'synthetic', 'numpy': 'synthetic'}}
SCENE = {'object_position': [.42, -.04, .31], 'target_position': [.54, .10, .31], 'seed': 0,
         'object_size': [.04, .04, .04], 'object_mass': .04, 'object_friction': .8,
         'jaw_force_limit': 15., 'joint_torque_limit': 40.}


def model():
    return {'schema_version': 1, 'kind': 'dvidia.numpy-rbf-movement-head', 'scope': 'simulation-only',
            'status': 'candidate', 'physical_robot_ready': False,
            'arm_profile': 'dvidia-authored-6dof-parallel-jaw-v0',
            'feature_contract': 'q6_object_minus_tcp3_target_minus_tcp3__capped_goal_delta3_upright_orientation_delta3_v1',
            'centers': [[0.] * 12 for _ in range(8)], 'context_mean': [0.] * 12,
            'context_scale': [1.] * 12, 'kernel_width': 1.,
            'weights': [[[0.] * 6 for _ in range(6)] for _ in range(8)],
            'dataset_sha256': 'b'*64, 'train_cases': ['train-0'],
            'training': {'algorithm': 'normalized-rbf-ridge', 'sample_count': 32,
                         'regularization': .001, 'seed': 0}}


def grounding():
    return {'schema_version': 1, 'task_type': 'place_cup', 'provenance': 'authored_simulation_adapter',
            'adapter_id': 'contact-pick-place-box-v0', 'arm_profile': 'dvidia-authored-6dof-parallel-jaw-v0',
            'frame': 'world', 'units': 'm', 'language_aliases': ['Set a cup on the mark.'],
            'workspace': {'min': [.25, -.2, .29], 'max': [.60, .2, .55]}, 'table_height': .29,
            'object': {'kind': 'box', 'position': SCENE['object_position'], 'size': SCENE['object_size'],
                       'mass': .04, 'friction': .8},
            'target': {'position': SCENE['target_position'], 'position_tolerance': .025, 'dwell_seconds': .25},
            'procedure': ['approach', 'descend', 'close', 'lift', 'transfer', 'place', 'release', 'retreat', 'verify']}


def capsule():
    return build_capsule((ROOT/'src/dvidia_training/place_cup.skill.json').read_bytes(), grounding(), model(), runtime=RUNTIME)


def rebind(value):
    value['payload_sha256'] = payload_sha256(value)
    return value


def evaluation(value):
    protocol = {'id': 'protocol-0', 'scope': 'simulation-only', 'scene_selection': 'synthetic declaration',
                'success_gate': 'unchanged native placement gate', 'split_kind': 'scene IDs',
                'controls': ['replay_open_jaw']}
    trial = {'id': 'eval-0', 'scene': normalize_scene(SCENE), 'scene_sha256': scene_sha256(SCENE),
             'success': True, 'valid': True, 'reason': 'success'}
    return {'id': 'evaluation-0', 'protocol': protocol, 'protocol_sha256': canonical_sha256(protocol),
            'movement_head_sha256': canonical_sha256(value['movement_head']),
            'training_dataset_sha256': value['movement_head']['dataset_sha256'],
            'runtime_sha256': canonical_sha256(RUNTIME), 'trials': [trial],
            'baseline_trials': [copy.deepcopy(trial)]}


def native_fixture(value, *, control=False):
    """Synthetic JSON consistency fixture. It was never run in native physics."""
    skill = load_skillspace(source_bytes(value), value['grounding'])
    config = {'timestep': .001, 'control_dt': .02, 'horizon': 18., 'settle_time': .4,
              'object_position': SCENE['object_position'], 'target_position': SCENE['target_position'],
              'object_half_size': [.02]*3, 'object_mass': .04, 'object_friction': .8,
              'table_height': .29, 'position_tolerance': .025, 'dwell_seconds': .25, 'scene_jitter': 0.,
              'joint_torque_limit': 40., 'jaw_force_limit': 15., 'joint_target_rate_limit': 3.,
              'jaw_width_rate_limit': .16, 'gravity': 9.81}
    steps = 900 if control else 25
    seconds = 18. if control else .5
    reason = 'horizon' if control else 'success'
    info = {'success': not control, 'valid': True, 'warnings': [], 'task_id': 'ArmPickPlace-v0',
            'arm_profile': 'dvidia-authored-6dof-parallel-jaw-v0', 'adapter_id': 'contact-pick-place-box-v0',
            'qualification': 'simulation-only', 'observations': 'privileged simulator state',
            'reason': reason, 'simulation_time': seconds, 'grasp_seen': not control, 'lift_seen': not control,
            'max_object_lift': .13 if not control else 0., 'lift_dwell_elapsed': 0.,
            'dwell_elapsed': .25 if not control else 0., 'target_distance': 0. if not control else .15,
            'table_contacts': 4}
    actions = [{'stage': 'verify', 'action': {'joint_targets': [0.]*6, 'gripper_width': .08}}
               for _ in range(steps)]
    terminal = {'object_position': list(SCENE['target_position']),
                'end_effector_position': [.54, .10, .44], 'object_velocity': [0.]*3,
                'object_angular_velocity': [0.]*3, 'grasp_contacts': {'left': 0, 'right': 0},
                'gripper_width': .08}
    episode = {'seed': 0, 'policy': 'replay_open_jaw' if control else 'distilled_placement',
               'success': not control, 'reason': reason, 'simulated_seconds': seconds,
               'control_steps': steps, 'final_info': info, 'actions': actions, 'trace': [terminal],
               'controller_diagnostics': {'movement_head_sha256': canonical_sha256(value['movement_head']),
                    'teacher_fallback_calls': 0, 'student_inference_calls': steps}}
    return {'schema_version': 1, 'task': 'ArmPickPlace-v0', 'scope': 'simulation-only',
            'runtime': copy.deepcopy(RUNTIME), 'skill': skill.to_dict(), 'environment_config': config,
            'episodes': [episode]}


class SkillCapsuleTests(unittest.TestCase):
    def test_export_roundtrip_exact_source_and_candidate(self):
        value = capsule()
        with patch('socket.create_connection', side_effect=AssertionError('Network was attempted')):
            result = inspect_capsule(encode_capsule(value), expected_runtime=RUNTIME)
        self.assertEqual(source_bytes(result), (ROOT/'src/dvidia_training/place_cup.skill.json').read_bytes())
        self.assertEqual(result['payload_sha256'], payload_sha256(result))
        self.assertEqual(capsule_summary(result)['status'], 'candidate')
        self.assertFalse(result['physical_robot_ready'])

    def test_source_and_payload_tampering_rejected(self):
        for change in (lambda c: c['source'].update(text=c['source']['text']+' '),
                       lambda c: c['movement_head']['weights'][0][0].__setitem__(0, .1)):
            value = capsule(); change(value)
            with self.assertRaises(SkillCapsuleError):
                inspect_capsule(value)
        value = capsule(); value['source']['text'] += ' '; rebind(value)
        with self.assertRaises(SkillCapsuleError):
            inspect_capsule(value)

    def test_exact_source_object_and_128_kib_bound_required(self):
        for raw in (b'', b'null', b'[]', b'{}'+b' '*(128*1024), b'\xff'):
            with self.subTest(raw=raw[:20]), self.assertRaises(SkillCapsuleError):
                build_capsule(raw, grounding(), model(), runtime=RUNTIME)
        value = capsule(); value['source']['text'] += ' '*(128*1024)
        value['source']['sha256'] = sha256(value['source']['text'].encode()).hexdigest(); rebind(value)
        with self.assertRaises(SkillCapsuleError):
            inspect_capsule(value)

    def test_duplicate_nonfinite_overflow_unknown_and_size_rejected(self):
        for raw in (b'{"format":1,"format":2}', b'{"value":NaN}', b'{"value":1e999}',
                    b'['*60+b'0'+b']'*60, b'{}'+b' '*MAX_CAPSULE_BYTES, b'\x80'):
            with self.subTest(raw=raw[:25]), self.assertRaises(SkillCapsuleError):
                inspect_capsule(raw)
        value = capsule(); value['executable'] = 'import os'; rebind(value)
        with self.assertRaises(SkillCapsuleError):
            inspect_capsule(value)
        value = capsule(); value['source']['extra'] = 'unknown'; rebind(value)
        with self.assertRaises(SkillCapsuleError):
            inspect_capsule(value)

    def test_profile_units_order_scope_and_runtime_are_exact(self):
        changes = [lambda c: c['compatibility']['joint_order'].reverse(),
                   lambda c: c['compatibility']['actions'][0].update(units='degree'),
                   lambda c: c['compatibility']['gripper'].update(maximum_jaw_actuator_force_n=100),
                   lambda c: c['compatibility'].update(supervisor_revision='different'),
                   lambda c: c.update(status='validated_simulation_scene'),
                   lambda c: c.update(physical_robot_ready=True)]
        for change in changes:
            value = capsule(); change(value); rebind(value)
            with self.assertRaises(SkillCapsuleError):
                inspect_capsule(value)
        runtime = copy.deepcopy(RUNTIME); runtime['dependencies']['mujoco'] = 'different'
        with self.assertRaises(SkillCapsuleError):
            inspect_capsule(capsule(), expected_runtime=runtime)
        self.assertEqual(arm_compatibility()['independent_commands'], 7)

    def test_model_unknown_boolean_bad_shapes_weights_and_training_rejected(self):
        changes = [lambda m: m.update(extra='unknown'), lambda m: m['weights'][0][0].__setitem__(0, True),
                   lambda m: m.update(weights=[]), lambda m: m.update(context_scale=[0.]*12),
                   lambda m: m.update(centers=[[0.]*12]*49),
                   lambda m: m['training'].update(sample_count=True),
                   lambda m: m.update(feature_contract='pixel_input_without_camera')]
        for change in changes:
            value = capsule(); change(value['movement_head']); rebind(value)
            with self.assertRaises(SkillCapsuleError):
                inspect_capsule(value)

    def test_finite_float32_overflow_traps_rejected_after_complete_rebinding(self):
        for name in ('centers', 'context_mean', 'context_scale', 'weights'):
            value = capsule()
            array = value['movement_head'][name]
            while type(array[0]) is list:
                array = array[0]
            array[0] = 1e30
            value['provenance'] = provenance_for_model(value['movement_head']); rebind(value)
            with self.subTest(name=name), self.assertRaisesRegex(SkillCapsuleError, 'numerical envelope'):
                inspect_capsule(value)
        value = capsule(); value['movement_head']['weights'][0][0][0] = 1e6
        value['provenance'] = provenance_for_model(value['movement_head']); rebind(value)
        self.assertEqual(inspect_capsule(value)['movement_head']['weights'][0][0][0], 1e6)

    def test_provenance_matches_model_and_evaluation_bindings(self):
        value = capsule(); value['provenance']['evaluation'] = evaluation(value); rebind(value)
        self.assertTrue(capsule_summary(inspect_capsule(value))['has_declared_evaluation'])
        self.assertEqual(inspect_capsule(value)['status'], 'candidate')
        for change in (lambda c: c['provenance']['dataset'].update(sha256='c'*64),
                       lambda c: c['provenance']['training'].update(movement_head_sha256='c'*64),
                       lambda c: c['provenance']['evaluation'].update(runtime_sha256='c'*64),
                       lambda c: c['provenance']['evaluation']['trials'][0].update(id='train-0'),
                       lambda c: c['provenance']['evaluation']['baseline_trials'][0].update(id='other'),
                       lambda c: c['provenance']['evaluation']['trials'][0]['scene'].update(object_mass=.08)):
            altered = copy.deepcopy(value); change(altered); rebind(altered)
            with self.assertRaises(SkillCapsuleError):
                inspect_capsule(altered)

    def test_scene_normalization_rejects_motor_expansion_and_mixed_units(self):
        self.assertEqual(scene_sha256(SCENE), canonical_sha256(normalize_scene(SCENE)))
        for change in (lambda s: s.update(object_size=[40, 40, 40]), lambda s: s.update(seed=True),
                       lambda s: s.update(jaw_force_limit=100), lambda s: s.update(object_mass='0.04'),
                       lambda s: s.update(object_position=[.60, .20, .31]),
                       lambda s: s.update(target_position=s['object_position'])):
            scene = copy.deepcopy(SCENE); change(scene)
            with self.assertRaises(SkillCapsuleError):
                normalize_scene(scene)

    def test_local_receipt_is_exact_scene_with_terminal_lift_counter_zero(self):
        value = capsule()
        receipt = qualification_receipt(value, SCENE, native_fixture(value), runtime=RUNTIME,
                                        controls=native_fixture(value, control=True))
        self.assertEqual(receipt['status'], 'validated_simulation_scene')
        self.assertFalse(receipt['physical_robot_ready'])
        self.assertEqual(check_qualification(receipt, value, SCENE, runtime=RUNTIME), receipt)
        # Importing the data again never adopts a qualification from elsewhere.
        self.assertEqual(inspect_capsule(encode_capsule(value))['status'], 'candidate')
        changed = copy.deepcopy(SCENE); changed['object_mass'] = .08
        with self.assertRaises(SkillCapsuleError):
            check_qualification(receipt, value, changed, runtime=RUNTIME)

    def test_native_task_configuration_success_gate_and_student_binding_rejected(self):
        value = capsule()
        changes = [lambda r: r['environment_config'].update(object_mass=.08),
                   lambda r: r['environment_config'].update(position_tolerance=.1),
                   lambda r: r['environment_config'].update(horizon=100),
                   lambda r: r['episodes'][0].update(seed=1),
                   lambda r: r['episodes'][0]['final_info'].update(lift_seen=False),
                   lambda r: r['episodes'][0]['controller_diagnostics'].update(teacher_fallback_calls=1),
                   lambda r: r['episodes'][0]['controller_diagnostics'].update(movement_head_sha256='c'*64),
                   lambda r: r['episodes'][0]['trace'][-1].update(gripper_width=.04),
                   lambda r: r['episodes'][0]['trace'][-1].update(end_effector_position=[.54, .1, .32])]
        for change in changes:
            result = native_fixture(value); change(result)
            with self.assertRaises(SkillCapsuleError):
                qualification_receipt(value, SCENE, result, runtime=RUNTIME, controls=native_fixture(value, control=True))

    def test_negative_control_is_same_recorded_arm_and_must_fail(self):
        value = capsule()
        for change in (lambda r: r['episodes'][0]['actions'][0]['action'].update(gripper_width=.04),
                       lambda r: r['episodes'][0]['actions'][0]['action'].update(joint_targets=[.1]*6),
                       lambda r: r['episodes'][0].update(policy='idle'),
                       lambda r: r['episodes'][0]['final_info'].update(warnings=['solver']),
                       lambda r: r['environment_config'].update(object_size=[.05]*3)):
            controls = native_fixture(value, control=True); change(controls)
            with self.assertRaises(SkillCapsuleError):
                qualification_receipt(value, SCENE, native_fixture(value), runtime=RUNTIME, controls=controls)

    def test_readiness_and_stale_qualification_digest_rejected(self):
        value = capsule()
        receipt = qualification_receipt(value, SCENE, native_fixture(value), runtime=RUNTIME,
                                        controls=native_fixture(value, control=True))
        for change in (lambda r: r.update(physical_robot_ready=True),
                       lambda r: r['student'].update(target_distance_m=.1),
                       lambda r: r.update(runtime_sha256='c'*64)):
            altered = copy.deepcopy(receipt); change(altered)
            with self.assertRaises(SkillCapsuleError):
                check_qualification(altered, value, SCENE, runtime=RUNTIME)

    def test_inspection_does_not_retain_mutable_source_dictionary_aliases(self):
        value = capsule(); checked = inspect_capsule(value)
        value['movement_head']['weights'][0][0][0] = .5
        self.assertEqual(checked['movement_head']['weights'][0][0][0], 0.)


if __name__ == '__main__':
    unittest.main()

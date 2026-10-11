"""PiPER protocol, negative-control and evidence boundaries without native stepping."""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dvidia_training import piper_task_benchmark as bench


LICENSE = b'MIT License\nCopyright upstream fixture\n'
BLOBS = {'agilex_piper/LICENSE': LICENSE, 'agilex_piper/README.md': b'Upstream model fixture\n'}


def protocol():
    value = json.loads(json.dumps(bench.make_protocol(seed=20, nominal=1, stress=0)))
    value['scenes'][0]['config']['timeout_s'] = .1
    return value


def provenance(plan):
    return deepcopy(bench._pinned_assets()[0])


def episode(plan, control, valid=True, tape=None):
    scene, initial = plan['scenes'][0], [0., 1.57, -1.3485, 0., 0., 0.]
    count = round(scene['config']['timeout_s']/scene['config']['control_dt'])
    tape = deepcopy(tape) if tape is not None else [initial.copy() for _ in range(count)]
    actions = [{'t_seconds': index*.02, 'joint_targets_rad': arm,
                'gripper_position_m': .035, 'phase': control} for index, arm in enumerate(tape)] if valid else []
    states = []
    for index in range(len(actions)+1 if valid else 0):
        pos, quat = list(scene['config']['object_position_m']), [1., 0., 0., 0.]
        states.append({'t_seconds': index*.02, 'qpos': initial+[.035, -.035]+pos+quat, 'qvel': [0.]*14,
                       'joint_position_rad': initial.copy(), 'joint_velocity_rad_s': [0.]*6,
                       'gripper_joint_position_m': [.035, -.035], 'tcp_position_m': [.3, 0., .3],
                       'tcp_quaternion_wxyz': quat.copy(), 'object_position_m': pos,
                       'object_quaternion_wxyz': quat.copy(), 'object_linear_velocity_m_s': [0.]*3,
                       'object_angular_velocity_rad_s': [0.]*3, 'target_position_m': list(scene['config']['target_position_m']),
                       'native_actuator_controls': initial+[.035], 'actuator_force_native': [0.]*7,
                       'grasp_contact_counts': {'left':0,'right':0}, 'pad_normal_forces_n': {'left':0.,'right':0.},
                       'table_object_contact_count':1, 'unexpected_contact_count':0, 'contact_count':1,
                       'active_contact_pairs':[{'geom_ids':[0,1],'normal_force_n':.392}],
                       'issued_joint_targets_rad': actions[index]['joint_targets_rad'] if index < len(actions) else None,
                       'issued_gripper_position_m': .035 if index < len(actions) else None,
                       'command': actions[index] if index < len(actions) else None,
                       'observation_origin': 'privileged_native_simulator_state'})
    return {'schema': bench.task.SCHEMA, 'valid': valid, 'success': False, 'warnings': [], 'reason': 'timeout' if valid else 'exception',
            'contract': plan['contract'], 'implementation_sha256': {name: plan['implementation_sha256'][name] for name in bench.EPISODE_SOURCE_FILES},
            'trajectory': states, 'actions': actions, 'arm_tape': tape if valid else [], 'simulated_seconds': .1 if valid else 0.,
            'asset_provenance': provenance(plan), 'metrics': {'valid': valid, 'success': False, 'physical_robot_ready': False},
            'compiled_model_sha256':'a'*64,
            'native_control_limits':[[-2.618,2.618],[0.,3.14],[-2.697,0.],[-1.832,1.832],[-1.22,1.22],[-3.14,3.14],[0.,.035]],
            'native_model_settings':{'dimensions':{'nq':15,'nv':14,'nu':7,'neq':1,'nmocap':0},
                                     'gravity_m_s2':[0.,0.,-9.81], 'body_gravcomp':[0.]+[1.]*9+[0.,0.]},
            'platform': {'system':'fixture','machine':'fixture','python':'fixture','mujoco_version':'fixture','numpy_version':'fixture'},
            'timing':{'setup_seconds':.1,'episode_wall_seconds':.5, 'control_boundaries':count,
                      'control_step_seconds':{'count':count,'median':.001,'p95':.002,'max':.003},
                      'missed_control_boundaries':0,'simulated_seconds_per_wall_second':.2}}


def run_fixture(root, *, invalid=False, incomplete=False):
    plan = protocol()
    seen = []

    def run(_assets, config, **kwargs):
        seen.append(kwargs)
        # The protocol has to exist before any episode, including failed ones.
        assert json.loads((root/'protocol.json').read_bytes()) == plan
        value = episode(plan, kwargs['control'], valid=not invalid, tape=kwargs.get('arm_tape'))
        if incomplete and kwargs['control'] == 'placement':
            value['valid'], value['metrics']['valid'] = False, False
            value['arm_tape'].pop()
            value['actions'].pop()
            value['trajectory'] = value['trajectory'][:len(value['actions'])+1]
            value['trajectory'][-1].update(command=None, issued_joint_targets_rad=None, issued_gripper_position_m=None)
            value['simulated_seconds'] = value['trajectory'][-1]['t_seconds']
        return value

    with patch.object(bench, 'make_protocol', return_value=plan), \
            patch.object(bench.robot_profiles, 'verify_assets', return_value=provenance(plan)), \
            patch.object(bench.robot_profiles, '_payload', return_value=({}, BLOBS)), \
            patch.object(bench.task, 'run_episode', side_effect=run):
        report = bench.run_benchmark('unused', root, nominal=1, stress=0)
    return report, seen


def reseal(root):
    names = ['protocol.json', 'report.json', 'report.html', bench.LICENSE_PATH, bench.MODEL_NOTE_PATH]
    names.extend(path.relative_to(root).as_posix() for path in sorted((root/'attempts').glob('*.json')))
    (root/'integrity.json').write_bytes(bench._json_bytes({'schema': bench.INTEGRITY_SCHEMA, 'files': {
        name: {'bytes': (root/name).stat().st_size, 'sha256': sha256((root/name).read_bytes()).hexdigest()} for name in names}}))


def mutate_attempt(root, report, index, mutate):
    row = report['attempts'][index]
    raw = json.loads((root/row['result_file']).read_bytes())
    mutate(raw)
    (root/row['result_file']).write_bytes(bench._json_bytes(raw))
    report['attempts'][index] = {**bench._summary(raw), 'result_file': row['result_file']}
    report['summary'] = bench.aggregate(report['attempts'], report['scenes'])
    report['valid_attempts'] = sum(row['valid'] for row in report['attempts'])
    (root/'report.json').write_bytes(bench._json_bytes(report))
    reseal(root)


class FixtureTests(unittest.TestCase):
    def setUp(self):
        # Evidence-boundary fixtures need no provisioned native model assets.
        # Canonical asset identity remains real; only these two notice hashes use
        # tiny authored fixture bytes. Native integration covers original bytes.
        expected, _ = bench._pinned_assets()
        notices = {name:sha256(value).hexdigest() for name,value in BLOBS.items()}
        patcher = patch.object(bench, '_pinned_assets', return_value=(expected,notices))
        patcher.start()
        self.addCleanup(patcher.stop)


class ProtocolTests(FixtureTests):
    def test_seeded_plan_freezes_units_defaults_and_all_controls(self):
        plan = bench.make_protocol()
        self.assertEqual(plan, bench.make_protocol())
        self.assertEqual(plan['planned_attempts'], 9)
        self.assertEqual(plan['native_command_units'], ['rad']*6+['m'])
        self.assertEqual([scene['config']['friction'] for scene in plan['scenes']], [.8, .8, .15])
        base = asdict(bench.task.PiperTaskConfig())
        for scene in plan['scenes']:
            for key in ('object_position_m', 'target_position_m'):
                self.assertTrue(all(abs(scene['config'][key][axis]-base[key][axis]) <= .005 for axis in (0, 1)))
                self.assertEqual(scene['config'][key][2], base[key][2])
        self.assertNotEqual(plan['scenes'], bench.make_protocol(seed=611048)['scenes'])
        for options in ({'seed': True}, {'seed':999988}, {'nominal':0}, {'stress':7}, {'cohort':'test'}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                bench.make_protocol(**options)

    def test_complete_controls_and_exact_original_license_are_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'run'
            report, seen = run_fixture(root)
            self.assertEqual(seen[1]['arm_tape'], json.loads((root/report['attempts'][0]['result_file']).read_bytes())['arm_tape'])
            self.assertEqual((root/bench.LICENSE_PATH).read_bytes(), LICENSE)
            checked = bench.inspect_task_bundle(root)
            self.assertEqual((checked['attempted'], checked['valid_attempts']), (3, 3))
            self.assertFalse(checked['physical_robot_ready'])
            self.assertNotIn('trajectory', report['attempts'][0])
            with self.assertRaises(FileExistsError):
                run_fixture(root)

    def test_invalid_or_incomplete_baseline_leaves_dependent_control_invalid(self):
        for options in ({'invalid':True}, {'incomplete':True}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)/'run'
                report, seen = run_fixture(root, **options)
                self.assertEqual([row['control'] for row in seen], ['placement', 'hold'])
                self.assertEqual(report['attempted'], 3)
                self.assertFalse(report['attempts'][1]['valid'])
                self.assertIn('unavailable', report['attempts'][1]['error']['message'])
                self.assertEqual(report['summary'][1]['attempted'], 1)


class InspectionTests(FixtureTests):
    def test_partial_invalid_terminal_state_is_retained_without_a_fake_complete_tick(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'run'
            report, _ = run_fixture(root)
            def partial(raw):
                raw['valid'] = raw['metrics']['valid'] = False
                raw['reason'], raw['warnings'], raw['simulated_seconds'] = 'solver_warning', ['fixture warning'], .033
                raw['actions'], raw['arm_tape'], raw['trajectory'] = raw['actions'][:2], raw['arm_tape'][:2], raw['trajectory'][:3]
                raw['trajectory'][-1].update(t_seconds=.033, command=None, issued_joint_targets_rad=None, issued_gripper_position_m=None)
            mutate_attempt(root, report, 2, partial)
            self.assertEqual(bench.inspect_task_bundle(root)['valid_attempts'], 2)
            self.assertEqual(bench.inspect_task_bundle(root)['attempted'], 3)
            mutate_attempt(root, report, 2, lambda raw: raw['trajectory'][1].update(t_seconds=.013))
            with self.assertRaisesRegex(ValueError, 'cadence'):
                bench.load_task_bundle(root)

    def test_full_valid_terminal_state_cannot_issue_an_unexecuted_command(self):
        for field, value in (('issued_joint_targets_rad',[0.]*6), ('issued_gripper_position_m',.035), ('command',{})):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)/'run'
                report, _ = run_fixture(root)
                mutate_attempt(root, report, 0, lambda raw: raw['trajectory'][-1].update({field:value}))
                with self.assertRaisesRegex(ValueError, 'terminal'):
                    bench.load_task_bundle(root)

    def test_resealed_model_timing_contact_and_asset_declarations_are_checked(self):
        mutations = [lambda raw:raw.update(compiled_model_sha256='broken'),
                     lambda raw:raw.update(native_control_limits=[]),
                     lambda raw:raw['native_model_settings']['dimensions'].update(nq=1),
                     lambda raw:raw['timing'].update(control_boundaries=2),
                     lambda raw:raw['asset_provenance'].update(commit='b'*40),
                     lambda raw:raw['trajectory'][0].update(contact_count='broken'),
                     lambda raw:raw['trajectory'][0].update(pad_normal_forces_n=[]),
                     lambda raw:raw['trajectory'][0].update(active_contact_pairs=[{'geom_ids':[0,True],'normal_force_n':1}])]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)/'run'
                report, _ = run_fixture(root)
                mutate_attempt(root,report,1,mutate)
                with self.assertRaises(ValueError):
                    bench.load_task_bundle(root)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'run'
            report,_ = run_fixture(root)
            mutate_attempt(root,report,1,lambda raw:raw.update(compiled_model_sha256='b'*64))
            with self.assertRaisesRegex(ValueError,'different native model'):
                bench.load_task_bundle(root)

    def test_resealed_success_requires_actual_endpoint_and_invalid_duration_is_finite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'run'
            report,_ = run_fixture(root)
            def fake_success(raw):
                raw['success'] = True
                raw['metrics'].update(success=True,lift_seen=True,grasp_seen=True,carry_complete=True,
                                      retention_lost=False,unexpected_contact_native_ticks=0,dwell_seconds=.25)
            mutate_attempt(root,report,0,fake_success)
            with self.assertRaisesRegex(ValueError,'release endpoint'):
                bench.load_task_bundle(root)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'run'
            report,_ = run_fixture(root,invalid=True)
            mutate_attempt(root,report,0,lambda raw:raw.update(simulated_seconds=-10.))
            with self.assertRaisesRegex(ValueError,'simulation duration'):
                bench.load_task_bundle(root)

    def test_resealed_control_cheats_are_rejected_independently_of_summary(self):
        def arm_cheat(raw):
            raw['arm_tape'][0][0] += .1
            raw['actions'][0]['joint_targets_rad'][0] += .1
            raw['trajectory'][0]['issued_joint_targets_rad'][0] += .1
            raw['trajectory'][0]['command']['joint_targets_rad'][0] += .1
        def jaw_cheat(raw):
            raw['actions'][0]['gripper_position_m'] = 0.
            raw['trajectory'][0]['issued_gripper_position_m'] = 0.
            raw['trajectory'][0]['command']['gripper_position_m'] = 0.
        for index, mutate in ((1, arm_cheat), (2, arm_cheat), (1, jaw_cheat)):
            with self.subTest(index=index, mutate=mutate.__name__), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)/'run'
                report, _ = run_fixture(root)
                mutate_attempt(root, report, index, mutate)
                with self.assertRaises(ValueError):
                    bench.load_task_bundle(root)

    def test_resealed_scene_scope_source_and_native_state_mismatch_rejected(self):
        for field, value in (('profile','wrong'), ('seed',True), ('config',{}), ('physical_robot_ready',True),
                             ('contract',{}), ('implementation_sha256',{})):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)/'run'
                report, _ = run_fixture(root)
                mutate_attempt(root, report, 0, lambda raw: raw.update({field:value}))
                with self.assertRaises(ValueError):
                    bench.load_task_bundle(root)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'run'
            report, _ = run_fixture(root)
            mutate_attempt(root, report, 0, lambda raw: raw['trajectory'][0]['gripper_joint_position_m'].append(0.))
            with self.assertRaisesRegex(ValueError, 'dimensions'):
                bench.load_task_bundle(root)

    def test_dropped_attempt_and_resealed_false_success_cannot_shrink_denominator(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'run'
            report, _ = run_fixture(root)
            report['attempts'].pop()
            report['attempted'] = 2
            (root/'report.json').write_bytes(bench._json_bytes(report))
            reseal(root)
            with self.assertRaisesRegex(ValueError,'planned attempts'):
                bench.load_task_bundle(root)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'run'
            report, _ = run_fixture(root)
            def success(raw):
                raw['success'] = raw['metrics']['success'] = True
            mutate_attempt(root, report, 0, success)
            with self.assertRaisesRegex(ValueError,'lift/carry/release'):
                bench.load_task_bundle(root)

    def test_inventory_license_and_resource_boundaries(self):
        for variant in ('unsealed', 'license', 'path', 'bound'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)/'run'
                run_fixture(root)
                if variant == 'unsealed':
                    (root/'unrelated.txt').write_text('unexpected')
                elif variant == 'license':
                    (root/bench.LICENSE_PATH).write_bytes(b'changed license')
                    reseal(root)
                elif variant == 'path':
                    manifest = json.loads((root/'integrity.json').read_bytes())
                    manifest['files']['../escape'] = {'bytes':0, 'sha256':'a'*64}
                    (root/'integrity.json').write_bytes(bench._json_bytes(manifest))
                if variant == 'bound':
                    with patch.object(bench, 'MAX_BUNDLE_BYTES', 1), self.assertRaisesRegex(ValueError, 'byte bound'):
                        bench.load_task_bundle(root)
                else:
                    with self.assertRaises(ValueError):
                        bench.load_task_bundle(root)


if __name__ == '__main__':
    unittest.main()

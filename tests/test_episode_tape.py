"""Evidence contract, time alignment and native recording checks."""
from copy import deepcopy
from contextlib import redirect_stdout
from importlib.util import find_spec
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from dvidia_training.episode_tape import (ARM_PROFILE, KIND, PROTOCOL, SIGNALS, TASK_ID,
                                          canonical, digest, inspect_tape, load_tape,
                                          main, record_episode, snapshot, validate_tape, write_tape)


def fixture():
    observation = {name: ([0.] * contract['shape'][0] if contract['shape'] else 0.)
                   for name, contract in SIGNALS.items()}
    observation.update(end_effector_quaternion=[1., 0., 0., 0.], object_quaternion=[1., 0., 0., 0.],
                       grasp_contacts={'left': 0, 'right': 0}, table_contacts=1,
                       pad_normal_forces={'left': 0., 'right': 0.}, gripper_width=.08)
    initial = snapshot(observation, 0.)
    after = snapshot(observation, .02)
    action = {'joint_targets': [0.] * 6, 'gripper_width': .08}
    config = {'timestep': .001, 'control_dt': .02, 'horizon': .02, 'settle_time': .4,
              'object_position': [.42, -.04, .31], 'target_position': [.54, .10, .31],
              'object_half_size': [.02, .02, .02], 'object_mass': .04, 'object_friction': .8,
              'table_height': .29, 'position_tolerance': .025, 'dwell_seconds': .25,
              'scene_jitter': 0., 'joint_torque_limit': 40., 'jaw_force_limit': 15.,
              'joint_target_rate_limit': 3., 'jaw_width_rate_limit': .16, 'gravity': 9.81}
    result = {'terminated': False, 'truncated': True, 'success': False, 'valid': True, 'reason': 'horizon'}
    return {'schema_version': 1, 'kind': KIND, 'scope': 'simulation-only', 'physical_robot_ready': False,
            'task_id': TASK_ID, 'arm_profile': ARM_PROFILE,
            'provenance': {'episode_id': 'fixture-00', 'source_recording_id': 'source-00', 'session_id': 'session-00',
                           'source_kind': 'native-simulation-telemetry', 'seed': 0, 'case': 'nominal',
                           'policy_origin': 'authored controller', 'policy_revision': 'test-fixture',
                           'engine': 'mujoco-3.15.0', 'scene_sha256': '0'*64, 'configuration_sha256': digest(config),
                           'source_files_sha256': {name: '0'*64 for name in ('arm_env.py', 'arm_policy.py', 'episode_tape.py')}},
            'clock': {'kind': 'simulation-elapsed-seconds', 'origin': 'after-reset-settle', 'native_time_offset_seconds': .4},
            'signal_contract': deepcopy(SIGNALS), 'protocol': deepcopy(PROTOCOL), 'protocol_sha256': digest(PROTOCOL),
            'configuration': config, 'initial': initial,
            'steps': [{'index': 0, 'before': deepcopy(initial),
                       'command': {'issued_at_seconds': 0., 'policy_requested': deepcopy(action), 'issued': deepcopy(action), 'intervention': 'none'},
                       'applied': {'interval_start_seconds': 0., 'interval_end_seconds': .02,
                                   'ramped_joint_targets': [0.] * 6, 'ramped_gripper_width': .08,
                                   'native_position_controls': [0.] * 6+[.04, .04], 'rate_limited': False},
                       'after': after, 'labels': {'origin': 'authored controller', 'action_phase': 'approach', 'next_phase': 'approach'},
                       'result': result}],
            'outcome': {'result': deepcopy(result), 'simulation_seconds': .02, 'control_steps': 1,
                        'policy_failure_reason': None, 'recovery_count': 0}}


class EpisodeTapeContractTests(unittest.TestCase):
    def test_validate_detaches_input_and_inspection_counts_unique_observations(self):
        tape = fixture()
        validated = validate_tape(tape)
        validated['initial']['signals']['joint_position']['value'][0] = 1.
        self.assertEqual(tape['initial']['signals']['joint_position']['value'][0], 0.)
        summary = inspect_tape(tape)
        self.assertEqual(summary['unique_observations'], 2)
        self.assertEqual(summary['signal_status_counts']['joint_position'], {'available': 2, 'stale': 0, 'missing': 0})
        self.assertFalse(summary['physical_robot_ready'])

    def test_read_only_operations_do_not_import_optional_native_or_numpy_packages(self):
        code = '''import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('mujoco', 'numpy'):
        raise RuntimeError('Optional dependency imported')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from dvidia_training.episode_tape import validate_tape, inspect_tape
from tests.test_episode_tape import fixture
assert inspect_tape(validate_tape(fixture()))['control_steps'] == 1
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_sensor_missingness_and_staleness_are_explicit_and_counted(self):
        tape = fixture()
        missing = {'status': 'missing', 'captured_at_seconds': None, 'age_seconds': None, 'value': None}
        for snapshot_ in (tape['initial'], tape['steps'][0]['before'], tape['steps'][0]['after']):
            snapshot_['signals']['pad_normal_forces'] = deepcopy(missing)
        # An observation may be stale without changing time order. It is not replaced with zeros.
        tape['configuration']['control_dt'] = .2
        tape['configuration']['horizon'] = .2
        tape['provenance']['configuration_sha256'] = digest(tape['configuration'])
        tape['steps'][0]['after']['timestamp_seconds'] = .2
        for sensor in tape['steps'][0]['after']['signals'].values():
            if sensor['status'] != 'missing':
                sensor.update(captured_at_seconds=.2)
        tape['steps'][0]['after']['signals']['joint_velocity'].update(status='stale', captured_at_seconds=0., age_seconds=.2)
        tape['steps'][0]['applied']['interval_end_seconds'] = .2
        tape['outcome']['simulation_seconds'] = .2
        summary = inspect_tape(tape)
        self.assertEqual(summary['signal_status_counts']['pad_normal_forces']['missing'], 2)
        self.assertEqual(summary['signal_status_counts']['joint_velocity']['stale'], 1)
        self.assertEqual(summary['max_signal_age_seconds']['joint_velocity'], .2)

    def test_absent_observation_field_is_not_a_zero_measurement(self):
        partial = snapshot({}, 0.)
        for sensor in partial['signals'].values():
            self.assertEqual(sensor, {'status': 'missing', 'captured_at_seconds': None, 'age_seconds': None, 'value': None})

    def test_duplicate_fields_and_nonfinite_json_are_refused(self):
        for raw in ('{"kind": "a", "kind": "b"}', '{"value": NaN}', '{"value": Infinity}', b'\xff'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                validate_tape(raw)

    def test_unknown_keys_are_refused_at_nested_contract_boundaries(self):
        paths = [(), ('provenance',), ('clock',), ('initial',), ('initial', 'signals', 'joint_position'),
                 ('steps', 0), ('steps', 0, 'command'), ('steps', 0, 'applied'), ('steps', 0, 'labels'),
                 ('steps', 0, 'result'), ('outcome',)]
        for path in paths:
            tape = fixture()
            target = tape
            for key in path:
                target = target[key]
            target['unexpected'] = 'unchecked'
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_tape(tape)

    def test_nonfinite_boolean_shape_and_quaternion_measurements_are_refused(self):
        corruptions = [('joint_position', [0.]*5), ('joint_position', [True]+[0.]*5),
                       ('joint_position', [float('nan')]+[0.]*5), ('joint_position', [10**1000]+[0.]*5),
                       ('object_quaternion', [0.]*4), ('grasp_contacts', [0., 1.]),
                       ('gripper_width', -.001), ('pad_normal_forces', [-1., 0.])]
        for name, value in corruptions:
            tape = fixture()
            tape['steps'][0]['after']['signals'][name]['value'] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                validate_tape(tape)

    def test_sensor_age_and_missingness_contradictions_are_refused(self):
        for update in ({'age_seconds': .001}, {'captured_at_seconds': .03}, {'status': 'stale'},
                       {'status': 'missing'}, {'status': 'unknown'}):
            tape = fixture()
            tape['steps'][0]['after']['signals']['joint_position'].update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_tape(tape)

    def test_time_reordering_input_discontinuity_and_application_mismatch_are_refused(self):
        for target, field, value in [('after', 'timestamp_seconds', 0.), ('applied', 'interval_start_seconds', .001),
                                     ('applied', 'interval_end_seconds', .01), ('command', 'issued_at_seconds', .001),
                                     ('command', 'issued_at_seconds', False)]:
            tape = fixture()
            tape['steps'][0][target][field] = value
            with self.subTest(target=target, field=field), self.assertRaises(ValueError):
                validate_tape(tape)
        tape = fixture()
        tape['steps'][0]['before']['signals']['gripper_width']['value'] = .02
        with self.assertRaises(ValueError):
            validate_tape(tape)

    def test_native_jaw_controls_must_match_ramped_gap_symmetrically(self):
        for controls in ([0.] * 8, [0.] * 6+[.02, .04], [0.] * 6+[.03, .03]):
            tape = fixture()
            tape['steps'][0]['applied']['native_position_controls'] = controls
            with self.subTest(controls=controls), self.assertRaisesRegex(ValueError, 'half the recorded ramped gap'):
                validate_tape(tape)
        tape = fixture()
        # Native joint controls include gravity compensation, so do not require
        # them to equal ramped joint targets.
        tape['steps'][0]['applied']['native_position_controls'][0] = .001
        validate_tape(tape)

    def test_native_clock_origin_matches_actual_declared_settling_ticks(self):
        tape = fixture()
        tape['clock']['native_time_offset_seconds'] = 987.
        with self.assertRaisesRegex(ValueError, 'settling ticks'):
            validate_tape(tape)
        tape = fixture()
        # This falls within the runtime's integer-ratio tolerance. Reset takes
        # exactly 400 native ticks, rather than adopting the nominal time.
        tape['configuration']['settle_time'] = .4000000000005
        tape['provenance']['configuration_sha256'] = digest(tape['configuration'])
        validate_tape(tape)

    def test_extreme_finite_numeric_values_are_cleanly_refused_by_api_and_cli(self):
        cases = []
        ratio = fixture()
        ratio['configuration'].update(timestep=1e-308, control_dt=1e308, horizon=1e308)
        ratio['provenance']['configuration_sha256'] = digest(ratio['configuration'])
        cases.append(ratio)
        large_finite_ratio = fixture()
        large_finite_ratio['configuration'].update(timestep=1e-308, control_dt=1., horizon=1.)
        large_finite_ratio['provenance']['configuration_sha256'] = digest(large_finite_ratio['configuration'])
        cases.append(large_finite_ratio)
        for quaternion in ([10**300, 0, 0, 0], [1e308, 1e308, 1e308, 1e308]):
            orientation = fixture()
            orientation['steps'][0]['after']['signals']['object_quaternion']['value'] = quaternion
            cases.append(orientation)
        with TemporaryDirectory() as directory:
            for index, tape in enumerate(cases):
                with self.subTest(index=index), self.assertRaises(ValueError):
                    validate_tape(tape)
                path = Path(directory)/f'extreme-{index}.json'
                path.write_bytes(canonical(tape))
                for operation in ('validate', 'inspect'):
                    result = subprocess.run([sys.executable, '-m', 'dvidia_training.episode_tape', operation, str(path)],
                                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
                    with self.subTest(index=index, operation=operation):
                        self.assertEqual(result.returncode, 1, result.stderr)
                        self.assertEqual(json.loads(result.stdout)['status'], 'error')
                        self.assertNotIn('Traceback', result.stderr)

    def test_terminal_single_native_tick_accepts_only_float_roundoff(self):
        tape = fixture()
        end = .00099999999999999
        tape['steps'][0]['after']['timestamp_seconds'] = end
        for sensor in tape['steps'][0]['after']['signals'].values():
            sensor['captured_at_seconds'] = end
        tape['steps'][0]['applied']['interval_end_seconds'] = end
        tape['steps'][0]['result'].update(terminated=True, truncated=False, success=True, reason='success')
        tape['outcome']['result'] = deepcopy(tape['steps'][0]['result'])
        tape['outcome']['simulation_seconds'] = end
        validate_tape(tape)
        tape['steps'][0]['after']['timestamp_seconds'] = .0005
        for sensor in tape['steps'][0]['after']['signals'].values():
            sensor['captured_at_seconds'] = .0005
        tape['steps'][0]['applied']['interval_end_seconds'] = .0005
        tape['outcome']['simulation_seconds'] = .0005
        with self.assertRaises(ValueError):
            validate_tape(tape)

    def test_false_success_scope_escalation_and_early_terminal_are_refused(self):
        for field, value in [('success', True), ('terminated', True), ('valid', False), ('truncated', 1)]:
            tape = fixture()
            tape['steps'][0]['result'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_tape(tape)
        for field, value in [('physical_robot_ready', True), ('schema_version', True), ('scope', 'hardware')]:
            tape = fixture()
            tape[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_tape(tape)
        tape = fixture()
        tape['steps'].append(deepcopy(tape['steps'][0]))
        with self.assertRaises(ValueError):
            validate_tape(tape)

    def test_commands_cannot_be_silently_changed_or_recast_as_measurements(self):
        tape = fixture()
        tape['steps'][0]['command']['issued']['gripper_width'] = .01
        with self.assertRaises(ValueError):
            validate_tape(tape)
        tape = fixture()
        tape['provenance']['case'] = 'open-jaw'
        tape['steps'][0]['command']['policy_requested']['gripper_width'] = .01
        tape['steps'][0]['command']['intervention'] = 'open-jaw'
        self.assertEqual(validate_tape(tape)['steps'][0]['after']['signals']['gripper_width']['value'], .08)

    def test_hashes_final_receipt_and_signal_contract_cannot_drift(self):
        tape = fixture()
        mutations = [lambda x: x['configuration'].update(object_mass=.08),
                     lambda x: x.update(protocol_sha256='0'*64),
                     lambda x: x['signal_contract']['object_position'].update(privileged=1),
                     lambda x: x['outcome'].update(control_steps=2),
                     lambda x: x['outcome'].update(simulation_seconds=.04)]
        for mutation in mutations:
            modified = deepcopy(tape)
            mutation(modified)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_tape(modified)

    def test_file_round_trip_is_fresh_and_cli_inspection_preserves_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'episode.json'
            receipt = write_tape(path, fixture())
            original = path.read_bytes()
            self.assertEqual(receipt['bytes'], len(original))
            self.assertEqual(load_tape(path), fixture())
            with self.assertRaises(FileExistsError):
                write_tape(path, fixture())
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(['inspect', str(path)]), 0)
            self.assertEqual(json.loads(output.getvalue())['unique_observations'], 2)
            self.assertEqual(path.read_bytes(), original)

    def test_invalid_identifier_rejected_before_optional_native_import(self):
        with self.assertRaises(ValueError):
            record_episode(episode_id='../private', source_recording_id='s', session_id='g')


@unittest.skipUnless(find_spec('mujoco') is not None, 'Requires optional native arm extra')
class EpisodeTapeNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nominal = record_episode(episode_id='native-nominal', source_recording_id='fresh-native-01', session_id='fresh-session-19050', seed=19050)
        cls.open_jaw = record_episode(episode_id='native-open-jaw', source_recording_id='fresh-native-02', session_id='fresh-session-19050', seed=19050, case='open-jaw')
        cls.released = record_episode(episode_id='native-released', source_recording_id='fresh-native-03', session_id='fresh-session-19050', seed=19050, case='release-after-lift')

    def test_native_tape_preserves_command_measurement_and_terminal_semantics(self):
        tape = self.nominal
        self.assertTrue(tape['outcome']['result']['success'])
        self.assertAlmostEqual(tape['clock']['native_time_offset_seconds'], .4, places=12)
        steps = tape['steps']
        self.assertTrue(any(step['applied']['rate_limited'] for step in steps))
        self.assertTrue(any(abs(step['command']['issued']['gripper_width']-step['after']['signals']['gripper_width']['value']) > .002 for step in steps))
        for step in steps:
            self.assertEqual(step['command']['issued_at_seconds'], step['before']['timestamp_seconds'])
            self.assertEqual(step['after']['signals']['joint_position']['age_seconds'], 0.)
            self.assertEqual(step['labels']['origin'], 'authored controller')
            self.assertEqual(step['applied']['native_position_controls'][6:], [step['applied']['ramped_gripper_width']/2]*2)
        self.assertTrue(tape['signal_contract']['object_position']['privileged'])

    def test_open_jaw_case_cannot_be_reported_as_successful_grasp(self):
        tape = self.open_jaw
        self.assertFalse(tape['outcome']['result']['success'])
        self.assertEqual(tape['outcome']['result']['reason'], 'horizon')
        self.assertEqual(inspect_tape(tape)['command_intervention_samples'], len(tape['steps']))
        self.assertTrue(all(step['after']['signals']['grasp_contacts']['value'] == [0, 0] for step in tape['steps']))
        self.assertEqual(tape['provenance']['scene_sha256'], self.nominal['provenance']['scene_sha256'])

    def test_released_case_records_intervention_and_authored_recovery_separately(self):
        tape = self.released
        summary = inspect_tape(tape)
        self.assertTrue(tape['outcome']['result']['success'])
        self.assertEqual(tape['outcome']['recovery_count'], 1)
        self.assertGreater(summary['command_intervention_samples'], 0)
        self.assertLessEqual(summary['command_intervention_samples'], 18)
        self.assertGreater(summary['authored_action_phase_samples'].get('recover_wait', 0), 0)
        intervened = [step for step in tape['steps'] if step['command']['intervention'] != 'none']
        self.assertTrue(any(step['command']['policy_requested']['gripper_width'] != .08 for step in intervened))
        self.assertTrue(all(step['command']['issued']['gripper_width'] == .08 for step in intervened))
        intervention_seconds = sum(step['applied']['interval_end_seconds']-step['applied']['interval_start_seconds'] for step in intervened)
        self.assertAlmostEqual(intervention_seconds, .36, places=9)
        self.assertEqual(tape['provenance']['scene_sha256'], self.nominal['provenance']['scene_sha256'])
        self.assertEqual(tape['provenance']['session_id'], self.nominal['provenance']['session_id'])

    def test_native_record_validate_and_inspect_cli_round_trip(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'native-cli.json'
            prefix = [sys.executable, '-m', 'dvidia_training.episode_tape']
            command = prefix+['record', '--output', str(path), '--episode-id', 'native-cli',
                             '--source-recording-id', 'native-cli-recording', '--session-id', 'native-cli-session',
                             '--seed', '19050']
            recorded = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60)
            self.assertEqual(recorded.returncode, 0, recorded.stderr or recorded.stdout)
            receipt = json.loads(recorded.stdout)
            self.assertTrue(receipt['inspection']['outcome']['result']['success'])
            original = path.read_bytes()
            for operation in ('validate', 'inspect'):
                result = subprocess.run(prefix+[operation, str(path)], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
                self.assertEqual(json.loads(result.stdout)['scope'], 'simulation-only')
            repeated = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60)
            self.assertEqual(repeated.returncode, 1, repeated.stderr)
            self.assertEqual(json.loads(repeated.stdout)['status'], 'error')
            self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()

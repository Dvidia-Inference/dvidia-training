"""Causal boundaries, action sensitivity and bounded portable model artifacts."""
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from dvidia_training.episode_tape import canonical, digest, validate_tape
from dvidia_training.transition_model import (
    CONTROL_DT_SECONDS, FEATURE_NAMES, MAX_MODEL_BYTES, OUTPUT_NAMES,
    extract_rows, fit_model, load_model, predict, predict_features,
    validate_model, write_model,
)
from tests.test_episode_tape import fixture


def learning_rows():
    """Matched states with independently varied issued joint and jaw commands."""
    rows = []
    for joint in (-.5, 0., .5):
        for gap in (0., .04, .08):
            for velocity in (-.1, .1):
                features = [0.]*6 + [velocity]*6 + [.04] + [joint]*6 + [gap]
                target = [.012*joint-.003*velocity]*6 + [.1*(gap-.04)]
                rows.append({'features': features, 'target_delta': target})
    return rows


def multi_step_fixture(count=7):
    tape = fixture()
    template = deepcopy(tape['steps'][0])
    tape['configuration']['horizon'] = count*.02
    tape['provenance']['configuration_sha256'] = digest(tape['configuration'])
    before = deepcopy(tape['initial'])
    tape['steps'] = []
    for index in range(count):
        step = deepcopy(template)
        end = (index+1)*.02
        after = deepcopy(tape['initial'])
        after['timestamp_seconds'] = end
        for sensor in after['signals'].values():
            sensor['captured_at_seconds'] = end
        step.update(index=index, before=deepcopy(before), after=after)
        step['command']['issued_at_seconds'] = before['timestamp_seconds']
        step['applied'].update(interval_start_seconds=before['timestamp_seconds'], interval_end_seconds=end)
        if index < count-1:
            step['result'].update(truncated=False, reason='running')
        tape['steps'].append(step)
        before = after
    tape['outcome'].update(result=deepcopy(tape['steps'][-1]['result']),
                           simulation_seconds=count*.02, control_steps=count)
    return validate_tape(tape)


def rehash(model):
    model['model_sha256'] = digest({key: value for key, value in model.items() if key != 'model_sha256'})
    return model


class TransitionExtractionTests(unittest.TestCase):
    def test_exact_input_and_output_contract_excludes_privileged_scene_and_phase(self):
        tape = fixture()
        step = tape['steps'][0]
        step['after']['signals']['joint_position']['value'][0] = .002
        step['after']['signals']['gripper_width']['value'] = .075
        result = extract_rows(tape)
        self.assertEqual(result['attempted_steps'], 1)
        self.assertEqual(result['exclusions'], [])
        row = result['rows'][0]
        self.assertEqual(row['features'], [0.]*12+[.08]+[0.]*6+[.08])
        self.assertEqual(row['target_delta'][:6], [.002]+[0.]*5)
        self.assertAlmostEqual(row['target_delta'][6], -.005)
        self.assertEqual((len(FEATURE_NAMES), len(OUTPUT_NAMES)), (20, 7))
        self.assertEqual(CONTROL_DT_SECONDS, .02)
        self.assertFalse(any(token in name for name in FEATURE_NAMES
                             for token in ('phase', 'case', 'seed', 'time', 'object', 'contact', 'after', 'applied')))

    def test_future_labels_object_state_and_applied_controls_do_not_change_inputs(self):
        tape = fixture()
        original = extract_rows(tape)['rows'][0]
        changed = deepcopy(tape)
        step = changed['steps'][0]
        step['labels'].update(action_phase='recover_up', next_phase='verify')
        step['after']['signals']['joint_position']['value'] = [.01]*6
        step['after']['signals']['object_position']['value'] = [.45, .1, .33]
        step['after']['signals']['grasp_contacts']['value'] = [1, 1]
        step['applied']['native_position_controls'][0] = .01
        step['applied']['ramped_joint_targets'][0] = .01
        new = extract_rows(changed)['rows'][0]
        self.assertEqual(new['features'], original['features'])
        self.assertNotEqual(new['target_delta'], original['target_delta'])

    def test_actual_issued_command_not_policy_intent_enters_input(self):
        tape = fixture()
        tape['provenance']['case'] = 'open-jaw'
        tape['steps'][0]['command'].update(intervention='open-jaw')
        tape['steps'][0]['command']['policy_requested']['gripper_width'] = 0.
        row = extract_rows(tape)['rows'][0]
        self.assertEqual(row['features'][19], .08)

    def test_required_missing_signals_excluded_with_reason_without_zero_fill(self):
        for boundary, name in [('before', 'joint_position'), ('before', 'joint_velocity'),
                               ('before', 'gripper_width'), ('after', 'joint_position'),
                               ('after', 'gripper_width')]:
            tape = fixture()
            missing = {'status': 'missing', 'captured_at_seconds': None, 'age_seconds': None, 'value': None}
            tape['steps'][0][boundary]['signals'][name] = deepcopy(missing)
            if boundary == 'before':
                tape['initial']['signals'][name] = deepcopy(missing)
            result = extract_rows(tape)
            with self.subTest(boundary=boundary, name=name):
                self.assertEqual(result['rows'], [])
                self.assertEqual(len(result['exclusions']), 1)
                self.assertIn(f'missing:{name}', result['exclusions'][0]['reason'])

    def test_unrequired_missing_contacts_do_not_block_boundary(self):
        tape = fixture()
        for snapshot in (tape['initial'], tape['steps'][0]['before'], tape['steps'][0]['after']):
            snapshot['signals']['grasp_contacts'] = {
                'status': 'missing', 'captured_at_seconds': None, 'age_seconds': None, 'value': None}
        self.assertEqual(len(extract_rows(tape)['rows']), 1)

    def test_stale_required_before_and_after_excluded(self):
        tape = multi_step_fixture()
        stale = {'status': 'stale', 'captured_at_seconds': 0., 'age_seconds': .12, 'value': [0.]*6}
        tape['steps'][5]['after']['signals']['joint_position'] = deepcopy(stale)
        tape['steps'][6]['before']['signals']['joint_position'] = deepcopy(stale)
        result = extract_rows(tape)
        self.assertEqual(result['attempted_steps'], 7)
        self.assertEqual(len(result['rows']), 5)
        self.assertEqual([item['index'] for item in result['exclusions']], [5, 6])
        self.assertTrue(all('stale:joint_position' in item['reason'] for item in result['exclusions']))

    def test_available_but_aged_required_before_and_after_are_excluded(self):
        tape = multi_step_fixture()
        # This signal is still available by the tape's 100 ms recording rule,
        # but a 20 ms label must represent the actual boundary acquisition.
        aged = {'status': 'available', 'captured_at_seconds': .019,
                'age_seconds': .001, 'value': [0.]*6}
        tape['steps'][0]['after']['signals']['joint_position'] = deepcopy(aged)
        tape['steps'][1]['before']['signals']['joint_position'] = deepcopy(aged)
        result = extract_rows(tape)
        self.assertEqual([item['index'] for item in result['exclusions']], [0, 1])
        self.assertTrue(all('noncontemporaneous:joint_position' in item['reason']
                            for item in result['exclusions']))

    def test_partial_terminal_boundary_is_excluded(self):
        tape = fixture()
        step = tape['steps'][0]
        step['after']['timestamp_seconds'] = .001
        for sensor in step['after']['signals'].values():
            sensor['captured_at_seconds'] = .001
        step['applied']['interval_end_seconds'] = .001
        step['result'].update(terminated=True, truncated=False, success=True, reason='success')
        tape['outcome'].update(result=deepcopy(step['result']), simulation_seconds=.001)
        result = extract_rows(tape)
        self.assertEqual(result['rows'], [])
        self.assertEqual(result['exclusions'][0]['reason'], 'incomplete_20ms_boundary')

    def test_tape_validation_precedes_extraction(self):
        tape = fixture()
        tape['physical_robot_ready'] = True
        with self.assertRaises(ValueError):
            extract_rows(tape)


class CompactTransitionTests(unittest.TestCase):
    def setUp(self):
        self.rows = learning_rows()
        self.model = fit_model(self.rows, alpha=1e-6)

    def test_action_sensitive_predictions_and_state_ablation_on_matched_states(self):
        passive = fit_model(self.rows, alpha=1e-6, feature_mode='state')
        first = [0.]*12+[.04]+[-.5]*6+[0.]
        second = [0.]*12+[.04]+[.5]*6+[.08]
        low, high = (predict_features(self.model, features) for features in (first, second))
        self.assertAlmostEqual(high[0]-low[0], .012, places=7)
        self.assertAlmostEqual(high[6]-low[6], .008, places=7)
        self.assertEqual(predict_features(passive, first), predict_features(passive, second))

    def test_weighted_train_only_normalization_and_unpenalized_intercept(self):
        rows = [{'features': [0.]*20, 'target_delta': [2.]*7},
                {'features': [2.]*20, 'target_delta': [4.]*7},
                {'features': [1000.]*20, 'target_delta': [1000.]*7}]
        model = fit_model(rows, alpha=1e6, sample_weights=[1., 3., 0.])
        self.assertEqual(model['normalization']['means'], [1.5]*20)
        self.assertAlmostEqual(model['normalization']['scales'][0], 3**.5/2)
        self.assertEqual(model['intercept'], [3.5]*7)
        # Inference on a novel state cannot change fitted normalization.
        baseline = deepcopy(model['normalization'])
        predict_features(model, [9999.]*20)
        self.assertEqual(model['normalization'], baseline)
        self.assertEqual(model['training']['rows'], 3)
        self.assertEqual(model['training']['positive_weight_rows'], 2)

    def test_constant_features_have_zero_coefficients_and_safe_scale(self):
        normalization = self.model['normalization']
        self.assertTrue(normalization['constant_features'][0])
        self.assertEqual(normalization['scales'][0], 1.)
        self.assertTrue(all(row[0] == 0. for row in self.model['coefficients']))
        changed = [row['features'][0]+1 for row in self.rows]
        self.assertEqual(len(changed), 18)

    def test_missing_and_stale_observations_abstain(self):
        snapshot = deepcopy(fixture()['initial'])
        command = {'joint_targets': [0.]*6, 'gripper_width': .04}
        self.assertEqual(predict(self.model, snapshot, command)['status'], 'predicted')
        missing = deepcopy(snapshot)
        missing['signals']['joint_position'].update(status='missing', captured_at_seconds=None,
                                                    age_seconds=None, value=None)
        result = predict(self.model, missing, command)
        self.assertEqual(result['status'], 'abstained')
        self.assertIn('missing:joint_position', result['reason'])
        snapshot['timestamp_seconds'] = .2
        for sensor in snapshot['signals'].values():
            sensor.update(captured_at_seconds=.2)
        snapshot['signals']['joint_velocity'].update(status='stale', captured_at_seconds=0., age_seconds=.2)
        result = predict(self.model, snapshot, command)
        self.assertEqual(result['status'], 'abstained')
        self.assertIn('stale:joint_velocity', result['reason'])

    def test_available_but_delayed_observation_abstains_without_age_propagation(self):
        snapshot = deepcopy(fixture()['initial'])
        snapshot['timestamp_seconds'] = .02
        for sensor in snapshot['signals'].values():
            sensor.update(captured_at_seconds=.02)
        snapshot['signals']['joint_position'].update(captured_at_seconds=.019, age_seconds=.001)
        result = predict(self.model, snapshot, {'joint_targets': [0.]*6, 'gripper_width': .04})
        self.assertEqual(result['status'], 'abstained')
        self.assertIn('noncontemporaneous:joint_position', result['reason'])

    def test_direct_prediction_rejects_clock_contradictions_and_invalid_commands(self):
        before = fixture()['initial']
        before['signals']['joint_position']['captured_at_seconds'] = .01
        self.assertEqual(predict(self.model, before, {'joint_targets': [0.]*6, 'gripper_width': .04})['status'], 'abstained')
        before = fixture()['initial']
        for command in ({'joint_targets': [0.]*5, 'gripper_width': .04},
                        {'joint_targets': [0.]*6, 'gripper_width': .2},
                        {'joint_targets': [0.]*6, 'gripper_width': .04, 'phase': 'approach'}):
            self.assertEqual(predict(self.model, before, command)['status'], 'abstained')

    def test_inference_load_and_extraction_do_not_import_numpy_or_mujoco(self):
        with TemporaryDirectory() as directory:
            model_path = Path(directory)/'model.json'
            write_model(model_path, self.model)
            code = '''import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('numpy', 'mujoco'):
        raise RuntimeError('Optional numeric/native dependency imported')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from dvidia_training.transition_model import load_model,predict_features,extract_rows
from tests.test_episode_tape import fixture
model=load_model(__import__('sys').argv[1])
assert len(predict_features(model,[0.]*20))==7
assert len(extract_rows(fixture())['rows'])==1
'''
            result = subprocess.run([sys.executable, '-c', code, str(model_path)],
                                    cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_serialization_roundtrip_hashes_detachment_and_no_overwrite(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'model.json'
            receipt = write_model(path, self.model)
            self.assertLess(receipt['bytes'], MAX_MODEL_BYTES)
            self.assertEqual(receipt['sha256'], sha256(path.read_bytes()).hexdigest())
            restored = load_model(path)
            self.assertEqual(predict_features(restored, self.rows[0]['features']),
                             predict_features(self.model, self.rows[0]['features']))
            restored['intercept'][0] = 100.
            self.assertNotEqual(restored['intercept'][0], self.model['intercept'][0])
            with self.assertRaises(FileExistsError):
                write_model(path, self.model)

    def test_loading_without_posix_flags_still_checks_regular_file_and_symlink(self):
        import dvidia_training.transition_model as transition_model
        portable_os = SimpleNamespace(O_RDONLY=os.O_RDONLY, O_BINARY=getattr(os,'O_BINARY',0),
                                      open=os.open, fdopen=os.fdopen, fstat=os.fstat)
        with TemporaryDirectory() as directory:
            path = Path(directory)/'model.json'
            write_model(path,self.model)
            with patch.object(transition_model,'os',portable_os):
                self.assertEqual(load_model(path),self.model)
                with self.assertRaises(ValueError):
                    load_model(Path(directory))
                link = Path(directory)/'link.json'
                try:
                    link.symlink_to(path)
                except OSError:
                    pass  # Some Windows installations do not grant symlink permission.
                else:
                    with self.assertRaises(ValueError):
                        load_model(link)

    def test_model_path_swap_is_detected_before_parsing_when_file_identity_exists(self):
        import dvidia_training.transition_model as transition_model
        with TemporaryDirectory() as directory:
            path = Path(directory)/'model.json'
            other = Path(directory)/'replacement.json'
            write_model(path,self.model)
            write_model(other,self.model)
            before = path.lstat()
            with path.open('rb') as stream:
                opened = os.fstat(stream.fileno())
            if not before.st_ino or not opened.st_ino:
                self.skipTest('Path/descriptor identity is unavailable on this platform.')
            original_open = os.open
            def swapped_open(name,flags):
                path.unlink()
                other.rename(path)
                return original_open(name,flags)
            with patch.object(transition_model.os,'open',side_effect=swapped_open):
                with self.assertRaisesRegex(ValueError,'identity changed'):
                    load_model(path)

    def test_unknown_fields_false_scope_and_model_hash_tampering_refused(self):
        for key, value in [('unexpected', 'executable'), ('physical_robot_ready', True),
                           ('is_controller', True), ('predicts_task_success', True),
                           ('scope', 'hardware'), ('control_dt_seconds', .04),
                           ('schema_version', True)]:
            model = deepcopy(self.model)
            model[key] = value
            rehash(model)
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_model(model)
        changed = deepcopy(self.model)
        changed['intercept'][0] += .1
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            validate_model(changed)

    def test_nonfinite_bool_extreme_values_bad_shapes_and_weights_refused(self):
        for value in (True, float('nan'), float('inf'), 10**1000, 1e7):
            rows = deepcopy(self.rows)
            rows[0]['features'][0] = value
            with self.subTest(value=type(value).__name__), self.assertRaises(ValueError):
                fit_model(rows)
            features = [0.]*20
            features[0] = value
            with self.assertRaises(ValueError):
                predict_features(self.model, features)
        for weights in ([1.], [-1.]*18, [0.]*18, [True]*18):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                fit_model(self.rows, sample_weights=weights)
        for features in ([0.]*19, [0.]*21, {'features': [0.]*20}):
            with self.assertRaises(ValueError):
                predict_features(self.model, features)

    def test_artifact_nested_contracts_are_bounded_and_finite_even_with_recomputed_hash(self):
        for path, value in [(('normalization', 'means'), [0.]*19),
                            (('normalization', 'scales'), [0.]*20),
                            (('normalization', 'constant_features'), [1]*20),
                            (('normalization', 'scope'), 'all-partitions'),
                            (('training', 'rows'), True),
                            (('training', 'rows_sha256'), 'x'*64),
                            (('training', 'extra'), 'unchecked'),
                            (('coefficients',), [[0.]*20]*6),
                            (('intercept',), [0.]*6)]:
            model = deepcopy(self.model)
            target = model
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            rehash(model)
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_model(model)
        model = deepcopy(self.model)
        model['intercept'][0] = float('nan')
        with self.assertRaises(ValueError):
            validate_model(model)

    def test_duplicate_nonfinite_oversized_and_symlink_artifacts_refused(self):
        for raw in ('{"kind":"first","kind":"second"}', '{"coefficient":NaN}',
                    b'\xff', '\ud800', b' '* (MAX_MODEL_BYTES+1)):
            with self.subTest(raw_type=type(raw).__name__), self.assertRaises(ValueError):
                validate_model(raw)
        with TemporaryDirectory() as directory:
            path = Path(directory)/'model.json'
            write_model(path, self.model)
            link = Path(directory)/'link.json'
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                load_model(link)


if __name__ == '__main__':
    unittest.main()

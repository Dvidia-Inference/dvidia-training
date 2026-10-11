"""Analytic paired replay errors and strict, portable offline trace loading."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dvidia_training import embodiment_calibration as calibration
from dvidia_training.embodiment_calibration import (
    MAX_JOINTS, MAX_SAMPLES, MAX_TRACE_BYTES, load_trace, score_trace, validate_trace,
)


FIXTURE = Path(__file__).parent / 'fixtures' / 'paired_fidelity.json'


def fixture():
    return json.loads(FIXTURE.read_text(encoding='utf-8'))


class PairedFidelityScoringTests(unittest.TestCase):
    def test_analytic_joint_per_axis_and_multi_dof_pooled_errors(self):
        report = score_trace(fixture())
        joint = report['metrics']['joint_position']
        self.assertEqual(joint['unit'], 'rad')
        shoulder, elbow = joint['per_joint']
        self.assertEqual(shoulder['name'], 'shoulder')
        self.assertEqual(shoulder['count'], 3)
        self.assertAlmostEqual(shoulder['rmse'], math.sqrt(.1 / 3))
        self.assertAlmostEqual(shoulder['mae'], .4 / 3)
        self.assertAlmostEqual(shoulder['max_abs'], .3)
        self.assertAlmostEqual(elbow['rmse'], math.sqrt(.2 / 3))
        self.assertAlmostEqual(elbow['mae'], .2)
        self.assertAlmostEqual(elbow['max_abs'], .4)
        self.assertEqual(joint['pooled']['count'], 6)
        self.assertAlmostEqual(joint['pooled']['rmse'], math.sqrt(.3 / 6))
        self.assertAlmostEqual(joint['pooled']['mae'], 1 / 6)
        self.assertAlmostEqual(joint['pooled']['max_abs'], .4)

    def test_position_euclidean_orientation_geodesic_and_jaw_errors(self):
        report = score_trace(fixture())
        metrics = report['metrics']
        self.assertAlmostEqual(metrics['tcp_position']['euclidean']['rmse'], math.sqrt(.000125 / 3))
        self.assertAlmostEqual(metrics['tcp_position']['euclidean']['mae'], .005)
        self.assertAlmostEqual(metrics['tcp_position']['euclidean']['max_abs'], .01)
        self.assertAlmostEqual(metrics['tcp_position']['per_axis'][0]['rmse'], math.sqrt(.000045 / 3))
        self.assertAlmostEqual(metrics['tcp_orientation']['geodesic']['rmse'], math.pi * math.sqrt(5 / 12))
        self.assertAlmostEqual(metrics['tcp_orientation']['geodesic']['mae'], math.pi / 2)
        self.assertAlmostEqual(metrics['tcp_orientation']['geodesic']['max_abs'], math.pi)
        self.assertAlmostEqual(metrics['gripper_gap']['rmse'], math.sqrt(.000005 / 3))
        self.assertAlmostEqual(metrics['gripper_gap']['mae'], .001)
        self.assertAlmostEqual(metrics['gripper_gap']['max_abs'], .002)

    def test_quaternion_sign_and_normalization_roundoff_invariance(self):
        trace = fixture()
        expected = score_trace(trace)['metrics']['tcp_orientation']
        for sample in trace['samples']:
            sample['physical']['tcp_quaternion_wxyz'] = [-(1 + 1e-7) * value for value in sample['physical']['tcp_quaternion_wxyz']]
            sample['simulator']['tcp_quaternion_wxyz'] = [-(1 - 1e-7) * value for value in sample['simulator']['tcp_quaternion_wxyz']]
        result = score_trace(trace)['metrics']['tcp_orientation']
        for metric in ('rmse', 'mae', 'max_abs'):
            self.assertAlmostEqual(result['geodesic'][metric], expected['geodesic'][metric], places=14)

    def test_quaternion_near_pi_and_tiny_angles_are_stable(self):
        for angle in (math.pi - 1e-10, math.pi, 1e-10):
            trace = fixture()
            for sample in trace['samples']:
                sample['physical']['tcp_quaternion_wxyz'] = [math.cos(angle / 2), 0., 0., math.sin(angle / 2)]
            result = score_trace(trace)['metrics']['tcp_orientation']['geodesic']
            with self.subTest(angle=angle):
                self.assertAlmostEqual(result['rmse'], angle, delta=1e-14)
                self.assertAlmostEqual(result['mae'], angle, delta=1e-14)
                self.assertLessEqual(result['max_abs'], math.pi)

    def test_zero_error_and_future_joint_count_are_supported_without_actuation(self):
        trace = fixture()
        trace['joint_names'] = [f'finger_joint_{index}' for index in range(30)]
        for sample in trace['samples']:
            sample['simulator']['joint_position_rad'] = [0.] * 30
            sample['physical'] = deepcopy(sample['simulator'])
        report = score_trace(trace)
        self.assertEqual(report['joint_count'], 30)
        self.assertEqual(report['metrics']['joint_position']['pooled']['count'], 90)
        self.assertEqual(report['metrics']['joint_position']['pooled']['rmse'], 0.)
        self.assertEqual(report['metrics']['tcp_orientation']['geodesic']['rmse'], 0.)
        self.assertFalse(report['physical_robot_ready'])

    def test_direct_joint_difference_does_not_hide_angle_wrap_disagreement(self):
        trace = fixture()
        for sample in trace['samples']:
            sample['physical']['joint_position_rad'][0] = 2 * math.pi
        report = score_trace(trace)
        self.assertAlmostEqual(report['metrics']['joint_position']['per_joint'][0]['rmse'], 2 * math.pi)

    def test_single_source_correlation_irregular_sampling_and_declaration_limits(self):
        report = score_trace(fixture())
        self.assertEqual(report['input_provenance'], 'synthetic_fixture')
        self.assertEqual(report['source_group_count'], 1)
        self.assertEqual(report['sample_count'], 3)
        self.assertFalse(report['declared_provenance_authenticated'])
        self.assertFalse(report['alignment']['interpolation_performed'])
        self.assertFalse(report['alignment']['time_shift_fitted'])
        self.assertEqual(report['alignment']['status'], 'review-declaration-only')
        self.assertAlmostEqual(report['sampling']['duration_seconds'], .05)
        self.assertAlmostEqual(report['sampling']['interval_min_seconds'], .02)
        self.assertAlmostEqual(report['sampling']['interval_median_seconds'], .025)
        self.assertAlmostEqual(report['sampling']['interval_max_seconds'], .03)
        self.assertIn('equal per supplied sample', report['sampling']['weighting'])
        limitations = ' '.join(report['limitations'])
        self.assertIn('not measured physical', limitations)
        self.assertIn('correlated', limitations)
        self.assertIn('contact fidelity', limitations)
        self.assertIn('not authenticity', limitations)

    def test_declared_physical_review_does_not_become_authenticated_or_qualified(self):
        trace = fixture()
        trace['provenance'] = 'declared_physical_replay'
        trace['alignment'] = {'time_alignment_reviewed': True, 'coordinate_alignment_reviewed': True}
        report = score_trace(trace)
        self.assertEqual(report['input_provenance'], 'declared_physical_replay')
        self.assertFalse(report['declared_provenance_authenticated'])
        self.assertFalse(report['physical_robot_ready'])
        self.assertIn('not authenticated', ' '.join(report['limitations']))

    def test_validation_and_scoring_detach_without_mutating_input(self):
        trace = fixture()
        original = deepcopy(trace)
        detached = validate_trace(trace)
        detached['samples'][0]['physical']['joint_position_rad'][0] = 123.
        self.assertEqual(trace, original)
        first, second = score_trace(trace), score_trace(deepcopy(trace))
        self.assertEqual(first, second)
        self.assertEqual(trace, original)
        trace['samples'][0]['physical']['joint_position_rad'][0] += .01
        self.assertNotEqual(score_trace(trace)['trace_sha256'], first['trace_sha256'])


class PairedFidelityValidationTests(unittest.TestCase):
    def test_unknown_and_missing_fields_rejected_at_every_level(self):
        paths = ((), ('alignment',), ('samples', 0), ('samples', 0, 'simulator'), ('samples', 0, 'physical'))
        for path in paths:
            for operation in ('extra', 'missing'):
                trace = fixture()
                value = trace
                for part in path:
                    value = value[part]
                if operation == 'extra':
                    value['unexpected'] = 1
                else:
                    del value[next(iter(value))]
                with self.subTest(path=path, operation=operation), self.assertRaises(ValueError):
                    score_trace(trace)

    def test_empty_short_repeated_or_reordered_samples_rejected(self):
        variants = ([], fixture()['samples'][:1],
                    [fixture()['samples'][0], fixture()['samples'][0]],
                    list(reversed(fixture()['samples'])))
        for samples in variants:
            trace = fixture()
            trace['samples'] = samples
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                score_trace(trace)

    def test_joint_names_require_ordered_unique_nonempty_bounded_list(self):
        for names in ([], ['shoulder', 'shoulder'], [''], [True], ('shoulder', 'elbow'),
                      [f'j{index}' for index in range(MAX_JOINTS + 1)]):
            trace = fixture()
            trace['joint_names'] = names
            with self.subTest(names=names), self.assertRaises(ValueError):
                score_trace(trace)

    def test_missing_state_observations_and_wrong_vectors_are_not_zero_filled(self):
        for key, value in (('joint_position_rad', [0]), ('tcp_position_m', [0, 0]),
                           ('tcp_quaternion_wxyz', [1, 0, 0]), ('gripper_gap_m', None),
                           ('joint_position_rad', [None, 0]), ('tcp_position_m', (0, 0, 0))):
            trace = fixture()
            trace['samples'][1]['physical'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                score_trace(trace)

    def test_booleans_nonfinite_and_unbounded_numbers_rejected(self):
        for value in (True, False, float('nan'), float('inf'), -float('inf'), 1e100, 10 ** 400):
            for field in ('t_seconds', 'joint_position_rad', 'tcp_position_m', 'tcp_quaternion_wxyz', 'gripper_gap_m'):
                trace = fixture()
                if field == 't_seconds':
                    trace['samples'][0][field] = value
                elif field == 'gripper_gap_m':
                    trace['samples'][0]['physical'][field] = value
                else:
                    trace['samples'][0]['physical'][field][0] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    score_trace(trace)

    def test_negative_time_gap_and_unnormalized_quaternions_rejected(self):
        for quaternion in ([0, 0, 0, 0], [.9, 0, 0, 0], [1.00001, 0, 0, 0], [1, 1, 0, 0]):
            trace = fixture()
            trace['samples'][0]['physical']['tcp_quaternion_wxyz'] = quaternion
            with self.subTest(quaternion=quaternion), self.assertRaises(ValueError):
                score_trace(trace)
        for field in ('t_seconds', 'gripper_gap_m'):
            trace = fixture()
            if field == 't_seconds':
                trace['samples'][0][field] = -.01
            else:
                trace['samples'][0]['physical'][field] = -.01
            with self.subTest(field=field), self.assertRaises(ValueError):
                score_trace(trace)

    def test_provenance_time_frame_and_hash_contract(self):
        variants = [('schema', 'dvidia.paired-fidelity/v2'), ('provenance', 'verified_hardware'),
                    ('time_basis', 'individual_clocks'), ('coordinate_frame', 'camera'),
                    ('physical_hardware_id', ''), ('source_group', ' '),
                    ('protocol_id', 'x' * 201), ('model_sha256', 'F' * 64),
                    ('command_replay_sha256', '0' * 63)]
        for key, value in variants:
            trace = fixture()
            trace[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                score_trace(trace)

    def test_physical_replay_requires_true_boolean_alignment_reviews(self):
        for time, frame in ((False, True), (True, False), (False, False), (1, True), (True, 'true')):
            trace = fixture()
            trace['provenance'] = 'declared_physical_replay'
            trace['alignment'] = {'time_alignment_reviewed': time, 'coordinate_alignment_reviewed': frame}
            with self.subTest(time=time, frame=frame), self.assertRaises(ValueError):
                score_trace(trace)

    def test_sample_count_bound_rejected_before_processing(self):
        trace = fixture()
        trace['samples'] = [trace['samples'][0]] * (MAX_SAMPLES + 1)
        with self.assertRaises(ValueError):
            score_trace(trace)


class PairedFidelityLoaderTests(unittest.TestCase):
    def test_real_utf8_fixture_loads_and_needs_no_optional_dependencies(self):
        trace = load_trace(FIXTURE)
        self.assertEqual(trace['provenance'], 'synthetic_fixture')
        self.assertEqual(score_trace(trace)['sample_count'], 3)

    def test_scorer_runs_without_site_packages_or_optional_runtime_imports(self):
        source = Path(__file__).resolve().parents[1] / 'src'
        script = (
            'import json,sys; sys.path.insert(0,sys.argv[1]); '
            'from dvidia_training.embodiment_calibration import load_trace,score_trace; '
            'report=score_trace(load_trace(sys.argv[2])); '
            'assert not any(name.split(".")[0] in ("numpy","mujoco","cv2") for name in sys.modules); '
            'print(json.dumps({"count":report["sample_count"],"ready":report["physical_robot_ready"]}))'
        )
        result = subprocess.run([sys.executable, '-S', '-c', script, str(source), str(FIXTURE)],
                                capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), {'count': 3, 'ready': False})

    def test_duplicate_keys_at_top_and_nested_levels_rejected(self):
        originals = FIXTURE.read_text(encoding='utf-8')
        variants = ('{"schema":"x","schema":"y"}',
                    originals.replace('"gripper_gap_m": 0.04', '"gripper_gap_m": 0.04, "gripper_gap_m": 0.05', 1))
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'duplicate.json'
            for text in variants:
                path.write_text(text, encoding='utf-8')
                with self.subTest(text=text[:50]), self.assertRaisesRegex(ValueError, 'Duplicate JSON key'):
                    load_trace(path)

    def test_nonfinite_json_invalid_utf8_malformed_and_deep_json_rejected(self):
        variants = (b'{"x":NaN}', b'{"x":Infinity}', b'\xff', b'{', b'[' * 2000 + b'0' + b']' * 2000)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            for raw in variants:
                path.write_bytes(raw)
                with self.subTest(raw=raw[:30]), self.assertRaises(ValueError):
                    load_trace(path)

    def test_loader_refuses_oversized_missing_directory_and_symlink_inputs(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'oversized.json'
            path.write_bytes(b' ' * (MAX_TRACE_BYTES + 1))
            for candidate in (path, Path(directory), Path(directory) / 'missing.json'):
                with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                    load_trace(candidate)
            link = Path(directory) / 'link.json'
            try:
                link.symlink_to(FIXTURE)
            except (OSError, NotImplementedError):
                return  # Symlink creation needs additional privileges on some Windows hosts.
            with self.assertRaises(ValueError):
                load_trace(link)

    def test_binary_loading_without_posix_only_flags_is_portable(self):
        portable = SimpleNamespace(O_RDONLY=os.O_RDONLY, open=os.open, fdopen=os.fdopen, fstat=os.fstat)
        with patch.object(calibration, 'os', portable):
            result = load_trace(FIXTURE)
        self.assertEqual(result['source_group'], 'original-authored-fixture-one-group')

    def test_after_open_regular_file_size_and_identity_are_checked(self):
        original_fstat = os.fstat
        for altered in ('mode', 'size', 'identity'):
            def fstat(descriptor):
                original = original_fstat(descriptor)
                values = {name: getattr(original, name) for name in ('st_mode', 'st_size', 'st_ino', 'st_dev')}
                if altered == 'mode':
                    values['st_mode'] = 0
                elif altered == 'size':
                    values['st_size'] = MAX_TRACE_BYTES + 1
                else:
                    values['st_ino'] += 1
                return SimpleNamespace(**values)
            portable = SimpleNamespace(O_RDONLY=os.O_RDONLY, open=os.open, fdopen=os.fdopen, fstat=fstat)
            if altered == 'identity' and not FIXTURE.stat().st_ino:
                continue  # Some filesystems expose no usable identity.
            with patch.object(calibration, 'os', portable), self.subTest(altered=altered), self.assertRaises(ValueError):
                load_trace(FIXTURE)


if __name__ == '__main__':
    unittest.main()

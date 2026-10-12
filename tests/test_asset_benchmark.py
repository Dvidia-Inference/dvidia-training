"""Frozen denominator, trace-derived outcomes and tamper-resistant inspection."""
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dvidia_training import asset_benchmark as bench


def trace(supported=True):
    records = []
    for index in range(51):
        t = index*.02
        z = max(0., .08-.5*9.81*t*t) if supported else .08-.5*9.81*t*t
        velocity = 0. if supported and z == 0 else -9.81*t
        shoe = {'body_position_m': [.4, .4, z], 'body_quaternion_wxyz': [1., 0., 0., 0.],
                'bottom_m': z, 'support_contacts': int(supported and z == 0),
                'other_contacts': 0, 'other_contact_geoms': [],
                'support_normal_force_n': 2. if supported and z == 0 else 0.,
                'linear_speed_m_s': abs(velocity), 'angular_speed_rad_s': 0., 'vertical_velocity_m_s': velocity}
        records.append({'t_seconds': t, 'shoes': {'trainer-left': shoe}})
    return records


class AssetBenchmarkTests(unittest.TestCase):
    def test_frozen_default_denominator_and_matched_controls(self):
        protocol = bench.make_protocol()
        self.assertEqual(protocol['planned_attempts'], 54)
        self.assertEqual(protocol['attempts'], bench.make_protocol()['attempts'])
        self.assertNotEqual(protocol['attempts'], bench.make_protocol(seed=612048)['attempts'])
        for first, second in zip(protocol['attempts'][::2], protocol['attempts'][1::2]):
            self.assertEqual(first['placements'], second['placements'])
            self.assertEqual((first['control'], second['control']), bench.CONTROLS)
        self.assertEqual(len({row['id'] for row in protocol['attempts']}), 54)
        self.assertEqual(protocol['catalog_sha256'], bench._hash(protocol['catalog']))

    def test_invalid_protocol_options_rejected_before_work(self):
        for options in ({'seed': True}, {'seed': -1}, {'seed': 10**20}, {'repeats': True},
                        {'repeats': 0}, {'repeats': 6}, {'source_commit': 'HEAD'}, {'source_commit': 42}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                bench.make_protocol(**options)

    def test_support_requires_contact_low_velocity_and_full_settling_window(self):
        stable = trace()
        self.assertTrue(bench._score_shoe(stable, 'trainer-left')['settled_supported'])
        for field, value in (('support_contacts', 0), ('support_normal_force_n', 0.),
                             ('bottom_m', -.01), ('linear_speed_m_s', .031), ('angular_speed_rad_s', .21)):
            changed = deepcopy(stable)
            changed[40]['shoes']['trainer-left'][field] = value
            self.assertFalse(bench._score_shoe(changed, 'trainer-left')['settled_supported'], field)
        self.assertFalse(bench._score_shoe(stable[:-1], 'trainer-left')['settled_supported'])

    def test_gravity_control_is_not_support_success(self):
        falling = bench._score_shoe(trace(False), 'trainer-left')
        self.assertTrue(falling['gravity_observed'])
        self.assertTrue(falling['dropped_without_support'])
        self.assertFalse(falling['settled_supported'])
        frozen = trace()
        frozen[2]['shoes']['trainer-left']['vertical_velocity_m_s'] = 0.
        self.assertFalse(bench._score_shoe(frozen, 'trainer-left')['gravity_observed'])

    def test_other_contact_stays_in_trace_and_fails_expected_outcome(self):
        samples = trace()
        samples[10]['shoes']['trainer-left']['other_contacts'] = 2
        samples[10]['shoes']['trainer-left']['other_contact_geoms'] = ['robot_fixture']
        score = bench._score_shoe(samples, 'trainer-left')
        self.assertTrue(score['settled_supported'])
        self.assertEqual((score['unexpected_contact_samples'], score['max_other_contacts']), (1, 2))
        attempt = {'plan': {'placements': [{'shoe_id': 'trainer-left'}], 'control': 'table_support'},
                   'trajectory': samples, 'error': None, 'warnings': [], 'native_warning_count': 0,
                   'simulated_seconds': 1., 'native_step_wall_seconds': [0.]*500, 'model_contract': {'passed': True}}
        scored = bench._scores(attempt)
        self.assertTrue(scored['valid'])
        self.assertTrue(scored['support_success'])
        self.assertTrue(scored['unexpected_contact_detected'])
        self.assertFalse(scored['expected_outcome'])

    def test_failure_retains_attempt_in_denominator(self):
        protocol = bench.make_protocol(repeats=1)
        from dvidia_training import asset_library
        with patch.object(asset_library, 'build_scene', side_effect=ValueError('fixture')):
            attempt = bench._attempt(protocol['attempts'][0], 'unused', protocol['catalog'])
        self.assertFalse(attempt['scores']['valid'])
        self.assertFalse(attempt['scores']['support_success'])
        self.assertEqual(attempt['error']['type'], 'ValueError')
        summary = bench.aggregate([attempt])
        self.assertEqual((summary[0]['attempted'], summary[0]['valid'], summary[0]['support_success']), (1, 0, 0))

    def test_timing_quantiles_are_derived_from_every_step(self):
        result = bench._percentiles([.001, .002, .003, .004, .005])
        self.assertEqual(result['count'], 5)
        self.assertAlmostEqual(result['p50_seconds'], .003)
        self.assertAlmostEqual(result['p99_seconds'], .00496)
        self.assertAlmostEqual(result['total_seconds'], .015)

    def test_bundle_inspection_rejects_resealed_scores_denominator_and_nan(self):
        from dvidia_training import asset_library
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'bundle'
            with patch.object(asset_library, 'build_scene', side_effect=ValueError('fixture')):
                report = bench.benchmark(dict.fromkeys(bench.ROBOTS, 'unused'), output, repeats=1)
            self.assertEqual((report['attempted'], bench.inspect_benchmark(output)['valid_attempts']), (18, 0))
            pristine = {path.name: path.read_bytes() for path in output.iterdir()}

            def restore():
                for name, content in pristine.items():
                    (output/name).write_bytes(content)

            def reseal():
                integrity = json.loads((output/'integrity.json').read_bytes())
                for name in bench.FILES:
                    data = (output/name).read_bytes()
                    integrity['files'][name] = {'sha256': sha256(data).hexdigest(), 'bytes': len(data)}
                (output/'integrity.json').write_bytes(bench._json(integrity))

            changed = json.loads(pristine['attempts.json'])
            changed['attempts'][0]['scores']['support_success'] = True
            (output/'attempts.json').write_bytes(bench._json(changed))
            changed_report = json.loads(pristine['report.json'])
            changed_report['attempts_sha256'] = bench._hash(changed)
            (output/'report.json').write_bytes(bench._json(changed_report))
            reseal()
            with self.assertRaisesRegex(ValueError, 'outcomes'):
                bench.inspect_benchmark(output)
            restore()
            changed = json.loads(pristine['protocol.json'])
            changed['attempts'].pop()
            changed['planned_attempts'] -= 1
            (output/'protocol.json').write_bytes(bench._json(changed))
            reseal()
            with self.assertRaisesRegex(ValueError, 'denominator'):
                bench.inspect_benchmark(output)
            restore()
            (output/'attempts.json').write_bytes(pristine['attempts.json'].replace(b'"simulated_seconds": 0.0', b'"simulated_seconds": NaN', 1))
            reseal()
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                bench.inspect_benchmark(output)
            restore()
            (output/'unexpected.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Unexpected'):
                bench.inspect_benchmark(output)


@unittest.skipUnless(os.environ.get('DVIDIA_ASSET_LIBRARY_ROOTS'), 'Set DVIDIA_ASSET_LIBRARY_ROOTS to JSON robot-to-assets paths')
class NativeAssetBenchmarkTests(unittest.TestCase):
    def test_three_arms_three_pairs_and_matched_no_support_controls(self):
        roots = json.loads(os.environ['DVIDIA_ASSET_LIBRARY_ROOTS'])
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/'native'
            report = bench.benchmark(roots, directory, seed=109731, repeats=1)
            inspected = bench.inspect_benchmark(directory)
            self.assertEqual((inspected['attempts'], inspected['valid_attempts']), (18, 18))
            for row in report['summary']:
                self.assertEqual(row['attempted'], 3)
                self.assertEqual(row['gravity_success'], 3)
                self.assertEqual(row['expected_outcome'], 3)
                self.assertEqual(row['unexpected_contact_attempts'], 0)
                self.assertEqual(row['support_success'], 3 if row['control'] == 'table_support' else 0)
            self.assertEqual(report['timing']['native_step']['count'], 9000)
            self.assertFalse(report['physical_robot_ready'])
            self.assertFalse(report['shoe_manipulation_qualified'])


if __name__ == '__main__':
    unittest.main()

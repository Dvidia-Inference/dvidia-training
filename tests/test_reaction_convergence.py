"""Timestep comparison measurements without importing/running native physics."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from dvidia_training.reaction_convergence import compare_timesteps, recompute_comparison


def protocol():
    return {'seed': 47, 'cases': [{'id': 'crossing'}, {'id': 'late_occluded_entry', 'onset_s': .45}],
            'fixture_config': {'timestep_s': .001, 'warmup_s': .4, 'horizon_s': 1.4,
                               'rest_dwell_s': .05, 'command_delay_s': .02},
            'sensor_period_s': .01, 'reaction_period_s': .001, 'sensor_prime_s': .15,
            'reaction_config': {'command_delay_s': .02},
            'convergence_diagnostics': {'event_time_absolute_s': .01,
                'contact_relative_change': .1, 'penetration_absolute_m': .001,
                'scope': 'authored development tolerances'}}


def record(config, case, variant, seed):
    return {'case_id': case['id'], 'variant': variant, 'seed': seed, 'valid': True, 'error': None,
            'detected_s': .5, 'stopped_s': .7 if variant == 'monitor' else None,
            'stopping_distance_m': .02 if variant == 'monitor' else None,
            'peak_unexpected_force_n': 10., 'unexpected_contact_impulse_ns': .2,
            'max_penetration_m': .001, 'minimum_clearance_m': -.001,
            'trace': [{'detail': 'complete records retained'}]}


class ConvergenceTests(unittest.TestCase):
    def test_eight_paired_attempts_fixed_clocks_and_detached_inputs(self):
        source = protocol()
        before = deepcopy(source)
        seen = []
        def run(config, case, variant, seed):
            seen.append((deepcopy(config), deepcopy(case), variant, seed))
            result = record(config, case, variant, seed)
            config['fixture_config']['timestep_s'] = .99  # Cannot mutate source or pretrial identities.
            case['id'] = 'mutated'
            return result
        with patch('dvidia_training.reaction_benchmark.run_trial', side_effect=run):
            report = compare_timesteps(source)
        self.assertEqual(source, before)
        self.assertEqual((report['attempted'], report['invalid'], len(report['pairs'])), (8, 0, 4))
        self.assertTrue(report['timing_stable'])
        self.assertTrue(report['contact_converged'])
        self.assertFalse(report['physical_robot_ready'])
        for first, second in zip(seen[::2], seen[1::2]):
            self.assertEqual(first[3], second[3])
            self.assertEqual(first[0]['sensor_period_s'], second[0]['sensor_period_s'])
            self.assertEqual(first[0]['reaction_period_s'], second[0]['reaction_period_s'])
            self.assertEqual((first[0]['fixture_config']['timestep_s'], second[0]['fixture_config']['timestep_s']), (.001, .0005))
        for base, fine in zip(report['attempts'][::2], report['attempts'][1::2]):
            self.assertNotEqual(base['fixture_config_sha256'], fine['fixture_config_sha256'])
            self.assertEqual(base['record']['trace'][0]['detail'], 'complete records retained')
        json.dumps(report, allow_nan=False)

    def test_absolute_and_signed_baseline_relative_changes(self):
        def run(config, case, variant, seed):
            result = record(config, case, variant, seed)
            if config['fixture_config']['timestep_s'] == .0005:
                result.update(detected_s=.505, peak_unexpected_force_n=10.5,
                              unexpected_contact_impulse_ns=.21, minimum_clearance_m=-.0015)
            return result
        with patch('dvidia_training.reaction_benchmark.run_trial', side_effect=run):
            report = compare_timesteps(protocol())
        pair = report['pairs'][0]
        self.assertAlmostEqual(pair['metrics']['detected_s']['absolute_change'], .005)
        self.assertAlmostEqual(pair['metrics']['minimum_clearance_m']['relative_change_to_abs_base'], .5)
        self.assertTrue(pair['timing_stable'])
        self.assertTrue(pair['contact_converged'])

    def test_zero_baseline_not_converted_to_fabricated_relative_zero(self):
        def run(config, case, variant, seed):
            result = record(config, case, variant, seed)
            result['peak_unexpected_force_n'] = 0. if config['fixture_config']['timestep_s'] == .001 else 1.
            result['unexpected_contact_impulse_ns'] = 0.
            return result
        with patch('dvidia_training.reaction_benchmark.run_trial', side_effect=run):
            pair = compare_timesteps(protocol())['pairs'][0]
        force, impulse = pair['metrics']['peak_unexpected_force_n'], pair['metrics']['unexpected_contact_impulse_ns']
        self.assertEqual(force['relative_status'], 'zero_to_nonzero')
        self.assertIsNone(force['relative_change_to_abs_base'])
        self.assertEqual(impulse['relative_status'], 'both_zero')
        self.assertIsNone(impulse['relative_change_to_abs_base'])
        self.assertTrue(pair['contact_checks']['unexpected_contact_impulse_ns'])
        self.assertFalse(pair['contact_converged'])

    def test_missing_measurements_remain_unknown_or_asymmetric_failure(self):
        def run(config, case, variant, seed):
            result = record(config, case, variant, seed)
            result['unexpected_contact_impulse_ns'] = None
            result['stopped_s'] = None
            if config['fixture_config']['timestep_s'] == .0005:
                result['detected_s'] = None
            return result
        with patch('dvidia_training.reaction_benchmark.run_trial', side_effect=run):
            pair = compare_timesteps(protocol())['pairs'][0]
        self.assertFalse(pair['timing_stable'])
        self.assertIsNone(pair['contact_converged'])
        self.assertEqual(pair['metrics']['stopped_s']['relative_status'], 'both_missing')
        self.assertIsNone(pair['metrics']['stopped_s']['absolute_change'])

    def test_invalid_trials_and_exceptions_are_retained_in_denominator(self):
        def run(config, case, variant, seed):
            if config['fixture_config']['timestep_s'] == .0005:
                raise RuntimeError('native fixture failed')
            return record(config, case, variant, seed)
        with patch('dvidia_training.reaction_benchmark.run_trial', side_effect=run):
            report = compare_timesteps(protocol())
        self.assertEqual((report['attempted'], report['invalid']), (8, 4))
        self.assertIsNone(report['timing_stable'])
        self.assertIsNone(report['contact_converged'])
        self.assertTrue(all(pair['errors'][0]['error'] == 'RuntimeError: native fixture failed' for pair in report['pairs']))
        self.assertIsNone(report['pairs'][0]['metrics']['peak_unexpected_force_n']['fine'])

    def test_invalid_record_placeholders_are_not_used_as_measurements(self):
        def run(config, case, variant, seed):
            result = record(config, case, variant, seed)
            if config['fixture_config']['timestep_s'] == .0005:
                result.update(valid=False, error='failed before measurement', peak_unexpected_force_n=0.)
            return result
        with patch('dvidia_training.reaction_benchmark.run_trial', side_effect=run):
            report = compare_timesteps(protocol())
        self.assertEqual(report['attempts'][1]['record']['peak_unexpected_force_n'], 0.)
        self.assertIsNone(report['pairs'][0]['metrics']['peak_unexpected_force_n']['fine'])
        self.assertIsNone(report['pairs'][0]['contact_converged'])

    def test_authored_tolerance_failures_are_visible(self):
        def run(config, case, variant, seed):
            result = record(config, case, variant, seed)
            if config['fixture_config']['timestep_s'] == .0005:
                result.update(detected_s=.52, max_penetration_m=.003)
            return result
        with patch('dvidia_training.reaction_benchmark.run_trial', side_effect=run):
            report = compare_timesteps(protocol())
        self.assertFalse(report['timing_stable'])
        self.assertFalse(report['contact_converged'])

    def test_configuration_is_validated_before_attempts(self):
        for mutate in (lambda p: p.pop('convergence_diagnostics'),
                       lambda p: p.update(reaction_period_s=.0007),
                       lambda p: p['fixture_config'].update(horizon_s=1.4001),
                       lambda p: p['cases'].pop(),
                       lambda p: p['cases'][0].update(sensor_delay_s=.0007)):
            value = protocol()
            mutate(value)
            with patch('dvidia_training.reaction_benchmark.run_trial') as run, self.assertRaises(ValueError):
                compare_timesteps(value)
            run.assert_not_called()

    def test_pure_recomputation_detects_rewritten_verdict_and_metrics(self):
        def run(config, case, variant, seed):
            result = record(config, case, variant, seed)
            if config['fixture_config']['timestep_s'] == .0005:
                result.update(detected_s=.52, max_penetration_m=.003)
            return result
        source = protocol()
        with patch('dvidia_training.reaction_benchmark.run_trial', side_effect=run):
            original = compare_timesteps(source)
        altered = deepcopy(original)
        altered['timing_stable'] = altered['contact_converged'] = True
        altered['pairs'][0]['metrics']['max_penetration_m']['absolute_change'] = 0.
        altered['pairs'][0]['contact_converged'] = True
        with patch('dvidia_training.reaction_benchmark.run_trial') as native:
            recalculated = recompute_comparison(source, altered['attempts'])
        native.assert_not_called()
        self.assertEqual(recalculated, original)
        self.assertNotEqual(recalculated, altered)
        self.assertFalse(recalculated['contact_converged'])

    def test_missing_duplicate_extra_and_altered_attempt_identities_rejected(self):
        source = protocol()
        with patch('dvidia_training.reaction_benchmark.run_trial', side_effect=record):
            attempts = compare_timesteps(source)['attempts']
        mutations = [lambda rows: rows.pop(), lambda rows: rows.append(deepcopy(rows[0])),
                     lambda rows: rows.__setitem__(1, deepcopy(rows[0])),
                     lambda rows: rows[0].update(seed=48),
                     lambda rows: rows[0].update(protocol_sha256='0'*64),
                     lambda rows: rows[0].update(fixture_config_sha256='0'*64),
                     lambda rows: rows[0].update(input_sha256='0'*64),
                     lambda rows: rows[0]['record'].update(case_id='other'),
                     lambda rows: rows[0]['record'].update(seed=48),
                     lambda rows: rows[0].update(timestep_s=.00025)]
        for mutate in mutations:
            altered = deepcopy(attempts)
            mutate(altered)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                recompute_comparison(source, altered)

    def test_all_invalid_attempts_retained_and_reordering_is_canonical(self):
        source = protocol()
        with patch('dvidia_training.reaction_benchmark.run_trial', side_effect=RuntimeError('native unavailable')):
            original = compare_timesteps(source)
        reordered = list(reversed(original['attempts']))
        before = deepcopy(reordered)
        recalculated = recompute_comparison(source, reordered)
        self.assertEqual(reordered, before)
        self.assertEqual(recalculated, original)
        self.assertEqual((recalculated['attempted'], recalculated['invalid']), (8, 8))
        self.assertIsNone(recalculated['timing_stable'])
        self.assertIsNone(recalculated['contact_converged'])
        self.assertEqual(len(recalculated['pairs']), 4)
        self.assertTrue(all(len(pair['errors']) == 2 for pair in recalculated['pairs']))


if __name__ == '__main__':
    unittest.main()

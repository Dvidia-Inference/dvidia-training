"""Evidence corruption and scientific-denominator regression tests."""
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dvidia_training import local_goal_benchmark as bench
from dvidia_training.local_goal_verifier import GoalContract, verify_goal
from tests.test_local_goal_verifier import packet

GOAL = GoalContract('green-shoe', 'workstation', (.4, -.1), 0.)


def case(identity, complete, expected, cohort='native'):
    return dict(id=identity, cohort=cohort, scenario=identity, physical_truth=dict(present=True, complete=complete),
                expected_verdict=expected, generation_valid=True)


def row(identity, verdict):
    return dict(case_id=identity, decision=dict(verdict=verdict), timing_ns=[100, 200, 300])


class BenchmarkTests(unittest.TestCase):
    def test_all_predeclared_faults_fail_closed_without_mutating_source(self):
        original = packet()
        for name in bench.FAULTS:
            with self.subTest(fault=name):
                changed, now, expected = bench.fault_packet(original, name, .6)
                self.assertEqual(verify_goal(changed, contract=GOAL, now_s=now)['verdict'], expected)
        self.assertEqual(original, packet())
        self.assertEqual(len(bench.FAULTS), 16)

    def test_consistent_forgery_counts_as_physical_false_completion(self):
        cases = [case('native-positive', True, 'complete'), case('native-negative', False, 'incomplete'),
                 case('stale', True, 'unknown', 'evidence_fault'), case('forged', False, 'complete', 'blind_spot')]
        rows = [row(c['id'], c['expected_verdict']) for c in cases]
        metrics = bench.score(rows, cases)
        self.assertEqual(metrics['overall']['expected_verdict_matches']['rate'], 1.)
        self.assertEqual(metrics['overall']['physical_false_completion'], bench.fraction(1, 2))
        self.assertEqual(metrics['overall']['physical_true_completion'], bench.fraction(1, 2))
        self.assertEqual(metrics['by_cohort']['blind_spot']['physical_false_completion'], bench.fraction(1, 1))
        self.assertEqual(metrics['by_cohort']['evidence_fault']['abstention']['rate'], 1.)
        self.assertEqual(metrics['overall']['latency_us']['samples'], 12)

    def test_missing_duplicates_unknown_results_and_invalid_timings_rejected(self):
        cases = [case('one', True, 'complete')]
        for rows in ([], [row('one', 'complete')]*2, [row('other', 'complete')]):
            with self.assertRaises(ValueError): bench.score(rows, cases)
        r = row('one', 'complete'); r['timing_ns'] = [True]
        with self.assertRaises(ValueError): bench.score([r], cases)
        r['timing_ns'] = [float('nan')]
        with self.assertRaises(ValueError): bench.score([r], cases)
        self.assertIsNone(bench.score([], [])['overall']['physical_false_completion']['rate'])

    def test_generation_invalid_is_not_removed(self):
        c = case('bad-generation', False, 'unknown'); c['generation_valid'] = False
        m = bench.score([row(c['id'], 'unknown')], [c])['overall']
        self.assertEqual(m['planned'], 1)
        self.assertEqual(m['generation_invalid'], 1)
        self.assertEqual(m['physical_false_completion']['denominator'], 1)

    def test_input_seal_detects_tampering_and_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            bench.write_json(root/'packet.json', packet())
            bench.write_json(root/'integrity.json', {'schema':'dvidia.local-goal-integrity/v1', 'files': {
                'packet.json': {'sha256':bench.sha(root/'packet.json'), 'bytes':(root/'packet.json').stat().st_size}}})
            bench.verify_integrity(root)
            (root/'packet.json').write_text('{}')
            with self.assertRaises(ValueError): bench.verify_integrity(root)
            for name in ('../packet.json', '/packet.json', 'a//b', 'a/./b'):
                with self.assertRaises(ValueError): bench.relative_file(root, name)

    def test_hard_native_failure_seals_diagnostics_without_partial_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp).resolve()/'failed-run'
            failed = {'id':'broken', 'packet_file':None, 'physical_truth':None, 'now_s':None}
            with patch.object(bench, 'source_identity', return_value={'commit':'test','files':{}}), patch('dvidia_training.local_goal_scenes.generate_native', return_value=[failed]):
                with self.assertRaisesRegex(RuntimeError, 'sealed diagnostics'):
                    bench.generate('/unused', output, count=1)
            self.assertEqual(bench.read_json(output/'generation.json')['status'], 'generation_failed')
            self.assertEqual(bench.read_json(output/'cases.json')['cases'], [failed])
            self.assertTrue((output/'integrity.json').is_file())
            self.assertFalse((output/'results.json').exists())
            with self.assertRaisesRegex(ValueError, 'failed native generation'):
                bench.verify_integrity(output)

    def test_unsealed_physical_references_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            bench.write_json(root/'cases.json', {'cases':[{'packet_file':'p.json', 'state_file':'s.json', 'current_image':'x.png', 'goal_image':'g.png'}]})
            bench.write_json(root/'integrity.json', {'schema':'dvidia.local-goal-integrity/v1', 'files': {
                'cases.json': {'sha256':bench.sha(root/'cases.json'), 'bytes':(root/'cases.json').stat().st_size}}})
            with self.assertRaisesRegex(ValueError, 'not sealed'):
                bench.verify_integrity(root)

    def test_controls_retain_physical_truth_and_native_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            bench.write_json(root/'packets/complete.json', packet())
            c = case('layout-complete', True, 'complete')
            c.update(layout_id='layout', scenario='complete', packet_file='packets/complete.json',
                     contract=asdict(GOAL), now_s=.6, current_image='complete.png', state_file='complete.json')
            moving = dict(c, id='layout-moving', scenario='moving_alias', physical_truth=dict(present=True, complete=False),
                          expected_verdict='incomplete', state_file='moving.json')
            cases = bench.add_controls(root, [c, moving])
            self.assertEqual(len(cases), 19)
            forged = cases[-1]
            self.assertFalse(forged['physical_truth']['complete'])
            self.assertEqual(forged['state_file'], 'moving.json')
            self.assertEqual(forged['expected_verdict'], 'complete')
            self.assertTrue(all(r['physical_truth']['complete'] for r in cases if r['cohort']=='evidence_fault'))
            self.assertEqual(verify_goal(bench.read_json(root/forged['packet_file']), contract=GOAL, now_s=.6)['verdict'], 'complete')


if __name__ == '__main__':
    unittest.main()

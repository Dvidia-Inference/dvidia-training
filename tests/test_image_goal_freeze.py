"""Freeze and offline execution tests; no native simulator or network required."""
from contextlib import redirect_stdout
from hashlib import sha256
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zlib

from dvidia_training import image_goal_benchmark as bench
from dvidia_training import jev_client as jev


def png(width=320, height=280):
    def chunk(kind, value):
        return (struct.pack('>I', len(value)) + kind + value
                + struct.pack('>I', zlib.crc32(kind + value) & 0xffffffff))
    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress((b'\0' + b'\0' * width * 3) * height))
            + chunk(b'IEND', b''))


def fixtures(directory):
    root = Path(directory) / 'fixtures'
    root.mkdir()
    (root / 'current.png').write_bytes(png())
    (root / 'goal.png').write_bytes(png(321))
    cases = []
    for name, present in [('present-case', True), ('absent-case', False)]:
        cases.append({'id': name, 'category': 'unit-fixture',
            'current_image': 'current.png', 'goal_image': 'goal.png',
            'state_file': f'{name}-state.json',
            'physical_truth': {'present': present, 'complete': False},
            'measured_inputs': {'observed_contact_force_n': .12 if present else 0.},
            'expected_verdict': {condition: 'incomplete' for condition in bench.CONDITIONS},
            'expected_presence': {condition: 'present' if present else 'absent' for condition in bench.CONDITIONS}})
    for case in cases:
        (root / case['state_file']).write_bytes(bench.json_bytes(
            {'scope': 'unit-test-only', 'native_contact_history_n': [.12, .12]}))
    (root / 'cases.json').write_bytes(bench.json_bytes({'cases': cases}))
    (root / 'protocol.json').write_bytes(bench.json_bytes({'fixture_type': 'unit-test-only'}))
    (root / 'protocol.sha256').write_text(sha256((root / 'protocol.json').read_bytes()).hexdigest() + '\n')
    entries = [{'path': path.name, **bench.digest(path)} for path in sorted(root.iterdir())]
    (root / 'integrity.json').write_bytes(bench.json_bytes({'files': entries}))
    return root, cases


def seal(root):
    return bench.freeze(root, task='Place the target shoe in the goal zone.', source_commit='a' * 40)


class ImageGoalFreezeTests(unittest.TestCase):
    def test_freeze_covers_every_paired_trial_and_digest_matches_exact_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root, cases = fixtures(directory)
            plan = seal(root)
            self.assertEqual(len(plan['trials']), len(cases) * len(bench.CONDITIONS))
            self.assertEqual({(row['case_id'], row['condition']) for row in plan['trials']},
                             {(case['id'], c) for case in cases for c in bench.CONDITIONS})
            self.assertEqual(set(plan['files']), {'current.png', 'goal.png', 'cases.json', 'protocol.json',
                'protocol.sha256', 'integrity.json', 'present-case-state.json', 'absent-case-state.json'})
            self.assertEqual(plan['automatic_retries'], 0)
            by_id = {case['id']: case for case in cases}
            for row in plan['trials']:
                case = by_id[row['case_id']]
                prepared = jev.build_goal_request(
                    (root / case['current_image']).read_bytes(), (root / case['goal_image']).read_bytes(),
                    task=plan['task'], measured_state=case['measured_inputs'] if row['condition'] == 'rgb_telemetry' else None)
                self.assertEqual(prepared.request_sha256, row['request_sha256'])
                serialized = json.dumps(prepared.request)
                self.assertNotIn(case['id'], serialized)
                self.assertNotIn('physical_truth', serialized)
                self.assertNotIn('expected_verdict', serialized)
            verified, verified_cases = bench.verify_plan(root)
            self.assertEqual(verified, plan)
            self.assertEqual(verified_cases, cases)

    def test_mutating_any_frozen_evidence_file_is_rejected(self):
        for name in ('current.png', 'goal.png', 'cases.json', 'protocol.json',
                     'protocol.sha256', 'integrity.json', 'present-case-state.json', 'absent-case-state.json'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root, _ = fixtures(directory)
                seal(root)
                with (root / name).open('ab') as stream:
                    stream.write(b' ')
                with self.assertRaises(ValueError):
                    bench.verify_plan(root)

    def test_duplicate_missing_trial_or_late_request_mutation_rejected_before_calls(self):
        for mutation in ('duplicate', 'missing', 'late_digest', 'task'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root, _ = fixtures(directory)
                plan = seal(root)
                if mutation == 'duplicate':
                    plan['trials'][-1] = plan['trials'][0]
                elif mutation == 'missing':
                    plan['trials'].pop()
                elif mutation == 'late_digest':
                    plan['trials'][-1]['request_sha256'] = '0' * 64
                else:
                    plan['task'] = 'A modified contract after freezing.'
                (root / 'evaluation-plan.json').write_bytes(bench.json_bytes(plan))
                with patch.object(jev.JevClient, 'evaluate') as provider:
                    with self.assertRaises(ValueError), redirect_stdout(io.StringIO()):
                        bench.evaluate(root, Path(directory) / 'results', api_key='test-not-real')
                    provider.assert_not_called()

    def test_freeze_and_results_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = fixtures(directory)
            seal(root)
            original = (root / 'evaluation-plan.json').read_bytes()
            with self.assertRaises(FileExistsError):
                seal(root)
            self.assertEqual((root / 'evaluation-plan.json').read_bytes(), original)
            output = Path(directory) / 'results'
            output.mkdir()
            (output / 'keep.txt').write_text('existing result')
            with self.assertRaises(FileExistsError):
                bench.evaluate(root, output, api_key='')
            self.assertEqual((output / 'keep.txt').read_text(), 'existing result')

    def test_interruption_preserves_full_scheduled_inventory_and_unscored_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = fixtures(directory)
            plan = seal(root)
            response = {'status': 'ok', 'model': jev.MODEL, 'attempts': 1,
                        'presence': 'present', 'verdict': 'unknown',
                        'p_present': .8, 'p_complete': .5,
                        'latency_ms': {'prepare': .1, 'http': 5., 'total': 5.1},
                        'estimated_input_cost_usd': .0001}
            output = Path(directory) / 'results'
            with patch.object(jev.JevClient, 'evaluate', side_effect=[response, KeyboardInterrupt()]):
                with self.assertRaises(KeyboardInterrupt), redirect_stdout(io.StringIO()):
                    bench.evaluate(root, output, api_key='not-a-real-key')
            report = json.loads((output / 'results.json').read_text())
            self.assertEqual(len(report['results']), len(plan['trials']))
            self.assertEqual({(row['case_id'], row['condition']) for row in report['results']},
                             {(row['case_id'], row['condition']) for row in plan['trials']})
            self.assertIsNone(report['metrics'])
            self.assertEqual(sum(row['status'] == 'ok' for row in report['results']), 1)
            self.assertNotEqual(report['status'], 'completed')
            self.assertEqual(report['active_trial']['case_id'], plan['trials'][1]['case_id'])
            self.assertEqual(report['active_trial']['condition'], plan['trials'][1]['condition'])
            self.assertIn('charge unknown', report['active_trial']['outcome'])
            self.assertNotIn('not-a-real-key', (output / 'results.json').read_text())

    def test_absent_credentials_keep_all_trials_without_model_scores_or_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root, cases = fixtures(directory)
            plan = seal(root)
            with patch.dict('os.environ', {}, clear=True), patch.object(jev, 'build_opener') as opener:
                opener.return_value.open.side_effect = AssertionError('Network must not be called')
                with redirect_stdout(io.StringIO()):
                    report = bench.evaluate(root, Path(directory) / 'results')
                opener.return_value.open.assert_not_called()
            self.assertEqual(report['status'], 'not_run')
            self.assertIsNone(report['active_trial'])
            self.assertIsNone(report['metrics'])
            self.assertEqual(len(report['results']), len(plan['trials']))
            for row in report['results']:
                self.assertEqual(row['status'], 'not_run')
                self.assertIsNone(row['p_complete'])
                self.assertIsNone(row['p_present'])
                self.assertIsNone(row['online_total_ms'])
                self.assertIsNone(row['estimated_input_cost_usd'])
                self.assertEqual(row['attempts'], 0)
            output = Path(directory) / 'results'
            self.assertEqual(json.loads((output / 'results.json').read_text()), report)
            ledger = [json.loads(line) for line in (output / 'attempts.jsonl').read_text().splitlines()]
            self.assertEqual(ledger, report['results'])
            self.assertEqual((output / 'evaluation-plan.json').read_bytes(), (root / 'evaluation-plan.json').read_bytes())


if __name__ == '__main__':
    unittest.main()

"""Offline CLI isolation, fresh outputs and untrusted report embedding."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from dvidia_training import embodiment_cli as cli
from dvidia_training import reference_sim as sim


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/paired_fidelity.json'


class EmbodimentCLITests(unittest.TestCase):
    def invoke(self, arguments):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(arguments)
        return code, out.getvalue(), err.getvalue()

    def test_top_and_subcommand_help_need_no_optional_runtime_or_site_packages(self):
        code = '''import builtins,sys
sys.path.insert(0,sys.argv[1])
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in ('numpy','mujoco','cv2'):
        raise RuntimeError('Optional runtime imported during help')
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from dvidia_training.embodiment_cli import main
main(sys.argv[2:])
'''
        for command in ([], ['benchmark'], ['score-paired'], ['fetch-assets'], ['inspect'], ['export-replay']):
            with self.subTest(command=command):
                result = subprocess.run([sys.executable, '-S', '-c', code, str(ROOT / 'src'), *command, '--help'],
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)

    def test_paired_scoring_runs_with_standard_library_and_preserves_fresh_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'score.json'
            code = '''import sys
sys.path.insert(0,sys.argv[1])
from dvidia_training.embodiment_cli import main
result=main(['score-paired',sys.argv[2],'--output',sys.argv[3]])
assert not any(name.split('.')[0] in ('numpy','mujoco','cv2') for name in sys.modules)
raise SystemExit(result)
'''
            result = subprocess.run([sys.executable, '-S', '-c', code, str(ROOT / 'src'),
                                     str(FIXTURE), str(output)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(report['sample_count'], 3)
            self.assertFalse(report['physical_robot_ready'])
            self.assertEqual(report['input_provenance'], 'synthetic_fixture')
            self.assertEqual(json.loads(result.stdout), report)
            saved = output.read_bytes()
            code, _, error = self.invoke(['score-paired', str(FIXTURE), '--output', str(output)])
            self.assertEqual(code, 1)
            self.assertIn('DVIDIA reference:', error)
            self.assertEqual(output.read_bytes(), saved)

    def test_invalid_paired_input_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid = root / 'invalid.json'
            invalid.write_text('{"schema":"x","schema":"y"}', encoding='utf-8')
            output = root / 'report.json'
            code, out, error = self.invoke(['score-paired', str(invalid), '--output', str(output)])
            self.assertEqual(code, 1)
            self.assertEqual(out, '')
            self.assertIn('Duplicate JSON key', error)
            self.assertFalse(output.exists())

    def test_verify_and_fetch_propagate_explicit_hands_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            assets = Path(directory) / 'assets'
            payload = {'commit': sim.COMMIT, 'files_verified': 2}
            with patch.object(sim, 'verify_assets', return_value=payload) as verify:
                code, out, error = self.invoke(['verify-assets', str(assets), '--hands'])
                self.assertEqual((code, error), (0, ''))
                self.assertEqual(json.loads(out), payload)
                verify.assert_called_once_with(assets, hands=True)
            with patch.object(sim, 'fetch_assets', return_value=payload) as fetch:
                code, _, error = self.invoke(['fetch-assets', '--output', str(assets)])
                self.assertEqual((code, error), (0, ''))
                fetch.assert_called_once_with(assets, hands=False)

    def test_audit_optional_output_is_exclusive_and_utf8(self):
        with tempfile.TemporaryDirectory() as directory:
            assets = Path(directory) / 'assets'
            output = Path(directory) / 'audit.json'
            payload = {'scope': 'DVIDIA — geometry Δ', 'physical_robot_ready': False}
            with patch.object(sim, 'audit_reference', return_value=payload) as audit:
                code, out, error = self.invoke(['audit', str(assets), '--output', str(output)])
                self.assertEqual((code, error), (0, ''))
                self.assertEqual(json.loads(out), payload)
                self.assertEqual(json.loads(output.read_text(encoding='utf-8')), payload)
                audit.assert_called_once_with(assets, hands=False)
            original = output.read_bytes()
            with patch.object(sim, 'audit_reference', return_value=payload):
                code, _, _ = self.invoke(['audit', str(assets), '--output', str(output)])
            self.assertEqual(code, 1)
            self.assertEqual(output.read_bytes(), original)

    def test_benchmark_invalid_protocol_is_rejected_before_reserving_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for repeats, seconds in ((0, .1), (6, .1), (1, .11), (1, 0.), (1, 11.), (1, float('nan'))):
                output = root / 'not-created'
                with self.subTest(repeats=repeats, seconds=seconds), \
                        patch.object(sim, 'verify_assets', return_value={}), \
                        patch.object(sim, 'benchmark_reference') as benchmark:
                    code, _, error = self.invoke(['benchmark', str(root / 'assets'), '--output', str(output),
                                                 '--repeats', str(repeats), '--seconds', str(seconds)])
                    self.assertEqual(code, 1)
                    self.assertIn('DVIDIA reference:', error)
                    self.assertFalse(output.exists())
                    benchmark.assert_not_called()

    def test_benchmark_asset_failure_does_not_reserve_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'not-created'
            with patch.object(sim, 'verify_assets', side_effect=ValueError('Asset digest mismatch')), \
                    patch.object(sim, 'benchmark_reference') as benchmark:
                code, _, error = self.invoke(['benchmark', str(root / 'assets'), '--output', str(output)])
                self.assertEqual(code, 1)
                self.assertIn('Asset digest mismatch', error)
                self.assertFalse(output.exists())
                benchmark.assert_not_called()

    def test_benchmark_preserves_failed_attempt_denominator_and_existing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'evidence'
            report = {'attempted': 9, 'valid_attempts': 8, 'physical_robot_ready': False}
            with patch.object(sim, 'verify_assets', return_value={}), \
                    patch.object(sim, 'benchmark_reference', return_value=report) as benchmark:
                code, out, error = self.invoke(['benchmark', str(root / 'assets'), '--output', str(output),
                                              '--repeats', '1', '--seconds', '.1'])
                self.assertEqual((code, error), (1, ''))
                self.assertEqual(json.loads(out)['valid_attempts'], 8)
                self.assertEqual(json.loads((output / 'report.json').read_text(encoding='utf-8')), report)
                self.assertTrue((output / 'report.html').is_file())
                benchmark.assert_called_once_with(root / 'assets', repeats=1, seconds=.1, hands=False)
            original = (output / 'report.json').read_bytes()
            with patch.object(sim, 'verify_assets', return_value={}), \
                    patch.object(sim, 'benchmark_reference') as benchmark:
                code, _, _ = self.invoke(['benchmark', str(root / 'assets'), '--output', str(output)])
                self.assertEqual(code, 1)
                benchmark.assert_not_called()
            self.assertEqual((output / 'report.json').read_bytes(), original)

    def test_report_embedding_escapes_markup_and_declares_offline_scope(self):
        report = {'scope': '</script><script>window.INJECTED=true</script>', 'physical_robot_ready': False}
        html = cli._report_html(report)
        self.assertNotIn('</script><script>window.INJECTED', html)
        self.assertNotIn('__REFERENCE_DATA__', html)
        self.assertIn('DVIDIA', html)
        self.assertNotIn('src="https:', html)
        embedded = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)
        if embedded is None:
            embedded = re.search(r'<script[^>]*type="application/json"[^>]*>(.*?)</script>', html, re.S)
        self.assertIsNotNone(embedded)
        self.assertEqual(json.loads(embedded.group(1)), report)
        with self.assertRaises(ValueError):
            cli._report_html({'invalid': float('nan')})


class ReferenceReceiptTests(unittest.TestCase):
    def make_bundle(self, directory, *, report=None):
        report = report or {'schema': 'dvidia.reference-benchmark/v1', 'physical_robot_ready': False,
                            'attempted': 2, 'valid_attempts': 1,
                            'attempts': [{'valid': True}, {'valid': False}]}
        cli._write_json(directory / 'report.json', report)
        (directory / 'report.html').write_text(cli._report_html(report), encoding='utf-8')
        cli._seal(directory)
        return report

    def test_inspect_reads_only_stdlib_and_does_not_rewrite_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_bundle(root)
            original = {name: (root / name).read_bytes() for name in ('report.json', 'report.html', 'integrity.json')}
            code = '''import sys
sys.path.insert(0,sys.argv[1])
from dvidia_training.embodiment_cli import main
result=main(['inspect',sys.argv[2]])
assert not any(name.split('.')[0] in ('numpy','mujoco','cv2') for name in sys.modules)
raise SystemExit(result)
'''
            result = subprocess.run([sys.executable, '-S', '-c', code, str(ROOT / 'src'), str(root)],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads(result.stdout)
            self.assertEqual((receipt['files_verified'], receipt['attempted'], receipt['valid_attempts']), (2, 2, 1))
            self.assertFalse(receipt['physical_robot_ready'])
            self.assertIn('not authenticity or metric recomputation', receipt['scope'])
            self.assertEqual(original, {name: (root / name).read_bytes() for name in original})
            with self.assertRaises(FileExistsError):
                cli._seal(root)
            self.assertEqual(original, {name: (root / name).read_bytes() for name in original})

    def test_changed_or_truncated_report_artifacts_are_rejected(self):
        for name, mutation in (('report.json', 'same-size'), ('report.html', 'truncate')):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.make_bundle(root)
                path = root / name
                raw = path.read_bytes()
                path.write_bytes((b'X' + raw[1:]) if mutation == 'same-size' else raw[:-1])
                with self.assertRaisesRegex(ValueError, 'digest/size'):
                    cli.inspect_bundle(root)

    def test_manifest_unknown_schema_keys_files_and_declarations_are_rejected(self):
        mutations = (
            lambda value: value.update(schema='dvidia.reference-integrity/v2'),
            lambda value: value.update(extra='untrusted'),
            lambda value: value['files'].update({'../outside.json': {'bytes': 1, 'sha256': '0' * 64}}),
            lambda value: value['files'].pop('report.html'),
            lambda value: value['files']['report.json'].update(extra=True),
            lambda value: value['files']['report.json'].update(bytes=True),
            lambda value: value['files']['report.json'].update(sha256='not-a-digest'),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.make_bundle(root)
                manifest = json.loads((root / 'integrity.json').read_text(encoding='utf-8'))
                mutate(manifest)
                (root / 'integrity.json').write_text(json.dumps(manifest), encoding='utf-8')
                with self.assertRaises(ValueError):
                    cli.inspect_bundle(root)

    def test_nonfinite_duplicate_non_utf8_and_deep_manifest_json_are_rejected(self):
        for raw in (b'{"schema":"x","schema":"y"}', b'{"x":NaN}', b'\xff',
                    b'[' * 1800 + b'0' + b']' * 1800):
            with self.subTest(raw=raw[:30]), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.make_bundle(root)
                (root / 'integrity.json').write_bytes(raw)
                with self.assertRaises(ValueError):
                    cli.inspect_bundle(root)

    def test_resealed_false_readiness_schema_and_inconsistent_counts_are_rejected(self):
        mutations = (
            lambda value: value.update(physical_robot_ready=True),
            lambda value: value.update(schema='hardware-qualified'),
            lambda value: value.update(attempted=3),
            lambda value: value.update(valid_attempts=2),
            lambda value: value.update(valid_attempts=True),
            lambda value: value['attempts'][0].update(valid=1),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                report = {'schema': 'dvidia.reference-benchmark/v1', 'physical_robot_ready': False,
                          'attempted': 2, 'valid_attempts': 1,
                          'attempts': [{'valid': True}, {'valid': False}]}
                mutate(report)
                self.make_bundle(root, report=report)
                with self.assertRaises(ValueError):
                    cli.inspect_bundle(root)

    def test_missing_oversize_and_redirected_receipts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / 'bundle'
            bundle.mkdir()
            self.make_bundle(bundle)
            with self.assertRaises((OSError, ValueError)):
                cli.inspect_bundle(root / 'absent')
            html = bundle / 'report.html'
            original = html.read_bytes()
            with html.open('wb') as stream:
                stream.truncate(16 * 1024 * 1024 + 1)
            with self.assertRaises(ValueError):
                cli.inspect_bundle(bundle)
            html.write_bytes(original)
            target = root / 'real-report.html'
            target.write_bytes(original)
            html.unlink()
            try:
                html.symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest('This host does not permit symlink creation')
            with self.assertRaises(ValueError):
                cli.inspect_bundle(bundle)
            root_link = root / 'redirected-bundle'
            root_link.symlink_to(bundle, target_is_directory=True)
            with self.assertRaises(ValueError):
                cli.inspect_bundle(root_link)


if __name__ == '__main__':
    unittest.main()

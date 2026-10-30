"""Offline identity, omission, and unknown-outcome packaging boundaries."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest

from publishing import prepare_grip_wait_recovery_compact_v04 as builder


REPO = Path(__file__).resolve().parents[1]


class CompactRecoveryBuildTests(unittest.TestCase):
    def test_payload_inventory_and_exact_original_bytes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            hf, research = root / 'hf', root / 'research'
            receipt = builder.build(REPO, hf, research)
            for output in (hf, research):
                inventory = json.loads((output / 'integrity.json').read_text())['files']
                actual = {p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}
                self.assertEqual(actual, set(inventory) | {'integrity.json'})
                self.assertLessEqual(len(actual), 48)
                self.assertLessEqual(sum(p.stat().st_size for p in output.rglob('*') if p.is_file()), 4 * 1024 * 1024)
                for name, identity in inventory.items():
                    self.assertEqual(builder._identity((output / name).read_bytes()), identity)
                for name, digest in builder.EXPECTED.items():
                    self.assertEqual(sha256((output / name).read_bytes()).hexdigest(), digest)
                    self.assertEqual((output / name).read_bytes(), (REPO / name).read_bytes())
                self.assertEqual((output / 'LICENSE').read_bytes(), (REPO / 'LICENSE').read_bytes())
            scope = json.loads((hf / 'scope.json').read_text())
            self.assertEqual(scope['schema'], builder.SCHEMA)
            self.assertEqual(scope['repo_id'], 'Dvidia/piper-grip-wait-recovery-v04')
            self.assertFalse(scope['full_raw_native_archive_uploaded'])
            self.assertFalse(scope['full_raw_report_uploaded'])
            self.assertFalse(scope['training_weights_included'])
            self.assertTrue(scope['full_raw_native_evidence_retained_locally'])
            for name, identity in scope['sources'].items():
                self.assertEqual((hf / name).read_bytes(), (REPO / identity['source_path']).read_bytes())
                self.assertEqual(builder._identity((hf / name).read_bytes()),
                                 {k: identity[k] for k in ('bytes', 'sha256')})
            self.assertEqual(receipt['huggingface']['files'], len(list(p for p in hf.rglob('*') if p.is_file())))
            self.assertIn('not standalone', (hf / 'source/README.md').read_text())
            self.assertIn('- agilex-piper', (hf / 'README.md').read_text())
            self.assertNotIn('\n- piper\n', (hf / 'README.md').read_text())

    def test_outputs_are_deterministic_and_cannot_overwrite(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            one = builder.build(REPO, root / 'hf1', root / 'research1')
            two = builder.build(REPO, root / 'hf2', root / 'research2')
            self.assertEqual(one, two)
            before = (root / 'hf1/integrity.json').read_bytes()
            with self.assertRaisesRegex(ValueError, 'fresh and absent'):
                builder.build(REPO, root / 'hf1', root / 'not-created')
            self.assertFalse((root / 'not-created').exists())
            self.assertEqual((root / 'hf1/integrity.json').read_bytes(), before)

    def test_changed_frozen_input_rejected_before_any_output(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / 'repo'
            for name in builder.EXPECTED:
                out = fixture / name
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPO / name, out)
            with (fixture / builder.PAPER).open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                builder.build(fixture, root / 'hf', root / 'research')
            self.assertFalse((root / 'hf').exists())
            self.assertFalse((root / 'research').exists())

    def test_unknowns_and_successes_cannot_be_relabelled(self):
        original = json.loads((REPO / builder.BENCH / 'report.json').read_text())
        builder._validate_report(original)
        changed = deepcopy(original)
        next(r for r in changed['rows'] if r['interrupted_unevaluable'])['raw_strict_success'] = False
        with self.assertRaisesRegex(ValueError, 'unevaluable'):
            builder._validate_report(changed)
        changed = deepcopy(original)
        changed['rows'][0]['raw_strict_success'] = False
        with self.assertRaisesRegex(ValueError, 'totals'):
            builder._validate_report(changed)

    def test_public_text_scan_rejects_private_paths_secrets_binary_and_size(self):
        for raw in (b'/Users/someone/private/file.json', b'/private/tmp/file',
                    b'token=ghp_' + b'A' * 25, b'\xff\x00', b'normal\x00text',
                    b'x' * (builder.MAX_FILE_BYTES + 1)):
            with self.assertRaises(ValueError):
                builder._scan('file.txt', raw)
        builder._scan('note.md', b'No credentials or private paths are included.')
        with self.assertRaisesRegex(ValueError, '48 files'):
            builder._with_integrity({f'{i}.txt': b'' for i in range(48)})

    def test_relative_paths_and_symlinks_are_rejected(self):
        for name in ('../file', '/file', 'a//b', 'a/./b', 'a\\b', 'a:b'):
            with self.assertRaises(ValueError):
                builder._relative(name)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'real.txt').write_text('real')
            (root / 'link.txt').symlink_to(root / 'real.txt')
            with self.assertRaisesRegex(ValueError, 'symlinks'):
                builder._ordinary(root / 'link.txt')

    @unittest.skipUnless(shutil.which('node'), 'Node is required for the viewer semantics check')
    def test_viewer_unknown_semantics_and_javascript_syntax(self):
        # Exercise the actual outcome function. An imputed false value must not
        # change an interrupted observation into an observed failure in the UI.
        with TemporaryDirectory() as temporary:
            script = Path(temporary) / 'viewer.js'
            script.write_text(builder.JS)
            subprocess.run(['node', '--check', str(script)], check=True, capture_output=True)
            head = builder.JS.split("document.getElementById('filters').addEventListener", 1)[0]
            head += "\nif(kind({interrupted_unevaluable:true,raw_strict_success:null,success:false})!=='unknown')throw Error('unknown');"
            head += "\nif(kind({interrupted_unevaluable:false,raw_strict_success:false,success:false})!=='failure')throw Error('failure');"
            head += "\nif(kind({interrupted_unevaluable:false,raw_strict_success:true,success:true})!=='success')throw Error('success');"
            head += "\nif(kind({interrupted_unevaluable:false,raw_strict_success:true,success:false,aborted:true})!=='failure')throw Error('abort');"
            subprocess.run(['node', '-e', head], check=True, capture_output=True)
        self.assertNotIn('innerHTML', builder.JS)
        self.assertIn('textContent', builder.JS)
        self.assertIn("credentials:'omit'", builder.JS)
        self.assertIn('No live physics', builder.HTML)


if __name__ == '__main__':
    unittest.main()

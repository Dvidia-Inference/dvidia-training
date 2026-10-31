"""Archive/readback boundaries and immutable-input checks; no native runs."""
from hashlib import sha256
import importlib.util
import io
import json
import gzip
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
import zipfile

_SPEC = importlib.util.spec_from_file_location('probe_package', Path(__file__).with_name('package_evidence.py'))
package = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(package)
verify = {'__name__': 'probe_archive_test'}
exec(compile(package.VERIFY_SOURCE, 'verify_archive.py', 'exec'), verify)


class PackagingTests(unittest.TestCase):
    def fixture(self, root):
        source = root/'source.json'
        source.write_text('{"unchanged":true}\n')
        return {'study/source.json': source}, {'README.md': b'Public research\n'}, {'study/source.json': package.identity(source)}

    def test_deterministic_archive_verification_reconstruction_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files, generated, frozen = self.fixture(root)
            a, b = root/'a.zip', root/'b.zip'
            package.write_archive(a, files, generated, frozen)
            package.write_archive(b, files, generated, frozen)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            result = verify['verify_archive'](a, root/'restored')
            self.assertEqual(result['files'], 2)
            self.assertEqual((root/'restored/study/source.json').read_bytes(), next(iter(files.values())).read_bytes())
            with self.assertRaises(ValueError):
                verify['verify_archive'](a, root/'restored')

    def test_changed_scientific_bytes_refused_before_indexing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files, generated, frozen = self.fixture(root)
            next(iter(files.values())).write_text('{"unchanged":nope}\n')
            with self.assertRaisesRegex(ValueError, 'Frozen evidence changed'):
                package.write_archive(root/'bad.zip', files, generated, frozen)
            self.assertFalse((root/'bad.zip').exists())

    def test_traversal_symlink_unknown_blob_and_digest_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files, generated, frozen = self.fixture(root)
            package.write_archive(root/'a.zip', files, generated, frozen)
            with zipfile.ZipFile(root/'a.zip') as archive:
                original = {name: archive.read(name) for name in archive.namelist()}
            for index, mutation in enumerate(('path', 'extra', 'data', 'link')):
                content = dict(original)
                if mutation == 'path':
                    inventory = json.loads(content['inventory.json'])
                    inventory['files'][0]['path'] = '../escape'
                    content['inventory.json'] = json.dumps(inventory).encode()
                elif mutation == 'extra':
                    content['unreviewed.txt'] = b'extra'
                elif mutation == 'data':
                    content['study/source.json'] += b'changed'
                path = root/f'bad-{index}.zip'
                with zipfile.ZipFile(path, 'x') as archive:
                    for name, raw in content.items():
                        info = zipfile.ZipInfo(name)
                        info.external_attr = ((stat.S_IFLNK if mutation == 'link' else stat.S_IFREG) | 0o644) << 16
                        archive.writestr(info, raw)
                with self.assertRaises(ValueError):
                    verify['verify_archive'](path)

    def test_plain_tree_and_scans_refuse_private_paths_credentials_and_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'original'; source.write_bytes(b'public')
            (root/'link').symlink_to(source)
            with self.assertRaises(ValueError):
                package.tree(root)
        for raw in (b'a'*65534+b'/'+b'Users/'+b'person', b'hf_'+b'x'*24):
            with self.assertRaises(ValueError):
                package.scan_stream(io.BytesIO(raw))
        for bad in ('../escape', '/absolute', 'a//b', 'a\\b'):
            with self.assertRaises(ValueError):
                package.safe_path(bad)

    def test_escaped_private_json_and_decoded_gzip_are_scanned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = b'{"path":"\\u002fUsers\\u002fperson"}'
            for name, content in (('data.json', raw), ('data.json.gz', gzip.compress(raw))):
                path = root/name
                path.write_bytes(content)
                with self.assertRaises(ValueError):
                    package.scan_file(path)

    def test_guarded_audit_refuses_optimized_mode_before_any_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrapper = Path(__file__).with_name('audit_guarded.py')
            completed = subprocess.run([sys.executable, '-O', str(wrapper), '--audit-script', str(root/'missing.py'),
                                        '--guard-receipt', str(root/'guard.json'), '--output', str(root/'audit.json')],
                                       capture_output=True, text=True, check=False)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn('refuses optimized Python', completed.stderr)
            self.assertEqual(list(root.iterdir()), [])


if __name__ == '__main__':
    unittest.main()

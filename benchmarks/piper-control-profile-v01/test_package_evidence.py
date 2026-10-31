"""Archive integrity, deduplication and reconstruction boundaries, no native work."""
from hashlib import sha256
import importlib.util
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

spec = importlib.util.spec_from_file_location('dvidia_package_evidence', Path(__file__).with_name('package_evidence.py'))
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)
helper = {'__name__': 'dvidia_test_verifier'}
exec(compile(package.VERIFY_SOURCE, 'verify_extract.py', 'exec'), helper)


class EvidencePackageTests(unittest.TestCase):
    def fixture(self, root):
        source = root/'original.txt'
        source.write_bytes(b'original scientific bytes\n')
        files = {'comparison/a.txt': source, 'comparison/b.txt': source}
        generated = {'README.md': b'Readme\n', 'LICENSE': b'Existing license\n',
                     'tools/verify_extract.py': package.VERIFY_SOURCE.encode()}
        return files, generated

    def mutate(self, archive, output, change):
        with zipfile.ZipFile(archive) as original:
            content = {name: original.read(name) for name in original.namelist()}
        change(content)
        with zipfile.ZipFile(output, 'x') as result:
            for name, value in content.items(): result.writestr(name, value)

    def test_deterministic_deduplication_verified_reconstruction_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); files, generated = self.fixture(root)
            first, second = root/'a.zip', root/'b.zip'
            manifest = package.build_archive(first, files, generated, [])
            package.build_archive(second, files, generated, [])
            self.assertEqual(first.read_bytes(), second.read_bytes())
            result = helper['verify'](first, root/'restored')
            self.assertEqual(result['logical_files'], 5)
            self.assertEqual(result['unique_blobs'], 4)
            self.assertEqual((root/'restored/comparison/a.txt').read_bytes(), files['comparison/a.txt'].read_bytes())
            self.assertEqual((root/'restored/comparison/b.txt').read_bytes(), files['comparison/b.txt'].read_bytes())
            with self.assertRaises(ValueError): helper['verify'](first, root/'restored')
            self.assertEqual(manifest['files'][0]['path'], 'LICENSE')

    def test_traversal_absolute_backslash_and_directory_collisions_reject_before_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); files, generated = self.fixture(root)
            archive = root/'original.zip'; package.build_archive(archive, files, generated, [])
            for index, bad in enumerate(('../escape', '/absolute', 'a/../b', 'a\\b', 'a//b', 'C:/drive')):
                def change(content, bad=bad):
                    manifest = json.loads(content['manifest.json']);manifest['files'][0]['path'] = bad
                    content['manifest.json'] = json.dumps(manifest).encode()
                candidate = root/f'bad-{index}.zip';self.mutate(archive, candidate, change)
                with self.assertRaises(ValueError):helper['verify'](candidate, root/f'restored-{index}')
                self.assertFalse((root/f'restored-{index}').exists())
            for bad in ('../escape', '/absolute', 'a/../b', 'a\\b'):
                with self.assertRaises(ValueError):package.safe_path(bad)

    def test_blob_tampering_missing_payload_extra_member_and_symlink_reject(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);files,generated=self.fixture(root);archive=root/'original.zip'
            package.build_archive(archive, files, generated, [])
            def tamper(content):
                name=next(name for name in content if name.startswith('blobs/'));content[name]+=b'changed'
            def missing(content):del content[next(name for name in content if name.startswith('blobs/'))]
            def extra(content):content['unexpected.txt']=b'not listed'
            for index,change in enumerate((tamper,missing,extra)):
                candidate=root/f'bad-{index}.zip';self.mutate(archive,candidate,change)
                with self.assertRaises(ValueError):helper['verify'](candidate)
            with zipfile.ZipFile(archive,'a') as bundle:
                entry=zipfile.ZipInfo('blobs/'+'0'*64);entry.create_system=3
                entry.external_attr=(stat.S_IFLNK|0o777)<<16;bundle.writestr(entry,b'target')
            with self.assertRaises(ValueError):helper['verify'](archive)

    def test_scan_detects_cross_chunk_private_paths_and_credential_shapes(self):
        private=b'/'+b'Users/'+b'person/'
        with self.assertRaises(ValueError):package.scan_stream(io.BytesIO(b'a'*65534+private))
        with self.assertRaises(ValueError):package.scan_stream(io.BytesIO(b'hf_'+b'a'*24))
        self.assertEqual(package.scan_stream(io.BytesIO(b'public hashes and counts')),24)
        with self.assertRaises(ValueError):package.scan_stream(io.BytesIO(b'x'*101),100)

    def test_prepare_rejects_mutated_frozen_scientific_payload_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);comparison=root/'comparison';comparison.mkdir()
            original=comparison/'comparison.json';original.write_bytes(b'{"all_passed":false}')
            native=comparison/'native.npz';native.write_bytes(b'original-native-bytes')
            frozen={p.name:package.file_identity(p) for p in (original,native)}
            audit={'passed':True,'evidence_unchanged':True,
                   'eligible_under_predeclared_rule_after_complete_readback':True,
                   'original_report':package.file_identity(original),'frozen_evidence':frozen}
            raw=json.dumps(audit).encode();(comparison/'readback-audit.json').write_bytes(raw)
            # Same length mutation must still be rejected by its audited digest.
            native.write_bytes(b'mutated!-native-bytes')
            with patch.object(package,'AUDIT_SHA',sha256(raw).hexdigest()):
                with self.assertRaisesRegex(ValueError,'pinned audit'):
                    package.prepare(root,comparison,root/'unused-profile',root/'release')
            self.assertFalse((root/'release').exists())

    def test_frozen_roster_rejects_unknown_and_missing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'native.npz';source.write_bytes(b'fixed')
            (root/'readback-audit.json').write_bytes(b'{}')
            audit={'frozen_evidence':{'native.npz':package.file_identity(source)}}
            self.assertEqual(package.check_frozen_evidence(root,audit),audit['frozen_evidence'])
            extra=root/'unreviewed.json';extra.write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError,'roster'):package.check_frozen_evidence(root,audit)
            extra.unlink();source.unlink()
            with self.assertRaisesRegex(ValueError,'roster'):package.check_frozen_evidence(root,audit)

    def test_archive_indexing_remains_bound_to_audited_identity_after_precheck(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);files,generated=self.fixture(root)
            expected={name:package.file_identity(source) for name,source in files.items()}
            next(iter(files.values())).write_bytes(b'changed after precheck\n')
            with self.assertRaisesRegex(ValueError,'before archive indexing'):
                package.build_archive(root/'bad.zip',files,generated,[],expected)
            self.assertFalse((root/'bad.zip').exists())


if __name__ == '__main__':unittest.main()

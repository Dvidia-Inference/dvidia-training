from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import package_evidence as package


def write_json(path, value):
    path.write_bytes(package.encoded(value))


class PackageEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.study = self.root / 'piper-retention-v02-run'
        self.audit = self.root / 'piper-retention-v02-audit'
        self.study.mkdir(); self.audit.mkdir()
        rows = [{'id': f'{scene}-{variant}', 'scene_id': f'scene-{scene}', 'variant': variant}
                for scene in range(12) for variant in sorted(package.VARIANTS)]
        write_json(self.study / 'report.json', {'schema': package.SCHEMA, 'planned_attempts': 36,
                                                'attempted': 36, 'rows': rows, 'source_sha256': {},
                                                'aggregates': {v: {k: 0 for k in ('admissions', 'false_initial_admissions',
                                                    'native_5s_pause_success', 'grip_retention_and_stationarity', 'final_both_qualified')}
                                                    for v in package.VARIANTS}})
        write_json(self.study / 'protocol.json', {'schema': 'synthetic-protocol', 'source_sha256': {}})
        write_json(self.study / 'inspection.json', {
            'schema': 'dvidia.piper-retention-inspection/v1', 'attempts': 36,
            'rows': [{'id': row['id'], 'evaluation_recomputed': True} for row in rows]})
        (self.study / 'attempt-0.bin').write_bytes(b'synthetic attempt evidence')
        (self.audit / 'independent.txt').write_text('synthetic independent audit fixture')
        study_inventory = package.tree(self.study)
        write_json(self.audit / 'audit.json', {
            'schema': 'dvidia.retention-independent-audit/v1', 'passed': True,
            'source_and_raw_inputs_unchanged': True,
            'protocol_sha256': package.identity(self.study / 'protocol.json')['sha256'],
            'source_sha256': {}, 'frozen_raw_files': study_inventory, 'rows': rows})

    def test_prepares_repeatable_complete_archive_with_verified_reconstruction(self):
        outputs = [self.root / 'package-a', self.root / 'package-b']
        receipts = [package.prepare(self.study, self.audit, output, Path(__file__).resolve().parents[2])
                    for output in outputs]
        archive_a = outputs[0] / 'huggingface' / package.ARCHIVE_NAME
        archive_b = outputs[1] / 'huggingface' / package.ARCHIVE_NAME
        self.assertEqual(archive_a.read_bytes(), archive_b.read_bytes())
        self.assertEqual(receipts[0]['verified']['verified'], True)
        self.assertEqual(receipts[0]['attempts'], 36)
        self.assertEqual({path.name for path in (outputs[0] / 'huggingface').iterdir()}, package.HF_NAMES)
        with zipfile.ZipFile(archive_a) as archive:
            self.assertIn('study/report.json', archive.namelist())
            self.assertIn('audit/audit.json', archive.namelist())
            self.assertIn('tools/verify_archive.py', archive.namelist())
            self.assertEqual(json.loads(archive.read('inventory.json'))['schema'],
                             'dvidia.piper-retention-inventory/v1')
        verifier = {'__name__': 'retention_archive_verifier'}
        exec(compile(package.VERIFY_SOURCE, 'verify_archive.py', 'exec'), verifier)
        destination = self.root / 'reconstructed'
        proof = verifier['verify'](archive_a, destination)
        self.assertTrue(proof['verified'])
        self.assertEqual((destination / 'study/report.json').read_bytes(),
                         (self.study / 'report.json').read_bytes())

    def test_requires_passed_audit_and_same_twelve_scene_roster(self):
        audit = json.loads((self.audit / 'audit.json').read_bytes())
        audit['passed'] = False
        write_json(self.audit / 'audit.json', audit)
        with self.assertRaisesRegex(ValueError, 'audit'):
            package.validate_inputs(self.study, self.audit)
        audit['passed'] = True
        write_json(self.audit / 'audit.json', audit)
        report = json.loads((self.study / 'report.json').read_bytes())
        report['rows'][-1]['scene_id'] = 'different-scene'
        write_json(self.study / 'report.json', report)
        with self.assertRaisesRegex(ValueError, 'twelve scenes'):
            package.validate_inputs(self.study, self.audit)

    def test_refuses_duplicate_nonfinite_and_private_evidence(self):
        (self.study / 'bad.json').write_bytes(b'{"x":1,"x":2}')
        with self.assertRaises(ValueError):
            package.tree(self.study)
        (self.study / 'bad.json').unlink()
        (self.study / 'bad.json').write_bytes(b'{"x":NaN}')
        with self.assertRaises(ValueError):
            package.tree(self.study)
        (self.study / 'bad.json').unlink()
        (self.study / 'attempt-0.bin').write_bytes(b'/'+b'Users/private/person/video.mp4')
        with self.assertRaisesRegex(ValueError, 'Private path'):
            package.tree(self.study)

    def test_verifier_rejects_modified_archive_and_refuses_overwrite(self):
        receipt = package.prepare(self.study, self.audit, self.root / 'package',
                                  Path(__file__).resolve().parents[2])
        archive_path = self.root / 'package/huggingface' / package.ARCHIVE_NAME
        verifier = {'__name__': 'retention_archive_verifier'}
        exec(compile(package.VERIFY_SOURCE, 'verify_archive.py', 'exec'), verifier)
        output = self.root / 'reconstructed'
        verifier['verify'](archive_path, output)
        with self.assertRaisesRegex(ValueError, 'new'):
            verifier['verify'](archive_path, output)
        altered = self.root / 'altered.zip'
        with zipfile.ZipFile(archive_path) as source, zipfile.ZipFile(altered, 'w') as dest:
            for name in source.namelist():
                raw = source.read(name)
                if name == 'study/report.json':
                    raw += b' '
                dest.writestr(name, raw)
        with self.assertRaisesRegex(ValueError, 'expected bytes|digest'):
            verifier['verify'](altered)
        self.assertEqual(receipt['archive']['sha256'], sha256(archive_path.read_bytes()).hexdigest())

    def test_prepare_rejects_transient_same_size_mutation_before_archive_indexing(self):
        target = self.study/'attempt-0.bin'
        original = target.read_bytes()
        real_archive = package.archive
        def mutate_then_archive(*args, **kwargs):
            target.write_bytes(b'X'*len(original))
            try:
                return real_archive(*args, **kwargs)
            finally:
                # Restoration would defeat a final-tree-only check.
                target.write_bytes(original)
        with patch.object(package, 'archive', side_effect=mutate_then_archive):
            with self.assertRaisesRegex(ValueError, 'pinned evidence identity'):
                package.prepare(self.study,self.audit,self.root/'mutated-index',
                                Path(__file__).resolve().parents[2])
        self.assertEqual(target.read_bytes(), original)

    def test_archive_stream_copy_rejects_mutation_after_indexing(self):
        target = self.study/'attempt-0.bin'
        original, real_zip = target.read_bytes(), zipfile.ZipFile
        expected = {'study/attempt-0.bin': package.identity(target)}
        def mutate_before_copy(*args, **kwargs):
            target.write_bytes(b'Y'*len(original))
            return real_zip(*args, **kwargs)
        try:
            with patch.object(package.zipfile, 'ZipFile', side_effect=mutate_before_copy):
                with self.assertRaisesRegex(ValueError, 'Copied archive input differs'):
                    package.archive(self.root/'mutated-copy.zip',{'study/attempt-0.bin':target},{},
                                    expected_identities=expected)
        finally:
            target.write_bytes(original)

    def test_parsed_metadata_is_bound_to_the_tree_snapshot(self):
        real_load = package.load_json
        def changed_read(path, *args, **kwargs):
            value, raw = real_load(path, *args, **kwargs)
            return (value, raw+b' ') if Path(path).name=='report.json' else (value,raw)
        with patch.object(package, 'load_json', side_effect=changed_read):
            with self.assertRaisesRegex(ValueError, 'Parsed metadata differs'):
                package.validate_inputs(self.study,self.audit)

    def test_archive_identity_has_its_own_larger_byte_bound(self):
        large = self.root/package.ARCHIVE_NAME
        with large.open('wb') as stream:
            stream.truncate(package.MAX_FILE+1)
        with self.assertRaisesRegex(ValueError, 'oversized'):
            package.identity(large)
        self.assertEqual(package.identity(large,package.MAX_ARCHIVE),package.archive_digest(large))
        real_identity, archive_calls = package.identity, []
        def checked_identity(path, limit=package.MAX_FILE):
            if Path(path).name==package.ARCHIVE_NAME:
                self.assertEqual(limit,package.MAX_ARCHIVE)
                archive_calls.append(path)
            return real_identity(path,limit)
        with patch.object(package, 'identity', side_effect=checked_identity):
            package.prepare(self.study,self.audit,self.root/'archive-limit',
                            Path(__file__).resolve().parents[2])
        self.assertEqual(len(archive_calls),1)


if __name__ == '__main__':
    unittest.main()

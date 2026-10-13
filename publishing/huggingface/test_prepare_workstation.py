"""Data-only workstation evidence staging, with no credentials or native engine."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zlib

import prepare_workstation as prepare
import publish_robot_sim as publisher


def png(metadata=False):
    def chunk(kind, raw):
        return struct.pack('>I', len(raw))+kind+raw+struct.pack('>I', zlib.crc32(kind+raw)&0xffffffff)
    return (b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0))
            +(chunk(b'tEXt', b'Comment\0metadata') if metadata else b'')
            +chunk(b'IDAT', zlib.compress(b'\0\x10\x20\x30'))+chunk(b'IEND', b''))


class WorkstationPublicationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.space = self.root/'base/space'
        self.space.mkdir(parents=True)
        self.original = {
            'README.md': b'---\nsdk: static\napp_file: index.html\nlicense: mit\n---\nPrior card\n',
            'index.html': b'<section aria-labelledby="assets-title">Prior assets</section>',
            'LICENSE': b'MIT', 'provenance.json': b'{"prior":"preserved"}',
            'tasks/prior/report.json': b'{"all_failures_retained":true}',
            'asset-library/v01/README.md': b'Original shoe library\n',
            'replay/geometry.bin': b'Original native geometry'}
        for name, raw in self.original.items():
            path = self.space/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        self.plan = self.root/'base/publication-plan.json'
        self.refresh_base()
        self.workstation = self.root/'native'
        self.workstation.mkdir()
        self.native_files = {
            'index.html': b'Live simulator', 'scene.xml': b'<mujoco/>',
            'mujoco.wasm': b'\0asm\1\0\0\0', 'model.json': b'{}',
            'workstation.json': b'{}', 'workstation-integrity.json': b'{}',
            'geometry.bin.gz': b'fixture geometry', 'LICENSE': b'MIT', 'README.md': b'Native scene guide'}
        for name, raw in self.native_files.items():
            (self.workstation/name).write_bytes(raw)
        self.evidence = self.root/'evidence'
        self.evidence.mkdir()
        self.evidence_files = {'README.md': b'Complete attempts and failures.\n',
                               'confirmation/report.json': b'{"passed":false,"attempts":20}',
                               'browser/desktop.png': png()}
        for name, raw in self.evidence_files.items():
            path = self.evidence/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        guard = patch.object(publisher, 'client', side_effect=AssertionError('Credentials forbidden'))
        guard.start()
        self.addCleanup(guard.stop)

    def refresh_base(self):
        self.plan.write_bytes(prepare.encoded({
            'kind': publisher.KIND, 'schema_version': 1, 'namespace': 'Dvidia', 'private': False,
            'source_commit': 'a'*40, 'expected_remote_commit': 'b'*40, 'remove_files': [],
            'repository': {'repo_id': publisher.REPO_ID, 'repo_type': 'space', 'space_sdk': 'static',
                'directory': 'space', 'files': [{'path': name, **prepare.record(raw)} for name, raw in self.original.items()]},
            'total_bytes': sum(map(len, self.original.values()))}))

    def inspect_native(self, directory):
        root = Path(directory)
        actual = {p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}
        if actual != self.native_files:
            raise ValueError('Strict native export received different bytes or extra evidence')
        return {'file_count':len(actual)}

    def stage(self, name='output', **kwargs):
        with patch('dvidia_training.workstation.inspect_export', side_effect=self.inspect_native):
            return prepare.stage(self.plan, self.workstation, self.root/name, 'c'*40, 'd'*40,
                                 evidence=kwargs.get('evidence', self.evidence))

    def test_preserves_prior_results_and_strict_native_namespace(self):
        result = self.stage()
        output = self.root/'output/space'
        for name, raw in self.original.items():
            if name not in {'README.md', 'index.html', 'provenance.json'}:
                self.assertEqual((output/name).read_bytes(), raw)
        self.inspect_native(output/prepare.NAMESPACE)
        for name, raw in self.evidence_files.items():
            self.assertEqual((output/prepare.EVIDENCE_NAMESPACE/name).read_bytes(), raw)
            if name in self.native_files:
                self.assertEqual((output/prepare.NAMESPACE/name).read_bytes(), self.native_files[name])
            else:
                self.assertFalse((output/prepare.NAMESPACE/name).exists())
        provenance = json.loads((output/'provenance.json').read_bytes())
        self.assertEqual(provenance['prior'], 'preserved')
        evidence = provenance['live_workstation']['evidence']
        self.assertEqual(evidence['path'], prepare.EVIDENCE_NAMESPACE)
        self.assertEqual(evidence['source_commit'], 'c'*40)
        seal_bytes = (output/prepare.EVIDENCE_NAMESPACE/prepare.EVIDENCE_SEAL).read_bytes()
        self.assertEqual(evidence['integrity'], prepare.record(seal_bytes))
        seal = json.loads(seal_bytes)
        self.assertEqual(seal['source_commit'], 'c'*40)
        self.assertEqual(seal['workstation_integrity'], prepare.record(self.native_files['workstation-integrity.json']))
        self.assertEqual(seal['files'], {name:prepare.record(raw) for name, raw in self.evidence_files.items()})
        self.assertIn(b'workstation-evidence/v01/README.md', (output/'README.md').read_bytes())
        plan = publisher.checked_plan(self.root/'output/publication-plan.json')
        self.assertEqual(plan['expected_remote_commit'], 'd'*40)
        self.assertEqual(plan['remove_files'], [])
        self.assertEqual(result['added_files'], len(self.native_files))
        self.assertEqual(result['evidence_files'], len(self.evidence_files)+1)

    def test_rejects_tampered_prior_data_and_existing_output_before_overwrite(self):
        (self.space/'tasks/prior/report.json').write_bytes(b'changed')
        with self.assertRaises(publisher.PublicationError):
            self.stage()
        self.assertFalse((self.root/'output').exists())
        (self.space/'tasks/prior/report.json').write_bytes(self.original['tasks/prior/report.json'])
        (self.root/'output').mkdir()
        (self.root/'output/keep.md').write_text('keep')
        with self.assertRaises(ValueError):
            self.stage()
        self.assertEqual((self.root/'output/keep.md').read_text(), 'keep')

    def test_prior_artifact_mutation_during_staging_is_rejected(self):
        def inspect_and_mutate(directory):
            result = self.inspect_native(directory)
            (self.space/'tasks/prior/report.json').write_bytes(b'changed after prior verification')
            return result
        with patch('dvidia_training.workstation.inspect_export', side_effect=inspect_and_mutate):
            with self.assertRaisesRegex(ValueError, 'Prior artifact changed'):
                prepare.stage(self.plan, self.workstation, self.root/'output', 'c'*40, 'd'*40, evidence=self.evidence)
        self.assertFalse((self.root/'output').exists())

    def test_rejects_replacement_of_either_prior_release_namespace(self):
        for prefix in (prepare.NAMESPACE, prepare.EVIDENCE_NAMESPACE):
            name = prefix+'/prior.json'
            self.original[name] = b'{}'
            path = self.space/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'{}')
            self.refresh_base()
            with self.subTest(prefix=prefix), self.assertRaisesRegex(ValueError, 'replace'):
                self.stage()
            self.assertFalse((self.root/'output').exists())
            del self.original[name]
            path.unlink()
        self.refresh_base()

    def test_evidence_requires_separate_input_tree_and_readme(self):
        for folder in (self.workstation, self.workstation/'evidence', self.root):
            with self.subTest(folder=folder), self.assertRaisesRegex(ValueError, 'separate'):
                self.stage(evidence=folder)
        (self.evidence/'README.md').unlink()
        with self.assertRaisesRegex(ValueError, 'README'):
            self.stage()
        self.assertFalse((self.root/'output').exists())

    def test_only_safe_data_paths_and_extensions_are_allowed(self):
        for name in ('extra.js', 'extra.html', 'extra.xml', 'model.wasm', '.hidden.json',
                     'space name.json', 'a:b.json', prepare.EVIDENCE_SEAL):
            path = self.evidence/name
            path.write_bytes(b'{}')
            with self.subTest(name=name), self.assertRaises((ValueError, publisher.PublicationError)):
                prepare.read_evidence(self.evidence)
            path.unlink()
        alias = self.evidence/'alias.json'
        alias.symlink_to(self.evidence/'confirmation/report.json')
        with self.assertRaises(publisher.PublicationError):
            prepare.read_evidence(self.evidence)

    def test_private_paths_and_tokens_are_checked_after_json_unescaping(self):
        values = [b'{"path":"\\u002fUsers\\u002fsomeone\\u002fprivate"}',
                  b'{"path":"C:\\\\Users\\\\someone\\\\private"}',
                  prepare.encoded({'token':'github_pat_'+'x'*40}),
                  prepare.encoded({'token':'hf_'+'x'*40}),
                  b'{"path":"/private/tmp/receipt.json"}']
        for raw in values:
            (self.evidence/'private.json').write_bytes(raw)
            with self.subTest(raw=raw), self.assertRaises((ValueError, publisher.PublicationError)):
                prepare.read_evidence(self.evidence)
        self.assertFalse((self.root/'output').exists())

    def test_json_rejects_nonfinite_duplicates_and_excessive_depth(self):
        for raw in (b'{"x":NaN}', b'{"x":1,"x":2}', b'['*65+b'0'+b']'*65, b'\xff'):
            (self.evidence/'invalid.json').write_bytes(raw)
            with self.subTest(raw=raw[:40]), self.assertRaises(ValueError):
                prepare.read_evidence(self.evidence)

    def test_png_rejects_text_metadata_bogus_images_and_trailing_payload(self):
        for raw in (b'not png', png(metadata=True), png()+b'private trailing payload', png()[:-5]):
            (self.evidence/'browser/desktop.png').write_bytes(raw)
            with self.subTest(raw=raw[:30]), self.assertRaises(ValueError):
                prepare.read_evidence(self.evidence)

    def test_evidence_and_total_budgets_apply_before_output_creation(self):
        for constant, value in [('MAX_EVIDENCE_FILES', 1), ('MAX_EVIDENCE_BYTES', 2)]:
            with patch.object(prepare, constant, value), self.assertRaises(ValueError):
                self.stage()
            self.assertFalse((self.root/'output').exists())
        with patch.object(publisher, 'MAX_TOTAL_BYTES', sum(map(len, self.original.values()))), self.assertRaises(ValueError):
            self.stage()
        self.assertFalse((self.root/'output').exists())

    def test_final_inspection_failure_cleans_staging_and_preserves_inputs(self):
        with patch('dvidia_training.workstation.inspect_export', side_effect=[{}, ValueError('changed bytes')]):
            with self.assertRaisesRegex(ValueError, 'changed bytes'):
                prepare.stage(self.plan, self.workstation, self.root/'output', 'c'*40, 'd'*40, evidence=self.evidence)
        self.assertFalse((self.root/'output').exists())
        self.assertFalse(list(self.root.glob('.workstation-publication-*')))
        self.inspect_native(self.workstation)


if __name__ == '__main__':
    unittest.main()

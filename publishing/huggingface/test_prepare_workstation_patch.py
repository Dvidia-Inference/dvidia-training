"""Download fixes retain physics, prior evidence and bounded publication inputs."""
import gzip
import json
import unittest
from unittest.mock import patch

import prepare_workstation as original
import prepare_workstation_delivery as delivery
import prepare_workstation_patch as prepare
import publish_robot_sim as publisher
import test_prepare_workstation as fixtures


class DownloadFixTests(unittest.TestCase):
    refresh_base = fixtures.WorkstationPublicationTests.refresh_base
    inspect_native = fixtures.WorkstationPublicationTests.inspect_native

    def setUp(self):
        fixtures.WorkstationPublicationTests.setUp(self)
        for name in ('workstation.bundle.js', 'workstation_worker.bundle.js'):
            self.native_files[name] = b'original runtime'
            (self.workstation/name).write_bytes(self.native_files[name])
        with patch('dvidia_training.workstation.inspect_export', side_effect=self.inspect_native):
            original.stage(self.plan, self.workstation, self.root/'first', 'c'*40, 'd'*40,
                           evidence=self.evidence)
        self.prior = self.root/'prior-export'
        self.prior.mkdir()
        self.prior_files = {**self.native_files,
            'workstation.json': original.encoded({'native_asset_bundle': {'path': 'native-assets.bin.gz'}}),
            'native-assets.bin.gz': b'prior transport fixture'}
        for name, raw in self.prior_files.items():
            (self.prior/name).write_bytes(raw)
        with patch('dvidia_training.workstation.inspect_export', return_value={}):
            delivery.stage(self.root/'first/publication-plan.json', self.prior, self.evidence,
                           self.root/'delivery', 'e'*40, 'f'*40)
        self.base = self.root/'delivery/publication-plan.json'
        self.new = self.root/'new-export'
        self.new.mkdir()
        self.updated = {**self.prior_files,
                        'index.html': b'Official DVIDIA workstation with bounded download recovery',
                        'workstation_worker.bundle.js': b'new loader',
                        'workstation.bundle.js': b'new download status',
                        'workstation-integrity.json': b'{"new":true}',
                        'mujoco.wasm.gz': gzip.compress(self.native_files['mujoco.wasm'], mtime=0)}
        for name, raw in self.updated.items():
            (self.new/name).write_bytes(raw)
        self.refresh_metadata()

    def refresh_metadata(self, compressed=None, native=None):
        native = self.prior_files['mujoco.wasm'] if native is None else native
        compressed = self.updated['mujoco.wasm.gz'] if compressed is None else compressed
        metadata = json.loads(self.prior_files['workstation.json'])
        metadata.update(native_wasm={'path': 'mujoco.wasm', **original.record(native)},
                        wasm_bundle={'path': 'mujoco.wasm.gz', 'compression': 'gzip',
                                     **original.record(compressed), 'uncompressed_bytes': len(native)})
        (self.new/'workstation.json').write_bytes(original.encoded(metadata))

    def stage(self, **overrides):
        args = {'base_plan': self.base, 'workstation': self.new, 'evidence': self.evidence,
                'output': self.root/'fixed', 'source_commit': '1'*40, 'expected_remote_commit': '2'*40}
        args.update(overrides)
        with patch('dvidia_training.workstation.inspect_export', return_value={}):
            return prepare.stage(**args)

    def test_preserves_every_old_receipt_and_native_byte_and_appends_provenance(self):
        result = self.stage()
        before, after = self.root/'delivery/space', self.root/'fixed/space'
        allowed = {'README.md', 'provenance.json'} | {original.NAMESPACE+'/'+name for name in prepare.MUTABLE}
        for path in before.rglob('*'):
            if path.is_file() and path.relative_to(before).as_posix() not in allowed:
                self.assertEqual(path.read_bytes(), (after/path.relative_to(before)).read_bytes())
        old_live = json.loads((before/'provenance.json').read_bytes())['live_workstation']
        new_live = json.loads((after/'provenance.json').read_bytes())['live_workstation']
        for key in ('evidence', 'delivery', 'original_source_commit'):
            self.assertEqual(new_live[key], old_live[key])
        self.assertEqual(new_live['source_commit'], '1'*40)
        self.assertEqual(new_live['download_fix']['prior_source_commit'], 'e'*40)
        self.assertEqual(new_live['download_fix']['prior_publication_commit'], '2'*40)
        seal_raw = (after/prepare.EVIDENCE_NAMESPACE/original.EVIDENCE_SEAL).read_bytes()
        self.assertEqual(new_live['download_fix']['integrity'], original.record(seal_raw))
        seal = json.loads(seal_raw)
        self.assertEqual(seal['files'], {name: original.record(raw) for name, raw in self.evidence_files.items()})
        self.assertEqual(seal['added_files'], ['mujoco.wasm.gz'])
        for name, raw in self.evidence_files.items():
            self.assertEqual((after/prepare.EVIDENCE_NAMESPACE/name).read_bytes(), raw)
        plan = publisher.checked_plan(self.root/'fixed/publication-plan.json')
        self.assertEqual(plan['expected_remote_commit'], '2'*40)
        self.assertEqual(plan['source_commit'], '1'*40)
        self.assertEqual(plan['remove_files'], [])
        self.assertEqual(result['added_workstation_files'], ['mujoco.wasm.gz'])

    def test_refuses_physics_engine_notice_or_native_bundle_changes(self):
        for name in ('scene.xml', 'mujoco.wasm', 'geometry.bin.gz', 'native-assets.bin.gz', 'LICENSE'):
            (self.new/name).write_bytes(b'changed')
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'changed physics'):
                self.stage()
            (self.new/name).write_bytes(self.updated[name])
        self.assertFalse((self.root/'fixed').exists())

    def test_refuses_missing_or_unexpected_inventory(self):
        for name in ('mujoco.wasm', 'mujoco.wasm.gz'):
            raw = (self.new/name).read_bytes()
            (self.new/name).unlink()
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'retain the prior inventory'):
                self.stage()
            (self.new/name).write_bytes(raw)
        (self.new/'extra.json').write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, 'add only mujoco.wasm.gz'):
            self.stage()

    def test_only_exact_new_descriptors_are_allowed(self):
        valid = json.loads((self.new/'workstation.json').read_bytes())
        invalid = []
        for key in ('native_wasm', 'wasm_bundle'):
            data = json.loads(original.encoded(valid))
            del data[key]
            invalid.append(data)
        data = json.loads(original.encoded(valid))
        data['native_asset_bundle'] = {'path': 'different'}
        invalid.append(data)
        for key in ('native_wasm', 'wasm_bundle'):
            for field, value in (('path', 'other.wasm'), ('sha256', '0'*64), ('bytes', 1), ('extra', True)):
                data = json.loads(original.encoded(valid))
                data[key][field] = value
                invalid.append(data)
        for data in invalid:
            (self.new/'workstation.json').write_bytes(original.encoded(data))
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.stage()

    def test_gzip_must_decode_exact_engine_without_trailing_or_extra_members(self):
        valid = self.updated['mujoco.wasm.gz']
        invalid = (b'not gzip', valid[:-2], valid+b'trailing', valid+valid,
                   gzip.compress(b'x'*1024*1024, mtime=0), gzip.compress(b'different', mtime=0))
        for compressed in invalid:
            (self.new/'mujoco.wasm.gz').write_bytes(compressed)
            self.refresh_metadata(compressed=compressed)
            with self.subTest(size=len(compressed)), self.assertRaisesRegex(ValueError, 'WASM bundle'):
                self.stage()
        self.assertFalse((self.root/'fixed').exists())

    def test_requires_exact_commit_names_fresh_output_and_separate_inputs(self):
        for field in ('source_commit', 'expected_remote_commit'):
            for value in ('main', 'a'*39, 'A'*40, None):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, 'Exact source'):
                    self.stage(**{field: value})
        for folder in (self.new, self.new/'evidence', self.root):
            with self.subTest(folder=folder), self.assertRaisesRegex(ValueError, 'separate'):
                self.stage(evidence=folder)
        for folder in (self.new/'output', self.evidence/'output', self.base.parent/'output'):
            with self.subTest(folder=folder), self.assertRaisesRegex(ValueError, 'outside input'):
                self.stage(output=folder)
        (self.root/'fixed').mkdir()
        (self.root/'fixed/keep.md').write_bytes(b'keep')
        with self.assertRaisesRegex(ValueError, 'fresh'):
            self.stage()
        self.assertEqual((self.root/'fixed/keep.md').read_bytes(), b'keep')

    def test_rejects_changed_prior_artifacts(self):
        old = self.root/'delivery/space/tasks/prior/report.json'
        old.write_bytes(b'changed')
        with self.assertRaises(publisher.PublicationError):
            self.stage()
        self.assertFalse((self.root/'fixed').exists())

    def test_refuses_replacing_prior_provenance_or_download_evidence(self):
        path = self.root/'delivery/space/provenance.json'
        prior = json.loads(path.read_bytes())
        plan = json.loads(self.base.read_bytes())
        for mutation in ('inventory', 'download_fix'):
            value = json.loads(original.encoded(prior))
            live = value['live_workstation']
            if mutation == 'inventory':
                live['files']['mujoco.wasm']['sha256'] = '0'*64
            else:
                live['download_fix'] = {'prior': 'must remain intact'}
            raw = original.encoded(value)
            path.write_bytes(raw)
            for entry in plan['repository']['files']:
                if entry['path'] == 'provenance.json':
                    entry.update(original.record(raw))
            plan['total_bytes'] = sum(entry['bytes'] for entry in plan['repository']['files'])
            self.base.write_bytes(original.encoded(plan))
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, 'provenance'):
                self.stage()
        self.assertFalse((self.root/'fixed').exists())

    def test_rejects_prior_mutation_during_inspection(self):
        def mutate(_):
            (self.root/'delivery/space/tasks/prior/report.json').write_bytes(b'changed')
        with patch('dvidia_training.workstation.inspect_export', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'Prior artifact changed'):
                prepare.stage(self.base, self.new, self.evidence, self.root/'fixed', '1'*40, '2'*40)

    def test_budgets_and_final_inspection_failure_leave_no_output(self):
        for constant, value in (('MAX_FILES', 1), ('MAX_TOTAL_BYTES', 1)):
            with patch.object(publisher, constant, value), self.assertRaises((ValueError, publisher.PublicationError)):
                self.stage()
        with patch('dvidia_training.workstation.inspect_export', side_effect=[{}, {}, ValueError('final mismatch')]):
            with self.assertRaisesRegex(ValueError, 'final mismatch'):
                prepare.stage(self.base, self.new, self.evidence, self.root/'fixed', '1'*40, '2'*40)
        self.assertFalse((self.root/'fixed').exists())
        self.assertFalse(list(self.root.glob('.workstation-download-fix-*')))

    def test_native_wasm_bundle_allowlist_is_scoped(self):
        publisher._artifact_name('workstation/v01/mujoco.wasm.gz')
        for name in ('other/mujoco.wasm.gz', 'workstation/v02/mujoco.wasm.gz',
                     'workstation/v01/other.wasm.gz', 'workstation/v01/assets/mujoco.wasm.gz'):
            with self.subTest(name=name), self.assertRaises(publisher.PublicationError):
                publisher._artifact_name(name)


if __name__ == '__main__':
    unittest.main()

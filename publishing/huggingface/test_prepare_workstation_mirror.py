"""Mirror staging cannot alter physics, binaries or earlier public evidence."""
import gzip
import json
import unittest
from unittest.mock import patch

import prepare_workstation as original
import prepare_workstation_mirror as prepare
import publish_robot_sim as publisher
import test_prepare_workstation as fixtures


class MirrorTests(unittest.TestCase):
    refresh_base = fixtures.WorkstationPublicationTests.refresh_base
    inspect_native = fixtures.WorkstationPublicationTests.inspect_native

    def setUp(self):
        fixtures.WorkstationPublicationTests.setUp(self)
        self.model = {'schema': 'model-fixture', 'bodies': ['fixed'], 'geometry_binary': {
            'href': 'geometry.bin.gz', 'compression': 'gzip', 'bytes': 16,
            'compressed_bytes': len(self.native_files['geometry.bin.gz']), 'sha256': 'a'*64}}
        self.native_files.update({
            'model.json': original.encoded(self.model),
            'workstation.bundle.js': b'prior renderer',
            'workstation_worker.bundle.js': b'prior worker',
            'native-assets.bin.gz': b'unchanged native bundle',
            'mujoco.wasm.gz': gzip.compress(self.native_files['mujoco.wasm'], mtime=0)})
        for name, raw in self.native_files.items():
            (self.workstation/name).write_bytes(raw)
        with patch('dvidia_training.workstation.inspect_export', side_effect=self.inspect_native):
            original.stage(self.plan, self.workstation, self.root/'prior', 'c'*40, 'd'*40,
                           evidence=self.evidence)
        self.base = self.root/'prior/publication-plan.json'
        self.prior = self.root/'prior/space'
        self.provenance = self.prior/'provenance.json'
        value = json.loads(self.provenance.read_bytes())
        value['live_workstation']['delivery'] = {'retained': True}
        value['live_workstation']['download_fix'] = {'source_commit': 'c'*40, 'retained': True}
        self.provenance.write_bytes(original.encoded(value))
        self.refresh_prior_plan()
        self.new = self.root/'mirror-export'
        self.new.mkdir()
        self.updated = {**self.native_files, 'workstation.bundle.js': b'verified mirror renderer',
                        'workstation_worker.bundle.js': b'verified mirror worker',
                        'workstation-integrity.json': b'{"mirror":true}',
                        'README.md': b'Verified DVIDIA mirror delivery'}
        self.new_model = {**self.model, 'geometry_binary': {**self.model['geometry_binary'],
            'compressed_sha256': original.record(self.native_files['geometry.bin.gz'])['sha256']}}
        self.updated['model.json'] = original.encoded(self.new_model)
        for name, raw in self.updated.items():
            (self.new/name).write_bytes(raw)

    def refresh_prior_plan(self):
        plan = json.loads(self.base.read_bytes())
        plan['repository']['files'] = [
            {'path': path.relative_to(self.prior).as_posix(), **original.record(path.read_bytes())}
            for path in sorted(self.prior.rglob('*')) if path.is_file()]
        plan['total_bytes'] = sum(entry['bytes'] for entry in plan['repository']['files'])
        self.base.write_bytes(original.encoded(plan))

    def stage(self, **overrides):
        args = {'base_plan': self.base, 'workstation': self.new, 'evidence': self.evidence,
                'output': self.root/'mirror', 'source_commit': '1'*40, 'expected_remote_commit': '2'*40}
        args.update(overrides)
        with patch('dvidia_training.workstation.inspect_export', return_value={}):
            return prepare.stage(**args)

    def test_preserves_all_prior_binaries_evidence_and_provenance(self):
        result = self.stage()
        after = self.root/'mirror/space'
        allowed = {'README.md', 'provenance.json'} | {original.NAMESPACE+'/'+name for name in prepare.MUTABLE}
        for path in self.prior.rglob('*'):
            if path.is_file() and path.relative_to(self.prior).as_posix() not in allowed:
                self.assertEqual(path.read_bytes(), (after/path.relative_to(self.prior)).read_bytes())
        before = json.loads(self.provenance.read_bytes())['live_workstation']
        live = json.loads((after/'provenance.json').read_bytes())['live_workstation']
        for key in ('evidence', 'delivery', 'download_fix'):
            self.assertEqual(live[key], before[key])
        self.assertEqual(live['mirror']['prior_source_commit'], 'c'*40)
        self.assertEqual(live['mirror']['prior_publication_commit'], '2'*40)
        self.assertEqual(live['source_commit'], '1'*40)
        self.assertEqual(live['files'], {name: original.record(raw) for name, raw in self.updated.items()})
        seal_raw = (after/prepare.EVIDENCE_NAMESPACE/original.EVIDENCE_SEAL).read_bytes()
        self.assertEqual(live['mirror']['integrity'], original.record(seal_raw))
        seal = json.loads(seal_raw)
        self.assertEqual(seal['added_files'], [])
        self.assertTrue(seal['compressed_resources_unchanged'])
        self.assertEqual(seal['files'], {name: original.record(raw) for name, raw in self.evidence_files.items()})
        for name, raw in self.evidence_files.items():
            self.assertEqual((after/prepare.EVIDENCE_NAMESPACE/name).read_bytes(), raw)
        plan = publisher.checked_plan(self.root/'mirror/publication-plan.json')
        self.assertEqual(plan['source_commit'], '1'*40)
        self.assertEqual(plan['expected_remote_commit'], '2'*40)
        self.assertEqual(plan['remove_files'], [])
        self.assertEqual(result['added_workstation_files'], [])
        self.assertIn(b'SHA-256 verification', (after/'README.md').read_bytes())

    def test_protects_every_binary_scene_manifest_branding_and_notice(self):
        for name in ('scene.xml', 'mujoco.wasm', 'mujoco.wasm.gz', 'geometry.bin.gz',
                     'native-assets.bin.gz', 'workstation.json', 'index.html', 'LICENSE'):
            (self.new/name).write_bytes(b'changed')
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'changed physics'):
                self.stage()
            (self.new/name).write_bytes(self.updated[name])
        self.assertFalse((self.root/'mirror').exists())

    def test_rejects_any_removed_or_added_runtime_file(self):
        (self.new/'mujoco.wasm').unlink()
        with self.assertRaisesRegex(ValueError, 'exact prior runtime inventory'):
            self.stage()
        (self.new/'mujoco.wasm').write_bytes(self.updated['mujoco.wasm'])
        (self.new/'extra.json').write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, 'exact prior runtime inventory'):
            self.stage()

    def test_model_change_is_only_exact_compressed_digest(self):
        invalid = []
        for field, value in (('sha256', '0'*64), ('bytes', 9), ('extra', True), ('compressed_sha256', 'f'*64)):
            model = json.loads(original.encoded(self.new_model))
            model['geometry_binary'][field] = value
            invalid.append(model)
        model = json.loads(original.encoded(self.new_model))
        del model['geometry_binary']['compressed_sha256']
        invalid.append(model)
        model = json.loads(original.encoded(self.new_model))
        model['bodies'] = ['changed']
        invalid.append(model)
        invalid.extend(([], {'geometry_binary': []}, {}))
        for model in invalid:
            (self.new/'model.json').write_bytes(original.encoded(model))
            with self.subTest(model=model), self.assertRaises(ValueError):
                self.stage()
        self.assertFalse((self.root/'mirror').exists())

    def test_accepts_preexisting_matching_digest_but_rejects_bad_prior_digest(self):
        old_model = self.prior/original.NAMESPACE/'model.json'
        for checksum in ('0'*64, self.new_model['geometry_binary']['compressed_sha256']):
            model = {**self.model, 'geometry_binary': {**self.model['geometry_binary'], 'compressed_sha256': checksum}}
            old_model.write_bytes(original.encoded(model))
            value = json.loads(self.provenance.read_bytes())
            value['live_workstation']['files']['model.json'] = original.record(old_model.read_bytes())
            self.provenance.write_bytes(original.encoded(value))
            self.refresh_prior_plan()
            if checksum == '0'*64:
                with self.assertRaisesRegex(ValueError, 'Prior compressed geometry checksum'):
                    self.stage()
            else:
                self.stage()

    def test_refuses_replacing_old_mirror_provenance_or_evidence(self):
        prior = self.provenance.read_bytes()
        for mode in ('inventory', 'mirror'):
            value = json.loads(prior)
            if mode == 'inventory':
                value['live_workstation']['files']['mujoco.wasm']['sha256'] = '0'*64
            else:
                value['live_workstation']['mirror'] = {'must_remain': True}
            self.provenance.write_bytes(original.encoded(value))
            self.refresh_prior_plan()
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'provenance'):
                self.stage()
        self.provenance.write_bytes(prior)
        path = self.prior/prepare.EVIDENCE_NAMESPACE/'README.md'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'Prior mirror evidence')
        self.refresh_prior_plan()
        with self.assertRaisesRegex(ValueError, 'fresh mirror evidence'):
            self.stage()

    def test_commit_and_input_bounds(self):
        for field in ('source_commit', 'expected_remote_commit'):
            for value in ('main', 'a'*39, 'A'*40, None):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, 'Exact source'):
                    self.stage(**{field: value})
        for folder in (self.new, self.new/'proof', self.root):
            with self.subTest(folder=folder), self.assertRaisesRegex(ValueError, 'separate'):
                self.stage(evidence=folder)
        for folder in (self.new/'output', self.evidence/'output', self.base.parent/'output'):
            with self.subTest(folder=folder), self.assertRaisesRegex(ValueError, 'outside input'):
                self.stage(output=folder)
        (self.root/'mirror').mkdir()
        (self.root/'mirror/keep.md').write_bytes(b'keep')
        with self.assertRaisesRegex(ValueError, 'fresh'):
            self.stage()
        self.assertEqual((self.root/'mirror/keep.md').read_bytes(), b'keep')

    def test_prior_tampering_and_failed_final_inspection_leave_no_output(self):
        old = self.prior/'tasks/prior/report.json'
        raw = old.read_bytes()
        old.write_bytes(b'changed')
        with self.assertRaises(publisher.PublicationError):
            self.stage()
        old.write_bytes(raw)
        with patch('dvidia_training.workstation.inspect_export', side_effect=[{}, {}, ValueError('final mismatch')]):
            with self.assertRaisesRegex(ValueError, 'final mismatch'):
                prepare.stage(self.base, self.new, self.evidence, self.root/'mirror', '1'*40, '2'*40)
        self.assertFalse((self.root/'mirror').exists())
        self.assertFalse(list(self.root.glob('.workstation-mirror-*')))


if __name__ == '__main__':
    unittest.main()

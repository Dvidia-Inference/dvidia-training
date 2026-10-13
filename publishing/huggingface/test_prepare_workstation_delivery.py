"""Delivery revisions preserve physics and every original public evidence byte."""
import json
import unittest
from unittest.mock import patch

import prepare_workstation as original
import prepare_workstation_delivery as delivery
import publish_robot_sim as publisher
import test_prepare_workstation as fixtures


class DeliveryTests(unittest.TestCase):
    refresh_base = fixtures.WorkstationPublicationTests.refresh_base
    inspect_native = fixtures.WorkstationPublicationTests.inspect_native

    def setUp(self):
        fixtures.WorkstationPublicationTests.setUp(self)
        with patch('dvidia_training.workstation.inspect_export', side_effect=self.inspect_native):
            original.stage(self.plan, self.workstation, self.root/'first', 'c'*40, 'd'*40, evidence=self.evidence)
        self.base = self.root/'first/publication-plan.json'
        self.updated = {**self.native_files,
            'workstation.json': original.encoded({'native_asset_bundle': {'path':'native-assets.bin.gz'}}),
            'workstation-integrity.json': b'{"updated":true}',
            'native-assets.bin.gz': b'bounded gzip fixture'}
        self.new = self.root/'new'
        self.new.mkdir()
        for name, raw in self.updated.items():
            (self.new/name).write_bytes(raw)

    def stage(self):
        with patch('dvidia_training.workstation.inspect_export', return_value={}):
            return delivery.stage(self.base, self.new, self.evidence, self.root/'delivery', 'e'*40, 'f'*40)

    def test_preserves_original_receipts_and_native_bytes_and_binds_new_transport(self):
        self.stage()
        old, new = self.root/'first/space', self.root/'delivery/space'
        for path in old.rglob('*'):
            if not path.is_file():
                continue
            name = path.relative_to(old).as_posix()
            if name in {'provenance.json', 'README.md'} or name in {
                    original.NAMESPACE+'/'+value for value in delivery.MUTABLE}:
                continue
            self.assertEqual(path.read_bytes(), (new/name).read_bytes())
        before = json.loads((old/'provenance.json').read_bytes())['live_workstation']
        after = json.loads((new/'provenance.json').read_bytes())['live_workstation']
        self.assertEqual(before['evidence'], after['evidence'])
        self.assertEqual(after['original_source_commit'], 'c'*40)
        self.assertEqual(after['source_commit'], 'e'*40)
        plan = publisher.checked_plan(self.root/'delivery/publication-plan.json')
        self.assertEqual(plan['expected_remote_commit'], 'f'*40)
        self.assertEqual(plan['remove_files'], [])

    def test_refuses_physics_changes_or_asset_removal(self):
        (self.new/'scene.xml').write_bytes(b'<changed/>')
        with self.assertRaisesRegex(ValueError, 'changed physics'):
            self.stage()
        (self.new/'scene.xml').write_bytes(self.native_files['scene.xml'])
        (self.new/'mujoco.wasm').unlink()
        with self.assertRaisesRegex(ValueError, 'retain'):
            self.stage()
        self.assertFalse((self.root/'delivery').exists())

    def test_refuses_changed_controller_metadata_and_unknown_extra_files(self):
        (self.new/'workstation.json').write_bytes(original.encoded({'controller':'changed', 'native_asset_bundle':{}}))
        with self.assertRaisesRegex(ValueError, 'controller'):
            self.stage()
        (self.new/'workstation.json').write_bytes(self.updated['workstation.json'])
        (self.new/'extra.json').write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, 'only its transport bundle'):
            self.stage()

    def test_native_bundle_allowlist_is_scoped(self):
        publisher._artifact_name('workstation/v01/native-assets.bin.gz')
        for name in ('other/native-assets.bin.gz', 'workstation/v02/native-assets.bin.gz'):
            with self.assertRaises(publisher.PublicationError):
                publisher._artifact_name(name)


if __name__ == '__main__':
    unittest.main()

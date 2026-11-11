"""Explicit reference-source revisions; no historical artifact migration."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import unittest
from unittest.mock import patch

from dvidia_training import camera_action_reference_export as ref
from tests import test_camera_action_reference_export as fixtures


class ReferenceSourceCompatibilityTests(unittest.TestCase):
    def test_legacy_pins_preserved_and_only_exact_reviewed_capture_revision_admitted(self):
        self.assertEqual(ref.SOURCE_PINS['piper_action_capture.py'],
            'da037b2198cf7d2f3a9aa40fc19a4c0f5cc375a22c5dd75606e2c1856f6efc7f')
        legacy = deepcopy(ref.SOURCE_PINS)
        reviewed = {**legacy,'piper_action_capture.py':ref.REVIEWED_CAPTURE_SHA256}
        self.assertEqual(ref._source_revision(legacy),'frozen-reference-v01')
        self.assertEqual(ref._source_revision(reviewed),'capture-latency-v02')
        self.assertNotEqual(ref.cache._digest(legacy),ref.cache._digest(reviewed))
        for source in (legacy,reviewed):
            for name in source:
                bad = {**source,name:'0'*64}
                with self.subTest(revision=source['piper_action_capture.py'],file=name), self.assertRaises(ValueError):
                    ref._source_revision(bad)
            for bad in ({**source,'unreviewed.py':'1'*64},
                        {name:value for name,value in source.items() if name!='piper_task.py'}):
                with self.assertRaises(ValueError): ref._source_revision(bad)

    def test_current_identity_contains_actual_bytes_named_revision_and_new_helper_hash(self):
        identity = ref.source_identity()
        package = Path(ref.__file__).parent
        actual = {name:sha256((package/name).read_bytes()).hexdigest() for name in ref.SOURCE_PINS}
        self.assertEqual(identity['source_sha256'],actual)
        self.assertEqual(identity['source_sha256']['piper_action_capture.py'],ref.REVIEWED_CAPTURE_SHA256)
        self.assertEqual(identity['reviewed_source_revision'],'capture-latency-v02')
        self.assertEqual(identity['helper_sha256'],sha256(Path(ref.__file__).read_bytes()).hexdigest())
        self.assertNotEqual(identity['helper_sha256'],
            'ccf6f74f6100721868ccece67e5b27dae066c1b644c8297efe111125d9517753')
        self.assertEqual(identity['parent_source'],ref.cache.source_identity())

    def test_changed_capture_limiter_model_and_export_bytes_still_refuse(self):
        read = ref.cache._bytes
        for changed in ('piper_action_capture.py','piper_task.py',
                        'camera_action_model.py','camera_action_export.py'):
            def tampered(path,maximum):
                body = read(path,maximum)
                return body+b'\n# unreviewed source change\n' if Path(path).name==changed else body
            with self.subTest(file=changed), patch.object(ref.cache,'_bytes',side_effect=tampered):
                with self.assertRaisesRegex(ValueError,'frozen source'): ref.source_identity()

    def test_historical_or_other_revision_caller_pin_cannot_authorize_new_export(self):
        fixture = fixtures.ReferenceExportTests(); fixture.setUp()
        self.addCleanup(fixture.doCleanups); self.addCleanup(fixture.tearDown)
        parent = fixture.parent()
        current = ref.source_identity()
        # Reconstruct the predecessor identity shape, including its exact helper
        # hash and prior capture source in the registered runtime. This is a
        # caller-pin negative fixture, not an imported historical cache/model.
        historical = deepcopy(current)
        historical.pop('reviewed_source_revision')
        historical['helper_sha256'] = 'ccf6f74f6100721868ccece67e5b27dae066c1b644c8297efe111125d9517753'
        historical['source_sha256']['piper_action_capture.py'] = ref.SOURCE_PINS['piper_action_capture.py']
        historical['parent_source']['runtime']['piper_action_capture.py'] = ref.SOURCE_PINS['piper_action_capture.py']
        other_revision = deepcopy(current)
        other_revision['reviewed_source_revision'] = 'frozen-reference-v01'
        other_revision['source_sha256']['piper_action_capture.py'] = ref.SOURCE_PINS['piper_action_capture.py']
        for index,other in enumerate((historical,other_revision)):
            output = fixture.root/f'refused-{index}'
            self.assertNotEqual(ref.cache._digest(other),ref.source_digest())
            with self.assertRaisesRegex(ValueError,'independent caller expectation'):
                fixture.build(parent,output=output,expected_source_sha256=ref.cache._digest(other))
            self.assertFalse(output.exists())


if __name__=='__main__': unittest.main()

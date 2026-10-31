from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import publish_arm_interfaces_v01 as publisher


class PublicationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.dataset = self.root/'dataset'; self.dataset.mkdir()
        for name in publisher.NAMES:
            (self.dataset/name).write_bytes(b'synthetic publication fixture')
        self.plan = self.root/'plan.json'
        self.receipt = publisher.stage(self.dataset, self.plan)

    def test_staging_and_checking_are_credential_free(self):
        with patch.object(publisher, 'client', side_effect=AssertionError('Credential accessed')):
            result = publisher.checked(self.plan, self.receipt['plan_sha256'])
        self.assertEqual(result['repo_id'], 'Dvidia/arm-interfaces-v01')
        self.assertEqual(len(result['files']), 6)

    def test_mutated_payload_and_unreviewed_extra_files_fail(self):
        (self.dataset/'README.md').write_bytes(b'changed')
        with self.assertRaises(publisher.PublicationError):
            publisher.checked(self.plan, self.receipt['plan_sha256'])
        (self.dataset/'README.md').write_bytes(b'synthetic publication fixture')
        (self.dataset/'unreviewed.json').write_text('{}')
        with self.assertRaises(publisher.PublicationError):
            publisher.checked(self.plan, self.receipt['plan_sha256'])

    def test_resealed_changed_destination_is_rejected(self):
        value = json.loads(self.plan.read_bytes())
        for field, replacement in [('repo_id', 'Dvidia/robot-simulation-lab'),
                                   ('repo_type', 'space'), ('private', True)]:
            changed = deepcopy(value); changed[field] = replacement
            raw = publisher.encoded(changed); self.plan.write_bytes(raw)
            with self.assertRaises(publisher.PublicationError):
                publisher.checked(self.plan, sha256(raw).hexdigest())

    def test_existing_dataset_cannot_be_overwritten(self):
        class Info:
            private = False
            sha = 'a'*40
            siblings = [type('File', (), {'rfilename': 'README.md'})()]
        with patch.object(publisher, 'client') as get_client:
            api = unittest.mock.Mock()
            api.repo_info.return_value = Info()
            get_client.return_value = api, 'synthetic-fixture'
            with self.assertRaisesRegex(publisher.PublicationError, 'not empty'):
                publisher.publish(self.plan, self.receipt['plan_sha256'], self.root/'cache')
            api.create_commit.assert_not_called()

    def test_mutation_at_payload_freeze_is_rejected_before_credentials(self):
        original = publisher.read
        def changed(path, *args, **kwargs):
            return b'changed after review' if Path(path).name == 'README.md' else original(path, *args, **kwargs)
        with patch.object(publisher, 'read', side_effect=changed), patch.object(publisher, 'client') as client:
            with self.assertRaisesRegex(publisher.PublicationError, 'freezing'):
                publisher.publish(self.plan, self.receipt['plan_sha256'], self.root/'cache')
            client.assert_not_called()


if __name__ == '__main__':
    unittest.main()

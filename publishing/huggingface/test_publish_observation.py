"""Offline publication-boundary tests; no credential or network use."""
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from publish import PublicationError
from publish_observation import checked_plan


class ObservationPublicationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.space = self.root / 'space'
        self.space.mkdir()
        for name, raw in {'README.md': b'---\nsdk: static\napp_file: index.html\n---\nSynthetic review demo.\n',
                          'index.html': b'<!doctype html><p>synthetic</p>', 'LICENSE': b'MIT'}.items():
            (self.space / name).write_bytes(raw)
        self.plan = {'kind': 'dvidia.observation-lab-publication', 'schema_version': 1,
                     'namespace': 'Dvidia', 'private': False, 'repository': {
                         'repo_id': 'Dvidia/observation-lab', 'repo_type': 'space',
                         'space_sdk': 'static', 'directory': 'space', 'files': []}}
        self.path = self.root / 'publication-plan.json'
        self.refresh()

    def refresh(self):
        self.plan['repository']['files'] = [{'path': p.name, 'bytes': p.stat().st_size,
            'sha256': sha256(p.read_bytes()).hexdigest()} for p in self.space.iterdir() if p.is_file()]
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.plan))

    def test_accepts_complete_hash_bound_static_space(self):
        self.assertEqual(checked_plan(self.path)['repository']['repo_id'], 'Dvidia/observation-lab')

    def test_rejects_unrelated_destination_and_nonstatic_space(self):
        for key, value in [('repo_id', 'Dvidia/other'), ('space_sdk', 'docker'), ('directory', '../outside')]:
            with self.subTest(key=key):
                previous = self.plan['repository'][key]
                self.plan['repository'][key] = value
                self.save()
                with self.assertRaises(PublicationError): checked_plan(self.path)
                self.plan['repository'][key] = previous

    def test_rejects_unlisted_file_and_tampered_bytes(self):
        (self.space / 'unlisted.json').write_text('{}')
        with self.assertRaises(PublicationError): checked_plan(self.path)
        (self.space / 'unlisted.json').unlink()
        (self.space / 'index.html').write_text('modified')
        with self.assertRaises(PublicationError): checked_plan(self.path)

    def test_rejects_weight_and_hidden_files(self):
        for name in ('weights.onnx', '.env'):
            with self.subTest(name=name):
                (self.space / name).write_text('x')
                self.refresh()
                with self.assertRaises(PublicationError): checked_plan(self.path)
                (self.space / name).unlink()

    def test_rejects_possible_credential_or_private_path(self):
        for raw in (b'hf_' + b'a' * 32, b'/Users/roguedev/private.mp4'):
            (self.space / 'index.html').write_bytes(raw)
            self.refresh()
            with self.assertRaises(PublicationError): checked_plan(self.path)

    def test_rejects_symlink_and_private_plan(self):
        (self.space / 'linked.json').symlink_to(self.space / 'README.md')
        with self.assertRaises(PublicationError): checked_plan(self.path)
        (self.space / 'linked.json').unlink()
        self.plan['private'] = True
        self.save()
        with self.assertRaises(PublicationError): checked_plan(self.path)

    def test_rejects_missing_static_entrypoint_declaration(self):
        (self.space / 'README.md').write_text('sdk: docker\n')
        self.refresh()
        with self.assertRaises(PublicationError): checked_plan(self.path)


if __name__ == '__main__':
    unittest.main()

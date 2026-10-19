"""Publication rejects changed history, hidden failed cases and fabricated scores."""
import base64
import json
from pathlib import Path
import sys
import tempfile
import unittest

import publish_image_goals as publisher
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from dvidia_training.image_goal_report import build_report

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jfZkAAAAASUVORK5CYII=')


class ImageGoalPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.parent, self.export = self.root / 'parent', self.root / 'export'
        self.parent.mkdir(); self.export.mkdir()
        self.original = {'index.html': b'<main><section aria-labelledby="graspability-title">prior work</section></main>',
            'README.md': b'---\nsdk: static\n---\nPrior releases\n',
            'provenance.json': b'{"previous":{"keep":true}}\n', '.gitattributes': b'*.bin filter=lfs diff=lfs merge=lfs -text\n'}
        self.tree = {name: {'blob': publisher.git_blob(raw), 'size': len(raw), 'lfs_sha256': None} for name, raw in self.original.items()}
        self.tree['prerequisites/v03/native.bin'] = {'blob': 'c'*40, 'size': 99, 'lfs_sha256': 'd'*64}
        for name, raw in self.original.items():
            (self.parent / name).write_bytes(raw)
        (self.parent / 'snapshot.json').write_bytes(publisher.encoded({'repo_id': publisher.REPO_ID,
            'namespace': publisher.NAMESPACE, 'parent_commit': 'a'*40, 'tree': self.tree}))
        case = {'id': 'missing', 'physical_truth': {'present': False, 'complete': False},
            'current_image': 'images/current.png', 'goal_image': 'images/goal.png',
            'expected_verdict': {key: 'incomplete' for key in publisher.CONDITIONS},
            'expected_presence': {key: 'absent' for key in publisher.CONDITIONS}}
        rows = [{'case_id': 'missing', 'condition': key, 'status': 'not_run'} for key in sorted(publisher.CONDITIONS)]
        self.release = {'images/current.png': PNG,
            'images/goal.png': PNG, 'cases.json': publisher.encoded({'cases': [case]}),
            'protocol.json': b'{"scope":"synthetic fixture"}\n', 'integrity.json': b'{"files":{}}\n', 'protocol.sha256': b'd'*64+b'\n', 'LICENSE': b'MIT test fixture',
            'provenance.json': publisher.encoded({'source_commit': 'b'*40, 'provider_status': 'not_run'})}
        plan = {'source_commit': 'b'*40, 'model': 'jev-1.14.0', 'conditions': sorted(publisher.CONDITIONS), 'trials': rows,
            'files': {name: publisher.record(self.release[name]) for name in ('cases.json', 'protocol.json', 'integrity.json', 'protocol.sha256', 'images/current.png', 'images/goal.png')}}
        self.release['evaluation-plan.json'] = publisher.encoded(plan)
        self.release['results.json'] = publisher.encoded({'model': 'jev-1.14.0', 'status': 'not_run', 'metrics': None,
            'results': rows, 'evaluation_plan_sha256': publisher.record(self.release['evaluation-plan.json'])['sha256']})
        self.save()
        viewer = self.root / 'viewer'
        build_report(self.export, viewer)
        for path in viewer.rglob('*'):
            if path.is_file():
                self.release[path.relative_to(viewer).as_posix()] = path.read_bytes()
        self.save()

    def save(self):
        for name, raw in self.release.items():
            path = self.export / name
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        (self.export / 'publication-integrity.json').write_bytes(publisher.encoded({'files': {name: publisher.record(raw) for name, raw in self.release.items()}}))

    def stage(self):
        return publisher.stage(self.export, self.parent, self.root / 'stage', 'b'*40)

    def test_new_namespace_preserves_history_and_visible_pending_status(self):
        result = self.stage()
        plan, payload = publisher.check(result['plan'])
        self.assertEqual(plan['parent_tree'], self.tree)
        self.assertEqual(set(payload) & self.tree.keys(), publisher.ROOT_FILES)
        self.assertIn(self.original['README.md'], payload['README.md'])
        self.assertIn(b'not yet measured', payload['index.html'])
        self.assertEqual(json.loads(payload['provenance.json'])['previous'], {'keep': True})
        self.assertEqual(payload[publisher.NAMESPACE + '/integrity.json'], self.release['integrity.json'])
        self.assertIn(publisher.NAMESPACE + '/publication-integrity.json', payload)

    def test_frozen_checksum_extension_is_narrowly_bounded(self):
        self.release['protocol.sha256'] = b'not a checksum\n'; self.save()
        with self.assertRaisesRegex(publisher.PublicationError, 'SHA-256 digest'):
            self.stage()

    def test_omitted_unrun_trial_cannot_shrink_denominator(self):
        result = json.loads(self.release['results.json']); result['results'].pop()
        self.release['results.json'] = publisher.encoded(result); self.save()
        with self.assertRaisesRegex(publisher.PublicationError, 'Every planned case'):
            self.stage()

    def test_pending_provider_cannot_publish_measured_accuracy(self):
        result = json.loads(self.release['results.json']); result['metrics'] = {'accuracy': 1.0}
        self.release['results.json'] = publisher.encoded(result); self.save()
        with self.assertRaisesRegex(publisher.PublicationError, 'no measured provider metrics'):
            self.stage()

    def test_raw_response_and_credentials_are_rejected(self):
        result = json.loads(self.release['results.json']); result['response'] = {'private': 'raw'}
        self.release['results.json'] = publisher.encoded(result); self.save()
        with self.assertRaisesRegex(publisher.PublicationError, 'raw provider responses'):
            self.stage()

    def test_resealed_changed_fixture_or_viewer_label_is_rejected(self):
        self.release['images/current.png'] = b'new image'; self.save()
        with self.assertRaisesRegex(publisher.PublicationError, 'Frozen fixture bytes'):
            self.stage()
        self.release['images/current.png'] = PNG
        report = json.loads(self.release['report.json']); report['cases'][0]['ground_truth']['complete'] = True
        self.release['report.json'] = publisher.encoded(report); self.save()
        with self.assertRaisesRegex(publisher.PublicationError, 'Viewer labels'):
            self.stage()

    def test_staged_payload_tampering_and_parent_tampering_are_rejected(self):
        result = self.stage()
        (self.root / 'stage/payload/index.html').write_text('changed')
        with self.assertRaisesRegex(publisher.PublicationError, 'Payload changed'):
            publisher.check(result['plan'])
        (self.parent / 'README.md').write_text('changed')
        with self.assertRaisesRegex(publisher.PublicationError, 'root bytes changed'):
            publisher.stage(self.export, self.parent, self.root / 'other-stage', 'b'*40)

    def test_resealed_fabricated_provider_answer_or_summary_is_rejected(self):
        public = json.loads(self.release['report.json'])
        public['results'][0].update(status='ok', verdict='complete', p_complete=1,
            decision={'present': True, 'complete': True})
        self.release['report.json'] = publisher.encoded(public)
        viewer_seal = json.loads(self.release['report-integrity.json'])
        viewer_seal['files']['report.json'] = publisher.record(self.release['report.json'])
        self.release['report-integrity.json'] = publisher.encoded(viewer_seal)
        self.save()
        with self.assertRaisesRegex(publisher.PublicationError, 'independently regenerated'):
            self.stage()


if __name__ == '__main__':
    unittest.main()

"""Append-only publication rejects forged verdicts, scores and stale artifacts."""
from copy import deepcopy
from hashlib import sha256
import base64
import json
from pathlib import Path
import sys
import tempfile
import unittest

import publish_local_goal_verifier as publisher
from publish_prerequisites import git_blob
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO))
from dvidia_training.local_goal_verifier import GoalContract, verify_goal
from dvidia_training.local_goal_benchmark import score
from dvidia_training.local_goal_verifier_report import build_report
from tests.test_local_goal_verifier import packet

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jfZkAAAAASUVORK5CYII=')


class LocalVerifierPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.export, self.parent = self.root / 'export', self.root / 'parent'
        self.export.mkdir(); self.parent.mkdir()
        self.original = {'index.html': b'<main><section aria-labelledby="image-goals-results-title">measured Jev</section><section aria-labelledby="image-goals-title">original protocol</section></main>',
            'README.md': b'---\nsdk: static\n---\nOriginal releases\n', 'provenance.json': b'{"image_goals_v01":{"keep":true},"image_goals_v02":{"keep":true}}\n', '.gitattributes': b'*.bin filter=lfs diff=lfs merge=lfs -text\n'}
        self.tree = {name: {'blob': git_blob(raw), 'size': len(raw), 'lfs_sha256': None} for name, raw in self.original.items()}
        for name in ('image-goals/v01/index.html', 'image-goals/v02/results.json', 'prerequisites/v03/native.bin'):
            self.tree[name] = {'blob': 'c'*40, 'size': 99, 'lfs_sha256': 'd'*64 if name.endswith('.bin') else None}
        for name, raw in self.original.items(): (self.parent / name).write_bytes(raw)
        (self.parent / 'snapshot.json').write_bytes(publisher.encoded({'repo_id': publisher.REPO_ID, 'namespace': publisher.NAMESPACE, 'parent_commit': 'a'*40, 'tree': self.tree}))
        p = packet(); goal = GoalContract('green-shoe', 'workstation', (.4, -.1), 0.)
        cases, rows = [], []
        for identifier, cohort, truth, now, expected in [('positive', 'native', True, .6, 'complete'), ('stale', 'evidence_fault', True, 1., 'unknown'), ('forged', 'blind_spot', False, .6, 'complete')]:
            cases.append({'id': identifier, 'cohort': cohort, 'scenario': identifier, 'layout_id': 'layout-01', 'contract': {'object_id': 'green-shoe', 'frame_id': 'workstation', 'goal_xy_m': [.4, -.1], 'goal_yaw_rad': 0.}, 'now_s': now, 'packet_file': 'packet.json', 'state_file': 'state.json', 'current_image': 'scene.png', 'goal_image': 'scene.png', 'physical_truth': {'present': True, 'complete': truth}, 'expected_verdict': expected, 'note': 'Synthetic fixture.', 'generation_valid': True})
            rows.append({'case_id': identifier, 'decision': verify_goal(p, contract=goal, now_s=now), 'timing_ns': [100000, 200000]})
        names = ['src/dvidia_training/'+name+'.py' for name in ('local_goal_verifier', 'local_goal_benchmark', 'local_goal_scenes', 'image_goal_scenes', 'image_goal_benchmark', 'jev_client', 'workstation', 'robot_profiles', 'reference_sim', 'asset_library')]
        names += ['src/dvidia_training/robot_profiles.lock.json', 'src/dvidia_training/asset_catalog.json']
        protocol = {'schema': 'dvidia.local-goal-protocol/v1', 'source': {'commit': 'b'*40, 'files': {name: sha256((REPO/name).read_bytes()).hexdigest() for name in names}}, 'planned_cases': len(cases), 'repeats': 2, 'provider_calls': 0}
        self.release = {'protocol.json': publisher.encoded(protocol), 'cases.json': publisher.encoded({'schema': 'dvidia.local-goal-cases/v1', 'cases': cases}), 'packet.json': publisher.encoded(p), 'state.json': b'{"native":true}\n', 'scene.png': PNG}
        self.release['integrity.json'] = publisher.encoded({'schema': 'dvidia.local-goal-integrity/v1', 'files': {name: publisher.record(raw) for name, raw in self.release.items()}})
        self.release.update({'results.json': publisher.encoded({'schema': 'dvidia.local-goal-results/v1', 'status': 'completed', 'source_commit': 'b'*40, 'protocol_sha256': publisher.record(self.release['protocol.json'])['sha256'], 'input_integrity_sha256': publisher.record(self.release['integrity.json'])['sha256'], 'provider_calls': 0, 'results': rows, 'metrics': score(rows, cases)}), 'provenance.json': publisher.encoded({'publication_source_commit': 'e'*40, 'benchmark_source_commit': 'b'*40}), 'LICENSE': b'MIT test fixture\n'})
        self.save()
        viewer = self.root / 'viewer'
        build_report(self.export, viewer)
        for path in viewer.rglob('*'):
            if path.is_file(): self.release[path.relative_to(viewer).as_posix()] = path.read_bytes()
        self.save()

    def save(self):
        for name, raw in self.release.items():
            p = self.export / name
            p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(raw)
        (self.export / 'publication-integrity.json').write_bytes(publisher.encoded({'files': {name: publisher.record(raw) for name, raw in self.release.items()}}))

    def alter(self, name, mutate):
        value = json.loads(self.release[name]); mutate(value)
        self.release[name] = publisher.encoded(value); self.save()

    def check(self): return publisher.check_export(self.export, 'e'*40)

    def test_stage_preserves_both_releases_history_and_attributes(self):
        out = self.root / 'stage'
        summary = publisher.stage(self.export, self.parent, out, 'e'*40)
        plan, payload = publisher.check(summary['plan'])
        self.assertEqual(set(payload) & set(self.tree), publisher.ROOT_FILES)
        self.assertEqual(plan['parent_tree'], self.tree)
        self.assertEqual((out / 'parent-gitattributes.txt').read_bytes(), self.original['.gitattributes'])
        self.assertIn(b'measured Jev', payload['index.html'])
        self.assertIn(b'original protocol', payload['index.html'])
        self.assertIn(b'zero new Jev calls', payload['index.html'])
        for name in payload:
            self.assertTrue(name in publisher.ROOT_FILES or name.startswith('image-goals/v03/'))

    def test_resealed_fake_viewer_metrics_and_html_rejected(self):
        self.alter('report.json', lambda x: x['summaries'][0]['positive_recognition'].update(correct=999))
        with self.assertRaises(publisher.PublicationError): self.check()

    def test_resealed_fake_saved_verdict_rejected(self):
        self.alter('results.json', lambda x: x['results'][1]['decision'].update(verdict='complete'))
        with self.assertRaises(publisher.PublicationError): self.check()

    def test_resealed_fake_score_rejected(self):
        self.alter('results.json', lambda x: x['metrics']['by_cohort']['blind_spot']['physical_false_completion'].update(numerator=0))
        with self.assertRaises(publisher.PublicationError): self.check()

    def test_running_missing_duplicate_and_new_provider_calls_rejected(self):
        original = self.release['results.json']
        mutations = [lambda x: x.update(status='running'), lambda x: x.update(active_trial='positive'), lambda x: x['results'].pop(), lambda x: x['results'].append(x['results'][0]), lambda x: x.update(provider_calls=1)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                self.release['results.json'] = original; self.alter('results.json', mutate)
                with self.assertRaises(publisher.PublicationError): self.check()

    def test_native_packet_mutation_even_after_outer_reseal_rejected(self):
        self.alter('packet.json', lambda x: x['samples'][0].update(robot_contact_force_n=.2))
        with self.assertRaises(publisher.PublicationError): self.check()

    def test_unsealed_image_reference_rejected(self):
        self.alter('cases.json', lambda x: x['cases'][0].update(current_image='elsewhere.png'))
        with self.assertRaises(publisher.PublicationError): self.check()

    def test_shortened_timing_repetitions_rejected(self):
        self.alter('results.json', lambda x: x['results'][0]['timing_ns'].pop())
        with self.assertRaises(publisher.PublicationError): self.check()

    def test_private_fields_rejected_recursively(self):
        self.alter('provenance.json', lambda x: x.update(extra={'response': 'not public'}))
        with self.assertRaises(publisher.PublicationError): self.check()

    def test_changed_parent_or_existing_namespace_rejected(self):
        (self.parent / 'README.md').write_bytes(b'changed after snapshot')
        with self.assertRaises(publisher.PublicationError): publisher.stage(self.export, self.parent, self.root/'stage', 'e'*40)
        (self.parent / 'README.md').write_bytes(self.original['README.md'])
        snapshot = json.loads((self.parent/'snapshot.json').read_bytes())
        snapshot['tree']['image-goals/v03/prior.json'] = {'blob': 'f'*40, 'size': 1, 'lfs_sha256': None}
        (self.parent/'snapshot.json').write_bytes(publisher.encoded(snapshot))
        with self.assertRaises(publisher.PublicationError): publisher.stage(self.export, self.parent, self.root/'stage', 'e'*40)

    def test_browser_block_does_not_silently_become_a_pass(self):
        pending = {"browser_qa_passed": False, "browser_qa_status": "blocked-macos-locked", "static_qa_passed": True,
                   "publication_authorized_with_browser_qa_pending": True}
        html = self.release['index.html']
        evidence = {'status': 'partial-browser-blocked', 'viewer_sha256': publisher.record(html)['sha256'],
                    'static_checks': {'unique_element_ids': True}, 'browser': {'screenshots': []}}
        payload = {publisher.NAMESPACE+'/index.html': html, publisher.NAMESPACE+'/ui-qa.json': publisher.encoded(evidence)}
        qa = publisher.checked_qa(pending, payload)
        self.assertFalse(qa['browser_qa_passed'])
        self.assertEqual(qa['status'], 'pending-macos-locked')
        with self.assertRaises(publisher.PublicationError):
            publisher.checked_qa({**pending, 'publication_authorized_with_browser_qa_pending': False}, payload)
        evidence['viewer_sha256'] = '0'*64
        payload[publisher.NAMESPACE+'/ui-qa.json'] = publisher.encoded(evidence)
        with self.assertRaises(publisher.PublicationError): publisher.checked_qa(pending, payload)
        self.assertTrue(publisher.checked_qa({'browser_qa_passed': True}, payload)['browser_qa_passed'])

    def test_unknown_source_or_protocol_drift_rejected(self):
        with self.assertRaises(publisher.PublicationError): publisher.check_export(self.export, 'f'*40)
        self.alter('protocol.json', lambda x: x.update(provider_calls=1))
        with self.assertRaises(publisher.PublicationError): self.check()


if __name__ == '__main__': unittest.main()

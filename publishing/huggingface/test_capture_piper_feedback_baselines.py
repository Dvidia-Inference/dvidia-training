"""Read-only public snapshots and model integrity checks against a FakeHub."""
from hashlib import sha1, sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import os
import sys
import unittest
from unittest.mock import patch

import capture_piper_feedback_baselines as capture


def encoded(value):
    return capture.study._encoded(value)


def entry(raw, lfs=False):
    return dict(blob=sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(), size=len(raw),
                lfs_sha256=sha256(raw).hexdigest() if lfs else None)


class FakeHub:
    def __init__(self, files):
        self.files = files
        self.heads = dict(model='a'*40, dataset='b'*40, space='c'*40)
        self.private = False
        self.calls = []
        self.bad_download = False
        self.changed_revision = False

    def repo_info(self, repo_id, *, repo_type, revision):
        assert repo_id == capture.study.REPOSITORIES[repo_type]
        self.calls.append(('info', repo_type, revision))
        return SimpleNamespace(sha=('f'*40 if self.changed_revision and revision else revision or self.heads[repo_type]),
                               private=self.private, siblings=[], sdk='static')

    def list_repo_tree(self, repo_id, *, repo_type, revision, recursive):
        assert revision == self.heads[repo_type] and recursive is True
        self.calls.append(('tree', repo_type, revision))
        for path, raw in self.files[repo_type].items():
            yield SimpleNamespace(path=path, blob_id=entry(raw)['blob'], size=len(raw),
                                  lfs={'sha256': sha256(raw).hexdigest()} if path.endswith('.npz') else None)

    def download(self, repo_id, name, repo_type, revision):
        assert repo_id == capture.study.REPOSITORIES[repo_type] and revision == self.heads[repo_type]
        self.calls.append(('download', repo_type, revision, name))
        return b'corrupted readback' if self.bad_download else self.files[repo_type][name]


class PublicCaptureTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        self.output = self.home/'captured'
        files = dict(dataset={'README.md': b'# Original scientific README\n', 'native.npz': b'original LFS data'},
            space={'README.md': b'---\nsdk: static\n---\nOriginal Space\n', 'index.html': b'<html>Original root</html>'})
        self.api = FakeHub(files)
        self.initial = dict(schema=capture.study.SCHEMA+'/receipt', anonymous_readback=True,
            physical_robot_ready=False, source_commit='d'*40, plan_sha256='e'*64,
            source_integrity_sha256='f'*64,
            bindings={name+'_sha256': 'a'*64 for name in ('protocol', 'report', 'frozen')},
            repositories=[dict(repo_type=kind, repo_id=repo, commit=self.api.heads[kind],
                verified_files=len(files.get(kind, {'model.json': b''})))
                for kind, repo in capture.study.REPOSITORIES.items()])
        self.receipt = self.home/'publication-receipt.json'
        self.receipt.write_bytes(encoded(self.initial))

    def run_capture(self):
        return capture.capture(self.receipt, self.output, api=self.api, download=self.api.download)

    def test_complete_capture_matches_both_helper_schemas_and_records_exact_root_bytes(self):
        with patch.object(capture, 'public_client', side_effect=AssertionError('no authenticated client')):
            result = self.run_capture()
        dataset = (self.output/'dataset/tree.json').read_bytes()
        capture.metadata._baseline(dataset, self.api.heads['dataset'])
        space = json.loads((self.output/'space/tree.json').read_bytes())
        capture.hosting._tree_document(space, self.api.heads['space'])
        self.assertTrue(result['anonymous_readback'])
        self.assertFalse(result['remote_mutations'])
        self.assertFalse(result['credentials_accessed'])
        self.assertEqual(result['repositories']['dataset']['lfs_objects'], 1)
        for kind in ('dataset', 'space'):
            self.assertEqual((self.output/kind/'README.md').read_bytes(), self.api.files[kind]['README.md'])
        for name, receipt in result['files'].items():
            self.assertEqual(capture.study._receipt((self.output/name).read_bytes()), receipt)
        self.assertTrue(all(call[2] in self.api.heads.values() for call in self.api.calls if call[0] in ('tree', 'download')))
        self.assertEqual(len(result['files']), 5)

    def test_unfinished_receipt_rejected_before_any_network_or_output(self):
        for update in (dict(anonymous_readback=False), dict(repositories=self.initial['repositories'][:2])):
            self.receipt.write_bytes(encoded({**self.initial, **update}))
            with self.subTest(update=update), self.assertRaises(capture.PublicationError): self.run_capture()
            self.assertFalse(self.api.calls)
            self.assertFalse(self.output.exists())

    def test_moved_head_private_repo_or_false_pinned_readback_fail_without_completed_snapshot(self):
        self.api.heads['dataset'] = 'd'*40
        with self.assertRaisesRegex(capture.PublicationError, 'head moved'): self.run_capture()
        self.assertFalse(self.output.exists())
        self.api.heads['dataset'] = 'b'*40; self.api.private = True
        with self.assertRaises(capture.PublicationError): self.run_capture()
        self.api.private = False; self.api.changed_revision = True
        with self.assertRaisesRegex(capture.PublicationError, 'different revision'): self.run_capture()
        self.api.changed_revision = False; self.api.bad_download = True
        with self.assertRaisesRegex(capture.PublicationError, 'readback differs'): self.run_capture()
        self.assertFalse(self.output.exists())

    def test_missing_completed_dataset_inventory_or_unsafe_tree_is_rejected(self):
        self.api.files['dataset']['added.json'] = b'{}'
        with self.assertRaisesRegex(capture.PublicationError, 'inventory'): self.run_capture()
        self.api.files['dataset'].pop('added.json')
        self.api.files['space']['../outside.html'] = b'unsafe'
        with self.assertRaisesRegex(capture.PublicationError, 'Malformed'): self.run_capture()
        self.assertFalse(self.output.exists())

    def test_existing_output_and_artifact_symlinks_rejected_before_reads(self):
        self.output.mkdir()
        with self.assertRaisesRegex(capture.PublicationError, 'fresh'): self.run_capture()
        self.assertFalse(self.api.calls)
        self.output.rmdir(); self.output.symlink_to(self.home)
        with self.assertRaisesRegex(capture.PublicationError, 'symlink'): self.run_capture()

    def test_real_sdk_boundary_always_uses_explicit_anonymous_auth(self):
        calls = []
        sdk = SimpleNamespace(HfApi=lambda **kwargs: calls.append(('api', kwargs)) or object(),
            hf_hub_download=lambda *args, **kwargs: calls.append(('download', args, kwargs)) or b'public')
        utility = SimpleNamespace(disable_progress_bars=lambda: None)
        with patch.dict(sys.modules, huggingface_hub=sdk, **{'huggingface_hub.utils': utility}), \
                patch.dict(os.environ, {'HF_TOKEN': 'unrelated existing environment value'}), \
                patch.object(capture.logging, 'disable'):
            _, download = capture.public_client()
            self.assertEqual(download(capture.study.MODEL_REPO, 'README.md', 'model', 'a'*40), b'public')
            self.assertEqual(os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'], '1')
        self.assertIs(calls[0][1]['token'], False)
        self.assertIs(calls[1][2]['token'], False)
        self.assertEqual(calls[1][2]['revision'], 'a'*40)

    def model_fixture(self):
        totals = dict(feedback=6, blind=8, feedforward=5, teacher=24, open_jaw=0)
        report = dict(schema=capture.study.STUDY_SCHEMA, physical_robot_ready=False,
            limitations=['Synthetic simulation only.'],
            rows=[dict(case_id=f'confirmation-{case:02}', variant=variant,
                       score=dict(valid=True, success=case < successes))
                  for variant, successes in totals.items() for case in range(24)],
            variants={v: dict(scheduled=24, valid=24, successes=n) for v, n in totals.items()})
        files = {'LICENSE': b'MIT License\n', 'README.md': capture.study._card('model', report),
            'base-model.json': b'{"base":"original"}', 'candidates/jaw4/model.json': b'{"jaw":"original"}',
            'candidates/blind4/model.json': b'{"blind":"original"}',
            'contracts/runtime.whl': b'original wheel', 'report.json': encoded(report),
            'frozen.json': b'{"frozen":"original"}', 'audit.json': b'{"audit":"original"}',
            'protocol.json': b'{"protocol":"original"}'}
        contract = dict(schema='dvidia.piper-learned-grip-skill/v0.2', publisher='DVIDIA',
            physical_robot_ready=False, teacher_at_student_inference=False, confirmation=report['variants'],
            models={key: dict(path=name, sha256=sha256(files[name]).hexdigest()) for key, name in
                [('base', 'base-model.json'), ('feedback', 'candidates/jaw4/model.json'), ('blind', 'candidates/blind4/model.json')]},
            wheel=dict(path='contracts/runtime.whl', sha256=sha256(files['contracts/runtime.whl']).hexdigest()),
            frozen_sha256=sha256(files['frozen.json']).hexdigest(), audit_sha256=sha256(files['audit.json']).hexdigest())
        files['contracts/skill.json'] = encoded(contract)
        plan = dict(schema=capture.study.SCHEMA, private=False,
            bindings={key+'_sha256': sha256(files[key+'.json']).hexdigest() for key in ('report', 'protocol', 'frozen')},
            repositories=[dict(repo_type='model', repo_id=capture.study.MODEL_REPO,
                files=[dict(path=name, **capture.study._receipt(raw)) for name, raw in files.items()])])
        # Use an existing allowed archive path to exercise the immutable plan.
        name = 'contracts/dvidia_training-0.1.0-py3-none-any.whl'
        files[name] = files.pop('contracts/runtime.whl')
        contract['wheel']['path'] = name; files['contracts/skill.json'] = encoded(contract)
        plan['repositories'][0]['files'] = [dict(path=n, **capture.study._receipt(raw)) for n, raw in files.items()]
        self.plan = self.home/'publication-plan.json'; self.plan.write_bytes(encoded(plan))
        self.api.files['model'] = files
        return plan

    def test_public_model_check_verifies_every_hash_card_license_contract_and_research_counts(self):
        self.model_fixture()
        result = capture.verify_model(self.plan, self.output, api=self.api, download=self.api.download)
        self.assertEqual(result['verified_files'], len(self.api.files['model']))
        self.assertEqual(result['license'], 'mit')
        self.assertEqual(result['confirmation']['feedback']['successes'], 6)
        self.assertFalse(result['physical_robot_ready'])
        self.assertFalse(result['remote_mutations'])
        self.assertTrue(result['anonymous_readback'])
        self.assertEqual(json.loads((self.output/'model-check.json').read_bytes()), result)

    def test_public_model_changed_file_or_rehashed_bad_contract_is_rejected(self):
        plan = self.model_fixture()
        self.api.files['model']['base-model.json'] = b'{"base":"changed"}'
        with self.assertRaisesRegex(capture.PublicationError, 'immutable publication plan'):
            capture.verify_model(self.plan, self.output, api=self.api, download=self.api.download)
        self.assertFalse(self.output.exists())
        plan = self.model_fixture()
        contract = json.loads(self.api.files['model']['contracts/skill.json'])
        contract['physical_robot_ready'] = True
        raw = encoded(contract); self.api.files['model']['contracts/skill.json'] = raw
        next(r for r in plan['repositories'][0]['files'] if r['path'] == 'contracts/skill.json').update(capture.study._receipt(raw))
        self.plan.write_bytes(encoded(plan))
        with self.assertRaisesRegex(capture.PublicationError, 'research contract'):
            capture.verify_model(self.plan, self.output, api=self.api, download=self.api.download)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()

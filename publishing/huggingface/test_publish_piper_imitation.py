"""Offline checks for the fixed PiPER imitation publication path."""
import contextlib
import gzip
import hashlib
from html.parser import HTMLParser
import io
import json
from posixpath import dirname, normpath
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import publish_piper_imitation as publisher


def _record(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def _reseal(root):
    files = {path.relative_to(root).as_posix(): _record(path.read_bytes())
             for path in root.rglob('*') if path.is_file() and path.name != 'integrity.json'}
    (root / 'integrity.json').write_text(json.dumps({
        'schema': 'dvidia.piper-imitation-integrity/v1', 'files': files,
    }, sort_keys=True), encoding='utf-8')


class _LocalResources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in {'href', 'src'} and value:
                self.urls.append(value)


def _make_run(root: Path):
    def digest(raw):
        return hashlib.sha256(raw).hexdigest()

    teacher_attempt = gzip.compress(b'{"trajectory":[]}', mtime=0)
    learned_attempt = gzip.compress(b'{"trajectory":[]}', mtime=0)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('features.npy', b'fixture-numeric-array')
    samples = buffer.getvalue()
    protocol = json.dumps({'schema':'dvidia.piper-imitation-study/v1', 'variants':['teacher','learned'],
                           'cases':{'train':[{'id':'train1'}], 'confirmation':[{'id':'case1'}]}}, sort_keys=True).encode()
    model = b'{"schema":"dvidia.piper-motor-primitive/v2","selected":true}'
    shuffled = b'{"architecture":"mlp","negative_control":true}'
    demonstration = gzip.compress(b'{"demonstration":[]}', mtime=0)
    dataset = json.dumps({'schema':'test-dataset','protocol_sha256':digest(protocol),
                          'samples_sha256':digest(samples), 'rows':[{'case_id':'train1',
                          'episode':'demonstrations/train1.json.gz','sha256':digest(demonstration)}]}, sort_keys=True).encode()
    frozen = json.dumps({'protocol_sha256':digest(protocol),'dataset_sha256':digest(dataset),
                         'model_sha256':digest(model),'shuffled_model_sha256':digest(shuffled)},sort_keys=True).encode()
    opened = json.dumps({'frozen_sha256':digest(frozen),'opened_at_utc':'2026-10-26T12:00:00Z'},sort_keys=True).encode()
    adapter = b'# bounded frozen-tape resume adapter\n'
    recovery_record = json.dumps({
        'id': 'case1', 'variant': 'learned', 'tape_sha256': digest(learned_attempt),
        'frozen_sha256': digest(frozen), 'adapter_sha256': digest(adapter),
        'reason': 'configuration_tuple_list_comparison', 'physical_attempt_repeated': False,
    }, sort_keys=True).encode()
    recovery = json.dumps({
        'schema': 'dvidia.piper-imitation-driver-recovery/v1',
        'frozen_sha256': digest(frozen), 'adapter_sha256': digest(adapter),
        'serialization_guard_resumes': 1, 'completed_tapes_reexecuted': 0,
        'frozen_runtime_changed': False,
        'issue': 'Live configuration tuples compare unequal to persisted JSON lists; resume scores the same saved tape.',
        'records': ['driver-recovery/case1-learned.json'],
    }, sort_keys=True).encode()
    report = json.dumps({'schema':'dvidia.piper-imitation-study/v1','status':'completed',
        'protocol_sha256':digest(protocol),'frozen_sha256':digest(frozen),
        'training':json.loads(dataset),
        'variants':{'teacher':{'scheduled':1,'successes':0},'learned':{'scheduled':1,'successes':1}},
        'limitations':['simulation-only'], 'attempts':[
            {'id':'case1','variant':'teacher','score':{'success':False},'episode':'attempts/case1-teacher.json.gz','sha256':digest(teacher_attempt)},
            {'id':'case1','variant':'learned','score':{'success':True},'episode':'attempts/case1-learned.json.gz','sha256':digest(learned_attempt)}]},
        sort_keys=True).encode()
    audit = json.dumps({'schema':'dvidia.piper-imitation-audit/v1','status':'ready',
                        'attempt_count':2},sort_keys=True).encode()
    files = {
        'protocol.json': protocol,
        'report.json': report,
        'model/model.json': model,
        'model/shuffled-model.json': shuffled,
        'training/samples.npz': samples,
        'training/dataset.json': dataset,
        'frozen.json': frozen,
        'confirmation-opened.json': opened,
        'audit.json': audit,
        'driver-recovery.json': recovery,
        'driver-recovery/case1-learned.json': recovery_record,
        'demonstrations/train1.json.gz': demonstration,
        'attempts/case1-teacher.json.gz': teacher_attempt,
        'attempts/case1-learned.json.gz': learned_attempt,
        'attempt-starts/case1-teacher.json': b'{"started":true}',
        'viewer/index.html': (
            b'<!doctype html><html><link href="screen.css">'
            b'<a href="report.json">Report</a>'
            b'<script src="piper_imitation_viewer.bundle.js"></script></html>'),
        'viewer/screen.css': b'body { color: black; }',
        'viewer/piper_imitation_viewer.bundle.js': b"fetch('scene.json'); fetch('motion.json');",
        'viewer/scene.json': b'{"geometry_binary":"geometry.bin"}',
        'viewer/motion.json': b'{"motion_binary":"motion.bin"}',
        'viewer/geometry.bin': b'geometry',
        'viewer/motion.bin': b'motion',
        'viewer/report.json': b'{"status":"completed"}',
        'viewer/selected-replay.json': b'{"frames":[]}',
        'sources/piper_imitation.py': b'# frozen source',
        'sources/piper_imitation_driver.py': adapter,
    }
    for name, raw in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    seal = {'schema': 'dvidia.piper-imitation-integrity/v1',
            'files': {name: _record(raw) for name, raw in sorted(files.items())}}
    (root / 'integrity.json').write_text(json.dumps(seal), encoding='utf-8')


class PiperImitationPublisherTests(unittest.TestCase):
    def test_stage_plan_and_dry_run_are_offline_and_hash_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            run, output = base / 'run', base / 'stage'
            run.mkdir()
            _make_run(run)
            plan = publisher.stage(run, 'a' * 40, output)
            self.assertEqual({row['repo_id'] for row in plan['repositories']}, {
                publisher.MODEL_REPO, publisher.DATASET_REPO, publisher.SPACE_REPO})
            space = next(row for row in plan['repositories'] if row['repo_type'] == 'space')
            self.assertEqual(space['append_only_namespace'], 'learned-placement/v01')
            self.assertIn('learned-placement/v01/index.html', {item['path'] for item in space['files']})
            staged_space = output / 'space' / 'learned-placement/v01'
            landing = (staged_space / 'index.html').read_text()
            self.assertIn('url=viewer/', landing)
            self.assertIn('href="viewer/"', landing)
            self.assertEqual((staged_space / 'viewer/index.html').read_bytes(),
                             (run / 'viewer/index.html').read_bytes())
            parser = _LocalResources()
            parser.feed((staged_space / 'viewer/index.html').read_text())
            for url in parser.urls:
                if url.startswith(('https://', 'http://', '//', '#', '/')):
                    continue
                resource = normpath(f"viewer/{dirname('index.html')}/{url}")
                self.assertTrue((staged_space / resource).is_file(), f'missing relative viewer asset: {url}')
            for name in ('report.json', 'scene.json', 'motion.json', 'geometry.bin', 'motion.bin'):
                self.assertTrue((staged_space / 'viewer' / name).is_file(), name)
            self.assertFalse(any(item['path'] in {'index.html', 'README.md', 'provenance.json'}
                                 for item in space['files']))
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                summary = publisher.dry_run(output / 'publication-plan.json')
            self.assertFalse(summary['network'])
            self.assertFalse(summary['credentials_read'])
            self.assertIn('sha256', stream.getvalue())
            for row in plan['repositories']:
                card_path = (output / row['directory'] /
                             ('learned-placement/v01/README.md' if row['repo_type'] == 'space' else 'README.md'))
                card = card_path.read_text()
                self.assertIn('synthetic', card.lower())
                self.assertIn('privileged simulator', card.lower())
                self.assertIn('establish', card.lower())
            model_card = (output / 'model' / 'README.md').read_text()
            self.assertIn(publisher.SPACE_VIEWER_URL, model_card)
            self.assertIn('dvidia-piper-learned --model model.json', model_card)
            self.assertIn('pip install --force-reinstall', model_card)
            self.assertIn('piper-imitation-v0.1/dvidia_training-0.1.0-py3-none-any.whl', model_card)
            self.assertIn('high-resolution time-conditioned RBF feedforward motion primitive', model_card)
            self.assertIn('16 cubic spatial terms × 401 temporal centers, σ=0.025 s', model_card)
            self.assertIn('does not learn reactive contact correction or recovery', model_card)
            self.assertIn('serialization guard', model_card)
            self.assertIn('did not rerun physics', model_card)
            self.assertIn('zero completed tapes reexecuted', model_card)
            self.assertIn('earlier failed MLP candidate checkpoints', model_card)
            mlp_card = publisher._card('model', {
                **publisher.inventory(run),
                'model/model.json': b'{"schema":"dvidia.piper-imitation-mlp/v2"}',
            }).decode()
            self.assertIn('multilayer perceptron policy', mlp_card)
            self.assertNotIn('does not learn contact feedback or recovery', mlp_card)
            dataset_card = (output / 'dataset' / 'README.md').read_text()
            self.assertIn('earlier failed MLP candidate checkpoints', dataset_card)
            self.assertIn('development traces', dataset_card)
            self.assertIn('serialization guard', dataset_card)

    def test_integrity_seal_must_cover_every_run_file(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            _make_run(run)
            (run / 'attempts/extra.json.gz').write_bytes(gzip.compress(b'{}'))
            with self.assertRaises(publisher.PublicationError):
                publisher.inventory(run)

    def test_recovery_provenance_must_match_tape_and_zero_rerun_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            _make_run(run)
            record_path = run / 'driver-recovery/case1-learned.json'
            record = json.loads(record_path.read_text())
            record['tape_sha256'] = '0' * 64
            record_path.write_text(json.dumps(record), encoding='utf-8')
            _reseal(run)
            with self.assertRaisesRegex(publisher.PublicationError, 'Per-attempt recovery record'):
                publisher.inventory(run)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            _make_run(run)
            manifest_path = run / 'driver-recovery.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['completed_tapes_reexecuted'] = 1
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            _reseal(run)
            with self.assertRaisesRegex(publisher.PublicationError, 'recovery summary'):
                publisher.inventory(run)

    def test_staged_mutation_and_unsealed_file_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            run, output = base / 'run', base / 'stage'
            run.mkdir()
            _make_run(run)
            publisher.stage(run, 'b' * 40, output)
            file = output / 'model' / 'model/model.json'
            file.write_text('{"changed":true}')
            with self.assertRaises(publisher.PublicationError):
                publisher.checked_plan(output / 'publication-plan.json')
            file.write_text('{"schema":"dvidia.piper-motor-primitive/v2","selected":true}')
            (output / 'space' / 'learned-placement/v01' / 'unplanned.json').write_text('{}')
            with self.assertRaises(publisher.PublicationError):
                publisher.checked_plan(output / 'publication-plan.json')

    def test_private_keys_token_shapes_and_incomplete_reports_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            _make_run(run)
            (run / 'viewer/report.json').write_text('{"token":"hidden"}')
            with self.assertRaises(publisher.PublicationError):
                publisher.inventory(run)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            _make_run(run)
            (run / 'report.json').write_text('{"schema":"dvidia.piper-imitation-study/v1",'
                '"status":"running","variants":{},"limitations":["test"],"attempts":[]}')
            with self.assertRaises(publisher.PublicationError):
                publisher.inventory(run)

    def test_space_card_and_payload_preserve_entire_sealed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            _make_run(run)
            source = publisher.inventory(run)
            payloads = publisher._destination_payloads(source, b'MIT License\n')
            space = payloads['space']
            self.assertEqual(space['learned-placement/v01/report.json'], source['report.json'])
            self.assertEqual(space['learned-placement/v01/audit.json'], source['audit.json'])
            self.assertEqual(space['learned-placement/v01/driver-recovery.json'], source['driver-recovery.json'])
            self.assertEqual(space['learned-placement/v01/driver-recovery/case1-learned.json'],
                             source['driver-recovery/case1-learned.json'])
            self.assertEqual(payloads['dataset']['audit.json'], source['audit.json'])
            self.assertEqual(payloads['dataset']['driver-recovery.json'], source['driver-recovery.json'])
            self.assertEqual(payloads['dataset']['driver-recovery/case1-learned.json'],
                             source['driver-recovery/case1-learned.json'])
            self.assertEqual(payloads['model']['driver-recovery.json'], source['driver-recovery.json'])
            self.assertEqual(payloads['model']['sources/piper_imitation_driver.py'],
                             source['sources/piper_imitation_driver.py'])
            self.assertEqual(payloads['dataset']['attempt-starts/case1-teacher.json'],
                             source['attempt-starts/case1-teacher.json'])
            self.assertEqual(space['learned-placement/v01/viewer/selected-replay.json'],
                             source['viewer/selected-replay.json'])
            self.assertEqual(space['learned-placement/v01/viewer/index.html'], source['viewer/index.html'])
            self.assertEqual(space['learned-placement/v01/LICENSE'], b'MIT License\n')
            self.assertTrue(payloads['model']['README.md'].startswith(b'---\nlicense: mit'))
            self.assertTrue(payloads['dataset']['README.md'].startswith(b'---\nlicense: mit'))

    def test_publish_freezes_token_checked_bytes_after_local_client_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            run, output = base / 'run', base / 'stage'
            run.mkdir()
            _make_run(run)
            publisher.stage(run, 'c' * 40, output)
            api = Mock()
            api.whoami.side_effect = RuntimeError('stop before repository preflight')
            with patch.object(publisher, 'client', return_value=(api, 'test-token')):
                with self.assertRaisesRegex(RuntimeError, 'stop before repository preflight'):
                    publisher.publish(output / 'publication-plan.json')

    def test_staged_decoded_bound_counts_identical_tapes_once_across_repositories(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            run, output = base / 'run', base / 'stage'
            run.mkdir()
            _make_run(run)
            source = publisher.inventory(run)
            unique_decoded_bytes = sum(publisher._inspect_compressed(name, raw)
                                       for name, raw in source.items())
            plan = publisher.stage(run, 'd' * 40, output)
            with patch.object(publisher, 'MAX_DECODED_TOTAL_BYTES', unique_decoded_bytes):
                publisher.checked_plan(output / 'publication-plan.json')


if __name__ == '__main__':
    unittest.main()

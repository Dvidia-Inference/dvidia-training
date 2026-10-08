"""Exercise the actual loopback boundary, job lifecycle, and result integrity gate."""
from __future__ import annotations

import hashlib
import http.client
import io
import json
from pathlib import Path
import stat
import tempfile
import threading
import time
import unittest
import zipfile

from dvidia_training.training_studio import TrainingStudioServer, UPLOAD_LIMIT, configure


def artifact_run(source, output, *, offline=True, seed=17):
    archive = output / 'training-result.zip'
    with zipfile.ZipFile(archive, 'w') as zip_file:
        zip_file.writestr('visual-result.json', json.dumps({'kind': 'visual-learning', 'seed': seed}))
    summary = {'kind': 'dvidia.footage-training-run', 'status': 'complete',
               'counts': {'episodes': 12, 'groups': 6, 'train': 8, 'dev': 2, 'test': 2},
               'capability': {'visual_trained': True, 'movement_candidate_trained': False,
                              'robot_skill_acquired': False, 'physical_robot_ready': False},
               'elapsed_seconds': .01, 'export_file': 'training-result.zip',
               'source': str(source), 'offline': offline, 'seed': seed,
               'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
    (output / 'receipt.json').write_text(json.dumps(summary))
    return summary


def artifact_inspect(output):
    summary = json.loads((output / 'receipt.json').read_text())
    archive = output / 'training-result.zip'
    if hashlib.sha256(archive.read_bytes()).hexdigest() != summary['archive_sha256']:
        raise ValueError('Archive integrity mismatch.')
    return summary


def prepare_data(source, output, *, offline=True, seed=17):
    dataset = output / 'dataset' / 'dataset.json'
    dataset.parent.mkdir()
    dataset.write_text(json.dumps({'source': str(source), 'seed': seed, 'offline': offline}))
    return {'status': 'prepared', 'dataset_path': str(dataset),
            'counts': {'episodes': 12, 'groups': 6, 'train': 8, 'dev': 2, 'test': 2},
            'capability': {'visual_ready': True, 'movement_ready': False}}


def archive_bytes(entries):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        for name, data in entries:
            archive.writestr(name, data)
    return stream.getvalue()


class TrainingStudioHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.servers = []
        self.events = []
        self.server = self.make_server()

    def make_server(self, *, runner=artifact_run, preparer=prepare_data, offline=True):
        server = configure(self.root / str(len(self.servers)), port=0, offline=offline,
                           runner=runner, preparer=preparer, inspector=artifact_inspect)
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        thread.start()
        self.servers.append((server, thread))
        return server

    def tearDown(self):
        for event in self.events:
            event.set()
        for server, thread in self.servers:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.temporary.cleanup()

    def request(self, path, value=None, *, method='POST', raw=None, headers=None, server=None):
        server = server or self.server
        port = server.server_address[1]
        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=3)
        request_headers = {'Content-Type': 'application/json', 'Origin': f'http://127.0.0.1:{port}'}
        request_headers.update(headers or {})
        body = raw if raw is not None else (None if value is None else json.dumps(value))
        try:
            connection.request(method, path, body=body, headers=request_headers)
            response = connection.getresponse()
            data = response.read()
            if response.getheader('Content-Type', '').startswith('application/json'):
                data = json.loads(data)
            return response.status, dict(response.getheaders()), data
        finally:
            connection.close()

    def outcome(self, server=None):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status, _, state = self.request('/api/state', method='GET', server=server)
            self.assertEqual(status, 200)
            if state['status'] != 'running':
                return state
            time.sleep(.01)
        self.fail('Injected worker did not finish.')

    def train(self, server=None, **fields):
        status, _, result = self.request('/api/train', {'source': '/local/export', **fields}, server=server)
        self.assertEqual(status, 202, result)
        return self.outcome(server)

    def test_initial_ui_and_state_do_not_claim_training_or_robot_readiness(self):
        status, headers, body = self.request('/', method='GET')
        self.assertEqual(status, 200)
        self.assertIn(b'Visual learning does not control an arm', body)
        self.assertIn(b'id="train-button" disabled', body)
        self.assertIn(b'id="download"', body)
        self.assertIn("frame-ancestors 'none'", headers['Content-Security-Policy'])
        self.assertEqual(headers['X-Content-Type-Options'], 'nosniff')
        self.assertNotIn('Access-Control-Allow-Origin', headers)
        state = self.outcome()
        self.assertEqual(state['status'], 'idle')
        self.assertFalse(state['allow_online'])
        self.assertFalse(state['robot_skill_acquired'])
        self.assertFalse(state['physical_robot_ready'])
        self.assertIsNone(state['summary'])
        self.assertEqual(self.request('/download', method='GET')[0], 409)
        self.assertEqual(self.request('/api/result', method='GET')[0], 409)

    def test_browser_origin_and_host_guards_cannot_replace_a_completed_result(self):
        self.assertEqual(self.train()['status'], 'complete')
        port = self.server.server_address[1]
        cases = ({'Host': 'evil.example'}, {'Host': f'localhost:{port + 1}'},
                 {'Origin': 'https://evil.example'}, {'Origin': 'null'}, {'Sec-Fetch-Site': 'cross-site'})
        for headers in cases:
            with self.subTest(headers=headers):
                status, _, result = self.request('/api/train', {'source': '/local/other'}, headers=headers)
                self.assertEqual(status, 400)
                self.assertIn('error', result)
                self.assertEqual(self.request('/download', method='GET')[0], 200)
        self.assertEqual(self.request('/api/state', method='GET', headers={'Origin': 'null'})[0], 400)
        with self.assertRaises(ValueError):
            TrainingStudioServer(('0.0.0.0', 0), self.root)

    def test_strict_json_bounds_and_offline_url_permission_reject_before_work(self):
        cases = [('[]', {}), ('{"source":"a","source":"b"}', {}),
                 ('{"source":"a","seed":NaN}', {}), ('{"source":"a","seed":1e999}', {}),
                 ('{"source":"a","seed":true}', {}), ('{"source":"a","online":1}', {}),
                 ('{"source":"a","extra":1}', {}), ('{"source":"a"}', {'Content-Type': 'text/plain'}),
                 ('{}', {'Content-Length': '32769'}), ('{}', {'Transfer-Encoding': 'chunked'}),
                 ('{"source":"https://dvidia.org/store/a/b","online":false}', {}),
                 ('{"source":"https://dvidia.org/store/a/b","online":true}', {})]
        for raw, headers in cases:
            with self.subTest(raw=raw, headers=headers):
                status, _, result = self.request('/api/train', raw=raw, headers=headers)
                self.assertEqual(status, 400)
                self.assertIn('error', result)
                self.assertIsNone(self.server.job)
        self.assertFalse((self.server.directory / 'jobs').exists())

    def test_online_mode_is_explicit_per_request_and_uses_https_only(self):
        server = self.make_server(offline=False)
        self.assertTrue(self.outcome(server)['allow_online'])
        for source in ('http://dvidia.org/a', 'https://user:secret@dvidia.org/a',
                       'https://dvidia.org/a#fragment', 'file:///etc/passwd'):
            with self.subTest(source=source):
                self.assertEqual(self.request('/api/train', {'source': source, 'online': True}, server=server)[0], 400)
        status, _, _ = self.request('/api/train', {'source': 'https://dvidia.org/export.zip', 'online': True}, server=server)
        self.assertEqual(status, 202)
        summary = self.outcome(server)['summary']
        self.assertFalse(summary['offline'])
        self.assertEqual(summary['source'], 'https://dvidia.org/export.zip')

    def test_seed_limit_matches_intake_and_training_before_work_is_queued(self):
        for path in ('/api/prepare', '/api/train'):
            with self.subTest(path=path):
                status, _, error = self.request(path, {'source': '/local/export', 'seed': 1000000})
                self.assertEqual(status, 400)
                self.assertIn('999999', error['error'])
                self.assertIsNone(self.server.job)
        self.assertEqual(self.train(seed=999999)['summary']['seed'], 999999)

    def test_checked_dataset_reused_only_for_matching_source_seed_and_permission(self):
        source = '/local/skillspace'
        request = {'source': source, 'seed': 29}
        self.assertEqual(self.request('/api/prepare', request)[0], 202)
        checked = self.outcome()
        self.assertEqual(checked['status'], 'prepared')
        dataset = checked['summary']['dataset_path']
        self.assertFalse(checked['summary']['capability']['movement_ready'])
        self.assertEqual(self.request('/download', method='GET')[0], 409)
        self.assertEqual(self.request('/api/train', request)[0], 202)
        trained = self.outcome()
        self.assertEqual(trained['summary']['source'], dataset)
        self.assertEqual(trained['summary']['seed'], 29)
        self.assertEqual(self.request('/api/train', {'source': source, 'seed': 43})[0], 202)
        changed = self.outcome()
        self.assertEqual(changed['summary']['source'], source)
        self.assertEqual(changed['summary']['seed'], 43)

    def test_one_active_job_rejects_concurrent_prepare_train_and_upload(self):
        started, release = threading.Event(), threading.Event()
        self.events.append(release)
        def blocked(source, output, **kwargs):
            started.set()
            if not release.wait(3):
                raise RuntimeError('Test worker timed out.')
            return artifact_run(source, output, **kwargs)
        server = self.make_server(runner=blocked)
        self.assertEqual(self.request('/api/train', {'source': '/local/export'}, server=server)[0], 202)
        self.assertTrue(started.wait(1))
        identity = server.job['id']
        for path in ('/api/train', '/api/prepare'):
            self.assertEqual(self.request(path, {'source': '/local/replacement'}, server=server)[0], 409)
        raw = archive_bytes([('clip.mp4', b'placeholder')])
        self.assertEqual(self.request('/api/upload', raw=raw, headers={'Content-Type': 'application/zip'}, server=server)[0], 409)
        self.assertEqual(server.job['id'], identity)
        self.assertEqual(self.outcome_running(server)['status'], 'running')
        release.set()
        self.assertEqual(self.outcome(server)['status'], 'complete')

    def outcome_running(self, server):
        return self.request('/api/state', method='GET', server=server)[2]

    def test_failed_job_and_failed_verification_never_offer_an_old_download(self):
        self.assertEqual(self.train()['status'], 'complete')
        status, headers, archive = self.request('/download', method='GET')
        self.assertEqual(status, 200)
        self.assertEqual(headers['Content-Type'], 'application/zip')
        self.assertTrue(zipfile.is_zipfile(io.BytesIO(archive)))
        self.server.archive_path().write_bytes(b'tampered')
        status, _, result = self.request('/api/result', method='GET')
        self.assertEqual(status, 409)
        self.assertIn('verification failed', result['error'])
        self.assertEqual(self.outcome()['status'], 'failed')
        self.assertEqual(self.request('/download', method='GET')[0], 409)
        def failed(*args, **kwargs):
            raise ValueError('No usable complete footage groups.')
        server = self.make_server(runner=failed)
        state = self.train(server)
        self.assertEqual(state['status'], 'failed')
        self.assertIsNone(state['summary'])
        self.assertIn('No usable', state['error'])
        self.assertEqual(self.request('/download', method='GET', server=server)[0], 409)

    def test_invalid_new_source_clears_previous_result_but_keeps_saved_artifacts(self):
        self.assertEqual(self.train()['status'], 'complete')
        path = self.server.archive_path()
        self.assertEqual(self.request('/api/train', {'source': ''})[0], 400)
        self.assertEqual(self.outcome()['status'], 'idle')
        self.assertTrue(path.is_file())
        self.assertEqual(self.request('/download', method='GET')[0], 409)

    def test_zip_upload_uses_generated_local_filename_and_rejects_unsafe_members(self):
        raw = archive_bytes([('skillspace/metadata.json', '{}'), ('skillspace/clip.mp4', b'placeholder')])
        status, _, uploaded = self.request('/api/upload', raw=raw, headers={'Content-Type': 'application/zip'})
        self.assertEqual(status, 201)
        path = Path(uploaded['source'])
        self.assertTrue(path.is_relative_to(self.server.directory / 'uploads'))
        self.assertRegex(path.name, r'^[0-9a-f]{32}\.zip$')
        self.assertEqual(path.read_bytes(), raw)
        self.assertFalse(uploaded['checked'])
        self.assertFalse(uploaded['robot_skill_acquired'])
        before = set(path.parent.iterdir())
        unsafe = [archive_bytes([('../outside.mp4', b'a')]), archive_bytes([('/absolute.mp4', b'a')]),
                  archive_bytes([('nested\\outside.mp4', b'a')]), archive_bytes([('clip.mp4', b'a'), ('CLIP.mp4', b'b')]), b'not a ZIP']
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            info = zipfile.ZipInfo('link.mp4')
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, '../outside')
        unsafe.append(stream.getvalue())
        for raw in unsafe:
            with self.subTest(raw_size=len(raw)):
                status, _, error = self.request('/api/upload', raw=raw, headers={'Content-Type': 'application/zip'})
                self.assertEqual(status, 400)
                self.assertIn('error', error)
        self.assertEqual(self.request('/api/upload', raw=b'x', headers={'Content-Type': 'application/zip', 'Content-Length': str(UPLOAD_LIMIT + 1)})[0], 400)
        self.assertEqual(set(path.parent.iterdir()), before)

    def test_prepare_cannot_escape_workspace_and_nonfinite_result_is_explicit_failure(self):
        outside = self.root / 'outside.json'
        outside.write_text('{}')
        def escaped(*args, **kwargs):
            return {'dataset_path': str(outside)}
        server = self.make_server(preparer=escaped)
        self.assertEqual(self.request('/api/prepare', {'source': '/local/source'}, server=server)[0], 202)
        self.assertEqual(self.outcome(server)['status'], 'failed')
        self.assertIsNone(server.prepared)
        def nonfinite(*args, **kwargs):
            return {'status': 'complete', 'loss': float('nan')}
        server = self.make_server(runner=nonfinite)
        state = self.train(server)
        self.assertEqual(state['status'], 'failed')
        json.dumps(state, allow_nan=False)
        self.assertIsNone(state['summary'])
        self.assertEqual(self.request('/download', method='GET', server=server)[0], 409)

    def test_unknown_routes_never_expose_arbitrary_local_paths(self):
        for path in ('/download?path=/etc/passwd', '/../outside.json', '/api/result/anything', '/uploads/file.zip'):
            with self.subTest(path=path):
                self.assertEqual(self.request(path, method='GET')[0], 404)
        self.assertEqual(self.request('/api/unknown', {})[0], 404)


if __name__ == '__main__':
    unittest.main()

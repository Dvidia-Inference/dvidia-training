"""The research viewer exports validated evidence without changing source files."""
from hashlib import sha256
import http.client
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest

from dvidia_training.improvement_studio import ImprovementServer


FIXTURE = Path(__file__).resolve().parents[1] / 'benchmarks/sample-count-19047'
FILES = ('report.json', 'protocol.json', 'protocol.sha256', 'models-frozen.json')


class ImprovementHTTPTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # macOS's /var and /tmp prefixes are symlinks; the loader deliberately
        # requires a canonical location rather than traversing either prefix.
        self.directory = Path(self.tmp.name).resolve()
        for name in FILES:
            shutil.copyfile(FIXTURE / name, self.directory / name)
        self.server = ImprovementServer(('127.0.0.1', 0), self.directory)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
        self.tmp.cleanup()

    def request(self, path='/api/improvement', *, method='GET', headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        try:
            conn.request(method, path, headers=headers or {})
            response = conn.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            conn.close()

    def fingerprint(self):
        return {p.name: sha256(p.read_bytes()).hexdigest() for p in self.directory.iterdir()}

    def test_service_and_all_exports_are_read_only(self):
        before = self.fingerprint()
        status, headers, html = self.request('/')
        self.assertEqual(status, 200)
        self.assertIn('text/html', headers['Content-Type'])
        self.assertIn(b'DVIDIA', html)
        status, headers, raw = self.request()
        dashboard = json.loads(raw)
        self.assertEqual(status, 200)
        self.assertEqual(dashboard['format'], 'dvidia.skillspace-improvement.v1')
        self.assertEqual(headers['Cache-Control'], 'no-store')
        self.assertEqual(headers['X-Content-Type-Options'], 'nosniff')
        self.assertIn("frame-ancestors 'none'", headers['Content-Security-Policy'])
        self.assertNotIn('Access-Control-Allow-Origin', headers)
        status, headers, raw = self.request('/api/improvement/report')
        self.assertEqual(status, 200)
        self.assertIn('attachment;', headers['Content-Disposition'])
        self.assertEqual(json.loads(raw), dashboard)
        status, headers, raw = self.request('/api/improvement/brief?id=candidate-investigation')
        self.assertEqual(status, 200)
        self.assertIn('text/plain', headers['Content-Type'])
        self.assertIn(b'Provisional investigation', raw)
        status, headers, raw = self.request('/api/improvement/coverage')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)['evidence'], [])
        self.assertEqual(self.fingerprint(), before)

    def test_unexpected_host_origin_and_cross_site_cannot_read_exports(self):
        for headers in ({'Host': 'evil.test'}, {'Host': '127.0.0.1'},
                        {'Origin': 'https://evil.test'}, {'Origin': 'null'},
                        {'Sec-Fetch-Site': 'cross-site'}):
            with self.subTest(headers=headers):
                self.assertEqual(self.request(headers=headers)[0], 403)

    def test_duplicate_host_and_origin_headers_are_rejected(self):
        for header in ('Host', 'Origin'):
            with self.subTest(header=header):
                conn = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
                try:
                    conn.putrequest('GET', '/api/improvement', skip_host=True)
                    host = f'127.0.0.1:{self.server.server_port}'
                    conn.putheader('Host', host)
                    if header == 'Host':
                        conn.putheader('Host', host)
                    else:
                        conn.putheader('Origin', 'http://' + host)
                        conn.putheader('Origin', 'http://' + host)
                    conn.endheaders()
                    response = conn.getresponse()
                    self.assertEqual(response.status, 403)
                    response.read()
                finally:
                    conn.close()

    def test_traversal_and_unknown_paths_do_not_serve_local_files(self):
        for path in ('/report.json', '/../report.json', '/%2e%2e/report.json',
                     '/api/improvement?path=report.json', '/api/improvement/coverage?path=../../private'):
            with self.subTest(path=path):
                status, _, raw = self.request(path)
                self.assertEqual(status, 404)
                self.assertNotIn(str(self.directory).encode(), raw)
        for query in ('id=../../private', 'id=a&id=b', 'id=candidate-investigation&path=x', 'id='):
            with self.subTest(query=query):
                self.assertEqual(self.request('/api/improvement/brief?' + query)[0], 409)

    def test_post_is_unsupported_and_never_changes_benchmark(self):
        before = self.fingerprint()
        status, _, raw = self.request(method='POST')
        self.assertEqual(status, 405)
        self.assertIn(b'read-only', raw)
        self.assertEqual(self.fingerprint(), before)

    def test_each_export_revalidates_file_bindings(self):
        for name in ('protocol.json', 'protocol.sha256', 'models-frozen.json'):
            with self.subTest(name=name):
                path = self.directory / name
                original = path.read_bytes()
                path.write_bytes(original + b' ' if name.endswith('.json') else b'0' * 64)
                for endpoint in ('/api/improvement', '/api/improvement/report',
                                 '/api/improvement/brief?id=candidate-investigation',
                                 '/api/improvement/coverage'):
                    status, _, raw = self.request(endpoint)
                    self.assertEqual(status, 409)
                    self.assertNotIn(str(self.directory).encode(), raw)
                path.write_bytes(original)
        self.assertEqual(self.request()[0], 200)

    def test_invalid_or_missing_report_returns_controlled_conflict(self):
        path = self.directory / 'report.json'
        for payload in ('{invalid', '[]', '{"same":1,"same":2}'):
            with self.subTest(payload=payload):
                path.write_text(payload)
                self.assertEqual(self.request()[0], 409)
        path.unlink()
        self.assertEqual(self.request()[0], 409)

    def test_malformed_nested_report_returns_controlled_conflict(self):
        source = json.loads((FIXTURE / 'report.json').read_text())
        for field in ('teacher', 'comparison', 'oversized_number'):
            with self.subTest(field=field):
                report = json.loads(json.dumps(source))
                if field == 'teacher':
                    report['teacher'] = None
                elif field == 'comparison':
                    report['comparisons'][0] = None
                else:
                    report['benchmark_wall_seconds'] = 10 ** 1000
                (self.directory / 'report.json').write_text(json.dumps(report))
                status, _, raw = self.request()
                self.assertEqual(status, 409)
                self.assertNotIn(str(self.directory).encode(), raw)

    def test_non_loopback_bind_rejected_before_opening_a_socket(self):
        with self.assertRaisesRegex(ValueError, '127.0.0.1'):
            ImprovementServer(('0.0.0.0', 0), self.directory)


if __name__ == '__main__':
    unittest.main()

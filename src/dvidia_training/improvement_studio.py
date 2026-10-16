"""Read-only, loopback Skillspace improvement evidence viewer."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .improvement import brief, build_dashboard, capture_plan, load_benchmark


class ImprovementServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, development, *, evaluation=None, baseline='groups-004', diagnostics=(), recovery=None):
        if address[0] != '127.0.0.1':
            raise ValueError('The research viewer binds only to 127.0.0.1.')
        self.development, self.evaluation, self.baseline = development, evaluation, baseline
        self.diagnostics = tuple(diagnostics)
        self.recovery = tuple(recovery or ())
        if (self.diagnostics or self.recovery) and not evaluation:
            raise ValueError('Experiments require the evaluated benchmark.')
        self.dashboard()  # Fail before binding if evidence is incomplete.
        super().__init__(address, ImprovementHandler)

    def dashboard(self):
        evaluation = load_benchmark(self.evaluation) if self.evaluation else None
        view = build_dashboard(load_benchmark(self.development), self.baseline, evaluation)
        if self.diagnostics:
            from .grip_diagnostic_view import attach_diagnostics, load_diagnostic
            view = attach_diagnostics(view, [load_diagnostic(path, evaluation) for path in self.diagnostics])
        if self.recovery:
            from .recovery_view import attach_recovery, load_recovery
            view = attach_recovery(view, [load_recovery(path, evaluation) for path in self.recovery])
        return view


class ImprovementHandler(BaseHTTPRequestHandler):
    timeout = 10

    def log_message(self, *_args):
        pass

    def local_request(self):
        port = self.server.server_address[1]
        valid = (f'127.0.0.1:{port}', f'localhost:{port}')
        hosts, origins = self.headers.get_all('Host', []), self.headers.get_all('Origin', [])
        if len(hosts) != 1 or hosts[0] not in valid:
            raise ValueError('Use the printed local viewer address.')
        if len(origins) > 1 or (origins and origins[0] not in tuple('http://' + h for h in valid)):
            raise ValueError('Use the local viewer for this request.')
        if self.headers.get('Sec-Fetch-Site') == 'cross-site':
            raise ValueError('Cross-site requests are not allowed.')

    def respond(self, value, status=200, *, mime='application/json; charset=utf-8', filename=None):
        raw = value if isinstance(value, bytes) else json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        if filename:
            self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        try:
            self.local_request()
        except ValueError as exc:
            self.respond({'error': str(exc)}, 403)
            return
        try:
            url = urlsplit(self.path)
            if url.path == '/' and not url.query:
                self.respond(Path(__file__).with_name('improvement_studio.html').read_bytes(), mime='text/html; charset=utf-8')
            elif url.path in ('/api/improvement', '/api/improvement/report') and not url.query:
                self.respond(self.server.dashboard(), filename='skillspace-improvement.json' if url.path.endswith('/report') else None)
            elif url.path == '/api/improvement/brief':
                query = parse_qs(url.query, strict_parsing=True)
                if set(query) != {'id'} or len(query['id']) != 1:
                    raise ValueError('Choose one investigation.')
                self.respond(brief(self.server.dashboard(), query['id'][0]).encode(),
                             mime='text/plain; charset=utf-8', filename='next-investigation.txt')
            elif url.path == '/api/improvement/coverage' and not url.query:
                self.respond(capture_plan(self.server.dashboard()), filename='skillspace.coverage.json')
            else:
                self.respond({'error': 'Not found.'}, 404)
        except (ValueError, KeyError, TypeError, OSError):
            self.respond({'error': 'The requested evidence is unavailable or no longer validates. Check the local benchmark files.'}, 409)

    def do_POST(self):
        self.respond({'error': 'This viewer is read-only. Run a bounded experiment from the local CLI.'}, 405)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('development', type=Path)
    parser.add_argument('--evaluation', type=Path)
    parser.add_argument('--diagnostic', type=Path, action='append', default=[],
                        help='Measured diagnostics in chronological order; repeat for a follow-up.')
    parser.add_argument('--recovery', type=Path, action='append', default=[],
                        help='Original-limit recovery comparisons in chronological order; repeat for a follow-up.')
    parser.add_argument('--baseline', default='groups-004')
    parser.add_argument('--port', type=int, default=8391)
    args = parser.parse_args()
    with ImprovementServer(('127.0.0.1', args.port), args.development,
                           evaluation=args.evaluation, baseline=args.baseline, diagnostics=args.diagnostic,
                           recovery=args.recovery) as server:
        print(f'DVIDIA Skillspace improvement: http://127.0.0.1:{server.server_address[1]}/', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()

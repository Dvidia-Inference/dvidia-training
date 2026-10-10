"""Loopback-only observation review, with private media and append-only revisions."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import mimetypes
import os
from pathlib import Path
import re
import threading
from urllib.parse import urlsplit

from .footage_data import _json, _local_file, canonical
from .observation_pipeline import load_episodes
from .observations import latest_review, validate_review


class ObservationServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, directory):
        if address[0] != '127.0.0.1':
            raise ValueError('The observation lab binds only to 127.0.0.1.')
        self.directory = Path(directory).resolve(strict=True)
        self.receipt, self.episodes = load_episodes(self.directory)
        self.lock = threading.RLock()
        self.media_signatures = {}
        super().__init__(address, ObservationHandler)

    @contextmanager
    def media(self, eid, *, verify=False):
        source = self.episodes[eid]['source']
        path = _local_file(self.directory/eid, source['video_path'])
        with path.open('rb') as handle:
            def signature():
                s = os.fstat(handle.fileno())
                return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            before = signature()
            if before[2] != source['bytes']:
                raise ValueError('Source media changed after validation.')
            with self.lock:
                if verify or self.media_signatures.get(eid) != before:
                    digest = hashlib.sha256()
                    while chunk := handle.read(1024*1024):
                        digest.update(chunk)
                    if signature() != before or digest.hexdigest() != source['sha256']:
                        raise ValueError('Source media changed after validation.')
                    self.media_signatures[eid] = before
                    handle.seek(0)
            yield path, handle, before[2]

    def episode(self, eid):
        result = deepcopy(self.episodes[eid])
        result['review'] = latest_review(self.directory/eid, result)
        return result

    def save(self, eid, payload):
        with self.lock:
            episode = self.episode(eid)
            with self.media(eid, verify=True):
                pass
            review = validate_review(payload, episode, episode['review'])
            target = self.directory/eid/f"review-{review['revision']:06d}.json"
            # Exclusive creation plus a same-directory atomic link provides
            # no-overwrite safety even if another server reviews this run.
            temporary = target.with_suffix(f'.{os.getpid()}.{threading.get_ident()}.tmp')
            try:
                with temporary.open('xb') as handle:
                    handle.write(canonical(review))
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    os.link(temporary, target)
                except FileExistsError as exc:
                    raise RuntimeError('This review changed. Reload before saving.') from exc
            finally:
                temporary.unlink(missing_ok=True)
            return review


class ObservationHandler(BaseHTTPRequestHandler):
    server_version = 'DvidiaObservation/1'

    def log_message(self, *_):
        pass

    def origin_ok(self, *, write=False):
        expected = f'127.0.0.1:{self.server.server_port}'
        origin = self.headers.get('Origin')
        return (self.headers.get('Host') == expected
                and self.headers.get('Sec-Fetch-Site') not in ('cross-site',)
                and (origin == 'http://'+expected if write else origin in (None, 'http://'+expected)))

    def send(self, status, data, content_type='application/json; charset=utf-8', extra=None):
        raw = data if type(data) is bytes else canonical(data)
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(raw)

    def video(self, eid):
        with self.server.media(eid) as (path, handle, size):
            return self.stream_video(path, handle, size)

    def stream_video(self, path, handle, size):
        start, end, status = 0, size-1, 200
        requested = self.headers.get('Range')
        if requested:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested)
            if not match or not any(match.groups()):
                return self.send(416, {'error': 'Unsupported byte range.'}, extra={'Content-Range': f'bytes */{size}'})
            left, right = match.groups()
            if left:
                start, end = int(left), min(int(right), size-1) if right else size-1
            else:
                start, end = max(0, size-int(right)), size-1
            if start > end or start >= size or not left and int(right) == 0:
                return self.send(416, {'error': 'Range is outside the video.'}, extra={'Content-Range': f'bytes */{size}'})
            status = 206
        self.send_response(status)
        self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        self.send_header('Content-Length', str(end-start+1))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Cache-Control', 'private, no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
        if status == 206:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        if self.command == 'HEAD':
            return
        handle.seek(start)
        remaining = end-start+1
        while remaining:
            chunk = handle.read(min(64*1024, remaining))
            if not chunk:
                break
            self.wfile.write(chunk)
            remaining -= len(chunk)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self.origin_ok():
            return self.send(403, {'error': 'Use this lab from its loopback address.'})
        path = urlsplit(self.path).path
        try:
            if path == '/':
                return self.send(200, Path(__file__).with_name('observation_studio.html').read_bytes(), 'text/html; charset=utf-8')
            if path == '/api/episodes':
                rows = []
                for eid in self.server.episodes:
                    e = self.server.episode(eid)
                    rows.append({'id': eid, 'title': e['title'], 'duration_seconds': e['duration_seconds'],
                                 'video_url': e['video_url'], 'status': e['review']['status']})
                return self.send(200, {'episodes': rows, 'failures': self.server.receipt['failures']})
            match = re.fullmatch(r'/api/episodes/([A-Za-z0-9_-]+)(/video)?', path)
            if match and match[1] in self.server.episodes:
                return self.video(match[1]) if match[2] else self.send(200, self.server.episode(match[1]))
            return self.send(404, {'error': 'No such observation resource.'})
        except (BrokenPipeError, ConnectionResetError):
            return
        except (ValueError, OSError, KeyError):
            return self.send(409, {'error': 'Source or review evidence changed. Inspect the run before continuing.'})

    def do_POST(self):
        if not self.origin_ok(write=True):
            return self.send(403, {'error': 'Review changes require the same local origin.'})
        path = urlsplit(self.path).path
        match = re.fullmatch(r'/api/episodes/([A-Za-z0-9_-]+)/review', path)
        if not match or match[1] not in self.server.episodes:
            return self.send(404, {'error': 'No such episode.'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if (self.headers.get_content_type() != 'application/json' or self.headers.get('Transfer-Encoding')
                    or not 0 < length <= 1024*1024):
                return self.send(400, {'error': 'Use bounded JSON with a content length.'})
            self.connection.settimeout(10)
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError('Incomplete review request.')
            payload = _json(raw, maximum=1024*1024)
            review = self.server.save(match[1], payload)
            return self.send(200, {'review': review})
        except RuntimeError as exc:
            return self.send(409, {'error': str(exc)})
        except (ValueError, OSError) as exc:
            return self.send(400, {'error': str(exc) if isinstance(exc, ValueError) else 'Could not store the review.'})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--port', type=int, default=8280)
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error('Use an unprivileged local port.')
    server = ObservationServer(('127.0.0.1', args.port), args.directory)
    print(f'Observation lab: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()

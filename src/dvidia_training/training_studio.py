"""A loopback-only, data-only footage training interface with one CPU worker."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import multiprocessing
from pathlib import Path, PurePosixPath
import re
import stat
import threading
import time
from urllib.parse import urlsplit
import uuid
import zipfile

JSON_LIMIT = 32 * 1024
UPLOAD_LIMIT = 24 * 1024 * 1024
CLIP_LIMIT = 128 * 1024 * 1024
EXPANDED_LIMIT = 512 * 1024 * 1024


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON keys are not allowed.')
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError('Nonfinite JSON values are not allowed.')

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
    json.dumps(value, allow_nan=False)
    return value


def _progress_file(output, event):
    if type(event) is not dict:
        return
    stage = event.get('stage')
    if stage not in ('intake', 'training', 'export', 'complete'):
        return
    message = event.get('message', '')
    if not isinstance(message, str):
        return
    target = Path(output) / 'studio-progress.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps({'stage': stage, 'message': message[:600]}, allow_nan=False))
    temporary.replace(target)


def _pipeline_worker(kind, source, output, offline, seed):
    # Import in the spawned worker: offline guards and CPU training stay isolated
    # from the local HTTP server. No source archive is imported as Python code.
    from . import footage_pipeline
    if kind == 'prepare':
        return footage_pipeline.prepare_run(source, Path(output), offline=offline, seed=seed)
    return footage_pipeline.run(source, Path(output), offline=offline, seed=seed,
                                progress=lambda event: _progress_file(output, event))


def _inspect_run(path):
    from .footage_pipeline import inspect_run
    return inspect_run(Path(path))


def validate_zip(raw):
    """Reject unsafe archive structure before saving; extraction is pipeline-owned."""
    if not raw or len(raw) > UPLOAD_LIMIT:
        raise ValueError('ZIP uploads must be at most 24 MiB. Use a local folder for larger footage.')
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = archive.infolist()
            if not members or len(members) > 4096:
                raise ValueError('Use a nonempty ZIP with at most 4096 entries.')
            expanded = 0
            names = set()
            for member in members:
                name = member.orig_filename
                parts = PurePosixPath(name).parts
                mode = member.external_attr >> 16
                if (not name or '\\' in name or '\x00' in name or ':' in name
                        or name.startswith('/') or any(part in ('.', '..') for part in name.split('/'))
                        or not parts or stat.S_ISLNK(mode)
                        or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR))
                        or member.flag_bits & 1):
                    raise ValueError('ZIP paths must be relative, unencrypted regular files or directories.')
                normalized = PurePosixPath(name).as_posix().casefold()
                if normalized in names:
                    raise ValueError('Duplicate ZIP member names are not allowed.')
                names.add(normalized)
                if member.file_size > CLIP_LIMIT:
                    raise ValueError('Each ZIP member must be at most 128 MiB.')
                expanded += member.file_size
                if expanded > EXPANDED_LIMIT:
                    raise ValueError('ZIP contents must expand to at most 512 MiB.')
    except zipfile.BadZipFile as exc:
        raise ValueError('Upload a valid ZIP archive.') from exc


class TrainingStudioServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, directory, offline=True, *, runner=None, preparer=None, inspector=None):
        if address[0] != '127.0.0.1':
            raise ValueError('The training studio must bind to 127.0.0.1.')
        if type(offline) is not bool:
            raise ValueError('Offline server policy must be a boolean.')
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.runner, self.preparer = runner, preparer
        self.inspector = inspector or _inspect_run
        self.lock = threading.RLock()
        self.job = None
        self.prepared = None
        self.pool = (ThreadPoolExecutor(max_workers=1) if runner is not None or preparer is not None
                     else ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')))
        try:
            super().__init__(address, TrainingStudioHandler)
        except Exception:
            self.pool.shutdown(wait=False, cancel_futures=True)
            raise

    def server_close(self):
        super().server_close()
        self.pool.shutdown(wait=True, cancel_futures=True)

    def _fail(self, message):
        self.prepared = None
        if self.job is not None:
            self.job.update(status='failed', error=str(message)[:2000], summary=None,
                            finished=time.monotonic(), finalized=True)

    def clear_completed(self):
        with self.lock:
            self.refresh()
            if self.job is None or self.job['status'] != 'running':
                self.job, self.prepared = None, None

    def refresh(self):
        with self.lock:
            job = self.job
            if job is None or job['finalized'] or not job['future'].done():
                return
            try:
                summary = job['future'].result()
                if type(summary) is not dict:
                    raise ValueError('The pipeline did not return a result object.')
                json.dumps(summary, allow_nan=False)
                if job['kind'] == 'prepare':
                    dataset = summary.get('dataset_path')
                    if not isinstance(dataset, str):
                        raise ValueError('The check did not produce a local dataset path.')
                    dataset = Path(dataset).resolve()
                    if not dataset.is_file() or not dataset.is_relative_to(job['output']):
                        raise ValueError('The checked dataset must stay inside its local job directory.')
                    self.prepared = {'source': job['original_source'], 'seed': job['seed'],
                                     'offline': job['offline'], 'dataset_path': str(dataset)}
                else:
                    summary = self.inspector(job['output'])
                    if type(summary) is not dict or summary.get('status') != 'complete':
                        raise ValueError('The pipeline did not verify a completed training result.')
                    json.dumps(summary, allow_nan=False)
                    self.archive_path()
                job.update(status='prepared' if job['kind'] == 'prepare' else 'complete',
                           summary=summary, finalized=True, finished=time.monotonic())
            except Exception as exc:
                self._fail(exc)

    def archive_path(self):
        if self.job is None:
            raise ValueError('No completed training result is available.')
        path = self.job['output'] / 'training-result.zip'
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(self.job['output']):
            raise ValueError('The completed training archive is unavailable.')
        return path

    def result(self):
        with self.lock:
            self.refresh()
            if self.job is None or self.job['status'] != 'complete':
                raise ValueError('Complete and verify a training job before downloading its result.')
            try:
                summary = self.inspector(self.job['output'])
                if type(summary) is not dict or summary.get('status') != 'complete':
                    raise ValueError('The training result is no longer verified.')
                json.dumps(summary, allow_nan=False)
                self.archive_path()
                self.job['summary'] = summary
                return summary
            except Exception as exc:
                self._fail(exc)
                raise ValueError('Training result verification failed: ' + str(exc)) from exc

    def state(self):
        with self.lock:
            self.refresh()
            result = {'status': 'idle', 'allow_online': not self.offline,
                      'scope': 'visual learning; movement readiness is separate',
                      'robot_skill_acquired': False, 'physical_robot_ready': False,
                      'job': None, 'summary': None, 'error': None}
            if self.job is not None:
                job = self.job
                elapsed = (job.get('finished') or time.monotonic()) - job['started']
                progress = {'stage': 'intake' if job['kind'] == 'prepare' else 'training',
                            'message': 'Checking footage and groups.' if job['kind'] == 'prepare'
                                       else 'Learning locally from the checked dataset.'}
                receipt = job['output'] / 'studio-progress.json'
                if job['status'] == 'running' and receipt.is_file() and receipt.stat().st_size <= 4096:
                    try:
                        event = _json(receipt.read_bytes())
                        if (type(event) is dict and event.get('stage') in ('intake', 'training', 'export', 'complete')
                                and isinstance(event.get('message'), str)):
                            progress = {'stage': event['stage'], 'message': event['message'][:600]}
                    except (OSError, ValueError):
                        pass
                result.update(status=job['status'], summary=job.get('summary'), error=job.get('error'),
                              job={'id': job['id'], 'kind': job['kind'], 'elapsed_seconds': max(0, elapsed),
                                   'offline': job['offline'], 'seed': job['seed'], 'progress': progress})
            return result

    def submit(self, kind, source, online, seed):
        with self.lock:
            self.refresh()
            if self.job is not None and self.job['status'] == 'running':
                raise RuntimeError('One CPU job is already running. Wait for its outcome.')
            offline = not online
            actual_source = source
            if kind == 'train' and self.prepared is not None:
                binding = self.prepared
                if (binding['source'], binding['seed'], binding['offline']) == (source, seed, offline):
                    actual_source = binding['dataset_path']
            if kind == 'prepare':
                self.prepared = None
            identity = uuid.uuid4().hex
            output = self.directory / 'jobs' / identity
            output.mkdir(parents=True, exist_ok=False)
            callback = self.preparer if kind == 'prepare' else self.runner
            if callback is None:
                future = self.pool.submit(_pipeline_worker, kind, actual_source, str(output), offline, seed)
            else:
                future = self.pool.submit(callback, actual_source, output, offline=offline, seed=seed)
            self.job = {'id': identity, 'kind': kind, 'output': output, 'future': future,
                        'original_source': source, 'offline': offline, 'seed': seed,
                        'status': 'running', 'started': time.monotonic(), 'finished': None,
                        'summary': None, 'error': None, 'finalized': False}
            return {'job_id': identity, 'status': 'running', 'kind': kind}


class TrainingStudioHandler(BaseHTTPRequestHandler):
    server_version = 'DVIDIA-Footage-Studio/0.1'
    timeout = 15

    def log_message(self, format, *args):
        # Source URLs, clip names, request bodies and local paths are not logged.
        pass

    def local_request(self):
        port = self.server.server_address[1]
        hosts = self.headers.get_all('Host', [])
        origins = self.headers.get_all('Origin', [])
        if len(hosts) != 1 or hosts[0] not in (f'127.0.0.1:{port}', f'localhost:{port}'):
            raise ValueError('Use the local training studio address printed by its server.')
        if len(origins) > 1 or (origins and origins[0] not in
                              (f'http://127.0.0.1:{port}', f'http://localhost:{port}')):
            raise ValueError('Use the local training studio UI for this request.')
        if self.headers.get('Sec-Fetch-Site') == 'cross-site':
            raise ValueError('Cross-site requests are not allowed.')

    def headers_for(self, length, mime, status=200, *, download=False):
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(length))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        if download:
            self.send_header('Content-Disposition', 'attachment; filename="training-result.zip"')
        self.end_headers()

    def respond(self, value, status=200, *, mime='application/json; charset=utf-8'):
        body = value if isinstance(value, bytes) else json.dumps(value, allow_nan=False).encode('utf8')
        self.headers_for(len(body), mime, status)
        self.wfile.write(body)

    def body(self, mime, limit):
        if self.headers.get('Transfer-Encoding') is not None:
            raise ValueError('Use a bounded Content-Length request.')
        if self.headers.get_all('Content-Type', []) != [mime]:
            raise ValueError('Send a ' + mime + ' request.')
        lengths = self.headers.get_all('Content-Length', [])
        if len(lengths) != 1 or not re.fullmatch(r'[1-9][0-9]*', lengths[0]):
            raise ValueError('Send one positive Content-Length.')
        length = int(lengths[0])
        if length > limit:
            raise ValueError(f'Request exceeds the {limit // 1024} KiB limit; use a local folder for large footage.')
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise ValueError('The request body is incomplete.')
        return raw

    def train_request(self):
        body = _json(self.body('application/json', JSON_LIMIT))
        if type(body) is not dict or set(body) - {'source', 'online', 'seed'} or 'source' not in body:
            raise ValueError('Supply source, optional online boolean, and optional seed.')
        source, online, seed = body['source'], body.get('online', False), body.get('seed', 17)
        if not isinstance(source, str) or not source.strip() or len(source.encode('utf8')) > 8192:
            raise ValueError('Supply a local Skillspace folder, ZIP or dataset path, or an allowed HTTPS link.')
        source = source.strip()
        if any(ord(char) < 32 for char in source):
            raise ValueError('The source contains invalid control characters.')
        if type(online) is not bool or type(seed) is not int or not 0 <= seed <= 999999:
            raise ValueError('Online must be a boolean and seed an integer between 0 and 999999.')
        if online and self.server.offline:
            raise ValueError('This server is offline. Restart with --allow-online to enable explicit HTTPS intake.')
        if '://' in source:
            url = urlsplit(source)
            if (url.scheme != 'https' or not url.hostname or url.username or url.password
                    or url.fragment or any(char.isspace() for char in source)):
                raise ValueError('Use an HTTPS source without credentials, fragments or whitespace.')
            if not online:
                raise ValueError('HTTPS intake requires explicit online permission; use a local export offline.')
        return source, online, seed

    def do_GET(self):
        local = False
        try:
            self.local_request()
            local = True
            if self.path == '/':
                self.respond(Path(__file__).with_name('training_studio.html').read_bytes(), mime='text/html; charset=utf-8')
            elif self.path == '/coverage':
                self.respond(Path(__file__).with_name('coverage_studio.html').read_bytes(), mime='text/html; charset=utf-8')
            elif self.path == '/api/coverage/example':
                from .coverage import example_plan
                self.respond(example_plan())
            elif self.path == '/api/state':
                self.respond(self.server.state())
            elif self.path == '/api/result':
                self.respond(self.server.result())
            elif self.path == '/download':
                with self.server.lock:
                    self.server.result()
                    path = self.server.archive_path()
                    with path.open('rb') as stream:
                        self.headers_for(path.stat().st_size, 'application/zip', download=True)
                        while block := stream.read(64 * 1024):
                            self.wfile.write(block)
            else:
                self.respond({'error': 'Not found.'}, 404)
        except ValueError as exc:
            self.respond({'error': str(exc)}, 409 if local and self.path in ('/api/result', '/download') else 400)
        except (OSError, TimeoutError) as exc:
            self.respond({'error': str(exc)}, 400)

    def do_POST(self):
        local = False
        try:
            self.local_request()
            local = True
            if self.path in ('/api/prepare', '/api/train'):
                source, online, seed = self.train_request()
                kind = 'prepare' if self.path == '/api/prepare' else 'train'
                self.respond(self.server.submit(kind, source, online, seed), 202)
            elif self.path == '/api/coverage/inspect':
                from .coverage import MAX_PLAN_BYTES, summarize_plan, validate_plan
                plan = validate_plan(self.body('application/json', MAX_PLAN_BYTES))
                self.respond({**summarize_plan(plan), 'plan': plan})
            elif self.path == '/api/upload':
                raw = self.body('application/zip', UPLOAD_LIMIT)
                validate_zip(raw)
                with self.server.lock:
                    self.server.refresh()
                    if self.server.job is not None and self.server.job['status'] == 'running':
                        raise RuntimeError('One CPU job is already running. Wait before replacing its source.')
                    self.server.clear_completed()
                    root = self.server.directory / 'uploads'
                    root.mkdir(exist_ok=True)
                    path = root / (uuid.uuid4().hex + '.zip')
                    path.write_bytes(raw)
                self.respond({'source': str(path), 'bytes': len(raw), 'status': 'uploaded',
                              'checked': False, 'robot_skill_acquired': False}, 201)
            else:
                self.respond({'error': 'Not found.'}, 404)
        except RuntimeError as exc:
            self.respond({'error': str(exc)}, 409)
        except Exception as exc:
            if local and self.path in ('/api/prepare', '/api/train', '/api/upload'):
                self.server.clear_completed()
            self.respond({'error': str(exc)}, 400)


def configure(directory=Path('runs/training-studio'), *, port=8270, offline=True,
              runner=None, preparer=None, inspector=None):
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError('Use a valid local port.')
    return TrainingStudioServer(('127.0.0.1', port), directory, offline,
                                runner=runner, preparer=preparer, inspector=inspector)


def run_server(directory=Path('runs/training-studio'), *, port=8270, offline=True):
    with configure(directory, port=port, offline=offline) as server:
        print(f'DVIDIA footage training: http://127.0.0.1:{server.server_address[1]}/', flush=True)
        print('One CPU worker. Local data stays local; visual learning is not an acquired robot skill.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser(description='Local, offline-first Skillspace footage training studio.')
    parser.add_argument('--directory', type=Path, default=Path('runs/training-studio'))
    parser.add_argument('--port', type=int, default=8270)
    parser.add_argument('--allow-online', action='store_true', help='Enable the explicit HTTPS-intake checkbox. Training still runs locally.')
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('Use a port between 1024 and 65535.')
    run_server(args.directory, port=args.port, offline=not args.allow_online)


if __name__ == '__main__':
    main()

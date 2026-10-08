"""Loopback authoring UI and a single isolated CPU physics worker."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import multiprocessing
from pathlib import Path
import re
import threading
from urllib.parse import urlsplit
import uuid

from .arm_runner import authored_grounding, run_skill, validate_scene, scene_config, write_run
from .arm_env import ArmConfig, ArmEnv
from .cli import NetworkGuard
from .installer import fetch_skill_url, install_source, read_installation, runtime_contract
from .capsule_installer import (capsule_worker, install_capsule, read_capsule_installation,
                               capsule_runtime)


def request_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate request JSON keys are not allowed.')
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('Nonfinite request values are not allowed.')
    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
    json.dumps(result, allow_nan=False)  # Reject overflowing numeric tokens too.
    return result


def preview_worker(scene):
    scene = validate_scene(scene)
    with NetworkGuard(True):
        environment = ArmEnv(scene_config(scene))
        observation, info = environment.reset(seed=scene['seed'])
    return {'observation': observation, 'info': info}


def run_worker(source, grounding, scene, policy, identity, source_url, output):
    result = run_skill(source, grounding, scene, policy=policy, offline=True)
    result.update(installation_id=identity, source_url=source_url)
    write_run(output, result, source=source, grounding=grounding)
    return result


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, directory, offline=False):
        self.directory, self.offline = Path(directory), offline
        self.directory.mkdir(parents=True, exist_ok=True)
        self.pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'))
        self.jobs, self.mutation_lock = {}, threading.Lock()
        super().__init__(address, StudioHandler)

    def server_close(self):
        super().server_close()
        self.pool.shutdown(wait=True, cancel_futures=True)


class StudioHandler(BaseHTTPRequestHandler):
    server_version = 'DVIDIA-Skillspace-Lab/0.2'

    def log_message(self, format, *args):
        # URL source, bodies and server paths are never logged.
        print(f'local request {self.command} {self.path.split("?")[0]}', flush=True)

    def respond(self, value, status=200, *, mime='application/json'):
        body = value if isinstance(value, bytes) else json.dumps(value, allow_nan=False).encode('utf8')
        self.send_response(status)
        self.send_header('Content-Type', mime+'; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' blob:; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(body)

    def local_request(self):
        port = self.server.server_address[1]
        if self.headers.get('Host') not in (f'127.0.0.1:{port}', f'localhost:{port}'):
            raise ValueError('Use the local lab address printed by its server.')
        origin = self.headers.get('Origin')
        if origin and origin not in (f'http://127.0.0.1:{port}', f'http://localhost:{port}'):
            raise ValueError('Use the local lab UI to change an environment or install a skill.')

    def do_GET(self):
        try:
            self.local_request()
            if self.path == '/':
                self.respond(Path(__file__).with_name('studio.html').read_bytes(), mime='text/html')
            elif self.path == '/api/installations':
                records = []
                root = self.server.directory/'installed'
                if root.exists():
                    for path in sorted(root.iterdir(), key=lambda p:p.stat().st_mtime, reverse=True)[:16]:
                        if re.fullmatch(r'[a-f0-9]{24}', path.name):
                            try:
                                records.append(read_installation(root, path.name))
                            except (ValueError, OSError, KeyError):
                                continue
                self.respond({'installations': records})
            elif self.path == '/api/capsules/example':
                from .skill_capsule import inspect_capsule
                data = Path(__file__).with_name('placement_candidate.skill.json').read_bytes()
                inspect_capsule(data, expected_runtime=capsule_runtime())
                self.respond(data)
            elif self.path == '/api/benchmark':
                summary = Path(__file__).with_name('arm_benchmark_summary.json')
                result = json.loads(summary.read_text()) if summary.exists() else {'status': 'pending'}
                if result.get('status') == 'complete':
                    current = runtime_contract()['files_sha256']
                    measured = result.get('source_hashes', {}).get('final', {})
                    if not measured or any(current.get(name) != digest for name, digest in measured.items()):
                        result = {'status': 'stale', 'description': 'The controller changed after this benchmark.'}
                self.respond(result)
            elif re.fullmatch(r'/api/jobs/[a-f0-9]{32}', self.path):
                job = self.server.jobs.get(self.path.rsplit('/', 1)[1])
                if job is None:
                    self.respond({'error': 'Unknown local execution.'}, 404)
                elif not job.done():
                    self.respond({'status': 'running'})
                else:
                    try:
                        result = job.result()
                    except Exception as exc:
                        self.respond({'status': 'failed', 'error': str(exc)})
                    else:
                        self.respond({'status': 'complete', 'run': result})
            else:
                self.respond({'error': 'Not found.'}, 404)
        except ValueError as exc:
            self.respond({'error': str(exc)}, 400)

    def do_POST(self):
        try:
            self.local_request()
            if self.headers.get('Content-Type') != 'application/json':
                raise ValueError('Send an application/json request.')
            length = int(self.headers.get('Content-Length', '0'))
            limit = 1024*1024 if self.path == '/api/capsules/install' else 256*1024
            if not 0 < length <= limit:
                raise ValueError(f'Request must contain at most {limit//1024} KiB of JSON.')
            body = request_json(self.rfile.read(length))
            if type(body) is not dict:
                raise ValueError('Request must be a JSON object.')
            if self.path == '/api/capsules/install':
                if set(body) != {'document'} or not isinstance(body['document'], str):
                    raise ValueError('Drop one data-only skill capsule JSON.')
                with self.server.mutation_lock:
                    record = install_capsule(body['document'], self.server.directory/'capsules')
                self.respond(record)
            elif self.path in ('/api/capsules/qualify', '/api/capsules/run'):
                if set(body) != {'installation_id', 'scene'}:
                    raise ValueError('Capsule execution requires its installation and scene.')
                scene = validate_scene(body['scene'])
                qualify = self.path.endswith('/qualify')
                with self.server.mutation_lock:
                    if any(not future.done() for future in self.server.jobs.values()):
                        raise ValueError('One CPU execution is already running. Wait for its outcome.')
                    read_capsule_installation(self.server.directory/'capsules', body['installation_id'],
                                              scene, require_qualified=not qualify)
                    identity = uuid.uuid4().hex
                    job = self.server.pool.submit(capsule_worker, self.server.directory/'capsules',
                               body['installation_id'], scene, self.server.directory/'executions'/identity,
                               qualify=qualify)
                    for old in list(self.server.jobs)[:-15]:
                        if self.server.jobs[old].done():
                            del self.server.jobs[old]
                    self.server.jobs[identity] = job
                self.respond({'job_id': identity, 'status': 'running'}, 202)
            elif self.path == '/api/install':
                if set(body)-{'url','bundled','document','scene'}:
                    raise ValueError('Unknown installation field.')
                modes = [key for key in ('url','bundled','document') if key in body]
                if len(modes)!=1:
                    raise ValueError('Supply a DVIDIA link, bundled source, or JSON document.')
                if modes == ['url']:
                    if self.server.offline:
                        raise ValueError('This lab is offline. Install bundled source or drop an exported JSON.')
                    data, url = fetch_skill_url(body['url'])
                elif modes == ['bundled'] and body['bundled'] is True:
                    data, url = Path(__file__).with_name('place_cup.skill.json').read_bytes(), None
                elif modes == ['document'] and isinstance(body['document'], str):
                    data, url = body['document'].encode('utf8'), None
                else:
                    raise ValueError('Invalid installation source.')
                grounding = authored_grounding(data, body.get('scene'))
                with self.server.mutation_lock:
                    record = install_source(data, grounding, self.server.directory/'installed', source_url=url)
                self.respond(record)
            elif self.path == '/api/preview':
                if set(body) != {'scene'}:
                    raise ValueError('Preview requires the environment scene.')
                scene = validate_scene(body['scene'])
                self.respond(self.server.pool.submit(preview_worker, scene).result(timeout=30))
            elif self.path == '/api/run':
                if set(body) != {'installation_id','scene','policy'}:
                    raise ValueError('Execution requires an installed skill, scene and controller.')
                scene = validate_scene(body['scene'])
                if body['policy'] not in ('placement','open_jaw','idle'):
                    raise ValueError('Unknown installed controller.')
                with self.server.mutation_lock:
                    if any(not future.done() for future in self.server.jobs.values()):
                        raise ValueError('One CPU execution is already running. Wait for its outcome.')
                    record = read_installation(self.server.directory/'installed', body['installation_id'])
                    source = (self.server.directory/'installed'/body['installation_id']/'source.json').read_bytes()
                    grounding = None if record['grounding_mode']=='inline' else record['grounding']
                    identity = uuid.uuid4().hex
                    job = self.server.pool.submit(run_worker, source, grounding, scene, body['policy'],
                                                  body['installation_id'], record['source_url'],
                                                  self.server.directory/'executions'/identity)
                    # Keep bounded memory; completed runs remain saved on disk.
                    for old in list(self.server.jobs)[:-15]:
                        if self.server.jobs[old].done():
                            del self.server.jobs[old]
                    self.server.jobs[identity] = job
                self.respond({'job_id': identity, 'status': 'running'}, 202)
            else:
                self.respond({'error': 'Not found.'}, 404)
        except Exception as exc:
            self.respond({'error': str(exc)}, 400)


def main():
    parser = argparse.ArgumentParser(description='Local DVIDIA Skillspace installation and native CPU arm lab.')
    parser.add_argument('--port', type=int, default=8240)
    parser.add_argument('--directory', type=Path, default=Path('runs/studio'))
    parser.add_argument('--offline', action='store_true', help='Disable URL downloads; bundled and dropped source still install.')
    args = parser.parse_args()
    if not 1024<=args.port<=65535:
        parser.error('Use a port between 1024 and 65535.')
    with StudioServer(('127.0.0.1', args.port), args.directory, args.offline) as server:
        print(f'DVIDIA Skillspace lab: http://127.0.0.1:{args.port}/', flush=True)
        print('One native CPU physics worker. Installed execution denies Python network calls.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()

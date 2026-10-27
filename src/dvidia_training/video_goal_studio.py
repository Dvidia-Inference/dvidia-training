"""Local video-goal studio: the CLI's data-only PiPER workflow, one CPU worker.

The HTTP interface stays on loopback. It neither accepts account credentials nor
provides a cross-origin bridge from a hosted website. Downloads are candidates,
not learned motor policies or physical-robot qualification.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import mimetypes
import multiprocessing
from pathlib import Path
import re
import shlex
import shutil
import threading
import time
import tempfile
from urllib.parse import parse_qs, urlsplit
import uuid
import zipfile

from .footage_data import _json, VIDEO_EXTENSIONS
from .video_goal_bridge import MAX_VIDEO_BYTES, PlanarVideoCalibration
from .video_goal_install import FILES, MAX_MANIFEST_BYTES, MAX_PROPOSAL_BYTES, MAX_CALIBRATION_BYTES
from .skillspace_video import SOURCE_PACKAGE_LIMIT, import_skillspace_video, inspect_skillspace_video

JSON_LIMIT = 32 * 1024
BUNDLE_LIMIT = 3 * 1024 * 1024
_ID = re.compile(r'^[a-f0-9]{24,32}$')


def provision_assets(directory):
    """Fetch a complete pinned profile before exposing it to the local runtime."""
    from .robot_profiles import verify_assets, fetch_assets
    root = Path(directory).expanduser()
    if root.is_symlink(): raise ValueError('Choose a plain asset directory.')
    try: return verify_assets('piper', root)
    except (ValueError, OSError): pass
    target = root/'agilex_piper'
    if target.exists() or target.is_symlink():
        raise ValueError('Existing PiPER assets failed verification. Choose a fresh --assets directory; existing files are preserved.')
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.piper-download-', dir=root) as temporary:
        fetch_assets('piper', temporary)
        (Path(temporary)/'agilex_piper').rename(target)
    return verify_assets('piper', root)


def _worker(kind, source, calibration, output, assets, object_xy, seed):
    # Spawned workers import only fixed installed code. No uploaded code runs.
    from .video_goal_install import create_candidate, execute_candidate
    if kind == 'create':
        result = create_candidate(source, calibration, output)
        return {'manifest': result}
    receipt = execute_candidate(source, assets, object_xy, output, seed=seed)
    from .video_goal_benchmark import score_reference
    from .video_goal_install_benchmark import _read_episode
    episode = _read_episode(Path(output)/'episode.json.gz')
    score = score_reference(episode, receipt['goal_xy_m'])
    return {'goal_score': score, 'wall_seconds': receipt['run_wall_seconds'],
            'simulated_seconds': episode['simulated_seconds'], 'goal_xy_m': receipt['goal_xy_m']}


def _calibration(value):
    if type(value) is not dict or set(value) != set(asdict(PlanarVideoCalibration())):
        raise ValueError('Choose a camera calibration JSON with every documented calibration field.')
    try: return asdict(PlanarVideoCalibration(**value))
    except TypeError as exc: raise ValueError('Invalid camera calibration fields.') from exc


def import_bundle(raw, output):
    """Unpack exactly three bounded JSON files, then validate with the CLI loader."""
    from .video_goal_install import load_candidate
    output = Path(output)
    if output.exists() or output.is_symlink(): raise ValueError('Use a new candidate directory.')
    if not raw or len(raw) > BUNDLE_LIMIT: raise ValueError('Candidate ZIP exceeds 3 MiB.')
    bounds = dict(zip(FILES, (MAX_MANIFEST_BYTES, MAX_PROPOSAL_BYTES, MAX_CALIBRATION_BYTES)))
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            infos = archive.infolist()
            if len(infos) != len(FILES) or {i.filename for i in infos} != set(FILES):
                raise ValueError('Candidate ZIP must contain exactly manifest.json, proposal.json and calibration.json.')
            payloads = {}
            for info in infos:
                mode = info.external_attr >> 16
                if (info.flag_bits & 1 or info.is_dir() or (mode & 0o170000) not in (0, 0o100000)
                        or not 0 < info.file_size <= bounds[info.filename]):
                    raise ValueError('Candidate files must be bounded, unencrypted regular JSON files.')
                with archive.open(info) as stream: data = stream.read(bounds[info.filename]+1)
                if len(data) != info.file_size: raise ValueError('Invalid expanded candidate size.')
                payloads[info.filename] = data
        output.mkdir(parents=True, exist_ok=False)
        for name, data in payloads.items(): (output/name).write_bytes(data)
        return load_candidate(output)['manifest']
    except (zipfile.BadZipFile, RuntimeError, OSError, ValueError) as exc:
        if output.is_dir(): shutil.rmtree(output)
        if isinstance(exc, ValueError): raise
        raise ValueError('Choose a valid data-only candidate ZIP.') from exc


class VideoGoalStudioServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, directory, assets=None, *, runner=None):
        if address[0] != '127.0.0.1': raise ValueError('The studio binds only to 127.0.0.1.')
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.assets = Path(assets).expanduser().resolve() if assets else self.directory/'robot-assets'
        self.lock = threading.RLock()
        self.source = self.candidate = self.job = None
        self.sources, self.candidates, self.receipts = {}, {}, {}
        self.runner = runner
        self.pool = (ThreadPoolExecutor(max_workers=1) if runner is not None else
                     ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')))
        self.asset_state = {'ready': False,
            'setup_command': 'dvidia-robot-sim fetch piper --output '+shlex.quote(str(self.assets))}
        self.refresh_assets()
        try: super().__init__(address, VideoGoalStudioHandler)
        except Exception:
            self.pool.shutdown(wait=False, cancel_futures=True)
            raise

    def refresh_assets(self):
        from .robot_profiles import verify_assets
        self.assets_checked_at = time.monotonic()
        try:
            verify_assets('piper', self.assets)
            self.asset_state.update(ready=True, message='Pinned PiPER assets verified.')
        except (ValueError, OSError):
            self.asset_state.update(ready=False, message='PiPER assets are needed to run the simulation.')

    def server_close(self):
        super().server_close()
        self.pool.shutdown(wait=True, cancel_futures=True)

    def refresh(self):
        if not self.job or self.job['status'] != 'running' or not self.job['future'].done(): return
        job = self.job
        try:
            result = job['future'].result()
            if job['kind'] == 'create':
                from .video_goal_install import load_candidate
                manifest = load_candidate(job['output'])['manifest']
                self.validate_source()
                if self.source['skillspace'] and manifest['video']['sha256'] != self.source['skillspace']['video_sha256']:
                    raise ValueError('The candidate does not match the imported Skillspace video. Import it again.')
                candidate = {'id': manifest['candidate_id'], 'directory': job['output'], 'manifest': manifest}
                self.candidates[candidate['id']] = self.candidate = candidate
                # Restore only the calibration explicitly supplied for this
                # source, never the example camera on a different upload.
                if self.source is not None:
                    self.source['calibration'] = job['calibration']
                result = {'goal_xy_m': manifest['goal_xy_m'], 'candidate_id': manifest['candidate_id']}
            else:
                receipt = job['output']/'receipt.json'
                # API responses only expose this registered file, never caller paths.
                if receipt.is_symlink() or not receipt.is_file(): raise ValueError('Missing execution receipt.')
                self.receipts[job['id']] = receipt
                result['receipt_url'] = '/api/receipt/'+job['id']+'.json'
            job.update(status='complete', result=result, finished=time.monotonic())
        except Exception as exc:
            message = str(exc)
            if 'Only a proposed video destination' in message:
                message = 'No stable visible green-block destination was found. Check the clip and camera calibration.'
            job.update(status='failed', error=message[:1000], result=None, finished=time.monotonic())
            if job['kind'] == 'run': self.refresh_assets()

    def state(self):
        with self.lock:
            self.refresh()
            if not self.asset_state['ready'] and time.monotonic()-self.assets_checked_at >= 3.:
                self.refresh_assets()
            source = None if self.source is None else {key: self.source[key] for key in
                ('id', 'name', 'size_bytes', 'media_url', 'sample', 'calibration', 'skillspace')}
            candidate = None if self.candidate is None else {
                'id': self.candidate['id'], 'goal_xy_m': self.candidate['manifest']['goal_xy_m'],
                'video_sha256': self.candidate['manifest']['video']['sha256'],
                'download_url': '/api/download/'+self.candidate['id']+'.zip',
                'scope': self.candidate['manifest']['scope'], 'policy_learned': False}
            job = None if self.job is None else {key: self.job[key] for key in
                ('id', 'kind', 'status', 'error', 'result')}
            if job is not None: job['elapsed_s'] = (self.job.get('finished') or time.monotonic())-self.job['started']
            return {'profile': 'piper', 'limits': {'video_bytes': MAX_VIDEO_BYTES,
                    'source_package_bytes': SOURCE_PACKAGE_LIMIT},
                'assets': dict(self.asset_state), 'example': {'available': self.example_available()},
                'source': source, 'candidate': candidate, 'job': job}

    def idle(self):
        self.refresh()
        if self.job and self.job['status'] == 'running': raise ValueError('Wait for the current job to finish.')

    def example_available(self):
        root = Path(__file__).parent
        return (root/'video_goal_example.mp4').is_file() and (root/'video_goal_example.json').is_file()

    def register_source(self, path, name, *, sample=False, calibration=None, skillspace=None):
        identity = path.parent.name
        self.source = {'id': identity, 'path': path, 'name': name,
            'size_bytes': path.stat().st_size, 'media_url': '/media/'+identity,
            'sample': sample, 'calibration': calibration, 'skillspace': skillspace,
            'file_signature': self.file_signature(path)}
        self.sources[identity] = self.source
        self.candidate = self.job = None

    @staticmethod
    def file_signature(path):
        info = path.stat()
        if path.is_symlink() or not path.is_file(): raise ValueError('Choose a regular local video file.')
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    def validate_source(self, *, verify_hash=True):
        source = self.source
        if source is None: raise ValueError('Choose the source video again.')
        if self.file_signature(source['path']) != source['file_signature']:
            raise ValueError('Local video changed. Import or upload it again before creating a candidate.')
        if verify_hash and source['skillspace']:
            manifest = inspect_skillspace_video(source['path'].parent)
            if (manifest['video']['sha256'] != source['skillspace']['video_sha256']
                    or manifest['contribution'] != {key: source['skillspace'][key] for key in
                        ('id', 'revision', 'skillspace_id', 'task_title')}):
                raise ValueError('Local Skillspace source changed. Import the package again.')

    def import_video(self, package):
        """Open an explicit local export without contacting the platform."""
        with self.lock:
            self.idle()
            output = self.directory/'uploads'/uuid.uuid4().hex
            manifest = import_skillspace_video(package, output)
            self.register_source(output/manifest['video']['filename'], manifest['video']['original_name'],
                skillspace={**manifest['contribution'], 'exported_at': manifest['exported_at'],
                            'video_sha256': manifest['video']['sha256']})

    def example(self):
        with self.lock:
            self.idle()
            if not self.example_available(): raise ValueError('The included example is unavailable.')
            metadata = _json(Path(__file__).with_name('video_goal_example.json').read_bytes())
            clip = Path(__file__).with_name('video_goal_example.mp4')
            if sha256(clip.read_bytes()).hexdigest() != metadata['sha256']:
                raise ValueError('The included example differs from its recorded source.')
            target = self.directory/'uploads'/uuid.uuid4().hex/'source.mp4'
            target.parent.mkdir(parents=True)
            shutil.copyfile(clip, target)
            self.register_source(target, 'PiPER block placement · synthetic example', sample=True,
                                 calibration=_calibration(metadata['calibration']))

    def submit(self, kind, payload):
        with self.lock:
            self.idle()
            if kind == 'create':
                if type(payload) is not dict or set(payload) != {'source_id', 'calibration'}:
                    raise ValueError('Supply a source and its camera calibration.')
                if self.source is None or payload['source_id'] != self.source['id']:
                    raise ValueError('Choose the current source video.')
                self.validate_source()
                source = self.source['path']; calibration = _calibration(payload['calibration'])
                object_xy = None; seed = 0
                self.candidate = None
            else:
                if type(payload) is not dict or set(payload) != {'candidate_id', 'object_xy', 'seed'}:
                    raise ValueError('Supply a candidate, object start XY and integer seed.')
                if self.candidate is None or payload['candidate_id'] != self.candidate['id']:
                    raise ValueError('Create or import the current candidate first.')
                if not self.asset_state['ready']:
                    self.refresh_assets()
                if not self.asset_state['ready']: raise ValueError(self.asset_state['message'])
                object_xy, seed = payload['object_xy'], payload['seed']
                if (type(object_xy) is not list or len(object_xy) != 2 or
                        type(seed) is not int or not 0 <= seed <= 999999):
                    raise ValueError('Use two finite start coordinates and an integer seed in [0,999999].')
                from .piper_task import PiperTaskConfig
                PiperTaskConfig(object_position_m=(*object_xy, .082),
                    target_position_m=(*self.candidate['manifest']['goal_xy_m'], .08))
                source = self.candidate['directory']; calibration = None
            identity = uuid.uuid4().hex
            output = self.directory/'jobs'/identity
            future = self.pool.submit(self.runner or _worker, kind, str(source), calibration,
                                      str(output), str(self.assets), object_xy, seed)
            self.job = {'id': identity, 'kind': kind, 'output': output, 'future': future,
                'status': 'running', 'started': time.monotonic(), 'error': None, 'result': None,
                'calibration': calibration}

    def archive(self, identity):
        from .video_goal_install import load_candidate
        if not _ID.fullmatch(identity) or identity not in self.candidates:
            raise ValueError('No such candidate in this local session.')
        root = self.candidates[identity]['directory']
        load_candidate(root)
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name in FILES: archive.writestr(name, (root/name).read_bytes())
        return stream.getvalue()


class VideoGoalStudioHandler(BaseHTTPRequestHandler):
    server_version = 'DVIDIA-Video-Skill-Studio/0.1'
    timeout = 30

    def log_message(self, *_): pass

    def local(self, *, write=False):
        expected = f'127.0.0.1:{self.server.server_port}'
        if self.headers.get_all('Host', []) != [expected]: raise PermissionError('Use the printed loopback address.')
        origins = self.headers.get_all('Origin', [])
        if (origins not in ([], ['http://'+expected]) or write and origins != ['http://'+expected]
                or self.headers.get('Sec-Fetch-Site') == 'cross-site'):
            raise PermissionError('Use this local studio for requests. Cross-origin requests are disabled.')

    def send(self, status, value, mime='application/json; charset=utf-8', *, filename=None):
        data = value if type(value) is bytes else json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; media-src 'self' blob:; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
        if filename: self.send_header('Content-Disposition', 'attachment; filename="'+filename+'"')
        self.end_headers()
        if self.command != 'HEAD': self.wfile.write(data)

    def length(self, mime, maximum):
        if self.headers.get('Transfer-Encoding') is not None: raise ValueError('A bounded Content-Length is required.')
        if self.headers.get_all('Content-Type', []) != [mime]: raise ValueError('Use Content-Type '+mime+'.')
        values = self.headers.get_all('Content-Length', [])
        if len(values) != 1 or not re.fullmatch(r'[1-9][0-9]{0,9}', values[0]):
            raise ValueError('Send exactly one positive Content-Length.')
        length = int(values[0])
        if length > maximum: raise ValueError('Upload exceeds the supported size limit.')
        return length

    def body(self, mime, maximum):
        length = self.length(mime, maximum)
        data = self.rfile.read(length)
        if len(data) != length: raise ValueError('The upload ended before its declared length.')
        return data

    def media(self, identity):
        if not _ID.fullmatch(identity) or identity not in self.server.sources:
            raise ValueError('No such source in this local session.')
        source = self.server.sources[identity]
        path = source['path']; size = path.stat().st_size
        if self.server.file_signature(path) != source['file_signature']:
            raise ValueError('Local video changed; upload it again.')
        start, end, status = 0, size-1, 200
        requested = self.headers.get('Range')
        if requested:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested)
            if not match or not any(match.groups()):
                return self.range_error(size)
            left, right = match.groups()
            if left: start, end = int(left), min(int(right), size-1) if right else size-1
            elif int(right): start, end = max(0, size-int(right)), size-1
            else: return self.range_error(size)
            if not 0 <= start <= end < size: return self.range_error(size)
            status = 206
        self.send_response(status)
        self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        self.send_header('Content-Length', str(end-start+1))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
        if status == 206: self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        if self.command == 'HEAD': return
        with path.open('rb') as stream:
            stream.seek(start); remaining = end-start+1
            while remaining:
                data = stream.read(min(64*1024, remaining))
                if not data: break
                self.wfile.write(data); remaining -= len(data)

    def range_error(self, size):
        self.send_response(416)
        self.send_header('Content-Range', f'bytes */{size}')
        self.send_header('Content-Length', '0')
        self.end_headers()

    def do_HEAD(self): self.do_GET()

    def do_GET(self):
        try:
            self.local()
            path = urlsplit(self.path).path
            if path == '/': return self.send(200, Path(__file__).with_name('video_goal_studio.html').read_bytes(), 'text/html; charset=utf-8')
            scripts = {'/studio-mapping.js': 'studio_mapping.js', '/studio-ui.js': 'studio_ui.js'}
            if path in scripts:
                return self.send(200, Path(__file__).with_name(scripts[path]).read_bytes(), 'text/javascript; charset=utf-8')
            if path == '/api/state': return self.send(200, self.server.state())
            if path.startswith('/media/'): return self.media(path[len('/media/'):])
            match = re.fullmatch(r'/api/download/([a-f0-9]{24})\.zip', path)
            if match:
                return self.send(200, self.server.archive(match[1]), 'application/zip', filename='dvidia-video-candidate.zip')
            match = re.fullmatch(r'/api/receipt/([a-f0-9]{32})\.json', path)
            if match and match[1] in self.server.receipts:
                receipt = self.server.receipts[match[1]]
                if receipt.is_symlink() or receipt.stat().st_size > 2*1024*1024: raise ValueError('Invalid receipt.')
                return self.send(200, receipt.read_bytes())
            return self.send(404, {'error': 'Not found.'})
        except PermissionError as exc: self.send(403, {'error': str(exc)})
        except (ValueError, OSError) as exc: self.send(400, {'error': str(exc)[:1000]})

    def do_POST(self):
        try:
            self.local(write=True)
            path = urlsplit(self.path).path
            if path == '/api/upload':
                name = parse_qs(urlsplit(self.path).query).get('name', [''])[0]
                if (not name or len(name) > 200 or '/' in name or '\\' in name
                        or any(ord(c) < 32 for c in name) or Path(name).suffix.lower() not in VIDEO_EXTENSIONS):
                    raise ValueError('Choose a supported local video file (MP4, MOV, WebM, AVI or MKV).')
                length = self.length('application/octet-stream', MAX_VIDEO_BYTES)
                with self.server.lock:
                    self.server.idle()
                    output = self.server.directory/'uploads'/uuid.uuid4().hex/('source'+Path(name).suffix.lower())
                    output.parent.mkdir(parents=True)
                    try:
                        with output.open('xb') as stream:
                            remaining = length
                            while remaining:
                                data = self.rfile.read(min(64*1024, remaining))
                                if not data: raise ValueError('Incomplete video upload.')
                                stream.write(data); remaining -= len(data)
                        self.server.register_source(output, name)
                    except Exception:
                        shutil.rmtree(output.parent)
                        raise
            elif path == '/api/import':
                raw = self.body('application/zip', BUNDLE_LIMIT)
                with self.server.lock:
                    self.server.idle()
                    output = self.server.directory/'imports'/uuid.uuid4().hex
                    manifest = import_bundle(raw, output)
                    candidate = {'id': manifest['candidate_id'], 'directory': output, 'manifest': manifest}
                    self.server.candidates[candidate['id']] = self.server.candidate = candidate
                    self.server.source = self.server.job = None
            elif path == '/api/import-video':
                length = self.length('application/zip', SOURCE_PACKAGE_LIMIT)
                # Stream to a private temporary file. The offline importer then
                # checks names, manifest, size and source digest before adoption.
                with tempfile.NamedTemporaryFile(dir=self.server.directory, suffix='.zip') as package:
                    remaining = length
                    while remaining:
                        data = self.rfile.read(min(64*1024, remaining))
                        if not data: raise ValueError('Incomplete video package upload.')
                        package.write(data); remaining -= len(data)
                    package.flush()
                    self.server.import_video(Path(package.name))
            elif path == '/api/example':
                if _json(self.body('application/json', JSON_LIMIT)) != {}: raise ValueError('Example takes no arguments.')
                self.server.example()
            elif path in ('/api/create', '/api/run'):
                self.server.submit(path.rsplit('/', 1)[1], _json(self.body('application/json', JSON_LIMIT)))
            else: return self.send(404, {'error': 'Not found.'})
            return self.send(200, self.server.state())
        except PermissionError as exc: self.send(403, {'error': str(exc)})
        except (ValueError, TypeError, OSError) as exc: self.send(400, {'error': str(exc)[:1000]})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('dvidia-skills'))
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--port', type=int, default=8290)
    parser.add_argument('--fetch-assets', action='store_true', help='explicitly download pinned public PiPER assets before starting')
    parser.add_argument('--import', dest='source_package', type=Path,
                        help='open a downloaded DVIDIA video package in the local studio')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535: parser.error('Port must be between 1 and 65535.')
    if args.fetch_assets:
        assets = args.assets or args.directory/'robot-assets'
        try: provision_assets(assets)
        except (ValueError, OSError) as exc: parser.exit(2, 'Asset setup: '+str(exc)+'\n')
    with VideoGoalStudioServer(('127.0.0.1', args.port), args.directory, args.assets) as server:
        if args.source_package:
            try: server.import_video(args.source_package)
            except (ValueError, OSError) as exc: parser.exit(2, 'Video package: '+str(exc)+'\n')
        print(f'DVIDIA Skill Studio: http://127.0.0.1:{server.server_port}', flush=True)
        print('Local research alpha · PiPER simulation · no motor policy trained', flush=True)
        try: server.serve_forever()
        except KeyboardInterrupt: pass


if __name__ == '__main__': main()

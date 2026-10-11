"""Read-only, loopback replay of hash-bound image-motion evidence."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from fractions import Fraction
import hashlib
from http.server import ThreadingHTTPServer
import os
from pathlib import Path
import re
import threading
from urllib.parse import urlsplit

from .footage_data import _json, _local_file, _read, canonical
from .motion import validate_motion_rows
from .observation_studio import ObservationHandler
from .observations import box, digest, identity

MAX_JSON_BYTES = 16 * 1024 * 1024


def _document(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Expected a local, regular evidence file.')
    return _json(_read(path, MAX_JSON_BYTES), maximum=MAX_JSON_BYTES)


def _file_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _bound_file(root, name, expected):
    path = _local_file(root, name)
    if type(expected) is not str or not re.fullmatch('[a-f0-9]{64}', expected) or _file_digest(path) != expected:
        raise ValueError('Evidence file differs from its recorded hash.')
    return path


def load_motion_run(path):
    """Validate declared files, causal rows and the copied source before replay.

    Hashes establish internal consistency, not source authenticity or independent
    motion accuracy. Trusted local bundles remain the supported input scope.
    """
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Use a regular motion receipt.')
    path = path.resolve(strict=True)
    run = _document(path)
    root = path.parent
    if type(run) is not dict or run.get('kind') != 'dvidia.motion-run.v1':
        raise ValueError('Expected a dvidia.motion-run.v1 receipt.')
    if run.get('status') not in ('completed', 'failed'):
        raise ValueError('A motion receipt requires a completed or failed status.')
    protocol = run.get('protocol')
    rows, frames = run.get('rows'), run.get('frames')
    if type(protocol) is not dict or type(rows) is not list or type(frames) is not list:
        raise ValueError('Missing motion protocol, rows or timestamp evidence.')
    if not 0 <= len(rows) <= 1800 or len(frames) != len(rows):
        raise ValueError('Motion frame counts exceed the replay envelope.')
    for filename, document, key in (('protocol.json', protocol, 'protocol_sha256'),
                                    ('rows.json', rows, 'rows_sha256'),
                                    ('frames.json', frames, 'frames_sha256')):
        if run.get(key) != digest(document) or _document(_local_file(root, filename)) != document:
            raise ValueError('Motion evidence bytes differ from the recorded receipt.')
    size = protocol.get('analysis_image_size_pixels')
    if size is None and not rows and run['status'] == 'failed':
        pass  # Intake may fail before the original image can be probed.
    else:
        if (type(size) is not list or len(size) != 2 or any(type(n) is not int or not 1 <= n <= 640 for n in size)):
            raise ValueError('Unsupported analysis-image dimensions.')
        validate_motion_rows(rows, *size)
    seed = run.get('seed')
    if (type(seed) is not dict or seed.get('kind') != 'human_authored_image_region'
            or digest(seed) != run.get('seed_sha256') or digest(seed) != protocol.get('seed_sha256')):
        raise ValueError('The human seed differs from the motion protocol.')
    box(seed.get('normalized_xyxy'))
    if type(seed.get('label')) is not str or not 1 <= len(seed['label']) <= 160:
        raise ValueError('The human seed requires a bounded text label.')
    if type(run.get('source')) is not dict:
        raise ValueError('A motion run requires a source receipt.')
    if run.get('episode_id') != protocol.get('episode_id'):
        raise ValueError('Episode identity differs from its motion protocol.')
    if (run.get('frame_count') != len(rows) or type(run.get('frame_count')) is not int
            or run.get('tracked_frame_count') != sum(row['status'] == 'tracked' for row in rows)):
        raise ValueError('Motion coverage counts differ from the retained evidence.')
    if any(row['estimator'] != protocol.get('estimator') for row in rows):
        raise ValueError('Motion estimator differs from the declared protocol.')
    seed_evidence = run.get('seed_evidence')
    if seed_evidence is not None:
        if type(seed_evidence) is not dict:
            raise ValueError('Malformed seed evidence.')
        _bound_file(root, seed_evidence.get('path'), seed_evidence.get('sha256'))
    for index, (frame, row) in enumerate(zip(frames, rows, strict=True)):
        if (type(frame) is not dict or frame.get('processed_index') != index
                or type(frame.get('processed_index')) is not int
                or frame.get('source_pts_seconds') != row['acquired_time_seconds']):
            raise ValueError('Motion rows do not match their source presentation times.')
        pts, base = frame.get('source_pts_integer'), frame.get('source_time_base')
        if (type(pts) is not int or type(base) is not list or len(base) != 2
                or any(type(n) is not int or n <= 0 for n in base)
                or float(pts * Fraction(*base)) != frame['source_pts_seconds']):
            raise ValueError('Motion timestamps differ from the retained integer presentation clock.')
    media = run.get('media')
    if media is not None:
        if (type(media) is not dict or type(media.get('bytes')) is not int
                or not 0 < media['bytes'] <= 128 * 1024 * 1024
                or type(media.get('sha256')) is not str
                or not re.fullmatch('[a-f0-9]{64}', media['sha256'])):
            raise ValueError('Invalid motion media receipt.')
        video = _local_file(root, media.get('path'))
        if video.stat().st_size != media['bytes']:
            raise ValueError('Recorded motion media size changed.')
        if _file_digest(video) != media['sha256']:
            raise ValueError('Recorded motion media bytes changed.')
        source = run.get('source')
        if (type(source) is not dict or source.get('sha256') != media['sha256']
                or source.get('bytes') != media['bytes'] or protocol.get('source_sha256') != media['sha256']):
            raise ValueError('Motion media no longer matches its declared original.')
    elif run['status'] == 'completed':
        raise ValueError('A completed motion run requires copied source media.')
    if run['status'] == 'completed' and not rows:
        raise ValueError('A completed motion run requires decoded evidence.')
    return run


def load_motion_bundle(directory):
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('Use an analysis or benchmark directory.')
    report_path, run_path = root / 'report.json', root / 'run.json'
    report, entries = None, []
    if run_path.is_file():
        run = load_motion_run(run_path)
        entries = [('episode', run_path, run, None)]
    elif report_path.is_file():
        report = _document(report_path)
        if type(report) is not dict or report.get('kind') != 'dvidia.motion-benchmark.v1':
            raise ValueError('Unsupported motion benchmark report.')
        cases = report.get('cases')
        if type(cases) is not list or not 1 <= len(cases) <= 50:
            raise ValueError('A motion benchmark requires bounded attempted cases.')
        from .motion_benchmark import score_motion
        protocol = _document(_bound_file(root, 'protocol.json', report.get('protocol_sha256')))
        if type(protocol) is not dict or protocol.get('kind') != 'dvidia.motion-benchmark-protocol.v1':
            raise ValueError('Unsupported benchmark protocol.')
        fixtures = protocol.get('fixtures')
        if type(fixtures) is not list or len(fixtures) != len(cases):
            raise ValueError('Benchmark attempts differ from the frozen fixture protocol.')
        fixture_map = {identity(fixture.get('id')): fixture for fixture in fixtures if type(fixture) is dict}
        if len(fixture_map) != len(cases):
            raise ValueError('Malformed or duplicate benchmark fixtures.')
        _bound_file(root, 'inputs/source.json', protocol.get('sources_manifest_sha256'))
        seen = set()
        for case in cases:
            if type(case) is not dict:
                raise ValueError('Malformed motion case.')
            eid = identity(case.get('id'))
            if eid in seen:
                raise ValueError('Duplicate motion case identifier.')
            seen.add(eid)
            if eid not in fixture_map:
                raise ValueError('Case differs from the frozen benchmark protocol.')
            target = _bound_file(root, case.get('run_path'), case.get('run_sha256'))
            run = load_motion_run(target)
            if run['status'] != case.get('status'):
                raise ValueError('Case status differs from its motion receipt.')
            fixture = fixture_map[eid]
            if (run.get('episode_id') != eid or run.get('source', {}).get('sha256') != fixture.get('video_sha256')
                    or run.get('seed', {}).get('normalized_xyxy') != fixture.get('seed_box_xyxy_normalized')):
                raise ValueError('Benchmark run differs from its frozen source or seed.')
            inputs = root / 'inputs'
            _bound_file(inputs, fixture.get('video_path'), fixture.get('video_sha256'))
            reference = _document(_bound_file(inputs, fixture.get('reference_path'), fixture.get('reference_sha256')))
            if (type(reference) is not dict or reference.get('kind') != 'dvidia.authored-motion-reference.v1'
                    or type(reference.get('rows')) is not list or not 1 <= len(reference['rows']) <= 1800):
                raise ValueError('Unsupported or unbounded authored motion reference.')
            scores = None
            if run['status'] == 'completed':
                scores = _document(_bound_file(root, case.get('scores_path'), case.get('scores_sha256')))
                if scores != case.get('scores') or score_motion(run['rows'], reference.get('rows'),
                        fps=protocol['fixture_recipe']['fps'], horizons=protocol['horizons_seconds']) != scores:
                    raise ValueError('Benchmark scores do not match the retained evaluator evidence.')
            entries.append((eid, target, run, scores))
        completed = sum(entry[2]['status'] == 'completed' for entry in entries)
        if (report.get('attempted') != len(entries) or report.get('completed') != completed
                or report.get('failed') != len(entries)-completed):
            raise ValueError('Benchmark completion counts differ from the retained attempts.')
    else:
        raise ValueError('No motion run or benchmark report found.')
    episodes = {}
    for eid, path, run, scores in entries:
        episodes[eid] = {'run_path': path, 'run': run, 'scores': scores}
    return report, episodes


class MotionServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, directory):
        if address[0] != '127.0.0.1':
            raise ValueError('The motion lab binds only to 127.0.0.1.')
        self.report, self.episodes = load_motion_bundle(directory)
        self.lock = threading.RLock()
        self.media_signatures = {}
        super().__init__(address, MotionHandler)

    @contextmanager
    def media(self, eid):
        entry = self.episodes[eid]
        media = entry['run']['media']
        path = _local_file(entry['run_path'].parent, media['path'])
        with path.open('rb') as handle:
            def signature():
                s = os.fstat(handle.fileno())
                return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns
            before = signature()
            if before[2] != media['bytes']:
                raise ValueError('Motion source media changed.')
            with self.lock:
                if self.media_signatures.get(eid) != before:
                    h = hashlib.file_digest(handle, 'sha256').hexdigest()
                    if signature() != before or h != media['sha256']:
                        raise ValueError('Motion source media changed.')
                    self.media_signatures[eid] = before
                    handle.seek(0)
            yield path, handle, before[2]


class MotionHandler(ObservationHandler):
    server_version = 'DvidiaMotion/0.2-alpha'

    def do_GET(self):
        if not self.origin_ok():
            return self.send(403, {'error': 'Use the motion lab from its loopback address.'})
        path = urlsplit(self.path).path
        try:
            if path == '/':
                return self.send(200, Path(__file__).with_name('motion_studio.html').read_bytes(), 'text/html; charset=utf-8')
            if path == '/api/episodes':
                episodes = []
                for eid, entry in self.server.episodes.items():
                    run = entry['run']
                    episodes.append({'id': eid, 'title': eid.replace('-', ' ').replace('_', ' '),
                                     'status': run['status'], 'frame_count': len(run['rows'])})
                return self.send(200, {'episodes': episodes, 'benchmark': self.server.report})
            match = re.fullmatch(r'/api/episodes/([A-Za-z0-9_-]+)(/video)?', path)
            if match and match[1] in self.server.episodes:
                eid = match[1]
                entry = self.server.episodes[eid]
                if match[2]:
                    if entry['run'].get('media') is None:
                        return self.send(404, {'error': 'No media retained for this failed attempt.'})
                    with self.server.media(eid) as (video, handle, size):
                        return self.stream_video(video, handle, size)
                return self.send(200, {'run': deepcopy(entry['run']), 'scores': entry['scores'],
                                       'video_url': f'/api/episodes/{eid}/video'})
            return self.send(404, {'error': 'No such motion resource.'})
        except (BrokenPipeError, ConnectionResetError):
            return
        except (ValueError, OSError, KeyError, TypeError):
            return self.send(409, {'error': 'Motion evidence changed or is inconsistent. Inspect the bundle.'})

    def do_POST(self):
        return self.send(405, {'error': 'Motion replay is read-only.'}, extra={'Allow': 'GET, HEAD'})


def serve(directory, port=8286):
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError('Use an unprivileged loopback port.')
    server = MotionServer(('127.0.0.1', port), directory)
    print(f'Motion lab: http://127.0.0.1:{port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

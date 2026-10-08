"""Local, data-only footage intake with original hashes and grouped splits.

This module prepares visual evidence; it does not turn human landmarks into
robot actions, execute imported code, or claim unseen-source generalization.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import http.client
import ipaddress
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import ssl
import stat
import subprocess
import tempfile
from urllib.parse import urljoin, urlsplit, urlunsplit
import zipfile

MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_VIDEO_BYTES = 128 * 1024 * 1024
MAX_TOTAL_BYTES = 500 * 1024 * 1024
MAX_SIDECAR_BYTES = 8 * 1024 * 1024
MAX_FILES = 4096
MAX_EPISODES = 1200
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.webm', '.m4v', '.avi', '.mkv'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.heic'}
UNSAFE_EXTENSIONS = {'.py', '.pyc', '.js', '.mjs', '.sh', '.bash', '.exe', '.dll', '.so', '.dylib', '.pkl', '.pickle', '.npy', '.npz'}
DVIDIA_HOSTS = {'dvidia.org', 'www.dvidia.org', 'research.dvidia.org'}
MEDIA_FIELDS = {'id', 'path', 'source_recording_id', 'session_id', 'shoe_pair_id', 'complete', 'task', 'actions_path'}
_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.:-]{0,119}$')
_HOST = re.compile(r'^[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?$')


class FootageDataError(ValueError):
    """Unavailable, malformed or unsafe data input."""


def _fail(message):
    raise FootageDataError(message)


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise FootageDataError('Expected finite JSON data.') from exc


def _json(raw, *, maximum=MAX_MANIFEST_BYTES):
    if type(raw) is not bytes or not raw or len(raw) > maximum:
        _fail(f'JSON is empty or exceeds its {maximum//1024} KiB limit.')
    def pairs(rows):
        obj = {}
        for key, value in rows:
            if key in obj:
                _fail('Duplicate JSON field: '+key)
            obj[key] = value
        return obj
    def constant(token):
        _fail('Nonfinite JSON value: '+token)
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=constant)
        canonical(value)
        def walk(row, depth=0):
            if depth > 48:
                _fail('JSON nesting exceeds 48 levels.')
            if type(row) is dict:
                for item in row.values():
                    walk(item, depth+1)
            elif type(row) is list:
                for item in row:
                    walk(item, depth+1)
        walk(value)
        return value
    except (ValueError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, FootageDataError):
            raise
        raise FootageDataError('Expected finite UTF-8 JSON.') from exc


def _read(path, maximum):
    if path.stat().st_size > maximum:
        _fail('Input exceeds its size limit.')
    with path.open('rb') as handle:
        raw = handle.read(maximum+1)
    if len(raw) > maximum:
        _fail('Input exceeds its size limit.')
    return raw


def _text(value, name, maximum=1200):
    if type(value) is not str or not 1 <= len(value) <= maximum or not value.strip():
        _fail(name+' must be bounded nonempty text.')
    if any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in value):
        _fail(name+' contains unsupported characters.')
    return value


def _relative(value):
    _text(value, 'Artifact path', 1000)
    path = PurePosixPath(value)
    if '\\' in value or ':' in value or path.is_absolute() or any(x in ('', '.', '..') for x in value.split('/')):
        _fail('Artifact paths must remain inside the export directory.')
    return path


def _local_file(root, value):
    relative = _relative(value)
    current = root
    for part in relative.parts:
        current = current/part
        if current.is_symlink():
            _fail('Symlink artifacts are not accepted.')
    try:
        current.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise FootageDataError('Artifact escaped the export directory.') from exc
    if not current.is_file():
        _fail('An explicitly listed artifact is missing: '+value)
    return current


def probe_video(path) -> dict:
    """Read only the supplied local media; external demuxer protocols are off."""
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        _fail('Use a supported video file; images are not complete-action footage.')
    try:
        result = subprocess.run(['ffprobe', '-v', 'error', '-protocol_whitelist', 'file',
                '-format_whitelist', 'mov,matroska,webm,avi',
                '-select_streams', 'v:0', '-show_entries',
                'stream=codec_name,width,height,avg_frame_rate,duration,start_time,nb_frames:format=duration,start_time,size',
                '-of', 'json', str(path)], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FootageDataError('ffprobe is unavailable or could not inspect this video within 10 seconds.') from exc
    if result.returncode != 0 or len(result.stdout) > 1024*1024:
        _fail('A supplied file could not be decoded as local video metadata.')
    info = _json(result.stdout)
    streams = info.get('streams') if type(info) is dict else None
    if type(streams) is not list or len(streams) != 1 or type(streams[0]) is not dict:
        _fail('A supplied video has no readable primary video stream.')
    stream, container = streams[0], info.get('format', {})
    if type(container) is not dict:
        _fail('Video container metadata must be an object.')
    try:
        duration = float(stream.get('duration', container.get('duration')))
        start = float(stream.get('start_time', container.get('start_time', '0')))
        numerator, denominator = stream['avg_frame_rate'].split('/')
        fps = float(numerator)/float(denominator)
        width, height = stream['width'], stream['height']
    except (TypeError, KeyError, ValueError, ZeroDivisionError) as exc:
        raise FootageDataError('Video duration, rate or dimensions are unavailable.') from exc
    if (not all(math.isfinite(x) for x in (duration, start, fps)) or not .01 <= duration <= 86400
            or not 0 <= start <= 86400 or not .1 <= fps <= 1000
            or type(width) is not int or type(height) is not int or not 1 <= width <= 16384 or not 1 <= height <= 16384):
        _fail('Video timing or dimensions exceed the supported envelope.')
    return {'duration_seconds': duration, 'start_seconds': start, 'width': width, 'height': height,
            'fps': fps, 'codec': str(stream.get('codec_name', 'unknown'))}


def _unzip(source, target):
    if source.stat().st_size > MAX_TOTAL_BYTES+MAX_MANIFEST_BYTES:
        _fail('ZIP input exceeds 502 MiB.')
    try:
        with zipfile.ZipFile(source) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_FILES:
                _fail('ZIP contains too many files.')
            total, names = 0, set()
            for item in infos:
                name = item.filename.rstrip('/')
                relative = _relative(name)
                if name in names:
                    _fail('ZIP contains duplicate paths.')
                names.add(name)
                mode = item.external_attr >> 16
                if stat.S_ISLNK(mode) or item.flag_bits & 1:
                    _fail('ZIP symlinks and encrypted entries are not accepted.')
                if item.is_dir():
                    continue
                if relative.suffix.lower() in UNSAFE_EXTENSIONS:
                    _fail('Executable code, pickles and opaque array files are not footage artifacts.')
                total += item.file_size
                if (item.file_size > MAX_VIDEO_BYTES or total > MAX_TOTAL_BYTES
                        or (item.file_size > 1024*1024 and item.file_size > max(item.compress_size, 1)*1000)):
                    _fail('ZIP expanded data exceeds its size or compression limits.')
                destination = target/relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as input_handle, destination.open('xb') as output_handle:
                    copied = 0
                    while block := input_handle.read(1024*1024):
                        copied += len(block)
                        if copied > item.file_size or copied > MAX_VIDEO_BYTES:
                            _fail('ZIP expanded entry exceeds its declared size.')
                        output_handle.write(block)
                if copied != item.file_size:
                    _fail('ZIP expanded entry size mismatch.')
    except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
        raise FootageDataError('ZIP could not be safely extracted.') from exc


def _url(value, hosts):
    if type(value) is not str or len(value) > 2000:
        _fail('Use a bounded public HTTPS artifact URL.')
    parts = urlsplit(value)
    try:
        port = parts.port
    except ValueError as exc:
        raise FootageDataError('Invalid URL port.') from exc
    host = parts.hostname
    if (parts.scheme != 'https' or not host or host not in hosts or not _HOST.fullmatch(host)
            or parts.username is not None or parts.password is not None or parts.query or parts.fragment
            or port not in (None, 443)):
        _fail('Only declared public HTTPS hosts without credentials, query tokens or fragments are accepted.')
    try:
        addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise FootageDataError('Public artifact host could not be resolved.') from exc
    ips = sorted({entry[4][0] for entry in addresses})
    if not ips or any(not ipaddress.ip_address(value).is_global for value in ips):
        _fail('Artifact hosts must resolve exclusively to public IP addresses.')
    return parts, ips[0]


def _fetch(value, maximum, hosts):
    """DNS-public validation plus IP-pinned TLS; redirects repeat both checks."""
    for attempt in range(4):
        parts, address = _url(value, hosts)
        class PinnedHTTPS(http.client.HTTPSConnection):
            def connect(self):
                raw = socket.create_connection((address, 443), timeout=self.timeout)
                try:
                    self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
                except BaseException:
                    raw.close(); raise
        connection = PinnedHTTPS(parts.hostname, timeout=15, context=ssl.create_default_context())
        try:
            connection.request('GET', parts.path or '/', headers={'Accept': 'application/json,video/*', 'User-Agent': 'DVIDIA-Footage-Intake/1'})
            response = connection.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                location = response.getheader('Location')
                if not location or attempt == 3:
                    _fail('Public artifact redirect did not resolve within its limit.')
                value = urljoin(value, location)
                continue
            if response.status != 200:
                _fail('Public artifact fetch failed with HTTP '+str(response.status)+'.')
            declared = response.getheader('Content-Length')
            if declared is not None and (not declared.isdecimal() or int(declared) > maximum):
                _fail('Public artifact exceeds its download limit.')
            raw = response.read(maximum+1)
            if not raw or len(raw) > maximum:
                _fail('Public artifact is empty or exceeds its download limit.')
            return raw, value
        except (OSError, http.client.HTTPException) as exc:
            raise FootageDataError('Public artifact download failed.') from exc
        finally:
            connection.close()
    _fail('Public artifact redirect limit exceeded.')


def _source_url(value):
    parts = urlsplit(value)
    if parts.scheme != 'https' or parts.hostname not in DVIDIA_HOSTS or parts.query or parts.fragment:
        _fail('Public intake starts with a DVIDIA HTTPS store or JSON export URL.')
    if parts.path.startswith('/store/'):
        segments = parts.path.rstrip('/').split('/')
        if len(segments) != 4 or any(not _ID.fullmatch(x) for x in segments[2:]):
            _fail('Invalid public DVIDIA store identifier.')
        return urlunsplit(('https', parts.netloc, '/packs/'+segments[2]+'/'+segments[3]+'/skill.json', '', ''))
    if not parts.path.lower().endswith('.json'):
        _fail('Use a public DVIDIA JSON export or store link.')
    return value


def _manifest(root, source_file=None):
    candidates = [source_file] if source_file else [root/name for name in ('skillspace.training.json', 'export.json', 'skill.json')
                                                   if (root/name).exists() or (root/name).is_symlink()]
    if candidates:
        path = candidates[0]
        if path.is_symlink():
            _fail('Symlink manifests are not accepted.')
        raw = _read(path, MAX_MANIFEST_BYTES)
        value = _json(raw)
        if type(value) is not dict:
            _fail('Footage manifest must be a JSON object.')
    else:
        value, raw = {}, None
    if 'media' not in value:
        files, images = [], False
        for index, path in enumerate(root.rglob('*')):
            if index >= MAX_FILES:
                _fail('Export contains too many filesystem entries.')
            if path.is_symlink():
                _fail('Symlink artifacts are not accepted.')
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                files.append(path)
                if len(files) > MAX_EPISODES:
                    _fail('Export contains more than 1200 videos.')
            elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                images = True
        files.sort()
        rows = [{'id': 'clip-'+sha256(path.relative_to(root).as_posix().encode()).hexdigest()[:20],
                 'path': path.relative_to(root).as_posix()} for path in files]
        if not rows:
            if images:
                _fail('This export contains images only; supply actual video clips.')
            _fail('No actual footage paths are available; demonstration counts and starter metadata are not videos.')
        value = {**value, 'media': rows}
        if raw is None:
            raw = canonical({'kind': 'dvidia.local-footage-discovery', 'media': rows})
    return value, raw


def _groups(episodes):
    parents = list(range(len(episodes)))
    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]; index = parents[index]
        return index
    def union(a, b):
        parents[find(b)] = find(a)
    seen = {}
    for index, episode in enumerate(episodes):
        identities = [('media_sha256', episode['media_sha256'])]
        identities += [(key, episode[key]) for key in ('source_recording_id', 'session_id', 'shoe_pair_id') if episode[key] is not None]
        for identity in identities:
            if identity in seen:
                union(index, seen[identity])
            else:
                seen[identity] = index
    members = {}
    for index, episode in enumerate(episodes):
        members.setdefault(find(index), []).append(episode['id'])
    groups = []
    for ids in members.values():
        group = 'group-'+sha256(canonical(sorted(ids))).hexdigest()[:24]
        groups.append({'id': group, 'episode_ids': sorted(ids)})
        for episode in episodes:
            if episode['id'] in ids:
                episode['group_id'] = group
    return sorted(groups, key=lambda x: x['id'])


def _split(groups, seed, ratios):
    if type(seed) is not int or not 0 <= seed <= 999999:
        _fail('Split seed must be an integer in [0, 999999].')
    if (type(ratios) not in (tuple, list) or len(ratios) != 3
            or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 < x < 1 for x in ratios)
            or not math.isclose(sum(ratios), 1., abs_tol=1e-9, rel_tol=0)):
        _fail('Train/dev/test ratios must be three positive finite values summing to one.')
    if len(groups) < 3:
        _fail('At least three independent recording/session/pair/content groups are required for train, dev and test.')
    ordered = sorted(groups, key=lambda row: sha256(canonical([seed, row['id']])).hexdigest())
    desired = [len(groups)*x for x in ratios]
    counts = [max(1, math.floor(x)) for x in desired]
    while sum(counts) > len(groups):
        eligible = [index for index in range(3) if counts[index] > 1]
        counts[max(eligible, key=lambda i: counts[i]-desired[i])] -= 1
    while sum(counts) < len(groups):
        counts[max(range(3), key=lambda i: desired[i]-counts[i])] += 1
    assignments, offset = {}, 0
    for name, count in zip(('train', 'dev', 'test'), counts):
        for row in ordered[offset:offset+count]:
            assignments[row['id']] = name
        offset += count
    return assignments


def _copy_local(path, target, maximum):
    if path.stat().st_size > maximum:
        _fail('A supplied artifact exceeds its size limit.')
    digest, size = sha256(), 0
    with path.open('rb') as source, target.open('xb') as destination:
        while block := source.read(1024*1024):
            size += len(block)
            if size > maximum:
                _fail('A supplied artifact exceeds its size limit.')
            digest.update(block); destination.write(block)
    if not size:
        _fail('An explicitly listed artifact is empty.')
    return digest.hexdigest(), size


def _rows(manifest):
    rows = manifest.get('media')
    if type(rows) is not list or not 1 <= len(rows) <= MAX_EPISODES:
        _fail('Manifest media must contain 1–1200 actual video entries.')
    ids, checked = set(), []
    for row in rows:
        if type(row) is not dict or set(row)-MEDIA_FIELDS or not {'id', 'path'} <= set(row):
            _fail('Media entries require id/path and only supported lineage, task and action fields.')
        identity = row['id']
        if type(identity) is not str or not _ID.fullmatch(identity) or identity in ids:
            _fail('Clip IDs must be bounded, distinct artifact identifiers.')
        ids.add(identity)
        _text(row['path'], 'Video path', 2000)
        value = {**row}
        for name in ('source_recording_id', 'session_id', 'shoe_pair_id'):
            value[name] = row.get(name)
            if value[name] is not None:
                _text(value[name], name, 200)
        complete = row.get('complete', False)
        if type(complete) is not bool:
            _fail('Complete-action review must be an explicit boolean.')
        value['complete'] = complete
        if row.get('task') is not None:
            _text(row['task'], 'Episode task')
        if row.get('actions_path') is not None:
            _text(row['actions_path'], 'Action sidecar path', 2000)
        checked.append(value)
    return checked


def prepare(source: str | Path, output: Path, *, offline=True, seed=17,
            split_ratios=(.7, .15, .15)) -> dict:
    """Copy exact footage bytes, probe the copies and freeze source-group splits.

    Supports local directories, JSON manifests, safe ZIPs and explicit public
    DVIDIA JSON/store exports when online is selected.  Output paths are relative
    to the prepared dataset directory.  Numeric action semantics remain a
    separate trainer qualification, even when a sidecar is present.
    """
    if type(offline) is not bool:
        _fail('offline must be boolean.')
    output = Path(output)
    if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        _fail('Choose a new or empty output directory; existing data is never overwritten.')
    if type(source) not in (str, Path) and not isinstance(source, Path):
        _fail('Supply a local directory, JSON export, ZIP or public DVIDIA link.')
    text = str(source)
    remote = text.startswith('https://') or '://' in text
    if remote and offline:
        _fail('Offline intake does not fetch links; supply a local export directory, JSON or ZIP.')
    with tempfile.TemporaryDirectory(prefix='dvidia-footage-intake-') as temporary:
        work = Path(temporary)
        staged = work/'prepared'
        staged.mkdir(); (staged/'data').mkdir()
        source_url = None
        if remote:
            source_url = _source_url(text)
            raw_manifest, source_url = _fetch(source_url, MAX_MANIFEST_BYTES, DVIDIA_HOSTS)
            manifest = _json(raw_manifest)
            if type(manifest) is not dict or 'media' not in manifest:
                _fail('Public source exposes metadata only; explicit media paths are required for footage intake.')
            media_hosts = manifest.get('media_hosts', [])
            if type(media_hosts) is not list or len(media_hosts) > 16 or any(type(x) is not str or not _HOST.fullmatch(x) for x in media_hosts):
                _fail('media_hosts must be up to 16 explicit public hostnames.')
            hosts = DVIDIA_HOSTS | set(media_hosts)
            source_kind, root = 'public_dvidia_manifest', None
        else:
            path = Path(source)
            if path.is_symlink() or not path.exists():
                _fail('Local input must exist and must not be a symlink.')
            if path.is_dir():
                root, source_kind = path.resolve(), 'local_directory'
                manifest, raw_manifest = _manifest(root)
            elif path.suffix.lower() == '.zip':
                root, source_kind = work/'export', 'local_zip'
                root.mkdir(); _unzip(path, root)
                children = list(root.iterdir())
                if len(children) == 1 and children[0].is_dir():
                    root = children[0]
                manifest, raw_manifest = _manifest(root)
            elif path.suffix.lower() == '.json':
                root, source_kind = path.resolve().parent, 'local_manifest'
                manifest, raw_manifest = _manifest(root, path)
            else:
                _fail('Supply a footage directory, data-only JSON manifest or safe ZIP export.')
            hosts = set()
        task = manifest.get('task', manifest.get('skill', manifest.get('title', 'unspecified footage task')))
        task = _text(task, 'Task')
        rows = _rows(manifest)
        warnings, episodes, total_bytes = [], [], 0
        for row in rows:
            extension = PurePosixPath(urlsplit(row['path']).path).suffix.lower()
            if extension not in VIDEO_EXTENSIONS:
                if extension in IMAGE_EXTENSIONS:
                    _fail('Manifest images are not video demonstrations; supply actual complete-action clips.')
                _fail('Unsupported video artifact extension.')
            safe_name = sha256(row['id'].encode('utf-8')).hexdigest()[:24]
            relative_media = 'data/'+safe_name+extension
            target = staged/relative_media
            if remote:
                raw, _ = _fetch(urljoin(source_url, row['path']), MAX_VIDEO_BYTES, hosts)
                target.write_bytes(raw)
                digest, size = sha256(raw).hexdigest(), len(raw)
            else:
                original = _local_file(root, row['path'])
                digest, size = _copy_local(original, target, MAX_VIDEO_BYTES)
            total_bytes += size
            if total_bytes > MAX_TOTAL_BYTES:
                _fail('Footage export exceeds its 500 MiB total artifact limit.')
            metadata = probe_video(target)
            episode = {'id': row['id'], 'media_path': relative_media, 'media_sha256': digest,
                       'media_bytes': size, **metadata, 'source_recording_id': row['source_recording_id'],
                       'session_id': row['session_id'], 'shoe_pair_id': row['shoe_pair_id'],
                       'complete': row['complete'], 'task': row.get('task') or task}
            if not any(row[name] is not None for name in ('source_recording_id', 'session_id', 'shoe_pair_id')):
                warnings.append({'episode_id': row['id'], 'kind': 'unknown_lineage',
                    'detail': 'Independent original/session/pair lineage is unknown; a per-file group does not establish novelty.'})
            if not row['complete']:
                warnings.append({'episode_id': row['id'], 'kind': 'complete_action_not_reviewed',
                                 'detail': 'A full task action has not been explicitly marked complete.'})
            if row.get('actions_path') is not None:
                action_path = row['actions_path']
                if PurePosixPath(urlsplit(action_path).path).suffix.lower() != '.json':
                    _fail('Action sidecars must be plain numeric-contract JSON; no pickles or imported code.')
                relative_actions = 'data/'+safe_name+'.actions.json'
                if remote:
                    action_raw, _ = _fetch(urljoin(source_url, action_path), MAX_SIDECAR_BYTES, hosts)
                    (staged/relative_actions).write_bytes(action_raw)
                    action_sha, action_size = sha256(action_raw).hexdigest(), len(action_raw)
                else:
                    action_sha, action_size = _copy_local(_local_file(root, action_path), staged/relative_actions, MAX_SIDECAR_BYTES)
                    action_raw = _read(staged/relative_actions, MAX_SIDECAR_BYTES)
                action_value = _json(action_raw, maximum=MAX_SIDECAR_BYTES)
                if type(action_value) not in (dict, list):
                    _fail('Action sidecar must contain a data-only JSON object or list.')
                total_bytes += action_size
                if total_bytes > MAX_TOTAL_BYTES:
                    _fail('Footage and action sidecars exceed 500 MiB.')
                episode.update(actions_path=relative_actions, actions_sha256=action_sha,
                               actions_bytes=action_size, actions_validation='unvalidated_numeric_contract')
            episodes.append(episode)
        groups = _groups(episodes)
        assignments = _split(groups, seed, split_ratios)
        for episode in episodes:
            episode['split'] = assignments[episode['group_id']]
        for group in groups:
            group['split'] = assignments[group['id']]
        duplicates = {}
        for episode in episodes:
            duplicates.setdefault(episode['media_sha256'], []).append(episode['id'])
        duplicate_groups = [sorted(value) for value in duplicates.values() if len(value) > 1]
        dataset = {'schema_version': 1, 'kind': 'dvidia.footage-dataset', 'task': task,
                   'source_kind': source_kind, 'source_url': source_url,
                   'source_manifest': manifest, 'source_manifest_sha256': sha256(raw_manifest).hexdigest(),
                   'source_manifest_path': 'source_manifest.json', 'episodes': episodes, 'groups': groups,
                   'splits': {name: [row['id'] for row in episodes if row['split'] == name] for name in ('train', 'dev', 'test')},
                   'split_protocol': {'seed': seed, 'ratios': list(split_ratios),
                       'grouping': 'Connected shared recording/session/shoe-pair identities and exact media hashes; unknown lineage defaults per-file.'},
                   'duplicate_media_groups': duplicate_groups, 'lineage_warnings': warnings,
                   'declared_demonstrations': manifest.get('demos'),
                   'actual_media_episodes': len(episodes), 'independent_groups': len(groups),
                   'total_artifact_bytes': total_bytes,
                   'capability': {'visual_ready': True, 'movement_ready': False,
                                  'action_sidecars_present': sum(row.get('actions_path') is not None for row in episodes)},
                   'limitations': ['Hashes and declared lineage establish consistency, not capture authenticity or unseen-source novelty.',
                       'Visual footage intake does not supply calibrated robot actions; movement training requires a separately validated action contract.',
                       'Group counts and complete flags are declared/reviewable inputs, not task-success measurements.']}
        (staged/'source_manifest.json').write_bytes(raw_manifest)
        (staged/'dataset.json').write_bytes(canonical(dataset)+b'\n')
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.rmdir()  # Only the empty output accepted above, never data.
        shutil.move(str(staged), str(output))
        return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('prepare')
    command.add_argument('source')
    command.add_argument('--output', type=Path, required=True)
    command.add_argument('--online', action='store_true', help='Allow explicit public DVIDIA artifact downloads.')
    command.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    dataset = prepare(args.source, args.output, offline=not args.online, seed=args.seed)
    print(json.dumps({'dataset': str(args.output/'dataset.json'), 'actual_media_episodes': dataset['actual_media_episodes'],
                      'independent_groups': dataset['independent_groups'], 'capability': dataset['capability']}))


if __name__ == '__main__':
    main()

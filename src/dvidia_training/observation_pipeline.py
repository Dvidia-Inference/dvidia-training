"""Offline video -> timestamped detections -> editable observation proposals."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time

from .footage_data import _json, _local_file, _read, probe_video, canonical
from .observations import (associate, digest, evaluate_reviews, file_hash, identity,
                           number, propose_relations, text)

MAX_BYTES = 128 * 1024 * 1024
RECIPE = {'id': 'dvidia-observations-v1.1', 'samples_per_second': 2,
          'max_frames': 120, 'max_duration_seconds': 60, 'max_image_side': 640,
          'max_track_gap_seconds': 0.8, 'minimum_association_iou': 0.2,
          'claim_scope': 'sampled_2d_observations', 'automatic_outcome': 'unknown',
          'timebase': 'zero-origin video and container; nonzero starts rejected'}


def validate_manifest(path):
    path = Path(path).resolve(strict=True)
    if path.stat().st_size > 1024*1024:
        raise ValueError('Manifest exceeds 1 MiB.')
    manifest = _json(_read(path, 1024*1024))
    if (type(manifest) is not dict or set(manifest) != {'kind', 'task', 'episodes'}
            or manifest['kind'] != 'dvidia.observation-source.v1'):
        raise ValueError('Expected a dvidia.observation-source.v1 manifest.')
    text(manifest['task'], 'Task')
    episodes = manifest['episodes']
    if type(episodes) is not list or not 1 <= len(episodes) <= 50:
        raise ValueError('Use between one and fifty explicitly permitted videos.')
    ids, total = set(), 0
    for row in episodes:
        required = {'id', 'path', 'title', 'session_id', 'source_recording_id', 'kind',
                    'credit', 'license', 'source_url', 'permission', 'sha256'}
        if type(row) is not dict or set(row) != required:
            raise ValueError('Each source must include the documented provenance fields.')
        eid = identity(row['id'])
        if eid in ids:
            raise ValueError('Duplicate episode id.')
        ids.add(eid)
        for key in ('title', 'session_id', 'source_recording_id', 'credit', 'license', 'source_url'):
            text(row[key], key, 1200)
        if row['kind'] not in ('public', 'synthetic', 'user_authorized') or row['permission'] != 'local_analysis':
            raise ValueError('Every source requires an explicit local-analysis permission declaration.')
        if type(row['sha256']) is not str or not re.fullmatch('[a-f0-9]{64}', row['sha256']):
            raise ValueError('Every source requires an original SHA-256.')
        video = _local_file(path.parent, row['path'])
        total += video.stat().st_size
        if not 0 < video.stat().st_size <= MAX_BYTES or total > 512*1024*1024:
            raise ValueError('Footage exceeds the 128 MiB/clip or 512 MiB/batch limit.')
    return path.parent, manifest


def validate_timebase(video, info):
    """Do not silently shift boxes relative to the original browser media clock."""
    result = subprocess.run(['ffprobe', '-v', 'error', '-protocol_whitelist', 'file',
        '-format_whitelist', 'mov,matroska,webm,avi', '-show_entries', 'format=start_time',
        '-of', 'json', str(video)], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=10, check=False)
    if result.returncode or len(result.stdout) > 1024*1024:
        raise ValueError('Could not establish the original video timeline.')
    container = _json(result.stdout).get('format', {})
    start = float(container.get('start_time', 'nan'))
    number(start, 0, 0, 'Container start time')
    number(info['start_seconds'], 0, 0, 'Video start time')


def extract_frames(video, output, duration):
    """One bounded decode; actual input PTS survive selection and scaling."""
    output.mkdir()
    filters = ("select='isnan(prev_selected_t)+gte(t-prev_selected_t,0.5)',"
               "scale=640:640:force_original_aspect_ratio=decrease,showinfo")
    command = ['ffmpeg', '-hide_banner', '-nostdin', '-nostats', '-loglevel', 'info',
               '-xerror', '-threads', '1', '-filter_threads', '1',
               '-protocol_whitelist', 'file', '-format_whitelist', 'mov,matroska,webm,avi',
               '-copyts', '-i', str(video), '-map', '0:v:0', '-an', '-sn', '-dn',
               '-vf', filters, '-frames:v', '120', '-fps_mode', 'vfr',
               '-threads', '1', str(output/'frame-%04d.png')]
    result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=120, check=False)
    if result.returncode:
        raise ValueError('Video decoding failed; the original is unchanged.')
    times = [float(v) for v in re.findall(rb'\bn:\s*\d+\s+pts:\s*-?\d+\s+pts_time:([-+0-9.eE]+)', result.stderr)]
    files = sorted(output.glob('frame-*.png'))
    if not files or len(times) != len(files) or len(files) > 120:
        raise ValueError('Could not bind every sampled frame to a presentation timestamp.')
    first_pts = times[0]
    number(first_pts, 0, 0, 'First decoded timestamp')
    rows = []
    for index, (stamp, file) in enumerate(zip(times, files, strict=True)):
        t = stamp-first_pts
        number(t, 0, duration+0.05, 'Decoded time')
        if index and stamp <= times[index-1]:
            raise ValueError('Decoded timestamps must strictly increase.')
        rows.append({'id': f'frame-{index+1}', 'time_seconds': t,
                     'source_pts_seconds': stamp, 'path': 'frames/'+file.name,
                     'sha256': file_hash(file)})
    return rows


def run(manifest_path, output, detector):
    started = time.monotonic()
    root, manifest = validate_manifest(manifest_path)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Choose a new run directory; previous evidence is never overwritten.')
    output.mkdir(parents=True)
    metadata = json.loads(canonical(detector.metadata))
    # Freeze the declared recipe and inputs before observing any model output.
    protocol = {'kind': 'dvidia.observation-protocol.v1', 'recipe': RECIPE,
                'model': metadata, 'manifest_sha256': digest(manifest),
                'sources': [{k: v for k, v in row.items() if k != 'path'} for row in manifest['episodes']],
                'created_at': datetime.now(timezone.utc).isoformat(),
                'hardware': {'platform': platform.system(), 'machine': platform.machine()},
                'execution': 'local_offline', 'limits': {'cpu_threads': 4, 'workers': 1}}
    (output/'protocol.json').write_bytes(canonical(protocol))
    entries, failed, seen_hashes, sessions = [], [], {}, {}
    for row in manifest['episodes']:
        episode_started = time.monotonic()
        directory = output/row['id']
        directory.mkdir()
        try:
            source = _local_file(root, row['path'])
            actual_hash = file_hash(source)
            if actual_hash != row['sha256']:
                raise ValueError('Source hash differs from the frozen manifest.')
            info = probe_video(source)
            if info['duration_seconds'] > 60 or max(info['width'], info['height']) > 4096:
                raise ValueError('Pilot inputs must be <=60 seconds and <=4096 pixels per side.')
            validate_timebase(source, info)
            target = directory/('original'+source.suffix.lower())
            shutil.copyfile(source, target)
            if file_hash(target) != actual_hash:
                raise ValueError('Original changed during intake.')
            frames = extract_frames(target, directory/'frames', info['duration_seconds'])
            inference_started = time.monotonic()
            for frame in frames:
                frame['detections'] = detector.detect(directory/frame['path'])
            inference_seconds = time.monotonic()-inference_started
            observations = associate(frames, info['duration_seconds'])
            observations.extend(propose_relations(frames, info['duration_seconds']))
            if len(observations) > 1200:
                raise ValueError('Too many fragmented proposals; keep this failed run for inspection.')
            source_meta = {key: row[key] for key in ('kind', 'credit', 'license', 'source_url',
                           'session_id', 'source_recording_id', 'permission')}
            source_meta.update(sha256=actual_hash, bytes=target.stat().st_size,
                               video_path=target.name, timing=info,
                               duplicate_of=seen_hashes.get(actual_hash))
            seen_hashes.setdefault(actual_hash, row['id'])
            sessions.setdefault(row['session_id'], []).append(row['id'])
            episode = {'kind': 'dvidia.observation-episode.v1', 'id': row['id'],
                'title': row['title'], 'task': manifest['task'], 'source': source_meta,
                'duration_seconds': info['duration_seconds'], 'recipe': {**RECIPE, 'model': metadata},
                'video_url': f"/api/episodes/{row['id']}/video", 'observations': observations,
                'frames': frames, 'metrics': {'elapsed_seconds': time.monotonic()-episode_started,
                    'inference_seconds': inference_seconds, 'frames': len(frames),
                    'detected_frames': sum(bool(f['detections']) for f in frames),
                    'accuracy_status': 'not_assessed'}}
            (directory/'episode.json').write_bytes(canonical(episode))
            entries.append({'id': row['id'], 'episode_sha256': digest(episode)})
        except Exception as exc:
            # Do not expose original machine paths, provider bodies or credentials.
            message = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
            failure = {'id': row['id'], 'error': message[:500], 'status': 'failed',
                       'elapsed_seconds': time.monotonic()-episode_started}
            (directory/'failure.json').write_bytes(canonical(failure))
            failed.append(failure)
    receipt = {'kind': 'dvidia.observation-run.v1', 'protocol_sha256': digest(protocol),
        'episodes': entries, 'failures': failed, 'attempted': len(manifest['episodes']),
        'completed': len(entries), 'elapsed_seconds': time.monotonic()-started,
        'timing_scope': 'intake, decode, inference, association and per-episode writes; excludes model setup and final receipt',
        'accuracy_status': 'not_assessed', 'sessions': sessions,
        'robot_skill_qualified': False, 'geometry': 'image_coordinates_only'}
    (output/'run.json').write_bytes(canonical(receipt))
    return receipt


def load_episodes(directory, *, verify_media=True):
    directory = Path(directory).resolve(strict=True)
    receipt = _json(_read(_local_file(directory, 'run.json'), 2*1024*1024))
    if (type(receipt) is not dict or receipt.get('kind') != 'dvidia.observation-run.v1'
            or type(receipt.get('episodes')) is not list or len(receipt['episodes']) > 50):
        raise ValueError('Unsupported observation run.')
    protocol = _json(_read(_local_file(directory, 'protocol.json'), 2*1024*1024))
    if digest(protocol) != receipt.get('protocol_sha256'):
        raise ValueError('Protocol was modified after the run.')
    episodes = {}
    for item in receipt['episodes']:
        eid = identity(item['id'])
        path = _local_file(directory, eid+'/episode.json')
        episode = _json(_read(path, 8*1024*1024), maximum=8*1024*1024)
        if digest(episode) != item['episode_sha256'] or episode['id'] != eid or eid in episodes:
            raise ValueError('Episode evidence was modified or duplicated.')
        video = _local_file(path.parent, episode['source']['video_path'])
        if verify_media and file_hash(video) != episode['source']['sha256']:
            raise ValueError('Video does not match its observation source.')
        episodes[eid] = episode
    return receipt, episodes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    analyze = commands.add_parser('analyze')
    analyze.add_argument('manifest', type=Path)
    analyze.add_argument('--output', required=True, type=Path)
    analyze.add_argument('--models', required=True, type=Path)
    inspect = commands.add_parser('inspect')
    inspect.add_argument('directory', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'analyze':
        from .observation_models import OpenCVDetector
        detector = OpenCVDetector(args.models)
        result = run(args.manifest, args.output, detector)
    else:
        from .observations import latest_review
        result, episodes = load_episodes(args.directory)
        for eid, episode in episodes.items():
            episode['review'] = latest_review(args.directory/eid, episode)
        result = {**result, 'review': evaluate_reviews(list(episodes.values()))}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 1 if result.get('failures') else 0


if __name__ == '__main__':
    raise SystemExit(main())

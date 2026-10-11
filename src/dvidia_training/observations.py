"""DVIDIA-owned, data-only observation and append-only review contracts."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

from .footage_data import _json, _read, canonical

ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$')
OUTCOMES = ('unknown', 'visible_completion', 'incomplete')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def number(value, low, high, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} must be finite and between {low} and {high}.')
    return value


def text(value, name, limit=300):
    if (type(value) is not str or not value.strip() or len(value) > limit
            or any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in value)):
        raise ValueError(f'{name} must be bounded plain text.')
    return value.strip()


def identity(value):
    if type(value) is not str or not ID.fullmatch(value):
        raise ValueError('Invalid observation identifier.')
    return value


def box(value):
    if type(value) is not list or len(value) != 4:
        raise ValueError('Boxes require four normalized xyxy coordinates.')
    for coordinate in value:
        number(coordinate, 0, 1, 'Box coordinate')
    if value[0] >= value[2] or value[1] >= value[3]:
        raise ValueError('Boxes must have positive width and height.')
    return value


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union else 0.0


def associate(frames, duration, *, max_gap=0.8, minimum_iou=0.2):
    """Greedy same-class IoU association. No optical flow or physical identity claim.

    Only fresh detections extend a track; no boxes are interpolated through
    occlusion. Sequence snippets split after a gap to avoid hiding uncertainty.
    """
    tracks = []
    last_time = -1.0
    for frame in frames:
        t = number(frame['time_seconds'], 0, duration, 'Frame time')
        if t <= last_time:
            raise ValueError('Frame times must increase strictly.')
        last_time = t
        rows = frame['detections']
        if type(rows) is not list or len(rows) > 60:
            raise ValueError('Too many detections in a frame.')
        used = set()
        for row in sorted(rows, key=lambda x: -x['score']):
            label = text(row['label'], 'Object label', 64)
            score = number(row['score'], 0, 1, 'Detector score')
            bounds = box(row['xyxy'])
            candidates = [(iou(track['boxes'][-1]['xyxy'], bounds), n)
                          for n, track in enumerate(tracks)
                          if n not in used and track['object_label'] == label
                          and 0 < t-track['boxes'][-1]['time_seconds'] <= max_gap]
            best = max(candidates, default=(0, -1))
            if best[0] >= minimum_iou:
                n = best[1]
            else:
                n = len(tracks)
                tracks.append({'id': f'track-{n+1}', 'object_label': label, 'boxes': []})
            used.add(n)
            tracks[n]['boxes'].append({'time_seconds': t, 'xyxy': bounds,
                                      'label': label, 'confidence': score})
    for track in tracks:
        times = [b['time_seconds'] for b in track['boxes']]
        # A sampled observation spans only a short evidence interval, not the gap
        # to a future detection or the full duration of a physical action.
        track.update(label=track['object_label'].capitalize()+' visible',
                     start_seconds=times[0], end_seconds=min(duration, times[-1]+min(max_gap, 0.5)),
                     confidence=sum(b['confidence'] for b in track['boxes'])/len(times),
                     evidence='detector', review_state='proposed')
    return tracks


def propose_relations(frames, duration):
    """2D proximity hypotheses, never contact, grasp, depth or task completion."""
    events = []
    for frame in frames:
        hands = [d for d in frame['detections'] if d['label'] == 'hand']
        for row in frame['detections']:
            if row['label'] not in ('cup', 'bottle', 'bowl'):
                continue
            a = row['xyxy']
            if any(iou(a, h['xyxy']) > 0.02 for h in hands):
                t = frame['time_seconds']
                events.append({'id': f'relation-{len(events)+1}',
                    'label': f"Hand near {row['label']} · image overlap",
                    'start_seconds': t, 'end_seconds': min(duration, t+0.5),
                    'boxes': [], 'evidence': '2d-overlap-heuristic',
                    'review_state': 'proposed'})
    return events


def initial_review(episode):
    return {'revision': 0, 'status': 'unreviewed', 'outcome': 'unknown',
            'events': [{'id': o['id'], 'observation_id': o['id'], 'label': o['label'],
                        'start_seconds': o['start_seconds'], 'end_seconds': o['end_seconds'],
                        'decision': 'pending'} for o in episode['observations']]}


def validate_review(payload, episode, current):
    if type(payload) is not dict or set(payload) != {'base_revision', 'events', 'outcome', 'status'}:
        raise ValueError('Expected base_revision, events, outcome and status.')
    if type(payload['base_revision']) is not int or payload['base_revision'] < 0:
        raise ValueError('A nonnegative base revision is required.')
    if payload['base_revision'] != current['revision']:
        raise RuntimeError('This review changed in another tab. Reload before saving.')
    if payload['outcome'] not in OUTCOMES or payload['status'] not in ('in_review', 'reviewed'):
        raise ValueError('Choose a supported review status and visible outcome.')
    events = payload['events']
    if type(events) is not list or len(events) > 1500:
        raise ValueError('Reviews support at most 1500 events.')
    observation_ids = {o['id'] for o in episode['observations']}
    seen, seen_sources, clean = set(), set(), []
    for event in events:
        if type(event) is not dict or set(event) != {'id', 'observation_id', 'label', 'start_seconds', 'end_seconds', 'decision'}:
            raise ValueError('An event contains unsupported fields.')
        eid = identity(event['id'])
        oid = event['observation_id']
        if eid in seen:
            raise ValueError('Duplicate event identifier.')
        seen.add(eid)
        if oid is not None:
            identity(oid)
            if oid not in observation_ids or oid in seen_sources or eid != oid:
                raise ValueError('A proposal must retain its unique original identity.')
            seen_sources.add(oid)
        elif not eid.startswith('manual-'):
            raise ValueError('New events require a manual- identifier.')
        start = number(event['start_seconds'], 0, episode['duration_seconds'], 'Start')
        end = number(event['end_seconds'], 0, episode['duration_seconds'], 'End')
        if start >= end:
            raise ValueError('Event end must be later than its start.')
        if event['decision'] not in ('pending', 'approved', 'rejected'):
            raise ValueError('Invalid event decision.')
        if payload['status'] == 'reviewed' and event['decision'] == 'pending':
            raise ValueError('Resolve all pending suggestions before finishing review.')
        clean.append({**event, 'label': text(event['label'], 'Event label')})
    if seen_sources != observation_ids:
        raise ValueError('Keep every original proposal; reject it rather than deleting its history.')
    return {'revision': current['revision']+1, 'status': payload['status'],
            'outcome': payload['outcome'], 'events': clean,
            'source_sha256': episode['source']['sha256'], 'episode_id': episode['id'],
            'annotation_sha256': digest(episode['observations']),
            'parent_sha256': digest(current), 'review_origin': 'local_operator',
            'created_at': datetime.now(timezone.utc).isoformat()}


def latest_review(directory, episode):
    current = initial_review(episode)
    for path in sorted(Path(directory).glob('review-*.json')):
        if path.is_symlink() or path.stat().st_size > 2*1024*1024:
            raise ValueError('Unsafe review revision.')
        row = _json(_read(path, 2*1024*1024))
        if (type(row) is not dict or row.get('parent_sha256') != digest(current)
                or row.get('source_sha256') != episode['source']['sha256']
                or row.get('annotation_sha256') != digest(episode['observations'])
                or row.get('episode_id') != episode['id']):
            raise ValueError('Review history does not match the source and prior revision.')
        checked = validate_review({'base_revision': current['revision'], 'events': row['events'],
                                   'outcome': row['outcome'], 'status': row['status']}, episode, current)
        if row.get('revision') != checked['revision']:
            raise ValueError('Review revisions are not consecutive.')
        current = row
    return current


def evaluate_reviews(episodes):
    """Review progress is not an independently referenced accuracy score."""
    reviewed = sum(e.get('review', {}).get('status') == 'reviewed' for e in episodes)
    return {'episodes': len(episodes), 'reviewed_episodes': reviewed,
            'accuracy': None, 'accuracy_status': 'not_assessed',
            'reason': 'No independent blinded reference annotations supplied.',
            'robot_skill_qualified': False, 'geometry': 'image_coordinates_only'}

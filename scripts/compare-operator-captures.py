#!/usr/bin/env python3
"""Verify a recorder optimization preserved observations, actions and outcomes."""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
import struct
import zlib

import numpy as np

from dvidia_training import operator_capture as capture, operator_export as export
from dvidia_training import piper_action_capture as files


def _png_pixels(body, width, height):
    """Bounded RGB8/filter-0 decoder for this capture writer; verify every CRC.

    The existing sensor-goal decoder accepts only its 640x480 fixtures. This
    variant binds the dimensions to the recording's validated calibration.
    No image-library dependency, lossy tolerance, ignored chunks or trailing data.
    """
    if (type(width) is not int or type(height) is not int
            or not 16 <= width <= 640 or not 16 <= height <= 480
            or len(body) > capture.tape.MAX_FRAME_BYTES
            or body[:8] != b'\x89PNG\r\n\x1a\n'):
        raise ValueError('Unsupported bounded capture PNG.')
    offset = 8; compressed = bytearray(); header = False; ended = False
    while offset < len(body):
        if offset+12 > len(body):
            raise ValueError('Truncated PNG chunk.')
        length = struct.unpack('>I', body[offset:offset+4])[0]
        if offset+12+length > len(body):
            raise ValueError('Truncated PNG chunk body.')
        kind = body[offset+4:offset+8]; data = body[offset+8:offset+8+length]
        crc = struct.unpack('>I', body[offset+8+length:offset+12+length])[0]
        if zlib.crc32(kind+data) & 0xffffffff != crc:
            raise ValueError('PNG CRC mismatch.')
        if kind == b'IHDR':
            if header or offset != 8 or length != 13:
                raise ValueError('Invalid PNG header.')
            if struct.unpack('>IIBBBBB', data) != (width, height, 8, 2, 0, 0, 0):
                raise ValueError('PNG dimensions or format differ from calibration.')
            header = True
        elif kind == b'IDAT':
            if not header:
                raise ValueError('PNG image precedes header.')
            compressed.extend(data)
        elif kind == b'IEND':
            if not header or not compressed or length or offset+12 != len(body):
                raise ValueError('Invalid PNG terminator.')
            ended = True; break
        else:
            raise ValueError('Unexpected capture PNG chunk.')
        offset += length+12
    if not ended:
        raise ValueError('Incomplete PNG.')
    expected = height*(1+width*3)
    try:
        decoder = zlib.decompressobj()
        pixels = decoder.decompress(compressed, expected+1)
    except zlib.error as exc:
        raise ValueError('Invalid PNG compressed pixels.') from exc
    if (len(pixels) != expected or not decoder.eof
            or decoder.unused_data or decoder.unconsumed_tail):
        raise ValueError('Invalid bounded PNG pixel length.')
    rows = np.frombuffer(pixels, dtype=np.uint8).reshape(height, 1+width*3)
    if np.any(rows[:, 0]):
        raise ValueError('Only capture PNG filter-0 rows are accepted.')
    return rows[:, 1:].reshape(height, width, 3).copy()


def _configuration(root, recording, *, normalize_sources):
    config = deepcopy(capture._read(root, 'context.json')['metadata']['configuration'])
    if capture.tape.digest(config) != recording['binding']['configuration_sha256']:
        raise ValueError('Configuration document does not match its original binding.')
    sources = config.get('implementation_sha256')
    if type(sources) is not dict or set(sources) != set(files.SOURCES):
        raise ValueError('Configuration source inventory differs.')
    for name, value in sources.items():
        capture.tape._sha(value, 'configuration source '+name)
        if (name in recording['operator_implementation_sha256']
                and recording['operator_implementation_sha256'][name] != value):
            raise ValueError('Configuration implementation differs from operator source binding.')
    if normalize_sources:
        config['implementation_sha256'] = {name: 'validated-source-identity' for name in sources}
    return config


def _frame(frame, identities, canonical):
    result = deepcopy(frame)
    ref = result['rgb']; path = ref['path']
    if path not in identities or ref != identities[path]:
        raise ValueError('RGB reference does not match the verified frame bytes.')
    result['rgb'] = deepcopy(canonical[path])
    return result


def _snapshot(snapshot, identities, canonical):
    result = deepcopy(snapshot)
    result['camera']['value'] = _frame(result['camera']['value'], identities, canonical)
    return result


def compare(left, right, *, lossless_png=False):
    if type(lossless_png) is not bool:
        raise ValueError('lossless_png must be explicitly boolean.')
    left, right = Path(left), Path(right)
    reports = [capture.inspect_capture(p) for p in (left, right)]
    capture_reports = [json.loads((p/'capture.json').read_text()) for p in (left,right)]
    recordings = [capture.load_capture(p) for p in (left, right)]
    a, b = recordings
    for key in ('initial_applied_targets', 'profile', 'calibration', 'limits', 'clock', 'goal_position_m',
                'schema', 'scope', 'physical_robot_ready', 'signal_contract', 'control_mode',
                'camera_cadence_seconds', 'source_authenticated', 'human_authenticated', 'outcome'):
        if a[key] != b[key]:
            raise ValueError('Initial observation/calibration contract differs: '+key)
    for key in ('profile_lock_sha256', 'profile_record_sha256', 'model_sha256',
                'scene_sha256', 'limits_sha256', 'calibration_sha256', 'reference_policy_sha256'):
        if a['binding'][key] != b['binding'][key]:
            raise ValueError('Stable recording binding differs: '+key)
    configs = [_configuration(root, recording, normalize_sources=lossless_png)
               for root, recording in zip((left, right), recordings)]
    if configs[0] != configs[1]:
        raise ValueError('Configuration differs beyond declared implementation identities.')
    for key in ('kind', 'source_group', 'case_id'):
        if a['provenance'][key] != b['provenance'][key]:
            raise ValueError('Source provenance differs: '+key)
    for key in ('status', 'reason', 'source_kind', 'case', 'requested_intervals',
                'started_intervals', 'recorded_complete_intervals',
                'uncommitted_native_intervals', 'strict_success', 'teacher_queries',
                'native_evidence_saved', 'error', 'sealed', 'camera_pairs'):
        if capture_reports[0][key] != capture_reports[1][key]:
            raise ValueError('Capture lifecycle/outcome differs: '+key)
    if len(a['steps']) != len(b['steps']):
        raise ValueError('Control interval counts differ.')
    names = [sorted(p.name for p in (root/'frames').iterdir()) for root in (left, right)]
    if names[0] != names[1]:
        raise ValueError('Frame inventory differs.')
    rgbd_files = 0; changed_png_bytes = 0; identities = [{}, {}]
    intrinsics = a['calibration']['intrinsics']
    for name in names[0]:
        bodies = [capture.tape._ordinary_bytes(root/'frames'/name, capture.tape.MAX_FRAME_BYTES)
                  for root in (left, right)]
        if lossless_png and re.fullmatch(r'rgb-\d{4}\.png', name):
            pixels = [_png_pixels(body, intrinsics['width'], intrinsics['height']) for body in bodies]
            if not np.array_equal(pixels[0], pixels[1]):
                raise ValueError('Decoded RGB pixels differ: '+name)
            changed_png_bytes += int(bodies[0] != bodies[1])
            for mapping, body in zip(identities, bodies):
                path = 'frames/'+name
                mapping[path] = dict(path=path, bytes=len(body), sha256=sha256(body).hexdigest())
        elif bodies[0] != bodies[1]:
            raise ValueError('Lossless RGB/depth bytes differ: '+name)
        rgbd_files += 1
    if lossless_png and len(identities[0]) != capture_reports[0]['camera_pairs']:
        raise ValueError('Decoded PNG count differs from camera-pair count.')
    if not lossless_png and capture_reports[0]['frame_bytes'] != capture_reports[1]['frame_bytes']:
        raise ValueError('Capture lifecycle/outcome differs: frame_bytes')
    def normalized(snapshot, side):
        return _snapshot(snapshot, identities[side], identities[0]) if lossless_png else snapshot
    if normalized(a['initial'], 0) != normalized(b['initial'], 1):
        raise ValueError('Initial observation differs.')
    for x, y in zip(a['steps'], b['steps']):
        if set(x) != set(y) or x['perception']['path'] != y['perception']['path']:
            raise ValueError('Control row or perception reference inventory differs.')
        for key in ('index', 'command', 'applied', 'input_event', 'teacher_query'):
            if x[key] != y[key]:
                raise ValueError('Control observation/action differs: '+key)
        for key in ('before', 'after'):
            if normalized(x[key], 0) != normalized(y[key], 1):
                raise ValueError('Control observation/action differs: '+key)
    perception_rows = 0
    expected_perception = [f'{index:04}.json' for index in range(len(a['steps'])+1)]
    if any(sorted(p.name for p in (root/'perception').iterdir()) != expected_perception for root in (left, right)):
        raise ValueError('Perception inventory differs from complete boundaries.')
    for index, name in enumerate(expected_perception):
        items = []
        for side, (root, recording) in enumerate(zip((left, right), recordings)):
            snapshot = recording['initial'] if index == 0 else recording['steps'][index-1]['after']
            body = capture.tape._ordinary_bytes(root/'perception'/name, capture.MAX_DOCUMENT_BYTES)
            reference = dict(path='perception/'+name, bytes=len(body), sha256=sha256(body).hexdigest())
            item = deepcopy(capture._perception(root, reference, snapshot, recording, index))
            # Validate the original digest above before replacing only consequences
            # of PNG byte identities already proven to decode to exact pixels.
            if lossless_png:
                item['frame'] = _frame(item['frame'], identities[side], identities[0])
                item['snapshot_sha256'] = capture.tape.digest(normalized(snapshot, side))
            item.pop('tracker_sha256')  # Source identity; all estimate fields still compare.
            # The sole numeric exclusion is diagnostic wall-clock computation.
            item['estimate']['diagnostics'].pop('processing_seconds', None)
            items.append(item)
        if items[0] != items[1]:
            raise ValueError('Current-frame perception/binding/missingness differs: '+name)
        perception_rows += 1
    native_values = 0
    with np.load(left/'native.npz',allow_pickle=False) as x, np.load(right/'native.npz',allow_pickle=False) as y:
        if set(x.files) != set(y.files):
            raise ValueError('Native array inventory differs.')
        for key in x.files:
            if x[key].dtype != y[key].dtype or not np.array_equal(x[key], y[key]):
                raise ValueError('Native values differ: '+key)
            native_values += x[key].size
    scores = [r['score'] for r in capture_reports]
    if scores[0] != scores[1]:
        raise ValueError('Independently reconstructed native task scores differ.')
    datasets = [export.export_episode(p, reviewed_indices=list(range(len(a['steps']))),
        review_reason='Scripted performance parity only; not reviewed useful training demonstrations.') for p in (left,right)]
    for key in ('features','targets'):
        if (datasets[0][key].dtype != datasets[1][key].dtype
                or not np.array_equal(datasets[0][key],datasets[1][key])):
            raise ValueError('Learning export differs: '+key)
    if datasets[0]['groups'] != datasets[1]['groups']:
        raise ValueError('Learning split groups differ.')
    exclusions = ['source implementation hashes and unique recording/session provenance IDs',
                  'perception diagnostics.processing_seconds and capture wall-time metadata']
    if lossless_png:
        exclusions.append('lossless PNG compressed bytes/length/hash and consequent bound RGB references/snapshot digests; exact decoded uint8 pixels required')
    return dict(schema='dvidia.operator-performance-parity/v1',exact=True,
        comparison_mode='decoded-lossless-png' if lossless_png else 'exact-artifact-bytes',
        png_files_with_different_compressed_bytes=changed_png_bytes,
        frame_bytes=[r['frame_bytes'] for r in capture_reports],
        native_values_compared=int(native_values),rgbd_artifacts_compared=rgbd_files,
        perception_rows_compared=perception_rows,control_intervals=len(a['steps']),
        features_shape=list(datasets[0]['features'].shape),targets_shape=list(datasets[0]['targets'].shape),
        strict_success=[r['strict_success'] for r in reports],
        integrity_sha256=[sha256((p/'integrity.json').read_bytes()).hexdigest() for p in (left,right)],
        exclusions=exclusions,
        physical_robot_ready=False,scope='Semantic and exact numeric parity, not task qualification.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',required=True,type=Path)
    parser.add_argument('--candidate',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--lossless-png',action='store_true',
                        help='Allow PNG compression changes only after exact decoded RGB equality and original binding checks.')
    args=parser.parse_args()
    result=compare(args.baseline,args.candidate,lossless_png=args.lossless_png)
    files._json(args.output,result)
    print(json.dumps(result,indent=2))

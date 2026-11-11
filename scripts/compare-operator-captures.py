#!/usr/bin/env python3
"""Verify a recorder optimization preserved observations, actions and outcomes."""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from dvidia_training import operator_capture as capture, operator_export as export
from dvidia_training import piper_action_capture as files


def compare(left, right):
    left, right = Path(left), Path(right)
    reports = [capture.inspect_capture(p) for p in (left, right)]
    capture_reports = [json.loads((p/'capture.json').read_text()) for p in (left,right)]
    recordings = [capture.load_capture(p) for p in (left, right)]
    a, b = recordings
    for key in ('initial', 'initial_applied_targets', 'profile', 'calibration', 'limits', 'clock', 'goal_position_m'):
        if a[key] != b[key]:
            raise ValueError('Initial observation/calibration contract differs: '+key)
    for key in ('profile_lock_sha256', 'profile_record_sha256', 'model_sha256',
                'scene_sha256', 'limits_sha256', 'calibration_sha256', 'reference_policy_sha256',
                'configuration_sha256'):
        if a['binding'][key] != b['binding'][key]:
            raise ValueError('Stable recording binding differs: '+key)
    for key in ('kind', 'source_group', 'case_id'):
        if a['provenance'][key] != b['provenance'][key]:
            raise ValueError('Source provenance differs: '+key)
    for key in ('status', 'reason', 'source_kind', 'case', 'requested_intervals',
                'started_intervals', 'recorded_complete_intervals',
                'uncommitted_native_intervals', 'strict_success', 'teacher_queries',
                'native_evidence_saved', 'error', 'sealed', 'camera_pairs', 'frame_bytes'):
        if capture_reports[0][key] != capture_reports[1][key]:
            raise ValueError('Capture lifecycle/outcome differs: '+key)
    if len(a['steps']) != len(b['steps']):
        raise ValueError('Control interval counts differ.')
    for x, y in zip(a['steps'], b['steps']):
        for key in ('index', 'before', 'command', 'applied', 'after', 'input_event', 'teacher_query'):
            if x[key] != y[key]:
                raise ValueError('Control observation/action differs: '+key)
    rgbd_files = 0
    for path in sorted((left/'frames').iterdir()):
        peer = right/'frames'/path.name
        if path.read_bytes() != peer.read_bytes():
            raise ValueError('Lossless RGB/depth bytes differ: '+path.name)
        rgbd_files += 1
    if rgbd_files != len(list((right/'frames').iterdir())):
        raise ValueError('Frame inventory differs.')
    perception_rows = 0
    for path in sorted((left/'perception').iterdir()):
        x = json.loads(path.read_text()); y = json.loads((right/'perception'/path.name).read_text())
        for key in ('index', 'snapshot_sha256', 'frame', 'calibration_sha256', 'input_calibration_sha256'):
            if x[key] != y[key]:
                raise ValueError('Perception binding differs: '+key)
        estimates = [deepcopy(z['estimate']) for z in (x,y)]
        for estimate in estimates:
            # The sole numeric exclusion is diagnostic wall-clock computation.
            estimate['diagnostics'].pop('processing_seconds', None)
        if estimates[0] != estimates[1]:
            raise ValueError('Current-frame perception or missingness differs: '+path.name)
        perception_rows += 1
    if perception_rows != len(list((right/'perception').iterdir())):
        raise ValueError('Perception inventory differs.')
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
    return dict(schema='dvidia.operator-performance-parity/v1',exact=True,
        native_values_compared=int(native_values),rgbd_artifacts_compared=rgbd_files,
        perception_rows_compared=perception_rows,control_intervals=len(a['steps']),
        features_shape=list(datasets[0]['features'].shape),targets_shape=list(datasets[0]['targets'].shape),
        strict_success=[r['strict_success'] for r in reports],
        integrity_sha256=[sha256((p/'integrity.json').read_bytes()).hexdigest() for p in (left,right)],
        exclusions=['source implementation hashes and unique recording/session provenance IDs',
                    'perception diagnostics.processing_seconds and capture wall-time metadata'],
        physical_robot_ready=False,scope='Semantic and exact numeric parity, not task qualification.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',required=True,type=Path)
    parser.add_argument('--candidate',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    result=compare(args.baseline,args.candidate)
    files._json(args.output,result)
    print(json.dumps(result,indent=2))

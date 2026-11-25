#!/usr/bin/env python3
"""Compare two trusted tracker revisions on every acquired frame of a sealed capture."""
import argparse
from collections import Counter
from hashlib import sha256
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import runpy
import sys
from time import perf_counter_ns

import numpy as np

from dvidia_training import action_demonstration as tape, operator_capture


def load_tracker(name, path):
    # Compile precisely the bytes named in the receipt, including when a checkout
    # has a stale .pyc file from another source revision.
    body = path.read_bytes()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(body, str(path), 'exec'), module.__dict__)
    return module.estimate_pose, sha256(body).hexdigest()


def comparable(estimate):
    fields = dict(estimate)
    fields['diagnostics'] = dict(fields['diagnostics'])
    del fields['diagnostics']['processing_seconds']
    return tape.canonical(fields)


def describe(times_ns):
    ms = np.asarray(times_ns, dtype=np.float64) / 1e6
    return {'calls': len(ms), 'total_ms': float(ms.sum()), 'mean_ms': float(ms.mean()),
            'p50_ms': float(np.median(ms)), 'p95_ms': float(np.quantile(ms, .95)),
            'max_ms': float(ms.max())}


def load_frame(root, reference, calibration, png_pixels):
    def verified_bytes(identity):
        body = tape._ordinary_bytes(root / str(tape._relative(identity['path'])), tape.MAX_FRAME_BYTES)
        if len(body) != identity['bytes'] or sha256(body).hexdigest() != identity['sha256']:
            raise ValueError('Frame bytes differ from their recorded identity.')
        return body
    k = calibration['intrinsics']
    rgb = png_pixels(verified_bytes(reference['rgb']), k['width'], k['height'])
    depth = np.load(io.BytesIO(verified_bytes(reference['depth'])), allow_pickle=False)
    rgb.flags.writeable = depth.flags.writeable = False
    return rgb, depth


def benchmark(capture, baseline, candidate):
    recording = operator_capture.load_capture(capture)
    if recording['outcome']['status'] != 'complete':
        raise ValueError('Replay requires a complete operator capture.')
    trackers, sources = {}, {}
    for name, path in [('baseline', baseline), ('candidate', candidate)]:
        trackers[name], sources[name] = load_tracker('replay_tracker_' + name, path)
    if sources['baseline'] != recording['tracker_sha256']:
        raise ValueError('Baseline source must match the tracker that produced this recording.')
    png_pixels = runpy.run_path(str(Path(__file__).with_name('compare-operator-captures.py')))['_png_pixels']
    calibration = recording['calibration']
    # The second 50 Hz boundary repeats a 25 Hz frame. Only replay its acquisition.
    frames = []
    for boundary, snapshot in enumerate([recording['initial'], *(step['after'] for step in recording['steps'])]):
        reference = snapshot['camera']['value']
        if reference['index'] == len(frames):
            item = operator_capture._read(capture, f'perception/{boundary:04}.json')
            if item['frame'] != reference or item['snapshot_sha256'] != tape.digest(snapshot):
                raise ValueError('Saved estimate disagrees with the recorded observation.')
            args = dict(frame_index=reference['index'], captured_at_seconds=snapshot['camera']['captured_at_seconds'],
                        available_at_seconds=snapshot['camera']['available_at_seconds'], decision_at_seconds=snapshot['timestamp_seconds'])
            frames.append((reference, args, comparable(item['estimate'])))
    first_rgb, first_depth = load_frame(capture, frames[0][0], calibration, png_pixels)
    for name in ('baseline', 'candidate', 'candidate', 'baseline') * 2:
        trackers[name](first_rgb, first_depth, calibration, **frames[0][1])

    rounds, statuses, differences = [], {name:Counter() for name in trackers}, Counter()
    first_difference = {}
    for repetition in range(2):
        times = {name: [] for name in trackers}
        for index, (reference, clocks, recorded) in enumerate(frames):
            rgb, depth = load_frame(capture, reference, calibration, png_pixels)
            values = {}
            order = ('baseline', 'candidate') if (index + repetition) % 2 == 0 else ('candidate', 'baseline')
            for name in order:
                start = perf_counter_ns()
                result = trackers[name](rgb, depth, calibration, **clocks)
                times[name].append(perf_counter_ns() - start)
                values[name] = comparable(result)
                statuses[name][result['status']] += 1
            for label, equal in [('between_revisions', values['baseline'] == values['candidate']),
                                 ('baseline_vs_recorded', values['baseline'] == recorded),
                                 ('candidate_vs_recorded', values['candidate'] == recorded)]:
                differences[label] += int(not equal)
                if not equal:
                    first_difference.setdefault(label, {'round': repetition + 1, 'frame_index': index})
        rounds.append(times)
    summary = {name: describe(rounds[0][name] + rounds[1][name]) for name in trackers}
    return {'schema':'dvidia.tracker-frame-replay/v1', 'source_sha256':sources,
            'benchmark_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
            'capture_integrity_sha256':sha256((capture/'integrity.json').read_bytes()).hexdigest(),
            'case':recording['provenance']['case_id'], 'provenance':recording['provenance']['kind'],
            'unique_frame_indices':len(frames), 'distinct_rgb_depth_contents':len({(ref['rgb']['sha256'],ref['depth']['sha256']) for ref, _, _ in frames}),
            'warmup_calls_per_variant':4, 'passes':2,
            'order':'One timed call per variant per frame. Alternate the first variant each frame, then reverse on pass 2.',
            'timing_scope':'Tracker function call on decoded, read-only RGB-D. I/O, decoding, comparisons and recorder/control excluded.',
            'sampling_scope':'Two passes replay the same frames from one episode. Scene states may repeat across frame indices.',
            'host':{'platform':platform.platform(), 'machine':platform.machine(), 'python':platform.python_version(), 'numpy':np.__version__,
                    'requested_threads':{key:os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS')}},
            'status_counts':statuses, 'differing_estimates':differences, 'first_difference':first_difference,
            'comparison_scope':'Canonical bytes of every estimate field except diagnostics.processing_seconds.',
            'per_pass':[{name:describe(times[name]) for name in trackers} for times in rounds], 'timings':summary,
            'tracker_time_reduction_percent':100 * (1 - summary['candidate']['total_ms'] / summary['baseline']['total_ms'])}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('capture','baseline','candidate','output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh receipt path.')
    result = benchmark(args.capture, args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as target:
        json.dump(result, target, indent=2, allow_nan=False)
        target.write('\n')
    print(json.dumps({key:result[key] for key in ('differing_estimates','timings','tracker_time_reduction_percent')}, indent=2))
    raise SystemExit(int(any(result['differing_estimates'].values())))

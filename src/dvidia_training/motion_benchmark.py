"""Authored image-motion tracking/forecast benchmark; not detector accuracy.

Future reference trajectories are evaluator-only. The video analyzer receives
only the permitted manifest, first-frame human-authored seed and cadence.
Original DVIDIA fixture generation and scoring code: MIT.
"""
from __future__ import annotations

from bisect import bisect_left
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

from .footage_data import canonical

FPS = 30
WIDTH, HEIGHT = 640, 360
FRAMES = 120
HORIZONS = (0.1, 0.25, 0.5)
CASES = ('stationary', 'constant_velocity', 'stop_start', 'direction_change',
         'brief_occlusion', 'low_texture')


def _hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def _write(path, value):
    Path(path).write_bytes(canonical(value))


def _finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def _point(value):
    return (type(value) in (tuple, list) and len(value) == 2
            and all(_finite(v) for v in value))


def distribution(values):
    """Linear interpolation between sorted samples; nulls for zero samples."""
    values = sorted(values)
    if any(not _finite(value) for value in values):
        raise ValueError('Scores must be finite numbers.')
    def percentile(q):
        position = (len(values)-1)*q
        lower = int(position)
        upper = min(lower+1, len(values)-1)
        return values[lower]+(values[upper]-values[lower])*(position-lower)
    return {'count': len(values), 'mean': sum(values)/len(values) if values else None,
            'p50': percentile(.5) if values else None,
            'p95': percentile(.95) if values else None,
            'max': values[-1] if values else None}


def _nearest(references, times, target, tolerance):
    index = bisect_left(times, target)
    candidates = [i for i in (index-1, index) if 0 <= i < len(times)]
    candidates = [i for i in candidates if abs(times[i]-target) <= tolerance+1e-12]
    if not candidates:
        return None
    # Resolve only floating-point arithmetic ties. Do not widen the declared
    # half-frame tolerance to accommodate a sparse or differently timed grid.
    index = min(candidates, key=lambda i: (round(abs(times[i]-target), 12), times[i]))
    return references[index]


def score_motion(rows, references, *, fps=FPS, horizons=HORIZONS):
    """Score only emitted image anchors/forecasts against independent frames.

    References contain frame_index, source_pts_seconds, anchor_xy_pixels,
    visible, velocity_scoreable and velocity_xy_pixels_per_second. Missing
    sampled rows remain coverage gaps. Missing future references and abstentions
    have explicit denominators and never become zero endpoint errors.
    """
    if not _finite(fps) or fps <= 0 or not references:
        raise ValueError('A positive frame rate and nonempty references are required.')
    times = [ref['source_pts_seconds'] for ref in references]
    if (any(not _finite(t) for t in times)
            or any(b <= a for a, b in zip(times, times[1:]))):
        raise ValueError('Reference acquisition times must strictly increase.')
    if len({ref['frame_index'] for ref in references}) != len(references):
        raise ValueError('Reference frame identities must be unique.')
    if any(not _point(ref.get('anchor_xy_pixels')) or type(ref.get('visible')) is not bool
           for ref in references):
        raise ValueError('References require finite image anchors and visibility.')
    if any(not _finite(h) or h <= 0 for h in horizons) or len(set(horizons)) != len(horizons):
        raise ValueError('Horizons must be unique positive seconds.')
    row_times = [row['acquired_time_seconds'] for row in rows]
    if (any(not _finite(t) for t in row_times)
            or any(b <= a for a, b in zip(row_times, row_times[1:]))):
        raise ValueError('Observation acquisition times must strictly increase.')
    tolerance = 1/(2*fps)
    current_time_tolerance = 2e-6
    matched, tracked_visible = set(), set()
    anchor_errors, velocity_errors = [], []
    lost_visible = false_tracking = unmatched_current = 0
    lost_rows = sum(row.get('status') != 'tracked' for row in rows)
    first_lost = None
    reacquired_after_loss = 0
    horizon_scores = {str(h): {'horizon_seconds': h, 'attempted_rows': len(rows),
        'future_reference_available': 0, 'terminal_censored': 0,
        'future_reference_gap': 0, 'current_reference_unmatched': 0,
        'lost_abstentions': 0, 'velocity_warmup_abstentions': 0,
        'missing_forecast_abstentions': 0, 'constant_velocity': [],
        'hold_current_paired': [], 'hold_current_all_tracked': [],
        'reference_time_offsets': [], 'region_radii': [], 'region_covered': 0}
        for h in horizons}
    for row in rows:
        t = row['acquired_time_seconds']
        ref = _nearest(references, times, t, current_time_tolerance)
        tracked = row.get('status') == 'tracked'
        center = row.get('estimated_center_xy_pixels')
        velocity = row.get('velocity_xy_pixels_per_second')
        if tracked and not _point(center):
            raise ValueError('Tracked observations require a finite image anchor.')
        if velocity is not None and not _point(velocity):
            raise ValueError('Velocity must be null or a finite pixel/s vector.')
        if not tracked and first_lost is None:
            first_lost = t
        elif tracked and first_lost is not None:
            reacquired_after_loss += 1
            first_lost = None
        if ref is None:
            unmatched_current += 1
        else:
            matched.add(ref['frame_index'])
            if tracked and ref['visible']:
                tracked_visible.add(ref['frame_index'])
                anchor_errors.append(math.dist(center, ref['anchor_xy_pixels']))
                if ref.get('velocity_scoreable') and _point(velocity):
                    truth_velocity = ref.get('velocity_xy_pixels_per_second')
                    if not _point(truth_velocity):
                        raise ValueError('Scoreable reference velocity must be finite.')
                    velocity_errors.append(math.dist(velocity, truth_velocity))
            elif tracked:
                false_tracking += 1
            elif ref['visible']:
                lost_visible += 1
        for h in horizons:
            score = horizon_scores[str(h)]
            target = t+h
            if target > times[-1]+1e-12:
                score['terminal_censored'] += 1
                continue
            future = _nearest(references, times, target, tolerance)
            if future is None:
                score['future_reference_gap'] += 1
                continue
            score['future_reference_available'] += 1
            score['reference_time_offsets'].append(future['source_pts_seconds']-target)
            if ref is None:
                score['current_reference_unmatched'] += 1
                continue
            if not tracked:
                score['lost_abstentions'] += 1
                continue
            hold_error = math.dist(center, future['anchor_xy_pixels'])
            score['hold_current_all_tracked'].append(hold_error)
            if velocity is None:
                score['velocity_warmup_abstentions'] += 1
                continue
            predictions = [p for p in row.get('forecasts', [])
                           if abs(p.get('horizon_seconds', -1)-h) <= 1e-8]
            if not predictions:
                score['missing_forecast_abstentions'] += 1
                continue
            if len(predictions) != 1:
                raise ValueError('A horizon must have at most one forecast.')
            prediction = predictions[0]
            if (not _point(prediction.get('center_xy_pixels'))
                    or not _finite(prediction.get('target_time_seconds'))
                    or abs(prediction['target_time_seconds']-target) > 1e-8):
                raise ValueError('Forecast must target its declared causal horizon.')
            error = math.dist(prediction['center_xy_pixels'], future['anchor_xy_pixels'])
            score['constant_velocity'].append(error)
            score['hold_current_paired'].append(hold_error)
            radius = prediction.get('uncertainty_radius_pixels')
            if radius is not None:
                if not _finite(radius) or radius < 0:
                    raise ValueError('Uncertainty radius must be null or nonnegative.')
                score['region_radii'].append(radius)
                score['region_covered'] += error <= radius
    visible_refs = sum(ref['visible'] for ref in references)
    for score in horizon_scores.values():
        for key in ('constant_velocity', 'hold_current_paired', 'hold_current_all_tracked'):
            score[key] = distribution(score[key])
        offsets = score.pop('reference_time_offsets')
        score['reference_time_offset_seconds'] = distribution(offsets)
        score['reference_max_absolute_time_offset_seconds'] = max(map(abs, offsets), default=None)
        radii = score.pop('region_radii')
        score['uncertainty'] = {'radius_pixels': distribution(radii),
            'covered': score.pop('region_covered'), 'evaluated': len(radii),
            'calibrated': False}
        scored = score['constant_velocity']['count']
        eligible = score['future_reference_available']
        score['forecast_answer_coverage'] = scored/eligible if eligible else None
        score['endpoint_error_units'] = 'pixels'
    return {'kind': 'dvidia.motion-scores.v1', 'units': {'position': 'pixels', 'velocity': 'pixels/second'},
        'reference_matching': {'rule': 'nearest actual reference PTS; earlier frame on a half-frame tie',
            'maximum_time_tolerance_seconds': tolerance, 'fps': fps,
            'current_reference_rule': 'same acquisition PTS within numerical serialization tolerance; off-grid rows unmatched',
            'current_time_tolerance_seconds': current_time_tolerance,
            'terminal_rule': 'targets after the last reference frame are censored, never rounded backward',
            'observations': 'unsampled references remain gaps; no extra tracking is run for scoring'},
        'tracking': {'reference_frames': len(references), 'reference_visible_frames': visible_refs,
            'attempted_observation_rows': len(rows), 'matched_reference_frames': len(matched),
            'unsampled_reference_frames': len(references)-len(matched),
            'unmatched_observation_rows': unmatched_current,
            'tracked_visible_reference_frames': len(tracked_visible),
            'track_coverage_visible': len(tracked_visible)/visible_refs if visible_refs else None,
            'lost_rows': lost_rows, 'lost_visible_rows': lost_visible,
            'tracked_during_reference_occlusion_rows': false_tracking,
            'reacquisitions_after_reported_loss': reacquired_after_loss,
            'automatic_reacquisition_supported': False,
            'anchor_error_pixels': distribution(anchor_errors),
            'velocity_error_pixels_per_second': distribution(velocity_errors),
            'velocity_scope': 'visible frames with available estimate; authored discontinuity neighborhoods excluded'},
        'forecasts': horizon_scores,
        'claim_scope': 'authored fixed-camera image-anchor tracking and passive forecasting only'}


def _trajectory(case, index):
    if case in ('stationary', 'low_texture'):
        x = 160
    elif case in ('constant_velocity', 'brief_occlusion'):
        x = 100+3*index
    elif case == 'stop_start':
        x = 100+3*min(index, 40)+3*max(0, index-80)
    elif case == 'direction_change':
        x = 100+3*min(index, 60)-3*max(0, index-60)
    else:
        raise ValueError('Unsupported authored fixture case.')
    return x, 180


def _frame_timestamps(video):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'frame=best_effort_timestamp:stream=time_base', '-of', 'json', str(video)],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=30, check=False)
    if result.returncode or len(result.stdout) > 1024*1024:
        raise ValueError('Could not establish generated frame identities.')
    decoded = json.loads(result.stdout)
    timebase = Fraction(decoded['streams'][0]['time_base'])
    times = [float(int(row['best_effort_timestamp'])*timebase) for row in decoded['frames']]
    if (len(times) != FRAMES or times[0] != 0
            or any(b <= a for a, b in zip(times, times[1:]))):
        raise ValueError('Generated video does not have the declared zero-origin frame grid.')
    if any(abs(t-index/FPS) > 1e-6 for index, t in enumerate(times)):
        raise ValueError('Generated video frame timing differs from the frozen recipe.')
    return times


def generate_fixtures(directory):
    """Encode own fixtures; write evaluator references before any tracking."""
    import numpy as np
    directory = Path(directory)
    directory.mkdir(parents=True)
    texture_seed = 4702
    random = np.random.default_rng(texture_seed)
    texture = random.integers(40, 235, size=(64, 80, 3), dtype=np.uint8)
    texture[::8, :, :] = 245
    texture[:, ::8, :] = 25
    episodes, fixture_receipts = [], []
    for case in CASES:
        video = directory/(case+'.mp4')
        raw = directory/(case+'.rgb')
        raw_hash = hashlib.sha256()
        with raw.open('wb') as handle:
            for index in range(FRAMES):
                frame = np.full((HEIGHT, WIDTH, 3), 230, dtype=np.uint8)
                x, y = _trajectory(case, index)
                visible = not (case == 'brief_occlusion' and 45 <= index < 65)
                if visible:
                    patch = np.full((64, 80, 3), 120, dtype=np.uint8) if case == 'low_texture' else texture
                    frame[y-32:y+32, x-40:x+40] = patch
                data = frame.tobytes()
                raw_hash.update(data)
                handle.write(data)
        command = ['ffmpeg', '-hide_banner', '-nostdin', '-loglevel', 'error',
            '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{WIDTH}x{HEIGHT}',
            '-r', str(FPS), '-i', str(raw), '-an', '-c:v', 'libx264', '-preset', 'fast',
            '-crf', '18', '-pix_fmt', 'yuv420p', '-threads', '1', '-map_metadata', '-1',
            '-movflags', '+faststart', str(video)]
        try:
            encoded = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, timeout=60, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ValueError('Authored fixture encoding requires FFmpeg and a completed 60-second encoding budget.') from exc
        finally:
            raw.unlink(missing_ok=True)
        if encoded.returncode:
            raise ValueError('Authored fixture encoding failed.')
        times = _frame_timestamps(video)
        refs = []
        boundaries = {'stop_start': (40, 80), 'direction_change': (60,)}.get(case, ())
        for index, stamp in enumerate(times):
            current = _trajectory(case, index)
            velocity = None
            if index:
                previous = _trajectory(case, index-1)
                dt = stamp-times[index-1]
                velocity = [(current[d]-previous[d])/dt for d in (0, 1)]
            refs.append({'frame_index': index, 'source_pts_seconds': stamp,
                'anchor_xy_pixels': list(current), 'velocity_xy_pixels_per_second': velocity,
                'velocity_scoreable': index > 0 and not any(abs(index-b) <= 2 for b in boundaries),
                'visible': not (case == 'brief_occlusion' and 45 <= index < 65)})
        reference_path = directory/(case+'.reference.json')
        _write(reference_path, {'kind': 'dvidia.authored-motion-reference.v1', 'rows': refs,
            'role': 'evaluator_only_never_supplied_to_tracker', 'license': 'MIT',
            'anchor_semantics': 'initial manually seeded image region center plus authored translation'})
        x, y = _trajectory(case, 0)
        seed = [(x-36)/WIDTH, (y-28)/HEIGHT, (x+36)/WIDTH, (y+28)/HEIGHT]
        episodes.append({'id': case, 'path': video.name, 'title': 'Authored '+case.replace('_', ' '),
            'session_id': 'dvidia-motion-authoring-v1', 'source_recording_id': case,
            'kind': 'synthetic', 'credit': 'DVIDIA authored geometric fixture', 'license': 'MIT',
            'source_url': 'https://github.com/Dvidia-Inference/dvidia-training',
            'permission': 'local_analysis', 'sha256': _hash(video)})
        fixture_receipts.append({'id': case, 'video_path': video.name, 'video_sha256': _hash(video),
            'video_bytes': video.stat().st_size, 'raw_rgb_sha256': raw_hash.hexdigest(),
            'raw_rgb_bytes': FRAMES*WIDTH*HEIGHT*3, 'reference_path': reference_path.name,
            'reference_sha256': _hash(reference_path), 'seed_box_xyxy_normalized': seed,
            'frame_count': len(times), 'source_duration_seconds': FRAMES/FPS})
    manifest = {'kind': 'dvidia.observation-source.v1',
                'task': 'authored image-anchor tracking and passive motion forecasts', 'episodes': episodes}
    _write(directory/'source.json', manifest)
    return fixture_receipts


def _version(command):
    try:
        output = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=10, check=False)
        return output.stdout.decode('utf-8', errors='replace').splitlines()[0][:200]
    except (OSError, IndexError, subprocess.TimeoutExpired):
        return 'unavailable'


def _peak_memory():
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {'bytes': int(peak if platform.system() == 'Darwin' else peak*1024),
                'scope': 'process-lifetime resident high-water mark including fixture generation; not a stage cap'}
    except (ImportError, AttributeError):
        return {'bytes': None, 'scope': 'unavailable on this platform'}


def run_benchmark(output):
    """Run all cases in a new directory, preserving every failed attempt."""
    from .motion_video import analyze_video
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Choose a new benchmark directory; prior evidence is never overwritten.')
    output.mkdir(parents=True)
    fixture_started = time.monotonic()
    fixtures = generate_fixtures(output/'inputs')
    generation_seconds = time.monotonic()-fixture_started
    protocol = {'kind': 'dvidia.motion-benchmark-protocol.v1',
        'created_at': datetime.now(timezone.utc).isoformat(), 'cases': list(CASES),
        'fixture_recipe': {'width': WIDTH, 'height': HEIGHT, 'fps': FPS, 'frames': FRAMES,
            'texture_rng': 'numpy.default_rng(4702)', 'object_size_pixels': [80, 64],
            'seed_inset_pixels': 4, 'background_rgb': [230, 230, 230],
            'encoder': 'libx264 fast crf18 yuv420p single-thread; no metadata',
            'trajectory_code_sha256': _hash(__file__)},
        'horizons_seconds': list(HORIZONS), 'sample_hz': FPS,
        'sources_manifest_sha256': _hash(output/'inputs/source.json'),
        'fixtures': fixtures, 'reference_access': 'scorer only; never a tracker argument',
        'baseline_comparison': 'hold current on the identical evaluable anchored rows with a constant-velocity forecast',
        'hardware': {'system': platform.system(), 'machine': platform.machine(),
            'processor': platform.processor() or None, 'python': platform.python_version(),
            'os_release': platform.release(), 'logical_cpu_count': os.cpu_count()},
        'software': {'ffmpeg': _version(['ffmpeg', '-version']), 'ffprobe': _version(['ffprobe', '-version'])},
        'claim_scope': 'authored synthetic image tracking and passive forecasting; no detection, real-world generalization or robot control'}
    _write(output/'protocol.json', protocol)
    started = time.monotonic()
    results = []
    for fixture in fixtures:
        case = fixture['id']
        run_dir = output/'cases'/case
        attempted = time.monotonic()
        try:
            receipt = analyze_video(output/'inputs/source.json', case, run_dir,
                seed_box=fixture['seed_box_xyxy_normalized'], label='authored textured region', sample_hz=FPS)
            if receipt.get('status') != 'completed':
                results.append({'id': case, 'status': 'failed', 'run_path': f'cases/{case}/run.json',
                    'run_sha256': _hash(run_dir/'run.json'),
                    'errors': receipt.get('errors', []),
                    'error': receipt.get('error', 'Analyzer failed; see retained run receipt'),
                    'elapsed_seconds': time.monotonic()-attempted})
                continue
            rows = receipt.get('rows')
            if rows is None:
                rows_doc = json.loads((run_dir/'rows.json').read_bytes())
                rows = rows_doc if type(rows_doc) is list else rows_doc['rows']
            reference_raw = (output/'inputs'/fixture['reference_path']).read_bytes()
            if hashlib.sha256(reference_raw).hexdigest() != fixture['reference_sha256']:
                raise ValueError('Evaluator reference differs from the frozen fixture protocol.')
            references = json.loads(reference_raw)['rows']
            score = score_motion(rows, references)
            _write(run_dir/'scores.json', score)
            results.append({'id': case, 'status': 'completed', 'run_path': f'cases/{case}/run.json',
                'run_sha256': _hash(run_dir/'run.json'), 'scores_path': f'cases/{case}/scores.json',
                'scores_sha256': _hash(run_dir/'scores.json'), 'scores': score,
                'elapsed_seconds': time.monotonic()-attempted, 'timings': receipt.get('timings', {})})
        except Exception as exc:
            run_dir.mkdir(parents=True, exist_ok=True)
            error = str(exc)[:300] if isinstance(exc, ValueError) else type(exc).__name__
            if not (run_dir/'run.json').exists():
                _write(run_dir/'run.json', {'kind': 'dvidia.motion-run.v1', 'status': 'failed', 'error': error})
            results.append({'id': case, 'status': 'failed', 'run_path': f'cases/{case}/run.json',
                            'run_sha256': _hash(run_dir/'run.json'),
                            'error': error, 'elapsed_seconds': time.monotonic()-attempted})
    elapsed = time.monotonic()-started
    completed = sum(row['status'] == 'completed' for row in results)
    report = {'kind': 'dvidia.motion-benchmark.v1', 'protocol_sha256': _hash(output/'protocol.json'),
        'attempted': len(fixtures), 'completed': completed, 'failed': len(fixtures)-completed,
        'cases': results, 'timing': {'batch_elapsed_seconds': elapsed,
            'fixture_generation_seconds': generation_seconds,
            'attempted_source_duration_seconds': len(fixtures)*FRAMES/FPS,
            'completed_source_duration_seconds': completed*FRAMES/FPS,
            'batch_processing_time_source_duration_ratio': elapsed/(len(fixtures)*FRAMES/FPS)
                if completed == len(fixtures) else None,
            'ratio_availability': 'all cases must complete; failed batches cannot appear faster through early failure',
            'scope': 'batch intake, probing, streaming decode, causal tracking/forecast, writes and scoring; fixture generation/provisioning excluded',
            'live_sensor_to_action_latency_measured': False}, 'memory': _peak_memory(),
        'limitations': ['Six authored fixtures are engineering integration evidence, not held-out real footage.',
            'Image anchors and pixel/s have no calibrated physical scale.',
            'Seeded sparse tracking is not object detection or semantic task understanding.',
            'Offline source-duration ratio does not establish live latency or deadline guarantees.',
            'No automatic reacquisition or robot commands; losses abstain.',
            'Uncertainty radii are heuristic and uncalibrated.'],
        'robot_skill_qualified': False, 'trained_model': False}
    _write(output/'report.json', report)
    return report

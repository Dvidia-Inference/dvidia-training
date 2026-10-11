"""Causal, CPU-only image motion from one manually seeded local video region.

The tracked quantity is an image anchor: the seed-box center translated by the
robust displacement of its initial features. It is not a physical centroid,
object detector, camera-motion compensation or robot control signal.
"""
from __future__ import annotations

from datetime import datetime, timezone
from fractions import Fraction
import math
import os
from pathlib import Path
import platform
import queue
import re
import shutil
import signal
import subprocess
import threading
import time

from .footage_data import _local_file, canonical, probe_video
from .observation_pipeline import validate_manifest, validate_timebase
from .observations import box, digest, file_hash, identity, number, text


MAX_IMAGE_SIDE = 640
MAX_DECODE_SECONDS = 120
MAX_STDERR_BYTES = 2 * 1024 * 1024
MAX_STDERR_LINE_BYTES = 8192
MAX_TIMESTAMP_QUEUE = 16
MAX_TRACK_GAP_SECONDS = 0.25
MIN_FEATURES = 6
FEATURE_RECIPE = {
    'maximum_initial_features': 100, 'quality_level': 0.01,
    'minimum_feature_distance_pixels': 5, 'feature_block_size': 7,
    'lk_window_pixels': 21, 'lk_pyramid_levels': 3,
    'maximum_forward_backward_error_pixels': 1.5,
    'minimum_features': MIN_FEATURES, 'minimum_valid_fraction': 0.5,
    'minimum_consensus_fraction': 0.7, 'minimum_consensus_radius_pixels': 2.0,
    'maximum_consensus_radius_pixels': 4.0,
    'maximum_track_gap_seconds': MAX_TRACK_GAP_SECONDS,
    'maximum_step_pixels': 'max(32, 0.10 * maximum analysis image side)',
    'lost_track_policy': 'terminal; no replenishment, reacquisition or interpolation',
}
LIMITATIONS = [
    'Fixed-camera scope; camera movement is not compensated or detected.',
    'Human-authored seed and label; no automatic semantic object detection.',
    'Image anchor from initial feature translation, not a physical object centroid.',
    'Units are analysis-image pixels and pixels/second, not metric 3D motion.',
    'Sparse optical flow can drift, fail under deformation or follow background features.',
    'Reported tracking uncertainty is an uncalibrated heuristic, not a probability guarantee.',
    'Lost tracks remain lost; no predictions bridge occlusion or large timestamp gaps.',
    'Offline causal replay; inference uses no later frames, but live robot latency is unmeasured.',
    'Local-analysis permission does not confer publication or redistribution rights.',
    'Resource settings are requests and input bounds; peak RAM/thread usage is unmeasured.',
    'No physical interaction, actuator command, depth or robot safety qualification.',
]
_PTS = re.compile(rb'\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:([-+0-9.eE]+)')
_TIMEBASE = re.compile(rb'\bconfig in time_base:\s*(\d+)/(\d+)')


class MotionVideoError(ValueError):
    """A bounded, user-readable motion intake or decoding error."""


def _dimensions(width, height):
    scale = min(1.0, MAX_IMAGE_SIDE / max(width, height))
    return max(1, int(width * scale)), max(1, int(height * scale))


def _stop_process(process):
    """Stop only the child process group created for this invocation."""
    try:
        if os.name == 'posix':
            os.killpg(process.pid, signal.SIGKILL)
        elif process.poll() is None:
            process.kill()
    except ProcessLookupError:
        pass
    except PermissionError:
        # Some hosted sandboxes forbid group signals but allow terminating an
        # owned direct child. Never signal an unrelated group or process.
        if process.poll() is None:
            process.kill()


def _decoded_frames(video, width, height, *, sample_hz, max_frames, duration,
                    timeout_seconds=MAX_DECODE_SECONDS):
    """Stream bounded raw frames with their selected *input* presentation times.

    FFmpeg showinfo runs before rawvideo output. A draining stderr thread binds
    one metadata row to each raw frame; a watchdog stops blocked decoding. PTS
    are never reconstructed from nominal frame rate or OpenCV position values.
    """
    interval = 1.0 / sample_hz
    filters = ("select='isnan(prev_selected_t)+gte(t-prev_selected_t,"
               f"{interval:.12f}-0.000000001)',"
               f"trim=end_frame={max_frames},scale={width}:{height},setsar=1,showinfo")
    command = ['ffmpeg', '-hide_banner', '-nostdin', '-nostats', '-loglevel', 'info',
               '-xerror', '-threads', '1', '-filter_threads', '1',
               '-protocol_whitelist', 'file', '-format_whitelist', 'mov,matroska,webm,avi',
               '-copyts', '-noautorotate', '-i', str(video), '-map', '0:v:0',
               '-an', '-sn', '-dn', '-t', str(duration), '-vf', filters,
               '-frames:v', str(max_frames), '-fps_mode', 'vfr',
               '-threads', '1', '-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1']
    try:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=(os.name == 'posix'))
    except OSError as exc:
        raise MotionVideoError('FFmpeg is unavailable for bounded local decoding.') from exc
    metadata = queue.Queue(maxsize=MAX_TIMESTAMP_QUEUE)
    errors, ended, timed_out = [], threading.Event(), threading.Event()
    stderr_bytes = [0]

    def consume_metadata():
        source_timebase = None
        try:
            while line := process.stderr.readline(MAX_STDERR_LINE_BYTES + 1):
                stderr_bytes[0] += len(line)
                if len(line) > MAX_STDERR_LINE_BYTES or stderr_bytes[0] > MAX_STDERR_BYTES:
                    errors.append('Decoder diagnostics exceeded the bounded metadata budget.')
                    _stop_process(process)
                    return
                clock = _TIMEBASE.search(line)
                if clock:
                    numerator, denominator = int(clock.group(1)), int(clock.group(2))
                    if numerator <= 0 or denominator <= 0:
                        errors.append('Decoder time base must be a positive rational clock.')
                        _stop_process(process)
                        return
                    clock_value = Fraction(numerator, denominator)
                    if source_timebase is not None and source_timebase != clock_value:
                        errors.append('Decoder presentation time base changed within the selected stream.')
                        _stop_process(process)
                        return
                    source_timebase = clock_value
                match = _PTS.search(line)
                if match:
                    if source_timebase is None:
                        errors.append('Decoder presented a frame before declaring its rational time base.')
                        _stop_process(process)
                        return
                    # pts_time text is rounded by FFmpeg; preserve the actual
                    # integer PTS clock rather than that formatted approximation.
                    integer_pts = int(match.group(2))
                    stamp = float(integer_pts * source_timebase)
                    if not math.isfinite(stamp):
                        errors.append('Decoder reported a nonfinite presentation timestamp.')
                        _stop_process(process)
                        return
                    try:
                        metadata.put((int(match.group(1)), stamp, integer_pts,
                                      [source_timebase.numerator, source_timebase.denominator]), timeout=1)
                    except queue.Full:
                        errors.append('Decoder presentation timestamps exceeded the bounded queue.')
                        _stop_process(process)
                        return
        except (OSError, ValueError):
            errors.append('Decoder metadata could not be read.')
            _stop_process(process)
        finally:
            ended.set()

    def expire():
        timed_out.set()
        _stop_process(process)

    reader = threading.Thread(target=consume_metadata, daemon=True,
                              name='dvidia-motion-timestamps')
    watchdog = threading.Timer(timeout_seconds, expire)
    watchdog.daemon = True
    reader.start()
    watchdog.start()
    frame_bytes, count, last_stamp = width * height * 3, 0, None
    try:
        while count < max_frames:
            raw = bytearray()
            while len(raw) < frame_bytes:
                block = process.stdout.read(frame_bytes - len(raw))
                if not block:
                    break
                raw.extend(block)
            if timed_out.is_set():
                raise MotionVideoError('Local video decoding exceeded its 120-second watchdog.')
            if errors:
                raise MotionVideoError(errors[0])
            if not raw:
                break
            if len(raw) != frame_bytes:
                raise MotionVideoError('Decoder returned an incomplete raw video frame.')
            try:
                index, stamp, integer_pts, source_timebase = metadata.get(timeout=2)
            except queue.Empty as exc:
                raise MotionVideoError('A decoded frame has no bounded presentation timestamp.') from exc
            if (index != count or not 0 <= stamp <= duration + 0.05
                    or (count == 0 and stamp != 0)
                    or (last_stamp is not None and stamp <= last_stamp)):
                raise MotionVideoError('Selected input timestamps must begin at zero and increase strictly.')
            yield count, stamp, bytes(raw), {'source_pts_integer': integer_pts,
                                           'source_time_base': source_timebase}
            count += 1
            last_stamp = stamp
        try:
            returncode = process.wait(timeout=5)
        except subprocess.TimeoutExpired as exc:
            raise MotionVideoError('Local decoder did not exit after its bounded frame stream.') from exc
        reader.join(timeout=2)
        if timed_out.is_set():
            raise MotionVideoError('Local video decoding exceeded its 120-second watchdog.')
        if errors:
            raise MotionVideoError(errors[0])
        if returncode or not ended.is_set():
            raise MotionVideoError('Local video decoding failed; no external media was fetched.')
        if count == 0 or not metadata.empty():
            raise MotionVideoError('Decoded frame and presentation timestamp counts do not match.')
    finally:
        watchdog.cancel()
        _stop_process(process)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        reader.join(timeout=2)
        process.stdout.close()
        process.stderr.close()


class _FeatureAnchor:
    """Initial-region LK translation, with explicit terminal track loss."""

    def __init__(self, cv2, np, width, height, seed_box):
        self.cv2, self.np = cv2, np
        self.width, self.height = width, height
        self.seed_box = seed_box
        self.center = np.array([(seed_box[0] + seed_box[2]) * width / 2,
                                (seed_box[1] + seed_box[3]) * height / 2], dtype='float64')
        self.previous_gray, self.points, self.previous_time = None, None, None
        self.lost_reason = None
        self.uncertainty = 1.0

    def _lost(self, reason, *, valid=0, inliers=0):
        self.lost_reason = reason
        self.points = None
        return None, 0, None, {'state': 'lost', 'reason': reason,
                              'valid_feature_count': valid, 'inlier_feature_count': inliers}

    def update(self, stamp, image):
        cv2, np = self.cv2, self.np
        if self.lost_reason:
            return self._lost(self.lost_reason)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.previous_gray is None:
            mask = np.zeros((self.height, self.width), dtype='uint8')
            x1, y1 = int(math.ceil(self.seed_box[0] * self.width)), int(math.ceil(self.seed_box[1] * self.height))
            x2, y2 = int(math.floor(self.seed_box[2] * self.width)), int(math.floor(self.seed_box[3] * self.height))
            if x2 <= x1 or y2 <= y1:
                return self._lost('seed_region_has_no_analysis_pixels')
            mask[y1:y2, x1:x2] = 255
            self.points = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01,
                                                minDistance=5, blockSize=7, mask=mask)
            if self.points is None or len(self.points) < MIN_FEATURES:
                return self._lost('insufficient_initial_features',
                                  valid=0 if self.points is None else len(self.points))
            self.previous_gray, self.previous_time = gray, stamp
            support = len(self.points)
            return self.center.tolist(), support, self.uncertainty, {
                'state': 'seeded', 'reason': None, 'valid_feature_count': support,
                'inlier_feature_count': support, 'forward_backward_median_pixels': None,
                'displacement_dispersion_pixels': None}
        if stamp - self.previous_time > MAX_TRACK_GAP_SECONDS:
            return self._lost('timestamp_gap_exceeds_tracking_envelope')
        options = {'winSize': (21, 21), 'maxLevel': 3,
                   'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)}
        forward, ok_forward, _ = cv2.calcOpticalFlowPyrLK(self.previous_gray, gray, self.points, None, **options)
        if forward is None or ok_forward is None:
            return self._lost('forward_flow_unavailable')
        backward, ok_backward, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous_gray, forward, None, **options)
        if backward is None or ok_backward is None:
            return self._lost('backward_flow_unavailable')
        old, new, back = self.points.reshape(-1, 2), forward.reshape(-1, 2), backward.reshape(-1, 2)
        fb = np.linalg.norm(old - back, axis=1)
        valid = ((ok_forward.reshape(-1) == 1) & (ok_backward.reshape(-1) == 1)
                 & np.isfinite(new).all(axis=1) & np.isfinite(back).all(axis=1)
                 & (fb <= 1.5) & (new[:, 0] >= 0) & (new[:, 0] < self.width)
                 & (new[:, 1] >= 0) & (new[:, 1] < self.height))
        valid_count = int(valid.sum())
        if valid_count < MIN_FEATURES or valid_count < 0.5 * len(old):
            return self._lost('insufficient_forward_backward_support', valid=valid_count)
        displacement = new[valid] - old[valid]
        median = np.median(displacement, axis=0)
        residual = np.linalg.norm(displacement - median, axis=1)
        median_residual = float(np.median(residual))
        inlier = residual <= max(2.0, min(4.0, 3.0 * median_residual))
        inlier_count = int(inlier.sum())
        if inlier_count < MIN_FEATURES or inlier_count < 0.7 * valid_count:
            return self._lost('ambiguous_feature_motion', valid=valid_count, inliers=inlier_count)
        shift = np.median(displacement[inlier], axis=0)
        if float(np.linalg.norm(shift)) > max(32, 0.1 * max(self.width, self.height)):
            return self._lost('feature_step_exceeds_tracking_envelope', valid=valid_count,
                              inliers=inlier_count)
        center = self.center + shift
        if not (0 <= center[0] < self.width and 0 <= center[1] < self.height):
            return self._lost('image_anchor_left_frame', valid=valid_count, inliers=inlier_count)
        fb_median = float(np.median(fb[valid][inlier]))
        dispersion = float(np.median(np.linalg.norm(displacement[inlier] - shift, axis=1)))
        self.uncertainty += fb_median + dispersion / math.sqrt(inlier_count)
        self.center = center
        self.points = new[valid][inlier].astype('float32').reshape(-1, 1, 2)
        self.previous_gray, self.previous_time = gray, stamp
        return center.tolist(), inlier_count, self.uncertainty, {
            'state': 'tracked', 'reason': None, 'valid_feature_count': valid_count,
            'inlier_feature_count': inlier_count, 'forward_backward_median_pixels': fb_median,
            'displacement_dispersion_pixels': dispersion}


def analyze_video(manifest_path, episode_id, output, *, seed_box, label='object',
                  sample_hz=30.0, max_duration_seconds=60, max_frames=1800):
    """Analyze one explicitly permitted source; preserve a receipt on attempt failure.

    Invalid arguments/manifests and an existing output are rejected before any
    output is created. Once intake begins, processing failures return a failed
    receipt with partial rows; no failed attempt is overwritten or hidden.
    """
    started = time.monotonic()
    episode_id = identity(episode_id)
    seed_box = list(box(seed_box))
    label = text(label, 'Seed label', 160)
    sample_hz = number(sample_hz, 1, 30, 'Sampling rate')
    max_duration_seconds = number(max_duration_seconds, 0.01, 60, 'Maximum duration')
    if type(max_frames) is not int or not 1 <= max_frames <= 1800:
        raise ValueError('Maximum frames must be an integer between one and 1800.')
    root, manifest = validate_manifest(manifest_path)
    sources = [row for row in manifest['episodes'] if row['id'] == episode_id]
    if len(sources) != 1:
        raise ValueError('Select exactly one episode id present in the source manifest.')
    source_row = sources[0]
    source_path = _local_file(root, source_row['path'])
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Choose a new run directory; previous motion evidence is never overwritten.')
    output.mkdir(parents=True, exist_ok=False)
    seed = {'kind': 'human_authored_image_region', 'normalized_xyxy': seed_box,
            'label': label, 'anchor_semantics': 'initial_seed_center_plus_robust_feature_translation'}
    protocol = {'kind': 'dvidia.motion-protocol.v1', 'created_at': datetime.now(timezone.utc).isoformat(),
                'algorithm': 'manual_seed_sparse_lk_translation', 'manifest_sha256': digest(manifest),
                'episode_id': episode_id, 'seed_sha256': digest(seed),
                'execution': 'local_offline_causal_replay', 'geometry': 'analysis_image_coordinates_only',
                'sample_hz_requested': sample_hz, 'max_duration_seconds': max_duration_seconds,
                'max_frames': max_frames, 'max_image_side': MAX_IMAGE_SIDE,
                'decoder_watchdog_seconds': MAX_DECODE_SECONDS,
                'ffmpeg_threads_requested': 1, 'ffmpeg_filter_threads_requested': 1,
                'opencv_threads_requested': 1, 'worker_count': 1,
                'tracking': FEATURE_RECIPE,
                'estimator': {'horizons_seconds': [0.1, 0.25, 0.5], 'max_gap_seconds': 0.15,
                              'max_prediction_age_seconds': 0.15},
                'cadence_note': 'Velocity needs two actual observations no more than 0.15 seconds apart; low source/sample cadence can prevent every forecast.',
                'processing_budget': {'seconds': 150, 'enforcement': 'cooperative check between processed frames; native tracking call is not preempted'},
                'timebase': 'FFmpeg showinfo integer input PTS multiplied by declared rational time base; zero origin required',
                'video_rotation': 'automatic rotation disabled; seed coordinates use decoded source raster',
                'hardware': {'platform': platform.system(), 'machine': platform.machine()},
                'implementation_sha256': {'motion_video.py': file_hash(Path(__file__)),
                                          'observation_pipeline.py': file_hash(Path(__file__).with_name('observation_pipeline.py'))},
                'limitations': LIMITATIONS}
    source = {key: value for key, value in source_row.items() if key != 'path'}
    rows, frames, failures, media, seed_evidence = [], [], [], None, None
    timings = {'manifest_validation_seconds': time.monotonic() - started,
               'intake_seconds': 0.0, 'dependency_setup_seconds': 0.0,
               'decode_wait_seconds': 0.0, 'tracking_estimation_seconds': 0.0,
               'seed_evidence_write_seconds': 0.0}
    stage = 'intake'
    (output / 'protocol.json').write_bytes(canonical(protocol))
    try:
        stage_started = time.monotonic()
        actual_hash = file_hash(source_path)
        if actual_hash != source_row['sha256']:
            raise MotionVideoError('Source hash differs from the frozen source manifest.')
        info = probe_video(source_path)
        if info['duration_seconds'] > max_duration_seconds or max(info['width'], info['height']) > 4096:
            raise MotionVideoError('Source exceeds the selected duration or 4096-pixel source-side limit.')
        validate_timebase(source_path, info)
        target = output / ('original' + source_path.suffix.lower())
        shutil.copyfile(source_path, target)
        if file_hash(target) != actual_hash:
            raise MotionVideoError('Original changed during local intake.')
        width, height = _dimensions(info['width'], info['height'])
        source.update(bytes=target.stat().st_size, timing=info)
        media = {'path': target.name, 'sha256': actual_hash, 'bytes': target.stat().st_size}
        protocol.update(original_image_size_pixels=[info['width'], info['height']],
                        analysis_image_size_pixels=[width, height],
                        source_to_analysis_scale_xy=[width / info['width'], height / info['height']],
                        source_sha256=actual_hash)
        timings['intake_seconds'] = time.monotonic() - stage_started
        stage = 'dependency_setup'
        stage_started = time.monotonic()
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise MotionVideoError('Install the observations extra to enable CPU optical flow.') from exc
        from .motion import MotionEstimator
        cv2.setNumThreads(1)
        protocol.update(opencv_threads_reported=cv2.getNumThreads(),
                        software={'opencv': cv2.__version__, 'numpy': np.__version__})
        motion_path = Path(__file__).with_name('motion.py')
        protocol['implementation_sha256']['motion.py'] = file_hash(motion_path)
        estimator = MotionEstimator(width, height, **protocol['estimator'])
        tracker = _FeatureAnchor(cv2, np, width, height, seed_box)
        timings['dependency_setup_seconds'] = time.monotonic() - stage_started
        (output / 'protocol.json').write_bytes(canonical(protocol))
        stage = 'decode_and_track'
        stream = _decoded_frames(target, width, height, sample_hz=sample_hz,
                                 max_frames=max_frames, duration=info['duration_seconds'])
        try:
            while True:
                decode_started = time.monotonic()
                try:
                    index, stamp, raw, clock_evidence = next(stream)
                except StopIteration:
                    timings['decode_wait_seconds'] += time.monotonic() - decode_started
                    break
                timings['decode_wait_seconds'] += time.monotonic() - decode_started
                stage_started = time.monotonic()
                image = np.frombuffer(raw, dtype='uint8').reshape(height, width, 3)
                center, support, uncertainty, tracking = tracker.update(stamp, image)
                row = estimator.update(stamp, center, support_count=support,
                                       status='tracked' if center is not None else 'lost',
                                       uncertainty_pixels=uncertainty)
                rows.append(row)
                frames.append({'processed_index': index, 'source_pts_seconds': stamp,
                               **clock_evidence, 'tracking': tracking})
                timings['tracking_estimation_seconds'] += time.monotonic() - stage_started
                if index == 0:
                    evidence_started = time.monotonic()
                    if not cv2.imwrite(str(output / 'seed.png'), image):
                        raise MotionVideoError('Could not retain the first decoded seed evidence image.')
                    seed_evidence = {'path': 'seed.png', 'sha256': file_hash(output / 'seed.png')}
                    timings['seed_evidence_write_seconds'] += time.monotonic() - evidence_started
                if time.monotonic() - started > 150:
                    raise MotionVideoError('Local motion replay exceeded its 150-second processing budget.')
        finally:
            stream.close()
    except Exception as exc:
        message = str(exc) if isinstance(exc, MotionVideoError) else f'Local {stage} failed ({type(exc).__name__}).'
        failures.append({'stage': stage, 'error': message[:300]})
    (output / 'protocol.json').write_bytes(canonical(protocol))
    (output / 'rows.json').write_bytes(canonical(rows))
    (output / 'frames.json').write_bytes(canonical(frames))
    timings['elapsed_seconds'] = time.monotonic() - started
    receipt = {'kind': 'dvidia.motion-run.v1', 'status': 'failed' if failures else 'completed',
               'episode_id': episode_id, 'task': manifest['task'], 'source': source,
               'seed': seed, 'seed_sha256': digest(seed), 'seed_evidence': seed_evidence,
               'protocol': protocol, 'protocol_sha256': digest(protocol),
               'rows': rows, 'rows_sha256': digest(rows), 'media': media,
               'frames': frames, 'frames_sha256': digest(frames),
               'timings': timings, 'errors': failures, 'limitations': LIMITATIONS,
               'frame_count': len(rows), 'tracked_frame_count': sum(row['status'] == 'tracked' for row in rows),
               'coverage': {'frame_limit_reached': len(rows) >= max_frames,
                            'last_observed_source_pts_seconds': frames[-1]['source_pts_seconds'] if frames else None,
                            'full_source_coverage_status': 'not_assessed',
                            'note': 'Only selected frames were analyzed; reaching the frame cap may leave the remainder unobserved.'},
               'accuracy_status': 'not_assessed', 'real_time_status': 'not_assessed',
               'robot_skill_qualified': False,
               'timing_scope': 'manifest validation, intake, dependency setup, streaming decode waits, tracking and evidence writes; excludes final run.json write',
               'timing_note': 'Decode and tracking overlap; decode_wait_seconds measures consumer wait, not total decoder compute.'}
    (output / 'run.json').write_bytes(canonical(receipt))
    return receipt

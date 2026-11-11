#!/usr/bin/env python3
"""Bounded scripted capture timing; no human data, fitting or hardware control.

Profiles are diagnostic sidecars outside the sealed capture. cProfile cumulative
times are nested and must not be added together. Run one worker per process.
"""
from __future__ import annotations

import argparse
import cProfile
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import pstats
import shutil
import sys
import time

import numpy as np

from dvidia_training import operator_capture as capture
from dvidia_training import godot_demonstration_bridge as bridge
from dvidia_training import piper_action_capture as files
from dvidia_training.piper_encoder_kinematics import PiperEncoderKinematics


def summary(values, deadline_seconds=None):
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return dict(count=0)
    result = dict(count=len(a), total_seconds=float(a.sum()),
                p50_ms=float(np.percentile(a, 50)*1000),
                p95_ms=float(np.percentile(a, 95)*1000),
                max_ms=float(a.max()*1000))
    if deadline_seconds is not None:
        result.update(deadline_ms=deadline_seconds*1000,
                      deadline_misses=int((a > deadline_seconds).sum()))
    return result


def peak_memory():
    try:
        import resource
    except ImportError:
        return dict(status='unavailable', bytes=None)
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(status='available', bytes=int(value if sys.platform == 'darwin' else value*1024),
                scope='Whole process high-water RSS, not incremental renderer/cache memory.')


def numerical_runtime():
    dependencies = np.show_config(mode='dicts')['Build Dependencies']
    return dict(build_dependencies={name:{key:value.get(key) for key in ('name','version','found')}
                                   for name,value in dependencies.items() if name in ('blas','lapack')},
        requested_thread_environment={name:os.environ.get(name) for name in
            ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS')},
        effective_thread_count=None,
        thread_scope='One worker process; environment requests are not measured effective BLAS thread counts.')


def profile_rows(profiler):
    if profiler is None:
        return []
    rows = []
    for (filename, line, function), (primitive, calls, own, cumulative, _) in pstats.Stats(profiler).stats.items():
        rows.append(dict(file=Path(filename).name, line=line, function=function,
                         calls=calls, primitive_calls=primitive,
                         own_seconds=own, cumulative_seconds=cumulative))
    return sorted(rows, key=lambda row: -row['cumulative_seconds'])[:100]


def run(assets, output, duration, profile):
    config = capture.camera.RolloutConfig(duration_s=duration, seed=0)
    output = Path(output).absolute()
    if output.exists() or output.is_symlink() or any(p.is_symlink() for p in output.parents):
        raise ValueError('Use a fresh ordinary output directory.')
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    upper = (round(duration/.04)+1)*(320*240*7+4096)+32*1024**2
    if shutil.disk_usage(parent).free < bridge.RESERVE_BYTES+upper:
        raise ValueError('Keep the 10 GiB reserve plus the full declared capture budget.')
    output.mkdir(parents=True, exist_ok=False)
    source = Path(capture.__file__).parent
    manifest = dict(schema='dvidia.operator-profile/v2', duration_s=duration,
        profile_enabled=profile, source_kind='simulation_scripted_input',
        source_sha256={p.name: sha256(p.read_bytes()).hexdigest() for p in [
            Path(__file__), source/'operator_capture.py', source/'piper_action_capture.py',
            source/'camera_action_capture.py', source/'piper_rgbd_tracker.py',
            source/'piper_encoder_kinematics.py']},
        case=bridge.DEFAULT_CASE, python=platform.python_version(), system=platform.platform(),
        numpy=np.__version__, numerical_runtime=numerical_runtime(),
        physics_dt_seconds=.001, control_dt_seconds=.02,
        camera_dt_seconds=.04, resolution=[320, 240],
        scope='Single-process local scripted capture including encoder-only IK; excludes sockets, '
              'Godot, real-time pacing, model fitting and hardware. cProfile cumulative scopes overlap.')
    files._json(output/'started.json', manifest)
    profiles = {name: cProfile.Profile() if profile else None for name in ('open', 'control', 'finish')}
    times, camera_times, encoder_times, ik_times = [], [], [], []
    sample_times, png_times, record_times, frame_processing_age, tracking_times = [], [], [], [], []
    latest_sample_start = None
    recorder = None
    began = time.perf_counter(); cpu_began = time.process_time()
    try:
        kinematics = PiperEncoderKinematics(assets)
        recorder = capture.OperatorCapture(assets, output/'capture', case=bridge.DEFAULT_CASE,
            duration_s=config.duration_s, source_kind='simulation_scripted_input')
        original_sample = recorder.session.sample
        def measured_sample(index):
            nonlocal latest_sample_start
            latest_sample_start = time.perf_counter()
            result = original_sample(index)
            sample_times.append(time.perf_counter()-latest_sample_start)
            return result
        recorder.session.sample = measured_sample
        # Stage wrappers only run in diagnostic profiles. Confirmation uses
        # the normal encoder and writer without additional per-file timers.
        original_png, original_record = files._png, files._record
        if profile:
            def timed_png(*args, **kwargs):
                start = time.perf_counter()
                result = original_png(*args, **kwargs)
                png_times.append(time.perf_counter()-start)
                return result
            def timed_record(*args, **kwargs):
                start = time.perf_counter()
                result = original_record(*args, **kwargs)
                record_times.append(time.perf_counter()-start)
                return result
            files._png, files._record = timed_png, timed_record
        opening = time.perf_counter()
        if profile: profiles['open'].enable()
        opened = recorder.open()
        if profile: profiles['open'].disable()
        open_seconds = time.perf_counter()-opening
        joints = opened['context']['snapshot']['signals']['joint_position_rad']['value']
        home = kinematics.fk(joints)['tcp_position_world_m']
        orientation = recorder.session.task_config.tool_orientation_wxyz
        # Same explicit path for every revision/run; camera and object truth
        # never choose the action. No teacher calls or task-completion retries.
        control_cpu_start = time.process_time()
        for index in range(round(duration/.02)):
            tick = time.perf_counter()
            if profile: profiles['control'].enable()
            context = recorder.context()
            angle = 2*math.pi*(index+1)/100
            target = [home[0]-.01*(1-math.cos(angle)), home[1]+.01*math.sin(angle), home[2]]
            ik_start = time.perf_counter()
            action = kinematics.pose_action(context['snapshot']['signals']['joint_position_rad']['value'],
                                           target, orientation, .035)
            ik_times.append(time.perf_counter()-ik_start)
            recorder.step(action, input_event=dict(kind='registered_scripted_sine', index=index,
                                                  target_world_m=target))
            if profile: profiles['control'].disable()
            elapsed = time.perf_counter()-tick
            times.append(elapsed)
            (camera_times if (index+1)%2 == 0 else encoder_times).append(elapsed)
            frame_processing_age.append(time.perf_counter()-latest_sample_start)
            if (index+1)%2 == 0:
                tracking_times.append(recorder._estimate_timing['estimator_wall_seconds'])
        control_cpu_seconds = time.process_time()-control_cpu_start
        finishing = time.perf_counter()
        if profile: profiles['finish'].enable()
        result = recorder.finish('capture_budget_complete')
        if profile: profiles['finish'].disable()
        finish_seconds = time.perf_counter()-finishing
        total_seconds = time.perf_counter()-began
        total_cpu_seconds = time.process_time()-cpu_began
        capture_peak_memory = peak_memory()
        if not result['sealed'] or result['error'] or not result['native_evidence_saved']:
            raise RuntimeError('Recorder did not seal complete native evidence.')
        inspection = capture.inspect_capture(output/'capture')
        receipt = dict(**manifest, complete=True, open_seconds=open_seconds,
            finish_seconds=finish_seconds, total_seconds=total_seconds,
            control=summary(times,.02), camera_control=summary(camera_times,.02),
            encoder_control=summary(encoder_times,.02), ik=summary(ik_times),
            control_cpu_seconds=control_cpu_seconds, total_cpu_seconds=total_cpu_seconds,
            capture_peak_memory=capture_peak_memory,
            render_encode=summary(sample_times), tracking=summary(tracking_times),
            first_camera_render_encode_ms=sample_times[0]*1000,
            steady_camera_render_encode=summary(sample_times[1:]),
            diagnostic_png_encode=summary(png_times), diagnostic_durable_write=summary(record_times),
            stage_scope='render_encode includes initial camera; tracking includes control camera updates only. '
                        'Diagnostic PNG/file-write timers include open/control/finish and overlap parent scopes. '
                        'File-write timer excludes journal append/fsync; cProfile retains that separate work.',
            offline_frame_processing_age=summary(frame_processing_age),
            frame_age_scope='Wall time from beginning of synthetic sample to step return; includes tracking '
                            'and durable recording. Offline without pacing; not physical capture or delivery age.',
            real_time_factor_control_only=duration/sum(times),
            interval_seconds=times, profiles={name:profile_rows(p) for name,p in profiles.items()},
            capture_integrity_sha256=sha256((output/'capture/integrity.json').read_bytes()).hexdigest(),
            capture_bytes=sum(p.stat().st_size for p in (output/'capture').rglob('*') if p.is_file()),
            strict_success=inspection['strict_success'], camera_pairs=result['camera_pairs'],
            recorded_intervals=result['recorded_complete_intervals'], teacher_queries=result['teacher_queries'],
            physical_robot_ready=False)
        files._json(output/'profile.json', receipt)
        print(json.dumps({k:receipt[k] for k in ('complete','profile_enabled','control',
            'camera_control','encoder_control','open_seconds','finish_seconds','total_seconds','capture_bytes')}),flush=True)
        return 0
    except BaseException as exc:
        for p in profiles.values():
            if p is not None: p.disable()
        if recorder is not None:
            try: recorder.finish('profile_interrupted')
            except BaseException: pass
        files._json(output/'failure.json', dict(schema=manifest['schema'],
            type=type(exc).__name__, message=str(exc), complete=False))
        raise
    finally:
        if 'original_png' in locals():
            files._png, files._record = original_png, original_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--duration', default=2., type=float)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    return run(args.assets, args.output, args.duration, args.profile)


if __name__ == '__main__':
    raise SystemExit(main())

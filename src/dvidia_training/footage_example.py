"""Explicitly synthetic footage and optional actual native arm telemetry.

The schematic videos are encoded media, not human demonstrations or camera
perception. They exercise the intake, learning and movement export tools.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from .footage_pipeline import _canonical, _fresh, _write

WIDTH, HEIGHT, FPS = 160, 128, 10


def _encode(path, frames):
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        raise ValueError('Install FFmpeg to create or train from encoded footage.')
    command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
               '-protocol_whitelist', 'file,pipe', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS), '-i', 'pipe:0', '-an',
               '-c:v', 'libx264', '-threads', '1', '-pix_fmt', 'yuv420p', str(path)]
    result = subprocess.run(command, input=b''.join(frame.tobytes() for frame in frames),
                            capture_output=True, timeout=60)
    if result.returncode:
        raise ValueError('FFmpeg could not encode the synthetic example.')


def _base(index):
    frame = np.full((HEIGHT, WIDTH, 3), 246, dtype=np.uint8)
    frame[102:] = [221, 220, 214]
    # Encoded visible per-recording marker, not a supervision label.
    frame[4:8, 4:4 + 4 * (index + 1)] = [160, 159, 152]
    return frame


def _disc(frame, x, y, radius, color):
    yy, xx = np.ogrid[:HEIGHT, :WIDTH]
    frame[(xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2] = color


def _line(frame, a, b, color, thickness=2):
    count = max(abs(int(a[0] - b[0])), abs(int(a[1] - b[1]))) + 1
    xs = np.rint(np.linspace(a[0], b[0], count)).astype(int)
    ys = np.rint(np.linspace(a[1], b[1], count)).astype(int)
    for dx in range(-thickness, thickness + 1):
        for dy in range(-thickness, thickness + 1):
            frame[np.clip(ys + dy, 0, HEIGHT - 1), np.clip(xs + dx, 0, WIDTH - 1)] = color


def _synthetic(index):
    frames = []
    start_x, end_x = 28 + index % 5 * 3, 121 - index % 4 * 3
    for t in range(60):
        frame = _base(index)
        alpha = t / 59
        x = start_x + (end_x - start_x) * alpha
        y = 95 - 38 * np.sin(np.pi * alpha)
        _disc(frame, end_x, 96, 11, [174, 203, 183])
        _disc(frame, x, y, 8, [72 + index * 3, 108, 170])
        frames.append(frame)
    return frames


def _native(index, root):
    from .arm_distill import FEATURE_CONTRACT, features
    from .arm_env import ArmConfig, ArmEnv
    from .arm_policy import ArmPickPlacePolicy
    config = ArmConfig(object_position=(.34 + .013 * (index % 6), -.11 + .03 * (index % 7), .31),
                       target_position=(.44 + .012 * (index % 6), .12 - .021 * (index % 7), .31),
                       scene_jitter=0)
    env = ArmEnv(config)
    observation, _ = env.reset(seed=8000 + index)
    samples, evidence, frames = [], [], []
    class Recorder:
        def __getattr__(self, name):
            return getattr(env, name)

        def joint_targets_for_pose(self, goal):
            targets = env.joint_targets_for_pose(goal)
            timestamp = round(env._elapsed, 9)
            if not samples or timestamp > samples[-1]['timestamp_seconds']:
                context, error = features(observation, goal)
                sample = {'timestamp_seconds': timestamp, 'context': context.tolist(),
                          'error': error.tolist(),
                          'delta': (np.asarray(targets) - np.asarray(observation['joint_position'])).tolist()}
                samples.append(sample)
                evidence.append({'timestamp_seconds': timestamp, 'observation': observation,
                                 'goal': np.asarray(goal).tolist(), 'joint_targets': targets})
            return targets
    policy = ArmPickPlacePolicy(Recorder())
    def project(position):
        x, y, z = position
        return (int(14 + 205 * (x + .25 * y)), int(118 - 190 * z + 12 * y))
    step = 0
    while True:
        if step % 5 == 0:
            frame = _base(index)
            points = [project(env.data.xpos[env.model.body(name).id]) for name in ('base', 'b0', 'b1', 'b2', 'b3', 'b4', 'b5')]
            points.append(project(observation['end_effector_position']))
            for a, b in zip(points, points[1:]):
                _line(frame, a, b, [109, 108, 101])
            _disc(frame, *project(observation['target_position']), 8, [166, 197, 174])
            _disc(frame, *project(observation['object_position']), 6, [65, 107, 172])
            _disc(frame, *points[-1], 3, [204, 132, 64])
            frames.append(frame)
        action = policy(observation)
        observation, _, terminated, truncated, info = env.step(action)
        step += 1
        if terminated or truncated:
            break
    path = root / 'videos' / f'example-{index:02}.mp4'
    _encode(path, frames)
    digest = sha256(path.read_bytes()).hexdigest()
    evidence_document = {'kind': 'dvidia.native-telemetry-evidence', 'schema_version': 1,
                         'scope': 'simulation-only', 'seed': 8000 + index,
                         'simulation_seconds': info['simulation_time'], 'final_info': info,
                         'samples': evidence,
                         'source_files_sha256': {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                                 for name in ('arm_env.py', 'arm_policy.py', 'arm_distill.py')}}
    evidence_path = root / 'evidence' / f'example-{index:02}.json'
    _write(evidence_path, evidence_document)
    validation = {'kind': 'dvidia.movement-alignment-validation', 'schema_version': 1,
                  'status': 'validated', 'scope': 'simulation-only', 'media_sha256': digest,
                  'samples_sha256': sha256(_canonical(samples)).hexdigest(),
                  'feature_contract': FEATURE_CONTRACT,
                  'method': 'Native simulator state and authored IK targets sampled at the same 20 ms control timestamps; schematic footage sampled at 10 Hz.',
                  'evidence_sha256': sha256(evidence_path.read_bytes()).hexdigest()}
    sidecar = {'kind': 'dvidia.aligned-movement-demonstration', 'schema_version': 1,
               'feature_contract': FEATURE_CONTRACT, 'media_sha256': digest, 'samples': samples,
               'provenance': {'source_kind': 'native-simulation-telemetry', 'validation': validation}}
    _write(root / 'actions' / f'example-{index:02}.json', sidecar)
    return info


def create_example(output, *, clips=10, native_arm=False):
    if type(clips) is not int or not 3 <= clips <= 20:
        raise ValueError('Use 3–20 independent synthetic example clips.')
    if native_arm:
        from .arm_entrypoints import require_arm
        require_arm()
    root = _fresh(output)
    (root / 'videos').mkdir()
    if native_arm:
        (root / 'actions').mkdir()
        (root / 'evidence').mkdir()
        (root / 'task.skill.json').write_bytes(Path(__file__).with_name('place_cup.skill.json').read_bytes())
    media, results = [], []
    for index in range(clips):
        if native_arm:
            result = _native(index, root)
            results.append({'id': f'example-{index:02}', 'success': result['success'],
                            'reason': result['reason'], 'simulation_seconds': result['simulation_time']})
        else:
            _encode(root / 'videos' / f'example-{index:02}.mp4', _synthetic(index))
        record = {'id': f'example-{index:02}', 'path': f'videos/example-{index:02}.mp4',
                  'source_recording_id': f'synthetic-recording-{index:02}',
                  'session_id': f'synthetic-session-{index:02}', 'complete': True}
        if native_arm:
            record['actions_path'] = f'actions/example-{index:02}.json'
        media.append(record)
    manifest = {'schema_version': 1, 'kind': 'dvidia.skillspace-training-source',
                'task': 'Synthetic native rigid-box placement' if native_arm else 'Synthetic moving-disc visual prediction',
                'synthetic': True, 'license': 'MIT', 'media': media,
                'description': 'Encoded schematic test fixture; no human footage, shoe task, physical arm or learned perception bridge.'}
    _write(root / 'skillspace.training.json', manifest)
    _write(root / 'example-receipt.json', {'schema_version': 1, 'synthetic': True,
                                          'native_arm': native_arm, 'clips': clips, 'episodes': results})
    return {'status': 'example_created', 'source': str(root), 'synthetic': True,
            'native_arm': native_arm, 'clips': clips, 'episodes': results}

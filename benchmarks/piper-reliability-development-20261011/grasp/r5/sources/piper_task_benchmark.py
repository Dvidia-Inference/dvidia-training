"""Frozen PiPER contact-task attempts and offline evidence inspection.

Inspection checks recorded declarations, controls and denominators. It neither
authenticates a publisher nor recomputes native contact dynamics.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
from html import escape
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import sys
import time

from . import piper_task as task, robot_profiles
from .reference_replay import _decode, _json_bytes, _read

SCHEMA = 'dvidia.piper-task-benchmark/v1'
PROTOCOL_SCHEMA = 'dvidia.piper-task-protocol/v1'
INTEGRITY_SCHEMA = 'dvidia.piper-task-integrity/v1'
CONTROLS = ('placement', 'replay_open_jaw', 'hold')
SOURCE_FILES = ('piper_task.py', 'piper_task_benchmark.py', 'robot_profiles.py',
                'robot_profiles.lock.json', 'reference_task.py', 'reference_sim.py', 'reference_replay.py')
EPISODE_SOURCE_FILES = ('piper_task.py', 'robot_profiles.py', 'robot_profiles.lock.json', 'reference_task.py', 'reference_sim.py')
LICENSE_PATH = 'licenses/agilex_piper/LICENSE'
MODEL_NOTE_PATH = 'licenses/agilex_piper/README.md'
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_BUNDLE_BYTES = 256 * 1024 * 1024
SCOPE = ('Authored PiPER free-block contact task, synthetic scenes and privileged native state. '
         'No learned policy, shoe competence, calibrated contact or physical robot qualification.')
TIMING_BOUNDARY = ('Attempt wall includes asset verification, compilation, settling, authored planning/IK, '
                   'physics, contact scoring and recorded state construction. Protocol/report/trace disk serialization, '
                   'rendering, training, physical sensing, transport and actuation are excluded. '
                   'Control computation windows are not sensor-to-actuator reaction times.')


def _write(path, value):
    with Path(path).open('xb') as stream:
        stream.write(_json_bytes(value))


def implementation_hashes():
    root = Path(__file__).parent
    return {name: sha256((root/name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def _pinned_assets():
    lock, item = robot_profiles._lock(), robot_profiles._profile('piper')
    metadata = {'profile': 'piper', 'repository': lock['repository'], 'commit': lock['commit'],
                'files_verified': len(item['files']), 'bytes_verified': sum(row['bytes'] for row in item['files']),
                'lock_sha256': sha256(robot_profiles.LOCK_PATH.read_bytes()).hexdigest(), 'simulation_asset_license': 'MIT'}
    notices = {row['path']: row['sha256'] for row in item['files'] if row['path'] in ('agilex_piper/LICENSE', 'agilex_piper/README.md')}
    return metadata, notices


def make_protocol(*, seed=611047, nominal=2, stress=1, cohort='development'):
    for name, value, low, high in (('seed', seed, 0, 999987), ('nominal', nominal, 1, 6), ('stress', stress, 0, 6)):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    if cohort not in ('development', 'confirmation'):
        raise ValueError('Cohort must be development or confirmation')
    base, rng, scenes = task.PiperTaskConfig(), random.Random(seed), []
    for stratum, count in (('nominal', nominal), ('low_friction', stress)):
        for index in range(count):
            start, target = list(base.object_position_m), list(base.target_position_m)
            for axis in (0, 1):
                start[axis] += rng.uniform(-.005, .005)
                target[axis] += rng.uniform(-.005, .005)
            config = replace(base, object_position_m=tuple(start), target_position_m=tuple(target),
                             friction=.15 if stratum == 'low_friction' else base.friction)
            scenes.append({'id': f'{stratum}-{index:02}', 'stratum': stratum,
                           'seed': seed+len(scenes), 'config': asdict(config)})
    return {'schema': PROTOCOL_SCHEMA, 'profile': task.PROFILE, 'contract': task.task_contract(),
            'scope': SCOPE, 'physical_robot_ready': False, 'learned_policy': False,
            'cohort': cohort, 'seed': seed, 'scenes': scenes, 'controls': list(CONTROLS),
            'planned_attempts': len(scenes)*len(CONTROLS), 'implementation_sha256': implementation_hashes(),
            'native_command_units': ['rad']*6+['m'], 'simulation_asset_license': 'MIT',
            'layout_offset_limit_m': .005, 'timing_boundary': TIMING_BOUNDARY,
            'comparison': 'Open-jaw replays the complete valid placement requested-arm tape with jaw target .035 m; '
                          'hold requests the initial six arm targets and an open jaw. Invalid or incomplete placement '
                          'tapes leave a retained invalid open-jaw attempt; they are never extended.',
            'split_note': 'Synthetic seeded scenes; repeats are not physical trials. Confirmation is a declaration, '
                          'not proof that these scenes were unseen during development.'}


def _summary(attempt):
    return {key: value for key, value in attempt.items() if key not in ('trajectory', 'actions', 'arm_tape')}


def aggregate(attempts, scenes):
    lookup = {scene['id']: scene['stratum'] for scene in scenes}
    result = []
    for stratum in dict.fromkeys(scene['stratum'] for scene in scenes):
        for control in CONTROLS:
            rows = [row for row in attempts if lookup[row['scene_id']] == stratum and row['control'] == control]
            successes = sum(row['valid'] is True and row['success'] is True for row in rows)
            result.append({'stratum': stratum, 'control': control, 'attempted': len(rows),
                           'valid': sum(row['valid'] is True for row in rows), 'success': successes,
                           'failed_or_invalid': len(rows)-successes, 'success_fraction': successes/len(rows) if rows else None})
    return result


def _peak_rss_bytes():
    if sys.platform not in ('darwin', 'linux'):
        return None
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == 'darwin' else value*1024)
    except (ImportError, OSError, ValueError):
        return None


def _html(report):
    rows = []
    for attempt in report['attempts']:
        status = 'Completed' if attempt['success'] else 'Failed' if attempt['valid'] else 'Invalid'
        rows.append('<tr>'+''.join('<td>'+escape(str(value))+'</td>' for value in
                    (attempt['scene_id'], attempt['control'], status, attempt.get('reason', 'exception'),
                     round(attempt['attempt_wall_seconds'], 3)))+
                    '<td><a href="'+escape(attempt['result_file'], quote=True)+'">Complete attempt</a></td></tr>')
    summary = ''.join('<li>'+escape(row['stratum']+' / '+row['control'])+': '+str(row['success'])+'/'+
                      str(row['attempted'])+' completed; '+str(row['valid'])+' valid.</li>' for row in report['summary'])
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>DVIDIA · PiPER contact benchmark</title><style>body{font:18px/1.6 Georgia,serif;margin:auto;max-width:1000px;'
            'padding:24px;background:#fafaf7;color:#292d27}a{color:#396751}.scroll{overflow:auto}td,th{padding:10px;text-align:left;'
            'border-bottom:1px solid #ddd}table{border-collapse:collapse;font:14px/1.5 system-ui}code{overflow-wrap:anywhere}</style>'
            '<main><p>DVIDIA RESEARCH</p><h1>PiPER: grasp, carry, release</h1><p>'+escape(SCOPE)+'</p><p>'+str(report['attempted'])+
            ' planned attempts · '+escape(report['cohort'])+' cohort. Every failed or invalid attempt stays in the denominator.</p>'
            '<p><a href="protocol.json">Frozen protocol</a> · <a href="report.json">Full report</a> · '
            '<a href="integrity.json">Integrity</a> · <a href="'+LICENSE_PATH+'">Original MIT asset license</a></p><ul>'+summary+
            '</ul><div class="scroll"><table><thead><tr><th>Scene</th><th>Control</th><th>Outcome</th><th>Reason</th><th>Wall seconds</th>'
            '<th>Evidence</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div><h2>Measurement scope</h2><p>'+escape(TIMING_BOUNDARY)+
            '</p><p>Six arm commands use radians; the one jaw command is single-finger travel in metres. '
            'A displayed block task does not establish shoe manipulation or hardware transfer.</p><p><code>'+escape(json.dumps(report['hardware']))+
            '</code></p><footer><a href="https://research.dvidia.org/">DVIDIA research</a> · '
            '<a href="https://github.com/Dvidia-Inference/dvidia-training">GitHub</a> · '
            '<a href="https://x.com/iammrriver">@iammrriver</a> · <a href="mailto:hello@dvidia.org">hello@dvidia.org</a></footer></main></html>')


def run_benchmark(assets, output, *, seed=611047, nominal=2, stress=1, cohort='development', progress=None):
    protocol = make_protocol(seed=seed, nominal=nominal, stress=stress, cohort=cohort)
    asset_root = task._asset_root(assets)
    provenance = robot_profiles.verify_assets('piper', asset_root)
    _, blobs = robot_profiles._payload('piper', asset_root)
    destination = Path(output)
    if destination.is_symlink():
        raise ValueError('Output must be a fresh regular directory')
    destination.mkdir(parents=True, exist_ok=False)
    (destination/'attempts').mkdir()
    (destination/'licenses/agilex_piper').mkdir(parents=True)
    for source, name in (('agilex_piper/LICENSE', LICENSE_PATH), ('agilex_piper/README.md', MODEL_NOTE_PATH)):
        with (destination/name).open('xb') as stream:
            stream.write(blobs[source])
    _write(destination/'protocol.json', protocol)  # Written before any native simulation.
    digest, attempts = sha256((destination/'protocol.json').read_bytes()).hexdigest(), []
    for scene in protocol['scenes']:
        tape = None
        for control in CONTROLS:
            if implementation_hashes() != protocol['implementation_sha256']:
                raise ValueError('Implementation changed after protocol freeze; preserve this incomplete run')
            began, attempt_id = time.perf_counter(), scene['id']+'--'+control
            try:
                if control == 'replay_open_jaw' and tape is None:
                    raise ValueError('Valid complete placement arm tape unavailable')
                result = task.run_episode(asset_root, task.PiperTaskConfig(**scene['config']), seed=scene['seed'],
                                          control=control, arm_tape=tape if control == 'replay_open_jaw' else None)
                if type(result.get('valid')) is not bool or type(result.get('success')) is not bool:
                    raise ValueError('Task runner omitted Boolean outcomes')
                if control == 'placement':
                    candidate = result.get('arm_tape')
                    ticks = round(scene['config']['timeout_s']/scene['config']['control_dt'])
                    tape = candidate if result['valid'] and type(candidate) is list and len(candidate) == ticks else None
            except Exception as exc:
                result = {'schema': task.SCHEMA, 'scope': task.SCOPE, 'valid': False, 'success': False, 'reason': 'exception',
                          'error': {'type': type(exc).__name__, 'message': str(exc).replace(str(asset_root.resolve()), '[assets]')[:500]},
                          'trajectory': [], 'actions': [], 'arm_tape': [], 'metrics': {}, 'timing': {}, 'warnings': [],
                          'simulated_seconds': 0., 'contract': protocol['contract'],
                          'implementation_sha256': {name: protocol['implementation_sha256'][name] for name in EPISODE_SOURCE_FILES}}
            result.update(id=attempt_id, scene_id=scene['id'], control=control, profile=task.PROFILE,
                          physical_robot_ready=False, seed=scene['seed'], config=scene['config'],
                          attempt_wall_seconds=time.perf_counter()-began)
            name = 'attempts/'+attempt_id+'.json'
            _write(destination/name, result)
            if implementation_hashes() != protocol['implementation_sha256']:
                raise ValueError('Implementation changed during an attempt; incomplete evidence retained')
            attempts.append({**_summary(result), 'result_file': name})
            if progress:
                progress({'id': attempt_id, 'valid': result['valid'], 'success': result['success']})
    runtimes = [row.get('platform') for row in attempts if type(row.get('platform')) is dict]
    report = {'schema': SCHEMA, 'profile': task.PROFILE, 'scope': SCOPE, 'physical_robot_ready': False, 'learned_policy': False,
              'cohort': cohort, 'protocol': protocol, 'protocol_sha256': digest, 'implementation_sha256': protocol['implementation_sha256'],
              'asset_provenance': provenance, 'simulation_asset_license': 'MIT',
              'asset_license': {'path': LICENSE_PATH, 'sha256': sha256(blobs['agilex_piper/LICENSE']).hexdigest()},
              'hardware': {'platform': platform.platform(), 'machine': platform.machine(), 'python': platform.python_version(),
                           'logical_cpu_count': os.cpu_count(), 'gpu_used': False, 'episode_runtimes': list({json.dumps(x, sort_keys=True): x for x in runtimes}.values()),
                           'process_lifetime_peak_rss_bytes': _peak_rss_bytes(),
                           'memory_scope': 'Process lifetime high-water RSS before final inspection/export, not per-episode memory or minimum RAM; null unavailable.'},
              'scenes': protocol['scenes'], 'attempted': len(attempts), 'valid_attempts': sum(row['valid'] for row in attempts),
              'summary': aggregate(attempts, protocol['scenes']), 'attempts': attempts}
    _write(destination/'report.json', report)
    with (destination/'report.html').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(_html(report))
    names = ['protocol.json', 'report.json', 'report.html', LICENSE_PATH, MODEL_NOTE_PATH]+[row['result_file'] for row in attempts]
    _write(destination/'integrity.json', {'schema': INTEGRITY_SCHEMA, 'files': {
        name: {'bytes': (destination/name).stat().st_size, 'sha256': sha256((destination/name).read_bytes()).hexdigest()} for name in names}})
    inspect_task_bundle(destination)
    return report


def _numeric(value, name, minimum=0., maximum=1e12):
    try:
        finite = math.isfinite(value)
    except (TypeError, OverflowError, ValueError):
        finite = False
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not finite or not minimum <= value <= maximum:
        raise ValueError('Invalid '+name)
    return value


def _vector(value, size, name):
    if type(value) is not list or len(value) != size:
        raise ValueError('Invalid '+name+' dimensions')
    for element in value:
        _numeric(element, name, -1e8, 1e8)
    return value


def _episode(raw, scene, protocol):
    if (raw.get('schema') != task.SCHEMA or raw.get('contract') != protocol['contract']
            or raw.get('implementation_sha256') != {name: protocol['implementation_sha256'][name] for name in EPISODE_SOURCE_FILES}):
        raise ValueError('Episode contract/source differs from frozen protocol')
    trajectory, actions, tape = (raw.get(name) for name in ('trajectory', 'actions', 'arm_tape'))
    config, valid = scene['config'], raw['valid']
    duration = _numeric(raw.get('simulated_seconds'), 'simulation duration', 0., config['timeout_s']+1e-8)
    count = round(config['timeout_s']/config['control_dt'])
    if any(type(rows) is not list for rows in (trajectory, actions, tape)) or len(actions) > count or len(tape) != len(actions):
        raise ValueError('Invalid bounded episode/tape lengths')
    if valid and (len(actions) != count or len(trajectory) != count+1):
        raise ValueError('Valid attempt must retain its full timeout and every boundary')
    if len(trajectory) > len(actions)+1:
        raise ValueError('Episode records more states than command boundaries')
    if trajectory and (trajectory[-1]['t_seconds'] > duration+1e-8 or duration > len(actions)*config['control_dt']+1e-8):
        raise ValueError('Recorded states and simulated duration disagree')
    limits = raw.get('native_control_limits')
    if valid or limits is not None:
        if type(limits) is not list or len(limits) != 7:
            raise ValueError('Invalid native control limits')
        for bounds in limits:
            _vector(bounds, 2, 'native control limits')
            if bounds[0] >= bounds[1]:
                raise ValueError('Invalid native control range')
        if limits[-1] != [0., .035]:
            raise ValueError('Native jaw limits disagree with metre command contract')
    for index, action in enumerate(actions):
        if type(action) is not dict or not math.isclose(_numeric(action.get('t_seconds'), 'action time'), index*config['control_dt'], abs_tol=1e-8, rel_tol=0):
            raise ValueError('Action timing disagrees with frozen cadence')
        arm = _vector(action.get('joint_targets_rad'), 6, 'radian arm targets')
        _numeric(action.get('gripper_position_m'), 'metre jaw target', 0., .035)
        if tape[index] != arm:
            raise ValueError('Arm tape differs from requested commands')
        if limits is not None and any(not bounds[0] <= value <= bounds[1] for value, bounds in zip(arm, limits)):
            raise ValueError('Requested arm command exceeds native limits')
        if raw['control'] != 'placement' and action['gripper_position_m'] != .035:
            raise ValueError('Negative control must request open jaws')
    for index, state in enumerate(trajectory):
        if type(state) is not dict:
            raise ValueError('Invalid native state record')
        clock = _numeric(state.get('t_seconds'), 'state time')
        scheduled = index*config['control_dt']
        # A warning can stop native physics partway through the last command.
        # Preserve this finite invalid terminal state without inventing a full tick.
        partial_terminal = (not valid and index > 0 and index == len(trajectory)-1 and index == len(actions)
                            and trajectory[index-1]['t_seconds'] < clock <= scheduled+1e-8)
        if not partial_terminal and not math.isclose(clock, scheduled, abs_tol=1e-8, rel_tol=0):
            raise ValueError('State timing disagrees with frozen cadence')
        qpos = _vector(state.get('qpos'), 15, 'native positions')
        _vector(state.get('qvel'), 14, 'native velocities')
        for field, size in (('joint_position_rad', 6), ('joint_velocity_rad_s', 6), ('gripper_joint_position_m', 2),
                            ('tcp_position_m', 3), ('object_position_m', 3), ('object_linear_velocity_m_s', 3),
                            ('object_angular_velocity_rad_s', 3), ('native_actuator_controls', 7), ('actuator_force_native', 7)):
            _vector(state.get(field), size, field)
        for field in ('tcp_quaternion_wxyz', 'object_quaternion_wxyz'):
            quat = _vector(state.get(field), 4, field)
            if not math.isclose(sum(x*x for x in quat), 1., abs_tol=1e-7, rel_tol=0):
                raise ValueError('Invalid native quaternion')
        if (state['joint_position_rad'] != qpos[:6] or state['gripper_joint_position_m'] != qpos[6:8]
                or state['object_position_m'] != qpos[8:11]
                or any(not math.isclose(a, b, abs_tol=1e-10, rel_tol=0) for a, b in zip(state['object_quaternion_wxyz'], qpos[11:15]))
                or state['joint_velocity_rad_s'] != state['qvel'][:6]
                or state.get('target_position_m') != config['target_position_m']
                or state.get('observation_origin') != 'privileged_native_simulator_state'):
            raise ValueError('Native state/units/provenance slices disagree')
        if index < len(actions) and (state.get('issued_joint_targets_rad') != actions[index]['joint_targets_rad']
                                    or state.get('issued_gripper_position_m') != actions[index]['gripper_position_m']
                                    or state.get('command') != actions[index]):
            raise ValueError('State and requested command alignment disagree')
        if index == len(actions) and any(state.get(name) is not None for name in
                                         ('issued_joint_targets_rad', 'issued_gripper_position_m', 'command')):
            raise ValueError('Recorded terminal state must have no next issued command')
        for field in ('grasp_contact_counts', 'pad_normal_forces_n'):
            if type(state.get(field)) is not dict or set(state[field]) != {'left', 'right'}:
                raise ValueError('Invalid native bilateral contact fields')
        for side in ('left', 'right'):
            value = state['grasp_contact_counts'][side]
            if type(value) is not int or not 0 <= value <= 100000:
                raise ValueError('Invalid native contact count')
            _numeric(state['pad_normal_forces_n'][side], 'pad normal force')
        for field in ('contact_count', 'table_object_contact_count', 'unexpected_contact_count'):
            if type(state.get(field)) is not int or not 0 <= state[field] <= 100000:
                raise ValueError('Invalid native contact count')
        pairs = state.get('active_contact_pairs')
        if type(pairs) is not list or len(pairs) > state['contact_count']:
            raise ValueError('Invalid native contact pair list')
        for pair in pairs:
            ids = pair.get('geom_ids') if type(pair) is dict else None
            if (type(ids) is not list or len(ids) != 2 or any(type(i) is not int or not 0 <= i <= 100000 for i in ids)
                    or ids[0] >= ids[1]):
                raise ValueError('Invalid native contact identities')
            _numeric(pair.get('normal_force_n'), 'active normal contact force', .01, 1e12)
    if valid:
        if not math.isclose(duration, config['timeout_s'], abs_tol=1e-8, rel_tol=0) or raw.get('warnings') != []:
            raise ValueError('Valid attempt duration/warnings disagree')
        if type(raw.get('compiled_model_sha256')) is not str or re.fullmatch(r'[0-9a-f]{64}', raw['compiled_model_sha256']) is None:
            raise ValueError('Invalid compiled model identity')
        settings, runtime, timing = (raw.get(name) for name in ('native_model_settings', 'platform', 'timing'))
        if (type(settings) is not dict or settings.get('dimensions') != {'nq':15,'nv':14,'nu':7,'neq':1,'nmocap':0}
                or settings.get('gravity_m_s2') != [0., 0., -9.81] or settings.get('body_gravcomp') != [0.]+[1.]*9+[0.,0.]):
            raise ValueError('Native model mechanics/gravity contract mismatch')
        if type(runtime) is not dict or any(type(runtime.get(key)) is not str or not runtime[key] for key in
                                             ('system','machine','python','mujoco_version','numpy_version')):
            raise ValueError('Missing actual native runtime')
        if type(timing) is not dict or type(timing.get('control_boundaries')) is not int or timing['control_boundaries'] != count:
            raise ValueError('Native timing boundary count mismatch')
        distribution = timing.get('control_step_seconds')
        if type(distribution) is not dict or type(distribution.get('count')) is not int or distribution['count'] != count:
            raise ValueError('Native timing distribution count mismatch')
        median, p95, worst = [_numeric(distribution.get(key), 'control computation seconds') for key in ('median','p95','max')]
        wall = _numeric(timing.get('episode_wall_seconds'), 'episode wall seconds', 1e-12)
        if (not median <= p95 <= worst or type(timing.get('missed_control_boundaries')) is not int
                or not 0 <= timing['missed_control_boundaries'] <= count
                or _numeric(timing.get('setup_seconds'), 'setup seconds') > wall
                or not math.isclose(_numeric(timing.get('simulated_seconds_per_wall_second'), 'throughput ratio'), duration/wall, rel_tol=1e-9, abs_tol=1e-9)):
            raise ValueError('Native computation timing/throughput mismatch')
        metrics = raw.get('metrics')
        if type(metrics) is not dict or metrics.get('valid') is not True or metrics.get('success') is not raw['success'] or metrics.get('physical_robot_ready') is not False:
            raise ValueError('Native outcome metrics disagree')
        if raw['success'] and (any(metrics.get(name) is not True for name in ('lift_seen', 'grasp_seen', 'carry_complete'))
                               or metrics.get('retention_lost') is not False or metrics.get('unexpected_contact_native_ticks') != 0
                               or _numeric(metrics.get('dwell_seconds'), 'release dwell')+1e-12 < config['goal_dwell_s']):
            raise ValueError('Success lacks declared native lift/carry/release evidence')
        if raw['success']:
            end = trajectory[-1]
            if (math.dist(end['object_position_m'], config['target_position_m']) > config['goal_position_tolerance_m']
                    or end['gripper_joint_position_m'][0] <= .028 or end['gripper_joint_position_m'][1] >= -.028
                    or any(end['grasp_contact_counts'].values()) or end['table_object_contact_count'] < 1
                    or end['unexpected_contact_count'] != 0 or math.hypot(*end['object_linear_velocity_m_s']) >= .035
                    or math.hypot(*end['object_angular_velocity_rad_s']) >= .3
                    or end['tcp_position_m'][2] < end['object_position_m'][2]+config['retreat_clearance_m']):
                raise ValueError('Success disagrees with recorded native release endpoint')
    _numeric(raw.get('attempt_wall_seconds'), 'attempt wall seconds')


def load_task_bundle(directory):
    """Load exact bounded artifacts; inspect consistency, not authenticity or physics."""
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Bundle must be a regular non-symlink directory')
    integrity = _read(root/'integrity.json', 65536)
    manifest = _decode(integrity)
    if type(manifest) is not dict or set(manifest) != {'schema', 'files'} or manifest['schema'] != INTEGRITY_SCHEMA:
        raise ValueError('Invalid PiPER integrity schema')
    declarations = manifest['files']
    if type(declarations) is not dict or not 8 <= len(declarations) <= 41:
        raise ValueError('Invalid bundle file count')
    static = {'protocol.json', 'report.json', 'report.html', LICENSE_PATH, MODEL_NOTE_PATH}
    files, total = {}, 0
    for name, record in declarations.items():
        if name not in static and (type(name) is not str or re.fullmatch(r'attempts/[a-z0-9_-]{1,100}\.json', name) is None):
            raise ValueError('Artifact path outside allowlist')
        path = root/name
        if any(parent.is_symlink() for parent in path.parents if parent != root and root in parent.parents):
            raise ValueError('Artifact parent must not be a symlink')
        data = _read(path, MAX_FILE_BYTES)
        total += len(data)
        if total > MAX_BUNDLE_BYTES:
            raise ValueError('Bundle exceeds byte bound')
        if (type(record) is not dict or set(record) != {'bytes', 'sha256'} or type(record['bytes']) is not int
                or record['bytes'] != len(data) or record['sha256'] != sha256(data).hexdigest()):
            raise ValueError('Artifact digest/size mismatch')
        files[name] = data
    paths = list(root.rglob('*'))
    if any(path.is_symlink() for path in paths) or {path.relative_to(root).as_posix() for path in paths if path.is_file()} != set(files)|{'integrity.json'}:
        raise ValueError('Bundle inventory contains unsealed or symlink artifacts')
    if not static <= files.keys():
        raise ValueError('Missing protocol, report or original license')
    report, protocol = _decode(files['report.json']), _decode(files['protocol.json'])
    if type(report) is not dict or report.get('schema') != SCHEMA or type(protocol) is not dict or protocol.get('schema') != PROTOCOL_SCHEMA:
        raise ValueError('Invalid report/protocol schema')
    for document in (report, protocol):
        if document.get('profile') != task.PROFILE or document.get('physical_robot_ready') is not False or document.get('learned_policy') is not False:
            raise ValueError('PiPER profile or qualification scope mismatch')
    hashes = protocol.get('implementation_sha256')
    if type(hashes) is not dict or set(hashes) != set(SOURCE_FILES) or any(type(x) is not str or re.fullmatch(r'[0-9a-f]{64}', x) is None for x in hashes.values()):
        raise ValueError('Malformed source hashes')
    if (report.get('implementation_sha256') != hashes or report.get('protocol_sha256') != sha256(files['protocol.json']).hexdigest()
            or report.get('protocol') != protocol or report.get('scenes') != protocol.get('scenes')
            or report.get('cohort') not in ('development', 'confirmation') or report['cohort'] != protocol.get('cohort')):
        raise ValueError('Frozen protocol/source/cohort mismatch')
    if (protocol.get('native_command_units') != ['rad']*6+['m'] or type(protocol.get('contract')) is not dict
            or protocol['contract'].get('profile') != task.PROFILE or protocol['contract'].get('physical_robot_ready') is not False
            or protocol.get('simulation_asset_license') != 'MIT'):
        raise ValueError('PiPER control units/contract mismatch')
    provenance = report.get('asset_provenance')
    pinned, notices = _pinned_assets()
    if (provenance != pinned or report.get('simulation_asset_license') != 'MIT'
            or provenance.get('lock_sha256') != hashes['robot_profiles.lock.json']
            or report.get('asset_license') != {'path': LICENSE_PATH, 'sha256': sha256(files[LICENSE_PATH]).hexdigest()}
            or sha256(files[LICENSE_PATH]).hexdigest() != notices['agilex_piper/LICENSE']
            or sha256(files[MODEL_NOTE_PATH]).hexdigest() != notices['agilex_piper/README.md']):
        raise ValueError('Original asset provenance/license mismatch')
    scenes = report['scenes']
    if type(scenes) is not list or not 1 <= len(scenes) <= 12:
        raise ValueError('Invalid scene count')
    lookup = {}
    for scene in scenes:
        if (type(scene) is not dict or type(scene.get('id')) is not str or re.fullmatch(r'[a-z0-9_-]{1,60}', scene['id']) is None
                or scene['id'] in lookup or scene.get('stratum') not in ('nominal', 'low_friction')
                or type(scene.get('seed')) is not int or not 0 <= scene['seed'] <= 999999 or type(scene.get('config')) is not dict):
            raise ValueError('Invalid or duplicate scene')
        task.PiperTaskConfig(**scene['config'])
        lookup[scene['id']] = scene
    expected = {(scene, control) for scene in lookup for control in CONTROLS}
    if protocol.get('controls') != list(CONTROLS) or type(protocol.get('planned_attempts')) is not int or protocol['planned_attempts'] != len(expected):
        raise ValueError('Frozen control denominator mismatch')
    attempts, seen, raw_attempts = report.get('attempts'), set(), {}
    if type(attempts) is not list or len(attempts) != len(expected) or type(report.get('attempted')) is not int or report['attempted'] != len(expected):
        raise ValueError('Missing planned attempts')
    for row in attempts:
        if type(row) is not dict or type(row.get('scene_id')) is not str or type(row.get('control')) is not str:
            raise ValueError('Invalid attempt identity')
        identity = row['scene_id'], row['control']
        if identity not in expected or identity in seen or type(row.get('valid')) is not bool or type(row.get('success')) is not bool or row['success'] and not row['valid']:
            raise ValueError('Duplicate or inconsistent attempt')
        scene, name = lookup[row['scene_id']], row.get('result_file')
        if (row.get('id') != row['scene_id']+'--'+row['control'] or row.get('profile') != task.PROFILE
                or row.get('physical_robot_ready') is not False or row.get('config') != scene['config']
                or type(row.get('seed')) is not int or row['seed'] != scene['seed']
                or name != 'attempts/'+row['id']+'.json' or name not in files):
            raise ValueError('Attempt disagrees with declared scene/profile')
        raw = _decode(files[name])
        if type(raw) is not dict or _summary(raw) != {key: value for key, value in row.items() if key != 'result_file'}:
            raise ValueError('Attempt summary differs from sealed trace')
        if raw['valid'] and raw.get('asset_provenance') != provenance:
            raise ValueError('Episode asset identity mismatch')
        _episode(raw, scene, protocol)
        seen.add(identity)
        raw_attempts[identity] = raw
    for scene in scenes:
        placement, opened, held = [raw_attempts[scene['id'], control] for control in CONTROLS]
        models = [{name: raw.get(name) for name in ('compiled_model_sha256','native_model_settings','native_control_limits',
                                                   'native_servo_gains','native_servo_bias','native_force_limits','contact_geometry')}
                  for raw in (placement, opened, held) if raw['valid']]
        if any(model != models[0] for model in models[1:]):
            raise ValueError('Matched controls use different native model mechanics')
        if opened['actions'] and (not placement['valid'] or opened['arm_tape'] != placement['arm_tape'][:len(opened['arm_tape'])]):
            raise ValueError('Open-jaw control differs from valid placement arm tape')
        if held['actions'] and (not held['trajectory'] or any(command != held['trajectory'][0]['joint_position_rad'] for command in held['arm_tape'])):
            raise ValueError('Hold control moves its initial arm targets')
    if set(files) != static|{row['result_file'] for row in attempts} or report.get('summary') != aggregate(attempts, scenes):
        raise ValueError('Inventory or aggregate denominator mismatch')
    if type(report.get('valid_attempts')) is not int or report['valid_attempts'] != sum(row['valid'] for row in attempts):
        raise ValueError('Valid-attempt denominator mismatch')
    files['integrity.json'] = integrity
    return report, files


def inspect_task_bundle(directory):
    report, files = load_task_bundle(directory)
    return {'schema': 'dvidia.piper-task-inspection/v1', 'scope': 'Artifact/control consistency only; no authenticity or physics recomputation',
            'files_verified': len(files)-1, 'attempted': report['attempted'], 'valid_attempts': report['valid_attempts'],
            'summary': report['summary'], 'physical_robot_ready': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description='DVIDIA PiPER CPU contact benchmark; simulation only')
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('benchmark')
    run.add_argument('assets', type=Path)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--seed', type=int, default=611047)
    run.add_argument('--nominal', type=int, default=2)
    run.add_argument('--stress', type=int, default=1)
    run.add_argument('--cohort', choices=('development', 'confirmation'), default='development')
    inspect = commands.add_parser('inspect')
    inspect.add_argument('bundle', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'benchmark':
            report = run_benchmark(args.assets, args.output, seed=args.seed, nominal=args.nominal, stress=args.stress,
                                   cohort=args.cohort, progress=lambda row: print(json.dumps(row), file=sys.stderr, flush=True))
            result = {'output': str(args.output), 'attempted': report['attempted'], 'summary': report['summary']}
        else:
            result = inspect_task_bundle(args.bundle)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, RuntimeError, TypeError) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()

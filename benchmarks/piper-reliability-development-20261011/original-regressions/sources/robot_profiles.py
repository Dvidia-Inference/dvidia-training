"""Pinned PiPER/SO101 CPU mechanics smoke tests, without hardware control.

Fetch is the only network operation. Audit, native stepping and inspection are
offline. Simulation asset licenses do not establish the openness of hardware.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import html
import json
import math
from pathlib import Path, PurePosixPath
import platform
import time
from urllib.request import Request, urlopen

from . import reference_sim as sim


LOCK_PATH = Path(__file__).with_name('robot_profiles.lock.json')
SCHEMA = 'dvidia.robot-profile-benchmark/v1'
INTEGRITY_SCHEMA = 'dvidia.robot-profile-integrity/v1'
CASES = ('hold', 'arm_sweep', 'gripper_cycle')
SCOPE = ('Unloaded native CPU mechanics smoke: authored position servos, upstream gravity settings, '
         'privileged joint/link state and synthetic commands. No object task, learned skill, measured '
         'hardware response, contact calibration or physical robot qualification.')


def _lock():
    return json.loads(LOCK_PATH.read_text(encoding='utf-8'))


def _profile(name):
    for item in _lock()['profiles']:
        if item['id'] == name:
            return item
    raise ValueError('Profile must be piper or so101')


def list_profiles():
    """Return explicit axes, units and asset-license claims without native imports."""
    return [{key: value for key, value in item.items() if key != 'files'} for item in _lock()['profiles']]


def _payload(profile, directory):
    item, root = _profile(profile), Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Assets require an existing non-symlink directory')
    blobs = {}
    for record in item['files']:
        path = root / record['path']
        if any(parent.is_symlink() for parent in path.parents if parent != root and root in parent.parents):
            raise ValueError('Asset subdirectories must not be symlinks')
        content = sim._regular_bytes(path, record['bytes'])
        if sha256(content).hexdigest() != record['sha256']:
            raise ValueError('Asset digest mismatch: '+record['path'])
        blobs[record['path']] = content
    return item, blobs


def verify_assets(profile, directory):
    item, blobs = _payload(profile, directory)
    return {'profile': item['id'], 'repository': _lock()['repository'], 'commit': _lock()['commit'],
            'files_verified': len(blobs), 'bytes_verified': sum(map(len, blobs.values())),
            'lock_sha256': sha256(LOCK_PATH.read_bytes()).hexdigest(),
            'simulation_asset_license': item['simulation_asset_license']}


def fetch_assets(profile, output):
    """Explicitly fetch the fixed packaged allowlist into a fresh asset root."""
    item, lock = _profile(profile), _lock()
    destination = Path(output)
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise ValueError('Asset root must be a plain directory')
    destination.mkdir(parents=True, exist_ok=True)
    (destination/item['folder']).mkdir(exist_ok=False)
    for record in item['files']:
        url = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{lock['commit']}/{record['path']}"
        with urlopen(Request(url, headers={'User-Agent': 'DVIDIA-Robot-Sim/0.1'}), timeout=30) as response:
            if response.geturl() != url:
                raise ValueError('Pinned asset download redirected')
            content = response.read(record['bytes']+1)
        if len(content) != record['bytes'] or sha256(content).hexdigest() != record['sha256']:
            raise ValueError('Downloaded asset differs from packaged lock: '+record['path'])
        path = destination / record['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(content)
    return verify_assets(profile, destination)


def build_model(profile, directory, *, timestep=.001):
    dt = sim._timestep(timestep)
    item, blobs = _payload(profile, directory)
    model_spec = sim._spec(blobs, item['folder'], item['model'])
    model_spec.option.timestep = dt
    model = model_spec.compile()
    mj, _ = sim._native_imports()
    actual = {name: int(getattr(model, name)) for name in ('nq', 'nv', 'nu', 'neq')}
    if actual != item['expected_dimensions']:
        raise ValueError('Native profile mechanical dimensions differ from the pinned contract')
    if [mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)] != item['actuator_names']:
        raise ValueError('Native profile actuator order differs from the pinned contract')
    expected_joints = item['arm_joint_names']+item['gripper_joint_names']
    if [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)] != expected_joints:
        raise ValueError('Native profile joint order differs from the pinned contract')
    if mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, item['tool_frame_body']) <= 0:
        raise ValueError('Native profile terminal link body is missing')
    for index, unit in enumerate(item['native_command_units']):
        joint = int(model.actuator_trnid[index, 0])
        expected = mj.mjtJoint.mjJNT_HINGE if unit == 'rad' else mj.mjtJoint.mjJNT_SLIDE
        if model.jnt_type[joint] != expected or not model.actuator_ctrllimited[index]:
            raise ValueError('Native command units/limits differ from the pinned contract')
    return model


def audit(profile, directory, *, timestep=.001):
    metadata = verify_assets(profile, directory)
    messages = []
    with sim.native_warnings(messages):
        model = build_model(profile, directory, timestep=timestep)
    if messages:
        raise ValueError('Native profile compilation emitted warnings: '+str(messages))
    mj, _ = sim._native_imports()
    return {'schema': 'dvidia.robot-profile-audit/v1', 'profile': list_profiles()[0 if profile == 'piper' else 1],
            'assets': metadata, 'native_version': mj.__version__, 'compiled_model_sha256': sim.model_digest(model),
            'dimensions': {name: int(getattr(model, name)) for name in ('nq', 'nv', 'nu', 'neq', 'ngeom', 'nbody')},
            'actuator_ctrlrange_native': model.actuator_ctrlrange.tolist(),
            'actuator_forcerange_native': model.actuator_forcerange.tolist(),
            'gravity_compensated_bodies': int(sum(model.body_gravcomp != 0)),
            'scope': SCOPE, 'task_qualified': False, 'physical_robot_ready': False}


def _bounded(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name+' must be finite numeric data')
    try:
        number = float(value)
    except (OverflowError, ValueError, TypeError) as exc:
        raise ValueError(name+' must be finite numeric data') from exc
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(name+' outside bounded range')
    return number


def _duration(value):
    number = _bounded(value, 'duration', .1, 2.)
    if not math.isclose(number/sim.CONTROL_DT, round(number/sim.CONTROL_DT), abs_tol=1e-9, rel_tol=0):
        raise ValueError('Duration must contain complete 20 ms control boundaries')
    return number


def _sample(model, data, item, elapsed):
    mj, np = sim._native_imports()
    if not all(np.isfinite(value).all() for value in (data.qpos, data.qvel, data.qacc, data.xpos)):
        raise RuntimeError('Nonfinite native profile state')
    body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, item['tool_frame_body'])
    return {'t_seconds': elapsed, 'qpos_native': data.qpos.tolist(), 'qvel_native_per_second': data.qvel.tolist(),
            'arm_joint_position_rad': data.qpos[:item['arm_axes']].tolist(),
            'arm_joint_velocity_rad_s': data.qvel[:item['arm_axes']].tolist(),
            'gripper_joint_position_native': data.qpos[item['arm_axes']:].tolist(),
            'terminal_link_position_m': data.xpos[body].tolist(), 'terminal_link_quaternion_wxyz': data.xquat[body].tolist(),
            'actuator_force_native': data.actuator_force.tolist(), 'contact_count': int(data.ncon),
            'issued_controls_native': None, 'observation_origin': 'privileged_native_simulator_state'}


def _attempt(profile, assets, case, repeat, duration, timestep):
    started, latencies = time.perf_counter(), []
    item = _profile(profile)
    result = {'id': f'{case}-{repeat}', 'case': case, 'repeat': repeat, 'valid': False,
              'error': None, 'warnings': [], 'simulated_seconds': 0., 'trajectory': [], 'timing': {}, 'metrics': {}}
    try:
        with sim.native_warnings(result['warnings']):
            model = build_model(profile, assets, timestep=timestep)
            mj, np = sim._native_imports()
            data = mj.MjData(model)
            if model.nkey:
                mj.mj_resetDataKeyframe(model, data, 0)
            else:
                data.ctrl[:] = data.qpos[model.jnt_qposadr[model.actuator_trnid[:, 0]]]
            base = data.ctrl.copy()
            mj.mj_forward(model, data)
            result['compiled_model_sha256'] = sim.model_digest(model)
            result['runtime'] = {'mujoco_version': mj.__version__, 'numpy_version': np.__version__}
            result['native_model_settings'] = {'gravity_m_s2': model.opt.gravity.tolist(),
                                               'gravity_compensated_bodies': [
                                                   {'name': mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i),
                                                    'fraction': float(model.body_gravcomp[i])}
                                                   for i in range(model.nbody) if model.body_gravcomp[i] != 0]}
            result['trajectory'].append(_sample(model, data, item, 0.))
            for index in range(round(duration/sim.CONTROL_DT)):
                tick_started = time.perf_counter()
                commands = base.copy()
                phase = math.sin(math.pi*(index+1)*sim.CONTROL_DT/duration)
                if case == 'arm_sweep':
                    commands[:item['arm_axes']] += .08*phase
                elif case == 'gripper_cycle':
                    commands[-1] += (.012 if item['id'] == 'piper' else .15)*phase
                commands = np.clip(commands, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
                result['trajectory'][-1]['issued_controls_native'] = commands.tolist()
                data.ctrl[:] = commands
                for _ in range(round(sim.CONTROL_DT/timestep)):
                    mj.mj_step(model, data)
                    mj.mj_forward(model, data)
                    _sample(model, data, item, 0.)
                    if result['warnings'] or any(warning.number for warning in data.warning):
                        raise RuntimeError('Native profile emitted a solver warning')
                result['simulated_seconds'] = (index+1)*sim.CONTROL_DT
                result['trajectory'].append(_sample(model, data, item, result['simulated_seconds']))
                latencies.append(time.perf_counter()-tick_started)
            result['valid'] = not result['warnings']
            errors = [abs(row['qpos_native'][int(model.jnt_qposadr[model.actuator_trnid[j, 0]])]-row['issued_controls_native'][j])
                      for row in result['trajectory'][:-1] for j in range(item['arm_axes'])]
            result['metrics'] = {'max_before_step_arm_target_error_rad': max(errors, default=0.),
                                 'max_recorded_contact_count': max(row['contact_count'] for row in result['trajectory']),
                                 'scope': 'Numerical state/command diagnostics only; no task success metric'}
    except Exception as exc:
        result['error'] = {'type': type(exc).__name__, 'message': str(exc)[:500]}
    elapsed = time.perf_counter()-started
    result['timing'] = {'episode_wall_seconds': elapsed, 'control_step_seconds': sim._summary(latencies) if latencies else None,
                        'control_boundaries': len(latencies), 'missed_control_boundaries': sum(x > sim.CONTROL_DT for x in latencies),
                        'simulated_seconds_per_wall_second': result['simulated_seconds']/elapsed,
                        'scope': 'Wall computation including compilation/state snapshots; not real-time sensor-to-actuator latency'}
    return result


def _json_bytes(value):
    return (json.dumps(value, allow_nan=False, indent=2)+'\n').encode()


def _record(path, content):
    return {'path': path, 'bytes': len(content), 'sha256': sha256(content).hexdigest()}


def benchmark(profile, assets, output, *, duration=.4, timestep=.001, repeats=1):
    """Seal every bounded unloaded attempt, including numerical/setup failures."""
    item, duration, timestep = _profile(profile), _duration(duration), sim._timestep(timestep)
    if type(repeats) is not int or not 1 <= repeats <= 3:
        raise ValueError('repeats must be an integer in [1,3]')
    metadata, blobs = verify_assets(profile, assets), _payload(profile, assets)[1]
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    report = {'schema': SCHEMA, 'profile': {key: value for key, value in item.items() if key != 'files'},
              'scope': SCOPE, 'task_qualified': False, 'physical_robot_ready': False, 'assets': metadata,
              'source': {'implementation_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
                         'reference_sim_sha256': sha256(Path(sim.__file__).read_bytes()).hexdigest(),
                         'lock_sha256': sha256(LOCK_PATH.read_bytes()).hexdigest(), 'asset_commit': _lock()['commit']},
              'settings': {'duration_seconds': duration, 'timestep_seconds': timestep,
                           'control_dt_seconds': sim.CONTROL_DT, 'repeats': repeats, 'cases': list(CASES)},
              'platform': {'system': platform.system(), 'machine': platform.machine(), 'python': platform.python_version()},
              'attempts': []}
    for case in CASES:
        for repeat in range(repeats):
            attempt = _attempt(profile, assets, case, repeat, duration, timestep)
            report['attempts'].append(attempt)
            path = destination/f'attempt-{attempt["id"]}.json'
            path.write_bytes(_json_bytes(attempt))
    report['summary'] = {'attempt_count': len(report['attempts']),
                         'valid_attempts': sum(bool(x['valid']) for x in report['attempts'])}
    compiled = next((attempt for attempt in report['attempts'] if 'runtime' in attempt), None)
    report['platform'].update(mujoco_version=compiled['runtime']['mujoco_version'] if compiled else None,
                              numpy_version=compiled['runtime']['numpy_version'] if compiled else None)
    report['native_model_settings'] = compiled['native_model_settings'] if compiled else None
    rows = ''.join(f'<tr><td>{html.escape(x["id"])}</td><td>{x["valid"]}</td><td>{x["simulated_seconds"]:.2f}</td>'
                   f'<td>{html.escape(json.dumps(x["error"]))}</td></tr>' for x in report['attempts'])
    page = (f'<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            f'<title>DVIDIA {html.escape(item["name"])} CPU mechanics</title><style>body{{max-width:900px;margin:40px auto;padding:20px;font:17px Georgia;line-height:1.6;background:#faf9f5}}td,th{{padding:8px;border-bottom:1px solid #ccc}}</style>'
            f'<h1>DVIDIA · {html.escape(item["name"])} CPU mechanics</h1><p>{html.escape(SCOPE)}</p>'
            f'<p>{item["arm_axes"]} arm axes + {item["gripper_axes"]} gripper command. Simulation asset license: {html.escape(item["simulation_asset_license"])}.</p>'
            f'<p>Native command units: {html.escape(", ".join(item["native_command_units"]))}. {html.escape(item["gripper_command"])}</p>'
            f'<p><a href="report.json">Complete sealed numerical receipt</a> · <a href="licenses/{profile}/LICENSE">Original asset license</a></p>'
            f'<table><tr><th>Attempt</th><th>Valid physics</th><th>Simulated seconds</th><th>Error</th></tr>{rows}</table>').encode()
    files = {'report.json': _json_bytes(report), 'report.html': page,
             f'licenses/{profile}/LICENSE': blobs[item['folder']+'/LICENSE']}
    for name, content in files.items():
        path = destination/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    for path in sorted(destination.glob('attempt-*.json')):
        files[path.name] = path.read_bytes()
    integrity = {'schema': INTEGRITY_SCHEMA, 'files': [_record(name, content) for name, content in sorted(files.items())]}
    (destination/'integrity.json').write_bytes(_json_bytes(integrity))
    return {'output': str(destination), 'profile': profile, **report['summary'], 'report_sha256': sha256(files['report.json']).hexdigest()}


def _bounded_file(path, limit):
    size = path.lstat().st_size
    if size > limit:
        raise ValueError('Benchmark file exceeds the bounded size limit')
    return sim._regular_bytes(path, size)


def inspect_benchmark(directory):
    """Verify the standalone numerical receipt without loading native physics."""
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Benchmark root must be a non-symlink directory')
    integrity = json.loads(_bounded_file(root/'integrity.json', 100000))
    if integrity.get('schema') != INTEGRITY_SCHEMA or type(integrity.get('files')) is not list or not 3 <= len(integrity['files']) <= 12:
        raise ValueError('Unsupported bounded profile integrity manifest')
    payload, total = {}, 0
    for record in integrity['files']:
        name = record.get('path')
        if type(name) is not str or name in payload or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts or '\\' in name:
            raise ValueError('Unsafe or duplicate benchmark file path')
        path = root/name
        if any(parent.is_symlink() for parent in path.parents if parent != root and root in parent.parents):
            raise ValueError('Benchmark subdirectories must not be symlinks')
        content = _bounded_file(path, 5000000)
        total += len(content)
        if total > 12000000 or len(content) != record.get('bytes') or sha256(content).hexdigest() != record.get('sha256'):
            raise ValueError('Benchmark integrity mismatch')
        payload[name] = content
    report = json.loads(payload['report.json'])
    if report.get('schema') != SCHEMA or report.get('task_qualified') is not False or report.get('physical_robot_ready') is not False:
        raise ValueError('Unsupported profile receipt or qualification claim')
    profile = _profile(report['profile']['id'])
    expected_profile = {key: value for key, value in profile.items() if key != 'files'}
    if report['profile'] != expected_profile:
        raise ValueError('Profile receipt axes/units/provenance contract mismatch')
    lock = _lock()
    lock_digest = sha256(LOCK_PATH.read_bytes()).hexdigest()
    expected_assets = {'profile': profile['id'], 'repository': lock['repository'], 'commit': lock['commit'],
                       'files_verified': len(profile['files']), 'bytes_verified': sum(x['bytes'] for x in profile['files']),
                       'lock_sha256': lock_digest, 'simulation_asset_license': profile['simulation_asset_license']}
    if report.get('assets') != expected_assets or report['source'].get('asset_commit') != lock['commit'] or report['source'].get('lock_sha256') != lock_digest:
        raise ValueError('Profile asset/source lock provenance mismatch')
    for key in ('implementation_sha256', 'reference_sim_sha256'):
        digest = report['source'].get(key)
        if type(digest) is not str or len(digest) != 64 or any(x not in '0123456789abcdef' for x in digest):
            raise ValueError('Profile implementation source digest is malformed')
    settings = report['settings']
    duration, timestep = _duration(settings['duration_seconds']), sim._timestep(settings['timestep_seconds'])
    repeats = settings['repeats']
    if type(repeats) is not int or not 1 <= repeats <= 3 or settings['cases'] != list(CASES) or settings['control_dt_seconds'] != sim.CONTROL_DT:
        raise ValueError('Profile benchmark settings mismatch')
    attempts = report['attempts']
    if type(attempts) is not list or len(attempts) != len(CASES)*repeats:
        raise ValueError('Profile benchmark attempt count mismatch')
    expected_files = {'report.json', 'report.html', f'licenses/{profile["id"]}/LICENSE'} | {
        f'attempt-{case}-{repeat}.json' for case in CASES for repeat in range(repeats)}
    actual_files = {path.relative_to(root).as_posix() for path in root.rglob('*') if not path.is_dir()}
    if set(payload) != expected_files or actual_files != expected_files | {'integrity.json'}:
        raise ValueError('Profile benchmark file inventory mismatch')
    for index, attempt in enumerate(attempts):
        case, repeat = CASES[index//repeats], index%repeats
        if attempt['id'] != f'{case}-{repeat}' or attempt['case'] != case or attempt['repeat'] != repeat or type(attempt['valid']) is not bool:
            raise ValueError('Profile attempt ordering/validity mismatch')
        if json.loads(payload[f'attempt-{attempt["id"]}.json']) != attempt:
            raise ValueError('Profile raw attempt and report differ')
        if attempt['valid'] and (attempt['error'] is not None or attempt['warnings'] or len(attempt['trajectory']) != round(duration/sim.CONTROL_DT)+1):
            raise ValueError('Valid profile attempt has incomplete or invalid native evidence')
        for row_index, row in enumerate(attempt['trajectory']):
            _bounded(row['t_seconds'], 'clock', 0, duration)
            if not math.isclose(row['t_seconds'], row_index*sim.CONTROL_DT, abs_tol=1e-9, rel_tol=0):
                raise ValueError('Profile trace clock mismatch')
            for key, count in (('qpos_native', profile['expected_dimensions']['nq']),
                               ('qvel_native_per_second', profile['expected_dimensions']['nv']),
                               ('arm_joint_position_rad', profile['arm_axes']),
                               ('arm_joint_velocity_rad_s', profile['arm_axes']),
                               ('gripper_joint_position_native', len(profile['gripper_joint_names'])),
                               ('terminal_link_position_m', 3), ('terminal_link_quaternion_wxyz', 4),
                               ('actuator_force_native', len(profile['actuator_names']))):
                if type(row[key]) is not list or len(row[key]) != count:
                    raise ValueError('Profile trace state shape mismatch')
                for value in row[key]:
                    _bounded(value, key, -1000000, 1000000)
            if row['arm_joint_position_rad'] != row['qpos_native'][:profile['arm_axes']] or row['gripper_joint_position_native'] != row['qpos_native'][profile['arm_axes']:] or row['arm_joint_velocity_rad_s'] != row['qvel_native_per_second'][:profile['arm_axes']]:
                raise ValueError('Profile redundant joint state differs from its native coordinates')
            if not math.isclose(sum(value*value for value in row['terminal_link_quaternion_wxyz']), 1., abs_tol=1e-8, rel_tol=0):
                raise ValueError('Profile terminal link quaternion must be normalized wxyz')
            if type(row['contact_count']) is not int or not 0 <= row['contact_count'] <= 100000 or row['observation_origin'] != 'privileged_native_simulator_state':
                raise ValueError('Profile contact count/observation provenance mismatch')
            issued = row['issued_controls_native']
            if attempt['valid'] and ((row_index == len(attempt['trajectory'])-1) != (issued is None)):
                raise ValueError('Valid profile trace must issue each decision and leave its terminal row unissued')
            if issued is not None:
                if type(issued) is not list or len(issued) != len(profile['native_command_units']):
                    raise ValueError('Profile native command shape mismatch')
                for value, (low, high) in zip(issued, profile['native_command_range']):
                    _bounded(value, 'native command', low, high)
    valid = sum(x['valid'] for x in attempts)
    if report['summary'] != {'attempt_count': len(attempts), 'valid_attempts': valid}:
        raise ValueError('Profile attempt summary mismatch')
    license_name = f'licenses/{profile["id"]}/LICENSE'
    record = next(x for x in profile['files'] if x['path'] == profile['folder']+'/LICENSE')
    if len(payload[license_name]) != record['bytes'] or sha256(payload[license_name]).hexdigest() != record['sha256']:
        raise ValueError('Upstream simulation asset license mismatch')
    return {'report': report, 'files': integrity['files'], 'valid_attempts': valid, 'attempts': len(attempts)}


def main(argv=None):
    parser = argparse.ArgumentParser(description='DVIDIA pinned CPU robot mechanics; no hardware control')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list')
    for name in ('fetch', 'audit', 'benchmark'):
        command = commands.add_parser(name)
        command.add_argument('profile', choices=('piper', 'so101'))
        if name != 'fetch':
            command.add_argument('assets')
        if name != 'audit':
            command.add_argument('--output', required=True)
        if name == 'benchmark':
            command.add_argument('--duration', type=float, default=.4)
            command.add_argument('--timestep', type=float, default=.001)
            command.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args(argv)
    try:
        if args.command == 'list':
            result = list_profiles()
        elif args.command == 'fetch':
            result = fetch_assets(args.profile, args.output)
        elif args.command == 'audit':
            result = audit(args.profile, args.assets)
        else:
            result = benchmark(args.profile, args.assets, args.output, duration=args.duration, timestep=args.timestep, repeats=args.repeats)
        print(json.dumps(result, allow_nan=False, indent=2))
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        parser.exit(2, 'error: '+str(exc)+'\n')


if __name__ == '__main__':
    main()

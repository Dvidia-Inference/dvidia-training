"""Bounded native loading, gravity and table-support evidence for shoe proxies.

This benchmark never commands shoe manipulation. Inspection recomputes recorded
outcomes and denominators; hashes detect modification, not publisher identity.
"""
from __future__ import annotations

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
import subprocess
import time

from . import reference_sim as sim

SCHEMA = 'dvidia.asset-library-benchmark/v1'
PROTOCOL_SCHEMA = 'dvidia.asset-library-protocol/v1'
ROBOTS = ('ur5e', 'piper', 'so101')
PAIRS = ('trainer', 'ankle-boot', 'slipper')
CONTROLS = ('table_support', 'table_disabled')
SEED = 612047
DT = .002
DURATION = 1.
SAMPLE_DT = .02
MAX_FILE_BYTES = 24 * 1024 * 1024
FILES = ('protocol.json', 'attempts.json', 'report.json', 'report.html')
SCOPE = ('Native CPU scene loading, free rigid-shoe gravity and passive table support. '
         'Original procedural shoe proxies with assumed physical properties. '
         'No shoe grasping, organizing policy, deformation calibration, learned skill or hardware qualification.')
TIMING_SCOPE = ('Native-step timings measure mj_step only on this host; state extraction, mj_forward, '
                'compilation, rendering, sensing, training and actuation are excluded. Attempt wall includes '
                'verification, compilation, physics and state extraction. Run wall ends before report serialization. '
                'Process peak RSS includes prior allocations and is not a minimum memory requirement. '
                'These measurements do not establish real-time robot performance.')
THRESHOLDS = {'support_window_seconds': .2, 'bottom_tolerance_m': .008,
              'linear_speed_limit_m_s': .03, 'angular_speed_limit_rad_s': .2,
              'contact_normal_force_min_n': 1e-6, 'gravity_probe_seconds': .04,
              'gravity_position_tolerance_m': .001, 'gravity_velocity_tolerance_m_s': .015,
              'control_min_drop_m': .1}


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def _hash(value):
    return sha256(_json(value)).hexdigest()


def implementation_hashes():
    root = Path(__file__).parent
    names = ('asset_benchmark.py', 'asset_library.py', 'asset_catalog.json',
             'reference_sim.py', 'reference_assets.lock.json', 'robot_profiles.py', 'robot_profiles.lock.json')
    return {name: sha256((root/name).read_bytes()).hexdigest() for name in names}


def _plan(seed, repeats):
    rng, result = random.Random(seed), []
    for family in PAIRS:
        for repeat in range(repeats):
            placements = []
            for side in range(2):
                placements.append({'shoe_id': family+('-left' if side == 0 else '-right'),
                                   'position_m': [round(.4+rng.uniform(-.015, .015), 8),
                                                  round(.38+.18*side+rng.uniform(-.015, .015), 8),
                                                  round(rng.uniform(.06, .10), 8)],
                                   'yaw_rad': round(rng.uniform(-.25, .25), 8)})
            for robot in ROBOTS:
                for control in CONTROLS:
                    result.append({'id': f'{robot}-{family}-{repeat:02}-{control}', 'robot': robot,
                                   'family': family, 'repeat': repeat, 'control': control, 'placements': placements})
    return result


def make_protocol(*, seed=SEED, repeats=3, source_commit=None):
    from . import asset_library
    if type(seed) is not int or not 0 <= seed <= 999999999:
        raise ValueError('seed must be an integer from 0 to 999999999')
    if type(repeats) is not int or not 1 <= repeats <= 5:
        raise ValueError('repeats must be an integer from 1 to 5')
    if source_commit is not None and (not isinstance(source_commit, str) or not re.fullmatch(r'[0-9a-f]{40}', source_commit)):
        raise ValueError('source_commit must be a full lowercase Git SHA or null')
    catalog = asset_library.catalog()
    plan = _plan(seed, repeats)
    return {'schema': PROTOCOL_SCHEMA, 'scope': SCOPE, 'seed': seed, 'repeats': repeats,
            'source_commit': source_commit, 'implementation_sha256': implementation_hashes(),
            'catalog': catalog, 'catalog_sha256': _hash(catalog), 'attempts': plan,
            'planned_attempts': len(plan), 'timestep_seconds': DT, 'duration_seconds': DURATION,
            'sample_interval_seconds': SAMPLE_DT, 'thresholds': THRESHOLDS,
            'gravity_m_s2': [0., 0., -9.81], 'timing_scope': TIMING_SCOPE,
            'contact_recording_scope': 'Loaded shoe contacts are sampled every 20 ms. Expected outcomes require no '
                                       'non-table shoe contacts in any recorded sample; transients between samples are not ruled out.',
            'independence_note': 'Matched seeded layouts are reused across robots and controls; these are synthetic '
                                 'numerical checks, not independent real shoes or held-out hardware trials.',
            'comparison': 'Both controls start at the identical free-joint positions, quaternions and held robot commands. '
                          'table_disabled changes only the support plane collision masks to zero. '
                          'Unexpected control support successes, errors and incomplete runs remain in the denominator.'}


def _shoe_rows(catalog):
    return {row['id']: row for row in catalog['shoes']}


def _mass(row):
    return float(row['properties']['mass_kg']['value'])


def _bottom(model, data, body):
    mj, np = sim._native_imports()
    values = []
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != body or not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        vertical = data.geom_xmat[geom].reshape(3, 3)[2]
        if int(model.geom_type[geom]) == int(mj.mjtGeom.mjGEOM_BOX):
            radius = float(np.dot(np.abs(vertical), model.geom_size[geom]))
        elif int(model.geom_type[geom]) == int(mj.mjtGeom.mjGEOM_ELLIPSOID):
            radius = float(np.linalg.norm(vertical*model.geom_size[geom]))
        else:
            raise ValueError('Shoe benchmark supports box and ellipsoid collision primitives only')
        values.append(float(data.geom_xpos[geom, 2])-radius)
    if not values:
        raise ValueError('Shoe has no collision geometry')
    return min(values)


def _sample(model, data, shoes, table, elapsed):
    mj, np = sim._native_imports()
    if not all(np.isfinite(value).all() for value in (data.qpos, data.qvel, data.qacc, data.xpos, data.actuator_force)):
        raise ValueError('Nonfinite native state')
    row = {'t_seconds': elapsed, 'qpos_native': data.qpos.tolist(), 'qvel_native_per_second': data.qvel.tolist(),
           'controls_native': data.ctrl.tolist(), 'shoes': {}}
    for name, body in shoes.items():
        force, contacts, other_contacts, other_geoms = 0., 0, 0, set()
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            first, second = int(contact.geom1), int(contact.geom2)
            if model.geom_bodyid[first] != body and model.geom_bodyid[second] != body:
                continue
            contact_force = np.zeros(6)
            mj.mj_contactForce(model, data, contact_id, contact_force)
            normal = max(0., float(contact_force[0]))
            if normal > THRESHOLDS['contact_normal_force_min_n']:
                if first == table or second == table:
                    contacts += 1
                    force += normal
                else:
                    other_contacts += 1
                    other = second if model.geom_bodyid[first] == body else first
                    other_geoms.add(mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, other) or f'unnamed_geom_{other}')
        joint = int(model.body_jntadr[body])
        velocity = data.qvel[int(model.jnt_dofadr[joint]):int(model.jnt_dofadr[joint])+6]
        row['shoes'][name] = {'body_position_m': data.xpos[body].tolist(),
                             'body_quaternion_wxyz': data.xquat[body].tolist(),
                             'bottom_m': _bottom(model, data, body), 'support_contacts': contacts,
                             'support_normal_force_n': force, 'linear_speed_m_s': float(np.linalg.norm(velocity[:3])),
                             'other_contacts': other_contacts, 'other_contact_geoms': sorted(other_geoms),
                             'angular_speed_rad_s': float(np.linalg.norm(velocity[3:])),
                             'vertical_velocity_m_s': float(velocity[2])}
    return row


def _score_shoe(trajectory, shoe_id):
    if len(trajectory) != round(DURATION/SAMPLE_DT)+1:
        observed = [row['shoes'][shoe_id]['other_contacts'] for row in trajectory]
        return {'settled_supported': False, 'gravity_observed': False, 'dropped_without_support': False,
                'unexpected_contact_samples': sum(value > 0 for value in observed),
                'max_other_contacts': max(observed, default=0)}
    records = [row['shoes'][shoe_id] for row in trajectory]
    final = [row['shoes'][shoe_id] for row in trajectory if row['t_seconds'] >= DURATION-THRESHOLDS['support_window_seconds']-1e-9]
    probe = records[round(THRESHOLDS['gravity_probe_seconds']/SAMPLE_DT)]
    t = THRESHOLDS['gravity_probe_seconds']
    gravity = (abs(probe['body_position_m'][2]-records[0]['body_position_m'][2]+.5*9.81*t*t)
               <= THRESHOLDS['gravity_position_tolerance_m'] and
               abs(probe['vertical_velocity_m_s']+9.81*t) <= THRESHOLDS['gravity_velocity_tolerance_m_s'])
    supported = all(row['support_contacts'] > 0 and row['support_normal_force_n'] > THRESHOLDS['contact_normal_force_min_n']
                    and abs(row['bottom_m']) <= THRESHOLDS['bottom_tolerance_m']
                    and row['linear_speed_m_s'] <= THRESHOLDS['linear_speed_limit_m_s']
                    and row['angular_speed_rad_s'] <= THRESHOLDS['angular_speed_limit_rad_s'] for row in final)
    drop = records[0]['body_position_m'][2]-records[-1]['body_position_m'][2]
    return {'settled_supported': supported, 'gravity_observed': gravity,
            'dropped_without_support': drop >= THRESHOLDS['control_min_drop_m'] and
                                       all(row['support_contacts'] == 0 for row in records),
            'unexpected_contact_samples': sum(row['other_contacts'] > 0 for row in records),
            'max_other_contacts': max(row['other_contacts'] for row in records),
            'vertical_drop_m': drop, 'final_bottom_m': records[-1]['bottom_m'],
            'final_linear_speed_m_s': records[-1]['linear_speed_m_s'],
            'final_angular_speed_rad_s': records[-1]['angular_speed_rad_s']}


def _scores(attempt):
    shoes = {row['shoe_id']: _score_shoe(attempt['trajectory'], row['shoe_id']) for row in attempt['plan']['placements']}
    valid = (attempt['error'] is None and not attempt['warnings'] and attempt['native_warning_count'] == 0
             and attempt['simulated_seconds'] == DURATION and len(attempt['trajectory']) == 51
             and len(attempt['native_step_wall_seconds']) == 500 and attempt['model_contract']['passed'] is True)
    support = valid and all(row['settled_supported'] for row in shoes.values())
    gravity = valid and all(row['gravity_observed'] for row in shoes.values())
    unexpected_contact = any(row['max_other_contacts'] > 0 for row in shoes.values())
    expected = gravity and not unexpected_contact and (support if attempt['plan']['control'] == 'table_support' else
                            not support and all(row['dropped_without_support'] for row in shoes.values()))
    return {'valid': valid, 'support_success': support, 'gravity_success': gravity,
            'unexpected_contact_detected': unexpected_contact, 'expected_outcome': expected, 'shoes': shoes}


def _attempt(plan, assets, catalog):
    from . import asset_library
    started = time.perf_counter()
    result = {'plan': plan, 'warnings': [], 'native_warning_count': 0, 'error': None, 'trajectory': [],
              'native_step_wall_seconds': [], 'simulated_seconds': 0., 'model_contract': {'passed': False}}
    try:
        with sim.native_warnings(result['warnings']):
            compiling = time.perf_counter()
            model = asset_library.build_scene(plan['robot'], assets, shoe_ids=[row['shoe_id'] for row in plan['placements']], timestep=DT)
            result['compile_wall_seconds'] = time.perf_counter()-compiling
            mj, np = sim._native_imports()
            data = asset_library.initialize(model)
            table = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, 'asset_table')
            if table < 0 or int(model.geom_type[table]) != int(mj.mjtGeom.mjGEOM_PLANE) or model.geom_pos[table, 2] != 0:
                raise ValueError('Expected asset_table support plane at z=0')
            if plan['control'] == 'table_disabled':
                model.geom_contype[table] = model.geom_conaffinity[table] = 0
            shoes, metadata, expected_shoes = {}, {}, _shoe_rows(catalog)
            for placement in plan['placements']:
                name = placement['shoe_id']
                body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'shoe_'+name.replace('-', '_'))
                if body <= 0 or model.body_jntnum[body] != 1:
                    raise ValueError('Expected one shoe free joint')
                joint = int(model.body_jntadr[body])
                if int(model.jnt_type[joint]) != int(mj.mjtJoint.mjJNT_FREE) or model.body_gravcomp[body] != 0:
                    raise ValueError('Shoes require gravity-active native free joints')
                if not math.isclose(float(model.body_mass[body]), _mass(expected_shoes[name]), rel_tol=1e-9):
                    raise ValueError('Native shoe mass differs from catalog')
                address = int(model.jnt_qposadr[joint])
                yaw = placement['yaw_rad']
                data.qpos[address:address+7] = placement['position_m']+[math.cos(yaw/2), 0., 0., math.sin(yaw/2)]
                data.qvel[int(model.jnt_dofadr[joint]):int(model.jnt_dofadr[joint])+6] = 0
                shoes[name] = body
                metadata[name] = {'mass_kg': float(model.body_mass[body]), 'free_joint': True,
                                  'gravity_compensation': float(model.body_gravcomp[body]),
                                  'qpos_address': address, 'qvel_address': int(model.jnt_dofadr[joint])}
            if not np.array_equal(model.opt.gravity, [0., 0., -9.81]) or model.opt.timestep != DT:
                raise ValueError('Native gravity or timestep differs from frozen protocol')
            result['model_contract'] = {'passed': True, 'gravity_m_s2': model.opt.gravity.tolist(),
                                        'timestep_seconds': float(model.opt.timestep), 'shoes': metadata,
                                        'dimensions': {key: int(getattr(model, key)) for key in ('nq', 'nv', 'nu', 'ngeom')},
                                        'table_collision_masks': [int(model.geom_contype[table]), int(model.geom_conaffinity[table])]}
            result['compiled_model_sha256'] = sim.model_digest(model)
            result['runtime'] = {'mujoco': mj.__version__, 'numpy': np.__version__}
            mj.mj_forward(model, data)
            result['trajectory'].append(_sample(model, data, shoes, table, 0.))
            for step in range(round(DURATION/DT)):
                tick = time.perf_counter()
                mj.mj_step(model, data)
                result['native_step_wall_seconds'].append(time.perf_counter()-tick)
                if not all(np.isfinite(value).all() for value in (data.qpos, data.qvel, data.qacc)):
                    raise ValueError('Nonfinite native state during stepping')
                result['native_warning_count'] = sum(int(warning.number) for warning in data.warning)
                if result['warnings'] or result['native_warning_count']:
                    raise ValueError('Native solver warning')
                result['simulated_seconds'] = (step+1)*DT
                if (step+1) % round(SAMPLE_DT/DT) == 0:
                    mj.mj_forward(model, data)
                    result['trajectory'].append(_sample(model, data, shoes, table, (step+1)*DT))
    except Exception as exc:
        result['error'] = {'type': type(exc).__name__, 'message': str(exc)[:500]}
    result['attempt_wall_seconds'] = time.perf_counter()-started
    result['scores'] = _scores(result)
    return result


def _percentiles(values):
    if not values:
        return {'count': 0, 'p50_seconds': None, 'p95_seconds': None, 'p99_seconds': None, 'max_seconds': None, 'total_seconds': 0.}
    ordered = sorted(values)
    def quantile(q):
        position = (len(ordered)-1)*q
        lower, upper = math.floor(position), math.ceil(position)
        return ordered[lower]+(ordered[upper]-ordered[lower])*(position-lower)
    return {'count': len(values), 'p50_seconds': quantile(.5), 'p95_seconds': quantile(.95),
            'p99_seconds': quantile(.99), 'max_seconds': max(values), 'total_seconds': sum(values)}


def aggregate(attempts):
    result = []
    for robot in ROBOTS:
        for control in CONTROLS:
            rows = [row for row in attempts if row['plan']['robot'] == robot and row['plan']['control'] == control]
            result.append({'robot': robot, 'control': control, 'attempted': len(rows),
                           **{key: sum(row['scores'][key] is True for row in rows) for key in
                              ('valid', 'support_success', 'gravity_success', 'expected_outcome')}})
            result[-1]['unexpected_contact_attempts'] = sum(row['scores']['unexpected_contact_detected'] is True for row in rows)
    return result


def _peak_rss():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == 'darwin' else value*1024) if sys.platform in ('darwin', 'linux') else None
    except (ImportError, OSError, ValueError):
        return None


def _hardware():
    result = {'platform': platform.system(), 'machine': platform.machine(), 'python': platform.python_version(),
              'logical_cpu_count': os.cpu_count(), 'cpu_model': platform.processor() or None, 'total_memory_bytes': None}
    if sys.platform == 'darwin':
        for field, key in (('cpu_model', 'machdep.cpu.brand_string'), ('total_memory_bytes', 'hw.memsize')):
            try:
                value = subprocess.run(['/usr/sbin/sysctl', '-n', key], check=True, capture_output=True,
                                       text=True, timeout=2).stdout.strip()
                result[field] = int(value) if field == 'total_memory_bytes' else value[:160]
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
    elif sys.platform == 'linux':
        try:
            result['total_memory_bytes'] = os.sysconf('SC_PHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')
            for line in Path('/proc/cpuinfo').read_text()[:65536].splitlines():
                if line.startswith('model name'):
                    result['cpu_model'] = line.split(':', 1)[1].strip()[:160]
                    break
        except (OSError, ValueError):
            pass
    return result


def _html(report):
    rows = ''.join('<tr>'+''.join('<td>'+escape(str(row[key]))+'</td>' for key in
                   ('robot', 'control', 'attempted', 'valid', 'support_success', 'gravity_success', 'unexpected_contact_attempts', 'expected_outcome'))+'</tr>'
                   for row in report['summary'])
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>DVIDIA · Shoe asset benchmark</title><style>body{font:18px/1.65 Georgia,serif;max-width:1000px;margin:auto;'
            'padding:24px;background:#fafaf7;color:#292d27}a{color:#396751}.scroll{overflow:auto}table{border-collapse:collapse;'
            'font:14px/1.5 system-ui}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;'
            'overflow-wrap:anywhere;font-size:13px}</style><main><p>DVIDIA RESEARCH</p><h1>Shoe assets: load, drop, settle</h1><p>'
            +escape(SCOPE)+'</p><p>'+str(report['attempted'])+' planned attempts. Failures and invalid attempts remain in the denominator. '
            'Support success is expected with the table; zero support successes are expected with its collisions disabled.</p>'
            '<p><a href="protocol.json">Frozen protocol and catalog</a> · <a href="attempts.json">Every native trajectory</a> · '
            '<a href="report.json">Report</a> · <a href="integrity.json">Integrity</a></p><div class="scroll"><table><thead><tr>'
            '<th>Arm</th><th>Control</th><th>Attempts</th><th>Valid</th><th>Supported</th><th>Gravity</th><th>Other contact</th><th>Expected outcome</th>'
            '</tr></thead><tbody>'+rows+'</tbody></table></div><h2>Measurement scope</h2><p>'+escape(TIMING_SCOPE)+
            '</p><pre>'+escape(json.dumps(report['timing'], indent=2, sort_keys=True))+'</pre><p>Sampled trajectories are privileged native state. '
            'The first library checks composition and passive support. A manipulation controller and measured shoe properties '
            'are separate evidence requirements.</p><footer><a href="https://research.dvidia.org/">DVIDIA research</a> · '
            '<a href="https://github.com/Dvidia-Inference/dvidia-training">GitHub</a> · '
            '<a href="https://x.com/iammrriver">@iammrriver</a> · <a href="mailto:hello@dvidia.org">hello@dvidia.org</a>'
            '</footer></main></html>')


def benchmark(asset_roots, output, *, seed=SEED, repeats=3, source_commit=None, progress=None):
    protocol = make_protocol(seed=seed, repeats=repeats, source_commit=source_commit)
    if not isinstance(asset_roots, dict) or set(asset_roots) != set(ROBOTS):
        raise ValueError('asset_roots requires ur5e, piper and so101 paths')
    destination = Path(output)
    if destination.is_symlink():
        raise ValueError('Output requires a fresh regular directory')
    destination.mkdir(parents=True, exist_ok=False)
    (destination/'protocol.json').write_bytes(_json(protocol))
    started, attempts = time.perf_counter(), []
    for plan in protocol['attempts']:
        attempts.append(_attempt(plan, asset_roots[plan['robot']], protocol['catalog']))
        if progress is not None:
            progress({'completed': len(attempts), 'planned': len(protocol['attempts']), 'id': plan['id'],
                      'valid': attempts[-1]['scores']['valid'], 'expected_outcome': attempts[-1]['scores']['expected_outcome']})
    elapsed = time.perf_counter()-started
    payload = {'schema': SCHEMA, 'protocol_sha256': _hash(protocol), 'attempts': attempts}
    native_times = [value for row in attempts for value in row['native_step_wall_seconds']]
    simulated = sum(row['simulated_seconds'] for row in attempts)
    report = {'schema': SCHEMA, 'scope': SCOPE, 'physical_robot_ready': False, 'shoe_manipulation_qualified': False,
              'protocol_sha256': _hash(protocol), 'catalog_sha256': protocol['catalog_sha256'],
              'attempts_sha256': _hash(payload), 'attempted': len(attempts), 'summary': aggregate(attempts),
              'hardware': _hardware(),
              'timing': {'scope': TIMING_SCOPE, 'native_step': _percentiles(native_times),
                         'compile': _percentiles([row['compile_wall_seconds'] for row in attempts if 'compile_wall_seconds' in row]),
                         'attempt_wall': _percentiles([row['attempt_wall_seconds'] for row in attempts]),
                         'simulated_seconds': simulated, 'run_wall_seconds': elapsed,
                         'simulated_seconds_per_run_wall_second': simulated/elapsed if elapsed else None,
                         'process_peak_rss_bytes': _peak_rss()}}
    (destination/'attempts.json').write_bytes(_json(payload))
    (destination/'report.json').write_bytes(_json(report))
    (destination/'report.html').write_text(_html(report), encoding='utf-8')
    integrity = {'schema': 'dvidia.asset-library-integrity/v1', 'files':
                 {name: {'sha256': sha256((destination/name).read_bytes()).hexdigest(), 'bytes': (destination/name).stat().st_size}
                  for name in FILES}}
    (destination/'integrity.json').write_bytes(_json(integrity))
    inspect_benchmark(destination)
    return report


def _read(path):
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_FILE_BYTES:
        raise ValueError('Benchmark files must be bounded regular files')
    return path.read_bytes()


def _decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def _finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite benchmark value')
    if isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item)


def inspect_benchmark(directory):
    """Verify seals and recorded protocol/trajectory outcomes, without native replay."""
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Benchmark requires a regular directory')
    try:
        if {path.name for path in root.iterdir()} != set(FILES) | {'integrity.json'}:
            raise ValueError('Unexpected or missing benchmark files')
        integrity = _decode(_read(root/'integrity.json'))
        if integrity['schema'] != 'dvidia.asset-library-integrity/v1' or set(integrity['files']) != set(FILES):
            raise ValueError('Unexpected integrity contract')
        blobs = {name: _read(root/name) for name in FILES}
        for name, raw in blobs.items():
            if integrity['files'][name] != {'sha256': sha256(raw).hexdigest(), 'bytes': len(raw)}:
                raise ValueError('Benchmark digest mismatch: '+name)
        protocol, payload, report = (_decode(blobs[name]) for name in FILES[:3])
        for value in (protocol, payload, report):
            _finite_tree(value)
        if protocol['schema'] != PROTOCOL_SCHEMA or payload['schema'] != SCHEMA or report['schema'] != SCHEMA:
            raise ValueError('Unsupported benchmark schema')
        if protocol['source_commit'] is not None and (not isinstance(protocol['source_commit'], str) or not re.fullmatch(r'[0-9a-f]{40}', protocol['source_commit'])):
            raise ValueError('Invalid source commit declaration')
        if set(protocol['implementation_sha256']) != set(implementation_hashes()) or any(not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest) for digest in protocol['implementation_sha256'].values()):
            raise ValueError('Invalid implementation digest declarations')
        if type(protocol['seed']) is not int or not 0 <= protocol['seed'] <= 999999999 or type(protocol['repeats']) is not int or not 1 <= protocol['repeats'] <= 5:
            raise ValueError('Invalid protocol seed or repeats')
        if protocol['attempts'] != _plan(protocol['seed'], protocol['repeats']) or protocol['planned_attempts'] != len(protocol['attempts']):
            raise ValueError('Frozen denominator or seeded layouts changed')
        if protocol['thresholds'] != THRESHOLDS or protocol['timestep_seconds'] != DT or protocol['duration_seconds'] != DURATION or protocol['sample_interval_seconds'] != SAMPLE_DT or protocol['gravity_m_s2'] != [0., 0., -9.81]:
            raise ValueError('Frozen scoring contract changed')
        if report['scope'] != SCOPE or protocol['scope'] != SCOPE or report['physical_robot_ready'] is not False or report['shoe_manipulation_qualified'] is not False:
            raise ValueError('Benchmark scope claims changed')
        if protocol['catalog_sha256'] != _hash(protocol['catalog']) or report['catalog_sha256'] != protocol['catalog_sha256'] or payload['protocol_sha256'] != _hash(protocol) or report['protocol_sha256'] != _hash(protocol) or report['attempts_sha256'] != _hash(payload):
            raise ValueError('Linked evidence digests differ')
        attempts = payload['attempts']
        if len(attempts) != protocol['planned_attempts'] or [row['plan'] for row in attempts] != protocol['attempts']:
            raise ValueError('Missing, reordered or substituted attempts')
        expected_shoes = _shoe_rows(protocol['catalog'])
        expected_robots = {row['id']: row for row in protocol['catalog']['robots']}
        for row in attempts:
            trajectory, contract = row['trajectory'], row['model_contract']
            if len(trajectory) > 51 or len(row['native_step_wall_seconds']) > 500 or any(type(t) not in (int, float) or t < 0 for t in row['native_step_wall_seconds']):
                raise ValueError('Invalid bounded trace or timing values')
            if type(row['attempt_wall_seconds']) not in (int, float) or row['attempt_wall_seconds'] < 0 or type(row['simulated_seconds']) not in (int, float) or not 0 <= row['simulated_seconds'] <= DURATION or ('compile_wall_seconds' in row and (type(row['compile_wall_seconds']) not in (int, float) or row['compile_wall_seconds'] < 0)):
                raise ValueError('Invalid attempt timing')
            if trajectory:
                dimensions = contract['dimensions']
                names = {placement['shoe_id'] for placement in row['plan']['placements']}
                base = expected_robots[row['plan']['robot']]['expected_dimensions']
                if any(type(dimensions[key]) is not int or dimensions[key] != base[key]+extra for key, extra in (('nq', 7*len(names)), ('nv', 6*len(names)), ('nu', 0))):
                    raise ValueError('Native dimensions differ from the catalog robot')
                if not isinstance(row['compiled_model_sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', row['compiled_model_sha256']) or row['runtime']['mujoco'] != '3.15.0' or not isinstance(row['runtime']['numpy'], str):
                    raise ValueError('Invalid native runtime or compiled model digest')
                for index, sample in enumerate(trajectory):
                    if not math.isclose(sample['t_seconds'], index*SAMPLE_DT, abs_tol=1e-12) or set(sample['shoes']) != names or len(sample['qpos_native']) != dimensions['nq'] or len(sample['qvel_native_per_second']) != dimensions['nv'] or len(sample['controls_native']) != dimensions['nu']:
                        raise ValueError('Trajectory dimensions, times or shoe identities differ')
                    if sample['controls_native'] != trajectory[0]['controls_native']:
                        raise ValueError('Passive benchmark changed robot commands')
                    for name, shoe in sample['shoes'].items():
                        meta = contract['shoes'][name]
                        qp, qv = meta['qpos_address'], meta['qvel_address']
                        pose = shoe['body_position_m']+shoe['body_quaternion_wxyz']
                        if len(pose) != 7 or not all(math.isclose(a, b, rel_tol=0., abs_tol=1e-12) for a, b in zip(pose, sample['qpos_native'][qp:qp+7])):
                            raise ValueError('Recorded shoe pose differs from native free-joint state')
                        velocity = sample['qvel_native_per_second'][qv:qv+6]
                        if not math.isclose(shoe['linear_speed_m_s'], math.sqrt(sum(v*v for v in velocity[:3])), rel_tol=1e-10, abs_tol=1e-12) or not math.isclose(shoe['angular_speed_rad_s'], math.sqrt(sum(v*v for v in velocity[3:])), rel_tol=1e-10, abs_tol=1e-12) or shoe['vertical_velocity_m_s'] != velocity[2]:
                            raise ValueError('Recorded shoe speed differs from native free-joint state')
                        if type(shoe['support_contacts']) is not int or shoe['support_contacts'] < 0 or shoe['support_normal_force_n'] < 0:
                            raise ValueError('Invalid support contact record')
                        if type(shoe['other_contacts']) is not int or shoe['other_contacts'] < 0 or not isinstance(shoe['other_contact_geoms'], list) or any(not isinstance(name, str) or not name for name in shoe['other_contact_geoms']) or shoe['other_contact_geoms'] != sorted(set(shoe['other_contact_geoms'])) or len(shoe['other_contact_geoms']) > shoe['other_contacts'] or bool(shoe['other_contact_geoms']) != bool(shoe['other_contacts']):
                            raise ValueError('Invalid non-table contact record')
                for placement in row['plan']['placements']:
                    name, yaw = placement['shoe_id'], placement['yaw_rad']
                    initial, meta = trajectory[0]['shoes'][name], contract['shoes'][name]
                    if initial['body_position_m'] != placement['position_m'] or not all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(initial['body_quaternion_wxyz'], [math.cos(yaw/2), 0, 0, math.sin(yaw/2)])):
                        raise ValueError('Initial native shoe pose differs from protocol')
                    if meta['free_joint'] is not True or meta['gravity_compensation'] != 0 or not math.isclose(meta['mass_kg'], _mass(expected_shoes[name]), rel_tol=1e-9):
                        raise ValueError('Native shoe mass or free joint differs from catalog')
                    if initial['linear_speed_m_s'] != 0 or initial['angular_speed_rad_s'] != 0:
                        raise ValueError('Initial shoe velocity differs from the resting release protocol')
                masks = contract['table_collision_masks']
                if (row['plan']['control'] == 'table_disabled' and masks != [0, 0]) or (row['plan']['control'] == 'table_support' and not any(masks)) or contract['gravity_m_s2'] != [0., 0., -9.81] or contract['timestep_seconds'] != DT:
                    raise ValueError('Native table, gravity or timestep contract differs')
            if _json(row['scores']) != _json(_scores(row)):
                raise ValueError('Recorded shoe outcomes do not match trajectory')
        if report['attempted'] != len(attempts) or report['summary'] != aggregate(attempts):
            raise ValueError('Report denominator or aggregate differs')
        native_times = [value for row in attempts for value in row['native_step_wall_seconds']]
        if report['timing']['native_step'] != _percentiles(native_times) or report['timing']['attempt_wall'] != _percentiles([row['attempt_wall_seconds'] for row in attempts]) or report['timing']['compile'] != _percentiles([row['compile_wall_seconds'] for row in attempts if 'compile_wall_seconds' in row]) or report['timing']['simulated_seconds'] != sum(row['simulated_seconds'] for row in attempts):
            raise ValueError('Timing aggregates differ from raw records')
        if report['timing']['scope'] != TIMING_SCOPE or blobs['report.html'] != _html(report).encode():
            raise ValueError('Report presentation or timing scope differs')
        wall = report['timing']['run_wall_seconds']
        if type(wall) not in (int, float) or wall <= 0 or report['timing']['simulated_seconds_per_run_wall_second'] != report['timing']['simulated_seconds']/wall:
            raise ValueError('Run throughput differs from recorded elapsed time')
        return {'schema': SCHEMA, 'verified': True, 'attempts': len(attempts),
                'valid_attempts': sum(row['scores']['valid'] for row in attempts), 'report': report,
                'verification_scope': 'Integrity and recorded invariants only; no physics rerun, contact calibration or publisher authentication.'}
    except (KeyError, TypeError, IndexError, OverflowError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError('Malformed asset benchmark evidence') from exc

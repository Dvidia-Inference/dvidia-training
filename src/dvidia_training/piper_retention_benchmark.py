"""Fixed fresh-scene comparison of authored PiPER short-pause hold controls.

This compares three closed-loop controllers, not learned policies or hardware.
Every attempt, refusal and native row is retained. The primary endpoint is a
continuous five-second native predicate after the first valid admission;
controller status at that endpoint and the final hold are separate outcomes.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

from . import piper_probe_benchmark as prior

SCHEMA = 'dvidia.piper-retention-study/v1'
VARIANTS = ('measured_joint', 'issued_joint', 'cartesian')
ORDERS = ((0, 1, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0), (1, 0, 2), (0, 2, 1))
HOLD_S = 5.
FORCE_CEILING_N = 10.
TCP_POSITION_CEILING_M = .004
TCP_ORIENTATION_CEILING_RAD = .05
BOUND_REFUSALS = frozenset(('initial_bias_exhausted', 'native_joint_margin_exhausted',
    'tcp_position_error_exhausted', 'tcp_orientation_error_exhausted', 'joint_correction_exhausted',
    'pose_solver_refused', 'missing_prior_issued_target'))
MAX_FILE = 32*1024*1024
SCOPE = ('Authored synthetic PiPER probe and short-pause controls with privileged simulator state. '
         'Fresh finite development cells; no learning, full placement, physical sensing or hardware qualification.')
encoded, digest, write, read = prior.encoded, prior.digest, prior.write, prior.read


def case_definitions():
    """Pure fixed cells/order; no native imports, asset fetch or parameter search."""
    from .piper_task import PiperTaskConfig
    base, cases = PiperTaskConfig(), []
    for mass in (.045, .055):
        for friction in (.75, .50, .20):
            for layout, start in enumerate(((.303, -.026), (.307, -.037))):
                index = len(cases)
                config = replace(base, mass_kg=mass, friction=friction, object_position_m=(*start, .082))
                cases.append({'id': f'm{round(mass*1000):02}-f{round(friction*100):02}-p{layout}',
                              'seed': 311100+index, 'config': asdict(config),
                              'variant_order': [VARIANTS[i] for i in ORDERS[index % len(ORDERS)]]})
    return cases


def roster(protocol):
    return [(case, variant) for case in protocol['cases'] for variant in case['variant_order']]


def make_protocol(assets):
    from .robot_profiles import verify_assets, LOCK_PATH
    from .piper_retention import retention_contract
    return {'schema': SCHEMA, 'scope': SCOPE, 'physical_robot_ready': False, 'learned_policy': False,
            'cohort': 'fresh_development_method_check', 'cases': case_definitions(), 'variants': list(VARIANTS),
            'planned_attempts': 36, 'source_sha256': prior.sources(), 'runtime': prior.runtime(),
            'asset_provenance': verify_assets('piper', assets),
            'asset_lock_sha256': sha256(LOCK_PATH.read_bytes()).hexdigest(),
            'controller_contract': retention_contract(), 'force_ceiling_n': FORCE_CEILING_N,
            'force_ceiling_scope': 'Unchanged arbitrary synthetic admission ceiling, not measured damage calibration or a force command.',
            'primary_endpoint': {'duration_s': HOLD_S, 'native_dt_s': .001,
                'definition': 'Correct 100 ms native admission prefix, then every native predicate sample through the next 5 s; complete valid warning-free run, no unexpected contact anywhere in the run and no force-ceiling violation on any subsequent hold row.',
                'predicate': 'Both contact counts positive, both pad forces strictly >0.01 N and <=10 N including the admission-prefix window, no table support or unexpected contact, rise >=0.006 m from this probe origin and TCP-local drift <=0.003 m; distance epsilon1e-12.',
                'missing_followup': 'No admission or less than five seconds of recorded followup is retained as failure, never padded.',
                'controller_status': 'Controller retained-at-5s and final hold are reported separately; native success never grants actuator permission.'},
            'secondary_endpoint': {'name': 'grip_retention_and_tcp_stationarity',
                'duration_s': HOLD_S, 'position_ceiling_m': TCP_POSITION_CEILING_M,
                'orientation_ceiling_rad': TCP_ORIENTATION_CEILING_RAD,
                'definition': 'Primary native grip retention succeeds, and every native TCP pose from admission through five seconds stays within 4 mm and 0.05 rad of the admission pose. Identical bounds for all three controllers, measured from recorded native poses.'},
            'order': 'Six balanced three-variant permutations repeated twice; each variant occupies each position four times. One attempt per cell/variant; no automatic resume, retry of a trial, selection or tuning.',
            'comparison': 'Shared acquisition native prefixes must be exactly equal through the earliest common treatment boundary: an admission or an explicit pre-admission hold-bound refusal in any variant. Commands strictly before that boundary must match. If no treatment begins, compare the full traces. Closed-loop hold controls may then change trajectories. These finite cells do not estimate population accuracy or isolate arbitrary physical mechanisms.',
            'timing': 'Unpaced shared-host policy-call and full control-step wall/CPU compute; step includes dense telemetry. Reset/compile and evidence serialization are separate from step samples. No sensor/hardware latency or real-time guarantee.'}


def prepare(output, assets):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError('Choose a fresh output directory; no evidence is overwritten.')
    protocol = make_protocol(assets)
    output.mkdir(parents=True)
    write(output/'protocol.json', protocol)
    write(output/'frozen.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(),
                                'protocol_sha256': sha256((output/'protocol.json').read_bytes()).hexdigest()})
    sources = output/'sources'; sources.mkdir()
    for name, expected in protocol['source_sha256'].items():
        raw = Path(__file__).with_name(name).read_bytes()
        if sha256(raw).hexdigest() != expected:
            raise ValueError('Source changed during preparation.')
        (sources/name).write_bytes(raw)
    return protocol


def _timing(values):
    result = prior._timing(values)
    if result is not None:
        result['p99_s'] = float(np.percentile(values, 99))
    return result


def _attempt(assets, case, variant):
    from .piper_task import PiperTaskConfig
    from .piper_feedback_env import FeedbackPiperTaskEnv
    from .piper_retention import RetentionPolicy
    from .reference_sim import model_digest
    env = FeedbackPiperTaskEnv(assets, PiperTaskConfig(**case['config']))
    result = {'schema': SCHEMA, 'id': case['id']+'-'+variant, 'scene_id': case['id'], 'variant': variant,
              'seed': case['seed'], 'config': case['config'], 'valid': False, 'physical_robot_ready': False,
              'trajectory': [], 'actions': [], 'error': None, 'metrics': {}, 'policy': None}
    timers = {name: [] for name in ('policy_wall', 'policy_cpu', 'step_wall', 'step_cpu')}
    policy = None
    began = time.perf_counter()
    try:
        start = time.perf_counter()
        obs, info = env.reset(case['seed'])
        result['setup_reset_wall_s'] = time.perf_counter()-start
        result.update(initial_object_position_m=env.initial_object_position.tolist(),
                      compiled_model_sha256=model_digest(env.model),
                      native_force_limits=env.model.actuator_forcerange.tolist())
        policy = RetentionPolicy(env, variant=variant)
        result['trajectory'].append(obs)
        for _ in range(round(env.config.timeout_s/env.config.control_dt)):
            wall, cpu = time.perf_counter(), time.process_time()
            action = policy(obs)
            timers['policy_wall'].append(time.perf_counter()-wall)
            timers['policy_cpu'].append(time.process_time()-cpu)
            record = {'t_seconds': obs['t_seconds'], **action}
            result['actions'].append(record)
            obs.update(command=record, issued_joint_targets_rad=list(action['joint_targets_rad']),
                       issued_gripper_position_m=action['gripper_position_m'])
            wall, cpu = time.perf_counter(), time.process_time()
            obs, info = env.step(action)
            timers['step_wall'].append(time.perf_counter()-wall)
            timers['step_cpu'].append(time.process_time()-cpu)
            if obs is not None:
                result['trajectory'].append(obs)
            if env.done:
                break
        if obs is not None:
            policy.observe_final(obs)
        result.update(valid=info['valid'], metrics=info, policy=policy.summary())
    except Exception as exc:
        result['error'] = {'type': type(exc).__name__, 'message': str(exc)[:300]}
        if env.ready:
            result['metrics'] = env.info()
        if policy is not None:
            try:
                result['policy'] = policy.summary()
            except Exception as summary_exc:
                result['policy_summary_error'] = {'type': type(summary_exc).__name__,
                                                   'message': str(summary_exc)[:300]}
    result['timing'] = {'attempt_wall_s': time.perf_counter()-began,
                        'summaries': {name: _timing(values) for name, values in timers.items()}, 'samples': timers}
    return result, env.native_arrays(), env.native_metadata()


def _index(timestamp, times):
    index = round(timestamp/.001)-1
    if not 0 <= index < len(times) or abs(times[index]-timestamp) > 1e-8:
        raise ValueError('Expected an observed native timestamp.')
    return index


def probe_series(arrays, probe, force_ceiling_n=FORCE_CEILING_N):
    """Pure causal native predicate; independent of controller status labels."""
    times = arrays['t_seconds']
    origin = _index(probe['start_time_s'], times)
    if (np.linalg.norm(arrays['object_position_m'][origin]-probe['object_origin_m']) > 1e-9
            or np.linalg.norm(arrays['object_minus_tcp_local_m'][origin]-probe['local_offset_origin_m']) > 1e-9):
        raise ValueError('Probe origin differs from native evidence.')
    rise = arrays['object_position_m'][:, 2]-probe['object_origin_m'][2]
    drift = np.linalg.norm(arrays['object_minus_tcp_local_m']-probe['local_offset_origin_m'], axis=1)
    condition = ((np.arange(len(times)) > origin)
                 & np.all(arrays['grasp_contact_counts'] > 0, axis=1)
                 & np.all(arrays['pad_normal_forces_n'] > .01, axis=1)
                 & np.all(arrays['pad_normal_forces_n'] <= force_ceiling_n, axis=1)
                 & (arrays['table_object_contact_count'] == 0)
                 & (arrays['unexpected_contact_count'] == 0)
                 & (rise+1e-12 >= .006) & (drift <= .003+1e-12))
    streak = np.empty(len(times), dtype=np.int64)
    count = 0
    for index, good in enumerate(condition):
        count = count+1 if good else 0
        streak[index] = count
    return condition, streak, streak >= 100, rise, drift


def _decomposition(arrays, start, end):
    obj = arrays['object_position_m'][end]-arrays['object_position_m'][start]
    tcp = arrays['tcp_position_m'][end]-arrays['tcp_position_m'][start]
    return {'object_world_z_delta_m': float(obj[2]), 'tcp_world_z_delta_m': float(tcp[2]),
            'relative_world_z_delta_m': float(obj[2]-tcp[2]),
            'local_offset_change_m': float(np.linalg.norm(arrays['object_minus_tcp_local_m'][end]-arrays['object_minus_tcp_local_m'][start])),
            'elapsed_s': (end-start)*.001,
            'scope': 'Kinematic components, not causal attribution or calibrated physical slip.'}


def _stationarity(arrays, start, end, enough, grip_success):
    positions = arrays['tcp_position_m'][start:end+1]
    quaternions = arrays['tcp_quaternion_wxyz'][start:end+1]
    displacement = np.linalg.norm(positions-positions[0], axis=1)
    # q and -q are the same orientation; shortest quaternion angular distance.
    angles = 2*np.arccos(np.clip(np.abs(quaternions @ quaternions[0]), 0., 1.))
    max_position, max_angle = float(np.max(displacement)), float(np.max(angles))
    within = max_position <= TCP_POSITION_CEILING_M+1e-12 and max_angle <= TCP_ORIENTATION_CEILING_RAD+1e-12
    return {'required_s': HOLD_S, 'complete_5s': bool(enough),
            'position_ceiling_m': TCP_POSITION_CEILING_M,
            'orientation_ceiling_rad': TCP_ORIENTATION_CEILING_RAD,
            'maximum_tcp_position_excursion_m': max_position,
            'maximum_tcp_orientation_excursion_rad': max_angle,
            'terminal_tcp_position_delta_m': (positions[-1]-positions[0]).tolist(),
            'terminal_tcp_orientation_delta_rad': float(angles[-1]),
            'tcp_pose_within_bounds': bool(enough and within),
            'grip_retention_and_stationarity': bool(grip_success and enough and within),
            'scope': 'Common recorded native TCP pose endpoint, not calibrated hardware stationarity.'}


def _corrections(policy):
    retention = (policy or {}).get('retention') or {}
    records = retention.get('hold_records', [])
    flags, refusals = {}, {}
    active, desired, axis_count = [], [], 0
    for record in records:
        for key, value in record.items():
            if type(value) is bool and ('saturat' in key or 'limit' in key or 'refus' in key):
                flags[key] = flags.get(key, 0)+int(value)
        if record.get('refused_bounds') is not None:
            reason = record['refused_bounds']
            refusals[reason] = refusals.get(reason, 0)+1
        axis_count += len(record.get('correction_rate_limited_axes') or [])
        for key, destination in (('correction_rad', active), ('desired_correction_rad', desired)):
            if record.get(key) is not None:
                vector = np.asarray(record[key], dtype=float)
                if vector.shape != (6,) or not np.isfinite(vector).all():
                    raise ValueError('Malformed correction diagnostic.')
                destination.append(vector)
    return {'hold_record_count': len(records), 'flag_counts': flags,
            'bound_refusal_counts': refusals, 'rate_limited_axis_occurrences': axis_count,
            'maximum_abs_active_correction_rad': float(np.max(np.abs(active))) if active else None,
            'maximum_abs_desired_correction_rad': float(np.max(np.abs(desired))) if desired else None,
            'input_refusals': deepcopy(retention.get('input_refusals', [])),
            'scope': 'Includes recorded final diagnostic checks; only command_issued rows represent issued controls.'}


def evaluate(result, arrays, metadata, protocol):
    """Retain invalid/refused attempts, with their primary endpoint false."""
    from .probe_evidence import _arrays
    base = {'status': 'refused', 'primary': {'required_s': HOLD_S, 'complete_5s': False,
            'native_pause_success': False, 'success': False, 'controller_retained_at_5s': False,
            'combined_qualified_pause': False, 'reason': 'invalid_or_incomplete_attempt'},
            'first_admission': None, 'probes': [], 'final': {'controller_qualified': False,
            'native_qualified': False, 'both_qualified': False, 'evidence_admissible': False},
            'stationarity': {'complete_5s': False, 'tcp_pose_within_bounds': False,
                             'grip_retention_and_stationarity': False},
            'decomposition': None, 'corrections': _corrections(result.get('policy')),
            'physical_robot_ready': False}
    try:
        data, array_digest = _arrays(arrays, metadata)
    except (ValueError, TypeError, KeyError) as exc:
        base['reason'] = 'native_sidecar_refused: '+str(exc)
        return base
    metrics, policy = result['metrics'], result.get('policy')
    complete = (len(result['actions']) == 1000 and len(result['trajectory']) == 1001
                and len(data['t_seconds']) == 20000)
    critical = int(np.count_nonzero(data['unexpected_contact_count']))
    max_force = float(np.max(data['pad_normal_forces_n']))
    native_valid = (result['valid'] and result['error'] is None and metrics.get('valid') is True
                    and not metrics.get('warnings') and complete
                    and critical == 0 and critical == metrics.get('unexpected_contact_native_ticks'))
    base['native'] = {'arrays_sha256': array_digest, 'rows': len(data['t_seconds']), 'complete': complete,
                      'valid': bool(native_valid), 'warnings': metrics.get('warnings'),
                      'unexpected_contact_native_ticks': critical, 'maximum_per_finger_force_n': max_force}
    if not native_valid or policy is None:
        return base
    times, by_probe = data['t_seconds'], {}
    for probe in policy['probes']:
        condition, streak, qualified, rise, drift = probe_series(data, probe, protocol['force_ceiling_n'])
        by_probe[probe['probe_index']] = (probe, condition, streak, qualified, rise, drift)
        events = [event for event in policy['events'] if event['type'] == 'evidence_qualified'
                  and event['probe_index'] == probe['probe_index']]
        claims = [{'t_seconds': event['t_seconds'], 'native_prefix_qualified': bool(qualified[_index(event['t_seconds'], times)])}
                  for event in events]
        base['probes'].append({'probe_index': probe['probe_index'], 'start_time_s': probe['start_time_s'],
                               'disposition': probe['disposition'], 'claims': claims,
                               'final_native_qualified_from_origin': bool(qualified[-1])})
    final_qualified = bool(by_probe[policy['probes'][-1]['probe_index']][3][-1]) if policy['probes'] else False
    base['final'].update(controller_qualified=bool(policy['admitted']), native_qualified=final_qualified,
                         both_qualified=bool(policy['admitted'] and final_qualified))
    try:
        base['receipt'] = prior._evaluate(result, data, metadata, protocol)
        base['final']['evidence_admissible'] = bool((base['receipt'].get('verification') or {}).get('admissible'))
    except (ValueError, TypeError, KeyError) as exc:
        base['receipt'] = {'status': 'audit_refused', 'reason': str(exc)[:300]}
    admissions = [event for event in policy['events'] if event['type'] == 'evidence_qualified']
    base['status'] = 'inspected'
    if not admissions:
        base['primary']['reason'] = 'no_admission'
        return base
    event = admissions[0]
    probe, condition, streak, qualified, rise, drift = by_probe[event['probe_index']]
    start = _index(event['t_seconds'], times)
    target = event['t_seconds']+HOLD_S
    enough = target <= times[-1]+1e-8
    end = _index(target, times) if enough else len(times)-1
    correct = bool(qualified[start])
    continuous = bool(np.all(condition[start:end+1]))
    # Every actual hold row must respect the unchanged synthetic ceiling,
    # including rows after the primary short-pause endpoint.
    ceiling_ok = bool(np.all(data['pad_normal_forces_n'][start:] <= protocol['force_ceiling_n']))
    checks = [check for check in policy['qualification_checks'] if abs(check['t_seconds']-target) <= 1e-8
              and check['probe_index'] == event['probe_index']]
    retained = enough and len(checks) == 1 and checks[0]['qualified'] is True
    success = correct and enough and continuous and ceiling_ok
    reason = ('false_initial_admission' if not correct else 'insufficient_followup' if not enough else
              'native_predicate_lost' if not continuous else 'force_ceiling_exceeded' if not ceiling_ok else 'success')
    base['first_admission'] = {'time_s': event['t_seconds'], 'probe_index': event['probe_index'],
                               'native_prefix_qualified': correct, 'native_qualifying_window_s': int(streak[start])*.001}
    failures = np.flatnonzero(~condition[start:end+1])
    base['primary'] = {'required_s': HOLD_S, 'target_time_s': target,
                       'followup_available_s': float(times[-1]-event['t_seconds']), 'complete_5s': bool(enough),
                       'native_pause_success': bool(success), 'success': bool(success),
                       'controller_retained_at_5s': bool(retained),
                       'combined_qualified_pause': bool(success and retained),
                       'all_native_hold_forces_within_ceiling': ceiling_ok,
                       'first_native_predicate_failure_s': float(times[start+int(failures[0])]) if failures.size else None,
                       'reason': reason}
    base['decomposition'] = _decomposition(data, start, end)
    base['stationarity'] = _stationarity(data, start, end, enough, success)
    return base


def _compact_policy(policy):
    if policy is None:
        return None
    result = {key: deepcopy(value) for key, value in policy.items()
              if key not in ('requested_controls', 'qualification_checks')}
    if 'retention' in result:
        result['retention'].pop('hold_records', None)
    return result


def aggregates(rows):
    result = {}
    for variant in VARIANTS:
        selected = [row for row in rows if row['variant'] == variant]
        admissions = [row['evaluation']['first_admission'] for row in selected if row['evaluation']['first_admission'] is not None]
        result[variant] = {'attempts': len(selected), 'valid': sum(row['valid'] for row in selected),
                           'admissions': len(admissions),
                           'false_initial_admissions': sum(not row['native_prefix_qualified'] for row in admissions),
                           'native_5s_pause_success': sum(row['evaluation']['primary']['success'] for row in selected),
                           'controller_retained_at_5s': sum(row['evaluation']['primary']['controller_retained_at_5s'] for row in selected),
                           'combined_qualified_5s_pause': sum(row['evaluation']['primary']['combined_qualified_pause'] for row in selected),
                           'grip_retention_and_stationarity': sum(row['evaluation']['stationarity']['grip_retention_and_stationarity'] for row in selected),
                           'final_both_qualified': sum(row['evaluation']['final']['both_qualified'] for row in selected)}
    return result


def run(output, assets):
    output = Path(output)
    protocol = read(output/'protocol.json')
    if encoded(protocol) != encoded(make_protocol(assets)):
        raise ValueError('Frozen protocol/source/runtime/assets changed; create a new study.')
    write(output/'started.json', {'protocol_sha256': sha256((output/'protocol.json').read_bytes()).hexdigest(),
                                 'started_at_utc': datetime.now(timezone.utc).isoformat()})
    rows = []
    for case, variant in roster(protocol):
        if prior.sources() != protocol['source_sha256']:
            raise ValueError('Source changed during the fixed study.')
        folder = output/(case['id']+'-'+variant); folder.mkdir()
        write(folder/'started.json', {'scene_id': case['id'], 'variant': variant,
                                      'protocol_sha256': sha256((output/'protocol.json').read_bytes()).hexdigest()})
        result, arrays, metadata = _attempt(assets, case, variant)
        (folder/'episode.json.gz').write_bytes(gzip.compress(encoded(result), mtime=0))
        with (folder/'native.npz').open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        write(folder/'native.json', metadata)
        try:
            evaluation = evaluate(result, arrays, metadata, protocol)
        except Exception as exc:
            evaluation = {'status': 'audit_refused', 'reason': type(exc).__name__+': '+str(exc)[:300],
                           'primary': {'success': False, 'controller_retained_at_5s': False,
                                       'combined_qualified_pause': False},
                           'first_admission': None, 'final': {'both_qualified': False}, 'physical_robot_ready': False}
            evaluation['stationarity'] = {'grip_retention_and_stationarity': False}
        write(folder/'evaluation.json', evaluation)
        files = {path.name: {'bytes': path.stat().st_size, 'sha256': sha256(path.read_bytes()).hexdigest()}
                 for path in sorted(folder.iterdir())}
        row = {'id': result['id'], 'scene_id': case['id'], 'variant': variant,
               'mass_kg': case['config']['mass_kg'], 'friction': case['config']['friction'],
               'valid': result['valid'], 'error': result['error'], 'metrics': result['metrics'],
               'policy': _compact_policy(result.get('policy')), 'timing': result['timing']['summaries'],
               'evaluation': evaluation, 'files': files}
        rows.append(row)
        write(folder/'summary.json', row)
        print(json.dumps({'attempt': result['id'], 'valid': result['valid'], 'primary': evaluation['primary']['success']}), flush=True)
        if prior.sources() != protocol['source_sha256']:
            raise ValueError('Source changed during an attempt; all evidence remains retained.')
    if prior.runtime() != protocol['runtime']:
        raise ValueError('Runtime changed during the study.')
    report = {'schema': SCHEMA, 'scope': SCOPE, 'planned_attempts': 36, 'attempted': len(rows),
              'valid_attempts': sum(row['valid'] for row in rows), 'rows': rows, 'aggregates': aggregates(rows),
              'physical_robot_ready': False, 'source_sha256': protocol['source_sha256'],
              'previous_learned_heldout_result': '5/12 unchanged. No learning or full-placement evaluation in this study.'}
    write(output/'report.json', report)
    return report


def _read_native(path):
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if (len(entries) != 22 or len({entry.filename for entry in entries}) != 22
                or sum(entry.file_size for entry in entries) > MAX_FILE
                or any(entry.file_size > 2*1024*1024 or '/' in entry.filename or not entry.filename.endswith('.npy') for entry in entries)):
            raise ValueError('Native NPZ exceeds the fixed numeric contract.')
    with np.load(path, allow_pickle=False) as native:
        return {name: native[name] for name in native.files}


def _controls(episode, arrays):
    actions, trajectory = episode['actions'], episode['trajectory']
    if len(actions) != 1000 or len(trajectory) != 1001 or len(arrays['t_seconds']) != 20000:
        raise ValueError('Complete control reconstruction requires 1000 actions/20000 native rows.')
    requested = episode['policy']['requested_controls']
    if len(requested) != 1000:
        raise ValueError('Missing policy control records.')
    control = np.array(trajectory[0]['native_actuator_controls'], dtype=float)
    maximum = np.array([episode['config']['joint_target_rate_rad_s']*.001]*6+[episode['config']['jaw_target_rate_m_s']*.001])
    for i, (action, request) in enumerate(zip(actions, requested)):
        if (abs(action['t_seconds']-i*.02) > 1e-8 or request['index'] != i
                or any(action[key] != request[key] for key in ('t_seconds', 'joint_targets_rad', 'gripper_position_m'))
                or trajectory[i]['command'] != action
                or trajectory[i]['issued_joint_targets_rad'] != action['joint_targets_rad']
                or trajectory[i]['issued_gripper_position_m'] != action['gripper_position_m']):
            raise ValueError('Issued action/boundary/policy records disagree.')
        target = np.array(action['joint_targets_rad']+[action['gripper_position_m']])
        for tick in range(i*20, (i+1)*20):
            control += np.clip(target-control, -maximum, maximum)
            actual = np.r_[arrays['arm_actuator_controls_rad'][tick], arrays['applied_jaw_target_m'][tick]]
            if (not np.allclose(control, actual, rtol=0, atol=1e-12)
                    or arrays['requested_jaw_target_m'][tick] != action['gripper_position_m']):
                raise ValueError('Native applied targets differ from unchanged slew law.')
        for field in ('object_position_m', 'joint_position_rad', 'tcp_position_m'):
            if not np.allclose(trajectory[i+1][field], arrays[field][(i+1)*20-1], rtol=0, atol=1e-9):
                raise ValueError('Boundary state differs from native record.')
    if any(trajectory[-1][key] is not None for key in ('command', 'issued_joint_targets_rad', 'issued_gripper_position_m')):
        raise ValueError('Terminal observation contains an unexecuted command.')


def _treatment_boundary(policy):
    events = (policy or {}).get('events', [])
    admissions = [event for event in events if event['type'] == 'evidence_qualified']
    admission = min((event['t_seconds'] for event in admissions), default=None)
    candidates = []
    for event in events:
        if event['type'] == 'evidence_qualified':
            reason = 'admission'
        elif (event['type'] == 'hold_invalidated' and event.get('reason') in BOUND_REFUSALS
              and (admission is None or event['t_seconds'] <= admission+1e-8)):
            reason = 'pre_admission_bound_refusal'
        else:
            continue
        candidates.append({'time_s': event['t_seconds'], 'reason': reason, 'detail': event.get('reason')})
    return min(candidates, key=lambda row: row['time_s']) if candidates else None


def _common_cutoff(rows):
    boundaries = []
    for row in rows:
        boundary = _treatment_boundary(row.get('policy'))
        if boundary is not None:
            boundaries.append({'variant': row['variant'], **boundary})
    boundaries.sort(key=lambda row: row['variant'])
    timestamp = min((row['time_s'] for row in boundaries), default=None)
    reasons = sorted({row['reason'] for row in boundaries if abs(row['time_s']-timestamp) <= 1e-8}) if timestamp is not None else []
    return {'time_s': timestamp, 'reason': '+'.join(reasons) if reasons else 'full_run_no_treatment',
            'treatment_boundaries': boundaries}


def _prefix(episode, arrays, *, cutoff=None):
    if cutoff is None:
        cutoff = _common_cutoff([episode])
    timestamp = cutoff['time_s']
    count = _index(timestamp, arrays['t_seconds'])+1 if timestamp is not None else len(arrays['t_seconds'])
    digest_native = sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name][:count])
        digest_native.update(encoded({'name': name, 'shape': list(value.shape), 'dtype': str(value.dtype)}))
        digest_native.update(value.tobytes())
    actions = [row for row in episode['actions'] if timestamp is None or row['t_seconds'] < timestamp-1e-8]
    return {'cutoff_time_s': timestamp, 'cutoff_reason': cutoff['reason'],
            'native_rows': count, 'native_sha256': digest_native.hexdigest(),
            'commands': len(actions), 'commands_sha256': digest(actions)}


def inspect(output):
    from .piper_grip_benchmark import decode_attempt
    from .probe_evidence import _arrays
    output = Path(output)
    protocol_path = output/'protocol.json'
    if protocol_path.is_symlink() or not protocol_path.is_file() or protocol_path.stat().st_size > 16*1024*1024:
        raise ValueError('Expected a bounded plain frozen protocol.')
    with protocol_path.open('rb') as stream:
        raw_protocol = stream.read(16*1024*1024+1)
    if (len(raw_protocol) > 16*1024*1024
            or sha256(raw_protocol).hexdigest() != read(output/'frozen.json')['protocol_sha256']):
        raise ValueError('Frozen protocol digest changed.')
    protocol, report = json.loads(raw_protocol), read(output/'report.json')
    if prior.sources() != protocol['source_sha256'] or report['source_sha256'] != protocol['source_sha256']:
        raise ValueError('Inspect with the frozen implementation.')
    planned = roster(protocol)
    ids = [case['id']+'-'+variant for case, variant in planned]
    if ([row['id'] for row in report['rows']] != ids or len(ids) != 36
            or report['attempted'] != 36 or protocol['planned_attempts'] != 36):
        raise ValueError('Attempt roster/denominator differs from frozen protocol.')
    checked, prefixes = [], {}
    cutoffs = {case['id']: _common_cutoff([row for row in report['rows'] if row['scene_id'] == case['id']])
               for case in protocol['cases']}
    for row, (case, variant) in zip(report['rows'], planned):
        folder = output/row['id']
        if set(row['files']) != {'started.json', 'episode.json.gz', 'native.npz', 'native.json', 'evaluation.json'}:
            raise ValueError('Unexpected attempt artifact roster.')
        for name, expected in row['files'].items():
            path = folder/name
            if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_FILE:
                raise ValueError('Expected bounded plain evidence file.')
            raw = path.read_bytes()
            if {'bytes': len(raw), 'sha256': sha256(raw).hexdigest()} != expected:
                raise ValueError('Frozen artifact digest changed.')
        episode = json.loads(decode_attempt((folder/'episode.json.gz').read_bytes()))
        if (episode['id'] != row['id'] or episode['variant'] != variant or episode['seed'] != case['seed']
                or episode['config'] != case['config'] or episode['metrics'] != row['metrics'] or episode['valid'] != row['valid']
                or encoded(_compact_policy(episode.get('policy'))) != encoded(row['policy'])):
            raise ValueError('Episode identity/config/metrics disagree.')
        arrays, metadata = _read_native(folder/'native.npz'), read(folder/'native.json')
        reevaluated = evaluate(episode, arrays, metadata, protocol)
        if encoded(reevaluated) != encoded(row['evaluation']) or encoded(read(folder/'evaluation.json')) != encoded(reevaluated):
            raise ValueError('Recomputed endpoint/receipt differs.')
        native_validated, reconstruction, reason = False, False, None
        try:
            _arrays(arrays, metadata)
            native_validated = True
            if episode['valid'] and episode['error'] is None:
                _controls(episode, arrays)
                reconstruction = True
                prefixes.setdefault(case['id'], []).append((variant, _prefix(episode, arrays, cutoff=cutoffs[case['id']])))
        except (ValueError, TypeError, KeyError) as exc:
            reason = str(exc)
            if episode['valid'] and episode['error'] is None:
                raise ValueError('Valid attempt failed native readback: '+reason) from exc
        checked.append({'id': row['id'], 'native_rows': len(arrays.get('t_seconds', [])),
                        'native_validated': native_validated, 'controls_recomputed': reconstruction,
                        'retained_refusal': not reconstruction, 'reason': reason, 'evaluation_recomputed': True})
    prefix_checks = []
    for case in protocol['cases']:
        group = prefixes.get(case['id'], [])
        if len(group) == 3:
            base = group[0][1]
            if any(encoded(value) != encoded(base) for _, value in group[1:]):
                raise ValueError('Acquisition prefixes differ before any hold treatment.')
            prefix_checks.append({'scene_id': case['id'], 'verified': True, **base,
                                  'treatment_boundaries': cutoffs[case['id']]['treatment_boundaries']})
        else:
            prefix_checks.append({'scene_id': case['id'], 'verified': False, 'reason': 'invalid_attempt_prevents_complete_triplet'})
    if encoded(aggregates(report['rows'])) != encoded(report['aggregates']):
        raise ValueError('Aggregate counts differ from retained rows.')
    return {'schema': 'dvidia.piper-retention-inspection/v1', 'attempts': len(checked),
            'native_rows': sum(row['native_rows'] for row in checked), 'rows': checked,
            'acquisition_prefixes': prefix_checks, 'aggregates': report['aggregates'], 'physical_robot_ready': False,
            'scope': 'Read-only recorded consistency and causal native predicate reconstruction; hashes do not authenticate external data or hardware.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'run'):
        command = sub.add_parser(name)
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--assets', type=Path, required=True)
    command = sub.add_parser('inspect'); command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = inspect(args.output) if args.command == 'inspect' else globals()[args.command](args.output, args.assets)
        print(json.dumps(result if args.command == 'inspect' else {'schema': SCHEMA, 'planned_attempts': 36,
                                                                  'physical_robot_ready': False}, indent=2))
        return 0
    except (ValueError, OSError, KeyError, ImportError) as exc:
        print('Retention study refused: '+str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

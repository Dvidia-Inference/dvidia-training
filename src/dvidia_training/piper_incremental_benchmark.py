"""One frozen incremental PiPER hold candidate against three unchanged controls.

Every scheduled attempt remains in the denominator. The primary task endpoint
requires native grip AND TCP stationarity; grip alone and controller qualification
remain separately inspectable. No physics runs at import time.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from . import piper_retention_benchmark as legacy
from . import piper_probe_benchmark as prior

SCHEMA = 'dvidia.piper-incremental-hold-study/v1'
INSPECTION_SCHEMA = 'dvidia.piper-incremental-hold-inspection/v1'
VARIANTS = (*legacy.VARIANTS, 'incremental')
ORDERS = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))
PLANNED_ATTEMPTS = 32
HOLD_S = legacy.HOLD_S
FORCE_CEILING_N = legacy.FORCE_CEILING_N
TCP_POSITION_CEILING_M = legacy.TCP_POSITION_CEILING_M
TCP_ORIENTATION_CEILING_RAD = legacy.TCP_ORIENTATION_CEILING_RAD
BOUND_REFUSALS = legacy.BOUND_REFUSALS | {'measured_joint_margin_exhausted'}
MAX_FILE = legacy.MAX_FILE
SCOPE = ('One authored bounded incremental pose-hold candidate and three frozen v0.2 controls. '
         'Eight fresh finite synthetic development scenes with privileged simulator state; '
         'no learning, full placement, physical sensing or hardware qualification.')
encoded, digest, write, read = prior.encoded, prior.digest, prior.write, prior.read
roster, _timing = legacy.roster, legacy._timing
_index, probe_series = legacy._index, legacy.probe_series
_read_native, _controls, _compact_policy = legacy._read_native, legacy._controls, legacy._compact_policy


def case_definitions():
    """The preregistered full 2 x 2 x 2 factorial with fixed Williams orders."""
    from .piper_task import PiperTaskConfig
    base, cases = PiperTaskConfig(), []
    for mass in (.0475, .0525):
        for friction in (.78, .52):
            for layout, start in enumerate(((.304, -.024), (.308, -.035))):
                index = len(cases)
                config = replace(base, mass_kg=mass, friction=friction, object_position_m=(*start, .082))
                cases.append({'id': f'm{round(mass*10000):04}-f{round(friction*100):02}-p{layout}',
                              'seed': 311200+index, 'config': asdict(config),
                              'variant_order': [VARIANTS[i] for i in ORDERS[index % len(ORDERS)]]})
    return cases


def _source_commit():
    return subprocess.check_output(['git', 'rev-parse', '--verify', 'HEAD'],
                                   cwd=Path(__file__).resolve().parents[2], text=True).strip()


def make_protocol(assets):
    from .piper_incremental_hold import incremental_hold_contract
    protocol = legacy.make_protocol(assets)
    grip = deepcopy(protocol['primary_endpoint'])
    stationary = deepcopy(protocol['secondary_endpoint'])
    if 'definition' in stationary:
        stationary['definition'] = stationary['definition'].replace('all three controllers', 'all four controllers')
    protocol.update(schema=SCHEMA, scope=SCOPE, cases=case_definitions(), variants=list(VARIANTS),
                    planned_attempts=PLANNED_ATTEMPTS, source_commit=_source_commit(),
                    candidate_controller_contract=incremental_hold_contract(),
                    baseline_source_commit='bc106d6f66367f8896787a2303aa0a124c69ea56',
                    grip_only_endpoint=grip)
    protocol['primary_endpoint'] = {
        **grip, 'name': 'grip_retention_and_tcp_stationarity',
        'position_ceiling_m': TCP_POSITION_CEILING_M,
        'orientation_ceiling_rad': TCP_ORIENTATION_CEILING_RAD,
        'definition': grip['definition']+' Additionally every native TCP pose from admission through five seconds must stay within 4 mm and 0.05 rad of its admission pose.',
        'predicate': grip['predicate']+' TCP displacement <=0.004 m and orientation error <=0.05 rad with epsilon 1e-12, identically for all four variants.',
    }
    protocol['secondary_endpoint'] = {
        'grip_only': grip, 'stationarity_details': stationary,
        'active_controller_conjunction': 'Primary native task success AND controller qualified at the five-second endpoint. Refusal is terminal, so endpoint qualification requires uninterrupted controller admission.',
        'maintained_claims': 'At every post-admission boundary claimed qualified, check the complete native history from admission through that boundary. Count grip and pose contradictions separately; false initial admission also invalidates subsequent grip claims.',
    }
    protocol['order'] = ('Four Williams orders repeated twice: [0,1,3,2], [1,2,0,3], [2,3,1,0], [3,0,2,1]. '
                         'Each variant occupies every position twice. One attempt per scene/variant; no resume, outcome-based retry, candidate selection or post-result tuning.')
    return protocol


def _frozen_protocol(output):
    path = Path(output)/'protocol.json'
    protocol = read(path)
    if sha256(path.read_bytes()).hexdigest() != read(Path(output)/'frozen.json')['protocol_sha256']:
        raise ValueError('Frozen protocol digest changed.')
    return protocol


def _failed_evaluation(reason):
    primary = {'required_s': HOLD_S, 'complete_5s': False, 'native_pause_success': False,
               'success': False, 'controller_retained_at_5s': False,
               'combined_qualified_pause': False, 'reason': reason,
               'first_tcp_pose_failure_s': None}
    return {'status': 'audit_refused', 'reason': reason, 'primary': primary,
            'grip_only': deepcopy(primary), 'first_admission': None,
            'final': {'controller_qualified': False, 'native_qualified': False,
                      'both_qualified': False, 'evidence_admissible': False},
            'stationarity': {'complete_5s': False, 'tcp_pose_within_bounds': False,
                             'grip_retention_and_stationarity': False},
            'claims': {'maintained_checks': 0, 'false_maintained_grip': 0, 'false_maintained_pose': 0},
            'physical_robot_ready': False}


def _json_record(result):
    """Retain invalid values explicitly without admitting nonfinite JSON numbers."""
    invalid = []

    def plain(value, path):
        if isinstance(value, (float, np.floating)) and not math.isfinite(value):
            label = 'nan' if math.isnan(value) else 'positive_infinity' if value > 0 else 'negative_infinity'
            invalid.append({'path': path, 'value': label})
            return {'invalid_nonfinite_number': label}
        if isinstance(value, dict):
            return {key: plain(item, path+'/'+str(key)) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(item, path+'/'+str(index)) for index, item in enumerate(value)]
        if isinstance(value, np.ndarray):
            return plain(value.tolist(), path)
        if isinstance(value, np.generic):
            return value.item()
        return value

    recorded = plain(result, '')
    if invalid:
        recorded['valid'] = False
        recorded['json_invalid_numbers'] = invalid
        if recorded.get('error') is None:
            recorded['error'] = {'type': 'NonfiniteEvidence',
                                 'message': 'Nonfinite JSON observations retained as explicit invalid-number markers.'}
    return recorded


def _unavailable_native(arrays=None):
    """A truthful zero-row/partial sidecar when native telemetry cannot export."""
    from .probe_evidence import FIELDS
    if arrays is None:
        arrays = {name: np.empty((0,) if width == 1 else (0, width), dtype=np.float64)
                  for name, width in FIELDS}
    metadata = {'schema': 'dvidia.piper-feedback-native/v1', 'native_timestep_s': .001,
                'rows': len(arrays['t_seconds']), 'fields': [{'name': name, 'width': width} for name, width in FIELDS],
                'side_order': ['left', 'right'], 'physical_robot_ready': False,
                'recording_status': 'unavailable_or_incomplete_after_exception'}
    return arrays, metadata


def _claims_and_pose_failure(result, arrays, evaluation):
    claims = {'maintained_checks': 0, 'false_maintained_grip': 0, 'false_maintained_pose': 0,
              'scope': 'Claimed qualified boundaries after admission, compared with every native row since admission; grip includes correct initial prefix. Counts are diagnostic, not independent trials.'}
    admission = evaluation['first_admission']
    if admission is None:
        return claims, None
    times = arrays['t_seconds']
    start = _index(admission['time_s'], times)
    position = np.linalg.norm(arrays['tcp_position_m'][start:]-arrays['tcp_position_m'][start], axis=1)
    angles = 2*np.arccos(np.clip(np.abs(arrays['tcp_quaternion_wxyz'][start:] @ arrays['tcp_quaternion_wxyz'][start]), 0., 1.))
    pose = (position <= TCP_POSITION_CEILING_M+1e-12) & (angles <= TCP_ORIENTATION_CEILING_RAD+1e-12)
    policy = result['policy']
    probe = next(row for row in policy['probes'] if row['probe_index'] == admission['probe_index'])
    condition = probe_series(arrays, probe)[0]
    grip = np.logical_and.accumulate(condition[start:]) & admission['native_prefix_qualified']
    pose_continuous = np.logical_and.accumulate(pose)
    for check in policy['qualification_checks']:
        if (check['probe_index'] == admission['probe_index'] and check['qualified'] is True
                and check['t_seconds'] > admission['time_s']+1e-8):
            index = _index(check['t_seconds'], times)-start
            claims['maintained_checks'] += 1
            claims['false_maintained_grip'] += int(not grip[index])
            claims['false_maintained_pose'] += int(not pose_continuous[index])
    in_window = times[start:] <= admission['time_s']+HOLD_S+1e-8
    failures = np.flatnonzero(~pose & in_window)
    return claims, float(times[start+int(failures[0])]) if failures.size else None


def evaluate(result, arrays, metadata, protocol):
    """Reuse the frozen native evaluator; promote its stationary-grip endpoint."""
    try:
        evaluation = legacy.evaluate(result, arrays, metadata, protocol)
        evaluation['grip_only'] = deepcopy(evaluation['primary'])
        primary, stationarity = evaluation['primary'], evaluation['stationarity']
        primary['success'] = bool(stationarity['grip_retention_and_stationarity'])
        primary['combined_qualified_pause'] = bool(primary['success'] and primary['controller_retained_at_5s'])
        if evaluation['grip_only']['success'] and not primary['success']:
            primary['reason'] = 'tcp_stationarity_lost'
        claims, failure = _claims_and_pose_failure(result, arrays, evaluation)
        evaluation['claims'] = claims
        primary['first_tcp_pose_failure_s'] = failure
        records = ((result.get('policy') or {}).get('retention') or {}).get('hold_records', [])
        increments, rates = [], []
        for row in records:
            if row.get('pose_error_increment_rad') is not None:
                increments.append(np.asarray(row['pose_error_increment_rad'], dtype=float))
            if row.get('command_issued') and row.get('correction_update_accepted'):
                rates.append((np.asarray(row['accumulated_correction_rad'])-np.asarray(row['correction_before_update_rad']))/.02)
        evaluation['corrections'].update(
            maximum_abs_pose_error_increment_rad=float(np.max(np.abs(increments))) if increments else None,
            maximum_abs_accepted_correction_rate_rad_s=float(np.max(np.abs(rates))) if rates else None)
        return evaluation
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        return _failed_evaluation(type(exc).__name__+': '+str(exc)[:300])


def aggregates(rows):
    result = {}
    for variant in VARIANTS:
        selected = [row for row in rows if row['variant'] == variant]
        admissions = [row['evaluation']['first_admission'] for row in selected if row['evaluation']['first_admission'] is not None]
        result[variant] = {
            'attempts': len(selected), 'valid': sum(row['valid'] for row in selected),
            'admissions': len(admissions),
            'false_initial_admissions': sum(not row['native_prefix_qualified'] for row in admissions),
            'native_5s_pause_success': sum(row['evaluation']['primary']['native_pause_success'] for row in selected),
            'native_5s_stationary_success': sum(row['evaluation']['primary']['success'] for row in selected),
            'controller_retained_at_5s': sum(row['evaluation']['primary']['controller_retained_at_5s'] for row in selected),
            'combined_qualified_5s_pause': sum(row['evaluation']['primary']['combined_qualified_pause'] for row in selected),
            'grip_retention_and_stationarity': sum(row['evaluation']['stationarity']['grip_retention_and_stationarity'] for row in selected),
            'false_maintained_grip': sum(row['evaluation']['claims']['false_maintained_grip'] for row in selected),
            'false_maintained_pose': sum(row['evaluation']['claims']['false_maintained_pose'] for row in selected),
            'final_both_qualified': sum(row['evaluation']['final']['both_qualified'] for row in selected),
        }
    return result


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

def _attempt(assets, case, variant):
    from .piper_task import PiperTaskConfig
    from .piper_feedback_env import FeedbackPiperTaskEnv
    from .piper_retention import RetentionPolicy
    from .piper_incremental_hold import IncrementalHoldPolicy
    from .reference_sim import model_digest
    env = None
    result = {'schema': SCHEMA, 'id': case['id']+'-'+variant, 'scene_id': case['id'], 'variant': variant,
              'seed': case['seed'], 'config': case['config'], 'valid': False, 'physical_robot_ready': False,
              'trajectory': [], 'actions': [], 'error': None, 'metrics': {}, 'policy': None}
    timers = {name: [] for name in ('policy_wall', 'policy_cpu', 'step_wall', 'step_cpu')}
    policy = None
    began = time.perf_counter()
    try:
        env = FeedbackPiperTaskEnv(assets, PiperTaskConfig(**case['config']))
        start = time.perf_counter()
        obs, info = env.reset(case['seed'])
        result['setup_reset_wall_s'] = time.perf_counter()-start
        result.update(initial_object_position_m=env.initial_object_position.tolist(),
                      compiled_model_sha256=model_digest(env.model),
                      native_force_limits=env.model.actuator_forcerange.tolist())
        policy = IncrementalHoldPolicy(env) if variant == 'incremental' else RetentionPolicy(env, variant=variant)
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
        if env is not None and env.ready:
            try:
                result['metrics'] = env.info()
            except Exception as info_exc:
                result['metrics_error'] = {'type': type(info_exc).__name__, 'message': str(info_exc)[:300]}
        if policy is not None:
            try:
                result['policy'] = policy.summary()
            except Exception as summary_exc:
                result['policy_summary_error'] = {'type': type(summary_exc).__name__,
                                                   'message': str(summary_exc)[:300]}
    arrays, metadata = _unavailable_native()
    if env is not None:
        try:
            arrays = env.native_arrays()
            metadata = env.native_metadata()
        except Exception as native_exc:
            result['valid'] = False
            result['native_recording_error'] = {'type': type(native_exc).__name__, 'message': str(native_exc)[:300]}
            if result['error'] is None:
                result['error'] = deepcopy(result['native_recording_error'])
            arrays, metadata = _unavailable_native(arrays)
    result['timing'] = {'attempt_wall_s': time.perf_counter()-began,
                        'summaries': {name: _timing(values) for name, values in timers.items()}, 'samples': timers}
    return result, arrays, metadata

def run(output, assets):
    output = Path(output)
    protocol = _frozen_protocol(output)
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
        result = _json_record(result)
        (folder/'episode.json.gz').write_bytes(gzip.compress(encoded(result), mtime=0))
        with (folder/'native.npz').open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        write(folder/'native.json', metadata)
        try:
            evaluation = evaluate(result, arrays, metadata, protocol)
        except Exception as exc:
            evaluation = _failed_evaluation(type(exc).__name__+': '+str(exc)[:300])
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
    report = {'schema': SCHEMA, 'scope': SCOPE, 'planned_attempts': PLANNED_ATTEMPTS, 'attempted': len(rows),
              'valid_attempts': sum(row['valid'] for row in rows), 'rows': rows, 'aggregates': aggregates(rows),
              'physical_robot_ready': False, 'source_sha256': protocol['source_sha256'],
              'previous_learned_heldout_result': '5/12 unchanged. No learning or full-placement evaluation in this study.'}
    write(output/'report.json', report)
    return report

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
    return legacy._prefix(episode, arrays, cutoff=_common_cutoff([episode]) if cutoff is None else cutoff)


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
    if protocol['schema'] != SCHEMA or protocol['variants'] != list(VARIANTS) or encoded(protocol['cases']) != encoded(case_definitions()):
        raise ValueError('Frozen candidate roster differs from the authored protocol.')
    planned = roster(protocol)
    ids = [case['id']+'-'+variant for case, variant in planned]
    if ([row['id'] for row in report['rows']] != ids or len(ids) != PLANNED_ATTEMPTS
            or report['attempted'] != PLANNED_ATTEMPTS or protocol['planned_attempts'] != PLANNED_ATTEMPTS):
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
        if len(group) == 4:
            base = group[0][1]
            if any(encoded(value) != encoded(base) for _, value in group[1:]):
                raise ValueError('Acquisition prefixes differ before any hold treatment.')
            prefix_checks.append({'scene_id': case['id'], 'verified': True, **base,
                                  'treatment_boundaries': cutoffs[case['id']]['treatment_boundaries']})
        else:
            prefix_checks.append({'scene_id': case['id'], 'verified': False, 'reason': 'invalid_attempt_prevents_complete_four_variant_comparison'})
    if encoded(aggregates(report['rows'])) != encoded(report['aggregates']):
        raise ValueError('Aggregate counts differ from retained rows.')
    return {'schema': INSPECTION_SCHEMA, 'attempts': len(checked),
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
        print(json.dumps(result if args.command == 'inspect' else {'schema': SCHEMA, 'planned_attempts': PLANNED_ATTEMPTS,
                                                                  'physical_robot_ready': False}, indent=2))
        return 0
    except (ValueError, OSError, KeyError, ImportError) as exc:
        print('Incremental hold study refused: '+str(exc), file=sys.stderr)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())

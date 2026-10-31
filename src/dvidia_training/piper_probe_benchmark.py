"""Frozen local PiPER short-lift study; authored control and simulator evidence.

No learning, physical sensing, force calibration or hardware commands occur.
The two closed-loop controllers may produce different trajectories: this is not
an isolated classifier comparison. All scheduled attempts remain in the report.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import gzip
import io
import json
from pathlib import Path
import platform
import sys
import time
import zipfile

import numpy as np

SCHEMA = 'dvidia.piper-probe-study/v1'
VARIANTS = ('evidence', 'timer')
SCOPE = ('Authored short-lift/hold controllers with privileged native simulator state. '
         'No full placement, learned policy, camera tracking, calibrated physical sensing or hardware qualification.')


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def digest(value):
    return sha256(encoded(value)).hexdigest()


def write(path, value):
    with Path(path).open('xb') as stream:
        stream.write(encoded(value))


def read(path, limit=16*1024*1024):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError('Expected a bounded plain evidence file.')
    with path.open('rb') as stream:
        raw = stream.read(limit+1)
    if len(raw) > limit:
        raise ValueError('Evidence exceeds its byte limit.')
    return json.loads(raw)


def sources():
    root = Path(__file__).parent
    return {p.name: sha256(p.read_bytes()).hexdigest()
            for p in sorted([*root.glob('*.py'), *root.glob('*.lock.json')])}


def runtime():
    import mujoco
    return dict(python=platform.python_version(), numpy=np.__version__, mujoco=mujoco.__version__,
                system=platform.system(), machine=platform.machine(),
                executable_sha256=sha256(Path(sys.executable).resolve().read_bytes()).hexdigest())


def make_protocol(assets):
    from .piper_task import PiperTaskConfig
    from .robot_profiles import verify_assets, LOCK_PATH
    from .piper_probe import probe_contract
    provenance = verify_assets('piper', assets)
    base, cases = PiperTaskConfig(), []
    for mass in (.04, .06):
        for friction in (.8, .55, .15):
            for index, start in enumerate(((.30, -.03), (.309, -.04))):
                config = replace(base, mass_kg=mass, friction=friction,
                                 object_position_m=(*start, .082))
                cases.append(dict(id=f'm{round(mass*1000):02}-f{round(friction*100):02}-p{index}',
                                  seed=311000+len(cases), config=asdict(config)))
    protocol = dict(schema=SCHEMA, scope=SCOPE, physical_robot_ready=False, learned_policy=False,
                    cohort='development_method_check', cases=cases, variants=list(VARIANTS),
                    planned_attempts=24, source_sha256=sources(), runtime=runtime(),
                    asset_provenance=provenance, asset_lock_sha256=sha256(LOCK_PATH.read_bytes()).hexdigest(),
                    controller_contract=probe_contract(),
                    evidence='Independent reconstruction from every post-physics 1 ms native row; continuous 100 ms grip, not a caller passed flag.',
                    force_ceiling_n=10., force_ceiling_scope='Arbitrary synthetic admission ceiling; not a measured damage limit or force controller.',
                    comparison='Same scene and authored acquisition prefix, followed by two closed-loop probe/hold controllers. '
                               'Early holds, alarms and supported retries can change the trajectories. Each admission is judged '
                               'against its own native history, not as an isolated causal effect of a classifier.',
                    timing='Policy-call and complete control-step wall/CPU compute, including dense recording for the step. '
                           'Unpaced shared CPU; excludes physical sensors, transport, hardware actuation and evidence serialization.',
                    selection='No parameter selection after these development attempts. Each scene runs once per variant; '
                              'deterministic cells are not independent hardware trials or a held-out learning test.')
    return protocol


def prepare(root, assets):
    root = Path(root)
    if root.exists() or root.is_symlink():
        raise ValueError('Choose a fresh output directory; existing evidence is never overwritten.')
    protocol = make_protocol(assets)
    root.mkdir(parents=True)
    write(root/'protocol.json', protocol)
    write(root/'frozen.json', {'created_at_utc':datetime.now(timezone.utc).isoformat(),
                              'protocol_sha256':sha256((root/'protocol.json').read_bytes()).hexdigest()})
    (root/'sources').mkdir()
    for name in ('piper_probe.py', 'probe_evidence.py', 'piper_probe_benchmark.py'):
        (root/'sources'/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    return protocol


def _timing(values):
    if not values:
        return None
    return dict(samples=len(values), mean_s=float(np.mean(values)), p50_s=float(np.percentile(values, 50)),
                p95_s=float(np.percentile(values, 95)), maximum_s=float(max(values)),
                over_20ms=sum(x > .02 for x in values))


def _feasibility(case):
    from .task_feasibility import template
    value = template()
    value.update(object_width_interval_m=[.04, .04], gripper_max_opening_interval_m=[.07, .07],
                 gripper_min_closure_m=0., mass_interval_kg=[case['config']['mass_kg']]*2,
                 friction_interval=[case['config']['friction']]*2,
                 per_finger_normal_force_budget_n=10., pose_specific_payload_limit_kg=.10,
                 planned_upward_accel_m_s2=.1, safety_factor=1.5,
                 pose_uncertainty_m=0., allowed_pose_tolerance_m=.003,
                 calibration_id='synthetic_probe_v01', test_lift_passed=False)
    return value


def _context(result, protocol, probe, planning):
    from .probe_evidence import feasibility_binding
    info = result['metrics']
    return dict(schema='dvidia.probe-context/v1', run_id=result['id'], object_id='task_block',
                grasp_id=result['id']+'-probe-'+str(probe['probe_index']), probe_index=probe['probe_index'], robot_profile='piper',
                model_sha256=result['compiled_model_sha256'], source_sha256=digest(protocol['source_sha256']),
                asset_lock_sha256=protocol['asset_lock_sha256'], command_sha256=digest(result['actions']),
                pose_binding_sha256=digest({'config':result['config'], 'probe':probe}),
                feasibility_sha256=feasibility_binding(planning), calibration_id='synthetic_probe_v01',
                calibration_scope='simulation_assumptions_not_hardware',
                initial_object_position_m=result['initial_object_position_m'], probe_start_s=probe['start_time_s'],
                probe_origin_object_position_m=probe['object_origin_m'],
                probe_origin_tcp_local_offset_m=probe['local_offset_origin_m'],
                per_finger_force_ceiling_n=planning['per_finger_normal_force_budget_n'],
                native_status={'valid':info['valid'], 'warnings':info['warnings'],
                               'unexpected_contact_native_ticks':info['unexpected_contact_native_ticks'],
                               'simulation_time_s':info['simulation_time']})


def _attempt(assets, case, variant, protocol):
    from .piper_task import PiperTaskConfig
    from .piper_feedback_env import FeedbackPiperTaskEnv
    from .piper_probe import ProbeStopPolicy
    from .reference_sim import model_digest
    env = FeedbackPiperTaskEnv(assets, PiperTaskConfig(**case['config']))
    result = dict(schema=SCHEMA, id=case['id']+'-'+variant, scene_id=case['id'], variant=variant,
                  config=case['config'], valid=False, physical_robot_ready=False,
                  trajectory=[], actions=[], error=None, metrics={}, policy=None)
    timers = {name:[] for name in ('policy_wall', 'policy_cpu', 'step_wall', 'step_cpu')}
    began = time.perf_counter()
    try:
        obs, info = env.reset(case['seed'])
        result.update(initial_object_position_m=env.initial_object_position.tolist(),
                      compiled_model_sha256=model_digest(env.model),
                      native_force_limits=env.model.actuator_forcerange.tolist())
        policy = ProbeStopPolicy(env, variant=variant)
        result['trajectory'].append(obs)
        for _ in range(round(env.config.timeout_s/env.config.control_dt)):
            wall, cpu = time.perf_counter(), time.process_time()
            action = policy(obs)
            timers['policy_wall'].append(time.perf_counter()-wall)
            timers['policy_cpu'].append(time.process_time()-cpu)
            record = dict(t_seconds=obs['t_seconds'], **action)
            result['actions'].append(record)
            obs['command'] = record
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
        result['error'] = {'type':type(exc).__name__, 'message':str(exc)[:300]}
        if env.ready:
            result['metrics'] = env.info()
    result['timing'] = dict(attempt_wall_s=time.perf_counter()-began,
                           summaries={k:_timing(v) for k,v in timers.items()}, samples=timers)
    return result, env.native_arrays(), env.native_metadata()


def _evaluate(result, arrays, metadata, protocol):
    from .probe_evidence import derive_probe_evidence, verify_probe_evidence, assess_with_probe
    planning = _feasibility({'config':result['config']})
    probes = (result.get('policy') or {}).get('probes', [])
    if not result['valid'] or not probes:
        return dict(status='not_qualified', reason='invalid_attempt' if not result['valid'] else 'no_probe',
                    receipt=None, planning_input=planning, physical_robot_ready=False)
    probe = probes[-1]
    context = _context(result, protocol, probe, planning)
    claims = [dict(t_seconds=e['t_seconds'], qualified=True, probe_index=e['probe_index'])
              for e in result['policy']['events']
              if e['type'] in ('evidence_qualified', 'timer_admitted') and e['probe_index']==probe['probe_index']]
    # A historical admission cannot resurrect a controller's terminal refusal.
    terminal = result['policy']['qualification_checks'][-1]
    if terminal['probe_index'] == probe['probe_index']:
        if claims and abs(claims[-1]['t_seconds']-terminal['t_seconds']) < 1e-8:
            claims[-1] = terminal
        else:
            claims.append(terminal)
    receipt = derive_probe_evidence(arrays, metadata, context, controller_qualifications=claims)
    now = result['metrics']['simulation_time']
    verification = verify_probe_evidence(receipt, arrays, metadata, context, now_s=now)
    admission = assess_with_probe(planning, receipt, arrays, metadata, context, now_s=now)
    return dict(status='inspected', planning_input=planning, context=context, receipt=receipt,
                verification=verification, admission=admission, physical_robot_ready=False)


def inspect(root):
    """Re-read seals, applied slew-limited commands and probe receipts; no physics."""
    from .piper_grip_benchmark import decode_attempt
    from .probe_evidence import _arrays
    root = Path(root)
    protocol, report = read(root/'protocol.json'), read(root/'report.json')
    if protocol['source_sha256'] != sources() or report['source_sha256'] != protocol['source_sha256']:
        raise ValueError('Inspect with the exact frozen study implementation.')
    roster = [(case, variant) for case in protocol['cases'] for variant in protocol['variants']]
    ids = [case['id']+'-'+variant for case, variant in roster]
    if ([r['id'] for r in report['rows']] != ids or len(ids) != protocol['planned_attempts']
            or report['attempted'] != len(ids)):
        raise ValueError('Recorded attempt denominator/order differs from the protocol.')
    checked = []
    for row, (case, variant) in zip(report['rows'], roster):
        folder = root/row['id']
        if set(row['files']) != {'started.json', 'episode.json.gz', 'native.npz', 'native.json', 'evaluation.json'}:
            raise ValueError('Unexpected attempt evidence roster.')
        for name, expected in row['files'].items():
            path = folder/name
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 32*1024*1024:
                raise ValueError('Invalid bounded evidence file.')
            raw = path.read_bytes()
            if {'bytes':len(raw),'sha256':sha256(raw).hexdigest()} != expected:
                raise ValueError('Recorded evidence digest changed: '+row['id']+'/'+name)
        episode = json.loads(decode_attempt((folder/'episode.json.gz').read_bytes()))
        if (episode['id'] != row['id'] or episode['config'] != case['config'] or episode['variant'] != variant
                or episode['valid'] != row['valid'] or episode['metrics'] != row['metrics']):
            raise ValueError('Episode and protocol/report identity differ.')
        with zipfile.ZipFile(folder/'native.npz') as archive:
            entries = archive.infolist()
            if (len({p.filename for p in entries}) != len(entries) or len(entries) != 22
                    or sum(p.file_size for p in entries) > 32*1024*1024
                    or any(p.file_size > 2*1024*1024 or '/' in p.filename or not p.filename.endswith('.npy') for p in entries)):
                raise ValueError('Native NPZ archive exceeds the numeric-only contract.')
        with np.load(folder/'native.npz', allow_pickle=False) as native:
            arrays = {name:native[name] for name in native.files}
        metadata = read(folder/'native.json')
        _arrays(arrays, metadata)  # Also validate refused/no-probe native trajectories.
        expected_evaluation = _evaluate(episode, arrays, metadata, protocol)
        if (encoded(expected_evaluation) != encoded(row['evaluation'])
                or encoded(read(folder/'evaluation.json')) != encoded(row['evaluation'])):
            raise ValueError('Recomputed native probe evidence differs: '+row['id'])
        actions, trajectory = episode['actions'], episode['trajectory']
        count = len(actions)
        if (not episode['valid'] or count != 1000 or len(trajectory) != count+1
                or len(arrays['t_seconds']) != count*20):
            raise ValueError('Incomplete/invalid native attempt retained; it cannot qualify.')
        requested = episode['policy']['requested_controls']
        if len(requested) != count:
            raise ValueError('Missing requested-control records.')
        control = np.array(trajectory[0]['native_actuator_controls'], dtype=float)
        step = np.array([episode['config']['joint_target_rate_rad_s']*.001]*6
                        +[episode['config']['jaw_target_rate_m_s']*.001])
        for i, (action, request) in enumerate(zip(actions, requested)):
            if (abs(action['t_seconds']-i*.02) > 1e-8 or request['index'] != i
                    or any(request[k] != action[k] for k in ('t_seconds','joint_targets_rad','gripper_position_m'))):
                raise ValueError('Issued commands differ from the policy records/cadence.')
            target = np.array(action['joint_targets_rad']+[action['gripper_position_m']])
            for tick in range(i*20, (i+1)*20):
                control += np.clip(target-control, -step, step)
                actual = np.r_[arrays['arm_actuator_controls_rad'][tick], arrays['applied_jaw_target_m'][tick]]
                if (not np.allclose(control, actual, atol=1e-12, rtol=0)
                        or arrays['requested_jaw_target_m'][tick] != action['gripper_position_m']):
                    raise ValueError('Applied native controls differ from the issued tape and fixed slew law.')
            for field in ('object_position_m', 'joint_position_rad', 'tcp_position_m'):
                if not np.allclose(trajectory[i+1][field], arrays[field][(i+1)*20-1], atol=1e-9, rtol=0):
                    raise ValueError('Boundary observation differs from native evidence.')
        checked.append(dict(id=row['id'], native_rows=len(arrays['t_seconds']), commands=count,
                            evaluation_recomputed=True, applied_controls_recomputed=True))
    return dict(schema='dvidia.piper-probe-inspection/v1', attempts=len(checked),
                native_rows=sum(r['native_rows'] for r in checked), rows=checked,
                scope='Recorded consistency and predicate reconstruction; not independent physics or authenticated sensors.',
                physical_robot_ready=False)


def run(root, assets):
    root = Path(root)
    protocol = read(root/'protocol.json')
    if encoded(protocol) != encoded(make_protocol(assets)):
        raise ValueError('Frozen protocol/source/runtime/assets no longer match; create a new study.')
    # One launch per study; a failed run remains evidence, never an implicit resume.
    write(root/'started.json', {'protocol_sha256':sha256((root/'protocol.json').read_bytes()).hexdigest(),
                               'started_at_utc':datetime.now(timezone.utc).isoformat()})
    rows = []
    for case in protocol['cases']:
        for variant in protocol['variants']:
            if sources() != protocol['source_sha256']:
                raise ValueError('Source changed during the frozen study.')
            folder = root/(case['id']+'-'+variant); folder.mkdir()
            write(folder/'started.json', {'scene_id':case['id'], 'variant':variant,
                                         'protocol_sha256':sha256((root/'protocol.json').read_bytes()).hexdigest()})
            result, arrays, metadata = _attempt(assets, case, variant, protocol)
            raw = encoded(result)
            (folder/'episode.json.gz').write_bytes(gzip.compress(raw, mtime=0))
            buffer = io.BytesIO(); np.savez_compressed(buffer, **arrays)
            (folder/'native.npz').write_bytes(buffer.getvalue())
            write(folder/'native.json', metadata)
            try:
                evaluation = _evaluate(result, arrays, metadata, protocol)
            except Exception as exc:
                evaluation = dict(status='audit_refused', error_type=type(exc).__name__, reason=str(exc)[:300],
                                  physical_robot_ready=False)
            write(folder/'evaluation.json', evaluation)
            files = {p.name:{'bytes':p.stat().st_size,'sha256':sha256(p.read_bytes()).hexdigest()} for p in sorted(folder.iterdir())}
            row = dict(id=result['id'], scene_id=case['id'], variant=variant,
                       mass_kg=case['config']['mass_kg'], friction=case['config']['friction'],
                       valid=result['valid'], error=result['error'], metrics=result['metrics'],
                       policy={k:v for k,v in (result.get('policy') or {}).items() if k != 'requested_controls'},
                       timing=result['timing']['summaries'], evaluation=evaluation, files=files)
            rows.append(row)
            write(folder/'summary.json', row)
            print(json.dumps({'attempt':result['id'],'valid':result['valid'], 'audit':evaluation['status']}), flush=True)
    if sources() != protocol['source_sha256'] or runtime() != protocol['runtime']:
        raise ValueError('Frozen source/runtime changed during the study.')
    report = dict(schema=SCHEMA, scope=SCOPE, planned_attempts=protocol['planned_attempts'], attempted=len(rows),
                  valid_attempts=sum(r['valid'] for r in rows), rows=rows,
                  previous_learned_heldout_result='5/12 unchanged; this study contains no training or full-placement evaluation.',
                  physical_robot_ready=False, source_sha256=protocol['source_sha256'])
    write(root/'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'run'):
        command = sub.add_parser(name)
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--assets', type=Path, required=True)
    command = sub.add_parser('inspect')
    command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        value = (inspect(args.output) if args.command == 'inspect'
                 else globals()[args.command](args.output, args.assets))
        print(json.dumps(value if args.command == 'inspect' else
                         {'schema':SCHEMA,'planned_attempts':value['planned_attempts'],
                          'physical_robot_ready':False},indent=2))
        return 0
    except (ValueError, OSError, KeyError, ImportError) as exc:
        print('Probe study refused: '+str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

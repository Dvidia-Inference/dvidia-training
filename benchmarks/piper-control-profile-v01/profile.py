"""Development-only profiling of one immutable, selected synthetic PiPER tape.

Run from a checkout with PYTHONPATH=src. No runtime modules or global simulator
functions are patched. Baseline cProfile runs are diagnostic and have profiling
overhead; their timings must not be compared with an unprofiled speed headline.
The separate replay mode checks the immutable saved action tape. It is a fixed
scenario mechanics diagnostic, not a new skill-success benchmark or speed comparison.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import gzip
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import platform
import pstats
import time

SCHEMA = 'dvidia.piper-control-profile/v1'
MODEL_SHA = '2976a605575d9edefb5db366e72c5e006c0ce85a5531ad3d61512436864c8ded'
EPISODE_SHA = '7954c32ae4bc689f8947d8bc7786e028bf18dc1a02a100a7a38bdc779bf8193d'
COMPARISON_FIELDS = ('config', 'seed', 'valid', 'success', 'reason', 'error', 'simulated_seconds',
                     'initial_object_position_m', 'compiled_model_sha256', 'model_sha256',
                     'trajectory', 'actions', 'metrics')
SCOPE = ('One selected synthetic PiPER development scenario, fixed model/action tape and unchanged '
         '1 ms mechanics/contact/outcome checks. Privileged native state, no physical robot. '
         'Not a held-out evaluation, real-time guarantee or a replacement for the frozen 5/12 benchmark.')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def identity(raw):
    return {'bytes': len(raw), 'sha256': sha256(raw).hexdigest()}


def write(path, value):
    with Path(path).open('xb') as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()+b'\n')


def decode(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError('Duplicate JSON field')
            result[key] = value
        return result
    def nonfinite(token):
        raise ValueError('Nonfinite JSON number: '+token)
    document = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
    canonical(document)
    return document


def compare_episode(expected, actual):
    """Exact finite JSON comparison; time measurements are deliberately excluded."""
    differences = []
    for key in COMPARISON_FIELDS:
        if key not in expected or key not in actual or canonical(expected[key]) != canonical(actual[key]):
            differences.append(key)
    return {'exact_match': not differences, 'different_fields': differences,
            'fields_compared': list(COMPARISON_FIELDS),
            'comparison_scope': 'Every recorded 20 ms observation including contacts/forces/commands; exact outcomes and cumulative native contact ticks. Not a saved full 1 ms state tape.'}


def summary(values):
    ordered = sorted(values)
    if not ordered:
        return {'count': 0}
    def percentile(q):
        rank = (len(ordered)-1)*q
        low, high = math.floor(rank), math.ceil(rank)
        return ordered[low]+(ordered[high]-ordered[low])*(rank-low)
    return {'count': len(ordered), 'sum_s': sum(ordered), 'mean_s': sum(ordered)/len(ordered),
            'p50_s': percentile(.5), 'p95_s': percentile(.95), 'p99_s': percentile(.99),
            'max_s': ordered[-1], 'over_nominal_20ms': sum(value > .02 for value in ordered)}


def load_inputs(episode_path, model_path):
    from dvidia_training.piper_imitation_benchmark import config, runtime_hashes
    from dvidia_training.piper_imitation_score import score_episode
    episode_raw, model_raw = Path(episode_path).read_bytes(), Path(model_path).read_bytes()
    if len(episode_raw) > 8*1024*1024 or sha256(episode_raw).hexdigest() != EPISODE_SHA:
        raise ValueError('Only the exact preserved selected episode is in this diagnostic protocol')
    if len(model_raw) > 8*1024*1024 or sha256(model_raw).hexdigest() != MODEL_SHA:
        raise ValueError('Only the exact pinned prior model is in this diagnostic protocol')
    with gzip.GzipFile(fileobj=io.BytesIO(episode_raw)) as stream:
        expanded = stream.read(32*1024*1024+1)
    if len(expanded) > 32*1024*1024:
        raise ValueError('Episode expands beyond the diagnostic budget')
    episode = decode(expanded)
    case = {'id': 'installed-known-goal1', 'start_xy': [.309, -.04],
            'goal_xy': [.40, -.08], 'seed': 261517}
    if (episode.get('implementation_sha256') != runtime_hashes()
            or episode.get('model_sha256') != MODEL_SHA
            or episode.get('seed') != case['seed']
            or canonical(episode.get('config')) != canonical(asdict(config(case)))
            or len(episode.get('trajectory', [])) != 1001 or len(episode.get('actions', [])) != 1000
            or not score_episode(episode, case['goal_xy'])['success']):
        raise ValueError('Preserved episode, runtime, config or fixed score differs from the protocol')
    return episode, case, {'episode': identity(episode_raw), 'model': identity(model_raw)}


def source_inventory(destination):
    from dvidia_training import piper_imitation_benchmark as module
    names = list(module.RUNTIME_FILES)+['env.py', 'piper_learned_skill.py']
    folder = destination/'sources'
    folder.mkdir()
    rows = []
    for name, path in [(name, Path(module.__file__).with_name(name)) for name in names]+[('profile.py', Path(__file__))]:
        raw = path.read_bytes()
        (folder/name).write_bytes(raw)
        rows.append({'name': name, **identity(raw)})
    return rows


def check_sources(inventory):
    from dvidia_training import piper_imitation_benchmark as module
    for row in inventory:
        path = Path(__file__) if row['name'] == 'profile.py' else Path(module.__file__).with_name(row['name'])
        if sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Profiling source changed during the diagnostic')


def profile_rows(profiler):
    stats = pstats.Stats(profiler)
    rows = []
    categories = defaultdict(list)
    for (filename, line, name), (primitive, total, own, cumulative, _) in stats.stats.items():
        row = {'file': Path(filename).name, 'line': line, 'function': name,
               'primitive_calls': primitive, 'total_calls': total,
               'self_seconds': own, 'inclusive_seconds': cumulative}
        rows.append(row)
        category = None
        if 'mj_step' in name: category = 'native_mj_step'
        elif 'mj_forward' in name: category = 'native_mj_forward'
        elif 'mj_contactForce' in name: category = 'native_contact_force'
        elif 'mj_objectVelocity' in name: category = 'native_object_velocity'
        elif name == '_contacts' and Path(filename).name == 'piper_task.py': category = 'contact_extraction_inclusive'
        elif name == '_finite' and Path(filename).name in ('piper_task.py', 'reference_task.py'): category = 'finite_checks_inclusive'
        elif name in ('observation', 'info') and Path(filename).name in ('piper_task.py', 'reference_task.py'): category = 'observation_and_info_inclusive'
        elif name == 'predict' and Path(filename).name == 'piper_imitation.py': category = 'policy_inclusive'
        elif name == 'step' and Path(filename).name == 'piper_task.py': category = 'environment_step_inclusive'
        elif name == 'reset' and Path(filename).name == 'piper_task.py': category = 'reset_inclusive'
        elif name == 'rollout' and Path(filename).name == 'piper_imitation_benchmark.py': category = 'rollout_orchestration_inclusive'
        if category: categories[category].append(row)
    return {'functions_by_self_time': sorted(rows, key=lambda row: row['self_seconds'], reverse=True),
            'functions_by_inclusive_time': sorted(rows, key=lambda row: row['inclusive_seconds'], reverse=True),
            'categories': dict(categories),
            'scope': 'Inclusive categories overlap and must not be added. cProfile self time is exclusive; rollout self includes setup/loop/recording bookkeeping, not recording alone.'}


def baseline(assets, model, case, expected, folder):
    # This script is called profile.py. Hide its directory briefly so cProfile
    # imports the standard-library profile module rather than this file again.
    import sys
    prior = list(sys.path)
    try:
        directory = Path(__file__).resolve().parent
        sys.path[:] = [entry for entry in sys.path if Path(entry or '.').resolve() != directory]
        import cProfile
    finally:
        sys.path[:] = prior
    from dvidia_training.piper_imitation_benchmark import rollout, save_episode
    from dvidia_training.piper_imitation_score import score_episode
    results = []
    for timer_name, timer in (('wall', time.perf_counter_ns), ('cpu', time.process_time_ns)):
        target = folder/('cprofile-'+timer_name)
        target.mkdir()
        write(target/'started.json', {'timer': timer_name, 'scope': SCOPE})
        profiler = cProfile.Profile(timer=timer, timeunit=1e-9)
        wall, cpu = time.perf_counter(), time.process_time()
        episode = profiler.runcall(rollout, assets, case, model)
        wall, cpu = time.perf_counter()-wall, time.process_time()-cpu
        profiler.dump_stats(str(target/'raw.pstats'))
        # Keep original raw stats locally; this second form strips local paths.
        pstats.Stats(profiler).strip_dirs().dump_stats(str(target/'portable.pstats'))
        save_episode(target/'episode.json.gz', episode)
        comparison = compare_episode(expected, episode)
        result = {'timer': timer_name, 'wall_seconds': wall, 'process_cpu_seconds': cpu,
                  'profiling_overhead': 'cProfile enabled throughout original rollout, including reset/model/asset verification. Not a performance headline.',
                  'comparison': comparison, 'score': score_episode(episode, case['goal_xy']),
                  'functions': profile_rows(profiler), 'runtime_reported_timing': {
                      key: value for key, value in episode['timing'].items() if not key.endswith('_samples_s')}}
        write(target/'receipt.json', result)
        results.append(result)
        print(json.dumps({'phase': 'cprofile-'+timer_name, 'wall_seconds': wall,
                          'process_cpu_seconds': cpu, 'exact_match': comparison['exact_match'],
                          'score_success': result['score']['success']}), flush=True)
        if not comparison['exact_match']:
            raise ValueError('Profiled original rollout differs from the preserved tape; evidence retained')
    return results


def replay(assets, expected):
    """Unmodified environment, original tape, exact 20 ms records and outcomes."""
    from dvidia_training.piper_task import PiperTaskEnv, PiperTaskConfig
    from dvidia_training import reference_sim as sim
    env = PiperTaskEnv(assets, PiperTaskConfig(**expected['config']))
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    obs, info = env.reset(expected['seed'])
    episode = {key: expected[key] for key in ('config', 'seed', 'model_sha256')}
    episode.update(trajectory=[obs], actions=[], initial_object_position_m=env.initial_object_position.tolist(),
                   compiled_model_sha256=sim.model_digest(env.model), error=None)
    step_wall, step_cpu = [], []
    for saved in expected['actions']:
        action = {key: saved[key] for key in ('joint_targets_rad', 'gripper_position_m')}
        record = {'t_seconds': obs['t_seconds'], **action}
        episode['actions'].append(record)
        obs.update(issued_joint_targets_rad=list(action['joint_targets_rad']),
                   issued_gripper_position_m=action['gripper_position_m'], command=record)
        wall, cpu = time.perf_counter(), time.process_time()
        obs, info = env.step(action)
        step_wall.append(time.perf_counter()-wall)
        step_cpu.append(time.process_time()-cpu)
        episode['trajectory'].append(obs)
        if env.done: break
    episode.update(valid=info['valid'], success=info['success'], reason=info['reason'],
                   simulated_seconds=info['simulation_time'],
                   metrics={**info, 'contact_pair_native_ticks': env.contact_pair_native_ticks})
    return episode, {'wall_seconds': time.perf_counter()-started_wall,
                     'process_cpu_seconds': time.process_time()-started_cpu,
                     'environment_step_wall': summary(step_wall), 'environment_step_cpu': summary(step_cpu)}


def tape_replay(assets, model, case, expected, folder):
    from dvidia_training.piper_imitation_benchmark import save_episode
    from dvidia_training.piper_imitation_score import score_episode
    write(folder/'started.json', {'variant': 'action_tape', 'scope': SCOPE})
    episode, timing = replay(assets, expected)
    save_episode(folder/'episode.json.gz', episode)
    result = {'variant': 'action_tape', 'timing': timing,
              'comparison': compare_episode(expected, episode), 'score': score_episode(episode, case['goal_xy']),
              'speed_scope': 'Single exact-tape replay, not a controlled speed comparison. No policy inference; setup/provenance work differs from original rollout. Do not compare against historical or cProfile timings.'}
    write(folder/'replay-receipt.json', result)
    print(json.dumps({'phase': 'action_tape', 'wall_seconds': timing['wall_seconds'],
                      'exact_match': result['comparison']['exact_match'], 'score_success': result['score']['success']}), flush=True)
    if not result['comparison']['exact_match']:
        raise ValueError('Saved-action replay differs from the preserved tape; evidence retained')
    return [result]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('baseline', 'replay'), default='baseline')
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error('Output must be a new directory; all prior evidence is preserved')
    expected, case, inputs = load_inputs(args.episode, args.model)
    from dvidia_training import robot_profiles, reference_sim as sim
    provenance = robot_profiles.verify_assets('piper', args.assets)
    mj, np = sim._native_imports()
    args.output.mkdir(parents=True, exist_ok=False)
    inventory = source_inventory(args.output)
    protocol = {'schema': SCHEMA, 'scope': SCOPE, 'mode': args.mode, 'inputs': inputs, 'case': case,
                'source_inventory': inventory, 'asset_provenance': provenance,
                'runtime': {'python': platform.python_version(), 'machine': platform.machine(),
                            'system': platform.system(), 'mujoco': mj.__version__, 'numpy': np.__version__},
                'roster': ['cprofile-wall', 'cprofile-cpu'] if args.mode == 'baseline' else ['action_tape'],
                'accuracy_freeze': 'Exact saved observation/action/outcome equality and unchanged scorer; timing fields excluded. No policy/physics changes or result-based retries.',
                'raw_profile_path_scope': 'raw.pstats may contain local source paths; portable.pstats and JSON function labels strip directories.'}
    write(args.output/'protocol.json', protocol)
    failure = None
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    try:
        results = (baseline if args.mode == 'baseline' else tape_replay)(args.assets, args.model, case, expected, args.output)
        check_sources(inventory)
    except Exception as exc:
        failure = {'type': type(exc).__name__, 'message': str(exc)}
        results = None
    receipt = {'schema': SCHEMA, 'scope': SCOPE, 'mode': args.mode, 'error': failure,
               'complete': failure is None, 'wall_seconds': time.perf_counter()-started_wall,
               'process_cpu_seconds': time.process_time()-started_cpu,
               'results': results, 'physical_robot_ready': False}
    write(args.output/'receipt.json', receipt)
    if failure: print(json.dumps({'error': failure}), flush=True)
    return 0 if failure is None else 1


if __name__ == '__main__':
    raise SystemExit(main())

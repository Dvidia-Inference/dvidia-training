"""Local rollout, parameter training and qualified-pack export."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
try:
    import resource
except ImportError:  # Windows has no POSIX resource module.
    resource = None
import socket
import time
import numpy as np
import mujoco
from .env import CableEndReachEnv, Config
from .pack import encode, export_pack, verify_pack
from .policies import FeedbackPolicy, ReleasePolicy, ZeroPolicy


def parse_seeds(value):
    try:
        seeds = [int(s.strip()) for s in value.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use comma-separated nonnegative integer seeds.') from exc
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds) or len(seeds) > 100:
        raise argparse.ArgumentTypeError('Use 1–100 distinct nonnegative seeds.')
    return seeds


class NetworkGuard:
    """A bounded Python socket test; this does not intercept native C networking."""
    def __init__(self, enabled):
        self.enabled, self.attempts, self.original = enabled, [], {}

    def __enter__(self):
        if not self.enabled:
            return self
        for name in ['socket', 'create_connection', 'getaddrinfo']:
            self.original[name] = getattr(socket, name)
            def denied(*args, _name=name, **kwargs):
                self.attempts.append(_name)
                raise RuntimeError('Python network access denied by offline run guard.')
            setattr(socket, name, denied)
        try:
            socket.create_connection(('offline-guard.invalid', 1))
        except RuntimeError:
            pass
        else:
            raise AssertionError('Offline guard failed its self-test.')
        self.attempts.clear()
        return self

    def __exit__(self, *args):
        for name, original in self.original.items():
            setattr(socket, name, original)

    def record(self):
        return {'enabled': self.enabled, 'scope': 'Python socket calls during rollouts/training after dependency imports; not native C or installation',
                'guard_self_test_passed': self.enabled, 'attempted_calls': self.attempts}


def rollout(config, seed, parameters, policy_name='feedback', trace=True):
    started = time.perf_counter()
    environment = CableEndReachEnv(config)
    try:
        observation, reset_info = environment.reset(seed=seed)
    except (ValueError, RuntimeError) as error:
        elapsed = time.perf_counter() - started
        return {'seed':seed, 'policy':policy_name, 'success':False,
                'reason':'rejected_start', 'reward':0., 'final_distance':None, 'final_speed':None,
                'control_steps':0, 'simulated_seconds':0., 'setup_seconds':elapsed,
                'episode_wall_seconds':elapsed, 'step_loop_seconds':0.,
                'control_latency_p50_seconds':None, 'control_latency_p95_seconds':None,
                'simulated_seconds_per_wall_second':0., 'final_observation_available':False,
                'reset_info':{'valid':False, 'error':str(error)},
                'final_info':{'valid':False, 'reason':'rejected_start'}, 'trace':[]}
    setup = time.perf_counter() - started
    feedback = FeedbackPolicy(**parameters)
    policy = {'feedback': feedback, 'zero': ZeroPolicy(), 'release': ReleasePolicy(feedback)}[policy_name]
    frames = [dict(time=0.0, **observation)] if trace else []
    rewards, latencies = [], []
    terminated = truncated = False
    info = reset_info
    while not (terminated or truncated):
        step_start = time.perf_counter()
        action = policy(observation)
        observation, reward, terminated, truncated, info = environment.step(action)
        latencies.append(time.perf_counter() - step_start)
        rewards.append(float(reward))
        if trace and info.get('observations_available', True):
            frames.append(dict(time=info['simulation_time'], **observation,
                               reward=float(reward), reason=info.get('reason'),
                               contact_count=info.get('contact_count', 0)))
        if len(rewards) > 100_000:
            raise RuntimeError('Environment did not respect a finite episode horizon.')
    elapsed = time.perf_counter() - started
    simulated = info['simulation_time']
    available = info.get('observations_available', True)
    distance = float(np.linalg.norm(np.asarray(observation['target'])-observation['endpoint_position'])) if available else None
    episode = {'seed': seed, 'policy': policy_name,
               'success': bool(info.get('success', False)), 'reason': info.get('reason', 'unknown'),
               'reward': sum(rewards), 'final_distance': distance,
               'final_speed': float(np.linalg.norm(observation['endpoint_velocity'])) if available else None,
               'final_observation_available': available,
               'control_steps': len(rewards), 'simulated_seconds': simulated,
               'setup_seconds': setup, 'episode_wall_seconds': elapsed,
               'step_loop_seconds': sum(latencies),
               'control_latency_p50_seconds': float(np.median(latencies)),
               'control_latency_p95_seconds': float(np.percentile(latencies, 95)),
               'simulated_seconds_per_wall_second': simulated/elapsed,
               'reset_info': reset_info, 'final_info': info, 'trace': frames}
    return episode


def payload_for(config, episodes, parameters, guard):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource is not None else None
    return {'schema_version': 1, 'task': 'CableEndReach-v0', 'scope': 'simulation-only',
            'actuation': 'ideal endpoint attachment', 'observations': 'privileged simulator state',
            'config': asdict(config), 'policy_parameters': parameters, 'episodes': episodes,
            'hardware': {'system': platform.system(), 'machine': platform.machine(),
                         'python': platform.python_version(), 'mujoco': mujoco.__version__, 'numpy': np.__version__,
                         'process_lifetime_peak_rss_bytes': (int(rss if platform.system() == 'Darwin' else rss*1024) if rss is not None else None)},
            'network_guard': guard.record(),
            'timing_scope': 'Episode clock includes construction/reset/settling, policy, physics, observations and trace collection. Step-loop clock excludes trace append/reset. Output I/O, viewer, installation and training optimization excluded; no GPU metric.',
            'summary': {'successes': sum(e['success'] for e in episodes), 'episodes': len(episodes),
                        'simulated_seconds': sum(e['simulated_seconds'] for e in episodes),
                        'episode_wall_seconds': sum(e['episode_wall_seconds'] for e in episodes)}}


def write_run(directory, payload, parameters):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'run.json').write_bytes(encode(payload))
    (directory/'policy.json').write_bytes(encode(parameters))
    manifest = export_pack(directory/'skill-pack.zip', payload, parameters)
    (directory/'manifest.json').write_bytes(encode(manifest))
    try:
        from .viewer import save_replay
    except ImportError:
        pass
    else:
        save_replay(payload, directory/'replay.html')
    summary = payload['summary']
    (directory/'RESULTS.md').write_text(
        '# CableEndReach-v0 run\n\n'
        f"{summary['successes']}/{summary['episodes']} simulated task successes. "
        'Ideal endpoint attachment; privileged simulator state. No physical or GPU qualification.\n\n'
        '| Seed | Outcome | Final distance (m) | Simulation/wall seconds |\n|---|---|---:|---:|\n'+
        ''.join(f"| {e['seed']} | {e['reason']} | {format(e['final_distance'], '.6f') if e['final_distance'] is not None else 'unavailable'} | {e['simulated_seconds_per_wall_second']:.2f} |\n" for e in payload['episodes'])+
        '\n'+payload['timing_scope']+'\n\nSee run.json for configuration, warnings, timings and traces; manifest.json for content hashes.\n', encoding='utf8')
    print(json.dumps({'output': str(directory), **summary, 'pack_verified': bool(verify_pack(directory/'skill-pack.zip'))}))


def train(config, args, base, guard):
    if set(args.seeds) & set(args.eval_seeds):
        raise ValueError('Training and evaluation seeds must be disjoint.')
    rng = np.random.default_rng(args.search_seed)
    mean = np.log([base['kp'], base['kd'], max(base['gravity_bias'], 1e-4)])
    spread = np.array([0.6, 0.6, 0.5])
    records = []
    def evaluate(parameters, label):
        episodes = [rollout(config, seed, parameters, trace=False) for seed in args.seeds]
        score = float(np.mean([10.0*e['success']-(e['final_distance'] if e['final_distance'] is not None else 1.)-0.02*(e['final_speed'] if e['final_speed'] is not None else 1.)-
                               (10 if e['reason'] not in ['success', 'horizon'] else 0) for e in episodes]))
        record = {'candidate': label, 'parameters': parameters, 'score': score,
                  'successes': sum(e['success'] for e in episodes), 'episodes': len(episodes),
                  'results': [{k:v for k,v in e.items() if k not in ['trace', 'reset_info', 'final_info']} for e in episodes]}
        records.append(record)
        return score
    best, best_score = dict(base), evaluate(base, 'baseline')
    for iteration in range(args.iterations):
        candidates = []
        for sample in range(args.population):
            vector = np.clip(rng.normal(mean, spread), np.log([0.1,0.005,0.0001]), np.log([50.,5.,config.force_limit]))
            kp,kd,bias = np.exp(vector)
            parameters = dict(force_limit=config.force_limit, kp=float(kp), kd=float(kd), gravity_bias=float(bias))
            score = evaluate(parameters, f'{iteration}:{sample}')
            candidates.append((score, vector, parameters))
            if score > best_score:
                best, best_score = parameters, score
        elites = sorted(candidates, key=lambda x:x[0], reverse=True)[:max(2,args.population//3)]
        values = np.array([entry[1] for entry in elites])
        mean, spread = values.mean(axis=0), np.maximum(values.std(axis=0), .12)
        print(json.dumps({'training_iteration': iteration+1, 'best_score': best_score}), flush=True)
    heldout = [rollout(config, seed, best) for seed in args.eval_seeds]
    payload = payload_for(config, heldout, best, guard)
    payload.update(training_seeds=args.seeds, training={'method':'cross-entropy search of explicit PD controller parameters',
                   'search_seed':args.search_seed, 'iterations':args.iterations, 'population':args.population,
                   'candidate_records':records, 'evaluation_policy_frozen_before_heldout':True})
    return payload, best


def main():
    parser = argparse.ArgumentParser(description='DVIDIA offline simulation lab: bounded cable endpoint control.')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ['run', 'train']:
        p = sub.add_parser(name)
        p.add_argument('--seeds', type=parse_seeds, default=parse_seeds('0,1,2'))
        p.add_argument('--output', type=Path, default=Path('runs')/('cable-'+name))
        p.add_argument('--offline', action='store_true', help='Deny Python socket calls during execution and record the guard test.')
        p.add_argument('--config', type=Path, help='JSON config or prior run.json configuration.')
        p.add_argument('--policy-file', type=Path)
        p.add_argument('--kp', type=float, default=10.)
        p.add_argument('--kd', type=float, default=.25)
        p.add_argument('--gravity-bias', type=float, default=.1)
        if name == 'run':
            p.add_argument('--policy', choices=['feedback','zero','release'], default='feedback')
        else:
            p.add_argument('--eval-seeds', type=parse_seeds, default=parse_seeds('100,101,102'))
            p.add_argument('--iterations', type=int, default=2)
            p.add_argument('--population', type=int, default=4)
            p.add_argument('--search-seed', type=int, default=31415)
    verify = sub.add_parser('verify-pack')
    verify.add_argument('path', type=Path)
    args = parser.parse_args()
    if args.command == 'verify-pack':
        manifest = verify_pack(args.path)
        print(json.dumps({'verified':True, 'task':manifest['task'], 'status':manifest['status'], 'files':len(manifest['files'])}))
        return
    configuration = json.loads(args.config.read_text()) if args.config else {}
    config = Config(**configuration.get('config', configuration))
    parameters = json.loads(args.policy_file.read_text()) if args.policy_file else dict(force_limit=config.force_limit,kp=args.kp,kd=args.kd,gravity_bias=args.gravity_bias)
    FeedbackPolicy(**parameters)
    if parameters['force_limit'] != config.force_limit:
        parser.error('Policy and environment force limits must match.')
    if args.command == 'train' and not (1 <= args.iterations <= 20 and 3 <= args.population <= 32):
        parser.error('Use 1–20 iterations and 3–32 candidates per iteration.')
    started = time.perf_counter()
    with NetworkGuard(args.offline) as guard:
        if args.command == 'run':
            episodes = [rollout(config, seed, parameters, args.policy) for seed in args.seeds]
            payload = payload_for(config, episodes, parameters, guard)
        else:
            payload, parameters = train(config, args, parameters, guard)
        payload['execution_wall_seconds_before_output'] = time.perf_counter()-started
        if guard.attempts:
            raise RuntimeError('Execution attempted network operations.')
    write_run(args.output, payload, parameters)


if __name__ == '__main__':
    main()

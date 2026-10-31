"""Controlled, offline PiPER source comparison; never a hardware latency claim.

The fixed ABBA timing runs use the uninstrumented environment and a preserved
1,000-command tape. Separate diagnostic processes record every post-scoring
1 ms tick. Their substantial copying/contact/FK overhead is excluded from the
timing comparison. Every scheduled attempt is retained, including failures.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

SCHEMA = 'dvidia.piper-control-comparison/v1'
EPISODE_SHA = '7954c32ae4bc689f8947d8bc7786e028bf18dc1a02a100a7a38bdc779bf8193d'
ALLOWED_SOURCE_DIFFERENCES = ('piper_task.py',)
SOURCE_FILES = ('__init__.py', 'piper_task.py', 'reference_task.py', 'reference_sim.py',
                'robot_profiles.py', 'robot_profiles.lock.json', 'reference_assets.lock.json',
                'env.py', 'piper_feedback_env.py', 'piper_imitation_score.py')
CASES = ('original', 'open-jaw', 'hold', 'feedback-no-pulse', 'feedback-pulse')
PULSE = {'start_s': 8.0, 'end_s': 8.12, 'minimum_opening_m': .035}
NATIVE_ARRAYS = ('qpos', 'qvel', 'qacc', 'qacc_warmstart', 'ctrl', 'act',
                 'actuator_force', 'qfrc_applied', 'xfrc_applied', 'qfrc_bias',
                 'qfrc_constraint', 'xpos', 'xquat', 'site_xpos', 'site_xmat')
SCORER_FIELDS = ('elapsed', 'steps', 'dwell', 'lift_dwell', 'max_lift', 'max_pad_force',
                 'critical_contact_ticks', 'lift_seen', 'grasp_seen', 'carry_complete',
                 'retention_lost', 'carry_contact_loss', 'max_carry_contact_loss',
                 'carry_contact_loss_events', 'first_success_time', 'reason', 'done')
EPISODE_FIELDS = ('config', 'seed', 'valid', 'success', 'reason', 'error',
                  'simulated_seconds', 'initial_object_position_m', 'compiled_model_sha256',
                  'trajectory', 'actions', 'metrics')
SCOPE = ('One previously checked synthetic PiPER scenario, preserved commands and privileged '
         'native simulator state. Allocation-only source comparison, not new skill training, '
         'held-out success, hardware latency, calibrated contact or a real-time guarantee.')
ADOPTION_RULE = {'required_exact_equivalence': True,
                 'required_lower_candidate_mean_each_pair': ['step_wall', 'step_cpu'],
                 'minimum_pooled_mean_reduction_percent_each_clock': 5.,
                 'on_failure': 'Do not promote; retain as negative or inconclusive. No post-result tuning.'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def identity(raw):
    return {'bytes': len(raw), 'sha256': sha256(raw).hexdigest()}


def write(path, value):
    with Path(path).open('xb') as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()+b'\n')


def write_gzip(path, value):
    with Path(path).open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as stream:
            stream.write(canonical(value))


def decode(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError('Duplicate JSON field')
            result[key] = value
        return result
    def nonfinite(token):
        raise ValueError('Nonfinite JSON token: '+token)
    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
    canonical(result)
    return result


def read_gzip(path, maximum=64*1024*1024):
    raw = Path(path).read_bytes()
    if len(raw) > maximum:
        raise ValueError('Compressed document exceeds bound')
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        expanded = stream.read(maximum+1)
    if len(expanded) > maximum:
        raise ValueError('Expanded document exceeds bound')
    return decode(expanded)


def inventory(root):
    folder = Path(root)/'src'/'dvidia_training'
    rows = {}
    for name in SOURCE_FILES:
        path = folder/name
        if path.is_symlink() or not path.is_file():
            raise ValueError('Source must be a regular file: '+name)
        rows[name] = identity(path.read_bytes())
    return rows


def check_inventory(root, expected):
    if inventory(root) != expected:
        raise ValueError('Frozen source changed during the comparison')


def source_differences(left, right):
    differences = sorted(name for name in set(left) | set(right) if left.get(name) != right.get(name))
    if set(differences) - set(ALLOWED_SOURCE_DIFFERENCES):
        raise ValueError('Source differs outside the declared allocation-only file: '+', '.join(differences))
    if not differences:
        raise ValueError('Baseline and candidate sources are identical')
    return differences


def schedule(pairs=2):
    if type(pairs) is not int or not 2 <= pairs <= 8 or pairs % 2:
        raise ValueError('Use an even number of paired runs in [2,8]')
    return [variant for pair in range(pairs)
            for variant in (('baseline', 'candidate') if pair % 2 == 0 else ('candidate', 'baseline'))]


def summary(values):
    values = sorted(values)
    if not values:
        return {'count': 0}
    def percentile(q):
        rank = (len(values)-1)*q
        low, high = math.floor(rank), math.ceil(rank)
        return values[low]+(values[high]-values[low])*(rank-low)
    return {'count': len(values), 'sum_s': sum(values), 'mean_s': sum(values)/len(values),
            'p50_s': percentile(.5), 'p95_s': percentile(.95), 'p99_s': percentile(.99),
            'max_s': values[-1], 'over_nominal_20ms': sum(item > .02 for item in values)}


def load_tape(path):
    raw = Path(path).read_bytes()
    if len(raw) > 8*1024*1024 or sha256(raw).hexdigest() != EPISODE_SHA:
        raise ValueError('Only the exact preserved installed-smoke episode is accepted')
    episode = read_gzip(path, 32*1024*1024)
    if (episode.get('seed') != 261517 or len(episode.get('actions', [])) != 1000
            or len(episode.get('trajectory', [])) != 1001
            or episode.get('config', {}).get('timestep') != .001
            or episode['config'].get('control_dt') != .02
            or episode['config'].get('timeout_s') != 20.):
        raise ValueError('Preserved tape dimensions, seed or cadence differ')
    commands = []
    for index, action in enumerate(episode['actions']):
        if (set(action) != {'t_seconds', 'joint_targets_rad', 'gripper_position_m'}
                or type(action['joint_targets_rad']) is not list
                or len(action['joint_targets_rad']) != 6
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       for value in action['joint_targets_rad'])
                or type(action['gripper_position_m']) not in (int, float)
                or not 0 <= action['gripper_position_m'] <= .035
                or not math.isclose(action['t_seconds'], index*.02, rel_tol=0, abs_tol=1e-8)):
            raise ValueError('Malformed preserved command tape')
        commands.append({key: action[key] for key in ('joint_targets_rad', 'gripper_position_m')})
    return episode, commands, identity(raw)


def exact_document(left, right, fields=None):
    keys = sorted(set(left) | set(right)) if fields is None else fields
    different = [key for key in keys if key not in left or key not in right
                 or canonical(left[key]) != canonical(right[key])]
    return {'exact_match': not different, 'different_fields': different, 'fields_compared': list(keys)}


def compare_arrays(left, right):
    """Bitwise numeric equality, including signed zero; never allow pickle."""
    import numpy as np
    differences = []
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        for key in sorted(set(a.files) | set(b.files)):
            if key not in a.files or key not in b.files:
                differences.append({'field': key, 'reason': 'missing_array'})
                continue
            x, y = a[key], b[key]
            if x.dtype.kind not in 'iufb' or y.dtype.kind not in 'iufb':
                raise ValueError('Dense evidence must contain numeric arrays only')
            if x.shape != y.shape or x.dtype != y.dtype:
                differences.append({'field': key, 'reason': 'shape_or_dtype'})
            elif x.tobytes() != y.tobytes():
                # Byte comparison catches signed-zero differences too.
                byte_diff = np.flatnonzero(x.view(np.uint8).reshape(-1) != y.view(np.uint8).reshape(-1))
                row_bytes = (x[0].nbytes if x.ndim and len(x) else x.dtype.itemsize)
                differences.append({'field': key, 'reason': 'bitwise_mismatch',
                                    'first_row': int(byte_diff[0]//max(row_bytes, 1))})
    return {'exact_match': not differences, 'differences': differences}


def _diagnostic_class(base, np, mj):
    class Dense(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._dense_arrays = {}
            self._dense_rows = []

        def _outcome_tick(self, **kwargs):
            super()._outcome_tick(**kwargs)
            self.capture(kwargs)

        def capture(self, scorer_inputs=None):
            arrays = {name: np.asarray(getattr(self.data, name)).copy() for name in NATIVE_ARRAYS}
            integration = np.empty(mj.mj_stateSize(self.model, mj.mjtState.mjSTATE_INTEGRATION))
            mj.mj_getState(self.model, self.data, integration, mj.mjtState.mjSTATE_INTEGRATION)
            arrays.update(integration_state=integration,
                          native_time=np.asarray(self.data.time, dtype=np.float64),
                          command_targets=np.asarray(self.command_targets).copy(),
                          command_gripper=np.asarray(self.command_gripper, dtype=np.float64))
            for name, value in arrays.items():
                self._dense_arrays.setdefault(name, []).append(value)
            counts, forces, supported, critical, pairs = self._contacts()
            raw_contacts = []
            for index, contact in enumerate(self.data.contact[:self.data.ncon]):
                force = np.zeros(6)
                mj.mj_contactForce(self.model, self.data, index, force)
                raw_contacts.append({'geom_ids': [int(contact.geom1), int(contact.geom2)],
                                     'dim': int(contact.dim), 'efc_address': int(contact.efc_address),
                                     'distance_m': float(contact.dist), 'position_m': contact.pos.tolist(),
                                     'frame': contact.frame.tolist(), 'friction': contact.friction.tolist(),
                                     'force_native': force.tolist()})
            self._dense_rows.append({
                'native_tick': len(self._dense_rows),
                'contacts': {'counts': counts, 'forces': forces, 'table': supported,
                             'critical': critical, 'active_pairs': pairs, 'raw': raw_contacts},
                'scorer_inputs': ({name: value.item() if isinstance(value, np.generic) else value
                                   for name, value in scorer_inputs.items()}
                                  if scorer_inputs is not None else None),
                'scorer': {name: getattr(self, name) for name in SCORER_FIELDS},
                'contact_pair_native_ticks': dict(self.contact_pair_native_ticks),
                'warnings': list(self.warnings),
                'native_warnings': [{'number': int(row.number), 'lastinfo': int(row.lastinfo)}
                                    for row in self.data.warning]})

        def save_dense(self, folder):
            # Each row was copied while current; no views or pickle cross ticks.
            arrays = {name: np.asarray(rows) for name, rows in self._dense_arrays.items()}
            with (folder/'native-arrays.npz').open('xb') as stream:
                np.savez_compressed(stream, **arrays)
            write_gzip(folder/'native-contacts-scorer.json.gz', self._dense_rows)
            return {'rows': len(self._dense_rows),
                    'array_fields': list(arrays),
                    'sample_timing': 'Initial post-reset state, then each post-physics/post-scoring native 1 ms tick; control-step bookkeeping is also compared at 20 ms boundaries.',
                    'overhead': 'Diagnostic callbacks copy state and re-read contacts; all diagnostic runs are excluded from the performance comparison.'}
    return Dense


def worker(args):
    folder = args.output
    protocol_raw = args.protocol.read_bytes()
    protocol = decode(protocol_raw)
    if sha256(protocol_raw).hexdigest() != args.protocol_sha:
        raise ValueError('Protocol changed before worker launch')
    root = args.source_root.resolve()
    frozen = protocol['sources'][args.variant]
    check_inventory(root, frozen)
    # -s/-P and PYTHONPATH isolate the requested source. Fail closed if an
    # editable install or external environment redirected package resolution.
    from dvidia_training import piper_task, reference_sim as sim, robot_profiles
    from dvidia_training.piper_imitation_score import score_episode
    import numpy as np
    import mujoco as mj
    if Path(piper_task.__file__).resolve() != root/'src/dvidia_training/piper_task.py':
        raise ValueError('Worker imported a different source checkout')
    expected, commands, tape_id = load_tape(args.episode)
    if tape_id != protocol['episode']:
        raise ValueError('Tape changed after protocol freeze')
    if identity(Path(sys.executable).read_bytes()) != protocol['interpreter']:
        raise ValueError('Worker interpreter differs from protocol')
    base = piper_task.PiperTaskEnv
    kwargs = {}
    if args.case.startswith('feedback-'):
        from dvidia_training.piper_feedback_env import FeedbackPiperTaskEnv, JawPulse
        base = FeedbackPiperTaskEnv
        kwargs['pulses'] = (JawPulse(**PULSE),) if args.case == 'feedback-pulse' else ()
    if args.mode == 'diagnostic':
        base = _diagnostic_class(base, np, mj)
    env = base(args.assets, piper_task.PiperTaskConfig(**expected['config']), **kwargs)
    episode = {'config': asdict(env.config), 'seed': expected['seed'], 'trajectory': [],
               'actions': [], 'error': None}
    timing = {'step_wall_samples_s': [], 'step_cpu_samples_s': []}
    result = {'schema': SCHEMA, 'variant': args.variant, 'mode': args.mode, 'case': args.case,
              'protocol_sha256': args.protocol_sha, 'status': 'started', 'scope': SCOPE,
              'runtime': {'python': platform.python_version(), 'machine': platform.machine(),
                          'system': platform.system(), 'mujoco': mj.__version__, 'numpy': np.__version__}}
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    try:
        setup = time.perf_counter()
        observation, info = env.reset(expected['seed'])
        timing['setup_reset_wall_s'] = time.perf_counter()-setup
        episode.update(initial_object_position_m=env.initial_object_position.tolist(),
                       compiled_model_sha256=sim.model_digest(env.model))
        result['assets'] = robot_profiles.verify_assets('piper', args.assets)
        result['model'] = {'sha256': episode['compiled_model_sha256'],
                           'dimensions': {name: int(getattr(env.model, name))
                                          for name in ('nq', 'nv', 'nu', 'neq', 'nbody', 'ngeom')},
                           'timestep_s': float(env.model.opt.timestep),
                           'native_gains': env.model.actuator_gainprm.tolist(),
                           'native_bias': env.model.actuator_biasprm.tolist(),
                           'native_force_limits': env.model.actuator_forcerange.tolist(),
                           'native_control_limits': env.model.actuator_ctrlrange.tolist()}
        if episode['compiled_model_sha256'] != expected['compiled_model_sha256']:
            raise ValueError('Compiled model differs from preserved tape')
        episode['trajectory'].append(observation)
        if args.mode == 'diagnostic':
            env.capture()
        hold_targets = observation['joint_position_rad'][:]
        for saved in commands:
            action = {'joint_targets_rad': saved['joint_targets_rad'][:],
                      'gripper_position_m': saved['gripper_position_m']}
            if args.case == 'open-jaw':
                action['gripper_position_m'] = .035
            elif args.case == 'hold':
                action = {'joint_targets_rad': hold_targets[:], 'gripper_position_m': .035}
            record = {'t_seconds': observation['t_seconds'], **action}
            episode['actions'].append(record)
            observation.update(issued_joint_targets_rad=action['joint_targets_rad'][:],
                               issued_gripper_position_m=action['gripper_position_m'], command=record)
            wall, cpu = time.perf_counter(), time.process_time()
            observation, info = env.step(action)
            timing['step_wall_samples_s'].append(time.perf_counter()-wall)
            timing['step_cpu_samples_s'].append(time.process_time()-cpu)
            episode['trajectory'].append(observation)
            if env.done:
                break
        episode.update(valid=info['valid'], success=info['success'], reason=info['reason'],
                       simulated_seconds=info['simulation_time'],
                       metrics={**info, 'contact_pair_native_ticks': dict(env.contact_pair_native_ticks)})
        result['score'] = score_episode(episode, expected['config']['target_position_m'][:2])
        result['boundary_comparison_to_preserved'] = exact_document(expected, episode, EPISODE_FIELDS)
        if len(episode['actions']) != 1000 or len(episode['trajectory']) != 1001:
            raise ValueError('Attempt did not complete its fixed command budget')
        if not info['valid'] or info['warnings']:
            raise ValueError('Native attempt invalid or emitted warnings')
        if args.case in ('original', 'feedback-no-pulse') and not result['boundary_comparison_to_preserved']['exact_match']:
            raise ValueError('Original/no-pulse mechanics differs from preserved episode')
        if args.case in ('open-jaw', 'hold') and result['score']['success']:
            raise ValueError('Negative control unexpectedly passed the independent scorer')
        check_inventory(root, frozen)
        result['status'] = 'complete'
    except Exception as exc:
        episode['error'] = type(exc).__name__+': '+str(exc)
        result.update(status='failed', error=episode['error'])
        (folder/'worker-traceback.txt').write_text(traceback.format_exc())
    finally:
        timing.update(worker_wall_s=time.perf_counter()-start_wall,
                      worker_process_cpu_s=time.process_time()-start_cpu)
        result['timing'] = {**timing,
                           'environment_step_wall': summary(timing['step_wall_samples_s']),
                           'environment_step_cpu': summary(timing['step_cpu_samples_s']),
                           'scope': 'Unpaced shared-host CPU compute. Step includes validation, 20 native 1 ms integrations, safety/contact/scoring and returned observation/info; no policy inference. Reset includes asset checks, model compile and settling. Worker total also includes provenance/record bookkeeping. Diagnostic samples are never used as speed evidence.'}
        if args.mode == 'diagnostic':
            result['dense'] = env.save_dense(folder)
        write_gzip(folder/'episode.json.gz', episode)
        try:
            check_inventory(root, frozen)
            result['source_unchanged'] = True
        except Exception as exc:
            result.update(status='failed', source_unchanged=False, source_error=str(exc))
        write(folder/'result.json', result)
    return 0 if result['status'] == 'complete' else 1


def compare_attempts(left, right, diagnostic=False):
    a, b = decode((left/'result.json').read_bytes()), decode((right/'result.json').read_bytes())
    checks = {'both_complete': a['status'] == b['status'] == 'complete',
              'same_model': exact_document(a.get('model', {}), b.get('model', {})),
              'same_runtime': exact_document(a.get('runtime', {}), b.get('runtime', {})),
              'same_assets': exact_document(a.get('assets', {}), b.get('assets', {})),
              'same_score': exact_document(a.get('score', {}), b.get('score', {})),
              'same_boundaries': exact_document(read_gzip(left/'episode.json.gz'),
                                                read_gzip(right/'episode.json.gz'), EPISODE_FIELDS)}
    if diagnostic:
        checks['same_native_arrays'] = compare_arrays(left/'native-arrays.npz', right/'native-arrays.npz')
        x, y = read_gzip(left/'native-contacts-scorer.json.gz'), read_gzip(right/'native-contacts-scorer.json.gz')
        first = next((index for index, (p, q) in enumerate(zip(x, y)) if canonical(p) != canonical(q)), None)
        checks['same_native_contacts_scorer'] = {'exact_match': len(x) == len(y) and first is None,
                                                'baseline_rows': len(x), 'candidate_rows': len(y),
                                                'first_different_row': first}
        checks['complete_native_cadence'] = len(x) == len(y) == 20001
    checks['exact_match'] = all(value if type(value) is bool else value['exact_match']
                                for value in checks.values())
    return checks


def check_inputs(script, protocol_path, protocol_sha, python, protocol, episode_path):
    if identity(Path(script).read_bytes()) != protocol['harness']:
        raise ValueError('Harness changed during the comparison')
    if sha256(Path(protocol_path).read_bytes()).hexdigest() != protocol_sha:
        raise ValueError('Protocol changed during the comparison')
    if identity(Path(python).read_bytes()) != protocol['interpreter']:
        raise ValueError('Interpreter changed during the comparison')
    if identity(Path(episode_path).read_bytes()) != protocol['episode']:
        raise ValueError('Preserved episode changed during the comparison')


def adoption_result(timing_runs, exact_equivalence):
    complete = len(timing_runs) >= 4 and len(timing_runs) % 2 == 0 and all(
        row['status'] == 'complete' for row in timing_runs)
    pair_checks, pooled = [], {}
    if complete:
        for index in range(0, len(timing_runs), 2):
            pair = {row['variant']: row for row in timing_runs[index:index+2]}
            if set(pair) != {'baseline', 'candidate'}:
                raise ValueError('Each timing pair must contain baseline and candidate')
            pair_checks.append({'pair': index//2, **{
                clock: pair['candidate'][clock]['mean_s'] < pair['baseline'][clock]['mean_s']
                for clock in ('step_wall', 'step_cpu')}})
        for clock in ('step_wall', 'step_cpu'):
            means = {}
            for variant in ('baseline', 'candidate'):
                rows = [row[clock] for row in timing_runs if row['variant'] == variant]
                means[variant] = sum(row['sum_s'] for row in rows)/sum(row['count'] for row in rows)
            pooled[clock] = {'mean_s': means,
                             'candidate_mean_reduction_percent': 100*(1-means['candidate']/means['baseline'])}
    eligible = (exact_equivalence and complete
                and all(row[clock] for row in pair_checks for clock in ('step_wall', 'step_cpu'))
                and all(row['candidate_mean_reduction_percent'] >= 5. for row in pooled.values()))
    return {'rule': ADOPTION_RULE, 'eligible_under_predeclared_rule': bool(eligible),
            'paired_lower_mean': pair_checks, 'pooled': pooled,
            'decision': 'Eligible for parent review' if eligible else 'Do not promote runtime change'}


def run(args):
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError('Output must be new; no evidence is overwritten')
    order = schedule(args.pairs)
    roots = {'baseline': args.baseline_root.resolve(), 'candidate': args.candidate_root.resolve()}
    if roots['baseline'] == roots['candidate']:
        raise ValueError('Baseline and candidate require independent source checkouts')
    sources = {name: inventory(root) for name, root in roots.items()}
    differences = source_differences(sources['baseline'], sources['candidate'])
    expected, commands, tape_id = load_tape(args.episode)
    output.mkdir(parents=True)
    script = Path(__file__).resolve()
    # Preserve a venv executable's spelling: resolving its symlink would launch
    # the global interpreter and silently lose that environment's dependencies.
    python = args.python.absolute()
    protocol = {'schema': SCHEMA, 'scope': SCOPE, 'sources': sources,
                'source_differences': differences, 'allowed_source_differences': list(ALLOWED_SOURCE_DIFFERENCES),
                'harness': identity(script.read_bytes()), 'interpreter': identity(python.read_bytes()),
                'episode': tape_id, 'commands_sha256': sha256(canonical(commands)).hexdigest(),
                'seed': expected['seed'], 'config': expected['config'],
                'timing_order': order, 'timing_pairs': args.pairs,
                'diagnostic_cases': list(CASES), 'diagnostic_pulse': PULSE,
                'adoption_rule': ADOPTION_RULE,
                'dense_native_state_fields': list(NATIVE_ARRAYS)+['integration_state', 'native_time', 'command_targets', 'command_gripper'],
                'dense_scorer_fields': list(SCORER_FIELDS),
                'comparison': 'Exact boundary JSON, bitwise numeric native arrays, exact ordered raw/active contacts and cumulative scorer. No timing or source hashes are present in mechanics documents; their identities are compared separately.',
                'timing_design': 'Two or more fixed matched pairs in alternating AB/BA order; no retry, tuning, warm-up selection, dropped attempts or best-run selection. Diagnostic callbacks run only after the independent performance schedule.'}
    write(output/'protocol.json', protocol)
    protocol_sha = sha256((output/'protocol.json').read_bytes()).hexdigest()
    (output/'preserved-episode.json.gz').write_bytes(args.episode.read_bytes())
    for variant, root in roots.items():
        destination = output/'sources'/variant
        destination.mkdir(parents=True)
        for name in SOURCE_FILES:
            (destination/name).write_bytes((root/'src/dvidia_training'/name).read_bytes())
    (output/'compare.py').write_bytes(script.read_bytes())
    planned = [{'id': f'timing-{index:02d}-{variant}', 'variant': variant, 'mode': 'timing', 'case': 'original'}
               for index, variant in enumerate(order)]
    planned += [{'id': f'dense-{case}-{variant}', 'variant': variant, 'mode': 'diagnostic', 'case': case}
                for case in CASES for variant in ('baseline', 'candidate')]
    write(output/'schedule.json', planned)
    attempts = []
    for row in planned:
        destination = output/row['id']
        destination.mkdir()
        write(destination/'started.json', {**row, 'protocol_sha256': protocol_sha})
        status = {**row, 'returncode': None}
        try:
            for variant, root in roots.items():
                check_inventory(root, sources[variant])
            check_inputs(script, output/'protocol.json', protocol_sha, python, protocol,
                         output/'preserved-episode.json.gz')
            command = [str(python), '-s', '-P', str(script), '--worker',
                       '--source-root', str(roots[row['variant']]), '--variant', row['variant'],
                       '--protocol', str(output/'protocol.json'), '--protocol-sha', protocol_sha,
                       '--episode', str(output/'preserved-episode.json.gz'), '--assets', str(args.assets.resolve()),
                       '--output', str(destination), '--mode', row['mode'], '--case', row['case']]
            environment = {key: value for key, value in os.environ.items()
                           if key not in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP')}
            environment.update(PYTHONPATH=str(roots[row['variant']]/'src'), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
            wall = time.perf_counter()
            with (destination/'stdout.txt').open('xb') as stdout, (destination/'stderr.txt').open('xb') as stderr:
                completed = subprocess.run(command, cwd=destination, env=environment, stdout=stdout,
                                           stderr=stderr, timeout=args.worker_timeout, check=False)
            status.update(returncode=completed.returncode, subprocess_wall_s=time.perf_counter()-wall)
            for variant, root in roots.items():
                check_inventory(root, sources[variant])
            check_inputs(script, output/'protocol.json', protocol_sha, python, protocol,
                         output/'preserved-episode.json.gz')
        except Exception as exc:
            status['error'] = type(exc).__name__+': '+str(exc)
        write(destination/'process.json', status)
        attempts.append(status)
        print(json.dumps({'attempt': row['id'], 'returncode': status['returncode'],
                          'error': status.get('error')}), flush=True)
    comparisons = []
    for index in range(0, len(order), 2):
        ids = planned[index:index+2]
        by_variant = {row['variant']: output/row['id'] for row in ids}
        try:
            result = compare_attempts(by_variant['baseline'], by_variant['candidate'])
        except Exception as exc:
            result = {'exact_match': False, 'error': type(exc).__name__+': '+str(exc)}
        comparisons.append({'kind': 'timing-pair', 'pair': index//2, **result})
    for case in CASES:
        try:
            result = compare_attempts(output/f'dense-{case}-baseline', output/f'dense-{case}-candidate', True)
        except Exception as exc:
            result = {'exact_match': False, 'error': type(exc).__name__+': '+str(exc)}
        comparisons.append({'kind': 'dense-equivalence', 'case': case, **result})
    timing_runs = []
    for row in planned[:len(order)]:
        try:
            result = decode((output/row['id']/'result.json').read_bytes())
            timing_runs.append({'id': row['id'], 'variant': row['variant'], 'status': result['status'],
                                'step_wall': result['timing']['environment_step_wall'],
                                'step_cpu': result['timing']['environment_step_cpu'],
                                'setup_reset_wall_s': result['timing'].get('setup_reset_wall_s')})
        except Exception:
            timing_runs.append({'id': row['id'], 'variant': row['variant'], 'status': 'missing-result'})
    paired = []
    for index in range(0, len(timing_runs), 2):
        pair = {row['variant']: row for row in timing_runs[index:index+2]}
        if all(row['status'] == 'complete' for row in pair.values()):
            a, b = pair['baseline']['step_wall'], pair['candidate']['step_wall']
            paired.append({'pair': index//2, 'baseline_sum_s': a['sum_s'], 'candidate_sum_s': b['sum_s'],
                           'candidate_total_step_reduction_percent': 100*(1-b['sum_s']/a['sum_s']),
                           'candidate_p95_reduction_percent': 100*(1-b['p95_s']/a['p95_s'])})
    unchanged = all(inventory(roots[name]) == frozen for name, frozen in sources.items())
    try:
        check_inputs(script, output/'protocol.json', protocol_sha, python, protocol,
                     output/'preserved-episode.json.gz')
    except Exception:
        unchanged = False
    report = {'schema': SCHEMA, 'protocol_sha256': protocol_sha, 'scope': SCOPE,
              'attempts': attempts, 'planned_attempts': len(planned), 'source_unchanged': unchanged,
              'comparisons': comparisons, 'timing_runs': timing_runs, 'paired_timing': paired,
              'all_passed': unchanged and all(row.get('returncode') == 0 and 'error' not in row for row in attempts)
                            and all(row['exact_match'] for row in comparisons),
              'interpretation': 'These unpaced shared-host CPU samples compare one unchanged synthetic command tape. No inference time is included. Do not infer new task accuracy, hardware response, minimum hardware or real-time safety. Report every paired result; two pairs cannot establish robust population speedup.'}
    report['adoption'] = adoption_result(timing_runs, report['all_passed'])
    write(output/'comparison.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--baseline-root', type=Path)
    parser.add_argument('--candidate-root', type=Path)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=2)
    parser.add_argument('--worker-timeout', type=float, default=300.)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--variant', choices=('baseline', 'candidate'))
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--protocol-sha')
    parser.add_argument('--mode', choices=('timing', 'diagnostic'))
    parser.add_argument('--case', choices=CASES)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args)
    if args.baseline_root is None or args.candidate_root is None:
        parser.error('Both independent source roots are required')
    report = run(args)
    print(json.dumps({'all_passed': report['all_passed'], 'attempts': len(report['attempts']),
                      'paired_timing': report['paired_timing']}), flush=True)
    return 0 if report['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

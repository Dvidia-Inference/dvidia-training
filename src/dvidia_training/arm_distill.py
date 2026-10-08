"""A data-only learned movement head, supervised by an authored placement policy.

The student replaces differential IK only. Contact/phase/gripper decisions remain
authored, observations remain privileged, and no teacher fallback is implicit.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import time

import numpy as np

from .arm_env import ArmConfig, ArmEnv, JOINT_LIMITS
from .arm_policy import ArmPickPlacePolicy
from .cli import NetworkGuard
from .skillspace import ARM_PROFILE

MAX_MODEL_BYTES = 128 * 1024
FEATURE_CONTRACT = 'q6_object_minus_tcp3_target_minus_tcp3__capped_goal_delta3_upright_orientation_delta3_v1'
MODEL_FIELDS = {'schema_version', 'kind', 'scope', 'status', 'physical_robot_ready',
                'arm_profile', 'feature_contract', 'centers', 'context_mean',
                'context_scale', 'kernel_width', 'weights', 'dataset_sha256',
                'train_cases', 'training'}
ERROR_SCALE = np.array([.006] * 3 + [.04] * 3, dtype=np.float32)


def canonical(document):
    return json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')


def digest(document):
    return sha256(canonical(document)).hexdigest()


def _decode(document):
    if type(document) is dict:
        return json.loads(canonical(document))
    if isinstance(document, (bytes, str)):
        raw = document.encode('utf8') if isinstance(document, str) else document
        if len(raw) > MAX_MODEL_BYTES:
            raise ValueError('Movement head exceeds 128 KiB.')
        def pairs(items):
            row = {}
            for key, value in items:
                if key in row:
                    raise ValueError('Duplicate movement-head field.')
                row[key] = value
            return row
        def nonfinite(value):
            raise ValueError('Nonfinite movement-head value.')
        try:
            return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
        except (UnicodeError, RecursionError) as exc:
            raise ValueError('Movement head must be finite UTF-8 JSON.') from exc
    raise ValueError('Movement head must be a plain dict or JSON.')


def _array(value, shape, label):
    def numbers(row):
        if type(row) is list:
            return all(numbers(x) for x in row)
        return type(row) in (int, float) and math.isfinite(row)
    if not numbers(value):
        raise ValueError(f'{label} must contain finite numbers.')
    try:
        result = np.asarray(value, dtype=np.float32)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f'{label} has invalid dimensions.') from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'{label} has invalid dimensions or float32 range.')
    return result


def validate_model(document):
    """Return a JSON-only candidate; integrity does not establish task mastery."""
    model = _decode(document)
    if type(model) is not dict or set(model) != MODEL_FIELDS:
        raise ValueError('Unknown or missing movement-head fields.')
    expected = {'schema_version': 1, 'kind': 'dvidia.numpy-rbf-movement-head',
                'scope': 'simulation-only', 'status': 'candidate',
                'physical_robot_ready': False, 'arm_profile': ARM_PROFILE,
                'feature_contract': FEATURE_CONTRACT}
    if any(type(model.get(k)) is not type(v) or model.get(k) != v for k, v in expected.items()):
        raise ValueError('Unsupported movement-head profile or scope.')
    if type(model['centers']) is not list or not 8 <= len(model['centers']) <= 48:
        raise ValueError('Movement head requires 8–48 RBF centers.')
    count = len(model['centers'])
    _array(model['centers'], (count, 12), 'centers')
    _array(model['context_mean'], (12,), 'context_mean')
    scale = _array(model['context_scale'], (12,), 'context_scale')
    if np.any(scale < 1e-4):
        raise ValueError('Context scales must be at least 1e-4.')
    _array(model['weights'], (count, 6, 6), 'weights')
    if type(model['kernel_width']) not in (int, float) or not math.isfinite(model['kernel_width']) or not .05 <= model['kernel_width'] <= 20:
        raise ValueError('Kernel width must lie in [0.05, 20].')
    ds = model['dataset_sha256']
    if type(ds) is not str or len(ds) != 64 or any(c not in '0123456789abcdef' for c in ds):
        raise ValueError('A dataset SHA256 is required.')
    cases = model['train_cases']
    if type(cases) is not list or not 1 <= len(cases) <= 64 or any(type(x) is not str or not x or len(x) > 80 for x in cases) or len(set(cases)) != len(cases):
        raise ValueError('Training case IDs must be distinct short strings.')
    training = model['training']
    if type(training) is not dict or set(training) != {'algorithm', 'sample_count', 'regularization', 'seed'}:
        raise ValueError('Invalid training receipt.')
    if training['algorithm'] != 'normalized-rbf-ridge' or type(training['sample_count']) is not int or not 32 <= training['sample_count'] <= 100000:
        raise ValueError('Invalid supervised sample count or algorithm.')
    if type(training['regularization']) not in (int, float) or not math.isfinite(training['regularization']) or not 1e-10 <= training['regularization'] <= 10:
        raise ValueError('Invalid ridge regularization.')
    if type(training['seed']) is not int or not 0 <= training['seed'] <= 999999:
        raise ValueError('Invalid training seed.')
    if len(canonical(model)) > MAX_MODEL_BYTES:
        raise ValueError('Movement head exceeds 128 KiB.')
    return model


def features(observation, goal):
    """Context and bounded pose error, computed without a Jacobian or IK call."""
    q = _array(observation['joint_position'], (6,), 'joint_position')
    tcp = _array(observation['end_effector_position'], (3,), 'end_effector_position')
    obj = _array(observation['object_position'], (3,), 'object_position')
    target = _array(observation['target_position'], (3,), 'target_position')
    goal = _array(goal.tolist() if isinstance(goal, np.ndarray) else list(goal), (3,), 'goal')
    quat = _array(observation['end_effector_quaternion'], (4,), 'end_effector_quaternion')
    norm = float(np.linalg.norm(quat))
    if not .99 <= norm <= 1.01:
        raise ValueError('TCP orientation must be a unit quaternion.')
    w, x, y, z = quat / norm
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], dtype=np.float32)
    orientation = np.array([rotation[1, 2]-rotation[2, 1],
                            rotation[2, 0]-rotation[0, 2],
                            rotation[0, 1]-rotation[1, 0]], dtype=np.float32) * .5
    translation = goal - tcp
    for error, cap in ((translation, .006), (orientation, .04)):
        length = float(np.linalg.norm(error))
        if length > cap:
            error *= cap / length
    context = np.concatenate((q, obj-tcp, target-tcp))
    error = np.concatenate((translation, orientation)) / ERROR_SCALE
    return context, error


def _kernel(context, centers, width):
    distance = np.sum((context[..., None, :]-centers) ** 2, axis=-1)
    logits = -distance / (2 * width * width)
    logits -= np.max(logits, axis=-1, keepdims=True)
    weights = np.exp(logits)
    return weights / np.sum(weights, axis=-1, keepdims=True)


class MovementHead:
    def __init__(self, document):
        self.model = validate_model(document)
        self.centers = np.asarray(self.model['centers'], dtype=np.float32)
        self.mean = np.asarray(self.model['context_mean'], dtype=np.float32)
        self.scale = np.asarray(self.model['context_scale'], dtype=np.float32)
        self.weights = np.asarray(self.model['weights'], dtype=np.float32).reshape(-1, 6)
        self.width = self.model['kernel_width']
        self.calls = 0
        self.inference_seconds = 0.

    def predict(self, observation, goal):
        started = time.perf_counter()
        context, error = features(observation, goal)
        kernel = _kernel((context-self.mean)/self.scale, self.centers, self.width)
        design = (kernel[:, None] * error[None, :]).reshape(-1)
        delta = np.clip(design @ self.weights, -.06, .06)
        limits = np.asarray(JOINT_LIMITS)
        result = np.clip(context[:6] + delta, limits[:, 0]+.02, limits[:, 1]-.02).tolist()
        self.calls += 1
        self.inference_seconds += time.perf_counter()-started
        return result


class _HeadEnvironment:
    def __init__(self, environment, head):
        self.environment, self.head, self.observation = environment, head, None

    def __getattr__(self, name):
        return getattr(self.environment, name)

    def joint_targets_for_pose(self, goal):
        return self.head.predict(self.observation, goal)


class DistilledPlacementPolicy:
    """Fresh instance per episode; learned joints and authored phase/gripper logic."""
    def __init__(self, environment, model):
        self.head = MovementHead(model)
        self.proxy = _HeadEnvironment(environment, self.head)
        self.supervisor = ArmPickPlacePolicy(self.proxy)

    @property
    def stage(self):
        return self.supervisor.stage

    @property
    def failure_reason(self):
        return self.supervisor.failure_reason

    def __call__(self, observation):
        self.proxy.observation = observation
        return self.supervisor(observation)

    def diagnostics(self):
        return {**self.supervisor.diagnostics(), 'policy_origin': 'learned movement head with authored phase/contact/gripper supervisor',
                'movement_head_sha256': digest(self.head.model), 'teacher_fallback_calls': 0,
                'student_inference_calls': self.head.calls,
                'student_inference_seconds': self.head.inference_seconds}


def make_protocol(seed=731927):
    """New layouts: development train/selection and a sealed one-time evaluation."""
    rng = np.random.default_rng(seed)
    seen = set()
    def cases(label, count, stress=0):
        rows = []
        while len(rows) < count:
            xy = np.round(rng.uniform([.27, -.17, .29, -.17], [.53, .17, .54, .17]), 6)
            if np.linalg.norm(xy[:2]) > .54 or np.linalg.norm(xy[2:]) > .54 or np.linalg.norm(xy[:2]-xy[2:]) < .10:
                continue
            size = rng.choice([.03, .035, .04, .045, .05], 3).tolist()
            z = .29 + size[2]/2
            scene = {'object_position': [*xy[:2].tolist(), z], 'target_position': [*xy[2:].tolist(), z],
                     'object_half_size': [x/2 for x in size], 'object_mass': float(rng.choice([.02, .04, .06, .08])),
                     'object_friction': float(rng.choice([.4, .8, 1.2, 1.5])), 'scene_jitter': 0.,
                     'jaw_force_limit': 15., 'joint_torque_limit': 40.}
            stressed = len(rows) >= count-stress
            if stressed:
                scene.update(jaw_force_limit=.3, joint_torque_limit=5.)
            key = digest(scene)
            if key in seen:
                continue
            seen.add(key)
            ArmConfig(**scene)
            index = len(rows)
            rows.append({'id': f'capsule-{label}-{index:02}', 'scene_sha256': key,
                         'seed': seed+index, 'group': 'actuator_stress' if stressed else 'nominal', 'config': scene})
        return rows
    return {'schema_version': 1, 'kind': 'movement-head-distillation-v1', 'scope': 'simulation-only',
            'seed': seed, 'development': cases('train', 16), 'selection': cases('dev', 4),
            'evaluation': cases('eval', 8, stress=2),
            'rules': {'historical_v03_heldout': 'untouched', 'selection': 'Development-only architecture/regularization selection; freeze weights before evaluation.',
                      'evaluation': 'One teacher/student paired evaluation on new disjoint scene digests; no tuning after results.',
                      'observations': 'privileged simulator state', 'teacher_fallback': 'none'}}


def _source_hashes():
    root = Path(__file__).parent
    return {name: sha256((root/name).read_bytes()).hexdigest() for name in ('arm_env.py', 'arm_policy.py', 'arm_distill.py', 'env.py')}


class _RecordingEnvironment:
    def __init__(self, env, samples, case_id, rng, stride, augment):
        self.environment, self.samples, self.case_id = env, samples, case_id
        self.rng, self.stride, self.augment = rng, stride, augment
        self.observation = None
        self.calls, self.ik_seconds = 0, 0.
        self.phase = 'observe'
        self.phase_ik = defaultdict(lambda: {'calls': 0, 'seconds': 0.})

    def __getattr__(self, name):
        return getattr(self.environment, name)

    def joint_targets_for_pose(self, goal):
        started = time.perf_counter()
        targets = self.environment.joint_targets_for_pose(goal)
        elapsed = time.perf_counter()-started
        self.ik_seconds += elapsed
        self.phase_ik[self.phase]['calls'] += 1
        self.phase_ik[self.phase]['seconds'] += elapsed
        if self.samples is not None and self.calls % self.stride == 0:
            self._record(goal, targets, 'native_teacher_call')
            for _ in range(self.augment):
                extra_goal = np.asarray(goal) + self.rng.normal(0., .008, 3)
                self._record(extra_goal, self.environment.joint_targets_for_pose(extra_goal), 'teacher_goal_augmentation')
        self.calls += 1
        return targets

    def _record(self, goal, targets, kind):
        context, error = features(self.observation, goal)
        delta = np.asarray(targets)-np.asarray(self.observation['joint_position'])
        self.samples.append({'case_id': self.case_id, 'kind': kind,
                             'context': context.tolist(), 'error': error.tolist(), 'delta': delta.tolist()})


def run_episode(case, model=None, *, record_samples=None, stride=3, augment=1):
    started = time.perf_counter()
    env = ArmEnv(ArmConfig(**case['config']))
    observation, info = env.reset(seed=case['seed'])
    reset = time.perf_counter()-started
    if model is None:
        proxy = _RecordingEnvironment(env, record_samples, case['id'], np.random.default_rng(case['seed']), stride, augment)
        policy = ArmPickPlacePolicy(proxy)
    else:
        proxy = None
        policy = ArmPickPlacePolicy(env) if model is None else DistilledPlacementPolicy(env, model)
    phases = defaultdict(lambda: {'calls': 0, 'policy_seconds': 0., 'native_step_seconds': 0.})
    info = {'success': False, 'valid': True}
    while True:
        phase = policy.stage
        if proxy is not None:
            proxy.observation = observation
            proxy.phase = phase
        before = time.perf_counter()
        action = policy(observation)
        after_policy = time.perf_counter()
        observation, reward, terminated, truncated, info = env.step(action)
        after_step = time.perf_counter()
        phases[phase]['calls'] += 1
        phases[phase]['policy_seconds'] += after_policy-before
        phases[phase]['native_step_seconds'] += after_step-after_policy
        if terminated or truncated:
            break
    result = {'id': case['id'], 'scene_sha256': case['scene_sha256'], 'group': case['group'],
              'success': info['success'], 'valid': info['valid'], 'reason': info['reason'],
              'target_distance_m': info['target_distance'], 'control_steps': info['steps'],
              'simulated_seconds': info['simulation_time'], 'reset_seconds': reset,
              'episode_wall_seconds': time.perf_counter()-started, 'phase_profile': dict(phases),
              'policy_diagnostics': policy.diagnostics(), 'final_info': info,
              'final_object_position': observation['object_position']}
    if proxy is not None:
        result.update(teacher_ik_calls=proxy.calls, teacher_ik_seconds=proxy.ik_seconds,
                      teacher_ik_phase_profile=dict(proxy.phase_ik),
                      collection_overhead=('policy timing includes recording and teacher-query augmentation; not used for clean speed comparison'
                                           if record_samples is not None else 'no recording or teacher-query augmentation'))
    serialization_started = time.perf_counter()
    serialized = canonical(result)
    result.update(serialization_seconds=time.perf_counter()-serialization_started,
                  serialized_bytes_before_timing_fields=len(serialized),
                  serialization_scope='compact episode receipt only; excludes full observation/action traces, model and file I/O')
    return result


def collect(cases, *, protocol_sha256=None, stride=3, augment=1):
    if type(stride) is not int or not 1 <= stride <= 20 or type(augment) is not int or not 0 <= augment <= 4:
        raise ValueError('Invalid collection stride or augmentation count.')
    samples = []
    with NetworkGuard(True) as guard:
        results = [run_episode(case, record_samples=samples, stride=stride, augment=augment) for case in cases]
    dataset = {'schema_version': 1, 'kind': 'teacher-movement-supervision', 'scope': 'simulation-only',
               'protocol_sha256': protocol_sha256, 'teacher_files_sha256': _source_hashes(),
               'feature_contract': FEATURE_CONTRACT, 'scene_ids': [c['id'] for c in cases],
               'scene_sha256': [c['scene_sha256'] for c in cases], 'samples': samples,
               'label': 'authored differential IK target minus current joint position; no video or physical demonstrations'}
    started = time.perf_counter()
    serialized = canonical(dataset)
    serialization = time.perf_counter()-started
    manifest = {'id': 'capsule-teacher-dataset-v1', 'sha256': sha256(serialized).hexdigest(),
                'sample_count': len(samples), 'scene_ids': dataset['scene_ids'], 'scene_sha256': dataset['scene_sha256'],
                'bytes': len(serialized), 'serialization_seconds': serialization}
    return {'dataset': dataset, 'manifest': manifest, 'episodes': results, 'network_guard': guard.record()}


def train(dataset, *, centers=32, regularization=1e-5, seed=491):
    """Fit learned conditional linear pose-to-joint matrices by ridge regression."""
    if type(centers) is not int or not 8 <= centers <= 48:
        raise ValueError('Use 8–48 centers.')
    rows = dataset['samples']
    if not 32 <= len(rows) <= 100000:
        raise ValueError('Use 32–100000 teacher samples.')
    context = np.asarray([r['context'] for r in rows], dtype=np.float64)
    errors = np.asarray([r['error'] for r in rows], dtype=np.float64)
    targets = np.asarray([r['delta'] for r in rows], dtype=np.float64)
    if context.shape != (len(rows), 12) or errors.shape != targets.shape or targets.shape != (len(rows), 6) or not all(np.isfinite(x).all() for x in (context, errors, targets)):
        raise ValueError('Malformed supervised movement samples.')
    mean = context.mean(axis=0)
    scale = np.maximum(context.std(axis=0), .02)
    normalized = (context-mean)/scale
    rng = np.random.default_rng(seed)
    # K-means++ centers; the center positions and all output matrices are learned.
    points = [normalized[rng.integers(len(normalized))]]
    distance = np.sum((normalized-points[0])**2, axis=1)
    for _ in range(1, centers):
        probabilities = distance/distance.sum() if distance.sum() else None
        chosen = normalized[rng.choice(len(normalized), p=probabilities)]
        points.append(chosen)
        distance = np.minimum(distance, np.sum((normalized-chosen)**2, axis=1))
    points = np.asarray(points)
    for _ in range(12):
        labels = np.argmin(np.sum((normalized[:, None, :]-points[None, :, :])**2, axis=-1), axis=1)
        for index in range(centers):
            if np.any(labels == index):
                points[index] = normalized[labels == index].mean(axis=0)
    nearest = np.min(np.sum((normalized[:, None, :]-points[None, :, :])**2, axis=-1), axis=1)
    width = float(np.clip(np.sqrt(np.median(nearest))*1.5, .3, 5.))
    kernel = _kernel(normalized, points, width)
    design = (kernel[:, :, None]*errors[:, None, :]).reshape(len(rows), -1)
    weights = np.linalg.solve(design.T@design + regularization*np.eye(centers*6), design.T@targets)
    def portable(array):
        return np.asarray(array, dtype=np.float32).astype(float).tolist()
    model = {'schema_version': 1, 'kind': 'dvidia.numpy-rbf-movement-head', 'scope': 'simulation-only',
             'status': 'candidate', 'physical_robot_ready': False, 'arm_profile': ARM_PROFILE,
             'feature_contract': FEATURE_CONTRACT, 'centers': portable(points),
             'context_mean': portable(mean), 'context_scale': portable(scale), 'kernel_width': width,
             'weights': portable(weights.reshape(centers, 6, 6)), 'dataset_sha256': digest(dataset),
             'train_cases': dataset['scene_ids'], 'training': {'algorithm': 'normalized-rbf-ridge',
             'sample_count': len(rows), 'regularization': regularization, 'seed': seed}}
    return validate_model(model)


def _write(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(document)+b'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['protocol', 'collect', 'train', 'selection', 'evaluate'])
    parser.add_argument('--output', type=Path, default=Path('runs/skill-capsule-v1'))
    parser.add_argument('--centers', type=int, default=32)
    parser.add_argument('--regularization', type=float, default=1e-5)
    args = parser.parse_args()
    root = args.output
    protocol_file = root/'protocol.json'
    if args.command == 'protocol':
        if protocol_file.exists():
            raise ValueError('Do not overwrite a declared protocol.')
        _write(protocol_file, make_protocol())
        print(json.dumps({'output': str(protocol_file), 'protocol_sha256': digest(make_protocol())}))
        return
    protocol = json.loads(protocol_file.read_text())
    if args.command == 'collect':
        if (root/'dataset.json').exists():
            raise ValueError('Do not overwrite teacher supervision.')
        result = collect(protocol['development'], protocol_sha256=digest(protocol))
        _write(root/'dataset.json', result.pop('dataset'))
        _write(root/'collection.json', result)
        print(json.dumps(result['manifest']))
    elif args.command == 'train':
        if (root/'evaluation-started.json').exists():
            raise ValueError('Evaluation is visible; new tuning requires a new protocol.')
        dataset = json.loads((root/'dataset.json').read_text())
        before = time.perf_counter()
        model = train(dataset, centers=args.centers, regularization=args.regularization)
        _write(root/'model.json', model)
        receipt = {'id': 'capsule-rbf-training-v1', 'dataset_sha256': model['dataset_sha256'],
                   'movement_head_sha256': digest(model), 'algorithm': model['training']['algorithm'],
                   'samples': model['training']['sample_count'], 'training_wall_seconds': time.perf_counter()-before,
                   'model_bytes': len(canonical(model)), 'source_hashes': _source_hashes()}
        _write(root/'training.json', receipt)
        print(json.dumps(receipt))
    else:
        model = validate_model((root/'model.json').read_bytes())
        mode = args.command
        if mode == 'evaluate':
            if (root/'evaluation-started.json').exists():
                raise ValueError('Frozen evaluation has already started; do not tune or repeat it.')
            _write(root/'evaluation-started.json', {'protocol_sha256': digest(protocol), 'movement_head_sha256': digest(model), 'source_hashes': _source_hashes()})
        cases = protocol['evaluation' if mode == 'evaluate' else 'selection']
        with NetworkGuard(True) as guard:
            teacher = [run_episode(c) for c in cases]
            student = [run_episode(c, model) for c in cases]
        result = {'id': f'capsule-{mode}-v1', 'movement_head_sha256': digest(model),
                  'training_dataset_sha256': model['dataset_sha256'], 'protocol_sha256': digest(protocol),
                  'trials': student, 'baseline_trials': teacher, 'network_guard': guard.record(),
                  'source_hashes': _source_hashes(), 'scope': 'learned movement head only; authored phase/contact/gripper supervisor; no teacher fallback; privileged simulation state',
                  'summary': {'teacher_successes': sum(r['success'] for r in teacher),
                              'student_successes': sum(r['success'] for r in student), 'cases': len(cases)}}
        _write(root/f'{mode}.json', result)
        print(json.dumps(result['summary']))


if __name__ == '__main__':
    main()

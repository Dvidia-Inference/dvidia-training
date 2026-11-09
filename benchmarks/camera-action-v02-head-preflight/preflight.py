#!/usr/bin/env python3
"""Synthetic reset-head algebra only; no policy fitting, recording or simulator.

Two candidate decoders share one seeded 269->64->64->7 float32 network.
Cabs = r0 + D*g(x); Cdelta = r + D*g(x). Decoding and endpoint loss are
float64. Outputs are compared BEFORE feasibility projection or joint clipping.
This explicit parameterization still needs registration before learning.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

BASE_COMMIT = '77624ca66ace1b3aa81d076fefc294fca9015711'
SOURCES = {
    'src/dvidia_training/piper_task.py': 'e66a44a8df4e7a8ee42e4f3866f889f8797cdfb8e2688a146e09b75d2c76664f',
    'src/dvidia_training/camera_action_model.py': 'e9a9c4d1d0f16fc7c91cac3ca2e6560d0532370ab78a2af5a4ec84425c23932f',
    'benchmarks/camera-action-v02-preflight/preflight.py': '39f948ce5100c93fd0f31d3aa3b49f8084fcdafa31ed937db88f2a6b1780fa96',
}
R0 = [0., 1.57, -1.3485, 0., 0., 0., .035]
ORDER = [f'joint{i}' for i in range(1, 7)] + ['single_finger_jaw']
LIMITS = [[-2.618, 2.618], [0., 3.14], [-2.697, 0.], [-1.832, 1.832],
          [-1.22, 1.22], [-3.14, 3.14], [0., .035]]
CONTRACT = {
    'schema': 'dvidia.camera-action-head-preflight/v1',
    'reference_kind': 'previous_applied_servo_reference', 'order': ORDER,
    'units': ['rad'] * 6 + ['single_finger_m'], 'reset_reference': R0,
    'increment_scale': [.03] * 6 + [.001], 'endpoint_loss_scale': [3.] * 6 + [.035],
    'loss': 'mean(((decoded_endpoint-target_endpoint)/endpoint_loss_scale)**2)',
    'network_dtype': 'float32', 'decoder_and_loss_dtype': 'float64',
    'dims': [269, 64, 64, 7], 'parameter_seed': 17, 'case_seed': 171711,
    'reset_cases': 64, 'later_cases': 64, 'projection_applied': False,
    'comparison_absolute_tolerance': 1e-12, 'relative_tolerance': 0.,
    'finite_difference_step': 1e-4, 'finite_difference_absolute_tolerance': 1e-8,
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return sha256(canonical(value)).hexdigest()


def check_contract(contract):
    if canonical(contract) != canonical(CONTRACT):
        raise ValueError('Changed reset, order, units, scales, dtype or experiment contract.')


def vector(value, size, label):
    if type(value) is not list or len(value) != size or any(type(v) not in (int, float) for v in value):
        raise ValueError(label + ' requires ordinary finite numeric coordinates.')
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(label + ' contains a nonfinite coordinate.')
    return result


def reference(value):
    return {'kind': CONTRACT['reference_kind'], 'order': list(ORDER), 'value': list(value)}


def read_reference(value):
    if type(value) is not dict or set(value) != {'kind', 'order', 'value'}:
        raise ValueError('Explicit prior-reference authority is required.')
    if value['kind'] != CONTRACT['reference_kind'] or value['order'] != ORDER:
        raise ValueError('Wrong reference kind or coordinate order.')
    result = vector(value['value'], 7, 'reference')
    limits = np.asarray(LIMITS)
    if np.any(result < limits[:, 0]) or np.any(result > limits[:, 1]):
        raise ValueError('Reference is outside original native bounds.')
    return result


def bind_sources(root):
    bodies = {}
    for relative, expected in SOURCES.items():
        body = (Path(root) / relative).read_bytes()
        if sha256(body).hexdigest() != expected:
            raise ValueError('Source binding changed: ' + relative)
        bodies[relative] = body
    tree = ast.parse(bodies['src/dvidia_training/piper_task.py'])
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PiperTaskEnv')
    reset = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'reset')
    assignments = {ast.unparse(n.targets[0]): n.value for n in ast.walk(reset)
                   if isinstance(n, ast.Assign) and len(n.targets) == 1}
    q = ast.literal_eval(assignments['self.data.qpos[:8]'])
    jaw = ast.literal_eval(assignments['self.command_gripper'])
    if q[:6] + [jaw] != R0 or ast.unparse(assignments['self.command_targets']) != 'self.data.qpos[:6].copy()':
        raise ValueError('Original constant reset-reference assignment changed.')
    return {'base_commit': BASE_COMMIT, 'source_sha256': SOURCES,
            'reset_reference': q[:6] + [jaw],
            'scope': 'Source literals and hashes only; no import or execution of native code.'}


def parameters():
    rng = np.random.default_rng(CONTRACT['parameter_seed'])
    dims = CONTRACT['dims']
    weights = [rng.normal(0., np.sqrt(2. / (a+b)), (a, b)).astype(np.float32)
               for a, b in zip(dims[:-1], dims[1:])]
    return weights, [np.zeros(b, dtype=np.float32) for b in dims[1:]]


def forward(x, params):
    weights, biases = params
    a = np.tanh(x @ weights[0] + biases[0])
    b = np.tanh(a @ weights[1] + biases[1])
    return b @ weights[2] + biases[2]


def decode(head, previous, mode, contract=CONTRACT):
    check_contract(contract)
    r = read_reference(previous)
    g = vector(head, 7, 'head')
    if mode not in ('absolute', 'delta'):
        raise ValueError('Unknown decoder.')
    base = np.asarray(R0, dtype=np.float64) if mode == 'absolute' else r
    return base + np.asarray(CONTRACT['increment_scale']) * g


def endpoint_loss_gradient(decoded, target, contract=CONTRACT):
    """Loss and dL/dg in the same native endpoint units for both decoders."""
    check_contract(contract)
    prediction = vector(decoded, 7, 'decoded endpoint')
    target = vector(target, 7, 'target endpoint')
    scale, increment = np.asarray(CONTRACT['endpoint_loss_scale']), np.asarray(CONTRACT['increment_scale'])
    error = prediction-target
    return float(np.mean((error/scale)**2)), 2.*error*increment/(7.*scale**2)


def cases():
    rng = np.random.default_rng(CONTRACT['case_seed'])
    limits = np.asarray(LIMITS)
    rows = []
    for i in range(64):
        x = np.zeros(269, dtype=np.float32) if i == 0 else rng.uniform(-3., 3., 269).astype(np.float32)
        r_later = rng.uniform(limits[:, 0], limits[:, 1])
        for kind, r in (('reset', np.asarray(R0)), ('later', r_later)):
            target = np.clip(r + rng.uniform(-1., 1., 7)*np.asarray(CONTRACT['increment_scale']), limits[:, 0], limits[:, 1])
            rows.append({'id': f'{kind}-{i:02}', 'kind': kind, 'features': x.tolist(),
                         'previous_reference': reference(r.tolist()), 'target_endpoint': target.tolist()})
    return rows


def measure(row, params):
    g = forward(np.asarray(row['features'], dtype=np.float32), params).astype(np.float64).tolist()
    r = row['previous_reference']
    absolute, delta = decode(g, r, 'absolute'), decode(g, r, 'delta')
    loss_a, grad_a = endpoint_loss_gradient(absolute.tolist(), row['target_endpoint'])
    loss_d, grad_d = endpoint_loss_gradient(delta.tolist(), row['target_endpoint'])
    expected_offset = read_reference(r)-np.asarray(R0)
    offset_error = float(np.max(np.abs((delta-absolute)-expected_offset)))
    # Both decoders have the same d(endpoint)/dg = D even away from reset.
    # A central-difference check measures the actual loss derivative per axis.
    h, fd_error = CONTRACT['finite_difference_step'], 0.
    for mode, gradient in (('absolute', grad_a), ('delta', grad_d)):
        for j in range(7):
            plus, minus = list(g), list(g)
            plus[j] += h; minus[j] -= h
            lp = endpoint_loss_gradient(decode(plus, r, mode).tolist(), row['target_endpoint'])[0]
            lm = endpoint_loss_gradient(decode(minus, r, mode).tolist(), row['target_endpoint'])[0]
            fd_error = max(fd_error, abs((lp-lm)/(2*h)-gradient[j]))
    reset = row['kind'] == 'reset'
    decoded_error = float(np.max(np.abs(delta-absolute)))
    gradient_error = float(np.max(np.abs(grad_d-grad_a)))
    tolerance = CONTRACT['comparison_absolute_tolerance']
    passed = offset_error <= tolerance and fd_error <= CONTRACT['finite_difference_absolute_tolerance']
    if reset:
        passed = passed and decoded_error <= tolerance and abs(loss_d-loss_a) <= tolerance and gradient_error <= tolerance
    limits = np.asarray(LIMITS)
    return {'case_id': row['id'], 'kind': row['kind'], 'passed': bool(passed),
            'preprojection_decoded_max_abs_difference_native': decoded_error,
            'decoded_offset_identity_max_abs_error_native': offset_error,
            'reset_bitwise_decoded_equal': absolute.tobytes() == delta.tobytes() if reset else None,
            'endpoint_loss_absolute': loss_a, 'endpoint_loss_delta': loss_d,
            'd_loss_d_head_max_abs_difference': gradient_error,
            'gradient_finite_difference_max_abs_error': fd_error,
            'absolute_output_exceeds_native_bounds': bool(np.any(absolute < limits[:, 0]) or np.any(absolute > limits[:, 1])),
            'delta_output_exceeds_native_bounds': bool(np.any(delta < limits[:, 0]) or np.any(delta > limits[:, 1]))}


def negative_checks():
    checks = []
    for name, field, value in (
        ('reset', 'reset_reference', [0.]*7), ('order', 'order', list(reversed(ORDER))),
        ('loss_units', 'endpoint_loss_scale', [1.]*7), ('increment_units', 'increment_scale', [1.]*7),
        ('dtype', 'decoder_and_loss_dtype', 'float32'), ('projection', 'projection_applied', True)):
        contract = deepcopy(CONTRACT); contract[field] = value
        try: decode([0.]*7, reference(R0), 'absolute', contract)
        except ValueError: checks.append({'case': name, 'rejected': True})
        else: checks.append({'case': name, 'rejected': False})
    for kind in ('measured_joint_position', 'current_teacher_target', 'future_applied_reference'):
        ref = reference(R0); ref['kind'] = kind
        try: decode([0.]*7, ref, 'delta')
        except ValueError: checks.append({'case': kind, 'rejected': True})
        else: checks.append({'case': kind, 'rejected': False})
    return checks


def run(root, output):
    root, output = Path(root), Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError('Use a fresh output directory; retain prior attempts.')
    source = bind_sources(root)
    params, rows = parameters(), cases()
    output.mkdir(parents=True)
    def write(name, value):
        with (output/name).open('xb') as stream: stream.write(canonical(value)+b'\n')
    # Seal complete synthetic case list, exact parameters and declared gates
    # before evaluating any case. This is random initialization, not fitting.
    write('cases.json', rows)
    write('initialization.json', {'weights': [x.tolist() for x in params[0]], 'biases': [x.tolist() for x in params[1]],
                                 'origin': 'seeded synthetic random initialization; never trained'})
    protocol = {'contract': CONTRACT, 'source_binding': source, 'script_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
                'cases_sha256': sha256((output/'cases.json').read_bytes()).hexdigest(),
                'initialization_sha256': sha256((output/'initialization.json').read_bytes()).hexdigest(),
                'negative_cases': ['reset', 'order', 'loss_units', 'increment_units', 'dtype', 'projection',
                                   'measured_joint_position', 'current_teacher_target', 'future_applied_reference'],
                'status': 'frozen_before_measurement', 'real_observation_vectors': False}
    write('protocol.json', protocol)
    started = time.perf_counter()
    measurements = [measure(row, params) for row in rows]
    negatives = negative_checks()
    write('measurements.json', measurements); write('negative-checks.json', negatives)
    elapsed = time.perf_counter()-started
    resets = [r for r in measurements if r['kind'] == 'reset']
    later = [r for r in measurements if r['kind'] == 'later']
    report = {'schema': CONTRACT['schema']+'/report', 'passed': all(r['passed'] for r in measurements) and all(r['rejected'] for r in negatives),
              'reset_cases': len(resets), 'later_cases': len(later),
              'reset_bitwise_equal': sum(r['reset_bitwise_decoded_equal'] for r in resets),
              'max_offset_identity_error_native': max(r['decoded_offset_identity_max_abs_error_native'] for r in measurements),
              'max_loss_gradient_finite_difference_error': max(r['gradient_finite_difference_max_abs_error'] for r in measurements),
              'later_cases_with_different_decoded_outputs': sum(r['preprojection_decoded_max_abs_difference_native'] > 1e-12 for r in later),
              'reset_cases_with_unprojected_out_of_bounds_output': sum(r['absolute_output_exceeds_native_bounds'] for r in resets),
              'negative_checks_rejected': sum(r['rejected'] for r in negatives),
              'measurement_wall_seconds': elapsed, 'timing_scope': 'Comparisons, central differences, negative checks and measurement-file writes; excludes imports, binding, cases/initialization/protocol and report writes.',
              'hardware': {'platform': platform.platform(), 'machine': platform.machine(), 'python': platform.python_version(), 'numpy': np.__version__},
              'native_runs': 0, 'fitting_runs': 0, 'recordings_read': False, 'trained_models_read': False,
              'qualified': False, 'learning_experiment_registered': False,
              'limitations': ['Synthetic normalized vectors are not registered reset observations.',
                              'Reference authority tags do not authenticate runtime telemetry.',
                              'Only reset decoded functions and common endpoint-loss derivatives are matched; later feedback and optimization outcomes differ.',
                              'No feasibility projection, native microstep simulation, complete learner gradients, Adam updates or controller performance is measured.',
                              'This common-D affine absolute parameterization must be explicitly registered before fitting; it is not automatically the original Cabs recipe.']}
    report['artifacts'] = {name: sha256((output/name).read_bytes()).hexdigest() for name in
                           ('cases.json', 'initialization.json', 'protocol.json', 'measurements.json', 'negative-checks.json')}
    write('report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root, args.output)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result['passed'] else 1)

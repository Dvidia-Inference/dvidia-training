#!/usr/bin/env python3
"""Numerical reference-limiter preflight; no simulator, fitting or recordings.

Execute only the two hash-pinned NumPy reference updates extracted from
PiperTaskEnv.step. MuJoCo, renderers and dvidia_training are never imported.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

BASE_COMMIT = 'ab9082e1ee2b2cc91c7c5fd3f1358727b3789c66'
SOURCE_HASHES = {
    'src/dvidia_training/piper_task.py': 'e66a44a8df4e7a8ee42e4f3866f889f8797cdfb8e2688a146e09b75d2c76664f',
    'src/dvidia_training/piper_imitation.py': '849f66512c58441a240cd20230fcc58691234ddd254c5bbcbfd76abc21b14868',
    'src/dvidia_training/camera_action_model.py': 'e9a9c4d1d0f16fc7c91cac3ca2e6560d0532370ab78a2af5a4ec84425c23932f',
    'src/dvidia_training/camera_action_export.py': '2c4e30587605edc0342150a93430a70ed0559c3ee3eda17d93c5670831370b02',
    'src/dvidia_training/piper_action_capture.py': 'da037b2198cf7d2f3a9aa40fc19a4c0f5cc375a22c5dd75606e2c1856f6efc7f',
    'src/dvidia_training/robot_profiles.lock.json': '4b2ee2ca675aceb2f826983cb3082a3e08e9aa0ff419b23078abcff08ebcc6fe',
}
ORDER = [f'joint{i}' for i in range(1, 7)]
LIMITS = [[-2.618, 2.618], [0., 3.14], [-2.697, 0.],
          [-1.832, 1.832], [-1.22, 1.22], [-3.14, 3.14], [0., .035]]
CONTRACT = {
    'schema': 'dvidia.camera-action-limiter-preflight/v1',
    'joint_order': ORDER,
    'action_units': ['rad'] * 6 + ['single_finger_m'],
    'limits_native': LIMITS,
    'reference_kind': 'previous_applied_servo_reference',
    'timestep_s': .001, 'control_dt_s': .02, 'microsteps': 20,
    'joint_rate_rad_s': 1.5, 'jaw_rate_m_s': .05,
    'absolute_tolerance_native': [1e-12] * 7, 'relative_tolerance': 0.,
}
EXPECTED_UPDATES = '''self.command_targets += np.clip(targets-self.command_targets,
    -self.config.joint_target_rate_rad_s*self.config.timestep,
    self.config.joint_target_rate_rad_s*self.config.timestep)
self.command_gripper += float(np.clip(jaw-self.command_gripper,
    -self.config.jaw_target_rate_m_s*self.config.timestep,
    self.config.jaw_target_rate_m_s*self.config.timestep))
'''
SEED = 171709
UNIFORM_CASES = 2048
NEAR_RATE_CASES = 2048


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return sha256(canonical(value)).hexdigest()


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _literal_assignment(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Missing source constant: ' + name)


def _check_contract(value):
    if type(value) is not dict or canonical(value) != canonical(CONTRACT):
        raise ValueError('The preflight requires unchanged original order, units, limits and rates.')


def _numbers(value, length, label):
    if type(value) not in (list, tuple) or len(value) != length:
        raise ValueError(label + ' requires exactly ' + str(length) + ' numeric values.')
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in value):
        raise ValueError(label + ' requires finite real numbers, not bool/string/coercion.')
    return np.asarray(value, dtype=np.float64)


def _action(value):
    if type(value) is not dict or set(value) != {'joint_targets_rad', 'gripper_position_m'}:
        raise ValueError('Action must contain only six joint targets and a single-finger jaw target.')
    joints = _numbers(value['joint_targets_rad'], 6, 'Joint targets')
    jaw = _numbers([value['gripper_position_m']], 1, 'Jaw target')
    result = np.concatenate((joints, jaw))
    limits = np.asarray(LIMITS)
    if np.any(result < limits[:, 0]) or np.any(result > limits[:, 1]):
        raise ValueError('Reference/action is outside the unchanged original native limits.')
    return result


def action(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return {'joint_targets_rad': vector[:6].tolist(), 'gripper_position_m': float(vector[6])}


def reference(vector):
    return {'kind': CONTRACT['reference_kind'], 'joint_order': list(ORDER), **action(vector)}


def _reference(value):
    fields = {'kind', 'joint_order', 'joint_targets_rad', 'gripper_position_m'}
    if type(value) is not dict or set(value) != fields:
        raise ValueError('Use an explicit preceding applied reference, not measured/future state.')
    if value['kind'] != CONTRACT['reference_kind'] or value['joint_order'] != ORDER:
        raise ValueError('Wrong reference authority or joint order.')
    return _action({k: value[k] for k in ('joint_targets_rad', 'gripper_position_m')})


def load_bindings(repo, piper_xml):
    """Hash-pin inputs, inspect constants/XML, then compile only two numeric lines."""
    repo, piper_xml = Path(repo), Path(piper_xml)
    for name, expected in SOURCE_HASHES.items():
        if file_hash(repo / name) != expected:
            raise ValueError('Frozen source hash differs: ' + name)
    source = (repo / 'src/dvidia_training/piper_task.py').read_text()
    tree = ast.parse(source)
    config = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PiperTaskConfig')
    expected = {'timestep': .001, 'control_dt': .02,
                'joint_target_rate_rad_s': 1.5, 'jaw_target_rate_m_s': .05}
    defaults = {n.target.id: ast.literal_eval(n.value) for n in config.body
                if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
                and n.target.id in expected}
    if any(defaults[k] != v for k, v in expected.items()):
        raise ValueError('Original configuration defaults changed.')
    imitation = ast.parse((repo / 'src/dvidia_training/piper_imitation.py').read_text())
    if ([list(row) for row in _literal_assignment(imitation, 'JOINT_LIMITS')] != LIMITS[:6]
            or _literal_assignment(imitation, 'JAW_MAX_M') != LIMITS[6][1]):
        raise ValueError('Original learning/native action bounds changed.')
    env = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PiperTaskEnv')
    step = next(n for n in env.body if isinstance(n, ast.FunctionDef) and n.name == 'step')
    loops = [n for n in ast.walk(step) if isinstance(n, ast.For) and n.body
             and isinstance(n.body[0], ast.AugAssign)
             and isinstance(n.body[0].target, ast.Attribute)
             and n.body[0].target.attr == 'command_targets']
    if len(loops) != 1:
        raise ValueError('Expected one original native stepping loop.')
    updates = loops[0].body[:2]
    expected_tree = ast.parse(EXPECTED_UPDATES)
    module = ast.Module(body=updates, type_ignores=[])
    if ast.dump(module, include_attributes=False) != ast.dump(expected_tree, include_attributes=False):
        raise ValueError('Original numeric limiter update expressions changed.')
    # We do not compile or execute the native step loop, _controls, mj_step or scoring.
    compiled = compile(ast.fix_missing_locations(module), '<hash-pinned-numeric-limiter-only>', 'exec')
    lock = json.loads((repo / 'src/dvidia_training/robot_profiles.lock.json').read_text())
    profile = next(p for p in lock['profiles'] if p['id'] == 'piper')
    pinned = next(f for f in profile['files'] if f['path'] == 'agilex_piper/piper.xml')
    if file_hash(piper_xml) != pinned['sha256'] or piper_xml.stat().st_size != pinned['bytes']:
        raise ValueError('PiPER XML differs from the existing pinned asset lock.')
    xml = ET.fromstring(piper_xml.read_bytes())
    if profile['actuator_names'] != ORDER + ['gripper'] or profile['arm_joint_names'] != ORDER:
        raise ValueError('Pinned profile joint/actuator order changed.')
    actuators = list(xml.find('actuator'))
    if [a.get('name') for a in actuators] != ORDER + ['gripper']:
        raise ValueError('Pinned XML actuator order changed.')
    joints = {j.get('name'): j for j in xml.findall('.//worldbody//joint')}
    xml_limits = [[float(x) for x in joints[n].get('range').split()] for n in ORDER + ['joint7']]
    if xml_limits != LIMITS:
        raise ValueError('Pinned native joint ranges differ from model action limits.')
    # The locked XML gives piper position servos inheritrange=1; finger inherits it.
    if xml.find("./default/default[@class='piper']/position").get('inheritrange') != '1':
        raise ValueError('Pinned servo joint-range inheritance changed.')
    bindings = {'base_source_commit': BASE_COMMIT, 'source_sha256': SOURCE_HASHES,
                'upstream_commit': lock['commit'], 'piper_xml': pinned,
                'limiter_ast_sha256': sha256(ast.dump(module, include_attributes=False).encode()).hexdigest(),
                'limiter_source_lines': [updates[0].lineno, updates[-1].end_lineno],
                'original_defaults': expected,
                'scope': 'Only the two reference updates are executed; XML is parsed without MuJoCo.'}
    return compiled, bindings


class Limiter:
    def __init__(self, compiled):
        self.compiled = compiled

    def trajectory(self, previous_reference, target, contract=CONTRACT):
        _check_contract(contract)
        r, a = _reference(previous_reference), _action(target)
        state = SimpleNamespace(command_targets=r[:6].copy(), command_gripper=float(r[6]),
                                config=SimpleNamespace(timestep=.001, control_dt=.02,
                                    joint_target_rate_rad_s=1.5, jaw_target_rate_m_s=.05))
        scope = {'self': state, 'targets': a[:6], 'jaw': float(a[6]), 'np': np, 'float': float}
        result = []
        for _ in range(20):
            exec(self.compiled, {'__builtins__': {}}, scope)
            result.append(np.concatenate((state.command_targets.copy(), [state.command_gripper])))
        return np.asarray(result, dtype=np.float64)

    def endpoint(self, previous_reference, target, contract=CONTRACT):
        """Closed-form P; no rollout, measured joints, object truth or teacher lookup."""
        _check_contract(contract)
        r, a = _reference(previous_reference), _action(target)
        return r + np.clip(a-r, -interval_allowance(), interval_allowance())


def interval_allowance():
    return np.asarray([1.5] * 6 + [.05], dtype=np.float64) * .02


def cases():
    """Fixed synthetic numerical training-development vectors, not scene episodes."""
    limits = np.asarray(LIMITS)
    low, high = limits[:, 0], limits[:, 1]
    middle = (low+high)/2.
    rows = []

    def add(name, r, a, group):
        rows.append({'case_id': name, 'group': group, 'usage': 'numerical_training_development_only',
                     'previous_reference': reference(r), 'teacher_target': action(a)})

    for name, r, a in [('hold_low', low, low), ('hold_high', high, high),
                       ('hold_middle', middle, middle), ('all_increase', low, high),
                       ('all_decrease', high, low),
                       ('mixed_directions', np.where(np.arange(7) % 2, low, high), middle)]:
        add(name, r, a, 'explicit_bounds')
    tick = interval_allowance()/20.
    for axis in range(7):
        for sign in (-1., 1.):
            for magnitude in (0., .5, 1., 2., 19., 20., 20.5, 21., 40.):
                a = middle.copy(); a[axis] += sign*tick[axis]*magnitude
                add(f'axis{axis}_sign{int(sign)}_ticks{magnitude:g}', middle, a, 'tick_boundaries')
        for bound, endpoint in [('low', low[axis]), ('high', high[axis])]:
            r, a = middle.copy(), middle.copy()
            r[axis] = np.nextafter(endpoint, middle[axis]); a[axis] = endpoint
            add(f'axis{axis}_one_ulp_from_{bound}', r, a, 'ulp_boundaries')
    rng = np.random.default_rng(SEED)
    for i in range(UNIFORM_CASES):
        add(f'uniform_{i:04}', rng.uniform(low, high), rng.uniform(low, high), 'uniform_bounded')
    for i in range(NEAR_RATE_CASES):
        r = rng.uniform(low, high)
        a = np.clip(r+rng.uniform(-1.25, 1.25, 7)*interval_allowance(), low, high)
        add(f'near_rate_{i:04}', r, a, 'near_interval_allowance')
    return rows


def negative_cases(limiter):
    midpoint = np.asarray(LIMITS).mean(axis=1)
    ref, target = reference(midpoint), action(midpoint)
    variants = []
    def add(name, ref_value=ref, target_value=target, contract=CONTRACT):
        variants.append((name, deepcopy(ref_value), deepcopy(target_value), deepcopy(contract)))
    for field, value in [('joint_rate_rad_s', 1.6), ('jaw_rate_m_s', .06),
                         ('timestep_s', .002), ('microsteps', 10), ('control_dt_s', .04)]:
        c = deepcopy(CONTRACT); c[field] = value; add('changed_'+field, contract=c)
    c = deepcopy(CONTRACT); c['limits_native'][0][0] -= .01; add('changed_joint_limit', contract=c)
    c = deepcopy(CONTRACT); c['limits_native'][6][1] += .01; add('changed_jaw_limit', contract=c)
    c = deepcopy(CONTRACT); c['joint_order'][0:2] = c['joint_order'][1::-1]; add('changed_contract_order', contract=c)
    r = deepcopy(ref); r['joint_order'][0:2] = r['joint_order'][1::-1]; add('changed_reference_order', ref_value=r)
    for kind in ('measured_joint_position', 'future_applied_reference', 'teacher_query_target'):
        r = deepcopy(ref); r['kind'] = kind; add('wrong_state_'+kind, ref_value=r)
    r = deepcopy(ref); r['measured_joint_position_rad'] = midpoint[:6].tolist(); add('extra_measured_state', ref_value=r)
    r = deepcopy(ref); del r['kind']; add('missing_reference_authority', ref_value=r)
    for name, value in [('nan', float('nan')), ('infinity', float('inf')), ('bool', True), ('string', '0')]:
        a = deepcopy(target); a['joint_targets_rad'][0] = value; add('invalid_target_'+name, target_value=a)
    a = deepcopy(target); a['joint_targets_rad'] = a['joint_targets_rad'][:-1]; add('target_length', target_value=a)
    a = deepcopy(target); a['joint_targets_rad'][0] = LIMITS[0][1]+1e-6; add('target_joint_bound', target_value=a)
    for jaw in (-1e-6, .035001):
        a = deepcopy(target); a['gripper_position_m'] = jaw; add('target_jaw_bound_'+str(jaw), target_value=a)
    r = deepcopy(ref); r['joint_targets_rad'][0] = LIMITS[0][0]-1e-6; add('reference_joint_bound', ref_value=r)
    r = deepcopy(ref); r['gripper_position_m'] = -.001; add('reference_jaw_bound', ref_value=r)
    results = []
    for name, r, a, c in variants:
        rejected = []
        for operation in (limiter.trajectory, limiter.endpoint):
            try:
                operation(r, a, c)
                rejected.append(False)
            except (ValueError, TypeError, OverflowError):
                rejected.append(True)
        results.append({'case_id': name, 'original_and_projection_rejected': all(rejected)})
    return results


def compare_case(limiter, row):
    r, a = row['previous_reference'], row['teacher_target']
    original = limiter.trajectory(r, a)
    endpoint = limiter.endpoint(r, a)
    projected = limiter.trajectory(r, action(endpoint))
    exact_endpoint = limiter.trajectory(r, action(original[-1]))
    error = np.abs(projected-original)
    exact_error = np.abs(exact_endpoint-original)
    limits = np.asarray(LIMITS)
    allowance = interval_allowance()
    delta = (endpoint-_reference(r))/allowance
    # Cdelta decode is tested here; no learned Cdelta or Cabs model exists yet.
    decoded = _reference(r)+delta*allowance
    decoded_trajectory = limiter.trajectory(r, action(decoded))
    decoded_error = np.abs(decoded_trajectory-original)
    within = bool(np.all(projected >= limits[:, 0]) and np.all(projected <= limits[:, 1]))
    tolerance = np.asarray(CONTRACT['absolute_tolerance_native'])
    return {'case_id': row['case_id'], 'group': row['group'],
            'max_abs_error_by_axis_native': error.max(axis=0).tolist(),
            'max_exact_endpoint_error_by_axis_native': exact_error.max(axis=0).tolist(),
            'max_delta_decode_error_by_axis_native': decoded_error.max(axis=0).tolist(),
            'bitwise_equal_analytic_endpoint': original.tobytes() == projected.tobytes(),
            'bitwise_equal_replayed_endpoint': original.tobytes() == exact_endpoint.tobytes(),
            'bitwise_equal_delta_decode': original.tobytes() == decoded_trajectory.tobytes(),
            'within_native_limits': within,
            'delta_within_unit_interval': bool(np.all(np.abs(delta) <= 1.+1e-12)),
            'passed': bool(np.all(error <= tolerance) and np.all(exact_error <= tolerance)
                           and np.all(decoded_error <= tolerance) and within
                           and np.all(np.abs(delta) <= 1.+1e-12))}


def _write_new(path, value):
    with Path(path).open('xb') as stream:
        stream.write(canonical(value)+b'\n')


def run(repo, piper_xml, output):
    compiled, bindings = load_bindings(repo, piper_xml)
    output = Path(output)
    if output.exists():
        raise ValueError('Evidence output must be new; previous outcomes are never overwritten.')
    script = Path(__file__)
    test_path = script.with_name('test_preflight.py')
    rows = cases()
    protocol = {'schema': 'dvidia.camera-action-limiter-preflight-protocol/v1',
                'contract': CONTRACT, 'bindings': bindings, 'seed': SEED,
                'uniform_cases': UNIFORM_CASES, 'near_rate_cases': NEAR_RATE_CASES,
                'case_count': len(rows), 'cases_sha256': digest(rows),
                'generator': 'numpy.default_rng PCG64, float64; all vectors generated before comparisons',
                'script_sha256': file_hash(script), 'tests_sha256': file_hash(test_path),
                'episode_data_read': False, 'training_or_confirmation_rollouts': 0,
                'training_fits': 0, 'simulator_import_or_execution': False,
                'gate': 'Every 20x7 reference entry agrees within 1e-12 absolute native units; '
                        'zero relative tolerance; all invalid contracts/states rejected; no skill qualification.'}
    output.mkdir(parents=True)
    _write_new(output/'cases.json', rows)
    _write_new(output/'protocol.json', protocol)
    began = time.perf_counter()
    limiter = Limiter(compiled)
    results = [compare_case(limiter, row) for row in rows]
    negatives = negative_cases(limiter)
    max_error = np.max([r['max_abs_error_by_axis_native'] for r in results], axis=0)
    delta_error = np.max([r['max_delta_decode_error_by_axis_native'] for r in results], axis=0)
    exact_error = np.max([r['max_exact_endpoint_error_by_axis_native'] for r in results], axis=0)
    allowed = all(r['passed'] for r in results) and all(r['original_and_projection_rejected'] for r in negatives)
    _write_new(output/'measurements.json', {'cases': results, 'negative_cases': negatives})
    report = {'schema': 'dvidia.camera-action-limiter-preflight-report/v1',
              'status': 'passed' if allowed else 'blocked',
              'reference_equivalence_gate_passed': allowed,
              'contrast_may_proceed_to_separately_registered_training': allowed,
              'bitwise_equivalence_claim': False,
              'protocol_sha256': file_hash(output/'protocol.json'),
              'cases_file_sha256': file_hash(output/'cases.json'),
              'measurements_sha256': file_hash(output/'measurements.json'),
              'case_count': len(rows), 'microsteps_per_case': 20, 'axes_per_microstep': 7,
              'matched_reference_entries_per_comparison': len(rows)*20*7,
              'numerically_equivalent_cases': sum(r['passed'] for r in results),
              'analytic_endpoint_bitwise_equal_cases': sum(r['bitwise_equal_analytic_endpoint'] for r in results),
              'replayed_endpoint_bitwise_equal_cases': sum(r['bitwise_equal_replayed_endpoint'] for r in results),
              'delta_decode_bitwise_equal_cases': sum(r['bitwise_equal_delta_decode'] for r in results),
              'max_abs_error_by_axis_native': max_error.tolist(),
              'max_delta_decode_error_by_axis_native': delta_error.tolist(),
              'max_replayed_endpoint_error_by_axis_native': exact_error.tolist(),
              'negative_cases': len(negatives),
              'negative_cases_rejected': sum(r['original_and_projection_rejected'] for r in negatives),
              'wall_seconds': time.perf_counter()-began,
              'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                          'platform': platform.platform(), 'implementation': platform.python_implementation()},
              'physical_robot_ready': False, 'skill_improvement_measured': False,
              'scope': 'Only constant-target position reference slew over 20 native intervals. '
                       'Not joint motion, actuator force, contacts, task success or model quality.',
              'limitations': ['Synthetic numeric vectors are not robot episodes or a distributional success test.',
                  'Float64 arithmetic is compared at a predeclared 1e-12 tolerance, not promised bitwise identity.',
                  'Typed reference authority rejects mislabeled fields; this standalone check cannot prove source telemetry provenance.',
                  'No policy fitting, model-head initialization equivalence or closed-loop evaluation was performed.',
                  'No altered limiter, variable-rate, pulse-actuation or measured-q-relative controller is covered.']}
    _write_new(output/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--piper-xml', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run(args.repo, args.piper_xml, args.output)
    except (ValueError, OSError, StopIteration, KeyError) as exc:
        print(json.dumps({'status': 'blocked', 'error': str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({k: report[k] for k in ('status', 'case_count', 'numerically_equivalent_cases',
                                          'negative_cases_rejected', 'wall_seconds')}, indent=2))
    return 0 if report['reference_equivalence_gate_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

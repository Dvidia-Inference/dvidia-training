"""Paired native timestep diagnostics for developmental reaction coupons.

Only native integration cadence changes. Sensor acquisition, reaction updates,
command delay, scripted trajectories and horizons remain fixed. Agreement is a
numerical diagnostic, not contact calibration or hardware qualification.
"""
from copy import deepcopy
import hashlib
import json
import math


TIMESTEPS_S = (.001, .0005)
METRICS = ('detected_s', 'stopped_s', 'stopping_distance_m',
           'peak_unexpected_force_n', 'unexpected_contact_impulse_ns',
           'max_penetration_m', 'minimum_clearance_m')


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _change(base, fine):
    row = {'base': base, 'fine': fine, 'absolute_change': None,
           'relative_change_to_abs_base': None, 'relative_status': None}
    if base is None or fine is None:
        row['relative_status'] = 'both_missing' if base is fine else 'missing_base' if base is None else 'missing_fine'
    elif not _finite(base) or not _finite(fine):
        row['relative_status'] = 'invalid_value'
    else:
        row['absolute_change'] = abs(fine-base)
        if base == 0:
            row['relative_status'] = 'both_zero' if fine == 0 else 'zero_to_nonzero'
        else:
            row['relative_change_to_abs_base'] = abs(fine-base)/abs(base)
            row['relative_status'] = 'comparable'
    return row


def _absolute_pass(row, tolerance):
    if row['absolute_change'] is None:
        return False if row['relative_status'] in ('missing_base', 'missing_fine') else None
    return row['absolute_change'] <= tolerance + 1e-12


def _relative_pass(row, tolerance):
    if row['relative_status'] == 'both_zero':
        return True
    if row['relative_status'] == 'zero_to_nonzero':
        return False
    if row['relative_change_to_abs_base'] is None:
        return False if row['relative_status'] in ('missing_base', 'missing_fine') else None
    return row['relative_change_to_abs_base'] <= tolerance + 1e-12


def _all(values):
    values = tuple(values)
    return False if False in values else None if None in values else True


def _validate(protocol):
    if not isinstance(protocol, dict):
        raise ValueError('protocol must be an object')
    diagnostics = protocol.get('convergence_diagnostics')
    names = ('event_time_absolute_s', 'contact_relative_change', 'penetration_absolute_m')
    if not isinstance(diagnostics, dict) or any(not _finite(diagnostics.get(name)) or diagnostics[name] < 0 for name in names):
        raise ValueError('freeze finite nonnegative convergence_diagnostics before trials')
    fixture = protocol.get('fixture_config', {})
    reaction = protocol.get('reaction_config', {})
    if not isinstance(fixture, dict) or not isinstance(reaction, dict):
        raise ValueError('fixture_config and reaction_config must be objects')
    required = ('warmup_s', 'horizon_s', 'rest_dwell_s', 'command_delay_s')
    clock = ('sensor_period_s', 'sensor_prime_s', 'reaction_period_s')
    durations = [fixture.get(name) for name in required] + [protocol.get(name) for name in clock]
    durations.append(reaction.get('command_delay_s'))
    for duration in durations:
        if not _finite(duration) or duration < 0:
            raise ValueError('protocol durations must be finite and nonnegative')
        for timestep in TIMESTEPS_S:
            if not math.isclose(duration/timestep, round(duration/timestep), rel_tol=0., abs_tol=1e-9):
                raise ValueError('all clocks/durations must contain whole ticks at both timesteps')
    if protocol['sensor_period_s'] <= 0 or protocol['reaction_period_s'] <= 0:
        raise ValueError('sensor and reaction periods must be positive')
    if type(protocol.get('seed')) is not int or protocol['seed'] < 0:
        raise ValueError('protocol seed must be a nonnegative integer')
    cases = protocol.get('cases', [])
    if not isinstance(cases, list) or any(not isinstance(case, dict) for case in cases):
        raise ValueError('cases must be a list of objects')
    selected = []
    for case_id in ('crossing', 'late_occluded_entry'):
        matches = [(index, case) for index, case in enumerate(cases) if case.get('id') == case_id]
        if len(matches) != 1:
            raise ValueError(f'protocol requires exactly one {case_id} case')
        for name in ('onset_s', 'sensor_delay_s', 'occluded_until_s'):
            duration = matches[0][1].get(name, 0.)
            if not _finite(duration) or duration < 0 or any(not math.isclose(duration/dt, round(duration/dt), rel_tol=0., abs_tol=1e-9) for dt in TIMESTEPS_S):
                raise ValueError('selected case timings must contain whole ticks at both timesteps')
        selected.append(matches[0])
    _hash(protocol)  # Reject non-JSON/non-finite configuration before any trial.
    return diagnostics, selected


def _inputs(source, cases):
    for case_index, case in cases:
        seed = source['seed'] + case_index*10
        for variant in ('monitor', 'disabled_control'):
            for timestep in TIMESTEPS_S:
                config = deepcopy(source)
                config['fixture_config']['timestep_s'] = timestep
                identity = {'case_id': case['id'], 'variant': variant, 'seed': seed,
                    'timestep_s': timestep, 'protocol_sha256': _hash(config),
                    'fixture_config_sha256': _hash(config['fixture_config']),
                    'input_sha256': _hash({'protocol': config, 'case': case, 'variant': variant, 'seed': seed})}
                yield identity, config, deepcopy(case)


def _checked_attempts(source, cases, attempts):
    if not isinstance(attempts, list) or len(attempts) != 8:
        raise ValueError('exactly eight convergence attempts required')
    expected = [identity for identity, _, _ in _inputs(source, cases)]
    by_identity = {(row['case_id'], row['variant'], row['timestep_s']): row for row in expected}
    found = {}
    for attempt in attempts:
        if not isinstance(attempt, dict) or set(attempt) != set(expected[0]) | {'record'}:
            raise ValueError('convergence attempt must contain exactly identity fields and record')
        if not isinstance(attempt['case_id'], str) or not isinstance(attempt['variant'], str) or not _finite(attempt['timestep_s']) or type(attempt['seed']) is not int:
            raise ValueError('invalid convergence attempt identity types')
        key = (attempt['case_id'], attempt['variant'], attempt['timestep_s'])
        if key not in by_identity or key in found:
            raise ValueError('unexpected or duplicate convergence attempt identity')
        identity = {name: attempt[name] for name in expected[0]}
        if identity != by_identity[key] or _hash(identity) != _hash(by_identity[key]):
            raise ValueError('convergence attempt configuration hashes/seed do not match protocol')
        record = attempt['record']
        if (not isinstance(record, dict) or not {'case_id', 'variant', 'seed', 'valid', 'error'} <= set(record)
                or type(record.get('valid')) is not bool
                or record.get('error') is not None and not isinstance(record['error'], str)):
            raise ValueError('convergence record requires boolean valid and null/string error')
        record_identity = {name: record.get(name) for name in ('case_id', 'variant', 'seed')}
        if _hash(record_identity) != _hash({name: identity[name] for name in record_identity}):
            raise ValueError('nested convergence record identity does not match attempt')
        if record['valid'] and any(name not in record for name in METRICS):
            raise ValueError('valid convergence records require all measurement fields')
        for name in METRICS:
            value = record.get(name)
            if value is not None and (not _finite(value) or name != 'minimum_clearance_m' and value < 0):
                raise ValueError(f'{name} must be finite measurement or null')
        _hash(attempt)  # All complete records must remain strict JSON evidence.
        found[key] = deepcopy(attempt)
    return [found[(row['case_id'], row['variant'], row['timestep_s'])] for row in expected]


def recompute_comparison(protocol, attempts) -> dict:
    """Validate all eight identities and derive canonical diagnostics from records.

    No native packages are imported. Stored verdicts are never consumed. Missing
    rest in disabled controls is inapplicable; other null measurements stay null.
    Invalid trials remain in the denominator and have unknown diagnostic status.
    """
    source = deepcopy(protocol)
    diagnostics, cases = _validate(source)
    attempts = _checked_attempts(source, cases, attempts)
    pairs = []
    for offset in range(0, len(attempts), 2):
        pair_attempts = attempts[offset:offset+2]
        identity = pair_attempts[0]
        variant = identity['variant']
        base, fine = (attempt['record'] for attempt in pair_attempts)
        valid = [record.get('valid') is True and record.get('error') is None for record in (base, fine)]
        both_valid = all(valid)
        changes = {name: _change(base.get(name) if valid[0] else None,
                                fine.get(name) if valid[1] else None) for name in METRICS}
        timing_checks = {'detected_s': _absolute_pass(changes['detected_s'], diagnostics['event_time_absolute_s'])}
        if variant == 'monitor':
            timing_checks['stopped_s'] = _absolute_pass(changes['stopped_s'], diagnostics['event_time_absolute_s'])
        contact_checks = {
            'peak_unexpected_force_n': _relative_pass(changes['peak_unexpected_force_n'], diagnostics['contact_relative_change']),
            'unexpected_contact_impulse_ns': _relative_pass(changes['unexpected_contact_impulse_ns'], diagnostics['contact_relative_change']),
            'max_penetration_m': _absolute_pass(changes['max_penetration_m'], diagnostics['penetration_absolute_m'])}
        pairs.append({'case_id': identity['case_id'], 'variant': variant, 'seed': identity['seed'],
            'both_valid': both_valid, 'base_timestep_s': TIMESTEPS_S[0], 'fine_timestep_s': TIMESTEPS_S[1],
            'metrics': changes, 'timing_checks': timing_checks, 'contact_checks': contact_checks,
            'rest_applicability': 'required monitored stop' if variant == 'monitor' else 'no brake intervention; rest comparison inapplicable',
            'timing_stable': _all(timing_checks.values()) if both_valid else None,
            'contact_converged': _all(contact_checks.values()) if both_valid else None,
            'errors': [{'timestep_s': attempt['timestep_s'], 'error': attempt['record'].get('error')}
                       for attempt in pair_attempts if attempt['record'].get('valid') is not True or attempt['record'].get('error') is not None]})
    invalid = sum(attempt['record'].get('valid') is not True or attempt['record'].get('error') is not None for attempt in attempts)
    return {'format': 'dvidia.reaction-timestep-comparison', 'schema_version': 1,
        'framework_version': '0.1', 'scope': 'two paired synthetic native coupons; authored numerical diagnostics',
        'benchmark_qualification': False, 'physical_robot_ready': False,
        'protocol_sha256': _hash(source), 'timesteps_s': list(TIMESTEPS_S),
        'fixed_clocks': {name: source[name] for name in ('sensor_period_s', 'reaction_period_s', 'sensor_prime_s')},
        'diagnostics': deepcopy(diagnostics), 'attempted': len(attempts), 'invalid': invalid,
        'attempts': attempts, 'pairs': pairs,
        'timing_stable': _all(pair['timing_stable'] for pair in pairs),
        'contact_converged': _all(pair['contact_converged'] for pair in pairs),
        'relative_change_scope': 'absolute fine-minus-base change divided by absolute base value; zero baseline remains null with explicit status',
        'limitations': ['One paired seed per case; neither independent confirmation nor material calibration.',
                        'Force is an external-force proxy, not calibrated tactile pressure.',
                        'Agreement at two native timesteps cannot establish continuous-time or hardware correctness.']}


def compare_timesteps(protocol) -> dict:
    """Run eight native attempts, then use the pure evidence recomputation path."""
    source = deepcopy(protocol)
    _, cases = _validate(source)
    from .reaction_benchmark import run_trial  # Optional native engine remains lazy.
    attempts = []
    for identity, config, case in _inputs(source, cases):
        try:
            record = run_trial(config, case, identity['variant'], identity['seed'])
            if not isinstance(record, dict):
                raise ValueError('run_trial returned a non-object record')
        except Exception as exc:
            record = {name: identity[name] for name in ('case_id', 'variant', 'seed')}
            record.update(valid=False, error=f'{type(exc).__name__}: {exc}')
        attempts.append({**identity, 'record': deepcopy(record)})
    return recompute_comparison(source, attempts)

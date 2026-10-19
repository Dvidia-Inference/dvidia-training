"""Reproducible local-goal benchmark with separate physical/evidence labels.

No provider calls. Generation requires native MuJoCo; evaluation is stdlib-only.
Input artifacts are sealed before evaluation, and every scheduled case counts.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

from .local_goal_verifier import GoalContract, contract_description, verify_goal

COHORTS = ('native', 'evidence_fault', 'blind_spot')
VERDICTS = ('complete', 'incomplete', 'unknown')
FAULTS = ('stale', 'future', 'gap', 'duplicate', 'wrong_units', 'wrong_frame',
          'wrong_identity', 'short_history', 'missing_measurement', 'unknown_presence',
          'threshold_uncertainty', 'single_tick_contact_spike', 'missing_uncertainty',
          'invalid_number', 'inconsistent_presence', 'injected_label')
TIMING_SCOPE = ('perf_counter_ns around verify_goal only, with parsed observations and trusted goal already in memory. '
                'Excludes acquisition, the required 200 ms history accumulation, serialization, I/O, rendering, networking and actuation. '
                'Repeated-call microbenchmark, not end-to-end robot latency.')


def read_json(path):
    return json.loads(Path(path).read_bytes(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative_file(root, name):
    if not isinstance(name, str) or '\\' in name or not name or name.startswith('/') or any(p in ('', '.', '..') for p in name.split('/')):
        raise ValueError('Expected a safe relative artifact path')
    root = Path(root).absolute()
    path = root / name
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError('Expected a plain existing artifact')
    return path


def quantiles(values):
    values = sorted(values)
    if not values:
        return dict(samples=0, p50=None, p95=None, p99=None, max=None)
    def percentile(f):
        i = (len(values)-1)*f
        lo, hi = math.floor(i), math.ceil(i)
        return values[lo] + (values[hi]-values[lo])*(i-lo)
    return dict(samples=len(values), p50=percentile(.5), p95=percentile(.95), p99=percentile(.99), max=values[-1])


def fraction(numerator, denominator):
    return dict(numerator=numerator, denominator=denominator, rate=numerator/denominator if denominator else None)


def score(rows, cases):
    catalog = {case['id']: case for case in cases}
    results = {row['case_id']: row for row in rows}
    if len(catalog) != len(cases) or len(results) != len(rows) or set(catalog) != set(results):
        raise ValueError('Exactly one result is required for every scheduled case')
    for identity, case in catalog.items():
        row = results[identity]
        if case['cohort'] not in COHORTS or case['expected_verdict'] not in VERDICTS or row['decision']['verdict'] not in VERDICTS:
            raise ValueError('Invalid cohort or verdict')
        truth = case['physical_truth']
        if any(type(truth[k]) is not bool for k in ('present', 'complete')) or (truth['complete'] and not truth['present']) or type(case['generation_valid']) is not bool:
            raise ValueError('Invalid physical truth')
        if not isinstance(row['timing_ns'], list) or not row['timing_ns'] or any(type(t) is not int or t < 0 or t > 10**12 for t in row['timing_ns']):
            raise ValueError('Every case needs bounded measured timings')
    def group(selected):
        decisions = [results[c['id']]['decision']['verdict'] for c in selected]
        positives = [c for c in selected if c['physical_truth']['complete']]
        negatives = [c for c in selected if not c['physical_truth']['complete']]
        return {
            'planned': len(selected),
            'verdict_counts': {v: decisions.count(v) for v in VERDICTS},
            'expected_verdict_matches': fraction(sum(results[c['id']]['decision']['verdict'] == c['expected_verdict'] for c in selected), len(selected)),
            'physical_true_completion': fraction(sum(results[c['id']]['decision']['verdict'] == 'complete' for c in positives), len(positives)),
            'physical_false_completion': fraction(sum(results[c['id']]['decision']['verdict'] == 'complete' for c in negatives), len(negatives)),
            'abstention': fraction(decisions.count('unknown'), len(selected)),
            'generation_invalid': sum(not c['generation_valid'] for c in selected),
            'latency_us': quantiles([t/1000 for c in selected for t in results[c['id']]['timing_ns']]),
        }
    return {'overall': group(cases),
            'by_cohort': {cohort: group([c for c in cases if c['cohort'] == cohort]) for cohort in COHORTS},
            'by_scenario': {name: group([c for c in cases if c['scenario'] == name]) for name in sorted({c['scenario'] for c in cases})}}


def fault_packet(packet, fault, now_s):
    """Predeclared evidence perturbations; physical state is never relabelled."""
    packet = deepcopy(packet)
    samples = packet['samples']
    expected = 'unknown'
    if fault == 'stale':
        now_s += .5
    elif fault == 'future':
        packet['history_start_s'] += .2
        for sample in samples:
            sample['t_s'] += .2
    elif fault == 'gap':
        del samples[50]
    elif fault == 'duplicate':
        samples[50]['t_s'] = samples[49]['t_s']
    elif fault == 'wrong_units':
        packet['units']['position'] = 'cm'
    elif fault == 'wrong_frame':
        samples[-1]['frame_id'] = 'camera'
    elif fault == 'wrong_identity':
        samples[-1]['object_id'] = 'different-shoe'
    elif fault == 'short_history':
        packet['samples'] = samples[-20:]
        packet['history_start_s'] = round(samples[-20]['t_s']-.002, 9)
    elif fault == 'missing_measurement':
        samples[-1]['robot_contact_force_n'] = None
    elif fault == 'unknown_presence':
        for sample in samples:
            sample['presence'] = 'unknown'
            for key in ('position_m', 'yaw_rad', 'linear_speed_m_s', 'angular_speed_rad_s', 'table_support_force_n', 'robot_contact_force_n'):
                sample[key] = None
    elif fault == 'threshold_uncertainty':
        packet['uncertainty_bounds']['table_support_force_n'] = 1.
    elif fault == 'single_tick_contact_spike':
        samples[50]['robot_contact_force_n'] = .10
        expected = 'incomplete'
    elif fault == 'missing_uncertainty':
        del packet['uncertainty_bounds']
    elif fault == 'invalid_number':
        samples[-1]['linear_speed_m_s'] = 'NaN'
    elif fault == 'inconsistent_presence':
        samples[-1]['presence'] = 'absent'
    elif fault == 'injected_label':
        packet['complete'] = True
    else:
        raise ValueError('Unknown predeclared fault')
    return packet, now_s, expected


def add_controls(root, native):
    root = Path(root)
    cases = list(native)
    for complete in [case for case in native if case['scenario'] == 'complete']:
        packet = read_json(relative_file(root, complete['packet_file']))
        for fault in FAULTS:
            altered, now_s, expected = fault_packet(packet, fault, complete['now_s'])
            case = deepcopy(complete)
            case.update(id=complete['layout_id']+'-fault-'+fault, cohort='evidence_fault', scenario=fault,
                        expected_verdict=expected, now_s=now_s,
                        note='Evidence perturbation only; underlying native physical state is unchanged. '+
                        ('A fabricated force spike is a definite failure in the supplied evidence.' if fault == 'single_tick_contact_spike' else 'A correctly conservative response is unknown.'))
            case['packet_file'] = 'packets/'+case['id']+'.json'
            write_json(root/case['packet_file'], altered)
            cases.append(case)
        moving = next(case for case in native if case['layout_id'] == complete['layout_id'] and case['scenario'] == 'moving_alias')
        forged = deepcopy(moving)
        forged.update(id=complete['layout_id']+'-consistent-forgery', cohort='blind_spot', scenario='consistent_forgery',
                      expected_verdict='complete', contract=deepcopy(complete['contract']), now_s=complete['now_s'],
                      note='Deliberately self-consistent false sensor stream: complete-state observations paired with a physically moving native state. Expected packet verdict is complete, but that is a physical false completion. Requires independent trusted sensing to detect.')
        forged['packet_file'] = 'packets/'+forged['id']+'.json'
        write_json(root/forged['packet_file'], packet)
        cases.append(forged)
    return cases


def source_identity():
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    paths = ['src/dvidia_training/'+name+'.py' for name in
             ('local_goal_verifier', 'local_goal_benchmark', 'local_goal_scenes', 'image_goal_scenes', 'image_goal_benchmark', 'jev_client', 'workstation', 'robot_profiles', 'reference_sim', 'asset_library')]
    paths += ['src/dvidia_training/robot_profiles.lock.json', 'src/dvidia_training/asset_catalog.json']
    files = {}
    for name in paths:
        committed = subprocess.check_output(['git', 'show', commit+':'+name], cwd=repo)
        current = sha(repo/name)
        if hashlib.sha256(committed).hexdigest() != current:
            raise ValueError('Commit benchmark sources before freezing a protocol: '+name)
        files[name] = current
    return {'commit': commit, 'files': files}


def generate(assets, output, *, seed=20261019, count=8, repeats=100, warmups=10):
    from .local_goal_scenes import make_layouts, generate_native
    output = Path(output)
    if output.exists():
        raise ValueError('Output must be a fresh directory; existing runs are immutable')
    if not 1 <= count <= 32 or not 1 <= repeats <= 1000 or not 0 <= warmups <= 100:
        raise ValueError('Invalid bounded run dimensions')
    layouts = make_layouts(seed, count)
    protocol = {'schema': 'dvidia.local-goal-protocol/v1', 'source': source_identity(),
                'seed': seed, 'layouts': layouts, 'layout_count': count, 'repeats': repeats, 'warmups': warmups,
                'faults': list(FAULTS), 'cohorts': list(COHORTS), 'contract': contract_description(),
                'planned_cases': count*25, 'timing_scope': TIMING_SCOPE,
                'scope': 'New native layouts, fixed narrow 30 g rigid shoe proxy and PiPER physics. Not learned perception, a controller, deformable shoe validation, or hardware transfer.',
                'oracle': 'Frozen native _measure conditions across the latest 100 samples, independently of the new verifier. Perturbed observations retain original physical truth.',
                'blind_spot': 'Self-consistent fabricated complete-state observations on moving native states; count resulting physical false completions.',
                'selection': 'All seeded native cases are attempted. Hard construction failure seals a generation_failed manifest and aborts scoring; valid packets with category mismatches remain scored. No outcome-based filtering or retries.',
                'development_exposure': 'One default-seed layout was inspected in an initial complete/moving-alias unit check; no thresholds or generation parameters were tuned. Full preflight uses separate seed20261018. Fresh layouts are not claimed to be fully unseen.',
                'provider_calls': 0}
    write_json(output/'protocol.json', protocol)
    start = time.perf_counter()
    native = generate_native(Path(assets), output, layouts)
    hard_failures = [c['id'] for c in native if c['packet_file'] is None or c['physical_truth'] is None or c['now_s'] is None]
    if hard_failures:
        write_json(output/'cases.json', {'schema': 'dvidia.local-goal-cases/v1', 'status': 'generation_failed', 'cases': native})
        write_json(output/'generation.json', {'status': 'generation_failed', 'hard_failures': hard_failures,
            'attempted_native_cases': len(native), 'planned_cases': protocol['planned_cases'],
            'derived_controls_not_generated': count*17, 'wall_seconds': time.perf_counter()-start})
        files = {str(path.relative_to(output)): {'sha256': sha(path), 'bytes': path.stat().st_size}
                 for path in sorted(output.rglob('*')) if path.is_file()}
        write_json(output/'integrity.json', {'schema': 'dvidia.local-goal-integrity/v1', 'files': files})
        raise RuntimeError('Native construction failed; sealed diagnostics retained and no verifier score generated')
    cases = add_controls(output, native)
    if len(cases) != protocol['planned_cases']:
        raise ValueError('Generated inventory differs from protocol')
    write_json(output/'cases.json', {'schema': 'dvidia.local-goal-cases/v1', 'cases': cases})
    write_json(output/'generation.json', {'wall_seconds': time.perf_counter()-start,
        'source': protocol['source'], 'case_count': len(cases), 'invalid_cases': [c['id'] for c in cases if not c['generation_valid']]})
    files = {str(path.relative_to(output)): {'sha256': sha(path), 'bytes': path.stat().st_size}
             for path in sorted(output.rglob('*')) if path.is_file()}
    write_json(output/'integrity.json', {'schema': 'dvidia.local-goal-integrity/v1', 'files': files})
    return {'run': str(output), 'cases': len(cases), 'generation_invalid': sum(not c['generation_valid'] for c in cases)}


def verify_integrity(root):
    seal = read_json(Path(root)/'integrity.json')
    if seal.get('schema') != 'dvidia.local-goal-integrity/v1' or not seal.get('files'):
        raise ValueError('Missing input seal')
    for name, entry in seal['files'].items():
        path = relative_file(root, name)
        if path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
            raise ValueError('Input integrity mismatch: '+name)
    catalog_path = Path(root)/'cases.json'
    if catalog_path.is_file():
        if 'cases.json' not in seal['files']:
            raise ValueError('Case catalog must be sealed')
        catalog = read_json(catalog_path)
        if catalog.get('status') == 'generation_failed':
            raise ValueError('Cannot score failed native generation')
        for case in catalog['cases']:
            for key in ('packet_file', 'state_file', 'current_image', 'goal_image'):
                if case[key] not in seal['files']:
                    raise ValueError('Physical/evidence reference is not sealed: '+case[key])
                relative_file(root, case[key])
    return seal


def evaluate(root):
    root = Path(root)
    if (root/'results.json').exists():
        raise ValueError('Results already exist; create a fresh run to repeat')
    seal = verify_integrity(root)
    protocol, catalog = read_json(root/'protocol.json'), read_json(root/'cases.json')
    if source_identity() != protocol['source']:
        raise ValueError('Source changed after protocol creation')
    cases = catalog['cases']
    if len(cases) != protocol['planned_cases'] or 'cases.json' not in seal['files'] or 'protocol.json' not in seal['files']:
        raise ValueError('Protocol inventory mismatch')
    rows = []
    for case in cases:
        if case['packet_file'] not in seal['files']:
            raise ValueError('Unsealed input packet')
        packet = read_json(relative_file(root, case['packet_file']))
        contract = GoalContract(**case['contract'])
        for _ in range(protocol['warmups']):
            verify_goal(packet, contract=contract, now_s=case['now_s'])
        timings = []
        decision = None
        for _ in range(protocol['repeats']):
            started = time.perf_counter_ns()
            current = verify_goal(packet, contract=contract, now_s=case['now_s'])
            timings.append(time.perf_counter_ns()-started)
            if decision is not None and current != decision:
                raise ValueError('Nondeterministic verdict')
            decision = current
        rows.append({'case_id': case['id'], 'decision': decision, 'timing_ns': timings})
    result = {'schema': 'dvidia.local-goal-results/v1', 'status': 'completed', 'source_commit': protocol['source']['commit'],
              'protocol_sha256': sha(root/'protocol.json'), 'input_integrity_sha256': sha(root/'integrity.json'),
              'results': rows, 'metrics': score(rows, cases), 'timing_scope': TIMING_SCOPE,
              'environment': {'python': platform.python_version(), 'system': platform.system(), 'release': platform.release(),
                              'machine': platform.machine(), 'logical_cpus': os.cpu_count(), 'processor': platform.processor()},
              'provider_calls': 0}
    verify_integrity(root)
    write_json(root/'results.json', result)
    return result['metrics']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    gen = commands.add_parser('generate')
    gen.add_argument('--assets', required=True)
    gen.add_argument('--output', required=True)
    gen.add_argument('--seed', type=int, default=20261019)
    gen.add_argument('--count', type=int, default=8)
    gen.add_argument('--repeats', type=int, default=100)
    gen.add_argument('--warmups', type=int, default=10)
    ev = commands.add_parser('evaluate'); ev.add_argument('--run', required=True)
    one = commands.add_parser('verify')
    one.add_argument('--packet', required=True); one.add_argument('--contract', required=True); one.add_argument('--now-s', type=float, required=True)
    args = parser.parse_args(argv)
    if args.command == 'generate':
        value = generate(args.assets, args.output, seed=args.seed, count=args.count, repeats=args.repeats, warmups=args.warmups)
    elif args.command == 'evaluate':
        value = evaluate(args.run)
    else:
        value = verify_goal(read_json(args.packet), contract=GoalContract(**read_json(args.contract)), now_s=args.now_s)
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

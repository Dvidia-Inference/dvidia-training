"""Paired v0.1/v0.2 grip experiments, with independent retention scoring."""
from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import platform
import re
import time
import json

from . import prerequisite_benchmark as evidence
from . import asset_preview, reference_replay, reference_sim as sim

SCHEMA = 'dvidia.continuous-shoe-grip/v2'
SCOPE = evidence.SCOPE
POLICIES = ('baseline', 'recovery')


def implementation_hashes():
    root = Path(__file__).parent
    files = ('continuous_grip.py', 'grip_protocol.py', 'grip_protocol_v02.json', 'grip_benchmark.py',
             'tool_reach.py', 'workstation.py', 'asset_library.py', 'asset_catalog.json',
             'robot_profiles.py', 'prerequisite_benchmark.py', 'asset_preview.py',
             'grip_lab.html', 'grip_lab.js', 'grip_lab.bundle.js')
    return {name: sha256((root / name).read_bytes()).hexdigest() for name in files}


def protocol(split='heldout'):
    from . import grip_protocol
    return grip_protocol.protocol(split)


def summarize(records):
    groups = []
    for policy in POLICIES:
        rows = [r for r in records if r['policy'] == policy]
        outcomes = {}
        for row in rows:
            outcome = row.get('scored_outcome', row['metrics'].get('outcome', 'failed')) if row['valid'] else 'invalid_attempt'
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        groups.append({'task': 'tool-reach', 'policy': policy, 'attempts': len(rows),
                       'valid': sum(r['valid'] for r in rows),
                       'successes': sum(r['success'] and r['valid'] for r in rows),
                       'outcomes': outcomes, 'wall_seconds': sum(r['wall_seconds'] for r in rows)})
    return groups


def paired_comparisons(records):
    pairs = []
    for a in (r for r in records if r['policy'] == 'baseline'):
        b = next(r for r in records if r['policy'] == 'recovery' and r['case'] == a['case'])
        same_model = bool(a.get('independent_score', {}).get('compiled_model_sha256')) and (
            a.get('independent_score', {}).get('compiled_model_sha256') == b.get('independent_score', {}).get('compiled_model_sha256'))
        valid = a['valid'] and b['valid'] and same_model
        pairs.append({'task': 'tool-reach', 'case_id': a['case']['id'], 'valid_pair': valid,
                      'baseline_success': a['success'], 'recovery_success': b['success'],
                      'gain': valid and not a['success'] and b['success'],
                      'loss': valid and a['success'] and not b['success'],
                      'same_compiled_model': same_model if a['valid'] and b['valid'] else None,
                      'comparison_scope': 'Same mounted rake, robot and shoe physics; authored controller change.'})
    return pairs


def benchmark(assets, output, *, split='heldout', source_commit=None):
    from . import continuous_grip, grip_protocol
    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise ValueError('Benchmark output must be a fresh directory')
    if source_commit is not None and re.fullmatch(r'[a-f0-9]{40}', source_commit) is None:
        raise ValueError('Source commit must be a full Git SHA')
    plan = protocol(split)
    plan.update(schema=SCHEMA, source_commit=source_commit, implementation_sha256=implementation_hashes())
    mj, _ = sim._native_imports()
    provenance, notices = asset_preview._licenses('piper', assets)
    destination.mkdir(parents=True)
    evidence._write(destination, 'protocol.json', evidence.encode(plan))
    source = {'commit': source_commit, 'protocol_sha256': sha256(evidence.encode(plan)).hexdigest(),
              'mujoco': mj.__version__, 'robot_assets': provenance}
    records = []
    for attempt in plan['attempts']:
        start = time.perf_counter()
        row = {'id': attempt['id'], 'task': attempt['task'], 'policy': attempt['policy'],
               'case': attempt['case'], 'valid': False, 'success': False, 'metrics': {}, 'events': [], 'errors': []}
        try:
            result = continuous_grip.run_case(assets, attempt['case'], policy=attempt['policy'], record=True)
            row['wall_seconds'] = time.perf_counter() - start
            if type(result['success']) is not bool or type(result['valid']) is not bool:
                raise ValueError('Attempt validity and success must be explicit booleans')
            row.update(valid=result['valid'], success=result['success'] and result['valid'],
                       metrics=result['metrics'], events=result['events'])
            for key in ('native_trace', 'native_trace_columns', 'trace_columns', 'control_compute_ms', 'warnings'):
                if key in result:
                    row[key] = result[key]
            score = grip_protocol.audit_result(result)
            score['compiled_model_sha256'] = sim.model_digest(result['model'])
            row['independent_score'] = score
            row['controller_reported_success'] = result['success']
            row['success'] = bool(score['task_success'] and row['valid'])
            row['scored_outcome'] = ('completed' if row['success'] else 'other_robot_support'
                                     if score.get('legacy_only_claim') else row['metrics'].get('outcome', 'incomplete'))
            # An independent scoring disagreement is invalid evidence, never a silent success.
            row['replay'] = evidence.export_trial(destination, attempt['id'], result, source)
            evidence.encode(row)
        except Exception as exc:
            row.update(valid=False, success=False)
            row['errors'].append(type(exc).__name__ + ': attempt, audit or export failed')
            try:
                evidence.encode(row)
            except (ValueError, TypeError):
                row.update(metrics={}, events=[])
                for key in ('native_trace', 'native_trace_columns', 'trace_columns', 'control_compute_ms', 'warnings', 'independent_score'):
                    row.pop(key, None)
        row.setdefault('wall_seconds', time.perf_counter() - start)
        row['export_seconds'] = max(0., time.perf_counter() - start - row['wall_seconds'])
        evidence._write(destination, 'trials/' + attempt['id'] + '.json', evidence.encode(row))
        records.append({k: v for k, v in row.items() if k != 'native_trace'})
        print(json.dumps({'attempt': row['id'], 'success': row['success'], 'valid': row['valid'],
                          'outcome': row['metrics'].get('outcome') if row['valid'] else 'invalid_attempt',
                          'errors': row['errors']}), flush=True)
    report = {'schema': SCHEMA, 'scope': SCOPE, 'source': source, 'split': split,
              'host': {'system': platform.system(), 'machine': platform.machine(), 'python': platform.python_version(),
                       'compute': 'CPU native simulation; no GPU required for benchmark. Browser replay needs WebGL2.'},
              'learned_policy': False, 'hardware_validated': False, 'attempts': records,
              'groups': summarize(records), 'paired_comparisons': paired_comparisons(records),
              'timing_scope': 'Native wall time includes model setup and recording. Export/audit timing is separate; no camera, network or hardware latency.'}
    evidence._write(destination, 'benchmark.json', evidence.encode(report))
    resource = Path(__file__).parent
    for name in ('grip_lab.html', 'grip_lab.bundle.js', 'reference_replay.third-party.txt'):
        evidence._write(destination, 'index.html' if name.endswith('.html') else name, (resource / name).read_bytes())
    for name, content in notices.items():
        evidence._write(destination, name, content)
    evidence._write(destination, 'LICENSE', reference_replay.MIT_LICENSE.encode())
    evidence._write(destination, 'README.md', ('# DVIDIA continuous shoe grip v0.2\n\n' + SCOPE + '\n\n'
        'Paired authored controllers use identical mounted-rake models. The v0.1 baseline is the original '
        'tool-enabled recovery controller, not the earlier no-tool capability control.\n\n'
        'Serve this export: `python -m http.server 8000`. Browser visualization replays recorded native poses.\n\n'
        'Reproduce using the exact source commit, install `.[arm]`, fetch PiPER assets with '
        '`dvidia-robot-sim fetch piper --output robot-assets`, then '
        '`dvidia-grip-benchmark --assets robot-assets --output fresh-results --split ' + split + '`.\n\n'
        'All attempts are retained. Native retention is independently checked from dense contact traces. '
        'Privileged simulator observations, rigid synthetic shoes, and assumed contact properties limit transfer claims.\n\n'
        'DVIDIA code and synthetic objects: MIT. Upstream robot, Three.js and font notices are retained. '
        'Branding does not grant trademark rights. CPU benchmark; WebGL2 replay. Minimum RAM is not established.\n').encode())
    evidence.seal(destination)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=('development', 'regression', 'heldout'), default='development')
    parser.add_argument('--source-commit')
    args = parser.parse_args()
    result = benchmark(args.assets, args.output, split=args.split, source_commit=args.source_commit)
    print(json.dumps({'groups': result['groups']}))
    return 0 if all(row['valid'] for row in result['attempts']) else 1


if __name__ == '__main__':
    raise SystemExit(main())

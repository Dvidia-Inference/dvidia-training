"""Independent, bounded readback of already frozen evidence; no native imports.

This does not rerun a simulation, modify a source, replace the original failed
64 MiB reader report, or change the predeclared adoption rule. It streams the
same canonical contact/scorer JSON with a 512 MiB expanded bound and checks
saved arrays, cadence, active forces, contact histograms and cumulative scorer.
"""
from __future__ import annotations
import argparse
import gzip
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import tempfile

MAX_BYTES = 512*1024*1024
CASES = ('original', 'open-jaw', 'hold', 'feedback-no-pulse', 'feedback-pulse')
EPISODE_FIELDS = ('config', 'seed', 'valid', 'success', 'reason', 'error', 'simulated_seconds',
                  'initial_object_position_m', 'compiled_model_sha256', 'trajectory', 'actions', 'metrics')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def unique(rows):
    result = {}
    for key, value in rows:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def nonfinite(token):
    raise ValueError('Nonfinite JSON token: '+token)


DECODER = json.JSONDecoder(object_pairs_hook=unique, parse_constant=nonfinite)


def read(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique, parse_constant=nonfinite)


def file_id(path):
    digest, size = sha256(), 0
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024*1024):
            digest.update(chunk)
            size += len(chunk)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def expanded_id(path):
    digest, size = sha256(), 0
    with gzip.open(path, 'rb') as stream:
        while chunk := stream.read(65536):
            size += len(chunk)
            if size > MAX_BYTES:
                raise ValueError('Expanded evidence exceeds 512 MiB')
            digest.update(chunk)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def rows(path, chunk_size=65536):
    """Incrementally decode one bounded top-level JSON array without loading it."""
    with gzip.open(path, 'rt', encoding='utf-8', errors='strict') as stream:
        buffer, total, ended = '', 0, False

        def fill():
            nonlocal buffer, total, ended
            chunk = stream.read(chunk_size)
            if not chunk:
                ended = True
                return False
            total += len(chunk.encode('utf-8'))
            if total > MAX_BYTES:
                raise ValueError('Expanded evidence exceeds 512 MiB')
            buffer += chunk
            return True

        def available():
            nonlocal buffer
            buffer = buffer.lstrip()
            while not buffer and not ended:
                fill()
                buffer = buffer.lstrip()
            if not buffer:
                raise ValueError('Truncated JSON array')

        available()
        if buffer[0] != '[':
            raise ValueError('Expected top-level JSON array')
        buffer = buffer[1:]
        may_end = True
        while True:
            available()
            if buffer[0] == ']':
                if not may_end:
                    raise ValueError('Trailing array comma')
                buffer = buffer[1:]
                while not ended:
                    fill()
                if buffer.strip():
                    raise ValueError('Data after JSON array')
                return
            while True:
                try:
                    item, offset = DECODER.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if len(buffer) > 1024*1024 or not fill():
                        raise ValueError('Malformed or oversized row') from None
            if type(item) is not dict:
                raise ValueError('Expected object row')
            canonical(item)
            buffer = buffer[offset:]
            yield item
            available()
            if buffer[0] == ',':
                buffer = buffer[1:]
                may_end = False
            elif buffer[0] == ']':
                may_end = True
            else:
                raise ValueError('Expected array delimiter')


def scorer_check(path, config):
    scorer = dict(elapsed=0., steps=0, dwell=0., lift_dwell=0., max_lift=0., max_pad_force=0.,
                  critical_contact_ticks=0, lift_seen=False, grasp_seen=False, carry_complete=False,
                  retention_lost=False, carry_contact_loss=0., max_carry_contact_loss=0.,
                  carry_contact_loss_events=0, first_success_time=None, reason='running', done=False)
    histogram, count = {}, 0
    for index, row in enumerate(rows(path)):
        assert row['native_tick'] == index
        assert abs(row['scorer']['elapsed']-index*.001) < 1e-8
        assert row['warnings'] == [] and all(item['number'] == 0 for item in row['native_warnings'])
        contacts = row['contacts']
        active = []
        for raw in contacts['raw']:
            assert all(math.isfinite(value) for value in raw['force_native'])
            if raw['force_native'][0] > .01:
                active.append({'geom_ids': sorted(set(raw['geom_ids'])),
                               'normal_force_n': raw['force_native'][0]})
        assert canonical(active) == canonical(contacts['active_pairs'])
        if index:
            for pair in {tuple(item['geom_ids']) for item in active}:
                name = ':'.join(map(str, pair))
                histogram[name] = histogram.get(name, 0)+1
            x = row['scorer_inputs']
            dt = config['timestep']
            scorer['elapsed'] += dt
            scorer['steps'] = (index-1)//20
            scorer['max_pad_force'] = max(scorer['max_pad_force'], x['pad_force'])
            scorer['critical_contact_ticks'] += bool(x['critical'])
            scorer['grasp_seen'] |= x['both']
            scorer['max_lift'] = max(scorer['max_lift'], x['lift'])
            loaded = x['both'] and not x['supported'] and x['lift'] >= config['min_lift_m']
            scorer['lift_dwell'] = scorer['lift_dwell']+dt if loaded else 0.
            scorer['lift_seen'] |= scorer['lift_dwell']+1e-12 >= config['min_lift_dwell_s']
            if scorer['lift_seen'] and not scorer['carry_complete']:
                scorer['carry_complete'] |= bool(x['both'] and x['supported'] and x['distance'] <= config['goal_position_tolerance_m'])
                if not scorer['carry_complete']:
                    if x['both']:
                        scorer['carry_contact_loss'] = 0.
                    else:
                        before = scorer['carry_contact_loss']
                        scorer['carry_contact_loss'] += dt
                        scorer['max_carry_contact_loss'] = max(scorer['max_carry_contact_loss'], scorer['carry_contact_loss'])
                        if before+1e-12 < .04 <= scorer['carry_contact_loss']+1e-12:
                            scorer['carry_contact_loss_events'] += 1
                            scorer['retention_lost'] = True
            qualifies = (scorer['lift_seen'] and scorer['grasp_seen'] and scorer['carry_complete']
                         and not scorer['retention_lost'] and x['released'] and x['supported']
                         and x['settled'] and x['clear'] and x['distance'] <= config['goal_position_tolerance_m']
                         and scorer['critical_contact_ticks'] == 0)
            scorer['dwell'] = scorer['dwell']+dt if qualifies else 0.
            if scorer['dwell']+1e-12 >= config['goal_dwell_s'] and scorer['first_success_time'] is None:
                scorer['first_success_time'] = scorer['elapsed']
            assert canonical(scorer) == canonical(row['scorer']), ('scorer', index)
        else:
            assert row['scorer_inputs'] is None
            assert canonical(scorer) == canonical(row['scorer'])
        assert histogram == row['contact_pair_native_ticks'], ('histogram', index)
        count += 1
    assert count == 20001
    return {'rows': count, 'native_cadence_s': .001, 'scorer_recomputed_exact': True,
            'active_pairs_recomputed_from_raw_forces': True,
            'contact_pair_native_histogram_exact': True, 'warnings': 0}


def arrays_check(left, right):
    import numpy as np
    fields = []
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        assert set(a.files) == set(b.files)
        for name in a.files:
            x, y = a[name], b[name]
            assert x.dtype.kind in 'iufb' and y.dtype == x.dtype
            assert x.shape == y.shape and x.shape[0] == 20001
            assert np.isfinite(x).all() and np.isfinite(y).all()
            assert x.tobytes() == y.tobytes(), name
            if name in ('qfrc_applied', 'xfrc_applied'):
                assert not np.any(x)
            fields.append({'name': name, 'shape': list(x.shape), 'dtype': str(x.dtype)})
        times = a['native_time']
        assert np.max(np.abs(times-times[0]-np.arange(20001)*.001)) < 1e-8
    return {'exact_bitwise_match': True, 'rows': 20001, 'all_finite': True,
            'external_forces_zero': True, 'native_clock_cadence_s': .001, 'fields': fields}


def read_episode(path):
    with gzip.open(path, 'rt') as stream:
        value = json.load(stream, object_pairs_hook=unique, parse_constant=nonfinite)
    canonical(value)
    return value


def self_test():
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary)/'rows.gz'
        value = [{'native_tick': n, 'x': [0., {'value': 'text'}]} for n in range(5)]
        path.write_bytes(gzip.compress(canonical(value)))
        for size in (1, 7, 65536):
            assert list(rows(path, size)) == value
        assert expanded_id(path) == {'bytes': len(canonical(value)), 'sha256': sha256(canonical(value)).hexdigest()}
        for raw in (b'[{"x":1},]', b'[{"x":1}]junk', b'[{"x":1}', b'[{"x":NaN}]', b'[{"x":1,"x":2}]'):
            path.write_bytes(gzip.compress(raw))
            try:
                list(rows(path, 2))
            except ValueError:
                pass
            else:
                raise AssertionError('Malformed input accepted')
    return {'passed': True, 'checks': 9}


def main():
    import sys
    if sys.flags.optimize:
        raise RuntimeError('Run this assertion-based research auditor without -O or PYTHONOPTIMIZE.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, help='New audit path; defaults to ROOT/readback-audit.json')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
        return
    root = args.root.resolve()
    script = Path(__file__).resolve()
    output = args.output if args.output is not None else root/'readback-audit.json'
    if output.exists():
        raise ValueError('Never overwrite an audit')
    baseline_report = read(root/'comparison.json')
    protocol = read(root/'protocol.json')
    frozen_files = {str(path.relative_to(root)): file_id(path) for path in sorted(root.rglob('*'))
                    if path.is_file() and path.name not in ('readback-audit.json',)}
    checks = []
    for case in CASES:
        a, b = root/('dense-'+case+'-baseline'), root/('dense-'+case+'-candidate')
        ra, rb = read(a/'result.json'), read(b/'result.json')
        assert ra['status'] == rb['status'] == 'complete'
        ea, eb = read_episode(a/'episode.json.gz'), read_episode(b/'episode.json.gz')
        assert all(canonical(ea[name]) == canonical(eb[name]) for name in EPISODE_FIELDS)
        assert all(canonical(ra[name]) == canonical(rb[name]) for name in ('model', 'runtime', 'assets', 'score'))
        left_id, right_id = expanded_id(a/'native-contacts-scorer.json.gz'), expanded_id(b/'native-contacts-scorer.json.gz')
        assert left_id == right_id
        arrays = arrays_check(a/'native-arrays.npz', b/'native-arrays.npz')
        # Canonical payload identity was checked independently for both files;
        # parsing one identical payload suffices for cadence/scorer invariants.
        scorer = scorer_check(a/'native-contacts-scorer.json.gz', protocol['config'])
        expected_success = case not in ('open-jaw', 'hold')
        assert ra['score']['success'] is expected_success
        checks.append({'case': case, 'exact_boundaries_model_runtime_assets_score': True,
                       'canonical_contact_scorer_payload_identity': left_id,
                       'arrays': arrays, 'scorer': scorer, 'score_success': expected_success})
        print(json.dumps({'case': case, 'exact': True, 'rows': scorer['rows'], 'expanded_bytes': left_id['bytes']}), flush=True)
    assert all(row['returncode'] == 0 and 'error' not in row for row in baseline_report['attempts'])
    assert all(row['exact_match'] for row in baseline_report['comparisons'] if row['kind'] == 'timing-pair')
    for relative, expected in frozen_files.items():
        assert file_id(root/relative) == expected, ('changed evidence', relative)
    rule = baseline_report['adoption']
    timing_eligible = (all(row[clock] for row in rule['paired_lower_mean'] for clock in ('step_wall', 'step_cpu'))
                       and all(row['candidate_mean_reduction_percent'] >= 5. for row in rule['pooled'].values()))
    result = {'schema': 'dvidia.piper-control-readback-audit/v1', 'passed': True,
              'scope': 'Read-only reanalysis of the same 14 frozen attempts, no simulation/timing reruns. Original fail-closed 64 MiB reader report is preserved.',
              'script': file_id(script), 'self_tests': self_test(),
              'original_report': file_id(root/'comparison.json'), 'protocol': file_id(root/'protocol.json'),
              'expanded_bound_bytes': MAX_BYTES, 'frozen_evidence': frozen_files,
              'evidence_unchanged': True, 'cases': checks,
              'total_dense_snapshots_compared': 20001*len(CASES)*2,
              'timing_runs': baseline_report['timing_runs'], 'timing_adoption_rule': protocol['adoption_rule'],
              'eligible_under_predeclared_rule_after_complete_readback': timing_eligible,
              'interpretation': 'Observed two-pair shared-host CPU improvement with exact saved mechanics. Not a general latency guarantee, accuracy gain, hardware test, or recovery-robustness evaluation; the fixed pulse also succeeds.'}
    with output.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'passed': True, 'eligible': timing_eligible, 'audit': file_id(output)}))


if __name__ == '__main__':
    main()

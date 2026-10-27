"""Measure saved jaw-actuator-target pulse exposure without running physics.

Scheduled pulse presence is distinct from a positive applied-target change.
This compares the actual applied target with the logged normal rate-limited
target. It does not recreate a no-pulse trajectory or prove object disturbance.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np

SCHEMA = 'dvidia.piper-jaw-pulse-exposure/v1'
VARIANTS = ('feedback', 'blind', 'feedforward', 'teacher', 'open_jaw')
DT = .001
THRESHOLD = 1e-9
MAX_FILE = 8*1024*1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def plain(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() and p not in {Path('/tmp'), Path('/var')} for p in (path, *path.parents)), 'Artifact symlinks are forbidden.')
    return path.resolve()


def read(path):
    path = plain(path)
    require(path.is_file() and path.stat().st_size <= MAX_FILE, 'Missing or oversized evidence file.')
    raw = path.read_bytes()
    require(len(raw) <= MAX_FILE, 'Evidence grew beyond its byte bound.')
    return raw


def document(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON evidence key.')
            result[key] = value
        return result
    value = json.loads(read(path), object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON evidence.')))
    require(isinstance(value, dict), 'Expected a JSON object.')
    return value


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()


def intervals(mask, times):
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)-1
    return [dict(start_s=float(times[first]-DT), end_s=float(times[last]), duration_s=float((last-first+1)*DT)) for first, last in zip(starts, ends)]


def quantiles(values):
    return dict(samples=len(values), p50=float(np.percentile(values, 50)), p95=float(np.percentile(values, 95)),
                maximum=float(max(values))) if values else dict(samples=0, p50=None, p95=None, maximum=None)


def analyze(run):
    root = plain(run)
    audit_raw = read(root/'audit.json')
    audit = document(root/'audit.json')
    report, protocol = document(root/'report.json'), document(root/'protocol.json')
    require(audit.get('status') == 'passed' and audit.get('report_sha256') == sha256(read(root/'report.json')).hexdigest()
            and audit.get('protocol_sha256') == sha256(read(root/'protocol.json')).hexdigest(), 'The existing independent audit does not bind this report and protocol.')
    cases = [case for case in protocol['cases']['confirmation'] if case['condition'] == 'jaw_interruption']
    require(len(cases) == 12 and all(isinstance(c['pulses'], list) and len(c['pulses']) == 1 for c in cases), 'Expected twelve literally scheduled single-pulse scenes.')
    audited = {row['path']: row for row in audit['attempts']}
    rows = []
    for case in cases:
        pulse = case['pulses'][0]
        for variant in VARIANTS:
            relative = f"confirmation/{case['id']}/{variant}"
            folder = root/relative
            receipt = document(folder/'receipt.json')
            native_raw = read(folder/'native.npz')
            native_sha = sha256(native_raw).hexdigest()
            require(receipt.get('files', {}).get('native.npz') == native_sha
                    and audited.get(relative, {}).get('native_npz_sha256') == native_sha
                    and receipt.get('binding', {}).get('case') == case
                    and receipt.get('binding', {}).get('variant') == variant,
                    'Native evidence differs from its original receipt or independent audit.')
            with zipfile.ZipFile(io.BytesIO(native_raw)) as archive:
                require(sum(i.file_size for i in archive.infolist()) <= 128*1024*1024, 'Native archive exceeds its decoded bound.')
            with np.load(io.BytesIO(native_raw), allow_pickle=False) as archive:
                names = ('t_seconds', 'native_tick', 'applied_jaw_target_m', 'normal_rate_limited_jaw_target_m',
                         'pulse_active', 'pulse_closing_tail', 'table_object_contact_count', 'grasp_contact_counts', 'pad_normal_forces_n')
                a = {name: archive[name] for name in names}
            require(all(v.dtype.kind in 'fiu' and np.isfinite(v).all() for v in a.values())
                    and all(a[name].shape == ((20000, 2) if name in ('grasp_contact_counts', 'pad_normal_forces_n') else (20000,)) for name in names)
                    and np.array_equal(a['native_tick'], np.arange(1, 20001))
                    and np.allclose(a['t_seconds'], np.arange(1, 20001)*DT, atol=1e-8, rtol=0), 'Incomplete or malformed native target trace.')
            delta = a['applied_jaw_target_m']-a['normal_rate_limited_jaw_target_m']
            positive, negative = delta > THRESHOLD, delta < -THRESHOLD
            loaded = np.all(a['grasp_contact_counts'] > 0, axis=1) & np.all(a['pad_normal_forces_n'] > .01, axis=1)
            supported = a['table_object_contact_count'] > 0
            first = int(np.flatnonzero(positive)[0]) if positive.any() else None
            before = None
            if first is not None:
                # The previous completed native tick precedes the first changed
                # actuator-target integration. Current schedules begin well
                # after reset; refuse to infer a missing pre-effect snapshot.
                require(first > 0, 'First effect at reset lacks a prior native snapshot.')
                previous = first-1
                streak = 0
                for value in loaded[:first][::-1]:
                    if not value:
                        break
                    streak += 1
                before = dict(t_seconds=float(a['t_seconds'][previous]), supported=bool(supported[previous]),
                    loaded_bilateral=bool(loaded[previous]), contiguous_loaded_duration_s=float(streak*DT),
                    loaded_for_at_least_20ms=streak >= 20,
                    ever_loaded_before_effect=bool(loaded[:first].any()),
                    contact_counts=a['grasp_contact_counts'][previous].tolist(), pad_normal_forces_n=a['pad_normal_forces_n'][previous].tolist())
            times = a['t_seconds']
            first_start = float(times[first]-DT) if first is not None else None
            rows.append(dict(case_id=case['id'], variant=variant, native_npz_sha256=native_sha,
                scheduled_pulse=pulse, strict_success=audited[relative]['success'], any_positive_applied_target_change=bool(positive.any()),
                positive_exposure_native_ticks=int(positive.sum()), positive_exposure_duration_s=float(positive.sum()*DT),
                positive_exposure_during_active_pulse_s=float(np.count_nonzero(positive & (a['pulse_active'] == 1))*DT),
                positive_exposure_during_closing_tail_s=float(np.count_nonzero(positive & (a['pulse_closing_tail'] == 1))*DT),
                maximum_positive_target_delta_m=max(0., float(delta.max())), minimum_target_delta_m=float(delta.min()),
                negative_target_delta_native_ticks=int(negative.sum()), positive_exposure_intervals=intervals(positive, times),
                first_effect_step_start_s=first_start, first_effect_completed_tick_s=float(times[first]) if first is not None else None,
                scheduled_onset_to_first_effect_step_s=float(first_start-pulse['start_s']) if first is not None else None,
                state_immediately_before_first_effect=before))
    require(len(rows) == 60, 'Analysis omitted a scheduled interrupted confirmation attempt.')
    aggregates = {}
    for variant in VARIANTS:
        selected = [r for r in rows if r['variant'] == variant]
        exposed = [r for r in selected if r['any_positive_applied_target_change']]
        aggregates[variant] = dict(scheduled=12, attempts_with_positive_applied_target_change=len(exposed),
            attempts_without_positive_applied_target_change=12-len(exposed),
            total_positive_exposure_duration_s=sum(r['positive_exposure_duration_s'] for r in selected),
            positive_duration_s_all_attempts=quantiles([r['positive_exposure_duration_s'] for r in selected]),
            maximum_positive_target_delta_m=max(r['maximum_positive_target_delta_m'] for r in selected),
            first_effect_step_start_s_exposed_attempts=quantiles([r['first_effect_step_start_s'] for r in exposed]),
            onset_to_effect_step_s_exposed_attempts=quantiles([r['scheduled_onset_to_first_effect_step_s'] for r in exposed]),
            before_effect_supported_attempts=sum(r['state_immediately_before_first_effect']['supported'] for r in exposed),
            before_effect_loaded_attempts=sum(r['state_immediately_before_first_effect']['loaded_bilateral'] for r in exposed),
            before_effect_supported_and_loaded_attempts=sum(r['state_immediately_before_first_effect']['supported'] and r['state_immediately_before_first_effect']['loaded_bilateral'] for r in exposed),
            before_effect_loaded_for_20ms_attempts=sum(r['state_immediately_before_first_effect']['loaded_for_at_least_20ms'] for r in exposed),
            strict_successes_with_exposure=sum(r['strict_success'] for r in exposed),
            strict_successes_without_exposure=sum(r['strict_success'] for r in selected if not r['any_positive_applied_target_change']))
    require(read(root/'audit.json') == audit_raw, 'Independent audit changed while exposure was analyzed.')
    return dict(schema=SCHEMA, status='completed', physical_robot_ready=False, native_physics_run=False,
        audit_sha256=sha256(audit_raw).hexdigest(), report_sha256=audit['report_sha256'], protocol_sha256=audit['protocol_sha256'],
        analysis_script_sha256=sha256(read(Path(__file__))).hexdigest(), native_dt_s=DT, positive_delta_threshold_m=THRESHOLD,
        definition='Positive exposure means applied_jaw_target_m - normal_rate_limited_jaw_target_m > 1e-9 metres at a logged 1 ms native tick, including a rate-limited closing tail.',
        timing_definition='Rows are post-physics. First effect step start is the first positive row timestamp minus 1 ms. The preceding completed native row supplies the pre-effect contact state.',
        loaded_definition='Both finger contact counts > 0 and both pad normal forces > 0.01 N; support means table-object contact count > 0.',
        limitations=['This is effective actuator-target exposure, not a measured physical displacement, force impulse or proven grip loss.',
            'The logged normal target is contemporaneous with the perturbed trajectory; it is not a rerun of the no-pulse counterfactual.',
            'A scheduled pulse can have zero positive target effect when the policy already requests an equally open jaw.',
            'Counts concern only the 60 scheduled interrupted confirmation attempts; nominal attempts are excluded by design.'],
        attempts=rows, aggregates=aggregates)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', type=Path, help='Fresh JSON; defaults to RUN/docs/pulse-exposure.json.')
    args = parser.parse_args(argv)
    try:
        root = plain(args.run)
        output = plain(args.output or root/'docs/pulse-exposure.json')
        require(not output.exists() and not output.is_symlink(), 'Exposure output already exists; refusing to overwrite it.')
        require(root not in output.parents or output == root/'docs/pulse-exposure.json', 'Only the new exposure document may be written inside the study.')
        result = analyze(root)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('xb') as stream:
            stream.write(encoded(result))
        print(json.dumps(dict(schema=SCHEMA, status='completed', attempts=len(result['attempts']), aggregates=result['aggregates'],
            output_sha256=sha256(encoded(result)).hexdigest(), audit_unchanged=True, native_physics_run=False), indent=2, sort_keys=True))
    except Exception as exc:
        print(str(exc) if isinstance(exc, ValueError) else f'Exposure analysis stopped: {type(exc).__name__}; raw error suppressed.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

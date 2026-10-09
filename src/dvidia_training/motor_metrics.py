"""Offline evidence summaries for the bounded motor-memory experiment v0.2.

No aggregate is a safety/readiness score. Development and confirmation evidence
remain separate, and failed or censored attempts remain in denominators.
"""
from collections import defaultdict
from html import escape
import json
import math
from statistics import mean, median


def _number(value, name, *, nonnegative=False):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be finite numeric or null')
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f'{name} must be finite') from exc
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f'{name} must be finite' + (' and nonnegative' if nonnegative else ''))
    return value


def _rate(numerator, denominator):
    if not denominator:
        return {'estimate': None, 'numerator': numerator, 'denominator': 0, 'ci95_wilson': None}
    z = 1.959963984540054
    p = numerator / denominator
    scale = 1 + z * z / denominator
    center = (p + z * z / (2 * denominator)) / scale
    radius = z * math.sqrt(p * (1 - p) / denominator + z * z / (4 * denominator ** 2)) / scale
    return {'estimate': p, 'numerator': numerator, 'denominator': denominator,
            'ci95_wilson': {'lower': max(0., center - radius), 'upper': min(1., center + radius)}}


def _distribution(values, attempted):
    values = sorted(value for value in values if value is not None)
    return {'count': len(values), 'missing': attempted - len(values),
            'mean': mean(values) if values else None,
            'median': median(values) if values else None,
            'p95': values[max(0, math.ceil(.95 * len(values)) - 1)] if values else None,
            'min': values[0] if values else None, 'max': values[-1] if values else None,
            'worst': values[-1] if values else None}


MEASUREMENTS = (
    'final_error_m', 'final_speed_m_s', 'overshoot_m', 'rms_error_m',
    'error_at_completion_m', 'speed_at_completion_m_s', 'peak_error_after_completion_m',
    'peak_acceleration_m_s2', 'peak_jerk_m_s3', 'peak_force_proxy_n',
    'contact_impulse_ns', 'max_penetration_m', 'reaction_s', 'sensor_unusable_s',
    'predictor_rmse_m', 'command_latency_s', 'controller_latency_p50_s',
    'controller_latency_p95_s', 'controller_latency_max_s',
)
_NUMERIC = set(MEASUREMENTS) | {
    'completed_s', 'truth_settled_s', 'simulated_seconds', 'wall_seconds', 'step_count',
    'target_final_m', 'speed_limit_m_s',
}
_BOOLS = ('holding', 'expected_completion', 'completion_false', 'completion_revoked',
          'rest_lost_after_confirmation', 'success', 'dropped')


def _checked(records):
    output = []
    for index, source in enumerate(records):
        if not isinstance(source, dict):
            raise ValueError(f'record {index} must be an object')
        record = dict(source)
        for key in ('case_id', 'variant'):
            if not isinstance(record.get(key), str) or not record[key]:
                raise ValueError(f'record {index}.{key} must be a nonempty string')
        if record.get('split') not in ('development', 'confirmation'):
            raise ValueError(f'record {index}.split must be development or confirmation')
        if type(record.get('valid')) is not bool:
            raise ValueError(f'record {index}.valid must be boolean')
        for key in _BOOLS:
            if record.get(key) is not None and type(record[key]) is not bool:
                raise ValueError(f'record {index}.{key} must be boolean or null')
        for key in _NUMERIC:
            record[key] = _number(record.get(key), f'record {index}.{key}', nonnegative=key != 'target_final_m')
        if record['speed_limit_m_s'] is None or record['speed_limit_m_s'] <= 0:
            raise ValueError(f'record {index}.speed_limit_m_s must be positive')
        if record['step_count'] is not None and not record['step_count'].is_integer():
            raise ValueError(f'record {index}.step_count must be an integer')
        if record.get('seed') is not None and (type(record['seed']) is not int or record['seed'] < 0):
            raise ValueError(f'record {index}.seed must be a nonnegative integer or null')
        for key in ('completed_s', 'truth_settled_s'):
            if (record[key] is not None and record['simulated_seconds'] is not None
                    and record[key] > record['simulated_seconds'] + 1e-12):
                raise ValueError(f'record {index}.{key} exceeds the simulated horizon')
        if record.get('trace') is not None and not isinstance(record['trace'], list):
            raise ValueError(f'record {index}.trace must be an array')
        output.append(record)
    return output


def _stratum(records, *, include_expectation=True):
    valid = [r for r in records if r['valid']]
    known_success = [r for r in valid if r.get('success') is not None]
    successes = [r for r in known_success if r['success']]
    # False flags can refer to an earlier claim superseded by a public goal change.
    claims = [r for r in valid if r.get('completed_s') is not None or r.get('completion_false') is True
              or bool(r.get('completion_claims'))]
    assessed_claims = [r for r in claims if r.get('completion_false') is not None]
    false_claims = sum(r['completion_false'] for r in assessed_claims)
    expected = [r for r in valid if r.get('expected_completion') is True]
    holding = [r for r in valid if r.get('holding') is True]
    known_load = [r for r in holding if r.get('dropped') is not None]
    verified = [r for r in successes if r.get('completed_s') is not None and r.get('completion_false') is False]
    measurements = {name: _distribution([r.get(name) for r in valid], len(valid)) for name in MEASUREMENTS}
    measurements['claimed_completion_s'] = _distribution([r.get('completed_s') for r in valid], len(valid))
    measurements['verified_success_completion_s'] = _distribution([r.get('completed_s') for r in verified], len(successes))
    measurements['truth_settled_s'] = _distribution([r.get('truth_settled_s') for r in valid], len(valid))
    wall = sum(r.get('wall_seconds') or 0. for r in records)
    simulation = sum(r.get('simulated_seconds') or 0. for r in records)
    steps = sum(r.get('step_count') or 0. for r in records)
    result = {
        'attempted': len(records), 'valid': len(valid), 'invalid': len(records) - len(valid),
        'errors': sum(r.get('error') is not None for r in records),
        'expected_completion': {
            'attempted_positive': sum(r.get('expected_completion') is True for r in records),
            'attempted_negative': sum(r.get('expected_completion') is False for r in records),
            'attempted_unknown': sum(r.get('expected_completion') is None for r in records),
            'invalid_positive': sum(not r['valid'] and r.get('expected_completion') is True for r in records),
            'invalid_negative': sum(not r['valid'] and r.get('expected_completion') is False for r in records),
            'valid_expected': len(expected),
        },
        'success': {'known_valid': len(known_success), 'unknown_valid': len(valid) - len(known_success),
                    'succeeded': len(successes), 'failed_known': len(known_success) - len(successes),
                    'conditional_valid': _rate(len(successes), len(known_success)),
                    'all_attempts': _rate(len(successes), len(records))},
        'completion': {'valid_claims': len(claims), 'assessed_claims': len(assessed_claims),
                       'claim_truth_unknown': len(claims) - len(assessed_claims), 'false_claims': false_claims,
                       'false_claims_without_final_completion': sum(r.get('completion_false') is True
                                                                  and r.get('completed_s') is None for r in valid),
                       'revoked': sum(r.get('completion_revoked') is True for r in valid),
                       'revocation_status_unknown': sum(r.get('completion_revoked') is None for r in valid),
                       'rest_lost_after_confirmation': sum(r.get('rest_lost_after_confirmation') is True for r in valid),
                       'rest_loss_status_unknown': sum(r.get('rest_lost_after_confirmation') is None for r in valid),
                       'false_given_assessed_claim': _rate(false_claims, len(assessed_claims)),
                       'false_claim_per_attempt': _rate(false_claims, len(records)),
                       'expected_censored': sum(r.get('completed_s') is None for r in expected),
                       'truth_unsettled': sum(r.get('truth_settled_s') is None for r in expected),
                       'verified_successes_with_completion': len(verified)},
        'load': {'holding_valid': len(holding), 'outcome_known': len(known_load),
                 'outcome_unknown': len(holding) - len(known_load),
                 'not_holding_or_unknown': len(valid) - len(holding),
                 'dropped': sum(r['dropped'] for r in known_load),
                 'drop_rate': _rate(sum(r['dropped'] for r in known_load), len(known_load))},
        'measurements': measurements,
        'throughput': {'wall_seconds': wall, 'simulated_seconds': simulation, 'step_count': int(steps),
                       'steps_per_wall_second': steps / wall if wall and all(
                           r.get('wall_seconds') is not None and r.get('step_count') is not None for r in records) else None,
                       'simulated_seconds_per_wall_second': simulation / wall if wall and all(
                           r.get('wall_seconds') is not None and r.get('simulated_seconds') is not None for r in records) else None,
                       'wall_time_missing': sum(r.get('wall_seconds') is None for r in records),
                       'simulated_time_missing': sum(r.get('simulated_seconds') is None for r in records),
                       'step_count_missing': sum(r.get('step_count') is None for r in records)}}
    if include_expectation:
        result['by_expected_completion'] = {
            name: _stratum([r for r in records if r.get('expected_completion') is expected], include_expectation=False)
            for name, expected in (('expected', True), ('not_expected', False), ('unknown', None))}
    return result


def _frontier(records):
    settings = defaultdict(list)
    for record in records:
        settings[(record['variant'], record['speed_limit_m_s'])].append(record)
    rows = []
    for (variant, speed), attempted in sorted(settings.items()):
        paired = [r for r in attempted if r['valid'] and r.get('success') is True
                  and r.get('completion_false') is False and r.get('completed_s') is not None
                  and r.get('error_at_completion_m') is not None]
        rows.append({'variant': variant, 'speed_limit_m_s': speed,
                     'summary': _stratum(attempted), 'plotted_verified_success_count': len(paired),
                     'completion_time_s': median(r['completed_s'] for r in paired) if paired else None,
                     'error_at_completion_m': median(r['error_at_completion_m'] for r in paired) if paired else None,
                     'final_error_m': median(r['final_error_m'] for r in paired
                                             if r.get('final_error_m') is not None) if any(
                                                 r.get('final_error_m') is not None for r in paired) else None,
                     'point_scope': 'medians over the same valid, verified successful trials only; counts and failures remain alongside'})
    return rows


def _matched(records):
    groups = defaultdict(lambda: defaultdict(list))
    for record in records:
        if record.get('seed') is not None:
            groups[(record['split'], record['case_id'], record['seed'], record['speed_limit_m_s'])][record['variant']].append(record)
    output = []
    for (split, case, seed, speed), variants in sorted(groups.items()):
        row = {'split': split, 'case_id': case, 'seed': seed, 'speed_limit_m_s': speed,
               'variant_attempt_counts': {k: len(v) for k, v in sorted(variants.items())}, 'comparisons': []}
        baseline = variants.get('slow_feedback', [])
        for variant in ('fast_fixed', 'adaptive_predictive'):
            candidate = variants.get(variant, [])
            unique = len(baseline) == len(candidate) == 1
            valid = unique and baseline[0]['valid'] and candidate[0]['valid']
            delta = {'variant': variant, 'baseline': 'slow_feedback', 'unique_pair': unique, 'both_valid': bool(valid)}
            for key in ('completed_s', 'error_at_completion_m', 'speed_at_completion_m_s',
                        'peak_error_after_completion_m', 'final_error_m', 'overshoot_m', 'rms_error_m',
                        'peak_acceleration_m_s2', 'peak_jerk_m_s3', 'peak_force_proxy_n',
                        'contact_impulse_ns', 'max_penetration_m'):
                a, b = (candidate[0].get(key), baseline[0].get(key)) if valid else (None, None)
                if key == 'completed_s' and valid and not all(
                    r.get('success') is True and r.get('completion_false') is False
                    for r in (candidate[0], baseline[0])):
                    a = b = None
                delta[key + '_candidate_minus_baseline'] = a - b if a is not None and b is not None else None
            delta['candidate_success'] = candidate[0].get('success') if len(candidate) == 1 else None
            delta['baseline_success'] = baseline[0].get('success') if len(baseline) == 1 else None
            for key in ('completion_false', 'completion_revoked'):
                delta['candidate_' + key] = candidate[0].get(key) if len(candidate) == 1 else None
                delta['baseline_' + key] = baseline[0].get(key) if len(baseline) == 1 else None
            delta['completion_delta_scope'] = 'both truth-successful with no false claim; other deltas use both valid attempts'
            row['comparisons'].append(delta)
        output.append(row)
    return output


def summarize(records):
    """Report development/confirmation independently and retain all attempts."""
    records = _checked(records)
    splits = defaultdict(list)
    for record in records:
        splits[record['split']].append(record)
    return {'attempted': len(records), 'valid': sum(r['valid'] for r in records),
            'invalid': sum(not r['valid'] for r in records),
            'by_split': {key: _stratum(splits.get(key, [])) for key in ('development', 'confirmation')},
            'frontier': {key: _frontier(splits.get(key, [])) for key in ('development', 'confirmation')},
            'matched_comparisons': _matched(records),
            'matched_seed_unknown': sum(r.get('seed') is None for r in records),
            'notes': [
                'Development and confirmation are separate; observed confirmation becomes regression evidence for later changes.',
                'All-attempt success includes invalid and unknown-outcome attempts in its denominator; conditional success does not.',
                'Success is truth-audited sustained task completion and retention; an appropriate reaction halt is not task success.',
                'Null completion times remain censored; they never become zero-second completion.',
                'Frontier dots summarize verified successes only, with their counts, failures and censoring printed alongside.',
                'Matching uses split, case, seed and speed setting; duplicate/missing variants cannot produce a unique comparison.',
                'Wilson intervals are descriptive; repeated authored fixtures are correlated and do not establish physical generalization.',
                'Nearest-rank p95 from a small sample is not a reliable tail bound.',
                'Simulated command delays and host controller call latency are separate clocks.',
                'No composite score, RGB-to-skill claim, whole-arm qualification or physical readiness is computed.',
            ]}


def _pretty(value):
    return escape(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _format_number(value):
    return 'unmeasured' if value is None else f'{value:.5g}'


def _format_rate(rate):
    if rate['estimate'] is None:
        return 'unmeasured (0/0)'
    interval = rate['ci95_wilson']
    return (f'{100 * rate["estimate"]:.1f}% ({rate["numerator"]}/{rate["denominator"]}); '
            f'95% Wilson {100 * interval["lower"]:.1f}–{100 * interval["upper"]:.1f}%')


_COLORS = {'slow_feedback': '#36576d', 'fast_fixed': '#a6663a', 'adaptive_predictive': '#42755b'}


def _frontier_svg(rows, split):
    points = [row for row in rows if row['completion_time_s'] is not None and row['error_at_completion_m'] is not None]
    if not points:
        return '<p>No verified-success speed–precision points are available for this split.</p>'
    x_max = max(.01, max(row['completion_time_s'] for row in points) * 1.1)
    y_max = max(.001, max(row['error_at_completion_m'] for row in points) * 1.1)
    markers = []
    for row in points:
        x, y = 80 + row['completion_time_s'] / x_max * 650, 300 - row['error_at_completion_m'] / y_max * 240
        title = escape(f'{row["variant"]}; speed limit {row["speed_limit_m_s"]:.3g} m/s; '
                       f'completion {row["completion_time_s"]:.4g} s; native error at claim {row["error_at_completion_m"]:.4g} m; '
                       f'N={row["plotted_verified_success_count"]}/{row["summary"]["attempted"]}')
        markers.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="6" fill="{_COLORS.get(row["variant"], "#555")}"><title>{title}</title></circle>')
        markers.append(f'<text x="{x+8:.3f}" y="{y-7:.3f}" font-size="11">{row["speed_limit_m_s"]:.3g}</text>')
    ticks = []
    for fraction in (0., .25, .5, .75, 1.):
        x, y = 80 + fraction * 650, 300 - fraction * 240
        ticks.append(f'<text x="{x:.1f}" y="322" text-anchor="middle">{fraction*x_max:.3g}</text>')
        ticks.append(f'<text x="70" y="{y+4:.1f}" text-anchor="end">{fraction*y_max*1000:.3g}</text>')
    return ('<svg viewBox="0 0 800 370" role="img" aria-label="' + escape(split) + ' speed and precision settings">'
            '<title>Completion time against native error at the observed claim; verified successes only</title>'
            '<rect width="800" height="370" fill="#faf9f5"/>'
            '<path d="M80 50V300H750" fill="none" stroke="#777"/>'
            '<g fill="#333" font-family="system-ui,sans-serif" font-size="12">' + ''.join(ticks) +
            '<text x="420" y="355" text-anchor="middle">Median verified completion time / simulated seconds</text>'
            '<text transform="translate(19 180) rotate(-90)" text-anchor="middle">Median native error at claim / mm</text>' +
            ''.join(markers) + '</g></svg>')


def render_report(report):
    """Render an offline escaped paper-style report; SVG contains no scripts."""
    records = report.get('records', [])
    summary = report.get('summary') or summarize(records)
    sections = []
    for split in ('development', 'confirmation'):
        s = summary['by_split'][split]
        rows = summary['frontier'][split]
        table_rows = []
        diagnostic_rows = []
        for row in rows:
            setting = row['summary']
            cells = (row['variant'], _format_number(row['speed_limit_m_s']), setting['attempted'],
                     setting['invalid'], _format_rate(setting['success']['all_attempts']),
                     _format_rate(setting['success']['conditional_valid']),
                     setting['success']['failed_known'], setting['success']['unknown_valid'],
                     setting['completion']['expected_censored'], setting['completion']['false_claims'],
                     setting['completion']['revoked'], setting['completion']['rest_lost_after_confirmation'],
                     row['plotted_verified_success_count'], _format_number(row['completion_time_s']),
                     _format_number(row['error_at_completion_m']))
            table_rows.append('<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in cells) + '</tr>')
            measurements = setting['measurements']
            cells = (row['variant'], _format_number(row['speed_limit_m_s']),
                     _format_number(measurements['final_error_m']['median']),
                     _format_number(measurements['overshoot_m']['worst']),
                     _format_number(measurements['peak_acceleration_m_s2']['worst']),
                     _format_number(measurements['peak_jerk_m_s3']['worst']),
                     _format_number(measurements['peak_force_proxy_n']['worst']),
                     setting['load']['dropped'], setting['load']['outcome_unknown'],
                     _format_number(measurements['controller_latency_p95_s']['median']),
                     _format_number(setting['throughput']['simulated_seconds_per_wall_second']))
            diagnostic_rows.append('<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in cells) + '</tr>')
        sections.append(f'<section><h2>{escape(split.title())}</h2><p>Attempts: {s["attempted"]}; '
                        f'valid: {s["valid"]}; invalid: {s["invalid"]}; completion claims with unknown truth: '
                        f'{s["completion"]["claim_truth_unknown"]}; revoked: {s["completion"]["revoked"]}; '
                        f'rest lost after confirmation: {s["completion"]["rest_lost_after_confirmation"]}. '
                        f'Expected-completion attempts: {s["expected_completion"]["attempted_positive"]}; '
                        f'completion-not-expected attempts: {s["expected_completion"]["attempted_negative"]}; '
                        f'expectation unknown: {s["expected_completion"]["attempted_unknown"]}.</p>' + _frontier_svg(rows, split) +
                        '<p class="legend"><span style="color:#36576d">● Slow feedback</span> · '
                        '<span style="color:#a6663a">● Fast fixed</span> · '
                        '<span style="color:#42755b">● Adaptive predictive</span>. Dot labels: speed limit / m/s.</p>'
                        '<p>Dots include verified successes only. Use the complete attempt counts below to interpret missing or selective results.</p>'
                        '<div class="table"><table><thead><tr><th>Controller</th><th>Speed limit / m/s</th>'
                        '<th>Attempts</th><th>Invalid</th><th>Success / all attempts</th><th>Success / known valid</th>'
                        '<th>Known failures</th><th>Unknown outcome</th>'
                        '<th>Censored expected completions</th><th>False claims</th><th>Revoked</th><th>Rest lost</th><th>Plot N</th>'
                        '<th>Median completion / s</th><th>Median native error at claim / m</th></tr></thead><tbody>' + ''.join(table_rows) +
                        '</tbody></table></div><h3>Motion, contact and computation</h3><div class="table"><table><thead><tr>'
                        '<th>Controller</th><th>Speed limit / m/s</th><th>Median final error / m</th><th>Worst overshoot / m</th>'
                        '<th>Sampled peak acceleration / m/s²</th><th>Sampled peak jerk / m/s³</th><th>Peak force proxy / N</th>'
                        '<th>Loads dropped</th><th>Load outcome unknown</th><th>Median trial p95 controller call / wall s</th>'
                        '<th>Simulation / wall seconds</th></tr></thead><tbody>' + ''.join(diagnostic_rows) +
                        '</tbody></table></div><details><summary>Complete split metrics, missingness and runtime</summary><pre>' +
                        _pretty(s) + '</pre></details><details><summary>Every controller and speed setting</summary><pre>' +
                        _pretty(rows) + '</pre></details></section>')
    receipts = []
    for index, record in enumerate(records):
        status = 'invalid' if not record.get('valid') else 'false completion' if record.get('completion_false') else 'success' if record.get('success') else 'incomplete/failed/unknown'
        receipt = {key: value for key, value in record.items() if key != 'trace'}
        trace = record.get('trace') or []
        receipt['trace_excerpt'] = {'count': len(trace), 'first': trace[0] if trace else None,
                                    'last': trace[-1] if trace else None,
                                    'scope': 'presentation excerpt; complete trace is in report.json and trials.jsonl'}
        receipts.append('<details class="receipt"><summary>' + escape(
            f'{index+1}. {record.get("split")} · {record.get("case_id")} · {record.get("variant")} · '
            f'{record.get("speed_limit_m_s")} m/s · {status}') + '</summary><pre>' + _pretty(receipt) + '</pre></details>')
    metadata = {key: value for key, value in report.items() if key not in ('records', 'summary')}
    notes = ''.join(f'<li>{escape(str(note))}</li>' for note in summary.get('notes', []))
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'">'
            '<title>DVIDIA motor memory v0.2 — speed and precision evidence</title><style>'
            'body{margin:0;background:#faf9f5;color:#252521;font:17px/1.6 Georgia,serif}'
            'main{max-width:1100px;margin:auto;padding:40px 24px}h1{font-size:2rem;line-height:1.2}'
            'h2{margin-top:2.4rem;font-size:1.4rem}p,li{max-width:84ch}.scope{border-left:3px solid #817754;padding-left:16px}'
            '.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font:14px/1.5 system-ui,sans-serif}'
            'th,td{text-align:left;vertical-align:top;border-bottom:1px solid #ddd;padding:10px 8px}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 monospace}'
            'details{margin:12px 0;border:1px solid #ddd;padding:12px}summary{cursor:pointer}'
            'svg{display:block;width:100%;max-width:800px;height:auto}.legend{font:14px/1.5 system-ui,sans-serif}'
            '@media(max-width:600px){main{padding:24px 16px}body{font-size:16px}}'
            '</style></head><body><main><p>DVIDIA · robotics research</p><h1>Motor memory v0.2</h1>'
            '<p class="scope">A fitted actuator model with an authored motion profile on a one-axis native mechanical '
            'coupon, using exact commanded goals and delayed/noisy robot state. Scope: simulation research.</p>'
            f'<p>Every attempt: <strong>{summary["attempted"]}</strong>; valid: {summary["valid"]}; '
            f'invalid: <strong>{summary["invalid"]}</strong>.</p>' + ''.join(sections) +
            '<h2>Matched controller comparisons</h2><p>Pairs require the same split, case, seed and speed setting. '
            'Invalid, duplicate and absent variants stay visible.</p><details><summary>Complete matched comparisons</summary><pre>' +
            _pretty(summary['matched_comparisons']) + '</pre></details><h2>Every attempted trial</h2>'
            '<p>Each receipt retains scalar outcomes, errors and completion history, with the trace count and first/last rows. '
            'Complete traces remain in <a href="report.json">report.json</a> and <a href="trials.jsonl">trials.jsonl</a>.</p>' + ''.join(receipts) +
            '<h2>Protocol, memory and machine identities</h2><pre>' + _pretty(metadata) + '</pre>'
            '<h2>Interpretation limits</h2><ul>' + notes + '</ul></main></body></html>')

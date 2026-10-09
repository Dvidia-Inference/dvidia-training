"""Standard-library measurements and readable evidence for reaction fixtures.

Rates describe the declared synthetic cases, not physical safety or readiness.
Unmeasured/censored values remain null and have explicit sample denominators.
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
        raise ValueError(f'{name} must be a finite number or null')
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f'{name} must be finite') from exc
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f'{name} must be finite' + (' and nonnegative' if nonnegative else ''))
    return value


def _rate(numerator, denominator):
    """95% Wilson score interval; null is retained for an empty denominator."""
    if denominator == 0:
        return {'estimate': None, 'numerator': numerator, 'denominator': 0,
                'ci95_wilson': None}
    z = 1.959963984540054
    p = numerator / denominator
    scale = 1 + z * z / denominator
    center = (p + z * z / (2 * denominator)) / scale
    radius = z * math.sqrt(p * (1 - p) / denominator + z * z / (4 * denominator ** 2)) / scale
    return {'estimate': p, 'numerator': numerator, 'denominator': denominator,
            'ci95_wilson': {'lower': max(0., center - radius),
                            'upper': min(1., center + radius)}}


def _distribution(values, attempted, *, worst='max'):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {'count': 0, 'missing': attempted, 'mean': None, 'median': None,
                'p95': None, 'min': None, 'max': None, 'worst': None}
    # Nearest-rank p95 is explicit; a small N must not imply a smooth tail estimate.
    return {'count': len(values), 'missing': attempted - len(values),
            'mean': mean(values), 'median': median(values),
            'p95': values[max(0, math.ceil(.95 * len(values)) - 1)],
            'min': values[0], 'max': values[-1],
            'worst': values[0] if worst == 'min' else values[-1]}


def _difference(record, end, start):
    a, b = record.get(end), record.get(start)
    if a is None or b is None:
        return None
    difference = a - b
    return 0. if abs(difference) < 1e-12 else difference


_TIMING_FIELDS = {
    'detection_latency_s': ('detected_s', 'hazard_onset_s'),
    'onset_to_sample_s': ('sensor_sample_s', 'hazard_onset_s'),
    'sample_to_delivery_s': ('sensor_delivered_s', 'sensor_sample_s'),
    'delivery_to_decision_s': ('decision_s', 'sensor_delivered_s'),
    'decision_to_brake_request_s': ('brake_requested_s', 'decision_s'),
    'detection_to_brake_request_s': ('brake_requested_s', 'detected_s'),
    'request_to_application_s': ('brake_applied_s', 'brake_requested_s'),
    'application_to_stop_s': ('stopped_s', 'brake_applied_s'),
    'hazard_to_stop_s': ('stopped_s', 'hazard_onset_s'),
}
_MEASUREMENT_FIELDS = (
    'stopping_distance_m', 'stopping_displacement_m', 'peak_unexpected_force_n',
    'unexpected_contact_impulse_ns', 'minimum_clearance_m', 'max_penetration_m',
    'request_to_rest_path_m', 'max_forward_excursion_m', 'rest_dwell_s',
    'observation_blackout_seconds', 'observation_stale_seconds', 'observation_unusable_seconds',
    'monitor_update_wall_seconds', 'monitor_call_latency_p50_s', 'monitor_call_latency_p95_s', 'monitor_call_latency_max_s',
)
_NONNEGATIVE = {
    'detected_s', 'hazard_onset_s', 'brake_requested_s', 'brake_applied_s',
    'stopped_s', 'decision_s',
    'stopping_distance_m', 'stopping_displacement_m', 'peak_unexpected_force_n',
    'unexpected_contact_impulse_ns', 'max_penetration_m', 'rest_dwell_s',
    'request_to_rest_path_m', 'max_forward_excursion_m', 'observation_blackout_seconds',
    'observation_stale_seconds', 'observation_unusable_seconds',
    'monitor_update_wall_seconds', 'monitor_call_latency_p50_s', 'monitor_call_latency_p95_s', 'monitor_call_latency_max_s',
    'simulated_seconds', 'wall_seconds', 'step_count',
}


def _checked(records):
    checked = []
    for index, source in enumerate(records):
        if not isinstance(source, dict):
            raise ValueError(f'record {index} must be an object')
        record = dict(source)
        for name in ('case_id', 'variant'):
            if not isinstance(record.get(name), str) or not record[name]:
                raise ValueError(f'record {index}.{name} must be a nonempty string')
        if type(record.get('valid')) is not bool:
            raise ValueError(f'record {index}.valid must be boolean')
        for name in ('expected_detection', 'dropped', 'observation_available', 'rest_lost_after_confirmation', 'contact_diagnostic_pass', 'required_observations_fresh'):
            if record.get(name) is not None and type(record[name]) is not bool:
                raise ValueError(f'record {index}.{name} must be boolean or null')
        group = record.get('group', 'ungrouped')
        if not isinstance(group, str) or not group:
            raise ValueError(f'record {index}.group must be a nonempty string')
        record['group'] = group
        for name in _NONNEGATIVE | {'minimum_clearance_m', 'sensor_sample_s', 'sensor_delivered_s'}:
            record[name] = _number(record.get(name), f'record {index}.{name}',
                                   nonnegative=name in _NONNEGATIVE)
        if record['step_count'] is not None and not record['step_count'].is_integer():
            raise ValueError(f'record {index}.step_count must be an integer')
        chain = ('sensor_sample_s', 'sensor_delivered_s', 'decision_s',
                 'brake_requested_s', 'brake_applied_s', 'stopped_s')
        present = [(name, record[name]) for name in chain if record.get(name) is not None]
        for (earlier, a), (later, b) in zip(present, present[1:]):
            if a > b + 1e-12:
                raise ValueError(f'record {index}: {later} precedes {earlier}')
        if (record.get('detected_s') is not None and record.get('brake_requested_s') is not None
                and record['detected_s'] > record['brake_requested_s'] + 1e-12):
            raise ValueError(f'record {index}: brake_requested_s precedes detected_s')
        checked.append(record)
    return checked


def _stratum(records):
    valid = [r for r in records if r['valid']]
    labeled = [r for r in valid if r.get('expected_detection') is not None]
    confusion = dict(tp=0, fp=0, tn=0, fn=0)
    for record in labeled:
        expected, detected = record['expected_detection'], record.get('detected_s') is not None
        confusion['tp' if expected and detected else 'fn' if expected else 'fp' if detected else 'tn'] += 1
    tp, fp, tn, fn = (confusion[key] for key in ('tp', 'fp', 'tn', 'fn'))
    attempted_positive = sum(r.get('expected_detection') is True for r in records)
    attempted_negative = sum(r.get('expected_detection') is False for r in records)
    invalid = [r for r in records if not r['valid']]
    confusion.update(expected_positive=tp + fn, expected_negative=tn + fp,
                     attempted_positive=attempted_positive, attempted_negative=attempted_negative,
                     attempted_unknown_label=len(records) - attempted_positive - attempted_negative,
                     invalid_positive=sum(r.get('expected_detection') is True for r in invalid),
                     invalid_negative=sum(r.get('expected_detection') is False for r in invalid),
                     invalid_unknown_label=sum(r.get('expected_detection') is None for r in invalid),
                     evaluated=len(labeled), unknown_label=len(valid) - len(labeled),
                     invalid_excluded=len(records) - len(valid),
                     valid_correct_per_attempted_labeled=_rate(tp + tn, attempted_positive + attempted_negative),
                     accuracy=_rate(tp + tn, len(labeled)),
                     precision=_rate(tp, tp + fp), recall=_rate(tp, tp + fn),
                     false_positive_rate=_rate(fp, fp + tn))
    metrics = {}
    for name, (end, start) in _TIMING_FIELDS.items():
        eligible = [r for r in valid if r.get('expected_detection') is True] if name in (
            'detection_latency_s', 'onset_to_sample_s', 'hazard_to_stop_s') else valid
        metrics[name] = _distribution([_difference(r, end, start) for r in eligible], len(eligible))
    for name in _MEASUREMENT_FIELDS:
        metrics[name] = _distribution([r.get(name) for r in valid], len(valid),
                                      worst='min' if name == 'minimum_clearance_m' else 'max')
    brakes = [r for r in valid if r.get('brake_requested_s') is not None]
    stopped = sum(r.get('stopped_s') is not None for r in brakes)
    rest_checks = [r for r in valid if r.get('rest_lost_after_confirmation') is not None]
    drops = [r for r in valid if r.get('dropped') is not None]
    diagnostics = [r for r in valid if r.get('contact_diagnostic_pass') is not None]
    negative_seconds = sum(r.get('simulated_seconds') or 0. for r in labeled
                           if r['expected_detection'] is False)
    # Throughput includes failed/invalid attempts: the work still consumed time.
    wall = sum(r.get('wall_seconds') or 0. for r in records)
    simulated = sum(r.get('simulated_seconds') or 0. for r in records)
    steps = sum(r.get('step_count') or 0. for r in records)
    return {'attempted': len(records), 'valid': len(valid), 'invalid': len(records) - len(valid),
            'error_records': sum(r.get('error') is not None for r in records),
            'detection': confusion, 'measurements': metrics,
            'stop': {'brake_requested': len(brakes), 'stopped': stopped,
                     'unstopped_at_horizon': len(brakes) - stopped,
                     'rest_lost_after_confirmation': sum(r['rest_lost_after_confirmation'] for r in rest_checks),
                     'rest_loss_status_known': len(rest_checks),
                     'rest_loss_status_unknown': len(valid) - len(rest_checks),
                     'stopped_fraction': _rate(stopped, len(brakes))},
            'load': {'known': len(drops), 'unknown': len(valid) - len(drops),
                     'dropped': sum(r['dropped'] for r in drops),
                     'drop_rate': _rate(sum(r['dropped'] for r in drops), len(drops))},
            'observations': {'unavailable': sum(r.get('observation_available') is False for r in valid),
                             'availability_unknown': sum(r.get('observation_available') is None for r in valid),
                             'required_freshness_failed': sum(r.get('required_observations_fresh') is False for r in valid),
                             'required_freshness_unknown': sum(r.get('required_observations_fresh') is None for r in valid)},
            'contact_diagnostics': {'assessed': len(diagnostics),
                                    'exceeded': sum(r['contact_diagnostic_pass'] is False for r in diagnostics),
                                    'unknown': len(valid)-len(diagnostics),
                                    'scope': 'authored penetration/force diagnostics; numerical validity and detection do not establish material accuracy'},
            'false_positive_episodes_per_simulated_hour': fp / negative_seconds * 3600 if negative_seconds else None,
            'negative_exposure_simulated_seconds': negative_seconds,
            'throughput': {'wall_seconds': wall, 'simulated_seconds': simulated, 'step_count': int(steps),
                           'steps_per_wall_second': steps / wall if wall and all(
                               r.get('wall_seconds') is not None and r.get('step_count') is not None for r in records) else None,
                           'simulated_seconds_per_wall_second': simulated / wall if wall and all(
                               r.get('wall_seconds') is not None and r.get('simulated_seconds') is not None for r in records) else None,
                           'wall_time_missing': sum(r.get('wall_seconds') is None for r in records),
                           'simulated_time_missing': sum(r.get('simulated_seconds') is None for r in records),
                           'step_count_missing': sum(r.get('step_count') is None for r in records)}}


def summarize(records):
    """Summarize every attempted record, preserving uncertainty and missingness.

    Detection is an episode-level decision against an authored label. A positive
    case remains positive even if timely braking prevents physical contact.
    Timestamp differences are simulated durations, not host response timings.
    """
    records = _checked(records)
    variants, groups, variant_groups = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(list))
    for record in records:
        variants[record['variant']].append(record)
        groups[record['group']].append(record)
        variant_groups[record['variant']][record['group']].append(record)
    overall = _stratum(records)
    return {'attempted': overall['attempted'], 'valid': overall['valid'], 'invalid': overall['invalid'],
            'overall': overall,
            'by_variant': {k: _stratum(v) for k, v in sorted(variants.items())},
            'by_group': {k: _stratum(v) for k, v in sorted(groups.items())},
            'by_variant_group': {k: {g: _stratum(rs) for g, rs in sorted(gs.items())}
                                 for k, gs in sorted(variant_groups.items())},
            'notes': ['Episode labels are authored independently of the measured trajectory.',
                      'Invalid attempts and unknown labels remain in counts; they do not enter accuracy denominators.',
                      'Valid-correct per attempted labeled cases includes invalid labeled attempts in its denominator; inspect it alongside conditional accuracy.',
                      'Wilson intervals describe uncertainty within these synthetic fixtures; repeated seeds are not independent physical evidence.',
                      'p95 uses nearest rank; small samples do not establish a tail bound.',
                      'Null stopping times are censored/unstopped attempts, not zero-duration stops.',
                      'Wall throughput includes all attempted episodes with recorded time, including invalid attempts.',
                      'The exposure rate counts false-positive negative episodes, at most one decision per episode; it is not a continuous-stream alarm frequency.',
                      'Negative detection latency means detection preceded the authored onset; interpret it using the protocol.',
                      'Sensor sample/delivery timestamps can be negative in the rollout-relative clock when primed before rollout.',
                      'No composite safety or physical-readiness score is computed.']}


def _pretty(value):
    return escape(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _format_number(value):
    return 'unmeasured' if value is None else f'{value:.5g}'


def _format_rate(value):
    if value['estimate'] is None:
        return f'unmeasured (0/{value["denominator"]})'
    interval = value['ci95_wilson']
    return (f'{100 * value["estimate"]:.1f}% ({value["numerator"]}/{value["denominator"]}); '
            f'95% Wilson {100 * interval["lower"]:.1f}–{100 * interval["upper"]:.1f}%')


def render_report(report):
    """Render a self-contained, escaped HTML report; no network or scripts."""
    summary = report.get('summary') or summarize(report.get('records', []))
    variants = summary['by_variant']
    counts = ''.join('<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in (
        name, s['attempted'], s['valid'], s['invalid'], s['detection']['unknown_label'],
        s['detection']['attempted_positive'], s['detection']['attempted_negative'],
        s['detection']['tp'], s['detection']['fp'], s['detection']['tn'], s['detection']['fn'])) + '</tr>'
        for name, s in variants.items())
    rates = ''.join('<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in (
        name, *(_format_rate(s['detection'][key]) for key in (
            'accuracy', 'precision', 'recall', 'false_positive_rate', 'valid_correct_per_attempted_labeled')))) + '</tr>'
        for name, s in variants.items())
    measurements = []
    for name, s in variants.items():
        rows = ''.join('<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in (
            metric, d['count'], d['missing'], _format_number(d['median']),
            _format_number(d['p95']), _format_number(d['worst']))) + '</tr>'
            for metric, d in s['measurements'].items())
        measurements.append(f'<h3>{escape(name)}</h3><p>Brake requests: {s["stop"]["brake_requested"]}; '
                            f'unstopped: {s["stop"]["unstopped_at_horizon"]}. '
                            f'Rest lost after confirmation: {s["stop"]["rest_lost_after_confirmation"]}. '
                            f'Load outcome unmeasured/not applicable: {s["load"]["unknown"]}; dropped: {s["load"]["dropped"]}. '
                            f'Contact diagnostics exceeded: {s["contact_diagnostics"]["exceeded"]}/{s["contact_diagnostics"]["assessed"]}.</p>'
                            '<div class="table"><table><thead><tr><th>Measurement / unit</th><th>N</th><th>Missing</th>'
                            '<th>Median</th><th>p95</th><th>Worst</th></tr></thead><tbody>' + rows + '</tbody></table></div>'
                            '<details><summary>Throughput and complete variant summary</summary><pre>' + _pretty(s) + '</pre></details>')
    attempts = []
    for index, r in enumerate(report.get('records', [])):
        status = 'invalid' if not r.get('valid') else 'unstopped' if r.get('brake_requested_s') is not None and r.get('stopped_s') is None else 'diagnostic' if r.get('contact_diagnostic_pass') is False else 'valid'
        label = 'positive' if r.get('expected_detection') is True else 'negative' if r.get('expected_detection') is False else 'unknown'
        attempts.append('<details class="attempt ' + status + '"><summary>' + escape(
            f'{index + 1}. {r.get("case_id", "unknown")} · {r.get("variant", "unknown")} · {status} · authored {label}') +
            '</summary><pre>' + _pretty(r) + '</pre></details>')
    notes = ''.join(f'<li>{escape(note)}</li>' for note in summary.get('notes', []))
    metadata = {key: report.get(key) for key in (
        'recorded_utc', 'framework_version', 'protocol', 'scope', 'versions', 'host', 'fixture_id', 'protocol_sha256',
        'source_sha256', 'timing_scope', 'total_trial_wall_seconds', 'convergence_wall_seconds', 'end_to_end_work_wall_seconds',
        'benchmark_qualification', 'physical_robot_ready')}
    paired = report.get('paired_comparisons', [])
    paired_rows = ''.join('<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in (
        pair.get('case_id'), pair.get('seed'), pair.get('both_valid'),
        _format_number(pair.get('peak_unexpected_force_n_monitor_minus_control')),
        _format_number(pair.get('unexpected_contact_impulse_ns_monitor_minus_control')),
        _format_number(pair.get('minimum_clearance_m_monitor_minus_control')),
        _format_number(pair.get('max_penetration_m_monitor_minus_control')),
        pair.get('monitor_dropped'), pair.get('control_dropped'))) + '</tr>' for pair in paired)
    paired_html = ('<h2>Matched intervention comparisons</h2><p>Deltas are monitor minus disabled control. '
                   'Lower force, impulse and penetration and greater clearance are favorable within the fixture. '
                   'Unmeasured or invalid pairs remain visible.</p><div class="table"><table><thead><tr>'
                   '<th>Case</th><th>Seed</th><th>Both valid</th><th>Δ peak force / N</th>'
                   '<th>Δ impulse / N·s</th><th>Δ clearance / m</th><th>Δ penetration / m</th>'
                   '<th>Monitor dropped</th><th>Control dropped</th></tr></thead><tbody>' + paired_rows +
                   '</tbody></table></div>') if paired else ''
    limitations = ''.join(f'<li>{escape(str(note))}</li>' for note in report.get('limitations', []))
    comparison = report.get('timestep_comparison')
    convergence_html = ''
    if comparison:
        convergence_rows = ''.join('<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in (
            pair['case_id'], pair['variant'], pair['both_valid'], pair['timing_stable'], pair['contact_converged'],
            _format_number(pair['metrics']['peak_unexpected_force_n']['relative_change_to_abs_base']),
            _format_number(pair['metrics']['unexpected_contact_impulse_ns']['relative_change_to_abs_base']),
            _format_number(pair['metrics']['max_penetration_m']['absolute_change']))) + '</tr>' for pair in comparison['pairs'])
        convergence_html = (f'<h2>Physics timestep comparison</h2><p>Additional attempts: {comparison["attempted"]}; '
            f'invalid: {comparison["invalid"]}. Timing within declared tolerance: {comparison["timing_stable"]}; '
            f'contact within declared tolerance: {comparison["contact_converged"]}. '
            'Only physics ticks change from 1 ms to 0.5 ms. Sensor and reaction clocks stay fixed. '
            'These are numerical diagnostics, with no physical calibration.</p><div class="table"><table><thead><tr>'
            '<th>Case</th><th>Variant</th><th>Both valid</th><th>Timing agreement</th><th>Contact agreement</th>'
            '<th>Relative peak-force change</th><th>Relative impulse change</th><th>Penetration change / m</th>'
            '</tr></thead><tbody>' + convergence_rows + '</tbody></table></div><details><summary>Complete timestep evidence</summary><pre>' + _pretty(comparison) + '</pre></details>')
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'">'
            '<title>DVIDIA reaction framework v0.1 — measured fixtures</title><style>'
            'body{margin:0;background:#faf9f5;color:#252521;font:17px/1.6 Georgia,serif}'
            'main{max-width:1050px;margin:auto;padding:40px 24px}h1{font-size:2rem;line-height:1.2}'
            'h2{margin-top:2.3rem;font-size:1.4rem}h3{font-size:1.1rem}p,li{max-width:82ch}'
            '.scope{border-left:3px solid #817754;padding-left:16px}.table{overflow-x:auto}'
            'table{border-collapse:collapse;width:100%;font:14px/1.45 system-ui,sans-serif}'
            'th,td{text-align:left;vertical-align:top;border-bottom:1px solid #ddd;padding:10px 8px}'
            'th{font-weight:650}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 monospace}'
            'details{margin:12px 0;border:1px solid #ddd;padding:12px}summary{cursor:pointer}'
            '.invalid,.unstopped,.diagnostic{border-left:4px solid #985c36}a{color:#344e40}'
            '@media(max-width:600px){main{padding:24px 16px}body{font-size:16px}}'
            '</style></head><body><main><p>DVIDIA · robotics research</p>'
            '<h1>Reaction framework v0.1</h1><p class="scope">Synthetic sensed-state fixtures: a native linear-actuator '
            'and parallel-jaw coupon in simulation, with a reusable reaction monitor. Whole-arm geometry protection is a future adapter. '
            'These measurements do not establish RGB perception accuracy, a learned skill, physical sensor accuracy, '
            'hardware stopping performance or physical safety.</p>'
            f'<p>All attempts: <strong>{summary["attempted"]}</strong> · numerically valid: {summary["valid"]} · '
            f'invalid: <strong>{summary["invalid"]}</strong>. Contact diagnostics exceeded: '
            f'<strong>{summary["overall"]["contact_diagnostics"]["exceeded"]}</strong>.</p>'
            '<details><summary>Recorded protocol, machine and source identities</summary><pre>' + _pretty(metadata) + '</pre></details>'
            '<h2>Detection decisions</h2><p>Positive/negative labels are fixed by the fixture. Braking that prevents contact '
            'does not turn a positive case into a negative. Invalid attempts and unknown labels are visible below.</p>'
            '<div class="table"><table><thead><tr><th>Variant</th><th>Attempts</th><th>Numerically valid</th><th>Invalid</th>'
            '<th>Unknown valid label</th><th>Attempted positive</th><th>Attempted negative</th>'
            '<th>TP</th><th>FP</th><th>TN</th><th>FN</th></tr></thead><tbody>' + counts + '</tbody></table></div>'
            '<div class="table"><table><thead><tr><th>Variant</th><th>Accuracy</th><th>Precision</th><th>Recall</th>'
            '<th>False-positive rate</th><th>Valid-correct / all labeled attempts</th></tr></thead><tbody>' + rates + '</tbody></table></div>'
            '<h2>Timing, stopping and contact</h2><p>Seconds in event differences are simulation time. Host wall time '
            'is reported separately. Missing stops remain censored; minimum clearance is worse when smaller.</p>' + ''.join(measurements) +
            paired_html + convergence_html + '<h2>Scenario strata</h2><details><summary>Every variant and group, including failed attempts</summary><pre>' +
            _pretty(summary['by_variant_group']) + '</pre></details><h2>Every attempted episode</h2>' + ''.join(attempts) +
            '<h2>Interpretation limits</h2><ul>' + notes + limitations + '</ul></main></body></html>')

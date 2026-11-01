#!/usr/bin/env python3
"""Render the bounded PiPER retention v0.2 JSON report as static HTML."""
import argparse
import html
import json
import math
from pathlib import Path

SCHEMA = 'dvidia.piper-retention-study/v1'
VARIANTS = ('measured_joint', 'issued_joint', 'cartesian')
MAX_BYTES = 16 * 1024 * 1024


def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate report JSON field.')
        result[key] = value
    return result


def constant(_value):
    raise ValueError('Nonfinite report JSON value.')


def finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Nonfinite report JSON number.')
    return result


def load(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError('Expected a bounded regular report.json.')
    raw = path.read_bytes()
    if len(raw) > MAX_BYTES:
        raise ValueError('Report grew beyond its byte limit.')
    report = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant, parse_float=finite_float)
    rows = report.get('rows') if type(report) is dict else None
    if (type(report) is not dict or report.get('schema') != SCHEMA
            or report.get('planned_attempts') != 36 or report.get('attempted') != 36
            or type(rows) is not list or len(rows) != 36
            or any(type(row) is not dict for row in rows)):
        raise ValueError('Expected all 36 completed scheduled attempt records.')
    seen = set()
    for row in rows:
        variant, scene = row.get('variant'), row.get('scene_id')
        if variant not in VARIANTS or type(scene) is not str or (scene, variant) in seen:
            raise ValueError('Report contains an unknown or duplicate scene/variant attempt.')
        seen.add((scene, variant))
        evaluation = row.get('evaluation')
        if (type(evaluation) is not dict or type(evaluation.get('primary')) is not dict
                or type(evaluation.get('stationarity')) is not dict):
            raise ValueError('Attempt lacks the frozen primary/stationarity evaluation fields.')
    scenes = {scene for scene, _ in seen}
    if (len(scenes) != 12 or seen != {(scene, variant) for scene in scenes for variant in VARIANTS}
            or {row['variant'] for row in rows} != set(VARIANTS)):
        raise ValueError('Expected the complete 12-scene by 3-variant roster.')
    return report, rows


def esc(value):
    return html.escape(str(value), quote=True)


def truth(value):
    return 'yes' if value is True else ('no' if value is False else 'not reported')


def fmt_seconds(value):
    return f'{value:.3f} s' if type(value) in (int, float) and math.isfinite(value) else '—'


def row_values(row):
    evaluation = row['evaluation']
    primary = evaluation['primary']
    admission = evaluation.get('first_admission')
    admitted_at = 'none'
    prefix = '—'
    if type(admission) is dict:
        admitted_at = f"{fmt_seconds(admission.get('time_s'))} · probe {admission.get('probe_index', '—')}"
        prefix = truth(admission.get('native_prefix_qualified'))
    complete = primary.get('complete_5s') is True
    native = primary.get('native_pause_success')
    controller = primary.get('controller_retained_at_5s')
    stationarity = evaluation.get('stationarity') or {}
    stationary_success = stationarity.get('grip_retention_and_stationarity')
    false_confirmation = complete and native is False and controller is True
    conservative = complete and native is True and controller is False
    timing = row.get('timing')
    step_wall = timing.get('step_wall') if type(timing) is dict else None
    p95 = step_wall.get('p95_s') if type(step_wall) is dict else None
    reason = primary.get('reason') or row.get('error') or '—'
    if type(reason) is dict:
        reason = reason.get('message') or reason.get('type') or '—'
    values = (row.get('id', '—'), row.get('scene_id', '—'), row.get('variant', '—'),
              truth(row.get('valid')), admitted_at, prefix,
              truth(native) if complete else 'incomplete follow-up',
              truth(stationary_success) if complete else 'incomplete follow-up',
              truth(controller) if complete else 'not evaluated',
              'yes' if false_confirmation else ('no' if complete else '—'),
              'yes' if conservative else ('no' if complete else '—'), fmt_seconds(p95), reason)
    return values, complete, native, controller


def render(report, rows):
    sections = []
    for variant in VARIANTS:
        group = [row for row in rows if row.get('variant') == variant]
        classified = [row_values(row) for row in group]
        complete = sum(item[1] for item in classified)
        native_pass = sum(item[1] and item[2] is True for item in classified)
        stationary_pass = sum(item[1] and row['evaluation']['stationarity'].get('grip_retention_and_stationarity') is True
                              for item, row in zip(classified, group))
        false_confirm = sum(item[1] and item[2] is False and item[3] is True for item in classified)
        abstain = sum(item[1] and item[2] is True and item[3] is False for item in classified)
        false_initial = sum(type(row['evaluation'].get('first_admission')) is dict
                            and row['evaluation']['first_admission'].get('native_prefix_qualified') is False
                            for row in group)
        sections.append(f'<h2>{esc(variant.replace("_", " ").title())}</h2>')
        sections.append(f'<p class="summary">12 scheduled · {sum(row.get("valid") is True for row in group)}/12 valid · {sum(row["evaluation"].get("first_admission") is not None for row in group)} first admissions · {false_initial} false initial admissions · {complete}/12 with complete five-second follow-up · {native_pass} native grip holds · {stationary_pass} grip plus stationary-tool pauses · {false_confirm} false confirmations · {abstain} conservative abstentions</p>')
        sections.append('<div class="table-wrap"><table><thead><tr><th>Attempt</th><th>Scene</th><th>Variant</th><th>Valid run</th><th>First admission</th><th>Native prefix qualified</th><th>Native five-second grip hold</th><th>Grip + stationary tool</th><th>Controller retained</th><th>False confirmation</th><th>Conservative abstention</th><th>Step p95</th><th>Primary reason / error</th></tr></thead><tbody>')
        for row in group:
            values, *_ = row_values(row)
            sections.append('<tr>' + ''.join(f'<td>{esc(value)}</td>' for value in values) + '</tr>')
        sections.append('</tbody></table></div>')
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Checking grip retention · DVIDIA PiPER v0.2</title><style>
:root{{--ink:#242321;--muted:#6d6a64;--line:#dedbd4;--paper:#f7f6f2;--white:#fff;--accent:#d6ff57}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:1320px;margin:auto;padding:42px 22px 70px}}
.eyebrow{{font-size:.75rem;letter-spacing:.16em;text-transform:uppercase;font-weight:700;color:var(--muted)}}h1{{font-size:clamp(2.2rem,5vw,4rem);line-height:1.04;letter-spacing:-.045em}}h2{{font-size:1.2rem;margin:2.2rem 0 .6rem}}.lede{{font-size:1.12rem;max-width:78ch}}.banner{{background:var(--accent);padding:14px 18px;border-radius:12px;font-weight:650;margin:24px 0}}
.summary{{color:var(--muted);font-size:.9rem}}.table-wrap{{overflow-x:auto;background:white;border:1px solid var(--line);border-radius:12px}}table{{width:100%;border-collapse:collapse;font-size:.82rem;min-width:1320px}}th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}}th{{background:#efeee8;position:sticky;top:0;font-size:.72rem;text-transform:uppercase;letter-spacing:.04em}}tr:last-child td{{border:0}}footer{{border-top:1px solid var(--line);margin-top:32px;padding-top:16px;color:var(--muted)}}a{{color:#274d00}}
</style><main><div class="eyebrow">DVIDIA · PiPER retention v0.2</div><h1>Checking grip retention without changing the squeeze.</h1>
<p class="lede">Three authored hold strategies across twelve fresh synthetic scenes. The primary question is whether an admitted grasp remains held continuously for five simulated seconds without a false confirmation. A second reported outcome also requires the TCP to stay within the same five-second pose bounds for every variant.</p>
<div class="banner">Simulation-only outcomes. No learned policy, transport, physical sensor validation or hardware qualification.</div>
<p>{esc(report.get('scope', ''))}</p>{''.join(sections)}
<h2>How to read the outcomes</h2><p>“Native five-second grip hold” is the simulator outcome after first admission. “Grip + stationary tool” additionally requires the TCP to remain within 4 mm and 0.05 rad of its admission pose throughout the five-second window. “Controller retained” is the controller's corresponding qualification claim. A false confirmation is counted only when complete follow-up exists, the native grip hold fails, and the controller claims retention. Conservative abstention is shown separately when the native hold succeeds and the controller does not claim retention. Initial admission against the native prefix is a separate event. Incomplete follow-up is not counted as either mismatch.</p>
<p>Step p95 is unpaced full control-step compute wall time (including dense telemetry); it is not sensor latency or a real-time guarantee. Runs use privileged simulator state and do not establish physical retention.</p>
<footer><a href="report.json">Report JSON</a> · <a href="protocol.json">Protocol</a> · <a href="https://research.dvidia.org/downloads/topics/physical-grounding/runs/2026-10-31-piper-retention-v02/report/audit.json">Independent audit</a> · <a href="inspection.json">Inspection</a> · <a href="https://research.dvidia.org/papers/piper-retention-v02/">Research paper</a> · <a href="https://github.com/Dvidia-Inference/dvidia-training/releases/tag/piper-retention-v0.2-preview">Source release</a> · <a href="https://huggingface.co/datasets/Dvidia/piper-retention-v02">Evidence dataset</a></footer></main></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report, rows = load(args.report)
    output = args.output or args.report.with_name('report.html')
    if output.exists() or output.is_symlink():
        raise ValueError('HTML output path must be new; existing files are not overwritten.')
    output.write_text(render(report, rows), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()

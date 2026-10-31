#!/usr/bin/env python3
"""Render the bounded piper-probe-study/v1 report as a self-contained HTML page."""
import argparse
import html
import json
from pathlib import Path

SCHEMA = 'dvidia.piper-probe-study/v1'
MAX_BYTES = 16 * 1024 * 1024


def load(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError('Expected a bounded regular report.json file.')
    with path.open('r', encoding='utf-8') as stream:
        value = json.load(stream, parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Non-finite JSON number')))
    if type(value) is not dict or value.get('schema') != SCHEMA or type(value.get('rows')) is not list:
        raise ValueError('Unsupported report schema.')
    return value


def esc(value):
    return html.escape(str(value), quote=True)


def ms(summary, name):
    value = (summary or {}).get(name)
    if type(value) is not dict or type(value.get('mean_s')) not in (int, float):
        return '—'
    p95 = value.get('p95_s')
    return f"{value['mean_s']*1000:.2f} / {p95*1000:.2f} ms" if type(p95) in (int, float) else f"{value['mean_s']*1000:.2f} ms"


def attempt(row):
    evaluation = row.get('evaluation') or {}
    receipt = evaluation.get('receipt') or {}
    audit = receipt.get('audit') or {}
    verification = evaluation.get('verification') or {}
    admission = evaluation.get('admission') or {}
    policy = row.get('policy') or {}
    probe_index = ((receipt.get('context') or {}).get('probe_index'))
    if type(probe_index) is not int:
        probes = policy.get('probes') or []
        probe_index = probes[-1].get('probe_index') if probes else None
    events = [e for e in (policy.get('events') or [])
              if type(e) is dict and e.get('probe_index') == probe_index]
    decisions = [e for e in events if e.get('type') in ('evidence_qualified', 'timer_admitted')]
    primary = decisions[-1] if decisions else None
    primary_text = (f"{primary['type']} @ {primary['t_seconds']:.2f}s"
                    if primary and type(primary.get('t_seconds')) in (int, float) else 'no admission')
    terminal = [e for e in events if e.get('type') in ('hold_invalidated', 'abort')]
    terminal_text = (f"{terminal[-1].get('type')}: {terminal[-1].get('reason', '—')}"
                     if terminal else f"no invalidation · {policy.get('status', 'unknown')}")
    false_n = audit.get('controller_false_qualification_count', '—')
    abstain_n = audit.get('controller_conservative_abstention_count', '—')
    disagreements = audit.get('controller_claim_mismatches')
    disagree_n = len(disagreements) if isinstance(disagreements, list) else '—'
    audit_text = f"false qualification {false_n} · conservative abstention {abstain_n} · disagreements {disagree_n}"
    state = 'valid' if row.get('valid') is True else 'invalid'
    native = verification.get('admissible')
    native_text = 'admissible' if native is True else ('not admissible' if native is False else 'not inspected')
    gate = admission.get('status') or (admission.get('planning') or {}).get('status') or 'unknown'
    return '<tr>' + ''.join(f'<td>{esc(x)}</td>' for x in (
        row.get('id', '—'), state, primary_text, terminal_text, native_text,
        audit_text, gate, ms(row.get('timing'), 'policy_cpu'), ms(row.get('timing'), 'step_cpu'))) + '</tr>'


def render(report):
    rows = report['rows']
    variants = ('evidence', 'timer')
    body = []
    for variant in variants:
        group = [r for r in rows if r.get('variant') == variant]
        admitted = sum(any(e.get('type') in ('evidence_qualified', 'timer_admitted')
                           for e in (r.get('policy') or {}).get('events', [])) for r in group)
        false_n = sum((((r.get('evaluation') or {}).get('receipt') or {}).get('audit') or {}).get(
            'controller_false_qualification_count', 0) for r in group)
        abstain_n = sum((((r.get('evaluation') or {}).get('receipt') or {}).get('audit') or {}).get(
            'controller_conservative_abstention_count', 0) for r in group)
        terminal = {}
        for row in group:
            status = (row.get('policy') or {}).get('status', 'unknown')
            terminal[status] = terminal.get(status, 0) + 1
        outcome = ', '.join(f'{esc(k)} {v}' for k, v in sorted(terminal.items())) or 'no attempts'
        body.append(f'<h2>{esc(variant.title())} policy · {len(group)} attempts</h2>')
        body.append(f'<p class="group-summary">Primary admissions {admitted}/{len(group)} · audited false qualifications {false_n} · conservative abstentions {abstain_n} · terminal policy statuses: {outcome}.</p>')
        body.append('<div class="table-wrap"><table><thead><tr><th>Attempt / scene</th><th>Native run</th><th>Primary admission event</th><th>Terminal event / state</th><th>Final 1 ms evidence</th><th>Audited claims: false qual · abstention · disagreement</th><th>Final planning gate</th><th>Policy CPU mean / p95</th><th>Whole-step CPU mean / p95</th></tr></thead><tbody>')
        body.extend(attempt(r) for r in group)
        body.append('</tbody></table></div>')
    valid = report.get('valid_attempts', sum(r.get('valid') is True for r in rows))
    return f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PiPER Probe Evidence v0.1 · DVIDIA Research</title>
<style>
:root{{color-scheme:light;--ink:#242321;--muted:#6d6a64;--line:#dedbd4;--paper:#f7f6f2;--white:#fff;--accent:#d6ff57}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1180px;margin:auto;padding:42px 22px 70px}}.eyebrow{{font-size:.75rem;letter-spacing:.16em;text-transform:uppercase;font-weight:700;color:var(--muted)}}
h1{{font-size:clamp(2.2rem,5vw,4rem);line-height:1.04;letter-spacing:-.045em;margin:.55rem 0 1rem}}h2{{font-size:1.2rem;margin:2.2rem 0 .7rem}}p{{max-width:78ch}}.lede{{font-size:1.12rem;max-width:78ch}}
.banner{{background:var(--accent);padding:14px 18px;border-radius:12px;font-weight:650;margin:24px 0}}.stats{{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}}.stat{{background:var(--white);border:1px solid var(--line);border-radius:12px;padding:14px 18px;min-width:150px}}.stat b{{display:block;font-size:1.45rem}}
.table-wrap{{overflow-x:auto;background:white;border:1px solid var(--line);border-radius:12px}}table{{width:100%;border-collapse:collapse;font-size:.86rem;min-width:1420px}}th,td{{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top}}th{{background:#efeee8;position:sticky;top:0;font-size:.75rem;letter-spacing:.04em;text-transform:uppercase}}tr:last-child td{{border-bottom:0}}.group-summary{{color:var(--muted);font-size:.9rem}}code{{background:#ebe9e2;padding:.12em .35em;border-radius:4px}}a{{color:#274d00}}footer{{border-top:1px solid var(--line);margin-top:36px;padding-top:18px;color:var(--muted)}}
</style><main><div class="eyebrow">DVIDIA · Research preview · PiPER</div>
<h1>Probe, verify, hold.</h1><p class="lede">A bounded native simulation study compares an evidence-based short-lift gate with a fixed timer gate. Both stop in a measured-position hold; neither performs full placement or transport.</p>
<div class="banner">Simulation-only evidence. No hardware qualification, learned policy, arbitrary-video transfer, or full-placement result.</div>
<div class="stats"><div class="stat"><b>{len(rows)}</b>attempt records</div><div class="stat"><b>{valid}/{len(rows)}</b>valid native runs</div><div class="stat"><b>{esc(report.get('planned_attempts', '—'))}</b>planned attempts</div></div>
<p>{esc(report.get('scope', ''))}</p>
{''.join(body)}
<h2>How to read this</h2><p>The 1 ms evidence audit independently checks recorded native simulator rows. The policy gate is sampled at control boundaries; the report preserves any disagreement. “Final planning gate” combines the simulation receipt with caller-declared load, friction, geometry, force-limit and pose assumptions. It is a planning result, not permission to actuate.</p>
<p>CPU values show per-attempt policy-call and complete-step computation summaries, in milliseconds (mean / p95). They exclude sensors, transport, physical actuation and evidence serialization. Host timing is not a real-time guarantee.</p>
<p>The evidence receipt uses privileged simulator state and an authored synthetic force ceiling. It does not measure real material properties, establish a calibrated force limit, or demonstrate that a physical gripper can hold the object.</p>
<footer><a href="report.json">Download report.json</a> · <a href="protocol.json">Study protocol</a> · <a href="audit.json">Independent audit</a> · <a href="inspection.json">Inspection receipt</a> · <a href="https://research.dvidia.org/papers/piper-probe-evidence-v01/">Research page</a> · <a href="https://github.com/Dvidia-Inference/dvidia-training/releases/tag/piper-probe-evidence-v0.1-preview">Source release</a> · <a href="https://huggingface.co/datasets/Dvidia/piper-probe-evidence-v01">Evidence archive</a></footer>
</main></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path, help='bounded report.json')
    parser.add_argument('--output', type=Path, help='HTML path (default: beside report.json)')
    args = parser.parse_args()
    report = load(args.report)
    output = args.output or args.report.with_name('report.html')
    output.write_text(render(report), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()

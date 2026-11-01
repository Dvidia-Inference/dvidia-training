"""Render all fixed incremental-hold attempts without simulator imports."""
import argparse
import html
import json
from pathlib import Path

VARIANTS = ('measured_joint', 'issued_joint', 'cartesian', 'incremental')


def unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError('Duplicate JSON field')
        out[key] = value
    return out


def load(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16*1024*1024:
        raise ValueError('Expected bounded report file')
    def invalid(_):
        raise ValueError('Nonfinite JSON')
    doc = json.loads(path.read_bytes(), object_pairs_hook=unique, parse_constant=invalid)
    rows = doc.get('rows', [])
    scenes = {r['scene_id'] for r in rows}
    if (doc.get('schema') != 'dvidia.piper-incremental-hold-study/v1'
            or doc.get('attempted') != 32 or doc.get('planned_attempts') != 32
            or len(rows) != 32 or len(scenes) != 8
            or {(r['scene_id'], r['variant']) for r in rows} != {(s,v) for s in scenes for v in VARIANTS}):
        raise ValueError('Expected complete eight-scene, four-controller study')
    return doc


def render(doc):
    esc = lambda value: html.escape(str(value), quote=True)
    truth = lambda value: 'yes' if value is True else 'no' if value is False else 'unknown'
    summary, details = [], []
    for variant in VARIANTS:
        group = [r for r in doc['rows'] if r['variant'] == variant]
        evaluations = [r['evaluation'] for r in group]
        count = lambda field: sum(e['primary'].get(field) is True for e in evaluations)
        summary.append('<tr>'+''.join(f'<td>{esc(v)}</td>' for v in (
            variant.replace('_',' '), f'{sum(e.get("first_admission") is not None for e in evaluations)}/8',
            f'{count("native_pause_success")}/8', f'{count("success")}/8',
            f'{count("combined_qualified_pause")}/8'))+'</tr>')
        body = []
        for r in group:
            e, p = r['evaluation'], r['evaluation']['primary']
            excursion = e.get('stationarity', {}).get('maximum_tcp_position_excursion_m')
            cells = (r['scene_id'], truth(r['valid']), truth(p.get('native_pause_success')),
                     truth(p.get('success')), truth(p.get('combined_qualified_pause')),
                     f'{excursion*1000:.3f} mm' if type(excursion) in (int,float) else 'unavailable',
                     p.get('reason','unknown'))
            body.append('<tr>'+''.join(f'<td>{esc(v)}</td>' for v in cells)+'</tr>')
        details.append(f'<details><summary>{esc(variant.replace("_"," "))} · all eight attempts</summary><div class="scroll"><table><thead><tr><th>Scene</th><th>Valid</th><th>Grip</th><th>Stationary grip</th><th>Active stationary grip</th><th>TCP excursion</th><th>Reason</th></tr></thead><tbody>{"".join(body)}</tbody></table></div></details>')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>A grip that stays in place · DVIDIA research</title><style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f8f7f3;color:#242824;font:16px/1.65 system-ui,sans-serif}main{max-width:1020px;margin:auto;padding:44px 22px 80px}nav,footer{font-size:.85rem;color:#60675f}a{color:#355b40;text-underline-offset:3px}h1{font:normal clamp(2.4rem,6vw,4.4rem)/1.07 Georgia,serif;letter-spacing:-.035em;max-width:15ch}h2{font:normal 1.7rem Georgia,serif;margin-top:2rem}.lede{font-size:1.2rem;max-width:65ch}.note{border-left:3px solid #7d947d;padding:12px 18px;background:#eeefe8}table{width:100%;border-collapse:collapse;font-size:.9rem}th,td{text-align:left;padding:13px 12px;border-bottom:1px solid #d8dcd1}th{font-size:.75rem;color:#626b60}.scroll{overflow:auto}summary{cursor:pointer;padding:16px 0;font-weight:600}details{border-bottom:1px solid #d8dcd1}footer{margin-top:40px}code{overflow-wrap:anywhere}
</style><main><nav><a href="https://research.dvidia.org/">DVIDIA research</a> / PiPER · v0.3</nav><h1>A grip that stays in place.</h1><p class="lede">A frozen comparison of four hold controllers on eight new simulated scenes. Holding the object and keeping the tool in place are measured together.</p><p class="note">32 scheduled attempts. Every failure stays in the denominator. Authored controls using simulator state; no model training or hardware qualification.</p><h2>Five-second outcomes</h2><div class="scroll"><table><thead><tr><th>Controller</th><th>Admitted</th><th>Native grip</th><th>Native stationary grip</th><th>Active stationary grip</th></tr></thead><tbody>'''+''.join(summary)+'''</tbody></table></div><p>The primary endpoint requires continuous grip plus TCP motion within 4 mm and 0.05 rad. “Active stationary grip” additionally requires the controller to remain qualified. A post-refusal fallback can retain an object; that does not establish successful active regulation.</p><h2>Every attempt</h2>'''+''.join(details)+'''<h2>What changed</h2><p>The new incremental candidate accumulates the current pose-error step around the target issued before admission. It retains the jaw request, limits each correction update to 0.002 rad and refuses a proposed total above 0.03 rad. Candidate measurements outside the native joint margin are refused before solving. The three previous controllers remain unchanged.</p><p>These are nearby deterministic development cells, not independent random trials or a real-world reliability estimate. The study does not test transport, placement, videos, cameras or cross-arm transfer. Timing in the JSON is shared-host offline computation, not sensor latency.</p><footer><a href="report.json">Report JSON</a> · <a href="protocol.json">Frozen protocol</a> · <a href="inspection.json">Inspection</a> · <a href="https://research.dvidia.org/downloads/topics/physical-grounding/runs/2026-10-31-piper-incremental-hold-v03/report/audit.json">Published independent audit</a> · <a href="https://research.dvidia.org/papers/piper-incremental-hold-v03/">Research note</a> · <a href="https://github.com/Dvidia-Inference/dvidia-training/releases/tag/piper-incremental-hold-v0.3-preview">GitHub</a> · <a href="https://huggingface.co/datasets/Dvidia/piper-incremental-hold-v03">Hugging Face</a></footer></main></html>'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.report.with_name('report.html')
    with output.open('x') as stream:
        stream.write(render(load(args.report)))

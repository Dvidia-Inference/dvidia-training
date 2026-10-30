"""Build the bounded-grip-wait recovery summaries offline, without new physics.

The fixed input identities are deliberately narrow. This utility neither reads
credentials nor contacts a remote service; publication is a separate operation.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re


SCHEMA = 'dvidia.piper-grip-wait-compact-release/v1'
REPO_ID = 'Dvidia/piper-grip-wait-recovery-v04'
BENCH = 'benchmarks/piper-grip-wait-v04/recovery-20261030'
PAPER = 'reports/Bounded grip waiting v04.md'
FULL_REPORT_SHA256 = '87d252d2ffe2aea5a1bca6b50d926abbd2181b4f51695471dfb5747af91e4e56'
EXPECTED = {
    BENCH + '/README.md': '9bd0b69e4e19c5e5fdaa61bc419edeb5df905e6f50a11803db536f3244792c3a',
    BENCH + '/audit-summary.json': 'a1cb4dd0a35228fd738a2881c50f6791d5fc799ac5b8f3b3cb27dd2797ad7541',
    BENCH + '/interruptions.json': '5e7724978a532d802bcefa63144dab9d180ebc41f9a7c3bbc64af895bea2e403',
    BENCH + '/report.json': 'acded121e3593e944b34c1cc5efde2b0d6bda511cb5ebdfffa16741243c65fe6',
    BENCH + '/source-lineage.json': '30a6c8f4fde7fb9d006d906f67ee1d66798a2a1cd059a187f3b7a74c3a6d6d9f',
    PAPER: '2557ac42b4ff2da720700185e62620a75ca835cb34fd000f4b470e587048dc50',
}
SOURCE_PATHS = (
    'src/dvidia_training/piper_grip_wait_v04.py',
    'src/dvidia_training/piper_grip_wait_v04_benchmark.py',
    'publishing/recover_piper_grip_wait_v04.py',
    'publishing/audit_piper_grip_wait_v04.py',
    'publishing/audit_piper_grip_clock_v03.py',
    'tests/test_piper_grip_wait_v04.py',
    'tests/test_piper_grip_wait_v04_benchmark.py',
    'publishing/test_recover_piper_grip_wait_v04.py',
    'publishing/test_audit_piper_grip_wait_v04.py',
    'publishing/test_audit_piper_grip_clock_v03.py',
)
MAX_FILES = 48
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 4 * 1024 * 1024
SUCCESS = {'guarded': 42, 'clock': 32, 'fixed': 44, 'teacher': 41}
PRIVATE_PATH = re.compile(r'(?<![A-Za-z0-9_.-])/(?:Users|private|var|tmp)/[^\s"\'<>]+')
SECRETS = (
    re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,})\b'),
    re.compile(r'\bAKIA[0-9A-Z]{16}\b'),
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    re.compile(r'(?i)\b(?:api[_-]?key|access[_-]?token|secret[_-]?key)\s*[:=]\s*["\']?[A-Za-z0-9_+/=-]{24,}'),
)


def _identity(raw):
    return {'bytes': len(raw), 'sha256': sha256(raw).hexdigest()}


def _json(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def _relative(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or '\\' in name or ':' in name
            or any(piece in ('', '.', '..') for piece in name.split('/'))):
        raise ValueError('Unsafe relative artifact path.')
    return path.as_posix()


def _ordinary(path):
    path = Path(path).absolute()
    platform_links = {Path('/tmp'), Path('/var')}
    if any(item.is_symlink() and item not in platform_links for item in (path, *path.parents)):
        raise ValueError('Artifact paths cannot traverse symlinks.')
    if not path.is_file():
        raise ValueError('Required ordinary input is absent: ' + path.name)
    return path


def _scan(name, raw):
    _relative(name)
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError('Artifact exceeds 2 MiB: ' + name)
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError as error:
        raise ValueError('Only UTF-8 textual artifacts are permitted.') from error
    if '\x00' in text:
        raise ValueError('Binary content is not permitted.')
    if PRIVATE_PATH.search(text):
        raise ValueError('Private absolute path found: ' + name)
    if any(pattern.search(text) for pattern in SECRETS):
        raise ValueError('Credential-shaped literal found: ' + name)


def _validate_report(report):
    if (report.get('scheduled_attempts') != 512
            or report.get('completed_scored_attempts') != 508
            or report.get('interrupted_unevaluable_attempts') != 4
            or report.get('full_raw_report', {}).get('sha256') != FULL_REPORT_SHA256
            or report.get('qualified_robot_skill') is not False
            or report.get('physical_robot_ready') is not False):
        raise ValueError('The compact report does not match the frozen recovery scope.')
    rows = report.get('rows', [])
    if len(rows) != 512 or len({(r['case_id'], r['variant']) for r in rows}) != 512:
        raise ValueError('Scheduled rows are missing or duplicated.')
    unknown = [r for r in rows if r.get('interrupted_unevaluable')]
    if len(unknown) != 4 or any(r.get('raw_strict_success') is not None for r in unknown):
        raise ValueError('Unknown outcomes must remain unevaluable.')
    for variant, success in SUCCESS.items():
        selected = [r for r in rows if r['variant'] == variant]
        if (len(selected) != 128 or sum(r.get('raw_strict_success') is True for r in selected) != success
                or report['variants'][variant]['success'] != success):
            raise ValueError('Controller totals differ from the frozen compact report.')


def _scope(sources=None):
    value = {
        'schema': SCHEMA, 'repo_id': REPO_ID, 'repo_type': 'dataset',
        'full_raw_native_archive_uploaded': False, 'full_raw_report_uploaded': False,
        'full_raw_native_evidence_retained_locally': True,
        'training_weights_included': False, 'newly_trained_model': False,
        'authored_acquisition_supervisor': True, 'learned_prediction_advisory': True,
        'physical_robot_ready': False, 'qualified_robot_skill': False,
        'scheduled_attempts': 512, 'completed_scored_attempts': 508,
        'interrupted_unevaluable_attempts': 4, 'successes': SUCCESS,
        'full_raw_report_sha256': FULL_REPORT_SHA256,
        'scope': 'Compact summaries of frozen simulated rigid-block evidence; no new training or physics.',
        'missingness': 'Four interrupted outcomes are unknown. Only the primary scheduled analysis imputes failure.',
    }
    if sources is not None:
        value['sources'] = sources
    return value


CARD = '''---
license: mit
language:
- en
task_categories:
- robotics
tags:
- robotics
- simulation
- agilex-piper
- manipulation
- benchmark
pretty_name: DVIDIA Bounded Grip Waiting v0.4 — Compact Recovery Evidence
size_categories:
- n<1K
---

# Bounded grip waiting v0.4

**Compact simulation evidence and source excerpts, with every scheduled outcome visible.**

This DVIDIA study tests an authored acquisition supervisor that permits approach
motion, waits for secure contact only when eligible, and aborts if its 200 ms
waiting budget expires or support is lost. Frozen learned predictions are advisory.
**No new model or training weights are included or were trained for this study.**

| Controller | Strict successes / scheduled | Complete, scored | Unknown interruption |
| --- | ---: | ---: | ---: |
| Bounded waiting | 42/128 | 127 | 1 |
| Learned clock | 32/128 | 127 | 1 |
| Fixed progression | 44/128 | 127 | 1 |
| Teacher | 41/128 | 127 | 1 |

The paired bounded-minus-learned contrast is +7.81 percentage points, with an
adjusted interval of −7.25 to +22.87. Bounded-minus-fixed is −1.56 points,
with an interval of −12.44 to +9.32. **Both comparisons are inconclusive.**
Four interrupted outcomes remain unknown; the primary 128-attempt denominator
per controller imputes those unknowns as failures only for that analysis.
The sensitivity analysis excludes their entire XY group, leaving 31 groups.

## Read the evidence

- [Research note](reports/Bounded%20grip%20waiting%20v04.md)
- [Research-site paper](https://research.dvidia.org/papers/bounded-grip-waiting-v04/)
  (publication is prepared separately; this builder does not verify availability)
- [Compact report and all 512 scheduled rows](benchmarks/piper-grip-wait-v04/recovery-20261030/report.json)
- [556 independently audited complete attempts, zero mismatches](benchmarks/piper-grip-wait-v04/recovery-20261030/audit-summary.json)
- [Four interruption byte manifests](benchmarks/piper-grip-wait-v04/recovery-20261030/interruptions.json)
- [Source and recovery lineage](benchmarks/piper-grip-wait-v04/recovery-20261030/source-lineage.json)
- [Source subset and its limitations](source/README.md)
- [Machine-readable scope](scope.json) and [byte inventory](integrity.json)

The 72 original complete receipts and four incomplete folders were preserved;
only the 436 never-started requests were executed during recovery. There are
508 scored, valid confirmation attempts, **not 512 valid attempts**.
All 48 development and 508 complete confirmation attempts were independently
audited. The 22 bounded-controller aborts are explicit failures, not recoveries.

## Contents and omissions

This dataset contains five compact evidence artifacts, one research note,
selected MIT-licensed source/test snapshots, and release metadata. It does
**not** contain native arrays, full decision traces, full per-step timing,
the full 35,327,217-byte recovery report, robot meshes, motor weights, or a
standalone simulator installation. Full raw native evidence remains retained
locally; **the full raw native archive has not been uploaded in this release**.

The compact artifacts bind the full local report to SHA256
`87d252d2ffe2aea5a1bca6b50d926abbd2181b4f51695471dfb5747af91e4e56`.
Repository-relative evidence paths identify omitted local evidence and are
not downloadable files within this compact dataset.

## Limits

One simulated rigid 40 g block, one PiPER reference arm, two known destinations,
and a narrow starting envelope. Contacts and state are privileged simulator
observations, not camera or human-video inputs. Intervention exposure differs
between controllers, so retention counts do not establish better slip recovery.
Original and resumed CPU timings come from different dates under shared load;
there is no algorithmic speedup, hard real-time, or physical safety claim.
This is **not a qualified robot skill or a physical-robot-ready release**.

MIT license applies to the authored materials and source excerpts included here.
No third-party robot assets or model weights are redistributed.

[DVIDIA research](https://research.dvidia.org/) ·
[Source repository](https://github.com/Dvidia-Inference/dvidia-training) ·
[@iammrriver](https://x.com/iammrriver) · hello@dvidia.org
'''

SOURCE_README = '''# Source snapshots — a documented subset

These files preserve selected controller, benchmark, recovery, independent-audit,
and test sources exactly as read from the repository at packaging time. Their
repository-relative origins and SHA256 identities are listed in `../scope.json`.

This directory is **not standalone software or the full runtime**. Imported
DVIDIA modules, Python dependencies, the frozen v0.3 motor model, native arrays,
robot assets and weights are excluded. Copying these excerpts alone will not
run the simulator or reconstruct every native audit. Consult the full
[DVIDIA training repository](https://github.com/Dvidia-Inference/dvidia-training)
and the compact lineage for the evidence identities. Publication tooling is
also available in that repository; no credential or publisher configuration
is included in this dataset.

No model was newly trained by the bounded supervisor. These are MIT-licensed
authored source excerpts; no upstream asset license is being replaced.
'''

NOTICE = '''DVIDIA bounded grip waiting v0.4 — compact recovery evidence
Copyright (c) 2026 DVIDIA contributors

Included authored documentation, compact benchmark summaries, source snapshots,
and viewer code are supplied under the accompanying MIT License.
No robot mesh, third-party physical asset, model weight, or raw video is included.
Repository-relative references to omitted evidence do not imply its inclusion.
This release is simulation research, not a hardware safety certification.
Contact: hello@dvidia.org
'''

RESEARCH_README = '''# Bounded grip waiting v0.4 — research evidence viewer

Serve this directory with any static HTTP server and open index.html. The viewer
reads only the bundled compact JSON; it is not a live physics simulation and
does not contact a model, API or hardware. All 512 scheduled records are visible.
Unknown interrupted outcomes are shown as Unknown, separately from analysis
failure imputation. This bundle contains no raw native archive or new weights.

[Research note](reports/Bounded%20grip%20waiting%20v04.md) ·
[Compact artifacts](benchmarks/piper-grip-wait-v04/recovery-20261030/README.md) ·
[Scope](scope.json) · [Integrity](integrity.json)

MIT license. DVIDIA · hello@dvidia.org
'''

HTML = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="DVIDIA bounded grip waiting v0.4: all scheduled simulation outcomes, including interrupted unknowns.">
<meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'">
<title>Bounded grip waiting v0.4 · DVIDIA research</title><link rel="stylesheet" href="viewer.css"><script src="viewer.js" defer></script></head>
<body><a class="skip" href="#main">Skip to evidence</a><header><a href="https://research.dvidia.org/">DVIDIA research</a><a href="https://huggingface.co/datasets/Dvidia/piper-grip-wait-recovery-v04">Dataset ↗</a></header>
<main id="main"><p class="eyebrow">MANIPULATION RESEARCH · V0.4</p><h1>Wait for the grip.<br>Keep a way forward.</h1>
<p class="lede">A finite acquisition contract for one simulated rigid-block task. Progress is explicit; reliable superiority remains unproven.</p>
<p class="scope">Recorded benchmark summaries · No live physics · No new training weights · Not qualified for physical robots</p>
<section aria-labelledby="findings"><h2 id="findings">What changed</h2><p>The authored supervisor allows approach and centering, then waits for supported secure contact. Its total wait budget is 200 ms. Exhausted budgets and loss of support produce an explicit abort; they do not force an insecure lift.</p>
<div class="numbers"><p><strong>512</strong> scheduled attempts</p><p><strong>508</strong> scored outcomes</p><p><strong>4</strong> interrupted, unknown</p><p><strong>556</strong> complete attempts audited; zero mismatches</p></div>
<table class="overview"><caption>Strict placements; 128 scheduled attempts per controller</caption><thead><tr><th scope="col">Controller</th><th scope="col">Successes</th><th scope="col">Scored</th><th scope="col">Unknown</th></tr></thead><tbody><tr><th scope="row">Bounded waiting</th><td>42/128</td><td>127</td><td>1</td></tr><tr><th scope="row">Learned clock</th><td>32/128</td><td>127</td><td>1</td></tr><tr><th scope="row">Fixed progression</th><td>44/128</td><td>127</td><td>1</td></tr><tr><th scope="row">Teacher</th><td>41/128</td><td>127</td><td>1</td></tr></tbody></table>
<p>Bounded minus learned: <strong>+7.81 percentage points</strong> (adjusted interval −7.25 to +22.87). Bounded minus fixed: <strong>−1.56 points</strong> (−12.44 to +9.32). Both comparisons are inconclusive.</p>
<p>Four interruptions are unknown observations. The primary analysis conservatively imputes failure; a sensitivity analysis removes the entire affected XY group. Timing sessions occurred on different dates under shared load. Unequal pulse exposure prevents a slip-recovery advantage claim.</p></section>
<section aria-labelledby="records"><h2 id="records">Every scheduled attempt</h2><p>Inspect outcomes and source identities. Unknown means the session ended before complete evidence existed; it is not an observed controller failure.</p>
<form id="filters"><label>Controller<select id="variant"><option value="">All controllers</option><option value="guarded">Bounded waiting</option><option value="clock">Learned clock</option><option value="fixed">Fixed progression</option><option value="teacher">Teacher</option></select></label><label>Outcome<select id="outcome"><option value="">All outcomes</option><option value="success">Success</option><option value="failure">Observed failure</option><option value="unknown">Unknown interruption</option></select></label><label>Condition<select id="condition"><option value="">Both conditions</option><option value="nominal">Nominal</option><option value="post_grip">Post-grip pulse assigned</option></select></label><label>Case search<input type="search" id="search" placeholder="e.g. xy04"></label><button type="reset">Reset filters</button></form>
<p id="status" role="status" aria-live="polite">Loading the compact evidence…</p><div class="table-wrap" role="region" aria-label="Scheduled attempt records" tabindex="0"><table id="attempts"><caption>All 512 scheduled records; filtered locally</caption><thead><tr><th scope="col">Case</th><th scope="col">Controller</th><th scope="col">Condition</th><th scope="col">Outcome</th><th scope="col">Wait</th><th scope="col">Acquisition</th><th scope="col">Receipt / interpretation</th></tr></thead><tbody id="rows"></tbody></table></div><noscript><p>JavaScript is needed to filter the table. The complete <a href="benchmarks/piper-grip-wait-v04/recovery-20261030/report.json">JSON report</a> is available directly.</p></noscript></section>
<section aria-labelledby="limits"><h2 id="limits">Read the boundaries</h2><p>One simulated rigid 40 g block, two known goals, and a narrow starting envelope. Privileged simulator contacts and state, rather than cameras or human videos. The acquisition latch does not continuously certify retention. There is no hardware, real-time, or general video-to-skill claim.</p><p>The compact bundle omits native arrays, complete traces, and the full raw report. Those remain retained locally and are identified by hashes. This publication does not upload that raw archive.</p>
<ul><li><a href="reports/Bounded%20grip%20waiting%20v04.md">Research note</a></li><li><a href="benchmarks/piper-grip-wait-v04/recovery-20261030/report.json">All outcomes and both statistical analyses</a></li><li><a href="benchmarks/piper-grip-wait-v04/recovery-20261030/audit-summary.json">556 audit identities</a></li><li><a href="benchmarks/piper-grip-wait-v04/recovery-20261030/interruptions.json">Four interruption manifests</a></li><li><a href="benchmarks/piper-grip-wait-v04/recovery-20261030/source-lineage.json">Source lineage</a></li><li><a href="scope.json">Release scope</a> · <a href="integrity.json">Byte inventory</a> · <a href="LICENSE">MIT License</a></li></ul></section>
</main><footer><span>DVIDIA · hello@dvidia.org</span><a href="https://x.com/iammrriver">@iammrriver ↗</a></footer></body></html>
'''

CSS = '''*{box-sizing:border-box}html{color-scheme:light;scroll-behavior:smooth}body{margin:0;color:#252722;background:#faf9f5;background-image:radial-gradient(#40473509 .5px,transparent .5px);background-size:3px 3px;font:17px/1.65 Georgia,'Times New Roman',serif}a{color:#295a48;text-underline-offset:.2em}a:hover{text-decoration-thickness:2px}header,footer{max-width:1160px;margin:auto;padding:25px 30px;display:flex;justify-content:space-between;gap:20px;font:14px/1.5 system-ui,sans-serif}header{border-bottom:1px solid #d8d9cf}main{max-width:1100px;margin:55px auto;padding:0 30px}h1{font-size:clamp(38px,5vw,65px);font-weight:400;letter-spacing:-.045em;line-height:1.08;margin:18px 0 25px}h2{font-size:28px;font-weight:400;letter-spacing:-.02em}.eyebrow,.scope{font:12px/1.6 system-ui,sans-serif;letter-spacing:.09em}.eyebrow{color:#5c6d60}.lede{max-width:710px;font-size:22px;color:#596057}.scope{letter-spacing:.02em;border-block:1px solid #d8d9cf;padding:13px 0;color:#555e52}section{margin-top:48px}section>p{max-width:850px}.numbers{display:grid;grid-template-columns:repeat(4,1fr);gap:22px;margin:30px 0;font:13px/1.5 system-ui,sans-serif}.numbers strong{display:block;font:35px/1.3 Georgia,serif;margin-bottom:6px}table{border-collapse:collapse;width:100%;font:13px/1.6 system-ui,sans-serif;text-align:left}caption{text-align:left;color:#5b6458;padding:10px 0}th,td{padding:12px 10px;border-bottom:1px solid #dcded4;vertical-align:top}th{font-weight:600}thead th{background:#efefe7;color:#424c40}.overview{max-width:760px}.table-wrap{overflow:auto;max-height:650px;overscroll-behavior:contain;border-block:1px solid #cdd2c6}.table-wrap thead{position:sticky;top:0}.table-wrap table{min-width:950px}.mono{font-family:ui-monospace,monospace;font-size:11px;overflow-wrap:anywhere}.result{white-space:nowrap;font-weight:600}.unknown{color:#85600a}.success{color:#2b6142}.failure{color:#895047}form{display:flex;flex-wrap:wrap;align-items:end;gap:15px;margin:25px 0;font:12px/1.4 system-ui,sans-serif}label{display:grid;gap:6px;flex:1;min-width:150px}input,select,button{font:14px/1.4 system-ui,sans-serif;border:1px solid #bec6b8;border-radius:5px;background:#fffefa;color:#252722;padding:11px 12px;min-height:43px}button{cursor:pointer}button:hover{background:#edeee5}:focus-visible{outline:3px solid #668a59;outline-offset:3px}#status{font:13px/1.5 system-ui,sans-serif;color:#566150}.skip{position:absolute;left:20px;top:-60px;background:#fff;padding:8px}.skip:focus{top:8px}footer{border-top:1px solid #d8d9cf;margin-top:60px}.small{font-size:11px;color:#5e665a}ul{padding-left:20px}@media(max-width:600px){header,footer{padding:20px}main{padding:0 20px;margin-top:35px}.numbers{grid-template-columns:repeat(2,1fr);gap:10px}section{margin-top:36px}h2{font-size:25px}.lede{font-size:20px}footer{flex-wrap:wrap}}
'''

JS = ''''use strict';
const labels={guarded:'Bounded waiting',clock:'Learned clock',fixed:'Fixed progression',teacher:'Teacher'};
const state={rows:[]};
function kind(row){return row.interrupted_unevaluable?'unknown':row.success===true?'success':'failure';}
function cell(text,cls){const td=document.createElement('td');td.textContent=text;if(cls)td.className=cls;return td;}
function render(){
 const variant=document.getElementById('variant').value,outcome=document.getElementById('outcome').value,condition=document.getElementById('condition').value,search=document.getElementById('search').value.trim().toLowerCase();
 const rows=state.rows.filter(r=>(!variant||r.variant===variant)&&(!outcome||kind(r)===outcome)&&(!condition||r.condition===condition)&&(!search||r.case_id.toLowerCase().includes(search)));
 const body=document.getElementById('rows'),frag=document.createDocumentFragment();
 for(const r of rows){
  const tr=document.createElement('tr');tr.dataset.outcome=kind(r);tr.dataset.variant=r.variant;
  const result=kind(r),supervisor=r.supervisor_summary;
  tr.append(cell(r.case_id,'mono'),cell(labels[r.variant]),cell(r.condition==='nominal'?'Nominal':'Pulse assigned'),cell(result==='unknown'?'Unknown':result==='success'?'Success':'Observed failure','result '+result));
  tr.append(cell(result==='unknown'?'—':r.variant==='guarded'&&supervisor?Math.round(supervisor.wait_seconds*1000)+' ms':'—'));
  tr.append(cell(result==='unknown'?'Not scored':r.variant!=='guarded'?'Not applicable':r.aborted?'Aborted: '+r.abort_reason:r.acquisition_released?'Released':'Not latched'));
  const evidence=document.createElement('td');
  if(result==='unknown'){evidence.append(document.createTextNode('Interrupted; native outcome unknown.'));const note=document.createElement('div');note.className='small';note.textContent='Primary analysis imputes failure; not an observed failure.';evidence.append(note);}
  else{const reason=document.createElement('div');reason.textContent=r.score_reason;const hash=document.createElement('div');hash.className='mono';hash.textContent='Receipt SHA256: '+r.receipt_sha256;evidence.append(reason,hash);}
  tr.append(evidence);frag.append(tr);
 }
 body.replaceChildren(frag);
 document.getElementById('status').textContent=rows.length+' of 512 scheduled records · '+rows.filter(r=>kind(r)==='unknown').length+' unknown in this view. Raw native arrays are not included.';
}
document.getElementById('filters').addEventListener('input',render);
document.getElementById('filters').addEventListener('change',render);
document.getElementById('filters').addEventListener('reset',()=>setTimeout(render,0));
fetch('benchmarks/piper-grip-wait-v04/recovery-20261030/report.json',{credentials:'omit',cache:'no-store'}).then(r=>{if(!r.ok)throw Error('unavailable');return r.json();}).then(report=>{
 if(report.schema!=='dvidia.piper-grip-wait-compact-recovery/v1'||report.rows.length!==512||report.rows.filter(r=>r.interrupted_unevaluable).length!==4)throw Error('scope');
 state.rows=report.rows;render();
}).catch(()=>{document.getElementById('status').textContent='Could not load the compact report. Serve this directory over HTTP, or read the JSON report directly below.';});
'''


def _checked_inputs(repo):
    result = {}
    for name, digest in EXPECTED.items():
        raw = _ordinary(repo / name).read_bytes()
        if sha256(raw).hexdigest() != digest:
            raise ValueError('Frozen compact input identity changed: ' + name)
        _scan(name, raw)
        result[name] = raw
    _validate_report(json.loads(result[BENCH + '/report.json']))
    audit = json.loads(result[BENCH + '/audit-summary.json'])
    if audit['audited_attempts'] != 556 or audit['audit_mismatches'] != 0:
        raise ValueError('Audit summary scope changed.')
    return result


def _with_integrity(files):
    if len(files) + 1 > MAX_FILES:
        raise ValueError('Payload exceeds 48 files.')
    for name, raw in files.items():
        _scan(name, raw)
    result = dict(files)
    result['integrity.json'] = _json({'schema': 'dvidia.compact-artifact-integrity/v1',
                                     'files': {name: _identity(raw) for name, raw in sorted(files.items())}})
    _scan('integrity.json', result['integrity.json'])
    if sum(map(len, result.values())) > MAX_TOTAL_BYTES:
        raise ValueError('Payload exceeds 4 MiB.')
    return result


def _fresh_output(path):
    path = Path(path).absolute()
    platform_links = {Path('/tmp'), Path('/var')}
    if any(item.is_symlink() and item not in platform_links for item in (path, *path.parents)):
        raise ValueError('Outputs cannot traverse symlinks.')
    if path.exists():
        raise ValueError('Output must be fresh and absent: ' + path.name)
    if not path.parent.is_dir():
        raise ValueError('Output parent must already exist.')
    return path


def build(repo, hf_output, research_output):
    repo = Path(repo).absolute()
    hf_output, research_output = _fresh_output(hf_output), _fresh_output(research_output)
    if hf_output == research_output or hf_output in research_output.parents or research_output in hf_output.parents:
        raise ValueError('Outputs cannot overlap.')
    inputs = _checked_inputs(repo)
    license_raw = _ordinary(repo / 'LICENSE').read_bytes()
    source_files, sources = {}, {}
    for name in SOURCE_PATHS:
        raw = _ordinary(repo / name).read_bytes()
        target = 'source/' + Path(name).name
        if target in source_files:
            raise ValueError('Source snapshot basename collision.')
        source_files[target] = raw
        sources[target] = dict(source_path=name, **_identity(raw))
    hf = _with_integrity(dict(inputs, **source_files, **{
        'README.md': CARD.encode(), 'LICENSE': license_raw, 'NOTICE.txt': NOTICE.encode(),
        'scope.json': _json(_scope(sources)), 'source/README.md': SOURCE_README.encode(),
    }))
    research = _with_integrity(dict(inputs, **{
        'README.md': RESEARCH_README.encode(), 'LICENSE': license_raw, 'NOTICE.txt': NOTICE.encode(),
        'scope.json': _json(_scope()), 'index.html': HTML.encode(),
        'viewer.js': JS.encode(), 'viewer.css': CSS.encode(),
    }))
    for root, files in ((hf_output, hf), (research_output, research)):
        root.mkdir()
        for name, raw in sorted(files.items()):
            destination = root / _relative(name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                stream.write(raw)
        for name, raw in files.items():
            if (root / name).read_bytes() != raw:
                raise ValueError('Written artifact readback differs.')
    return {'schema': 'dvidia.compact-recovery-build/v1',
            'huggingface': {'files': len(hf), 'bytes': sum(map(len, hf.values())),
                            'integrity_sha256': sha256(hf['integrity.json']).hexdigest(),
                            'card_sha256': sha256(hf['README.md']).hexdigest()},
            'research': {'files': len(research), 'bytes': sum(map(len, research.values())),
                         'integrity_sha256': sha256(research['integrity.json']).hexdigest()},
            'source_snapshots': sources}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--hf-output', type=Path, required=True)
    parser.add_argument('--research-output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.repo, args.hf_output, args.research_output), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

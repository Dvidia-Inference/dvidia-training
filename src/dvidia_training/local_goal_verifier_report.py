"""Portable CPU-verifier report; pixels illustrate scenes, not verifier inputs.

This is separate from the frozen Jev report. It exports only declared public
fields, independently counts each cohort, and never calls a service or robot.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re

COHORTS = ("native", "evidence_fault", "blind_spot")
VERDICTS = {"complete", "incomplete", "unknown"}
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}\Z")
PREDICATES = ("contract", "schema", "units", "cadence", "timestamps", "freshness", "identity", "frame", "measurement_validity", "uncertainty_validity", "history_span", "presence", "xy_position", "yaw", "linear_speed", "angular_speed", "support", "release")


def _plain(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Report paths must not contain symlinks.")
    return path


def _json(path):
    path = _plain(path)
    if not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Expected a bounded plain JSON artifact.")
    value = json.loads(path.read_bytes(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON.")))
    if not isinstance(value, dict):
        raise ValueError("Artifact must be a JSON object.")
    return value


def _encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def _name(value):
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise ValueError("Invalid public identifier.")
    return value


def _text(value, maximum=2000):
    if not isinstance(value, str) or len(value) > maximum:
        raise ValueError("Invalid bounded public text.")
    return value


def _finite(value):
    if type(value) not in (int, float) or not -1e100 <= value <= 1e100 or not math.isfinite(value):
        raise ValueError("Expected a finite number.")
    return value


def _relative(root, value):
    if not isinstance(value, str) or "\\" in value:
        raise ValueError("Expected a relative artifact path.")
    if not value or value.startswith("/") or any(p in {"", ".", ".."} for p in value.split("/")):
        raise ValueError("Artifact path must remain inside the run.")
    if any(not NAME.fullmatch(part) for part in value.split("/")):
        raise ValueError("Invalid artifact path characters.")
    path = _plain(root / value)
    if not path.is_file():
        raise ValueError("Referenced artifact does not exist.")
    return value


def _image(root, name, assets):
    path = root / _relative(root, name)
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Image exceeds its byte bound.")
    raw = path.read_bytes()
    suffix = ".png" if raw.startswith(b"\x89PNG\r\n\x1a\n") else ".jpg" if raw.startswith(b"\xff\xd8\xff") else None
    if suffix is None:
        raise ValueError("Only PNG and JPEG images are supported.")
    name = "images/" + sha256(raw).hexdigest() + suffix
    assets[name] = raw
    return name


def quantiles(values):
    if not values:
        return {"n": 0, "p50": None, "p95": None, "p99": None, "max": None}
    values = sorted(values)
    def p(fraction):
        at = (len(values) - 1) * fraction
        low, high = math.floor(at), math.ceil(at)
        return values[low] + (values[high] - values[low]) * (at - low)
    return {"n": len(values), "p50": p(.5), "p95": p(.95), "p99": p(.99), "max": values[-1]}


def _decision(source):
    if not isinstance(source, dict) or source.get("schema") != "dvidia.local-goal-verdict/v1" or source.get("verdict") not in VERDICTS:
        raise ValueError("Expected an explicit local-verifier verdict.")
    predicates = source.get("predicates")
    if not isinstance(predicates, dict) or set(predicates) != set(PREDICATES):
        raise ValueError("Expected all declared predicates.")
    clean = {}
    for key in PREDICATES:
        value = predicates[key]
        if not isinstance(value, dict) or value.get("status") not in {"pass", "fail", "unknown"}:
            raise ValueError("Invalid predicate status.")
        row = {"status": value["status"], "reason": _name(value.get("reason"))}
        for field in ("observed", "threshold", "age_s", "max_age_s", "absent_samples", "asserted_uncertainty_bound"):
            if field in value:
                row[field] = _finite(value[field])
        if "interval" in value:
            interval = value["interval"]
            if not isinstance(interval, dict) or set(interval) != {"lower", "upper"}:
                raise ValueError("Invalid uncertainty interval.")
            row["interval"] = {key: _finite(interval[key]) for key in ("lower", "upper")}
        for field in ("statistic", "operator", "detail"):
            if field in value:
                row[field] = _text(value[field])
        clean[key] = row
    reasons = source.get("reasons")
    if not isinstance(reasons, list) or len(reasons) > 32:
        raise ValueError("Invalid reason list.")
    window = source.get("window")
    if not isinstance(window, dict):
        raise ValueError("Invalid observation window.")
    window_keys = {"source_sample_count", "newest_t_s", "age_s", "declared_sample_period_s", "evaluated_sample_count", "evaluated_start_s", "evaluated_end_s", "evaluated_duration_s"}
    return {"verdict": source["verdict"], "reasons": [_name(reason) for reason in reasons], "predicates": clean,
            "window": {key: _finite(value) for key, value in window.items() if key in window_keys}}


def summarize(cases):
    """Every denominator includes all scheduled cases in its cohort."""
    summaries = []
    for cohort in COHORTS:
        rows = [case for case in cases if case["cohort"] == cohort]
        positives = [case for case in rows if case["physical_truth"]["complete"]]
        negatives = [case for case in rows if not case["physical_truth"]["complete"]]
        unknown = [case for case in rows if case["expected_verdict"] == "unknown"]
        answered = [case for case in rows if case["decision"]["verdict"] != "unknown"]
        summaries.append({"cohort": cohort, "planned": len(rows),
            "contract_agreement": {"correct": sum(case["decision"]["verdict"] == case["expected_verdict"] for case in rows), "total": len(rows)},
            "physical_agreement": {"correct": sum((case["decision"]["verdict"] == "complete") == case["physical_truth"]["complete"] for case in answered), "answered": len(answered), "total": len(rows)},
            "positive_recognition": {"correct": sum(case["decision"]["verdict"] == "complete" for case in positives), "total": len(positives)},
            "negative_recognition": {"correct": sum(case["decision"]["verdict"] == "incomplete" for case in negatives), "total": len(negatives)},
            "false_completion": {"count": sum(case["decision"]["verdict"] == "complete" for case in negatives), "total": len(negatives)},
            "abstentions": sum(case["decision"]["verdict"] == "unknown" for case in rows),
            "expected_abstention": {"correct": sum(case["decision"]["verdict"] == "unknown" for case in unknown), "total": len(unknown)},
            "generation_invalid": sum(not case["generation_valid"] for case in rows),
            "cpu_latency_us": quantiles([value / 1000 for case in rows for value in case["timing_ns"]])})
    return summaries


def build_report(run_dir, output_dir):
    """Render one completed local benchmark into a fresh directory."""
    root, output = _plain(run_dir), _plain(output_dir)
    if output.exists() or output == root or root in output.parents or output in root.parents:
        raise ValueError("Report output must be fresh and separate from the run.")
    catalog, result = _json(root / "cases.json"), _json(root / "results.json")
    if catalog.get("schema") != "dvidia.local-goal-cases/v1" or result.get("status") != "completed" or result.get("active_trial") is not None:
        raise ValueError("Only a completed, inactive local benchmark can be exported.")
    source_cases, source_rows = catalog.get("cases"), result.get("results")
    if not isinstance(source_cases, list) or not 1 <= len(source_cases) <= 1024 or not isinstance(source_rows, list):
        raise ValueError("Expected a bounded complete case cohort.")
    rows = {}
    for row in source_rows:
        identifier = _name(row.get("case_id"))
        if identifier in rows:
            raise ValueError("Duplicate result row.")
        rows[identifier] = row
    cases, assets, seen = [], {}, set()
    for source in source_cases:
        identifier = _name(source.get("id"))
        if identifier in seen or identifier not in rows:
            raise ValueError("Duplicate case or missing result.")
        seen.add(identifier)
        if source.get("cohort") not in COHORTS or source.get("expected_verdict") not in VERDICTS:
            raise ValueError("Unknown cohort or reference verdict.")
        truth = source.get("physical_truth", {})
        if any(type(truth.get(key)) is not bool for key in ("present", "complete")) or type(source.get("generation_valid")) is not bool:
            raise ValueError("Explicit physical truth and generation status are required.")
        row = rows[identifier]
        timings = row.get("timing_ns")
        if not isinstance(timings, list) or not 1 <= len(timings) <= 10000 or any(type(value) is not int or not 0 < value <= 10**12 for value in timings):
            raise ValueError("Require positive integer CPU timing samples.")
        case = {"id": identifier, "cohort": source["cohort"], "scenario": _name(source["scenario"]),
            "layout_id": _name(str(source["layout_id"])), "physical_truth": {key: truth[key] for key in ("present", "complete")},
            "expected_verdict": source["expected_verdict"], "generation_valid": source["generation_valid"],
            "note": _text(source.get("note", "")), "now_s": _finite(source["now_s"]),
            "packet_file": _relative(root, source["packet_file"]),
            "current_image": _image(root, source["current_image"], assets), "goal_image": _image(root, source["goal_image"], assets),
            "decision": _decision(row.get("decision")), "timing_ns": timings}
        if source.get("state_file") is not None:
            case["state_file"] = _relative(root, source["state_file"])
        cases.append(case)
    if seen != rows.keys():
        raise ValueError("Unexpected result rows outside the planned cases.")
    public = {"schema": "dvidia.local-goal-report/v1", "status": "completed", "mode": "local-cpu-verifier",
        "model_calls": 0, "cases": cases, "summaries": summarize(cases),
        "timing_scope": "Repeated verify_goal calls on already loaded Python objects. Excludes sensing, image inference, rendering, parsing, network and actuation. Samples are repeated compute measurements, not independent physical trials.",
        "limits": ["Privileged native simulation measurements are the verifier input. Images illustrate the scenes; no perception model reads them here.",
            "The goal and clock are trusted caller inputs. A valid-looking fabricated observation stream can produce a physically wrong completion.",
            "Sampled 200 ms history is not continuous-time safety or hardware validation. No Jev calls, trained weights or robot-control commands."]}
    payload = {"report.json": _encoded(public), **assets}
    embedded = json.dumps(public, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    payload["index.html"] = _HTML.replace("__PUBLIC_DATA__", embedded).encode()
    seal = {"schema": "dvidia.local-goal-report-integrity/v1", "files": {name: {"bytes": len(raw), "sha256": sha256(raw).hexdigest()} for name, raw in sorted(payload.items())}}
    output.mkdir(parents=True)
    for name, raw in payload.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    (output / "report-integrity.json").write_bytes(_encoded(seal))
    return seal


_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>Local goal verifier · DVIDIA research</title>
<style>
:root{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#26302d;background:#faf9f6;font-synthesis:none}*{box-sizing:border-box}body{margin:0;background-image:radial-gradient(#28332c09 .55px,transparent .65px);background-size:5px 5px}a{color:inherit;text-underline-offset:4px}button,select{font:inherit;min-height:44px;touch-action:manipulation}button:focus-visible,select:focus-visible,a:focus-visible{outline:3px solid #3c6cff;outline-offset:3px}.skip{position:absolute;top:-80px;background:white;padding:14px;z-index:2}.skip:focus{top:10px}.masthead{padding:21px 5vw;border-bottom:1px solid #dedfd6;display:flex;align-items:center;justify-content:space-between;gap:20px}.brand{display:flex;gap:11px;align-items:center;text-decoration:none;font-weight:750;font-size:20px;letter-spacing:2px}.brand svg{width:34px;height:34px}.top-links{display:flex;gap:22px;font-size:13px}main{max-width:1440px;margin:auto;padding:45px 5vw}.eyebrow{font-size:11px;letter-spacing:2px;text-transform:uppercase;color:#69726b;font-weight:650}h1,h2,h3{font-family:Georgia,"Times New Roman",serif;font-weight:400}h1{font-size:clamp(38px,5vw,64px);letter-spacing:-2px;line-height:1.05;margin:15px 0 20px}.intro{max-width:800px;color:#59665f;line-height:1.8}.notice{border:1px solid #d7ded2;background:#f1f3ec;padding:16px 20px;margin:25px 0;font-size:14px;line-height:1.7}.warning{background:#f6eee3;border-color:#e3d3ba}.toolbar{margin:28px 0 18px;display:flex;gap:25px;align-items:end;justify-content:space-between}label{font-size:12px;color:#5b6b60;display:block;margin-bottom:7px}select{border:1px solid #cfd6cb;border-radius:5px;background:#fffefb;padding:10px 34px 10px 12px;color:inherit;max-width:100%}.meta,.note{font-size:12px;color:#637166;line-height:1.75}.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border-block:1px solid #dcded5;margin:20px 0 30px}.metric{padding:20px 20px 22px 0}.metric+.metric{padding-left:20px;border-left:1px solid #dcded5}.metric dt{font-size:12px;color:#627064}.metric dd{font:30px Georgia,serif;margin:10px 0 8px}.metric small{display:block;font-size:11px;color:#637166;line-height:1.5}.workspace{display:grid;grid-template-columns:240px minmax(0,1fr);gap:28px}.case-nav{border-top:1px solid #dcded5}.case-nav h2{font:12px system-ui;letter-spacing:1px;text-transform:uppercase;margin:17px 0}.case-nav select{width:100%;font-size:12px;margin-bottom:12px}.case-list{display:grid;gap:4px;max-height:760px;overflow:auto;padding:2px}.case-list button{text-align:left;border:0;background:transparent;padding:11px 10px;color:#516057;cursor:pointer;border-radius:4px;font-size:12px;line-height:1.5}.case-list button[aria-pressed=true]{background:#e9eee4;color:#263a2a;box-shadow:inset 3px 0 #5f775d}.case-list small{display:block;margin-top:3px;color:#657264}.case-heading{display:flex;justify-content:space-between;align-items:baseline;gap:15px}.case-heading h2{font-size:25px;margin:0 0 17px}.images{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0;min-width:0}figure img{width:100%;aspect-ratio:4/3;object-fit:contain;background:#e9e9df;border:1px solid #ddded4;display:block}figcaption{font-size:12px;color:#5c6a5e;margin:10px 0 16px}.comparison{display:grid;grid-template-columns:1fr 1fr;gap:20px;border-top:1px solid #dbdfd3;padding-top:20px;margin-top:10px}.comparison h3{font-size:22px;margin:0 0 14px}dl.facts{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:11px;font-size:12px}.facts dt{color:#617064}.facts dd{margin:0;text-align:right;max-width:240px;overflow-wrap:anywhere}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px}th,td{text-align:left;padding:10px 12px 10px 0;border-bottom:1px solid #e0e2d9}th{font-weight:550;color:#637165}.predicates td:last-child{min-width:190px}.badge{display:inline-block;border-radius:3px;background:#e9eee4;padding:3px 7px;white-space:nowrap}.badge.fail{background:#f3e5dd}.badge.unknown{background:#f0eee2}.downloads{display:flex;flex-wrap:wrap;gap:16px;font-size:13px;line-height:1.8;margin:22px 0}.fine{margin-top:35px;border-top:1px solid #dcded5;padding-top:22px}.fine h2{font-size:28px}.fine p,.fine li{font-size:13px;line-height:1.8;color:#5b695e;max-width:950px}.empty{padding:25px;font-size:13px;color:#657264}footer{border-top:1px solid #dcdfd5;margin-top:35px;padding-top:22px;display:flex;justify-content:space-between;gap:20px;font-size:12px;line-height:1.8;color:#657264}@media(max-width:950px){.workspace{grid-template-columns:200px minmax(0,1fr);gap:18px}.images{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.metric:nth-child(3){border-left:0;padding-left:0}.metric:nth-child(n+3){border-top:1px solid #dcded5}}@media(max-width:640px){main{padding:30px 18px}.masthead{padding:16px 18px}.top-links a:first-child{display:none}.workspace{grid-template-columns:1fr}.case-list{grid-template-columns:repeat(2,minmax(0,1fr));max-height:220px}.toolbar{display:block}.toolbar .meta{margin-top:15px}.comparison{grid-template-columns:1fr}.metric dd{font-size:27px}.case-heading{display:block}h1{letter-spacing:-1px}footer{display:block}}
</style></head><body><a class="skip" href="#experiment">Skip to experiment</a><header class="masthead"><a class="brand" href="https://dvidia.org/" aria-label="DVIDIA home"><svg viewBox="0 0 64 64" aria-hidden="true"><rect width="64" height="64" rx="16" fill="#3c6cff"/><path d="M16 0h32c8.8 0 16 7.2 16 16v5C45 8 25 8 0 32V16C0 7.2 7.2 0 16 0Z" fill="#fff" opacity=".16"/><g fill="#f7f6f2"><rect x="8" y="20" width="4.6" height="24" rx="1.2" opacity=".28"/><rect x="14.1" y="20" width="4.6" height="24" rx="1.2" opacity=".38"/><rect x="20.2" y="20" width="4.6" height="24" rx="1.2" opacity=".48"/><rect x="26.3" y="20" width="4.6" height="24" rx="1.2" opacity=".58"/><rect x="32.4" y="20" width="4.6" height="24" rx="1.2" opacity=".68"/><rect x="38.5" y="20" width="4.6" height="24" rx="1.2" opacity=".78"/><rect x="44.6" y="20" width="4.6" height="24" rx="1.2" opacity=".9"/><rect x="50.7" y="20" width="4.6" height="24" rx="1.2" fill="#fff"/></g></svg><span>DVIDIA</span></a><nav class="top-links" aria-label="External resources"><a href="https://github.com/Dvidia-Inference/dvidia-training">Source code ↗</a><a href="https://research.dvidia.org/">Research notes ↗</a></nav></header>
<main id="experiment"><p class="eyebrow">Physical grounding · local verifier v0.3</p><h1>What does “finished” require?</h1><p class="intro">A small CPU verifier checks a declared placement goal against timestamped observations. Inspect the passing measurements, the reasons to refuse completion, and the cases where believable readings still describe the wrong physical state.</p>
<div class="notice" id="run-state"></div><div class="toolbar"><div><label for="cohort">Evidence cohort</label><select id="cohort"><option value="native">Fresh native scenes</option><option value="evidence_fault">Injected evidence faults</option><option value="blind_spot">Consistent fabricated readings</option></select></div><div class="meta">Recorded local CPU evaluation · no model service calls<br>Images illustrate scenes; the verifier reads observation packets.</div></div><div id="cohort-note" class="notice"></div><dl class="metrics" id="metrics"></dl>
<div class="workspace"><aside class="case-nav" aria-label="Benchmark cases"><h2 id="case-count">Cases</h2><label for="filter">Show cases</label><select id="filter"><option value="all">All cases in cohort</option><option value="positive">Physically complete</option><option value="negative">Physically incomplete</option><option value="unknown">Verifier abstained</option><option value="disagree">Disagrees with physical state</option><option value="freshness">Freshness or history fault</option></select><div class="case-list" id="case-list"></div></aside><article id="case-detail" aria-live="polite"></article></div>
<section class="fine"><h2>Every denominator stays visible.</h2><p>Contract agreement measures the expected decision from the supplied evidence. Physical agreement asks whether the decision matches the actual native scene. A fabricated but consistent packet can satisfy the contract while disagreeing with the scene. Repeated timing samples do not increase the number of physical cases.</p><div class="table-wrap"><table><thead><tr><th>Cohort</th><th>Cases</th><th>Contract agreement</th><th>Positive detections</th><th>Negative detections</th><th>False completions</th><th>Abstentions</th><th>CPU p95</th></tr></thead><tbody id="summary-table"></tbody></table></div><h2>Timing and limits</h2><p id="timing-scope"></p><ul id="limits"></ul><p>Fresh simulation measurements are privileged inputs, not a calibrated camera, identity estimator or force sensor. This is a diagnostic verifier, not a robot controller, learned skill or hardware safety qualification. The earlier Jev experiment remains a separate hosted-service measurement.</p><div class="downloads"><a href="results.json">Full recorded results</a><a href="cases.json">Case catalog</a><a href="protocol.json">Protocol</a><a href="publication-integrity.json">Artifact checksums</a><a href="https://research.dvidia.org/papers/local-goal-verifier-v03/">Research article ↗</a><a href="https://dvidia-robot-simulation-lab.static.hf.space/image-goals/v02/index.html">Earlier Jev results ↗</a></div></section><footer><span>DVIDIA research<br><a href="mailto:hello@dvidia.org">hello@dvidia.org</a></span><span><a href="https://x.com/iammrriver">@iammrriver</a> · Open research<br>MIT code · synthetic simulation evidence</span></footer></main>
<script id="public-data" type="application/json">__PUBLIC_DATA__</script><script>
'use strict';const data=JSON.parse(document.getElementById('public-data').textContent), $=id=>document.getElementById(id), names={native:'Fresh native scenes',evidence_fault:'Injected evidence faults',blind_spot:'Consistent fabricated readings'};let selected=null;const el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n}, fraction=(n,d)=>d?`${n} / ${d}`:'— (0 cases)', fmt=n=>n===null||n===undefined?'—':Number(n).toLocaleString(undefined,{maximumFractionDigits:3}), verdict=v=>({complete:'Complete',incomplete:'Incomplete',unknown:'Unknown — abstained'}[v]||v), pretty=s=>s.replaceAll('_',' '), latency=q=>q.n?`${fmt(q.p95)} μs`:'Unmeasured';function fact(dl,label,value){dl.append(el('dt',label),el('dd',value))}function metric(title,value,note){const n=el('div',undefined,'metric');n.append(el('dt',title),el('dd',value),el('small',note));return n}function rows(){return data.cases.filter(c=>c.cohort===$('cohort').value).filter(c=>{const f=$('filter').value;if(f==='positive')return c.physical_truth.complete;if(f==='negative')return !c.physical_truth.complete;if(f==='unknown')return c.decision.verdict==='unknown';if(f==='disagree')return c.decision.verdict!=='unknown'&&((c.decision.verdict==='complete')!==c.physical_truth.complete);if(f==='freshness')return ['freshness','timestamps','history_span'].some(k=>c.decision.predicates[k].status!=='pass'&&c.decision.predicates[k].reason!=='not_evaluated');return true})}
function render(){const group=$('cohort').value,s=data.summaries.find(x=>x.cohort===group),list=rows();if(!list.some(c=>c.id===selected))selected=list[0]?.id||null;$('case-count').textContent=`${list.length} of ${s.planned} cases`;$('case-list').replaceChildren();for(const c of list){const b=el('button',c.id);b.type='button';b.setAttribute('aria-pressed',String(c.id===selected));b.append(el('small',`${pretty(c.scenario)} · ${verdict(c.decision.verdict)}`));b.onclick=()=>{selected=c.id;render()};$('case-list').append(b)}$('metrics').replaceChildren(metric('Recognized physically complete',fraction(s.positive_recognition.correct,s.positive_recognition.total),'All physically complete cases, including abstentions.'),metric('False completion',fraction(s.false_completion.count,s.false_completion.total),'All physically incomplete cases in this cohort.'),metric('Contract agreement',fraction(s.contract_agreement.correct,s.contract_agreement.total),`${s.abstentions} abstentions; ${s.physical_agreement.answered} explicit decisions.`),metric('CPU p95',latency(s.cpu_latency_us),`Requires 200 ms observation history. ${s.cpu_latency_us.n} repeated compute calls.`));const notes={native:'Fresh scenes test physical predicates with privileged simulator observations. Positive and negative recognition are both reported.',evidence_fault:'Faults change or remove supplied evidence while retaining the source scene. Unknown is the expected refusal where the packet cannot support a decision; this is not a failure to see the object.',blind_spot:'These packets are intentionally fabricated but internally consistent. A complete verdict on an incomplete scene exposes the boundary of arithmetic verification: trusted sensing and provenance are still required.'};$('cohort-note').textContent=notes[group];$('cohort-note').className='notice'+(group==='blind_spot'?' warning':'');detail(list.find(c=>c.id===selected));}
function detail(c){const target=$('case-detail');target.replaceChildren();if(!c){target.append(el('p','No cases match this filter.','empty'));return}const heading=el('div',undefined,'case-heading');heading.append(el('h2',pretty(c.scenario)),el('span',c.id,'meta'));target.append(heading);const images=el('div',undefined,'images');for(const [key,label]of[['current_image','Current native scene'],['goal_image','Reference placement goal']]){const fig=el('figure'),im=el('img');im.src=c[key];im.alt=`${label}: ${pretty(c.scenario)} (${c.id})`;im.width=640;im.height=480;fig.append(im,el('figcaption',label));images.append(fig)}target.append(images,el('p',c.note,'note'));const compare=el('div',undefined,'comparison'),left=el('section'),right=el('section'),lf=el('dl',undefined,'facts'),rf=el('dl',undefined,'facts');left.append(el('h3','Local verdict'),lf);fact(lf,'Completion decision',verdict(c.decision.verdict));fact(lf,'Reasons',c.decision.reasons.length?c.decision.reasons.map(pretty).join('; '):'Every required predicate passed');fact(lf,'Expected from supplied evidence',verdict(c.expected_verdict));fact(lf,'Physically complete',c.physical_truth.complete?'Yes':'No');fact(lf,'Target actually present',c.physical_truth.present?'Yes':'No');right.append(el('h3','Observation history'),rf);fact(rf,'Trusted clock (s)',fmt(c.now_s));fact(rf,'Newest reading (s)',fmt(c.decision.window.newest_t_s));fact(rf,'Age (s)',fmt(c.decision.window.age_s));fact(rf,'Scored samples',fmt(c.decision.window.evaluated_sample_count));fact(rf,'Scored interval (s)',fmt(c.decision.window.evaluated_duration_s));fact(rf,'Generation validated',c.generation_valid?'Yes':'No');compare.append(left,right);target.append(compare);const wrap=el('div',undefined,'table-wrap'),table=el('table',undefined,'predicates'),thead=el('thead'),tr=el('tr');for(const title of['Predicate','State','Measurement / reason'])tr.append(el('th',title));thead.append(tr);table.append(thead);const body=el('tbody');for(const[key,p]of Object.entries(c.decision.predicates)){const r=el('tr'),state=el('td');state.append(el('span',p.status,'badge '+p.status));const note=p.observed!==undefined?`${p.statistic} ${fmt(p.observed)}; interval [${fmt(p.interval?.lower)}, ${fmt(p.interval?.upper)}] ${p.operator} ${fmt(p.threshold)} · ${pretty(p.reason)}`:(p.detail||pretty(p.reason));r.append(el('td',pretty(key)),state,el('td',note));body.append(r)}table.append(body);wrap.append(table);target.append(wrap);const links=el('div',undefined,'downloads');for(const[key,label]of[['packet_file','Exact observation packet'],['state_file','Native state evidence']])if(c[key]){const a=el('a',label);a.href=c[key];links.append(a)}target.append(links)}
$('run-state').textContent=`${data.cases.length} cases evaluated locally across three separately reported cohorts. Zero Jev calls. This release checks timestamped measurements; it does not infer physical state from the displayed images.`;$('timing-scope').textContent=data.timing_scope;for(const limit of data.limits)$('limits').append(el('li',limit));for(const s of data.summaries){const r=el('tr');for(const value of[names[s.cohort],s.planned,fraction(s.contract_agreement.correct,s.contract_agreement.total),fraction(s.positive_recognition.correct,s.positive_recognition.total),fraction(s.negative_recognition.correct,s.negative_recognition.total),fraction(s.false_completion.count,s.false_completion.total),s.abstentions,latency(s.cpu_latency_us)])r.append(el('td',value));$('summary-table').append(r)}$('cohort').onchange=()=>{selected=null;$('filter').value='all';render()};$('filter').onchange=render;render();
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_report(args.run, args.output), indent=2))


if __name__ == "__main__":
    main()

"""Build a portable, credential-free viewer for an image-goal experiment.

Only the explicit public fields below are exported. Provider payloads, headers,
raw responses and arbitrary free-form errors never enter the report. The viewer
does not call Jev, issue robot commands or turn an unrun experiment into a score.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}\Z")
_SUCCESS = {"ok", "success", "completed"}
_PENDING = {"pending", "not_run", "not-run", "unrun", "missing_api_key", "missing_credentials"}
_MAX_JSON = 8 * 1024 * 1024
_MAX_IMAGE = 4 * 1024 * 1024


def _json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_JSON:
        raise ValueError("Expected a bounded, plain experiment JSON file.")
    value = json.loads(path.read_bytes(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON.")))
    if not isinstance(value, dict):
        raise ValueError("Experiment JSON must be an object.")
    return value


def _name(value, label):
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError(f"Invalid public {label}.")
    return value


def _number(value, probability=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    if value < 0 or (probability and value > 1):
        return None
    return value


def _boolean(value):
    return value if type(value) is bool else None


def _image(root, name, assets):
    if not isinstance(name, str) or "\\" in name:
        raise ValueError("Image reference must be a relative artifact path.")
    relative = Path(name)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("Image reference must remain inside the experiment.")
    path = root / relative
    if any(item.is_symlink() for item in (path, *path.parents)) or not path.is_file():
        raise ValueError("Image reference must be a plain local file.")
    if path.stat().st_size > _MAX_IMAGE:
        raise ValueError("Report image exceeds its byte bound.")
    raw = path.read_bytes()
    suffix = ".png" if raw.startswith(b"\x89PNG\r\n\x1a\n") else ".jpg" if raw.startswith(b"\xff\xd8\xff") else None
    if suffix is None:
        raise ValueError("Report images must be PNG or JPEG.")
    public = "images/" + sha256(raw).hexdigest() + suffix
    assets[public] = raw
    return public


def _decisions(row):
    """Read explicit decisions only; probabilities do not imply a threshold."""
    decision = row.get("decision", {})
    if not isinstance(decision, dict):
        decision = {}
    fallbacks = {"present": {"present": True, "absent": False}.get(row.get("presence")),
                 "complete": {"complete": True, "incomplete": False}.get(row.get("verdict"))}
    return {key: _boolean(decision.get(key, row.get("predicted_" + key, fallbacks[key]))) for key in ("present", "complete")}


def _quantiles(values):
    if not values:
        return {"n": 0, "p50": None, "p95": None, "p99": None, "max": None}
    values = sorted(values)
    def percentile(p):
        offset = (len(values) - 1) * p
        low, high = math.floor(offset), math.ceil(offset)
        return values[low] + (values[high] - values[low]) * (offset - low)
    return {"n": len(values), "p50": percentile(.5), "p95": percentile(.95), "p99": percentile(.99), "max": values[-1]}


def _summary(cases, rows, conditions):
    truth = {case["id"]: case["ground_truth"] for case in cases}
    case_by_id = {case["id"]: case for case in cases}
    summaries = []
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition]
        attempted = [row for row in selected if row["status"] not in _PENDING]
        answered = [row for row in selected if row["status"] in _SUCCESS]
        complete_answers = [row for row in answered if row["decision"]["complete"] is not None and truth[row["case_id"]]["complete"] is not None]
        incomplete = [case for case in cases if case["ground_truth"]["complete"] is False]
        complete = [case for case in cases if case["ground_truth"]["complete"] is True]
        absent = [case for case in cases if case["ground_truth"]["present"] is False]
        incomplete_answers = [row for row in complete_answers if truth[row["case_id"]]["complete"] is False]
        absent_answers = [row for row in answered if truth[row["case_id"]]["present"] is False and row["decision"]["present"] is not None]
        false_complete = sum(row["decision"]["complete"] is True for row in incomplete_answers)
        positive_answers = [row for row in complete_answers if truth[row["case_id"]]["complete"] is True]
        false_present = sum(row["decision"]["present"] is True for row in absent_answers)
        known = [case for case in cases if case["expected_verdict"].get(condition) in {"complete", "incomplete"}]
        ambiguous = [case for case in cases if case["expected_verdict"].get(condition) == "unknown"]
        expected_cases = [case for case in cases if case["expected_verdict"].get(condition) is not None]
        expected_correct = sum(row["verdict"] == case_by_id[row["case_id"]]["expected_verdict"].get(condition)
            for row in answered if case_by_id[row["case_id"]]["expected_verdict"].get(condition) is not None)
        known_correct = sum(row["verdict"] == case_by_id[row["case_id"]]["expected_verdict"].get(condition)
                            for row in answered if case_by_id[row["case_id"]]["expected_verdict"].get(condition) in {"complete", "incomplete"})
        justified_abstentions = sum(row["verdict"] == "unknown" for row in answered
            if case_by_id[row["case_id"]]["expected_verdict"].get(condition) == "unknown")
        durations = lambda group: [row["timing_ms"]["total_ms"] for row in group if row["timing_ms"].get("total_ms") is not None]
        summaries.append({"condition": condition, "planned": len(cases), "attempted": len(attempted),
            "responses": len(answered), "errors": len(attempted) - len(answered),
            "not_run": len(cases) - len(attempted),
            "completion_accuracy": {"correct": sum(row["decision"]["complete"] == truth[row["case_id"]]["complete"] for row in complete_answers), "answered": len(complete_answers), "planned": sum(case["ground_truth"]["complete"] is not None for case in cases)},
            "completion_abstentions": sum(row["decision"]["complete"] is None for row in answered),
            "observability_aware_accuracy": {"correct": expected_correct, "planned": len(expected_cases)},
            "known_verdict_accuracy": {"correct": known_correct, "planned": len(known)},
            "ambiguous_abstention": {"correct": justified_abstentions, "planned": len(ambiguous)},
            "false_completion": {"count": false_complete, "answered_incomplete": len(incomplete_answers), "planned_incomplete": len(incomplete),
                                 "valid_incomplete": sum(truth[row["case_id"]]["complete"] is False for row in answered)},
            "true_completion": {"count": sum(row["decision"]["complete"] is True for row in positive_answers),
                                "planned_complete": len(complete), "answered_complete": len(positive_answers),
                                "valid_complete": sum(truth[row["case_id"]]["complete"] is True for row in answered)},
            "false_presence": {"count": false_present, "answered_absent": len(absent_answers), "planned_absent": len(absent),
                               "valid_absent": sum(truth[row["case_id"]]["present"] is False for row in answered)},
            "known_usage_estimated_input_cost_usd": sum(row["estimated_input_cost_usd"] for row in attempted if row["estimated_input_cost_usd"] is not None)
                if any(row["estimated_input_cost_usd"] is not None for row in attempted) else None,
            "unknown_cost_attempts": sum(row["estimated_input_cost_usd"] is None for row in attempted),
            "all_attempt_latency_ms": _quantiles(durations(attempted)),
            "successful_response_latency_ms": _quantiles(durations(answered))})
    return summaries


def build_report(run_dir, output_dir, *, results_path=None):
    """Export an offline report into a fresh directory and return its manifest.

    The public data contract is deliberately small: cases contain id, category,
    current_image, goal_image and physical_truth.{present,complete}; result rows
    contain case_id, condition, status, explicit decision booleans, probabilities,
    verdict and finite timing_ms values. Extra input fields are discarded.
    """
    root, output = Path(run_dir).absolute(), Path(output_dir).absolute()
    if any(path.is_symlink() for path in (root, *root.parents, output, *output.parents)):
        raise ValueError("Report paths must not contain symlinks.")
    if output.exists():
        raise ValueError("Report output must be a fresh directory.")
    catalog = _json(root / "cases.json")
    result_path = Path(results_path).absolute() if results_path is not None else root / "results.json"
    if any(path.is_symlink() for path in (result_path, *result_path.parents)):
        raise ValueError("Results path must not contain symlinks.")
    if results_path is not None and not result_path.is_file():
        raise ValueError("Explicit results path does not exist.")
    result = _json(result_path) if result_path.exists() else {"status": "not_run", "results": []}
    if result.get("status") == "running" or result.get("active_trial") is not None:
        raise ValueError("An active or interrupted request must be resolved before exporting a report.")
    source_cases = catalog.get("cases")
    if not isinstance(source_cases, list) or not 1 <= len(source_cases) <= 128:
        raise ValueError("A report requires one to 128 explicit cases.")
    cases, assets = [], {}
    for case in source_cases:
        ground_truth = case.get("physical_truth", case.get("ground_truth", {}))
        if not isinstance(ground_truth, dict):
            raise ValueError("Ground truth must be an object.")
        cases.append({"id": _name(case.get("id", case.get("case_id")), "case id"),
            "category": _name(case.get("category", "unspecified"), "case category"),
            "current_image": _image(root, case.get("current_image"), assets),
            "goal_image": _image(root, case.get("goal_image"), assets),
            "ground_truth": {key: _boolean(ground_truth.get(key)) for key in ("present", "complete")},
            "expected_verdict": {key: value for key, value in case.get("expected_verdict", {}).items()
                                 if key in {"rgb", "rgb_telemetry"} and value in {"complete", "incomplete", "unknown"}},
            "expected_presence": {key: value for key, value in case.get("expected_presence", {}).items()
                                  if key in {"rgb", "rgb_telemetry"} and value in {"present", "absent", "unknown"}}})
    ids = {case["id"] for case in cases}
    if len(ids) != len(cases):
        raise ValueError("Case ids must be unique.")
    raw_rows = result.get("results", result.get("rows", []))
    if not isinstance(raw_rows, list) or len(raw_rows) > 1024:
        raise ValueError("Expected a bounded list of result rows.")
    declared = result.get("planned_conditions", catalog.get("conditions", ["rgb", "rgb_telemetry"]))
    if not isinstance(declared, list) or not 1 <= len(declared) <= 8:
        raise ValueError("Declare one to eight input conditions.")
    conditions = [_name(value, "condition") for value in declared]
    if len(set(conditions)) != len(conditions):
        raise ValueError("Conditions must be unique.")
    rows, seen = [], set()
    for row in raw_rows:
        case_id, condition = _name(row.get("case_id"), "case id"), _name(row.get("condition"), "condition")
        if case_id not in ids or condition not in conditions or (case_id, condition) in seen:
            raise ValueError("Every result must name one unique planned case and condition.")
        seen.add((case_id, condition))
        timing = row.get("timing_ms", {})
        if isinstance(timing, (int, float)):
            timing = {"total_ms": timing}
        if not isinstance(timing, dict):
            timing = {}
        # Restrict names as well as values: metadata is not a covert error log.
        public_timing = {key: _number(timing.get(key)) for key in ("total_ms", "prepare_ms", "request_ms", "capture_ms", "encode_ms", "upload_ms", "decode_ms") if key in timing}
        verdict = row.get("verdict")
        if verdict not in {"complete", "not_complete", "incomplete", "abstain", "unknown", "missing", "present", "error", None}:
            verdict = "unknown"
        status = _name(row.get("status", "unknown"), "status")
        pending = status in _PENDING
        rows.append({"case_id": case_id, "condition": condition,
            "status": status, "decision": {"present": None, "complete": None} if pending else _decisions(row),
            "p_present": None if pending else _number(row.get("p_present"), True),
            "p_complete": None if pending else _number(row.get("p_complete"), True),
            "verdict": None if pending else verdict, "timing_ms": {} if pending else public_timing,
            "estimated_input_cost_usd": None if pending else _number(row.get("estimated_input_cost_usd"))})
    data = {"schema": "dvidia.image-goal-public-report/v1", "status": _name(result.get("status", "not_run"), "run status"),
        "model": _name(result.get("model", "jev-1.14.0"), "model"),
        "conditions": conditions, "cases": cases, "results": rows,
        "summary": _summary(cases, rows, conditions),
        "scope": "Synthetic simulation fixtures; passive image-goal judgments; no robot control or trained weights."}
    encoded = json.dumps(data, indent=2, allow_nan=False).encode() + b"\n"
    embedded = json.dumps(data, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    html = _HTML.replace("__PUBLIC_DATA__", embedded).encode()
    payload = {"index.html": html, "report.json": encoded, **assets}
    manifest = {"schema": "dvidia.image-goal-report-files/v1", "files": {name: {"bytes": len(raw), "sha256": sha256(raw).hexdigest()} for name, raw in payload.items()}}
    output.mkdir(parents=True)
    for name, raw in payload.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    (output / "report-integrity.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>Image goals · DVIDIA Research</title>
<style>
:root{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#26302d;background:#faf9f6;font-synthesis:none}*{box-sizing:border-box}body{margin:0;background-image:radial-gradient(#28332c09 .55px,transparent .65px);background-size:5px 5px}a{color:inherit;text-underline-offset:4px}button,select{font:inherit}button,select,a{touch-action:manipulation}button,select{min-height:44px}button:focus-visible,select:focus-visible,a:focus-visible{outline:3px solid #3c6cff;outline-offset:3px}.skip{position:absolute;left:20px;top:-60px;padding:12px;background:white;z-index:10}.skip:focus{top:8px}.masthead{min-height:84px;padding:20px 5vw;border-bottom:1px solid #dedfd6;display:flex;align-items:center;justify-content:space-between;gap:20px}.brand{display:flex;gap:11px;align-items:center;text-decoration:none;font-weight:750;font-size:20px;letter-spacing:2px}.brand svg{width:34px;height:34px}.top-links{font-size:13px;display:flex;gap:22px}main{max-width:1400px;margin:auto;padding:46px 5vw 36px}.eyebrow{font-size:11px;letter-spacing:2px;text-transform:uppercase;color:#69726b;font-weight:650}h1,h2,h3{font-family:Georgia,"Times New Roman",serif;font-weight:400}h1{font-size:clamp(38px,5vw,64px);letter-spacing:-2px;margin:15px 0 18px;line-height:1.05}.intro{max-width:760px;color:#59665f;line-height:1.8;font-size:16px}.run-state{padding:16px 20px;border:1px solid #d7ded2;background:#f1f3ec;margin:28px 0;font-size:14px;line-height:1.6}.toolbar{display:flex;justify-content:space-between;align-items:end;gap:22px;margin:28px 0 20px}.toolbar label{display:block;font-size:12px;color:#5b6b60;margin-bottom:7px}select{border:1px solid #cfd6cb;border-radius:5px;background:#fffefb;color:inherit;padding:10px 34px 10px 12px;max-width:100%}.meta{font-size:12px;color:#637166;line-height:1.6}.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border-block:1px solid #dcded5;margin:18px 0 30px}.metric{padding:19px 20px 21px 0}.metric+.metric{padding-left:20px;border-left:1px solid #dcded5}.metric dt{font-size:12px;color:#627064;line-height:1.4}.metric dd{font:29px Georgia,serif;margin:10px 0 7px}.metric small{font-size:11px;color:#637166;line-height:1.5;display:block}.workspace{display:grid;grid-template-columns:230px minmax(0,1fr);gap:28px}.case-nav{border-top:1px solid #dcded5}.case-nav h2{font:12px system-ui;letter-spacing:1px;text-transform:uppercase;margin:16px 0}.filter{width:100%;font-size:12px;margin-bottom:12px}.case-list{display:grid;gap:4px;max-height:690px;overflow:auto;padding:2px}.case-list button{text-align:left;border:0;background:transparent;padding:11px 10px;color:#516057;border-radius:4px;cursor:pointer;font-size:12px;line-height:1.5}.case-list button[aria-pressed=true]{background:#e9eee4;color:#263a2a;box-shadow:inset 3px 0 #5f775d}.case-list small{display:block;color:#6d786d;margin-top:3px}.case-list .error{color:#924733}.case-heading{display:flex;justify-content:space-between;align-items:baseline;gap:15px}.case-heading h2{margin:0 0 16px;font-size:25px}.image-pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0;min-width:0}figure img{width:100%;height:auto;aspect-ratio:4/3;object-fit:contain;display:block;background:#e9e9df;border:1px solid #ddded4}figcaption{font-size:12px;color:#5c6a5e;margin:10px 0 16px}.comparison{display:grid;grid-template-columns:1fr 1fr;gap:16px;border-top:1px solid #dbdfd3;padding-top:19px;margin-top:6px}.comparison h3{font-size:20px;margin:0 0 14px}.facts{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:11px;font-size:12px;margin:0}.facts dt{color:#617064}.facts dd{margin:0;text-align:right;font-variant-numeric:tabular-nums}.note{font-size:12px;color:#657264;line-height:1.75}.notice{background:#f2f2e9;padding:13px 15px;border-left:3px solid #a8b398;margin:18px 0;font-size:13px;line-height:1.6}.fine{margin-top:34px;padding-top:24px;border-top:1px solid #dcded5;max-width:950px}.fine h2{font-size:25px}.fine p{font-size:13px;line-height:1.8;color:#5b695e}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}th,td{text-align:left;padding:10px 14px 10px 0;border-bottom:1px solid #e0e2d9}th{font-weight:550;color:#637165}.downloads{display:flex;flex-wrap:wrap;gap:14px;font-size:13px;margin-top:22px}footer{border-top:1px solid #dcdfd5;margin-top:32px;padding-top:23px;font-size:12px;color:#657264;display:flex;justify-content:space-between;gap:22px;line-height:1.8}.empty{padding:25px;color:#657264;font-size:13px}@media(max-width:900px){.workspace{grid-template-columns:190px minmax(0,1fr);gap:18px}.image-pair{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.metric:nth-child(3){border-left:0;padding-left:0}.metric:nth-child(n+3){border-top:1px solid #dcded5}}@media(max-width:620px){main{padding:30px 18px}.masthead{padding:16px 18px;min-height:74px}.top-links a:first-child{display:none}.toolbar{display:block}.toolbar .meta{margin-top:14px}.workspace{grid-template-columns:1fr}.case-list{grid-template-columns:repeat(2,minmax(0,1fr));max-height:195px}.case-nav{margin-bottom:14px}.comparison{grid-template-columns:1fr}.comparison>section+section{border-top:1px solid #dcded5;padding-top:17px}.case-heading{display:block}.metric dd{font-size:26px}footer{display:block}footer p{margin:0 0 12px}.intro{font-size:15px}h1{letter-spacing:-1px}}
</style></head><body><a class="skip" href="#experiment">Skip to experiment</a><header class="masthead"><a class="brand" href="https://dvidia.org/" aria-label="DVIDIA home"><svg viewBox="0 0 64 64" aria-hidden="true"><rect width="64" height="64" rx="16" fill="#3c6cff"/><path d="M16 0h32c8.8 0 16 7.2 16 16v5C45 8 25 8 0 32V16C0 7.2 7.2 0 16 0Z" fill="#fff" opacity=".16"/><g fill="#f7f6f2"><rect x="8" y="20" width="4.6" height="24" rx="1.2" opacity=".28"/><rect x="14.1" y="20" width="4.6" height="24" rx="1.2" opacity=".38"/><rect x="20.2" y="20" width="4.6" height="24" rx="1.2" opacity=".48"/><rect x="26.3" y="20" width="4.6" height="24" rx="1.2" opacity=".58"/><rect x="32.4" y="20" width="4.6" height="24" rx="1.2" opacity=".68"/><rect x="38.5" y="20" width="4.6" height="24" rx="1.2" opacity=".78"/><rect x="44.6" y="20" width="4.6" height="24" rx="1.2" opacity=".9"/><rect x="50.7" y="20" width="4.6" height="24" rx="1.2" fill="#fff"/></g></svg><span>DVIDIA</span></a><nav class="top-links" aria-label="External resources"><a href="https://github.com/Dvidia-Inference/dvidia-training">Source code ↗</a><a href="https://research.dvidia.org/">Research notes ↗</a></nav></header>
<main id="experiment"><p class="eyebrow">Robotics research · Image goals / 0.1</p><h1>Does it only look finished?</h1><p class="intro">A goal image describes the destination. This experiment asks Jev to judge the current scene, then compares its answer with the simulator’s physical state. Missing shoes and misleading arrangements stay in the results.</p><div id="run-state" class="run-state" role="status">Loading the recorded experiment.</div><noscript><p class="notice">Enable JavaScript to inspect individual cases. The complete public report remains available as JSON below.</p></noscript><div class="toolbar"><div><label for="condition">Model input</label><select id="condition"></select></div><p class="meta" id="run-meta"></p></div><dl id="metrics" class="metrics"></dl>
<div class="workspace"><aside class="case-nav" aria-label="Benchmark cases"><h2>Inspect every case</h2><label class="meta" for="filter">Show</label><select id="filter" class="filter"><option value="all">All cases</option><option value="absent">Missing target</option><option value="incomplete">Physically incomplete</option><option value="complete">Physically complete</option><option value="false_complete">False completion</option><option value="error">Request errors</option><option value="not_run">Not run</option></select><div id="case-list" class="case-list"></div></aside><section aria-labelledby="case-title"><div class="case-heading"><h2 id="case-title"></h2><span id="case-category" class="meta"></span></div><div class="image-pair"><figure><img id="current" alt="Current simulated scene"><figcaption>Current observation</figcaption></figure><figure><img id="goal" alt="Requested goal arrangement"><figcaption>Goal reference</figcaption></figure></div><p id="case-notice" class="notice"></p><div class="comparison"><section><h3>Physical reference</h3><dl id="truth" class="facts"></dl><p class="note">Recorded simulator labels. A single image may not reveal hidden contact or sustained support.</p></section><section><h3>Jev judgment</h3><dl id="prediction" class="facts"></dl><p class="note">A remote research judgment. This page sends no images, runs no model and commands no robot.</p></section></div></section></div>
<section class="fine"><h2>Count the misses. Keep the clock honest.</h2><p>Judgment accuracy uses the frozen, observability-aware verdict for every planned case: complete, incomplete, or unknown. A correct abstention matters when an image cannot establish hidden motion, contact or dwell. Errors and cases not run remain in the planned denominator; no score is claimed before requests are attempted. Physical false-completion and missing-target counts retain every planned negative case, with answered coverage shown separately.</p><div class="table-wrap"><table><thead><tr><th>Decision check</th><th>Correct</th><th>Denominator</th></tr></thead><tbody id="decisions"></tbody></table></div><div class="table-wrap"><table><thead><tr><th>Latency scope</th><th>Timed requests</th><th>p50</th><th>p95</th><th>p99</th><th>Maximum</th></tr></thead><tbody id="latency"></tbody></table></div><p id="cost" class="note"></p><p>Timing is recorded wall-clock latency, not the robot’s reaction time or simulator timestep. Quantiles use linear interpolation and retain timed errors in the all-attempt row. A small cohort cannot establish production tail latency. Missing measurements remain unreported.</p><p>This benchmark uses synthetic simulation fixtures and a hosted decision model. It does not publish trained robot weights or demonstrate hardware transfer. The DVIDIA code license does not grant rights to the DVIDIA marks or to an external model.</p><div class="downloads"><a href="report.json" download>Public results · JSON</a><a href="report-integrity.json" download>File checksums</a><a href="https://docs.typesafe.ai/images">Jev image documentation ↗</a></div></section><footer><p>Published by DVIDIA · <a href="mailto:hello@dvidia.org">hello@dvidia.org</a><br>Original report code: MIT. External models retain their own terms.</p><p><a href="https://huggingface.co/spaces/Dvidia/robot-simulation-lab">Hugging Face</a> · <a href="https://x.com/iammrriver">@iammrriver</a></p></footer></main>
<script id="public-data" type="application/json">__PUBLIC_DATA__</script><script>
"use strict";
const data=JSON.parse(document.getElementById("public-data").textContent),byId=id=>document.getElementById(id),human=value=>String(value).replaceAll("_"," ").replaceAll("-"," "),labelCondition=value=>({rgb:"Images only",rgb_telemetry:"Images + measured state"}[value]||human(value)),success=new Set(["ok","success","completed"]),pending=new Set(["pending","not_run","not-run","unrun","missing_api_key","missing_credentials"]);let selected=data.cases[0].id;
const text=(id,value)=>{byId(id).textContent=value},bool=value=>value===true?"Yes":value===false?"No":"Unreported",prob=value=>value===null?"Unreported":(value*100).toFixed(1)+"%",ms=value=>value===null?"Unreported":value.toFixed(1)+" ms",rowFor=(id,condition)=>data.results.find(row=>row.case_id===id&&row.condition===condition),condition=()=>byId("condition").value;
for(const value of data.conditions){const option=document.createElement("option");option.value=value;option.textContent=labelCondition(value);byId("condition").append(option)}
function pairs(id,values){const node=byId(id);node.replaceChildren();for(const [key,value] of values){const dt=document.createElement("dt"),dd=document.createElement("dd");dt.textContent=key;dd.textContent=value;node.append(dt,dd)}}
function metric(title,value,detail){const wrap=document.createElement("div");wrap.className="metric";const dt=document.createElement("dt"),dd=document.createElement("dd"),small=document.createElement("small");dt.textContent=title;dd.textContent=value;small.textContent=detail;wrap.append(dt,dd,small);return wrap}
function renderSummary(){
const summary=data.summary.find(row=>row.condition===condition()),a=summary.observability_aware_accuracy,f=summary.false_completion,p=summary.false_presence;
byId("metrics").replaceChildren(
metric("Observability-aware judgments",summary.attempted&&a.planned?`${a.correct} / ${a.planned}`:"Not measured",`${summary.responses} responses · ${summary.completion_abstentions} abstentions · ${summary.errors} errors · ${summary.not_run} not run`),
metric("Physical false completion",summary.attempted&&f.planned_incomplete?`${f.count} / ${f.planned_incomplete}`:"Not measured",`${f.count}/${f.valid_incomplete} valid responses · ${f.answered_incomplete}/${f.planned_incomplete} explicit-decision coverage`),
metric("False target presence",summary.attempted&&p.planned_absent?`${p.count} / ${p.planned_absent}`:"Not measured",`${p.count}/${p.valid_absent} valid responses · ${p.answered_absent}/${p.planned_absent} explicit-decision coverage`),
metric("Response latency · p95",ms(summary.successful_response_latency_ms.p95),`${summary.successful_response_latency_ms.n} timed successful responses`));
text("run-meta",`${data.model} · ${data.cases.length} synthetic cases · ${summary.attempted}/${summary.planned} attempted in this condition`);
const total=data.summary.reduce((n,s)=>n+s.attempted,0);text("run-state",total?`Recorded benchmark · ${total} requests attempted. Every planned case remains visible, including errors and missing responses. Physically complete cases recognized in this input condition: ${summary.true_completion.count}/${summary.true_completion.planned_complete}.`:"Jev has not been run. These are prepared simulation cases; model accuracy and service latency have not been measured.");
const decisionBody=byId("decisions");decisionBody.replaceChildren();for(const [name,count,denominator,scope] of [["Physical completion detected",summary.true_completion.count,summary.true_completion.planned_complete,"all physically complete cases, including abstentions and errors"],["Known verdicts",summary.known_verdict_accuracy.correct,summary.known_verdict_accuracy.planned,"planned cases with an observable verdict"],["Justified abstention",summary.ambiguous_abstention.correct,summary.ambiguous_abstention.planned,"planned ambiguous cases"],["Selective physical agreement",summary.completion_accuracy.correct,summary.completion_accuracy.answered,"explicit answered decisions"]]){const tr=document.createElement("tr");for(const value of [name,summary.attempted&&denominator?`${count} / ${denominator}`:"Not measured",scope]){const td=document.createElement("td");td.textContent=value;tr.append(td)}decisionBody.append(tr)}
text("cost",`Known-usage estimated input cost: ${summary.known_usage_estimated_input_cost_usd===null?"unreported":"$"+summary.known_usage_estimated_input_cost_usd.toFixed(6)}. ${summary.unknown_cost_attempts} attempted requests have unknown cost. This estimate is not a billing receipt.`);
const tbody=byId("latency");tbody.replaceChildren();for(const [name,values] of [["All timed attempts",summary.all_attempt_latency_ms],["Successful responses",summary.successful_response_latency_ms]]){const tr=document.createElement("tr");for(const value of [name,values.n,ms(values.p50),ms(values.p95),ms(values.p99),ms(values.max)]){const td=document.createElement("td");td.textContent=value;tr.append(td)}tbody.append(tr)}}
function visible(caseRow){const row=rowFor(caseRow.id,condition()),filter=byId("filter").value;return filter==="all"||filter==="absent"&&caseRow.ground_truth.present===false||filter==="incomplete"&&caseRow.ground_truth.complete===false||filter==="complete"&&caseRow.ground_truth.complete===true||filter==="false_complete"&&caseRow.ground_truth.complete===false&&row&&success.has(row.status)&&row.decision.complete===true||filter==="error"&&row&&!success.has(row.status)&&!pending.has(row.status)||filter==="not_run"&&(!row||pending.has(row.status))}
function renderCases(){const shown=data.cases.filter(visible),list=byId("case-list");list.replaceChildren();if(!shown.length){const empty=document.createElement("p");empty.className="empty";empty.textContent="No cases match this filter.";list.append(empty)}if(shown.length&&!shown.some(row=>row.id===selected))selected=shown[0].id;for(const caseRow of shown){const row=rowFor(caseRow.id,condition()),button=document.createElement("button"),small=document.createElement("small");button.type="button";button.setAttribute("aria-pressed",String(caseRow.id===selected));button.textContent=human(caseRow.id);small.textContent=row?human(row.status):"Not run";if(row&&!success.has(row.status)&&!pending.has(row.status))small.className="error";button.append(small);button.addEventListener("click",()=>{selected=caseRow.id;renderCases()});list.append(button)}renderCase()}
function renderCase(){
const current=data.cases.find(row=>row.id===selected),row=rowFor(current.id,condition()),expected=current.expected_verdict[condition()],measured=row&&success.has(row.status);
text("case-title",human(current.id));text("case-category",human(current.category));byId("current").src=current.current_image;byId("goal").src=current.goal_image;byId("current").alt=`Current simulated scene for ${human(current.id)}`;byId("goal").alt=`Goal reference for ${human(current.id)}`;
pairs("truth",[["Target present",bool(current.ground_truth.present)],["Physically complete",bool(current.ground_truth.complete)],["Expected from this input",expected?human(expected):"Unreported"]]);
pairs("prediction",[["Request status",row?human(row.status):"Not run"],["Target present",measured?bool(row.decision.present):"Unmeasured"],["Completion verdict",measured&&row.verdict?human(row.verdict):"Unmeasured"],["Presence probability",measured?prob(row.p_present):"Unmeasured"],["Completion probability",measured?prob(row.p_complete):"Unmeasured"],["Total latency",row&&!pending.has(row.status)?ms(row.timing_ms.total_ms??null):"Unmeasured"]]);
let message="No Jev result is recorded for this case and input condition.";
if(row&&success.has(row.status)){
if(expected)message=row.verdict===expected?"The model judgment agrees with the frozen expected verdict for this input.":"Judgment disagreement: the model did not return the frozen expected verdict for this input.";
else message=row.decision.complete===null?"The model did not make an explicit completion decision.":current.ground_truth.complete===null?"No physical completion reference was reported.":row.decision.complete!==current.ground_truth.complete?"Completion disagreement: the model judgment does not match the physical reference.":"Completion judgment agrees with the physical reference for this case.";
if(expected==="unknown")message+=" This image does not establish every physical requirement; unknown is the expected answer.";
if(current.ground_truth.complete===false&&row.decision.complete===true)message+=" False completion: the physical task is incomplete.";
if(current.ground_truth.present===false&&row.decision.present===true)message+=" The model reported a target that is absent.";
}else if(row&&!pending.has(row.status))message="The request failed. This case remains in the planned cohort and error count.";
text("case-notice",message)}
byId("condition").addEventListener("change",()=>{renderSummary();renderCases()});byId("filter").addEventListener("change",renderCases);renderSummary();renderCases();
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    result = build_report(args.run_dir, args.output_dir, results_path=args.results)
    print(json.dumps({"files": len(result["files"]), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()

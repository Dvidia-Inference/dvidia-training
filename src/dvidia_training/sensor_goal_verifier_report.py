"""Portable paired report for RGB/drive corroboration of a local goal verdict.

This module is separate from all earlier published renderers. The camera's
input frames and the actuator history are evidence, while physical truth is a
separate native reference. No model service or robot commands are used.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re

COHORTS = ("native", "primary_forgery", "sensor_fault", "cross_channel")
VERDICTS = {"complete", "incomplete", "unknown"}
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z")
PRIVATE = {"authorization", "api_key", "apikey", "credential", "credentials", "token", "headers", "raw_response", "response"}
STAGES = ("input_json_io", "png_io_decode", "baseline", "camera_observation", "actuator_observation", "fusion", "processed_window")


def _plain(path):
    path = Path(path).absolute()
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("Report paths must not contain symlinks.")
    return path


def _json(path):
    path = _plain(path)
    if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Expected a bounded plain JSON file.")
    value = json.loads(path.read_bytes(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON.")))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return value


def _encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def _name(value):
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise ValueError("Invalid bounded identifier.")
    return value


def _number(value):
    if type(value) not in (int, float) or not -1e100 <= value <= 1e100 or not math.isfinite(value):
        raise ValueError("Expected a finite number.")
    return value


def _public(value, depth=0):
    """Bound selected observer objects; reject private fields at every depth."""
    if depth > 12:
        raise ValueError("Public evidence is too deeply nested.")
    if value is None or type(value) is bool:
        return value
    if type(value) in (int, float):
        return _number(value)
    if isinstance(value, str):
        if len(value) > 4000:
            raise ValueError("Public text exceeds its bound.")
        return value
    if isinstance(value, list):
        if len(value) > 2048:
            raise ValueError("Public evidence list exceeds its bound.")
        return [_public(item, depth+1) for item in value]
    if isinstance(value, dict):
        if len(value) > 100 or any(not isinstance(key, str) or not NAME.fullmatch(key) or key.lower() in PRIVATE for key in value):
            raise ValueError("Private or invalid public evidence field.")
        return {key: _public(item, depth+1) for key, item in value.items()}
    raise ValueError("Unsupported public evidence type.")


def _relative(root, value):
    if (not isinstance(value, str) or not value or "\\" in value or value.startswith("/")
            or any(not NAME.fullmatch(part) or part in {".", ".."} for part in value.split("/"))):
        raise ValueError("Expected a bounded relative evidence path.")
    if not _plain(root / value).is_file():
        raise ValueError("Referenced evidence is missing.")
    return value


def _image(root, value, assets):
    name = _relative(root, value)
    path = root / name
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Image exceeds its publication bound.")
    raw = path.read_bytes()
    if not raw.startswith(b"\x89PNG\r\n\x1a\n") or not name.endswith(".png"):
        raise ValueError("Expected a PNG evidence frame.")
    assets[name] = raw
    return name


def quantiles(values):
    if not values:
        return {"n": 0, "p50": None, "p95": None, "p99": None, "max": None}
    values = sorted(values)
    def at(fraction):
        position = (len(values)-1)*fraction
        low, high = math.floor(position), math.ceil(position)
        return values[low] + (values[high]-values[low])*(position-low)
    return {"n": len(values), "p50": at(.5), "p95": at(.95), "p99": at(.99), "max": values[-1]}


def summarize(cases):
    """Paired physical denominators include all cases, including unknowns."""
    def group(rows, label):
        positives = [row for row in rows if row["physical_truth"]["complete"]]
        negatives = [row for row in rows if not row["physical_truth"]["complete"]]
        result = {"cohort": label, "planned": len(rows), "generation_invalid": sum(not row["generation_valid"] for row in rows)}
        for policy in ("baseline", "fused"):
            result[policy] = {
                "positive_recognition": {"numerator": sum(row[policy]["verdict"] == "complete" for row in positives), "denominator": len(positives)},
                "negative_recognition": {"numerator": sum(row[policy]["verdict"] == "incomplete" for row in negatives), "denominator": len(negatives)},
                "false_completion": {"numerator": sum(row[policy]["verdict"] == "complete" for row in negatives), "denominator": len(negatives)},
                "abstention": {"numerator": sum(row[policy]["verdict"] == "unknown" for row in rows), "denominator": len(rows)}}
        false_baseline = [row for row in negatives if row["baseline"]["verdict"] == "complete"]
        true_baseline = [row for row in positives if row["baseline"]["verdict"] == "complete"]
        result["paired"] = {
            "caught_false_completions": {"numerator": sum(row["fused"]["verdict"] != "complete" for row in false_baseline), "denominator": len(false_baseline)},
            "caught_as_incomplete": sum(row["fused"]["verdict"] == "incomplete" for row in false_baseline),
            "caught_as_unknown": sum(row["fused"]["verdict"] == "unknown" for row in false_baseline),
            "lost_true_completions": {"numerator": sum(row["fused"]["verdict"] != "complete" for row in true_baseline), "denominator": len(true_baseline)},
            "new_false_completions": sum(row["baseline"]["verdict"] != "complete" and row["fused"]["verdict"] == "complete" for row in negatives)}
        stages = sorted({key for row in rows for key in row["timing_ms"]})
        result["latency_ms"] = {stage: quantiles([sample for row in rows for sample in row["timing_ms"].get(stage, [])]) for stage in stages}
        return result
    return [group([case for case in cases if case["cohort"] == cohort], cohort) for cohort in COHORTS] + [group(cases, "all")]


def _actuator_trace(packet):
    samples = packet.get("samples")
    if not isinstance(samples, list) or len(samples) > 100:
        raise ValueError("Expected a bounded drive trace.")
    result = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Expected a recorded drive sample.")
        units = sample.get("units", {})
        positions = sample.get("finger_position_m")
        gap = None
        if units.get("finger_position") == "m" and isinstance(positions, list) and len(positions) == 2:
            gap = _number(positions[0]) - _number(positions[1])
        effort = _number(sample["gripper_effort_n"]) if units.get("gripper_effort") == "N" else None
        command = _number(sample["gripper_command_m"]) if units.get("gripper_command") == "m" else None
        result.append({"t_s": _number(sample["t_s"]), "jaw_gap_m": gap, "gripper_effort_n": effort, "gripper_command_m": command})
    return result


def build_report(run_dir, output_dir):
    """Render all completed paired cases and only selected public evidence."""
    root, output = _plain(run_dir), _plain(output_dir)
    if output.exists() or output == root or root in output.parents or output in root.parents:
        raise ValueError("Report output must be fresh and separate from its input.")
    catalog, result = _json(root / "cases.json"), _json(root / "benchmark.json")
    if (catalog.get("schema") != "dvidia.sensor-goal-cases/v1"
            or result.get("schema") != "dvidia.sensor-goal-results/v1"
            or result.get("status") != "completed" or result.get("active_trial") is not None
            or result.get("provider_calls") != 0):
        raise ValueError("Only a completed, inactive, local-only benchmark can be rendered.")
    source_cases, source_rows = catalog.get("cases"), result.get("cases")
    if not isinstance(source_cases, list) or not 1 <= len(source_cases) <= 1088 or not isinstance(source_rows, list):
        raise ValueError("Expected a bounded complete paired cohort.")
    rows = {}
    for row in source_rows:
        identifier = _name(row.get("id"))
        if identifier in rows:
            raise ValueError("Duplicate paired result.")
        rows[identifier] = row
    cases, assets, seen = [], {}, set()
    for source in source_cases:
        identifier = _name(source.get("id"))
        if identifier in seen or identifier not in rows:
            raise ValueError("Duplicate case or missing paired result.")
        seen.add(identifier)
        row = rows[identifier]
        if any(row.get(key) != value for key, value in source.items()):
            raise ValueError("Case metadata differs between the frozen catalog and result.")
        if source.get("cohort") not in COHORTS:
            raise ValueError("Unknown paired cohort.")
        truth = source.get("physical_truth", {})
        if any(type(truth.get(key)) is not bool for key in ("present", "complete")) or type(source.get("generation_valid")) is not bool:
            raise ValueError("Physical truth and generation status must be explicit.")
        timing = row.get("timing_ms")
        if not isinstance(timing, list) or not 1 <= len(timing) <= 20:
            raise ValueError("Expected complete bounded timing repetitions.")
        times = {stage: [] for stage in STAGES}
        for repetition in timing:
            if not isinstance(repetition, dict) or set(repetition) != set(STAGES):
                raise ValueError("Missing or unknown compute stage.")
            for stage in STAGES:
                value = _number(repetition[stage])
                if not 0 < value <= 3600000:
                    raise ValueError("Compute measurements must be positive and bounded.")
                times[stage].append(value)
        decisions = {}
        for key, schema, allowed in (
            ("baseline", "dvidia.local-goal-verdict/v1", ("verdict", "reasons", "predicates", "window")),
            ("fused", "dvidia.sensor-goal-verdict/v1", ("verdict", "primary_verdict", "reasons", "primary_reasons", "limitations"))):
            value = row.get(key)
            if not isinstance(value, dict) or value.get("schema") != schema or value.get("verdict") not in VERDICTS:
                raise ValueError("Expected explicit baseline and fused verdicts.")
            decisions[key] = _public({field: value[field] for field in allowed if field in value})
        paths = {key: _relative(root, source[key]) for key in ("primary_packet_file", "camera_packet_file", "actuator_packet_file", "state_file")}
        camera_packet, drive_packet = _json(root / paths["camera_packet_file"]), _json(root / paths["actuator_packet_file"])
        frames = camera_packet.get("frames")
        if not isinstance(frames, list) or len(frames) > 11:
            raise ValueError("Expected a bounded camera frame inventory.")
        camera_frames = [{"t_s": _number(frame["t_s"]), "image": _image(root, frame["image"], assets)} for frame in frames]
        camera = row.get("camera")
        actuator = row.get("actuator")
        if not isinstance(camera, dict) or not isinstance(actuator, dict) or any(value.get("status") not in {"observed", "unknown"} for value in (camera, actuator)):
            raise ValueError("Expected explicit camera and drive observations.")
        camera = _public({key: value for key, value in camera.items() if key in {
            "schema", "status", "reason", "observations", "start_s", "end_s", "span_s", "sample_intervals_s", "net_displacement_m", "net_displacement_interval_m", "net_average_speed_interval_m_s", "edges", "scope"}})
        actuator = _public({key: value for key, value in actuator.items() if key in {
            "schema", "status", "reason", "start_s", "end_s", "release_corroborated", "reasons"}})
        if "start_s" in actuator and "end_s" in actuator:
            actuator["span_s"] = actuator["end_s"] - actuator["start_s"]
        case = {"id": identifier, "cohort": source["cohort"], "scenario": _name(source["scenario"]),
            "layout_id": _name(str(source["layout_id"])), "physical_truth": {key: truth[key] for key in ("present", "complete")},
            "generation_valid": source["generation_valid"], "now_s": _number(source["now_s"]),
            "prescribed_intervention": _public(source.get("prescribed_intervention", "")), "note": _public(source.get("note", "")),
            **paths, **decisions, "camera": camera, "actuator": actuator, "camera_frames": camera_frames,
            "actuator_trace": _actuator_trace(drive_packet), "timing_ms": times,
            "current_image": _image(root, source["current_image"], assets), "goal_image": _image(root, source["goal_image"], assets)}
        cases.append(case)
    if seen != rows.keys():
        raise ValueError("Unplanned paired results cannot be silently omitted.")
    public = {"schema": "dvidia.sensor-goal-report/v1", "status": "completed", "mode": "local-rgb-drive-corroboration",
        "model_calls": 0, "cases": cases, "summaries": summarize(cases), "timing_scope": _public(result.get("timing_scope", "")),
        "limits": ["The primary verifier remains necessary. RGB and drive observations corroborate a claim; they do not independently prove every completion condition.",
            "A declared green elongated proxy and fixed orthographic calibration are used. This is not a learned shoe detector, authenticated identity estimate, or depth reconstruction.",
            "Axial orientation, instantaneous motion, elevation, lookalike objects and consistently refreshed replay remain possible blind spots.",
            "Drive effort is not target contact force. An open settled jaw cannot prove table support or absence of contact with every robot link.",
            "All native channels share one simulator; separately processed paths are not independent physical sensors. Declared uncertainty and calibration IDs are not authentication.",
            "Every scheduled case remains visible. Unknowns remain in physical-positive and physical-negative denominators. No policy training, model service calls, robot commands or hardware qualification."]}
    payload = {"report.json": _encoded(public), **assets}
    embedded = json.dumps(public, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    payload["index.html"] = _HTML.replace("__PUBLIC_DATA__", embedded).encode()
    seal = {"schema": "dvidia.sensor-goal-report-integrity/v1", "files": {name: {"bytes": len(raw), "sha256": sha256(raw).hexdigest()} for name, raw in sorted(payload.items())}}
    output.mkdir(parents=True)
    for name, raw in payload.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    (output / "report-integrity.json").write_bytes(_encoded(seal))
    return seal


_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>Corroborating completion · DVIDIA research</title>
<style>
:root{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#26332d;background:#faf9f6;font-synthesis:none}*{box-sizing:border-box}body{margin:0;background-image:radial-gradient(#28332c09 .55px,transparent .65px);background-size:5px 5px}a{color:inherit;text-underline-offset:4px}button,select,input{font:inherit;min-height:44px;touch-action:manipulation}button:focus-visible,select:focus-visible,a:focus-visible,input:focus-visible,summary:focus-visible{outline:3px solid #3c6cff;outline-offset:3px}.skip{position:absolute;top:-90px;background:white;padding:14px;z-index:2}.skip:focus{top:10px}.masthead{padding:20px 5vw;border-bottom:1px solid #dcded5;display:flex;align-items:center;justify-content:space-between;gap:18px}.brand{display:flex;gap:11px;align-items:center;text-decoration:none;font-weight:750;font-size:20px;letter-spacing:2px}.brand svg{width:34px;height:34px}.top-links{display:flex;gap:22px;font-size:13px}main{max-width:1480px;margin:auto;padding:42px 5vw}.eyebrow{font-size:11px;letter-spacing:2px;text-transform:uppercase;color:#68766d;font-weight:650}h1,h2,h3{font-family:Georgia,"Times New Roman",serif;font-weight:400}h1{font-size:clamp(38px,5vw,64px);letter-spacing:-2px;line-height:1.1;margin:14px 0 20px}.intro{max-width:860px;line-height:1.8;color:#59675e}.notice{border:1px solid #d8dfd1;background:#f1f3ec;padding:16px 20px;margin:25px 0;font-size:13px;line-height:1.8}.warning{background:#f8f0e6;border-color:#e7d5ba}.toolbar{display:flex;align-items:end;justify-content:space-between;gap:20px;margin:28px 0 18px}label{display:block;font-size:12px;color:#5c6c60;margin-bottom:7px}select{background:#fffefb;border:1px solid #cfd5ca;padding:10px 34px 10px 12px;border-radius:4px;color:inherit;max-width:100%}.note,.meta{font-size:12px;color:#637367;line-height:1.8}.summary-grid{display:grid;grid-template-columns:1.25fr 1fr;gap:28px;margin-bottom:28px}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px;line-height:1.65}th,td{padding:11px 12px 11px 0;border-bottom:1px solid #dce1d5;text-align:left}th{font-weight:550;color:#617263}th:not(:first-child),td:not(:first-child){text-align:right}.paired{padding:16px 20px;background:#f0f2ea;border:1px solid #dce2d4;display:grid;grid-template-columns:1fr 1fr;gap:20px}.paired h2{font:12px system-ui;margin:0;color:#627262}.paired strong{display:block;font:31px Georgia,serif;margin:14px 0 8px}.paired p{font-size:12px;line-height:1.65;margin:0;color:#657266}.workspace{display:grid;grid-template-columns:225px minmax(0,1fr);gap:28px}.case-nav{border-top:1px solid #dce1d5}.case-nav h2{font:12px system-ui;letter-spacing:1px;text-transform:uppercase;margin:17px 0}.case-nav select{width:100%;font-size:12px;margin-bottom:10px}.case-list{display:grid;gap:4px;max-height:860px;overflow:auto;padding:2px}.case-list button{text-align:left;border:0;background:transparent;padding:11px 10px;color:#526257;cursor:pointer;border-radius:4px;font-size:12px;line-height:1.6}.case-list button[aria-pressed=true]{background:#e8eee2;color:#263c2a;box-shadow:inset 3px 0 #5f775d}.case-list small{display:block;color:#69776c}.case-heading{display:flex;align-items:baseline;justify-content:space-between;gap:20px}.case-heading h2{font-size:26px;margin:0 0 8px}.case-heading p{margin:0}.decisions{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:18px 0}.decision{padding:15px;border:1px solid #dbe0d4;background:#fffefb}.decision h3{font:11px system-ui;text-transform:uppercase;letter-spacing:1px;margin:0;color:#647368}.decision strong{display:block;font:24px Georgia,serif;margin:12px 0}.decision p{font-size:11px;line-height:1.7;margin:0;overflow-wrap:anywhere;color:#647266}.badge{display:inline-block;background:#e8eee1;border-radius:3px;padding:2px 6px}.badge.unknown{background:#f2eddf}.badge.incomplete,.badge.fail{background:#f5e6dd}.evidence{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(250px,1fr);gap:20px}.camera{min-width:0}.camera img,.scene-images img{width:100%;aspect-ratio:4/3;object-fit:contain;background:#e9e9df;border:1px solid #dde0d5;display:block}figure{margin:0;min-width:0}figcaption{font-size:12px;color:#5d6d61;margin:10px 0;line-height:1.7}.frame-control{display:flex;gap:12px;align-items:center}.frame-control input{flex:1;min-width:50px;accent-color:#536f57}.frame-control output{font-size:11px;white-space:nowrap;color:#637165}.panel{border-top:1px solid #dce1d5;padding-top:15px;margin-top:16px}.panel h3{font-size:23px;margin:0 0 12px}.features{display:grid;grid-template-columns:1fr auto;gap:9px;font-size:11px;line-height:1.7}.features dt{color:#607164}.features dd{margin:0;text-align:right;overflow-wrap:anywhere;max-width:230px}.trace{width:100%;height:135px;background:#f3f4ed;border:1px solid #dde1d5}.trace text{font:10px system-ui;fill:#657467}.trace polyline{fill:none;stroke:#547660;stroke-width:2;vector-effect:non-scaling-stroke}.trace-label{font-size:11px;color:#5f7063;margin:12px 0 6px}.empty{padding:25px;background:#f1f3eb;color:#65746a;font-size:13px;line-height:1.8}.scene-images{display:grid;grid-template-columns:1fr 1fr;gap:14px}details{border-top:1px solid #dce1d5;margin:20px 0;padding-top:14px}summary{cursor:pointer;font-size:13px;min-height:36px;color:#435c4b}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px;line-height:1.7;background:#f0f2ea;padding:16px;max-height:450px;overflow:auto}.downloads{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;line-height:1.8;margin:22px 0}.fine{margin-top:36px;border-top:1px solid #dce1d5;padding-top:24px}.fine h2{font-size:29px}.fine p,.fine li{font-size:13px;color:#5c6c60;line-height:1.9;max-width:1000px}footer{border-top:1px solid #dce1d5;margin-top:32px;padding-top:22px;display:flex;justify-content:space-between;gap:20px;font-size:12px;line-height:1.8;color:#657467}@media(max-width:1100px){.evidence{grid-template-columns:1fr}.summary-grid{grid-template-columns:1fr}.workspace{grid-template-columns:200px minmax(0,1fr);gap:20px}}@media(max-width:640px){main{padding:28px 18px}.masthead{padding:16px 18px}.top-links a:first-child{display:none}.workspace{grid-template-columns:1fr}.case-list{grid-template-columns:repeat(2,minmax(0,1fr));max-height:240px}.toolbar,.case-heading{display:block}.toolbar .meta{margin-top:12px}.decisions{grid-template-columns:1fr}.decision strong{display:inline-block;margin:8px 0}.scene-images{grid-template-columns:1fr}.paired{padding:14px;gap:15px}.paired strong{font-size:27px}h1{letter-spacing:-1px}footer{display:block}}
</style></head><body><a class="skip" href="#main">Skip to evidence</a>
<header class="masthead"><a class="brand" href="https://research.dvidia.org/" aria-label="DVIDIA research"><svg viewBox="0 0 64 64" aria-hidden="true"><rect width="64" height="64" rx="16" fill="#3c6cff"/><path d="M16 0h32c8.8 0 16 7.2 16 16v5C45 8 25 8 0 32V16C0 7.2 7.2 0 16 0Z" fill="#fff" opacity=".16"/><g fill="#f7f6f2"><rect x="8" y="20" width="4.6" height="24" rx="1.2" opacity=".28"/><rect x="14.1" y="20" width="4.6" height="24" rx="1.2" opacity=".38"/><rect x="20.2" y="20" width="4.6" height="24" rx="1.2" opacity=".48"/><rect x="26.3" y="20" width="4.6" height="24" rx="1.2" opacity=".58"/><rect x="32.4" y="20" width="4.6" height="24" rx="1.2" opacity=".68"/><rect x="38.5" y="20" width="4.6" height="24" rx="1.2" opacity=".78"/><rect x="44.6" y="20" width="4.6" height="24" rx="1.2" opacity=".9"/><rect x="50.7" y="20" width="4.6" height="24" rx="1.2" fill="#fff"/></g></svg><span>DVIDIA</span></a><nav class="top-links" aria-label="Research links"><a href="https://research.dvidia.org/papers/sensor-goal-verifier-v04/">Research notes ↗</a><a href="https://github.com/Dvidia-Inference/dvidia-training">Source ↗</a></nav></header>
<main id="main"><div class="eyebrow">Manipulation research · observation contracts v0.4</div><h1>Does the evidence agree?</h1><p class="intro">An image sequence and a gripper-drive history corroborate a completion claim. Compare the original verifier with the combined decision on the same physical scene, including the failures that survive every check.</p>
<div class="notice warning"><strong>Corroboration has limits.</strong> A green shape is a candidate, not authenticated identity. A visible position does not prove support or instantaneous rest. Drive effort is not object contact force. All channels here originate in one simulator. <strong>Zero model calls; no robot commands.</strong></div>
<div class="toolbar"><div><label for="cohort">Evidence cohort</label><select id="cohort"><option value="native">Fresh native scenes</option><option value="primary_forgery">Forged primary readings</option><option value="sensor_fault">Auxiliary sensor faults</option><option value="cross_channel">Cross-channel and blind controls</option><option value="all">All cohorts — mixed interventions</option></select></div><div class="meta" id="cohort-meta" aria-live="polite"></div></div>
<div class="summary-grid"><div class="table-wrap"><table aria-label="Paired physical outcomes"><thead><tr><th>Physical reference</th><th>Baseline v0.3</th><th>Corroborated</th></tr></thead><tbody id="summary"></tbody></table></div><div class="paired"><section><h2>False completions blocked</h2><strong id="caught"></strong><p id="caught-detail"></p></section><section><h2>True completions lost</h2><strong id="lost"></strong><p>Original successes that become incomplete or unknown. All physically complete cases remain counted.</p></section></div></div>
<div class="workspace"><aside class="case-nav" aria-label="Case selection"><h2>Inspect a case</h2><label for="filter">Case filter</label><select id="filter"><option value="all">Every case</option><option value="positive">Physically complete</option><option value="negative">Physically incomplete</option><option value="caught">False completion blocked</option><option value="survives">False completion survives</option><option value="lost">True completion lost</option><option value="unknown">Combined decision unknown</option><option value="changed">Decisions differ</option></select><p id="case-count" class="note" aria-live="polite"></p><div id="case-list" class="case-list"></div></aside>
<article id="case"><div class="case-heading"><h2 id="case-title"></h2><p id="case-id" class="note"></p></div><p id="intervention" class="note"></p><div class="decisions"><section class="decision"><h3>Physical reference</h3><strong id="physical-verdict"></strong><p id="physical-detail"></p></section><section class="decision"><h3>Baseline v0.3</h3><strong id="baseline-verdict"></strong><p id="baseline-reasons"></p></section><section class="decision"><h3>Corroborated verdict</h3><strong id="fused-verdict"></strong><p id="fused-reasons"></p></section></div>
<div class="evidence"><section class="camera"><figure><img id="camera-image" alt=""><figcaption id="camera-caption"></figcaption></figure><label for="frame">Consumed camera frame</label><div class="frame-control"><input id="frame" type="range" min="0" max="0" value="0" step="1"><output id="frame-count" for="frame"></output></div><div class="panel"><h3>What the pixels support</h3><dl id="camera-features" class="features"></dl><p class="note">Axial yaw is modulo 180°. Bounds come from the declared color, shape and camera model; they are not guarantees under arbitrary lighting or occlusion.</p></div></section>
<section><div class="panel" style="margin-top:0"><h3>What the drive reports</h3><p class="note" id="drive-status"></p><p class="trace-label">Jaw gap · mm</p><svg id="gap-trace" class="trace" viewBox="0 0 420 135" role="img" aria-label="Recorded jaw gap over the actuator window"></svg><p class="trace-label">Gripper drive effort · N</p><svg id="effort-trace" class="trace" viewBox="0 0 420 135" role="img" aria-label="Recorded gripper drive effort over the actuator window"></svg><dl id="drive-features" class="features"></dl><p class="note">Command, encoder and effort consistency can corroborate an open, settled jaw. It cannot establish table support or absence of contact with every arm link.</p></div></section></div>
<details><summary>Physical scene and reference goal</summary><div class="scene-images"><figure><img id="current-image" alt=""><figcaption>Physical scene reference. This view is separate from the consumed RGB sequence.</figcaption></figure><figure><img id="goal-image" alt=""><figcaption>Reference completed placement for this layout.</figcaption></figure></div></details>
<details><summary>Decision reasons and channel checks</summary><pre id="checks"></pre></details><details><summary>Selected observer outputs · full outputs in benchmark</summary><pre id="observations"></pre></details><div class="downloads" id="case-downloads"></div></article><div id="empty" class="empty" hidden>No cases match this filter.</div></div>
<section class="fine"><h2>Time, with its scope intact.</h2><p id="timing-scope"></p><p><strong>Requires 200 ms of observation history.</strong> Repeated pipeline calls measure compute variation; they are not additional physical trials or a hard real-time guarantee. These measurements are not a matched speedup comparison with a hosted Jev request or the previous parsed-state-only verifier.</p><div class="table-wrap"><table aria-label="Recorded compute stages in milliseconds"><thead><tr><th>Stage</th><th>Calls</th><th>Median · ms</th><th>p95 · ms</th><th>Max · ms</th></tr></thead><tbody id="timings"></tbody></table></div></section>
<section class="fine"><h2>Keep the blind spots visible.</h2><ul id="limits"></ul><p>Physical reference, evidence sufficiency and algorithmic agreement are different measurements. An unknown decision can block an unsafe success claim without proving that the task failed. Shared replay, consistent fabricated inputs and unobservable physical properties remain explicit controls.</p><div class="downloads"><a href="benchmark.json">Full benchmark</a><a href="cases.json">Every planned case</a><a href="protocol.json">Frozen protocol</a><a href="report.json">Viewer data</a><a href="publication-integrity.json">Package checksums</a><a href="README.md">Reproduction and licenses</a></div></section>
<footer><span>DVIDIA research · Original code and authored records: MIT<br>Upstream asset notices retained in the package.</span><span><a href="mailto:hello@dvidia.org">hello@dvidia.org</a> · <a href="https://x.com/iammrriver">@iammrriver</a></span></footer></main>
<script type="application/json" id="evidence-data">__PUBLIC_DATA__</script>
<script>
'use strict';
const DATA=JSON.parse(document.getElementById('evidence-data').textContent), $=id=>document.getElementById(id);
const words=value=>String(value??'unavailable').replaceAll('_',' '), number=(v,d=3)=>typeof v==='number'&&Number.isFinite(v)?v.toFixed(d):'unavailable';
const ratio=r=>r&&r.denominator?`${r.numerator}/${r.denominator}`:'n/a · no eligible cases';
const el=(tag,text)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n;};
let selected=DATA.cases[0]?.id, frame=0;
function verdict(id,value){const n=$(id);n.textContent=words(value);n.className='badge '+value;}
function facts(id,values){const n=$(id);n.replaceChildren();for(const [label,value] of values){n.append(el('dt',label),el('dd',value));}}
function interval(value,scale=1,unit=''){return Array.isArray(value)&&value.length===2?`${number(value[0]*scale)} to ${number(value[1]*scale)} ${unit}`:'unavailable';}
function filtered(){return DATA.cases.filter(c=>($('cohort').value==='all'||c.cohort===$('cohort').value)&&({all:true,positive:c.physical_truth.complete,negative:!c.physical_truth.complete,caught:!c.physical_truth.complete&&c.baseline.verdict==='complete'&&c.fused.verdict!=='complete',survives:!c.physical_truth.complete&&c.fused.verdict==='complete',lost:c.physical_truth.complete&&c.baseline.verdict==='complete'&&c.fused.verdict!=='complete',unknown:c.fused.verdict==='unknown',changed:c.baseline.verdict!==c.fused.verdict}[$('filter').value]));}
function summary(){const s=DATA.summaries.find(s=>s.cohort===$('cohort').value);$('cohort-meta').textContent=`${s.planned} planned cases · ${s.generation_invalid} generation mismatches · zero model calls`;$('summary').replaceChildren();for(const [label,key] of [['Complete cases recognized','positive_recognition'],['Incomplete cases recognized','negative_recognition'],['False completion claims','false_completion'],['Unknown decisions','abstention']]){const tr=el('tr');tr.append(el('td',label),el('td',ratio(s.baseline[key])),el('td',ratio(s.fused[key])));$('summary').append(tr);}$('caught').textContent=ratio(s.paired.caught_false_completions);$('caught-detail').textContent=`${s.paired.caught_as_incomplete} become incomplete; ${s.paired.caught_as_unknown} become unknown.`;$('lost').textContent=ratio(s.paired.lost_true_completions);$('timings').replaceChildren();for(const [stage,t] of Object.entries(s.latency_ms)){const tr=el('tr');[words(stage),String(t.n),number(t.p50),number(t.p95),number(t.max)].forEach(v=>tr.append(el('td',v)));$('timings').append(tr);}}
function svgText(parent,x,y,text){const n=document.createElementNS('http://www.w3.org/2000/svg','text');n.setAttribute('x',x);n.setAttribute('y',y);n.textContent=text;parent.append(n);}
function plot(id,rows,key,scale){const svg=$(id);svg.replaceChildren();const pts=rows.filter(r=>typeof r.t_s==='number'&&typeof r[key]==='number').map(r=>[r.t_s,r[key]*scale]);if(!pts.length){svgText(svg,20,65,'No valid recorded samples');return;}const ts=pts.map(p=>p[0]),ys=pts.map(p=>p[1]),t0=Math.min(...ts),t1=Math.max(...ts),lo=Math.min(...ys),hi=Math.max(...ys),span=hi-lo||Math.max(Math.abs(hi)*.1,.001);const line=document.createElementNS('http://www.w3.org/2000/svg','polyline');line.setAttribute('points',pts.map(([t,v])=>`${45+350*(t-t0)/(t1-t0||1)},${105-80*(v-lo)/span}`).join(' '));svg.append(line);svgText(svg,4,25,number(hi,3));svgText(svg,4,105,number(lo,3));svgText(svg,45,126,number(t0,3)+' s');svgText(svg,350,126,number(t1,3)+' s');}
function showFrame(c){const frames=c.camera_frames;frame=Math.min(Math.max(frame,0),frames.length-1);const f=frames[frame];$('frame').max=Math.max(0,frames.length-1);$('frame').value=frame;$('frame').disabled=frames.length<2;if(!f){$('camera-image').hidden=true;$('camera-caption').textContent='No readable camera frames are supplied for this planned case.';$('frame-count').textContent='0 frames';facts('camera-features',[['Observation','Unavailable']]);return;}$('camera-image').hidden=false;$('camera-image').src=f.image;$('camera-image').alt=`Consumed RGB frame ${frame+1} of ${frames.length} for ${c.id}`;$('camera-caption').textContent=`Exact input PNG · timestamp ${number(f.t_s,3)} s · ${f.image.split('/').pop()}`;$('frame-count').textContent=`${frame+1} / ${frames.length}`;const o=c.camera.observations?.[frame]||{};facts('camera-features',[['Status',words(o.status)],['Reason',words(o.reason)],['Candidates',String(o.candidate_count??'unavailable')],['XY position',Array.isArray(o.position_xy_m)?o.position_xy_m.map(v=>number(v)).join(', ')+' m':'unavailable'],['Declared position bound',number(o.position_bound_m)+' m'],['Axial yaw',number(o.yaw_axis_rad)+' rad'],['Declared yaw bound',number(o.yaw_bound_rad)+' rad'],['Observed green pixels',String(o.visible_green_pixels??'unavailable')]]);}
function show(c){$('case-title').textContent=words(c.scenario);$('case-id').textContent=c.id;$('intervention').textContent=c.prescribed_intervention||c.note||'Native evidence without an injected intervention.';verdict('physical-verdict',c.physical_truth.complete?'complete':'incomplete');$('physical-detail').textContent=`Target ${c.physical_truth.present?'present':'absent'} · independent native reference. ${c.generation_valid?'':'Intended generation category did not hold.'}`;for(const key of ['baseline','fused']){verdict(key+'-verdict',c[key].verdict);$(key+'-reasons').textContent=(c[key].reasons||[]).map(words).join(' · ')||'All declared checks passed.';}showFrame(c);const drive=c.actuator;$('drive-status').textContent=words(drive.status)+' · '+words(drive.reason);plot('gap-trace',c.actuator_trace,'jaw_gap_m',1000);plot('effort-trace',c.actuator_trace,'gripper_effort_n',1);facts('drive-features',[['Recorded samples',String(c.actuator_trace.length)],['History span',number(drive.span_s)+' s'],['Channel scope','Drive encoders and effort only']]);$('current-image').src=c.current_image;$('current-image').alt=`Native scene reference for ${c.id}`;$('goal-image').src=c.goal_image;$('goal-image').alt=`Reference completed placement for layout ${c.layout_id}`;$('checks').textContent=JSON.stringify({baseline:c.baseline,fused:c.fused},null,2);$('observations').textContent=JSON.stringify({camera:c.camera,actuator:c.actuator},null,2);$('case-downloads').replaceChildren();for(const [label,key] of [['Primary packet','primary_packet_file'],['Camera packet','camera_packet_file'],['Drive packet','actuator_packet_file'],['Native state','state_file']]){if(c[key]){const link=el('a',label);link.href=c[key];$('case-downloads').append(link);}}}
function render(restoreFocus=false){summary();const rows=filtered();if(!rows.some(c=>c.id===selected)){selected=rows[0]?.id;frame=0;}$('case-count').textContent=`${rows.length} cases shown`;$('case-list').replaceChildren();for(const c of rows){const b=el('button');b.type='button';b.setAttribute('aria-pressed',String(c.id===selected));b.append(el('span',words(c.scenario)),el('small',`${c.layout_id} · ${c.baseline.verdict} → ${c.fused.verdict}`));b.addEventListener('click',()=>{selected=c.id;frame=0;render(true);});$('case-list').append(b);}$('case').hidden=!rows.length;$('empty').hidden=!!rows.length;if(rows.length)show(rows.find(c=>c.id===selected));if(restoreFocus)$('case-list').querySelector('[aria-pressed="true"]')?.focus();}
$('cohort').addEventListener('change',()=>render());$('filter').addEventListener('change',()=>render());$('frame').addEventListener('input',()=>{frame=Number($('frame').value);showFrame(DATA.cases.find(c=>c.id===selected));});$('timing-scope').textContent=DATA.timing_scope;for(const text of DATA.limits)$('limits').append(el('li',text));render();
</script></body></html>'''



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    seal = build_report(args.run, args.output)
    print(json.dumps({"files": len(seal["files"]), "output": str(Path(args.output).absolute())}))


if __name__ == "__main__":
    main()

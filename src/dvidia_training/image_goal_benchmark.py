"""Frozen, bounded hosted-model evaluation against independent simulator labels.

This evaluates observation decisions. It does not command a robot or train a
policy. Prerecorded rendering time and hosted request time remain separate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import random
import time

CONDITIONS = ("rgb", "rgb_telemetry")
MODEL = "jev-1.14.0"
MAX_REQUESTS = 64


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    raw = Path(path).read_bytes()
    return {"sha256": sha256(raw).hexdigest(), "bytes": len(raw)}


def relative_file(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("Expected a relative artifact path")
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Artifact must be a file inside the fixture directory")
    return path


def validate_cases(document):
    cases = document.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= MAX_REQUESTS // 2:
        raise ValueError("Expected 1–32 cases")
    ids = set()
    for case in cases:
        identity = case.get("id")
        if not isinstance(identity, str) or not identity or identity in ids:
            raise ValueError("Case identities must be unique strings")
        ids.add(identity)
        truth = case.get("physical_truth", {})
        if any(type(truth.get(key)) is not bool for key in ("present", "complete")):
            raise ValueError("Independent presence and completion labels are required")
        if truth["complete"] and not truth["present"]:
            raise ValueError("An absent target cannot be complete")
        if not isinstance(case.get("measured_inputs"), dict):
            raise ValueError("Measured inputs must be explicit")
        for condition in CONDITIONS:
            if case.get("expected_verdict", {}).get(condition) not in {"complete", "incomplete", "unknown"}:
                raise ValueError("Observability-aware verdict labels are required")
            if case.get("expected_presence", {}).get(condition) not in {"present", "absent", "unknown"}:
                raise ValueError("Observability-aware presence labels are required")
    return cases


def freeze(root, *, task, source_commit):
    """Seal images, labels, semantic requests and evaluation rules before calls."""
    from .jev_client import build_goal_request
    root = Path(root)
    target = root / "evaluation-plan.json"
    if target.exists():
        raise FileExistsError("A frozen evaluation plan cannot be overwritten")
    cases = validate_cases(read_json(root / "cases.json"))
    files = {name: digest(root / name) for name in ("cases.json", "protocol.json")}
    for name in ("integrity.json", "protocol.sha256"):
        if (root / name).is_file():
            files[name] = digest(root / name)
    trials = []
    for case in cases:
        current = relative_file(root, case["current_image"])
        goal = relative_file(root, case["goal_image"])
        files[case["current_image"]] = digest(current)
        files[case["goal_image"]] = digest(goal)
        if case.get("state_file"):
            files[case["state_file"]] = digest(relative_file(root, case["state_file"]))
        for condition in CONDITIONS:
            prepared = build_goal_request(current.read_bytes(), goal.read_bytes(), task=task,
                measured_state=case["measured_inputs"] if condition == "rgb_telemetry" else None)
            trials.append({"case_id": case["id"], "condition": condition,
                           "request_sha256": prepared.request_sha256})
    random.Random(20261018).shuffle(trials)
    source_files = {}
    for name in ("image_goal_benchmark.py", "image_goal_scenes.py", "jev_client.py"):
        source_files[name] = digest(Path(__file__).with_name(name))
    plan = {"schema": "dvidia.image-goal-evaluation-plan/v1", "model": MODEL,
        "frozen_at": datetime.now(timezone.utc).isoformat(), "source_commit": source_commit,
        "task": task, "conditions": list(CONDITIONS), "trials": trials, "files": files,
        "source_files": source_files, "max_requests": MAX_REQUESTS,
        "timeout_s": 20., "timeout_scope": "Per socket operation; not a hard wall-clock deadline.",
        "automatic_retries": 0, "presence_thresholds": [.2, .8],
        "completion_decision": "Unmodified Jev Choice verdict; unknown is abstention.",
        "documented_input_price_usd_per_million_tokens": .042,
        "price_checked_date": "2026-10-18",
        "planning_allowance_per_request_usd": .003,
        "cost_scope": "Planning allowance assumes documented roughly 64k default context. Not a billing guarantee; receipts retain known usage and unknown-cost attempts.",
        "cohort": "Authored diagnostic fixtures; four scene layouts are correlated clusters, not 32 independent real-world tasks.",
        "latency_scope": "Preloaded image bytes + request construction + image upload + hosted response + parsing; excludes fixture file loading, prerecorded rendering, live camera, actuation and physical task duration.",
        "rules": ["Keep every scheduled trial, including errors and unrun trials.",
                  "No retries, prompt tuning, threshold tuning or case selection after evaluation begins.",
                  "No direct actuator commands and no model training or distillation.",
                  "Report physical false completion separately from observability-aware accuracy.",
                  "No remote calls when credentials are absent; missing results are never scored as accurate."]}
    target.write_bytes(json_bytes(plan))
    return plan


def verify_plan(root):
    root = Path(root)
    plan = read_json(root / "evaluation-plan.json")
    if plan.get("model") != MODEL or len(plan.get("trials", [])) > MAX_REQUESTS:
        raise ValueError("Invalid frozen model or request budget")
    for name, expected in plan["files"].items():
        if digest(relative_file(root, name)) != expected:
            raise ValueError("Frozen fixture changed: " + name)
    for name, expected in plan["source_files"].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError("Frozen implementation changed: " + name)
    cases = validate_cases(read_json(root / "cases.json"))
    expected = {(case["id"], condition) for case in cases for condition in CONDITIONS}
    actual = [(row["case_id"], row["condition"]) for row in plan["trials"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Frozen trial inventory differs from complete paired cohort")
    return plan, cases


def quantiles(values):
    values = sorted(float(value) for value in values if type(value) in (int, float) and math.isfinite(value))
    if not values:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}
    def percentile(q):
        position = (len(values) - 1) * q
        lo, hi = math.floor(position), math.ceil(position)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)
    return {"count": len(values), "p50": percentile(.5), "p95": percentile(.95),
            "p99": percentile(.99), "max": values[-1]}


def fraction(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def score(rows, cases):
    """Errors stay in scheduled denominators; abstention is not binary success."""
    by_id = {case["id"]: case for case in cases}
    if len({(r["case_id"], r["condition"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate trial result")
    for row in rows:
        if row.get("case_id") not in by_id or row.get("condition") not in CONDITIONS:
            raise ValueError("Unknown trial identity")
        if row.get("status") not in {"ok", "error", "not_run"}:
            raise ValueError("Invalid trial status")
        if row["status"] == "ok":
            if row.get("verdict") not in {"complete", "incomplete", "unknown"} or row.get("presence") not in {"present", "absent", "unknown"}:
                raise ValueError("Invalid typed decision")
            for name in ("p_present", "p_complete"):
                value = row.get(name)
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Invalid probability")
    expected = {(case["id"], condition) for case in cases for condition in CONDITIONS}
    if {(r["case_id"], r["condition"]) for r in rows} != expected:
        raise ValueError("Scoring requires every scheduled case and condition, including explicit unrun rows")
    def group(selected):
        valid = [r for r in selected if r["status"] == "ok"]
        answered = [r for r in valid if r["verdict"] != "unknown"]
        incomplete = [r for r in selected if not by_id[r["case_id"]]["physical_truth"]["complete"]]
        absent = [r for r in selected if not by_id[r["case_id"]]["physical_truth"]["present"]]
        complete = [r for r in selected if by_id[r["case_id"]]["physical_truth"]["complete"]]
        known = [r for r in selected if by_id[r["case_id"]]["expected_verdict"][r["condition"]] != "unknown"]
        ambiguous = [r for r in selected if by_id[r["case_id"]]["expected_verdict"][r["condition"]] == "unknown"]
        false_complete = sum(r["status"] == "ok" and r["verdict"] == "complete" for r in incomplete)
        true_complete = sum(r["status"] == "ok" and r["verdict"] == "complete" for r in complete)
        probability_rows = [r for r in valid if type(r.get("p_complete")) in (float, int)]
        brier = (sum((r["p_complete"] - by_id[r["case_id"]]["physical_truth"]["complete"]) ** 2
                     for r in probability_rows) / len(probability_rows) if probability_rows else None)
        return {"scheduled": len(selected), "completed_requests": len(valid),
            "statuses": dict(Counter(r["status"] for r in selected)),
            "verdicts": dict(Counter(r["verdict"] for r in valid)),
            "observability_aware_accuracy": fraction(sum(r["status"] == "ok" and r["verdict"] == by_id[r["case_id"]]["expected_verdict"][r["condition"]] for r in selected), len(selected)),
            "known_verdict_accuracy": fraction(sum(r["status"] == "ok" and r["verdict"] == by_id[r["case_id"]]["expected_verdict"][r["condition"]] for r in known), len(known)),
            "ambiguous_abstention": fraction(sum(r["status"] == "ok" and r["verdict"] == "unknown" for r in ambiguous), len(ambiguous)),
            "physical_false_completion": fraction(false_complete, len(incomplete)),
            "physical_false_completion_valid_requests": fraction(false_complete, sum(r["status"] == "ok" for r in incomplete)),
            "physical_true_completion": fraction(true_complete, len(complete)),
            "missing_object_false_presence": fraction(sum(r["status"] == "ok" and r["presence"] == "present" for r in absent), len(absent)),
            "presence_accuracy": fraction(sum(r["status"] == "ok" and r["presence"] == by_id[r["case_id"]]["expected_presence"][r["condition"]] for r in selected), len(selected)),
            "completion_coverage": fraction(len(answered), len(selected)),
            "selective_physical_accuracy": fraction(sum((r["verdict"] == "complete") == by_id[r["case_id"]]["physical_truth"]["complete"] for r in answered), len(answered)),
            "physical_completion_brier": {"count": len(probability_rows), "mean": brier},
            "online_ms_all_attempts": quantiles([r.get("online_total_ms") for r in selected]),
            "online_ms_successful_requests": quantiles([r.get("online_total_ms") for r in valid]),
            "request_ms_successful_requests": quantiles([r.get("latency_ms", {}).get("http") for r in valid]),
            "known_usage_estimated_input_cost_usd": sum(r["estimated_input_cost_usd"] for r in selected if r.get("estimated_input_cost_usd") is not None) if any(r.get("estimated_input_cost_usd") is not None for r in selected) else None,
            "unknown_cost_attempts": sum(r.get("attempts", int(r["status"] != "not_run")) > 0 and r.get("estimated_input_cost_usd") is None for r in selected)}
    output = {"overall": group(rows), "by_condition": {}, "by_category": {}}
    for condition in CONDITIONS:
        output["by_condition"][condition] = group([r for r in rows if r["condition"] == condition])
    for category in sorted({case.get("category", "unspecified") for case in cases}):
        output["by_category"][category] = group([r for r in rows if by_id[r["case_id"]].get("category", "unspecified") == category])
    return output


def evaluate(root, output, *, api_key=None):
    from .jev_client import JevClient, build_goal_request
    root, output = Path(root), Path(output)
    plan, cases = verify_plan(root)
    if output.exists():
        raise FileExistsError("Use a new directory; existing attempts are immutable")
    by_id = {case["id"]: case for case in cases}
    # Check the entire request cohort before contacting the provider. Cache the
    # verified bytes so a later fixture mutation cannot alter a paid trial.
    verified = {}
    for trial in plan["trials"]:
        case = by_id[trial["case_id"]]
        current = relative_file(root, case["current_image"]).read_bytes()
        goal = relative_file(root, case["goal_image"]).read_bytes()
        measured = case["measured_inputs"] if trial["condition"] == "rgb_telemetry" else None
        prepared = build_goal_request(current, goal, task=plan["task"], measured_state=measured)
        if prepared.request_sha256 != trial["request_sha256"]:
            raise ValueError("Frozen semantic request changed")
        verified[(case["id"], trial["condition"])] = (current, goal, measured)
    client = JevClient(api_key=api_key, timeout_s=plan["timeout_s"])
    output.mkdir(parents=True)
    (output / "evaluation-plan.json").write_bytes((root / "evaluation-plan.json").read_bytes())
    rows = [{"case_id": trial["case_id"], "condition": trial["condition"],
             "status": "not_run", "attempts": 0, "presence": "unknown", "verdict": "unknown",
             "p_present": None, "p_complete": None, "online_total_ms": None,
             "error": {"kind": "not_attempted", "message": "Scheduled trial has not run"}}
            for trial in plan["trials"]]
    base_report = {"schema": "dvidia.image-goal-results/v1", "model": MODEL,
        "planned_conditions": list(CONDITIONS),
        "evaluation_plan_sha256": digest(root / "evaluation-plan.json")["sha256"],
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "limitations": [plan["cohort"], plan["latency_scope"],
            "Telemetry is privileged simulator state, not validated real sensor readings.",
            "Jev is a hosted experimental image evaluator; no robot learning or local Jev inference is demonstrated."]}
    def checkpoint(status, active_trial=None):
        report = {**base_report, "status": status, "results": rows,
                  "active_trial": active_trial,
                  "metrics": None if any(row["status"] == "not_run" for row in rows) else score(rows, cases),
                  "updated_at": datetime.now(timezone.utc).isoformat()}
        temporary = output / ".results.json.tmp"
        temporary.write_bytes(json_bytes(report))
        temporary.replace(output / "results.json")
        return report
    checkpoint("running")
    ledger = output / "attempts.jsonl"
    for index, trial in enumerate(plan["trials"]):
        case = by_id[trial["case_id"]]
        start = time.perf_counter()
        current, goal, measured = verified[(case["id"], trial["condition"])]
        # If the process stops inside the request, this marker prevents an
        # unreturned, potentially billed attempt from being called "not sent".
        checkpoint("running", {"case_id": case["id"], "condition": trial["condition"],
                               "outcome": "Request may have reached provider; result and charge unknown until receipt is recorded."})
        start = time.perf_counter()
        result = client.evaluate(current, goal, task=plan["task"], measured_state=measured)
        total = (time.perf_counter() - start) * 1000
        row = {**result, "case_id": case["id"], "condition": trial["condition"],
               "online_total_ms": total if result["status"] != "missing_credentials" else None,
               "decision": {"present": {"present": True, "absent": False}.get(result.get("presence")),
                            "complete": {"complete": True, "incomplete": False}.get(result.get("verdict"))}}
        if row["status"] == "missing_credentials":
            row["status"] = "not_run"
        row["timing_ms"] = {"total_ms": row["online_total_ms"],
                            "prepare_ms": row.get("latency_ms", {}).get("prepare"),
                            "request_ms": row.get("latency_ms", {}).get("http")}
        rows[index] = row
        with ledger.open("ab") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False).encode() + b"\n")
            stream.flush()
        checkpoint("running")
        print(json.dumps({"trial": index + 1, "total": len(plan["trials"]), "case_id": row["case_id"],
                          "condition": row["condition"], "status": row["status"]}), flush=True)
    pending = any(row["status"] == "not_run" for row in rows)
    return checkpoint("not_run" if pending else "completed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    make = sub.add_parser("generate")
    make.add_argument("--assets", required=True); make.add_argument("--output", required=True)
    seal = sub.add_parser("freeze")
    seal.add_argument("--fixtures", required=True); seal.add_argument("--source-commit", required=True)
    run = sub.add_parser("evaluate")
    run.add_argument("--fixtures", required=True); run.add_argument("--output", required=True)
    run.add_argument("--keychain-service"); run.add_argument("--keychain-account")
    args = parser.parse_args(argv)
    if args.action == "generate":
        from .image_goal_scenes import generate
        generate(args.assets, args.output)
    elif args.action == "freeze":
        from .image_goal_scenes import TASK
        plan = freeze(args.fixtures, task=TASK, source_commit=args.source_commit)
        print(json.dumps({"frozen_trials": len(plan["trials"]), "model": plan["model"]}))
    else:
        if bool(args.keychain_service) != bool(args.keychain_account):
            parser.error("Specify both Keychain service and account")
        from .jev_client import load_api_key
        key = load_api_key(keychain_service=args.keychain_service, keychain_account=args.keychain_account)
        evaluate(args.fixtures, args.output, api_key=key)


if __name__ == "__main__":
    main()

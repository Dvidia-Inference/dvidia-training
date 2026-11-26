#!/usr/bin/env python3
"""Independently cross-check downloaded table results using only the Python standard library."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from math import isclose
from pathlib import Path

REGISTRATION_SHA256 = "0eb3c1eeca94f26441038bed4e874e6500ef80eb237fe6c0326e87d43c6dc2f8"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(file):
    return sha256(file.read_bytes()).hexdigest()


def percentile(data, q):
    order = sorted(data)
    rank = q*(len(order)-1)
    lo = int(rank)
    return (1-(rank-lo))*order[lo] + (rank-lo)*order[min(lo+1, len(order)-1)]


def audit(protocol_path, out):
    require(file_hash(protocol_path) == REGISTRATION_SHA256, "Unknown registration digest")
    protocol = json.loads(protocol_path.read_text())
    summary = json.loads((out/"summary.json").read_text())
    require(summary["registration_sha256"] == REGISTRATION_SHA256, "Summary protocol mismatch")
    for filename in ("environment.json", "reference.json", "attempts.jsonl"):
        require(summary["files_sha256"][filename] == file_hash(out/filename), f"Bad digest for {filename}")
    saved = json.loads((out/"reference.json").read_text())
    require(saved["capture"]["observation"] == protocol["reference"]["initial"], "Wrong remembered table")
    rows = [json.loads(line) for line in (out/"attempts.jsonl").read_text().splitlines()]
    require(len(rows) == len(protocol["cases"]), "Missing or extra attempts")
    for row, fixture in zip(rows, protocol["cases"]):
        key = fixture["id"]
        ref = saved["references"][fixture["reference"]]
        require(row["reference_id"] == ref["id"], f"{key} reference mismatch")
        require((row["case_id"], row["group"], row["seed"], row["initial"], row["offsets_xy"])
                == (key, fixture["group"], fixture["seed"], fixture["initial"], fixture["offsets_xy"]),
                f"{key} fixture mismatch")
        expected_first_motion = None
        for i, move in enumerate(row["moves"]):
            require(row["decisions"][i]["plan"]["moves"][0] == move["requested"], f"{key} plan/issue mismatch")
            if move["first_motion_ms"] is not None and expected_first_motion is None:
                expected_first_motion = move["call_start_elapsed_ms"] + move["first_motion_ms"]
            if move["verified"]:
                require(move["settled_at_destination"] and move["bilateral_lift_dwell_s"] >= .1
                        and not move["critical_contact_native_ticks"], f"{key} unsupported verified move")
        require(row["time_to_first_move_ms"] == expected_first_motion, f"{key} first motion clock mismatch")
        if row["decisions"]:
            require(row["first_decision_ms"] == row["decisions"][0]["elapsed_wall_ms"], f"{key} decision clock")
        if expected_first_motion is not None:
            require(row["first_decision_ms"] <= expected_first_motion <= row["elapsed_wall_ms"],
                    f"{key} inconsistent clock order")
        score = row["native_score"]
        physical_success = False
        if score:
            physical_success = (all(score["assignments"][obj] in zones
                                    and score["settled_samples"][obj] >= 13
                                    for obj, zones in ref["allowed_zones"].items())
                                and score["critical_contact_native_ticks"] == 0)
            require(physical_success == score["success"], f"{key} native outcome disagrees")
        complete = physical_success and row["camera_claimed_tidy"] and row["status"] in ("completed", "tidy")
        require(complete == row["success"], f"{key} outcome mismatch")
        if fixture["group"].startswith("tidy_"):
            complete = complete and not row["moves"] and expected_first_motion is None
        require(complete == row["benchmark_success"], f"{key} incorrect qualification")
    groups = {"overall": rows,
              "already_tidy": [r for r in rows if r["group"].startswith("tidy_")],
              "shuffles": [r for r in rows if not r["group"].startswith("tidy_")]}
    for group in protocol["counts"]:
        groups[group] = [r for r in rows if r["group"] == group]
    for key, subset in groups.items():
        info = summary["groups"][key] if key in protocol["counts"] else summary[key]
        require(len(subset) == info["cases"] and sum(r["benchmark_success"] for r in subset)
                == info["successes"] and dict(Counter(r["status"] for r in subset)) == info["status_counts"],
                f"{key} incorrect totals")
        for value, field in (("time_to_first_move_ms", "first_motion_wall"), ("first_decision_ms", "first_decision_wall")):
            points = [r[value] for r in subset if r[value] is not None]
            require(info[field]["n"] == len(points), f"{key} {field} denominator mismatch")
            for q, name in ((.50, "p50_ms"), (.95, "p95_ms")):
                number = info[field][name]
                require(isclose(number, percentile(points, q), rel_tol=1e-12) if points else number is None,
                        f"{key} {field} wrong {name}")
    return dict(cases=len(rows), checked="registration, reference capture, move verification, independent end states, null motion and timing distributions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--registration", type=Path,
                        default=Path(__file__).resolve().parents[1]/"benchmarks/tidy-table-v01/registration.json")
    args = parser.parse_args()
    print(json.dumps(audit(args.registration, args.directory), indent=2))

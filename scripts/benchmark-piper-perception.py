#!/usr/bin/env python3
"""Run the six registered PiPER captures and compare their sealed evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


CANDIDATE = Path(__file__).resolve().parents[1]
PLAN = CANDIDATE / "benchmarks/piper-perception-latency-v03/plan.json"
TRACKER = Path("src/dvidia_training/piper_rgbd_tracker.py")
PROFILER = Path("scripts/profile-operator-capture.py")
COMPARATOR = Path("scripts/compare-operator-captures.py")
THREAD_SETTINGS = {
    name: "1" for name in (
        "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
    )
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, document):
    with path.open("x", encoding="utf-8") as output:
        json.dump(document, output, indent=2, allow_nan=False)
        output.write("\n")


def verify_sources(checkouts, plan):
    baseline = checkouts["baseline"]
    head = subprocess.check_output(["git", "-C", str(baseline), "rev-parse", "HEAD"], text=True)
    if head.strip() != plan["baseline_commit"]:
        raise ValueError("Baseline checkout does not match the registered commit.")
    subprocess.run(["git", "-C", str(baseline), "diff", "--quiet", "HEAD", "--",
                    "src/dvidia_training"], check=True)
    for variant, checkout in checkouts.items():
        if digest(checkout / TRACKER) != plan[f"tracker_{variant}_sha256"]:
            raise ValueError(f"The {variant} tracker differs from its registered source.")
        if digest(checkout / PROFILER) != plan["profiler_sha256"]:
            raise ValueError(f"The {variant} profiler differs from its registered source.")
    # Other modules imported by this capture must come from equivalent packages.
    files = subprocess.check_output(["git", "-C", str(baseline), "ls-files", "-z",
                                     "--", "src/dvidia_training"], text=True)
    for name in filter(None, files.split("\0")):
        if Path(name) != TRACKER and digest(baseline / name) != digest(CANDIDATE / name):
            raise ValueError(f"Source outside the tracker changed: {name}")


def run_script(script, checkout, arguments, workdir, log):
    env = dict(os.environ, PYTHONPATH=str(checkout / "src"), **THREAD_SETTINGS)
    with log.open("xb") as output:
        process = subprocess.run([sys.executable, str(script), *map(str, arguments)],
                                 cwd=workdir, env=env, stdout=output, stderr=subprocess.STDOUT)
    return process.returncode


def latency(intervals):
    seconds = np.array(intervals, dtype=float)
    return {
        "intervals": len(seconds),
        "deadline_misses_over_20ms": int(np.count_nonzero(seconds > .020)),
        "p50_ms": float(np.percentile(seconds, 50) * 1000),
        "p95_ms": float(np.percentile(seconds, 95) * 1000),
        "max_ms": float(np.max(seconds) * 1000),
    }


def summarize_variants(profiles):
    for variant in ("baseline", "candidate"):
        chosen = [profile for row, profile in profiles if row["variant"] == variant]
        intervals = [t for profile in chosen for t in profile["interval_seconds"]]
        cameras = [t for profile in chosen for t in profile["interval_seconds"][1::2]]
        yield variant, {
            "pooled_control": latency(intervals),
            "pooled_camera": latency(cameras),
            "control_mean_seconds": float(np.mean([p["control"]["total_seconds"] for p in chosen])),
            "tracker_mean_seconds_per_episode": float(np.mean([p["tracking"]["total_seconds"] for p in chosen])),
            "peak_rss_bytes_per_episode": [p["capture_peak_memory"]["bytes"] for p in chosen],
        }


def compare_pairs(profiles, pairs, workdir):
    for pair_index in range(3):
        selected = dict((row["variant"], (row, profile))
                        for row, profile in profiles[pair_index * 2:pair_index * 2 + 2])
        baseline, bp = selected["baseline"]
        candidate, cp = selected["candidate"]
        source_keys = set(bp["source_sha256"]) | set(cp["source_sha256"])
        differences = sorted(key for key in source_keys
                             if bp["source_sha256"].get(key) != cp["source_sha256"].get(key))
        pair = {"baseline": baseline["id"], "candidate": candidate["id"],
                "source_inventory_matches": set(bp["source_sha256"]) == set(cp["source_sha256"]),
                "changed_sources": differences}
        if not pair["source_inventory_matches"] or differences != [TRACKER.name]:
            raise ValueError("Paired profiler source inventories differ outside the tracker.")
        for name, key in (("control", "control"), ("tracker", "tracking")):
            before, after = bp[key]["total_seconds"], cp[key]["total_seconds"]
            pair[name] = {"baseline_seconds": before, "candidate_seconds": after,
                          "reduction_percent": 100 * (before - after) / before}
        parity_name = f"pair-{pair_index + 1}-parity.json"
        code = run_script(CANDIDATE / COMPARATOR, CANDIDATE,
                          ["--baseline", workdir / baseline["id"] / "capture",
                           "--candidate", workdir / candidate["id"] / "capture",
                           "--lossless-png", "--output", workdir / parity_name],
                          workdir, workdir / f"pair-{pair_index + 1}.log")
        pair["parity_exit_code"] = code
        if code == 0:
            proof = json.loads((workdir / parity_name).read_text())
            pair["parity"] = proof
            pair["exact_including_png_bytes"] = (
                proof["exact"] and proof["png_files_with_different_compressed_bytes"] == 0
                and proof["frame_bytes"][0] == proof["frame_bytes"][1]
            )
        pairs.append(pair)


def run(baseline, assets, output):
    plan = json.loads(PLAN.read_text())
    order = plan["order"]
    if order != ["baseline", "candidate", "candidate", "baseline", "baseline", "candidate"]:
        raise ValueError("Unexpected registered run order.")
    checkouts = {"baseline": baseline.resolve(strict=True), "candidate": CANDIDATE}
    verify_sources(checkouts, plan)
    duration = plan["simulated_duration_seconds"]
    assets = assets.resolve(strict=True)
    output = output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Choose a fresh output directory.")
    parent = output.parent.resolve(strict=True)
    output = parent / output.name
    # Same per-capture upper bound used by profile-operator-capture.py.
    upper = (round(duration / .04) + 1) * (320 * 240 * 7 + 4096) + 32 * 1024**2
    if shutil.disk_usage(parent).free < plan["reserve_bytes"] + len(order) * upper:
        raise ValueError("Insufficient free space for six captures plus the registered reserve.")
    output.mkdir()
    runs, pairs, profiles = [], [], []
    result = {"schema": "dvidia.piper-perception-native-benchmark/v1",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "plan_sha256": digest(PLAN), "runner_sha256": digest(Path(__file__)),
              "comparator_sha256": digest(CANDIDATE / COMPARATOR),
              "thread_settings_requested": THREAD_SETTINGS, "runs": runs, "pairs": pairs,
              "complete": False}
    write_json(output / "started.json", result)
    try:
        for index, variant in enumerate(order, 1):
            name = f"run-{index:02}-{variant}"
            row = {"id": name, "variant": variant}
            runs.append(row)
            print(f"Starting {name}", flush=True)
            checkout = checkouts[variant]
            code = run_script(checkout / PROFILER, checkout,
                              ["--assets", assets, "--output", output / name, "--duration", duration],
                              output, output / f"{name}.log")
            row["exit_code"] = code
            if code:
                continue
            path = output / name / "profile.json"
            profile = json.loads(path.read_text())
            expected_steps = round(duration * plan["control_hz"])
            expected_frames = round(duration * plan["camera_hz"]) + 1
            if (not profile["complete"] or profile["profile_enabled"] or profile["teacher_queries"]
                    or len(profile["interval_seconds"]) != expected_steps
                    or profile["camera_pairs"] != expected_frames
                    or profile["recorded_intervals"] != expected_steps
                    or profile["source_sha256"][TRACKER.name] != plan[f"tracker_{variant}_sha256"]
                    or profile["source_sha256"][PROFILER.name] != plan["profiler_sha256"]):
                raise ValueError(f"{name} differs from the registered capture.")
            row["profile_sha256"] = digest(path)
            for key in ("system", "python", "numpy", "numerical_runtime", "control",
                        "camera_control", "tracking", "open_seconds", "finish_seconds",
                        "total_seconds", "control_cpu_seconds", "total_cpu_seconds", "capture_peak_memory",
                        "capture_bytes", "capture_integrity_sha256", "strict_success", "recorded_intervals",
                        "camera_pairs", "render_encode", "offline_frame_processing_age"):
                row[key] = profile[key]
            profiles.append((row, profile))
        if len(profiles) == len(order):
            verify_sources(checkouts, plan)
            result["variants"] = dict(summarize_variants(profiles))
            compare_pairs(profiles, pairs, output)
            result["complete"] = all(pair.get("exact_including_png_bytes") for pair in pairs)
    finally:
        result["ended_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "results.json", result)
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(run(args.baseline, args.assets, args.output))

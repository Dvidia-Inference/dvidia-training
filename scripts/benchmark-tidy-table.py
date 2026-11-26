#!/usr/bin/env python3
"""Run the registered PiPER remembrance and rearrangement benchmark, once per output."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess

from dvidia_training import tidy_table

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmarks/tidy-table-v01/registration.json"
EXPECTED_DIGEST = "0eb3c1eeca94f26441038bed4e874e6500ef80eb237fe6c0326e87d43c6dc2f8"


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def git_commit():
    result = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def cpu_model():
    if platform.system() == "Darwin":
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    if platform.system() == "Linux" and Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[-1].strip()
    return platform.processor() or None


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentiles(values):
    data = sorted(x for x in values if x is not None)
    if not data:
        return dict(n=0, p50_ms=None, p95_ms=None)
    def quantile(q):
        index = (len(data)-1)*q
        low = int(index)
        return data[low] + (index-low)*(data[min(low+1, len(data)-1)]-data[low])
    return dict(n=len(data), p50_ms=quantile(.50), p95_ms=quantile(.95))


def summarize(rows):
    motions = percentiles([r.get("time_to_first_move_ms") for r in rows])
    decisions = percentiles([r.get("first_decision_ms") for r in rows])
    return dict(cases=len(rows), successes=sum(r["benchmark_success"] for r in rows),
                status_counts=dict(Counter(r["status"] for r in rows)),
                issued_moves=sum(len(r.get("moves", [])) for r in rows),
                moved_cases=motions["n"], no_recorded_first_motion=len(rows)-motions["n"],
                first_motion_wall=motions, first_decision_wall=decisions,
                no_move_calls=sum(not r.get("moves") for r in rows),
                verified_no_op=sum(r.get("status") == "tidy" and r["benchmark_success"] for r in rows),
                camera_false_completions=sum(bool(r.get("camera_claimed_tidy") and
                    r.get("native_score") and not r["native_score"]["success"]) for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path,
                        help="A new directory for the reference, per-case results and summary")
    args = parser.parse_args()
    if file_sha(PROTOCOL) != EXPECTED_DIGEST:
        parser.error("The registered fixture hash differs from the published protocol.")
    if not args.assets.is_dir():
        parser.error("Select the existing PiPER asset directory.")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Results already exist. Select an empty directory; prior evidence is preserved.")
    args.out.mkdir(parents=True, exist_ok=True)
    series = json.loads(PROTOCOL.read_text())
    ref_spec = series["reference"]
    # Capture the visual reference once. Then derive both rule sets from that same frame.
    approved = tidy_table.capture_table(ref_spec["initial"], assets=args.assets,
        seed=ref_spec["seed"], offsets_xy=ref_spec["offsets_xy"])
    if approved["observation"] != ref_spec["initial"]:
        dump(args.out / "invalid-reference.json", approved)
        raise RuntimeError("The observed reference disagrees with the registered approved table.")
    references = {key: tidy_table.remember_table(approved["observation"], ref_spec[key])
                  for key in ("exact", "equivalent")}
    dump(args.out / "reference.json", dict(capture=approved, references=references))
    import dvidia_training.tidy_table_native as native
    files = [Path(__file__), Path(tidy_table.__file__), Path(native.__file__)]
    info = dict(schema="dvidia.tidy-table-benchmark/v1", registration_sha256=EXPECTED_DIGEST,
                started_utc=datetime.now(timezone.utc).isoformat(), git_commit=git_commit(),
                source_sha256={path.name: file_sha(path) for path in files},
                system=platform.system(), os_release=platform.release(), machine=platform.machine(), cpu_model=cpu_model(),
                python=platform.python_version(), mujoco=version("mujoco"), numpy=version("numpy"),
                timing_scope=series["timing"], worker_count=1)
    dump(args.out / "environment.json", info)
    rows = []
    with (args.out / "attempts.jsonl").open("x") as evidence:
        for case in series["cases"]:
            row = tidy_table.run_table_case(references[case["reference"]], case["initial"],
                    assets=args.assets, seed=case["seed"], offsets_xy=case["offsets_xy"])
            needs_noop = case["group"].startswith("tidy_")
            row.update(case_id=case["id"], group=case["group"], reference_kind=case["reference"],
                       benchmark_success=bool(row["success"] and (not needs_noop or
                           (not row["moves"] and row["time_to_first_move_ms"] is None))))
            evidence.write(json.dumps(row, separators=(",", ":"), allow_nan=False)+"\n")
            evidence.flush()
            os.fsync(evidence.fileno())
            rows.append(row)
            print(f"{row['case_id']}: {row['status']} ({len(row['moves'])} move calls)", flush=True)
    groups = {group: summarize([r for r in rows if r["group"] == group])
              for group in series["counts"]}
    summary = dict(**info, finished_utc=datetime.now(timezone.utc).isoformat(), overall=summarize(rows),
                   shuffles=summarize([r for r in rows if not r["group"].startswith("tidy_")]),
                   already_tidy=summarize([r for r in rows if r["group"].startswith("tidy_")]),
                   groups=groups, files_sha256={name: file_sha(args.out/name) for name in
                   ("reference.json", "attempts.jsonl", "environment.json")})
    dump(args.out / "summary.json", summary)
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()

"""Summarize and verify the frozen paired confirmation receipts."""
import argparse
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

CONTROLLERS = {"confirmation-candidate": "corner-v8", "confirmation-fixed": "fixed",
               "confirmation-timed": "corner-v8-timed"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def summarize(root, study):
    roster = read(study / "confirmation.json")
    roster_sha = digest(study / "confirmation.json")
    calibration_sha = digest(study / "workspace/nominal.json")
    assert roster_sha == read(study / "plan.json")["roster_sha256"]["confirmation"]
    summaries, results = {}, {}
    for folder, controller in CONTROLLERS.items():
        directory = root / folder
        run = read(directory / "run.json")
        assert run["registered_cases"] == roster and run["roster_sha256"] == roster_sha
        assert run["retry_count"] == 0 and run["controller"] == controller
        assert len(list(directory.glob("attempt-????.json"))) == len(roster)
        rows = []
        for index, case in enumerate(roster):
            row = read(directory / f"attempt-{index:04}.json")
            assert row["case"] == case and row["calibration_file_sha256"] == calibration_sha
            assert row["controller"] == controller and row["negative"] is None
            assert row["status"] == "completed" and row["valid"]
            assert row["native_ticks"] == 20000 and row["control_steps"] == 1000
            assert row["teacher_calls"] == 0 and row["initial_camera_pairs_delivered"] == 1
            assert row["success"] == row["score"]["native_recomputed"]["success"]
            tape = row["action_log"]
            assert tape["shape"] == [1000, 7] and digest(directory / tape["path"]) == tape["sha256"]
            rows.append(row)
        results[controller] = rows
        successes = [r for r in rows if r["success"]]
        med = statistics.median
        by_mass = {}
        for mass in sorted({r["case"]["mass_kg"] for r in rows}):
            group = [r for r in rows if r["case"]["mass_kg"] == mass]
            by_mass[str(mass)] = [sum(r["success"] for r in group), len(group)]
        summaries[controller] = dict(placements=[len(successes), len(rows)], by_mass_kg=by_mass,
            failure_stages=dict(Counter(r["failure_stage"] for r in rows if not r["success"])),
            successful_time_median_s=med(r["first_success_time_s"] for r in successes) if successes else None,
            failure_censored_time_mean_s=statistics.mean(r["first_success_time_s"] if r["success"] else 20 for r in rows),
            successful_final_xy_median_mm=med(r["final_xy_error_m"]*1000 for r in successes) if successes else None,
            critical_contact_episodes=sum(r["critical_contact_native_ticks"] > 0 for r in rows),
            maximum_pad_force_n=max(max(r["peak_pad_force_n"].values()) for r in rows),
            median_attempt_wall_s=med(r["attempt_wall_s"] for r in rows),
            median_episode_control_loop_p95_ms=med(r["control_loop_wall"]["p95_ms"] for r in rows),
            median_episode_inference_p95_ms=med(r["inference_wall"]["p95_ms"] for r in rows))
    full = results["corner-v8"]
    paired = {}
    for name in ("fixed", "corner-v8-timed"):
        groups = defaultdict(list)
        for a,b in zip(full, results[name], strict=True):
            assert a["initial_frame_sha256"] == b["initial_frame_sha256"]
            groups[a["case"]["group_id"]].append(int(a["success"]) - int(b["success"]))
        assert len(groups) == 18 and all(len(g)==2 for g in groups.values())
        deltas = [statistics.mean(g) for g in groups.values()]
        rng = random.Random(11252602)
        boot = sorted(sum(rng.choices(deltas, k=len(deltas)))/len(deltas) for _ in range(10000))
        paired[name] = dict(mean_strict_rate_difference=statistics.mean(deltas),
            grouped_bootstrap_95=[boot[249],boot[9749]], grouped_bootstrap_seed=11252602,
            identical_actuator_tapes=sum(a["action_log"]["sha256"]==b["action_log"]["sha256"]
                for a,b in zip(full,results[name],strict=True)))
    return dict(scope="simulation", controllers=summaries, candidate_minus_comparator=paired)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.receipts, Path(__file__).resolve().parent), indent=2))

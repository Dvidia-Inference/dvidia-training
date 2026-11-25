"""Headless verification and summary for the supplementary frozen-skill study."""
import argparse
import collections
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import time

from dvidia_training.piper_first_attempt_replay import replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--receipt-dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wait", type=float, default=0., help="Wait up to this many seconds for receipts")
    args = parser.parse_args()
    plan = Path(__file__).with_name("registration.json")
    h = sha256(plan.read_bytes()).hexdigest()
    cases = json.loads(plan.read_text())["cases"]
    validated = {}
    args.output.mkdir(parents=True,exist_ok=False)
    deadline = time.monotonic() + args.wait
    while True:
        for case in cases:
            key=case["id"]
            if key in validated:
                continue
            paths=[folder/(key+"-learned_camera_yaw.json") for folder in args.receipt_dirs]
            found=[p for p in paths if p.exists()]
            if len(found)>1:
                raise ValueError(f"Multiple receipts for {key}")
            if not found:
                continue
            try:
                result=json.loads(found[0].read_text())
            except json.JSONDecodeError:
                continue
            if result["yaw_plan_sha256"]!=h or result["case"]!={k:v for k,v in case.items() if k!="yaw_degrees"}:
                raise ValueError("Unregistered receipt "+key)
            if result["status"]=="completed":
                # All registered scenes used the stock upright identity quaternion.
                check=replay(args.assets,found[0])
            else:
                check=dict(reproduced=False,mismatches=[result["reason"]])
            info=dict(id=key,group=case["group_id"],goal=case["goal_id"],mass_kg=case["mass_kg"],friction=case["friction"],
                strict_success=result["success"],status=result["status"],failure_stage=result["failure_stage"],
                source_receipt_sha256=sha256(found[0].read_bytes()).hexdigest(),reproduced=check["reproduced"],
                mismatches=check["mismatches"],critical_ticks=result.get("critical_contact_native_ticks"),
                pad_peak_n=max(result.get("peak_pad_force_n",{}).values(),default=None),
                completion_seconds=result.get("first_success_time_s"),final_xy_error_m=result.get("final_xy_error_m"))
            with (args.output/(key+".json")).open("x") as out:
                json.dump(info,out,indent=2,sort_keys=True)
                out.write("\n")
            validated[key]=info
            print(key,info["strict_success"],info["reproduced"],flush=True)
        if len(validated)==len(cases) or time.monotonic()>=deadline:
            break
        time.sleep(1.)
    values=list(validated.values())
    successes=[row for row in values if row["strict_success"]]
    summary=dict(planned=len(cases),attempts_received=len(values),fully_reproduced=sum(row["reproduced"] for row in values),
        successes=len(successes),critical_episodes=sum(bool(row["critical_ticks"]) for row in values),
        failures=dict(collections.Counter(row["failure_stage"] for row in values if not row["strict_success"])),
        median_completion_s=median(row["completion_seconds"] for row in successes) if successes else None,
        median_placed_xy_mm=1000*median(row["final_xy_error_m"] for row in successes) if successes else None,
        max_pad_force_n=max((row["pad_peak_n"] for row in values if row["pad_peak_n"] is not None),default=None))
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()

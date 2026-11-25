"""Run the published local skill on each registered upright confirmation start once."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
SKILL = BENCH / "local-skill/piper-camera-yaw-skill-0.1.0.zip"
EVIDENCE = BENCH / "local-skill/evidence.zip"
PLAN = HERE / "registration.json"
STUDY = "runs/first-attempt/corner-distilled-yaw"
CONTROLLER = "learned_camera_yaw"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as writer:
        json.dump(value, writer, allow_nan=False, sort_keys=True, indent=2)
        writer.write("\n")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def register():
    conditions = json.loads((BENCH / "confirmation.json").read_text())
    assert len(conditions) == 36 and len({c["group_id"] for c in conditions}) == 18
    hashes = {str(p.relative_to(BENCH)): digest(p) for p in
              (HERE/"run.py", BENCH/"confirmation.json", BENCH/"workspace/nominal.json", SKILL, EVIDENCE)}
    save(PLAN, dict(schema="dvidia.supplemental-learned-confirmation/v1",
        registered_utc=datetime.now(timezone.utc).isoformat(), base_commit="e13ff91", portable_skill_commit="4ebdee2",
        method="Supplementary, frozen locally packaged 32-demo model plus its camera yaw adapter, untouched terminal prediction entrypoint and validation. Each case at nominal upright yaw, one execution, both goals per group, no teacher, no tuning, no retry. One initial RGB-D observation and 50 Hz measured encoders, 20 s horizon, strict independent 1 ms physical scorer. The primary authors may already be running their separate controllers; their results were not inspected.",
        files=hashes, controllers=[CONTROLLER], cases=[dict(c,yaw_degrees=0.) for c in conditions]))


def run(args):
    plan = json.loads(PLAN.read_text())
    for name, expected in plan["files"].items():
        if digest(BENCH / name) != expected:
            raise ValueError("Preregistration mismatch " + name)
    args.output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="piper-supplemental-") as temp:
        folder = Path(temp)
        with ZipFile(SKILL) as archive:
            archive.extractall(folder)
        sys.path.insert(0,str(folder))
        with ZipFile(EVIDENCE) as evidence:
            (folder/"driver.py").write_bytes(evidence.read(f"{STUDY}/driver.py"))
        terminal = load(folder/"run_yaw.py", "frozen_terminal")
        bundle = terminal.Bundle()
        driver = load(folder/"driver.py", "frozen_driver")
        class PackagedController:
            def __init__(self, base, *, adaptive):
                assert adaptive
                self.actual = bundle.controller()

            def __getattr__(self, key):
                return getattr(self.actual, key)

            def __call__(self, obs):
                keys = terminal.SAMPLE_KEYS | (terminal.INITIAL_KEYS if self.actual.base.index == 0 else set())
                return terminal.predict(self.actual, {k: obs[k] for k in keys})

        driver.InitialYawPolicy = PackagedController
        params = SimpleNamespace(assets=args.assets, model=folder/terminal.MODEL, policy_source=folder/"policy.py",
            plan=PLAN, calibration=BENCH/"workspace/nominal.json", policy_class=driver.load_policy(folder/"policy.py"))
        attempts = []
        for case in plan["cases"]:
            if not args.begin <= int(case["group_id"].split("-")[-1]) < args.end:
                continue
            row=driver.run(params,case,CONTROLLER,args.output)
            row["installed_skill_sha256"] = digest(SKILL)
            row["supplemental_driver_sha256"] = digest(Path(__file__))
            save(args.output/(case["id"]+"-"+CONTROLLER+".json"), row)
            attempts.append(dict(case=case["id"],strict_success=row["success"],status=row["status"],
                                 stage=row["failure_stage"],reason=row["reason"]))
            print(json.dumps(attempts[-1]),flush=True)
        save(args.output/"summary.json",attempts)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--register",action="store_true")
    p.add_argument("--assets",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--begin",type=int,default=0)
    p.add_argument("--end",type=int,default=18)
    args=p.parse_args()
    if args.register:
        register()
    elif args.assets and args.output and 0 <= args.begin < args.end <= 18:
        run(args)
    else:
        p.error("Supply --assets and a fresh --output, with a valid group range.")

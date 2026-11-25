"""Run one camera-initialized placement in the local PiPER physics simulator."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
SKILL = "piper-camera-yaw-skill-0.1.0.zip"
EVIDENCE = "evidence.zip"
SKILL_HASH = "45a8636d2e90db9a4643a73268cf0463b6943d4d63f97ec93fff5ec0b470b1f9"
EVIDENCE_HASH = "cac4c33eccdf0ef95adfccdcd0ac3647c2270c1cdd0562cdfcb5f7cc759e43de"
STUDY = "runs/first-attempt/corner-distilled-yaw"
PLAN = f"{STUDY}/protocol.json"
CALIBRATION = "benchmarks/piper-first-attempt-v01/workspace/nominal.json"
CONTROLLER = "learned_camera_yaw"


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def copy_member(archive, name, destination):
    destination.write_bytes(archive.read(name))
    return destination


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute(args, evidence, skill, folder, case, reference):
    import dvidia_training

    # Keep installed package data and the native scorer available in the
    # isolated source tree. Leave the installation itself untouched.
    original = Path(dvidia_training.__file__).resolve().parent
    overlay = folder / "dvidia_training"
    shutil.copytree(original, overlay, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for source in ("piper_first_attempt_eval.py", "piper_first_attempt_policy.py",
                   "piper_first_attempt_replay.py", "piper_workspace_calibration.py"):
        copy_member(evidence, f"src/dvidia_training/{source}", overlay / source)
    for source, expected in reference["source_sha256"].items():
        if not (overlay / source).is_file() or digest(overlay / source) != expected:
            raise ValueError(f"Installed simulator source differs from the benchmark: {source}")
    dvidia_training.__path__ = [str(overlay)]

    model = copy_member(skill, "model/camera-xy-primitive.npz", folder / "model.npz")
    policy = copy_member(skill, "policy.py", folder / "policy.py")
    copy_member(skill, "adapter.py", folder / "adapter.py")
    plan = copy_member(evidence, PLAN, folder / "protocol.json")
    calibration = copy_member(evidence, CALIBRATION, folder / "calibration.json")
    driver_path = copy_member(evidence, f"{STUDY}/driver.py", folder / "driver.py")
    sys.path.insert(0, str(folder))
    driver = import_file("benchmark_driver", driver_path)
    setup = SimpleNamespace(assets=args.assets.resolve(), model=model, plan=plan,
                            policy_source=policy, calibration=calibration,
                            policy_class=driver.load_policy(policy))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipt = driver.run(setup, case, CONTROLLER, output)
    receipt["skill_archive_sha256"] = SKILL_HASH
    receipt["matches_registered_commands"] = (
        receipt.get("executed_commands_sha256") is not None
        and receipt.get("executed_commands_sha256") == reference["executed_commands_sha256"])
    with (output / "receipt.json").open("x") as handle:
        json.dump(receipt, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(dict(case=case["id"], status=receipt["status"], strict_success=receipt["success"],
                          reason=receipt["reason"], simulated_completion_s=receipt.get("first_success_time_s"),
                          matches_registered_commands=receipt["matches_registered_commands"],
                          receipt=str(output / "receipt.json")), indent=2))
    return 0 if receipt["status"] == "completed" and receipt["success"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, help="Pinned native PiPER asset directory")
    parser.add_argument("--output", type=Path, help="Fresh directory for the receipt and actual commands")
    parser.add_argument("--case", default="development-01-g0-yaw+24", help="Registered simulated scene")
    parser.add_argument("--list", action="store_true", help="List available scene IDs")
    args = parser.parse_args()
    if digest(ROOT / EVIDENCE) != EVIDENCE_HASH or digest(ROOT / SKILL) != SKILL_HASH:
        parser.error("Skill or evidence archive differs from the verified local release.")
    with ZipFile(ROOT / EVIDENCE) as evidence, ZipFile(ROOT / SKILL) as skill:
        cases = {entry["id"]: entry for entry in json.loads(evidence.read(PLAN))["cases"]}
        if args.list:
            print("\n".join(cases))
            return 0
        if args.case not in cases:
            parser.error("Unknown case. Use --list to see the 24 registered scene IDs.")
        if args.assets is None or args.output is None:
            parser.error("--assets and --output are required to simulate.")
        if not args.assets.is_dir() or args.output.exists():
            parser.error("Provide an existing asset directory and a fresh output directory.")
        ref_name = f"{STUDY}/development/{args.case}-{CONTROLLER}.json"
        reference = json.loads(evidence.read(ref_name))
        with tempfile.TemporaryDirectory(prefix="piper-local-sim-") as temporary:
            return execute(args, evidence, skill, Path(temporary), cases[args.case], reference)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, ImportError, OSError) as error:
        raise SystemExit(f"Local PiPER simulator: {error}")

"""Read-only evidence audit and deterministic public kit for USD workstation v0.1.

This reconstructs recorded 10 ms trajectory scores and checks archived data
consistency. It never steps or recompiles MuJoCo, authenticates physical truth,
or reconstructs the unrecorded 1 ms goal-monitor observations.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import gzip
import io
import json
import math
from pathlib import Path
import re
import shutil
import zipfile

from dvidia_training import asset_preview, robot_profiles, usd_scene as bridge
from dvidia_training import usd_scene_benchmark as bench


SCHEMA = "dvidia.usd-workstation-publication/v1"
ROOT = Path(__file__).resolve().parents[1]
MAX_FILE = 16*1024*1024
MAX_TOTAL = 64*1024*1024
COMPILED_CHECKS = {"object_mass", "object_inertia", "object_center_of_mass", "object_inertia_axes",
    "object_gravity_compensation_disabled", "object_size", "table_size", "object_position",
    "object_orientation", "table_position", "table_orientation", "gravity", "object_friction",
    "table_friction", "free_object", "native_dimensions"}
ROOT_STUDY_FILES = {"execution-intent.json", "geometry.bin.gz", "input-inventory.json", "model.json",
    "motion.json", "preparation-seal.json", "protocol.json", "report.json"}
CASE_FILES = {"started.json", "receipt.json", "trajectory.json", "poses.bin.gz", "result.json",
              "input/workstation.usda", "input/recipe.json"}


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False)+"\n").encode()


def identity(raw):
    return {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}


def read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected a regular evidence file")
    with path.open("rb") as stream:
        raw = stream.read(MAX_FILE+1)
    if len(raw) > MAX_FILE:
        raise ValueError("Evidence file exceeds the public byte limit")
    return raw


def parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate evidence JSON key")
            result[key] = value
        return result
    def constant(_):
        raise ValueError("Nonfinite evidence JSON value")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def unpack(raw):
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        result = stream.read(MAX_FILE+1)
    if len(result) > MAX_FILE:
        raise ValueError("Compressed evidence exceeds its expanded byte limit")
    return result


def inventory(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Expected a local evidence directory")
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise ValueError("Public evidence cannot contain symlinks or special files")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = identity(read(path))
    if sum(item["bytes"] for item in result.values()) > MAX_TOTAL:
        raise ValueError("Public evidence exceeds the total byte limit")
    return result


def public_text_check(name, raw):
    if not name.endswith((".json", ".py", ".md", ".txt", ".html", ".js", ".usda", "LICENSE")):
        return
    raw.decode("utf-8")
    private = (b"/"+b"Users/", b"/"+b"private/", b"/"+b"var/folders/", b"api"+b"key_")
    if any(pattern in raw for pattern in private) or re.search(rb"(?:hf_|ghp_)[A-Za-z0-9]{24,}", raw):
        raise ValueError("Public text contains a private path or credential-shaped value")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_study(study):
    """Re-score all retained attempts without executing a native rollout."""
    import numpy as np
    root = Path(study)
    files = inventory(root)
    protocol = parse(read(root/"protocol.json"))
    seal = parse(read(root/"preparation-seal.json"))
    require(seal == {"schema": bench.SCHEMA, "protocol": identity(read(root/"protocol.json")),
                     "inputs": identity(read(root/"input-inventory.json"))}, "Preparation seal mismatch")
    require(protocol["schema"] == bench.SCHEMA and protocol["cases"] == bench.roster(), "Frozen roster mismatch")
    require(protocol["source"] == bench.sources(), "Installed runtime differs from the frozen source")
    expected = set(ROOT_STUDY_FILES) | {"source/"+name for name in protocol["source"]}
    expected |= {"cases/"+case["id"]+"/"+name for case in protocol["cases"] for name in CASE_FILES}
    require(set(files) == expected, "Study evidence membership differs from the complete 45-case archive")
    for name, want in protocol["source"].items():
        require(files["source/"+name] == want, "Retained runtime source mismatch")
    inputs = parse(read(root/"input-inventory.json"))
    require(inputs["schema"] == bench.SCHEMA, "Input inventory schema mismatch")
    expected_inputs = {name for name in files if "/input/" in name}
    require(set(inputs["files"]) == expected_inputs, "Input inventory membership mismatch")
    for name, want in inputs["files"].items():
        require(files[name] == want, "Frozen USD input mismatch")
    intent = parse(read(root/"execution-intent.json"))
    require(intent == {"schema": bench.SCHEMA, "protocol": seal["protocol"], "inputs": seal["inputs"],
                       "scheduled": 45, "worker_count": 1}, "Execution intent mismatch")
    report = parse(read(root/"report.json"))
    motion = parse(read(root/"motion.json"))
    model = parse(read(root/"model.json"))
    require(report["schema"] == motion["schema"] == bench.SCHEMA, "Output schema mismatch")
    require(len(report["results"]) == len(motion["cases"]) == 45, "Output attempt denominator mismatch")
    require(report["protocol_sha256"] == seal["protocol"]["sha256"], "Report protocol identity mismatch")
    bodies = model["bodies"]
    require([body["id"] for body in bodies] == list(range(len(bodies))), "Native body order mismatch")
    require(motion["body_count"] == len(bodies) == 11, "PiPER body count mismatch")
    objects = [body["id"] for body in bodies if body["name"] == "usd_object"]
    require(len(objects) == 1 and bodies[0]["name"] == "world", "Object/world body identity mismatch")
    object_index = objects[0]
    geometry = unpack(read(root/"geometry.bin.gz"))
    require(model["geometry_binary"] == {"href": "geometry.bin.gz", "compression": "gzip", **identity(geometry)},
            "Native geometry payload identity mismatch")
    cases = []
    max_xyz_error = 0.
    max_quaternion_error = 0.
    recorded_frames = 0
    for index, case in enumerate(protocol["cases"]):
        folder = root/"cases"/case["id"]
        started = parse(read(folder/"started.json"))
        require(started == {"case": case, "protocol_sha256": seal["protocol"]["sha256"], "source": protocol["source"]},
                "Attempt start identity mismatch")
        receipt = parse(read(folder/"receipt.json"))
        recipe = bridge.load_recipe(folder/"input/recipe.json")
        parsed_usd = bridge.inspect_stage(folder/"input/workstation.usda")
        require(receipt["schema"] == bridge.RECEIPT_SCHEMA and receipt["recipe"] == recipe and receipt["usd"] == parsed_usd,
                "Recorded recipe or USD properties mismatch")
        require(receipt["recipe_sha256"] == files["cases/"+case["id"]+"/input/recipe.json"]["sha256"],
                "Recipe byte identity mismatch")
        require(parsed_usd["stage_sha256"] == recipe["stage_sha256"], "Parsed USD byte identity mismatch")
        compiled = receipt["compiled_checks"]
        require(set(compiled) == COMPILED_CHECKS and all(value is True for value in compiled.values()),
                "Missing or failed compiled property check")
        trace = parse(read(folder/"trajectory.json"))
        require(trace["schema"] == bench.SCHEMA, "Trajectory schema mismatch")
        rows = trace["rows"]
        result = parse(read(folder/"result.json"))
        rescored = bench.score(case, rows, protocol)
        comparable = {key: value for key, value in result["checks"].items() if key != "no_native_warning"}
        require(comparable == rescored["checks"] and result["metrics"] == rescored["metrics"]
                and result["valid"] == rescored["valid"], "Recorded trajectory score mismatch")
        require(set(result["checks"]) == set(rescored["checks"]) | {"no_native_warning"}, "Unexpected score checks")
        require(isinstance(result["checks"]["no_native_warning"], bool), "Malformed native warning claim")
        require(result["success"] == all(result["checks"].values()), "Result success differs from its checks")
        require(result["case"] == case and result["control"] == case["control"]
                and result["compiled_property_checks"] == compiled and result["native_timestep_s"] == .001,
                "Result case or native property identity mismatch")
        require(report["results"][index] == result, "Report result ordering or bytes mismatch")
        replay_case = motion["cases"][index]
        require(replay_case["case"] == case and replay_case["result"] == result and replay_case["rows"] == rows,
                "Replay result/trajectory mismatch")
        require(replay_case["pose_file"] == "cases/"+case["id"]+"/poses.bin.gz", "Replay pose path mismatch")
        pose_raw = unpack(read(folder/"poses.bin.gz"))
        require(identity(pose_raw) == replay_case["pose_identity"], "Replay pose payload identity mismatch")
        require(len(pose_raw) == len(rows)*len(bodies)*7*4, "Recorded pose frame count mismatch")
        poses = np.frombuffer(pose_raw, dtype="<f4").reshape((len(rows), len(bodies), 7))
        require(bool(np.isfinite(poses).all()), "Nonfinite replay pose")
        xyz = np.asarray([row["position_m"] for row in rows], dtype=np.float64)
        xyz_error = float(np.max(np.abs(poses[:, object_index, :3].astype(np.float64)-xyz)))
        quaternion_error = float(np.max(np.abs(np.linalg.norm(poses[:, :, 3:].astype(np.float64), axis=2)-1)))
        require(xyz_error <= 1e-6, "Recorded object XYZ differs from native trajectory")
        require(quaternion_error <= 1e-6, "Recorded quaternion is not unit length")
        max_xyz_error = max(max_xyz_error, xyz_error)
        max_quaternion_error = max(max_quaternion_error, quaternion_error)
        recorded_frames += len(rows)
        cases.append({"case": case["id"], "valid": rescored["valid"], "trajectory_checks_match": True,
                      "recorded_success": result["success"], "frames": len(rows), "usd_properties_match": True,
                      "compiled_checks_present": 16, "object_xyz_error_m": xyz_error,
                      "quaternion_norm_error": quaternion_error})
    counts = {"scheduled": 45, "valid": sum(item["valid"] for item in report["results"]),
        "passed": sum(item["success"] for item in report["results"]),
        "normal_scheduled": 36, "normal_passed": sum(item["success"] for item in report["results"] if item["case"]["control"] == "normal"),
        "negative_scheduled": 9, "negative_passed": sum(item["success"] for item in report["results"] if item["case"]["control"] != "normal")}
    require(all(report[key] == value for key, value in counts.items()), "Report denominator or count mismatch")
    require(math.isclose(report["wall_seconds"], sum(item["wall_seconds"] for item in report["results"]), abs_tol=1e-12),
            "Recorded wall timing aggregation mismatch")
    for name in files:
        public_text_check(name, read(root/name))
    return {"schema": SCHEMA, "audit_kind": "read-only recorded evidence consistency", "counts": counts,
        "protocol": seal["protocol"], "study_files": len(files), "study_bytes": sum(value["bytes"] for value in files.values()),
        "recorded_frames": recorded_frames, "frame_interval_s": .01, "pose_dtype": "little-endian float32 wxyz",
        "body_count": 11, "max_object_xyz_error_m": max_xyz_error, "max_quaternion_norm_error": max_quaternion_error,
        "mismatches": 0, "cases": cases,
        "limitations": ["This checks 10 ms saved trajectories and body poses; it does not authenticate physics or physical truth.",
            "The 1 ms native goal-monitor dwell and native-warning claims cannot be reconstructed from 10 ms samples.",
            "Sixteen compiled property checks are retained claims and checked for consistency; this audit does not recompile models.",
            "USD properties are independently parsed again; no new rollout or policy execution is performed."],
        "auditor": identity(read(Path(__file__))), "runtime_source": protocol["source"]}


def deterministic_zip(folder, target):
    root, target = Path(folder), Path(target)
    if target.exists() or root in target.parents:
        raise ValueError("Use a fresh archive path outside the public directory")
    entries = inventory(root)
    with target.open("xb") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in entries:
            member = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = 0o100644 << 16
            archive.writestr(member, read(root/name))
    return {"files": len(entries), **identity(read(target))}


def prepare(study, assets, output):
    root, destination = Path(study), Path(output)
    archive = destination.with_suffix(".zip")
    if destination.exists() or destination.is_symlink() or archive.exists():
        raise ValueError("Use a fresh public directory and archive; nothing is overwritten")
    audited = audit_study(root)
    provenance, notices = asset_preview._licenses("piper", assets)
    require(all(parse(read(root/"cases"/case["id"]/"receipt.json"))["robot"] == provenance for case in bench.roster()),
            "Verified local asset provenance differs from the study")
    shutil.copytree(root, destination)
    additions = {"index.html": read(ROOT/"src/dvidia_training/usd_scene_viewer.html"),
        "usd_scene_viewer.bundle.js": read(ROOT/"src/dvidia_training/usd_scene_viewer.bundle.js"),
        "LICENSE": read(ROOT/"LICENSE"),
        "THIRD_PARTY_NOTICES.txt": read(ROOT/"src/dvidia_training/reference_replay.third-party.txt"),
        "docs/usd-workstation-v01.md": read(ROOT/"docs/usd-workstation-v01.md"),
        "environment.json": read(ROOT/"runs/usd-scene-tools/environment-20261030.json"),
        "tools/prepare_usd_workstation_v01.py": read(Path(__file__)),
        "audit.json": encoded(audited), **notices}
    additions["README.md"] = ("# DVIDIA USD workstation v0.1\n\n"
        "A recorded PiPER workstation and complete passive physics benchmark: 45 valid cases, "
        "36 normal checks and 9 deliberately disabled-physics controls. These are physics checks, not a robot skill success rate.\n\n"
        "Serve this folder with `python -m http.server 8000` and open `http://127.0.0.1:8000/`. "
        "The browser replays recorded native poses; it does not run physics. See `docs/usd-workstation-v01.md` "
        "for source installation, pinned assets and native CLI reproduction. Frozen sources here are evidence snapshots, "
        "not a complete Python distribution.\n\n"
        "`audit.json` reconstructs all 45 recorded trajectory scores and checks the saved pose/trace/receipt identities. "
        "It cannot reconstruct the unrecorded 1 ms goal-monitor interval or authenticate physical truth. "
        "`manifest.json` hashes every other public file; the manifest omits itself. The deterministic ZIP lives outside this folder.\n\n"
        "Original DVIDIA code and scene: MIT. PiPER simulation assets: MIT. Viewer dependencies retain their notices. "
        "No calibrated shoe physics, policy learning or physical-robot readiness is claimed.\n").encode()
    for name, raw in additions.items():
        public_text_check(name, raw)
        path = destination/name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
    payload = inventory(destination)
    manifest = {"schema": SCHEMA, "files": payload, "excluded_self": "manifest.json",
        "audit": identity(additions["audit.json"]), "protocol": audited["protocol"],
        "counts": audited["counts"], "scope": "Recorded passive rigid physics; no learned skill or hardware qualification"}
    with (destination/"manifest.json").open("xb") as stream:
        stream.write(encoded(manifest))
    archived = deterministic_zip(destination, archive)
    receipt = {"schema": SCHEMA, "public_files": len(payload)+1, "public_bytes": sum(value["bytes"] for value in payload.values())+len(encoded(manifest)),
        "manifest": identity(encoded(manifest)), "archive": {"name": archive.name, **archived},
        "audit": identity(additions["audit.json"]), "counts": audited["counts"]}
    with archive.with_suffix(".receipt.json").open("xb") as stream:
        stream.write(encoded(receipt))
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.study, args.assets, args.output), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

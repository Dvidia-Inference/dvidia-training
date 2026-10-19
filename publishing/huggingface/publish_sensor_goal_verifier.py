"""Publish an append-only v0.4 camera/drive benchmark to the existing Space.

The release is locally re-scored from sealed packets and pixels, bound to its
frozen source commit, and uploaded as regular static-Space files. Credentials
are read only by the existing memory-only Keychain adapter in publish.py.
"""
from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from publish import PublicationError, checked_remote, client, verified_downloads
from publish_prerequisites import (encoded, matches_tree, plain, public_bytes,
                                   read, record, tree)

REPO_ID = "Dvidia/robot-simulation-lab"
NAMESPACE = "image-goals/v04"
ROOT_FILES = {"index.html", "README.md", "provenance.json"}
SHA = re.compile(r"[a-f0-9]{40}\Z")
MAX_FILE = 16 * 1024 * 1024
MAX_TOTAL = 112 * 1024 * 1024
MAX_FILES = 1600
PRIVATE_KEYS = {"authorization", "api_key", "apikey", "credential", "credentials", "token", "headers", "raw_response", "response"}


def _json(raw):
    return json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(PublicationError("Nonfinite JSON.")))


def _no_private(value):
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            if any(str(key).lower() in PRIVATE_KEYS for key in current):
                raise PublicationError("Credential or raw response fields cannot be published.")
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)


def inventory(directory):
    root = plain(directory)
    if not root.is_dir():
        raise PublicationError("Expected a local sensor-goal export directory.")
    payload, total = {}, 0
    for path in sorted(root.rglob("*")):
        plain(path)
        if path.is_dir():
            continue
        name = path.relative_to(root).as_posix()
        raw = read(path, MAX_FILE)
        if name.endswith(".json"):
            _no_private(_json(raw))
        if re.search(rb"hf_[A-Za-z0-9]{30,}", raw):
            raise PublicationError("Possible Hugging Face credential in a public file.")
        payload[name] = raw
        total += len(raw)
        if len(payload) > MAX_FILES or total > MAX_TOTAL:
            raise PublicationError("Sensor-goal export exceeds its public size bounds.")
    return payload


def prepare(run, report, output, publication_source_commit):
    """Assemble the exact static viewer and its complete replayable inputs."""
    run, report, destination = plain(run), plain(report), plain(output)
    if SHA.fullmatch(publication_source_commit) is None:
        raise PublicationError("Export requires an exact publication source commit.")
    if destination.exists() or any(destination == item or item in destination.parents or destination in item.parents for item in (run, report)):
        raise PublicationError("Export destination must be fresh and separate from its inputs.")
    repo = Path(__file__).resolve().parents[2]
    source_root = str(repo / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from dvidia_training.sensor_goal_benchmark import verify_integrity
    from dvidia_training.sensor_goal_benchmark import read_json
    verify_integrity(run)
    protocol = read_json(run / "protocol.json")
    result = read_json(run / "benchmark.json")
    if result.get("status") != "completed" or result.get("provider_calls") != 0 or result.get("source") != protocol.get("source"):
        raise PublicationError("Only a completed local benchmark matching its frozen protocol can publish.")
    destination.mkdir(parents=True, exist_ok=False)
    for path in run.rglob("*"):
        relative = path.relative_to(run)
        out = destination / relative
        if path.is_dir():
            out.mkdir(parents=True, exist_ok=True)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(read(path, MAX_FILE))
    for name in ("index.html", "report.json", "report-integrity.json"):
        raw = read(report / name, MAX_FILE)
        (destination / name).write_bytes(raw)
    (destination / "LICENSE").write_bytes(read(repo / "LICENSE", 1024 * 1024))
    provenance = {
        "schema": "dvidia.sensor-goal-v04-provenance/v1",
        "repo_id": REPO_ID,
        "namespace": NAMESPACE,
        "publication_source_commit": publication_source_commit,
        "benchmark_source_commit": protocol["source"]["commit"],
        "protocol_sha256": record(read(run / "protocol.json", MAX_FILE))["sha256"],
        "benchmark_sha256": record(read(run / "benchmark.json", MAX_FILE))["sha256"],
        "native_integrity_sha256": record(read(run / "integrity.json", MAX_FILE))["sha256"],
        "report_integrity_sha256": record(read(report / "report-integrity.json", MAX_FILE))["sha256"],
        "status": "completed",
        "physical_robot_qualification": False,
        "provider_calls": 0,
    }
    (destination / "provenance.json").write_bytes(encoded(provenance))
    (destination / "README.md").write_text(
        "---\n"
        "title: DVIDIA Sensor Corroboration v0.4\n"
        "sdk: static\n"
        "---\n\n"
        "# Independent camera and actuator corroboration v0.4\n\n"
        "A local synthetic PiPER benchmark tests separately processed RGB and drive checks against primary completion readings. "
        "It prevents 69 of 112 injected false completions; 43 known orientation, identity, elevation, sampling and replay cases remain. "
        "The conservative gate also abstains on 73 of 80 baseline true completions. This is not physical sensor independence, a robot controller or hardware qualification.\n\n"
        "Open the [interactive evidence viewer](index.html), inspect [the paired results](benchmark.json), "
        "[case catalog](cases.json), [frozen protocol](protocol.json), and [integrity seal](integrity.json).\n\n"
        "Reproduce using the pinned source commit in `protocol.json` and the commands in the [repository guide](https://github.com/Dvidia-Inference/dvidia-training/blob/main/docs/sensor-goal-verifier.md).\n",
        encoding="utf-8")
    files = inventory(destination)
    (destination / "publication-integrity.json").write_bytes(encoded({
        "schema": "dvidia.sensor-goal-v04-export-integrity/v1",
        "files": {name: record(raw) for name, raw in sorted(files.items())}}))
    return {"files": len(files) + 1, "bytes": sum(map(len, files.values())), "path": str(destination)}


def check_export(export, publication_source_commit):
    payload = inventory(export)
    required = {"index.html", "report.json", "report-integrity.json", "publication-integrity.json",
        "protocol.json", "cases.json", "generation.json", "integrity.json", "benchmark.json",
        "provenance.json", "README.md", "LICENSE"}
    if not required <= payload.keys():
        raise PublicationError("Sensor-goal release lacks its viewer, protocol or full native evidence.")
    outer = _json(payload["publication-integrity.json"])
    if outer.get("schema") != "dvidia.sensor-goal-v04-export-integrity/v1" or outer.get("files") != {
            name: record(raw) for name, raw in sorted(payload.items()) if name != "publication-integrity.json"}:
        raise PublicationError("Export differs from its outer integrity seal.")
    protocol, result = _json(payload["protocol.json"]), _json(payload["benchmark.json"])
    provenance = _json(payload["provenance.json"])
    source = protocol.get("source", {})
    source_commit = source.get("commit", "")
    if (SHA.fullmatch(source_commit) is None or provenance.get("publication_source_commit") != publication_source_commit
            or provenance.get("benchmark_source_commit") != source_commit or result.get("source") != source
            or result.get("schema") != "dvidia.sensor-goal-results/v1" or result.get("status") != "completed"
            or result.get("provider_calls") != 0 or protocol.get("provider_calls") != 0
            or result.get("protocol_sha256") != record(payload["protocol.json"])["sha256"]
            or result.get("input_integrity_sha256") != record(payload["integrity.json"])["sha256"]):
        raise PublicationError("Result, protocol and publication source pins do not agree.")
    if any(provenance.get(key) != record(payload[name])["sha256"] for key, name in (
            ("protocol_sha256", "protocol.json"), ("benchmark_sha256", "benchmark.json"),
            ("native_integrity_sha256", "integrity.json"), ("report_integrity_sha256", "report-integrity.json"))):
        raise PublicationError("Provenance hashes do not match the saved files.")
    native = _json(payload["integrity.json"])
    if native.get("schema") != "dvidia.sensor-goal-integrity/v1" or any(
            name not in payload or record(payload[name]) != entry for name, entry in native.get("files", {}).items()):
        raise PublicationError("Native evidence differs from its input seal.")
    catalog = _json(payload["cases.json"]).get("cases", [])
    rows = result.get("cases", [])
    if len(catalog) != protocol.get("planned_cases") or len(rows) != len(catalog):
        raise PublicationError("Every scheduled case must have exactly one scored result.")
    ids = [case.get("id") for case in catalog]
    if len(set(ids)) != len(ids) or set(ids) != {row.get("id") for row in rows}:
        raise PublicationError("Duplicate, missing or unplanned case result.")
    repo = Path(__file__).resolve().parents[2]
    for name, digest in source.get("files", {}).items():
        if not isinstance(name, str) or name.startswith("/") or ".." in Path(name).parts or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise PublicationError("Invalid frozen source inventory.")
        committed = subprocess.check_output(["git", "show", f"{source_commit}:{name}"], cwd=repo)
        if sha256(committed).hexdigest() != digest:
            raise PublicationError("Frozen source commit differs from its recorded fingerprint.")
    if (provenance.get("publication_source_commit") != publication_source_commit
            or _json(payload["report-integrity.json"]).get("schema") != "dvidia.sensor-goal-report-integrity/v1"):
        raise PublicationError("Unexpected export provenance or report seal.")
    source_root = str(repo / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from dvidia_training.sensor_goal_benchmark import evaluate_case, score, verify_integrity
    verify_integrity(export)
    for case, row in zip(catalog, rows):
        for key in ("primary_packet_file", "camera_packet_file", "actuator_packet_file", "state_file", "current_image", "goal_image"):
            if case[key] not in native["files"]:
                raise PublicationError("Case references evidence outside the sealed run.")
        timing = row.get("timing_ms")
        if not isinstance(timing, list) or len(timing) != protocol.get("repeats"):
            raise PublicationError("Every case must retain all declared timing repetitions.")
        replay = evaluate_case(export, case, timings=False)
        if any(row.get(key) != replay[key] for key in ("baseline", "camera", "actuator", "fused")):
            raise PublicationError("Saved decisions differ from independent replay of the exact inputs.")
    if result.get("metrics") != score(rows, catalog, repeats=protocol["repeats"]):
        raise PublicationError("Published metrics differ from all saved case results.")
    from dvidia_training.sensor_goal_verifier_report import build_report
    with tempfile.TemporaryDirectory(prefix="dvidia-sensor-goal-export-") as temporary:
        viewer = Path(temporary) / "viewer"
        expected_seal = build_report(export, viewer)
        if _json(payload["report-integrity.json"]) != expected_seal:
            raise PublicationError("Viewer differs from the regenerated report seal.")
        for name, meta in expected_seal["files"].items():
            if name not in payload or payload[name] != (viewer / name).read_bytes() or record(payload[name]) != meta:
                raise PublicationError("Published viewer differs from regenerated evidence.")
    return payload


def snapshot(output, parent_commit):
    destination = plain(output)
    if SHA.fullmatch(parent_commit) is None or destination.exists():
        raise PublicationError("Snapshot requires a fresh destination and exact parent commit.")
    from huggingface_hub import HfApi
    api = HfApi(token=False)
    info = api.repo_info(REPO_ID, repo_type="space")
    checked_remote(info, "space")
    if info.sha != parent_commit:
        raise PublicationError("Space moved before its snapshot.")
    previous = tree(api, parent_commit)
    if any(name == NAMESPACE or name.startswith(NAMESPACE + "/") for name in previous):
        raise PublicationError("The v0.4 namespace already exists.")
    if not ROOT_FILES | {".gitattributes"} <= previous.keys():
        raise PublicationError("Existing Space root is incomplete.")
    destination.mkdir(parents=True)
    for name in ROOT_FILES | {".gitattributes"}:
        (destination / name).write_bytes(public_bytes(name, parent_commit, MAX_FILE))
    (destination / "snapshot.json").write_bytes(encoded({"schema": "dvidia.sensor-goal-v04-parent/v1",
        "repo_id": REPO_ID, "namespace": NAMESPACE, "parent_commit": parent_commit, "tree": previous}))
    return {"parent_commit": parent_commit, "retained_tree_entries": len(previous)}


def updated_roots(parent, export, publication_source_commit):
    count = len(_json(export["cases.json"])["cases"])
    card = ('<section aria-labelledby="sensor-goal-v04-title"><h2 id="sensor-goal-v04-title">Can a second sensor catch a false finish?</h2>'
        f'<p>Inspect {count} synthetic paired cases. Independent RGB and PiPER drive checks prevent 69 of 112 injected false completion readings; 43 remain. '
        'This conservative gate also abstains on 73 of 80 baseline true completions. The full report shows both errors and replay limits.</p>'
        f'<a href="{NAMESPACE}/index.html">Inspect camera and actuator evidence →</a></section>')
    index = parent["index.html"].decode()
    marker = '<section aria-labelledby="local-goal-verifier-title">'
    if index.count(marker) != 1 or 'id="sensor-goal-v04-title"' in index:
        raise PublicationError("Expected the existing v0.3 Space card exactly once.")
    index = index.replace(marker, card + marker)
    readme = parent["README.md"] + ("\n\n## Independent sensor corroboration v0.4\n\n"
        f"[Inspect the RGB and drive-channel benchmark]({NAMESPACE}/index.html). "
        "It prevents 69/112 injected false completions; 43 replay, identity, orientation, support and sampling blind spots remain. "
        "The conservative gate abstains on 73/80 baseline true completions. Synthetic evidence only; not hardware-qualified.\n\n"
        f"[Pinned implementation](https://github.com/Dvidia-Inference/dvidia-training/tree/{publication_source_commit}) · "
        "[Research](https://research.dvidia.org/papers/sensor-corroboration-v04/)\n").encode()
    provenance = _json(parent["provenance.json"])
    if "sensor_goal_v04" in provenance:
        raise PublicationError("v0.4 provenance already exists at the Space root.")
    provenance["sensor_goal_v04"] = {"namespace": NAMESPACE, "publication_source_commit": publication_source_commit,
        "export_integrity": record(export["publication-integrity.json"]), "status": "completed",
        "provider_calls": 0, "physical_robot_qualification": False}
    return {"index.html": index.encode(), "README.md": readme, "provenance.json": encoded(provenance)}


def stage(export_dir, parent_dir, output, publication_source_commit):
    export_dir, parent_dir, destination = plain(export_dir), plain(parent_dir), plain(output)
    if destination.exists() or any(destination == item or item in destination.parents or destination in item.parents for item in (export_dir, parent_dir)):
        raise PublicationError("Stage output must be fresh and separate from its inputs.")
    parent = _json(read(parent_dir / "snapshot.json", 2 * 1024 * 1024))
    if parent.get("repo_id") != REPO_ID or parent.get("namespace") != NAMESPACE or SHA.fullmatch(parent.get("parent_commit", "")) is None:
        raise PublicationError("Unexpected pinned Space snapshot.")
    previous = parent["tree"]
    roots = {name: read(parent_dir / name, MAX_FILE) for name in ROOT_FILES}
    attributes = read(parent_dir / ".gitattributes", 65536)
    if any(not matches_tree(raw, previous[name]) for name, raw in {**roots, ".gitattributes": attributes}.items()):
        raise PublicationError("Pinned Space root bytes changed.")
    export = check_export(export_dir, publication_source_commit)
    payload = {NAMESPACE + "/" + name: raw for name, raw in export.items()}
    payload.update(updated_roots(roots, export, publication_source_commit))
    if set(payload) & set(previous) != ROOT_FILES:
        raise PublicationError("Only the three Space root pointers may replace existing files.")
    destination.mkdir(parents=True)
    for name, raw in payload.items():
        path = destination / "payload" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    (destination / "parent-gitattributes.txt").write_bytes(attributes)
    plan = {"schema": "dvidia.sensor-goal-v04-publication/v1", "repo_id": REPO_ID, "namespace": NAMESPACE,
        "publication_source_commit": publication_source_commit, "parent_commit": parent["parent_commit"],
        "parent_tree": previous, "parent_attributes": record(attributes),
        "files": {name: record(raw) for name, raw in sorted(payload.items())}}
    plan_path = destination / "plan.json"
    plan_path.write_bytes(encoded(plan))
    check(plan_path)
    return {"plan": str(plan_path), "files": len(payload), "bytes": sum(map(len, payload.values()))}


def check(path):
    path = plain(path)
    plan = _json(read(path, 2 * 1024 * 1024))
    if (plan.get("schema") != "dvidia.sensor-goal-v04-publication/v1" or plan.get("repo_id") != REPO_ID
            or plan.get("namespace") != NAMESPACE or SHA.fullmatch(plan.get("publication_source_commit", "")) is None
            or SHA.fullmatch(plan.get("parent_commit", "")) is None):
        raise PublicationError("Unexpected v0.4 Hub plan.")
    payload = inventory(path.parent / "payload")
    if plan.get("files") != {name: record(raw) for name, raw in sorted(payload.items())}:
        raise PublicationError("Payload differs from its reviewed plan.")
    previous = plan["parent_tree"]
    if (set(payload) & set(previous) != ROOT_FILES
            or any(not (name in ROOT_FILES or name.startswith(NAMESPACE + "/")) for name in payload)
            or any(name == NAMESPACE or name.startswith(NAMESPACE + "/") for name in previous)):
        raise PublicationError("The plan may add only the new namespace and update its three root pointers.")
    attributes = read(path.parent / "parent-gitattributes.txt", 65536)
    if record(attributes) != plan["parent_attributes"] or not matches_tree(attributes, previous[".gitattributes"]):
        raise PublicationError("Pinned Hub attributes changed.")
    check_export(path.parent / "payload" / NAMESPACE, plan["publication_source_commit"])
    return plan, payload


def publish(path, gate_path):
    path = plain(path)
    plan, payload = check(path)
    gate = _json(read(gate_path, 1024 * 1024))
    if (gate.get("publication_source_commit") != plan["publication_source_commit"]
            or any(gate.get(key) is not True for key in ("tests_passed", "benchmark_reviewed", "static_qa_passed"))
            or gate.get("export_integrity_sha256") != plan["files"][NAMESPACE + "/publication-integrity.json"]["sha256"]):
        raise PublicationError("Publication requires reviewed tests, metrics and sealed static output.")
    api, token = client(path.parent / "publisher-cache")
    if any(token.encode() in raw for raw in payload.values()):
        raise PublicationError("Credential occurs in an upload byte sequence.")
    if "Dvidia" not in {org.get("name") for org in api.whoami().get("orgs", [])}:
        raise PublicationError("Publisher is not a member of the Dvidia organization.")
    pending_path = plain(path.parent / "pending.json")
    plan_hash = record(read(path, 2 * 1024 * 1024))["sha256"]
    if pending_path.exists():
        pending = _json(read(pending_path, 1024 * 1024))
        if pending.get("plan_sha256") != plan_hash or SHA.fullmatch(pending.get("commit", "")) is None:
            raise PublicationError("Retry receipt belongs to a different plan.")
        revision = pending["commit"]
    else:
        info = api.repo_info(REPO_ID, repo_type="space")
        checked_remote(info, "space")
        if info.sha != plan["parent_commit"] or tree(api, info.sha) != plan["parent_tree"]:
            raise PublicationError("Space changed after its reviewed snapshot.")
        from huggingface_hub import CommitOperationAdd
        operations = []
        for name, raw in payload.items():
            operation = CommitOperationAdd(path_in_repo=name, path_or_fileobj=raw)
            operation._upload_mode = "regular"
            operation._should_ignore = False
            operations.append(operation)
        revision = api.create_commit(REPO_ID, repo_type="space", parent_commit=plan["parent_commit"],
            operations=operations, commit_message="Publish independent RGB and actuator verifier v0.4").oid
        pending_path.write_bytes(encoded({"plan_sha256": plan_hash, "commit": revision}))
    after = tree(api, revision)
    if set(after) != set(plan["parent_tree"]) | set(payload):
        raise PublicationError("Published Space has unexpected added or removed files.")
    unchanged = set(plan["parent_tree"]) - set(payload)
    if any(after[name] != plan["parent_tree"][name] for name in unchanged):
        raise PublicationError("An earlier Space file changed during publication.")
    if any(after[name].get("lfs_sha256") is not None or not matches_tree(raw, after[name]) for name, raw in payload.items()):
        raise PublicationError("Published files differ from the sealed regular-Git payload.")
    verified_downloads({"repo_id": REPO_ID, "repo_type": "space", "files": [
        {"path": name, **expected} for name, expected in plan["files"].items()]}, revision, False)
    receipt = {"schema": "dvidia.sensor-goal-v04-hf-publication/v1", "repo_id": REPO_ID,
        "namespace": NAMESPACE, "publication_source_commit": plan["publication_source_commit"],
        "parent_commit": plan["parent_commit"], "commit": revision, "plan_sha256": plan_hash,
        "verified_public_files": len(payload), "unchanged_historical_tree_entries": len(unchanged),
        "all_new_files_regular_git": True, "viewer": f"https://dvidia-robot-simulation-lab.static.hf.space/{NAMESPACE}/index.html",
        "files": plan["files"]}
    (path.parent / "after-tree.json").write_bytes(encoded(after))
    (path.parent / "publication-receipt.json").write_bytes(encoded(receipt))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("run"); prepare_parser.add_argument("report"); prepare_parser.add_argument("output")
    prepare_parser.add_argument("--publication-source-commit", required=True)
    snap = sub.add_parser("snapshot"); snap.add_argument("output"); snap.add_argument("--parent-commit", required=True)
    stage_parser = sub.add_parser("stage")
    for name in ("export", "parent", "output"): stage_parser.add_argument(name)
    stage_parser.add_argument("--publication-source-commit", required=True)
    check_parser = sub.add_parser("check"); check_parser.add_argument("plan")
    pub = sub.add_parser("publish"); pub.add_argument("plan"); pub.add_argument("gate")
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            result = prepare(args.run, args.report, args.output, args.publication_source_commit)
        elif args.action == "snapshot":
            result = snapshot(args.output, args.parent_commit)
        elif args.action == "stage":
            result = stage(args.export, args.parent, args.output, args.publication_source_commit)
        elif args.action == "check":
            plan, payload = check(args.plan); result = {"checked_files": len(payload), "publication_source_commit": plan["publication_source_commit"]}
        else:
            result = publish(args.plan, args.gate)
        print(json.dumps(result, indent=2))
    except Exception as error:
        print(json.dumps({"status": "stopped", "error_type": type(error).__name__,
            "http_status": getattr(getattr(error, "response", None), "status_code", None)}))
        raise SystemExit(1)


if __name__ == "__main__": main()

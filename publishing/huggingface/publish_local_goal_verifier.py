"""Append a reviewed local-verifier benchmark to the public simulation Space.

Fixed namespace image-goals/v03. Prior releases and Hub attributes stay exact;
only three existing root pointers can change. Credentials reuse the memory-only
Keychain adapter. No API key is accepted as a command-line argument.
"""
from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
import re
import sys
import tempfile

from publish import PublicationError, checked_remote, client, verified_downloads
from publish_prerequisites import (encoded, inspect_bytes, matches_tree, plain,
                                  public_bytes, read, record, tree)

REPO_ID = "Dvidia/robot-simulation-lab"
NAMESPACE = "image-goals/v03"
ROOT_FILES = {"index.html", "README.md", "provenance.json"}
SHA = re.compile(r"[a-f0-9]{40}\Z")
MAX_FILE = 8 * 1024 * 1024
MAX_TOTAL = 96 * 1024 * 1024
MAX_FILES = 640
PRIVATE_KEYS = {"authorization", "api_key", "apikey", "credential", "credentials", "token", "headers", "raw_response", "response"}


def inventory(directory):
    directory = plain(directory)
    if not directory.is_dir():
        raise PublicationError("Expected a plain local-verifier export directory.")
    payload, size = {}, 0
    for path in sorted(directory.rglob("*")):
        plain(path)
        if path.is_dir():
            continue
        name = path.relative_to(directory).as_posix()
        raw = read(path, MAX_FILE)
        if name in {"protocol.sha256", NAMESPACE + "/protocol.sha256"}:
            if re.fullmatch(rb"[a-f0-9]{64}\n", raw) is None:
                raise PublicationError("Expected one newline-terminated SHA-256 digest.")
            inspect_bytes(name + ".txt", raw)
        else:
            inspect_bytes(name, raw)
        if name.endswith(".json"):
            stack = [json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(PublicationError("Nonfinite JSON.")))]
            while stack:
                value = stack.pop()
                if isinstance(value, dict):
                    if any(key.lower() in PRIVATE_KEYS for key in value):
                        raise PublicationError("Private credential or raw provider fields cannot be published.")
                    stack.extend(value.values())
                elif isinstance(value, list):
                    stack.extend(value)
        payload[name] = raw
        size += len(raw)
        if len(payload) > MAX_FILES or size > MAX_TOTAL:
            raise PublicationError("Local-verifier export exceeds its publication bound.")
    return payload


def check_export(export, source_commit):
    """Bind the viewer to the saved packets, decisions, scores and native files."""
    payload = inventory(export)
    required = {"index.html", "report.json", "report-integrity.json", "integrity.json", "publication-integrity.json",
                "cases.json", "protocol.json", "results.json", "provenance.json", "LICENSE"}
    if not required <= payload.keys():
        raise PublicationError("Missing local-verifier protocol, native evidence, results or license.")
    seal = json.loads(payload["publication-integrity.json"])
    if seal.get("files") != {name: record(raw) for name, raw in payload.items() if name != "publication-integrity.json"}:
        raise PublicationError("Export differs from its complete publication seal.")
    provenance = json.loads(payload["provenance.json"])
    result = json.loads(payload["results.json"])
    protocol = json.loads(payload["protocol.json"])
    if (provenance.get("publication_source_commit") != source_commit
            or SHA.fullmatch(result.get("source_commit", "")) is None
            or provenance.get("benchmark_source_commit") != result["source_commit"]
            or protocol.get("source", {}).get("commit") != result["source_commit"]):
        raise PublicationError("Benchmark and publication source pins must agree with recorded provenance.")
    if (result.get("status") != "completed" or result.get("active_trial") is not None
            or result.get("protocol_sha256") != record(payload["protocol.json"])["sha256"]
            or result.get("input_integrity_sha256") != record(payload["integrity.json"])["sha256"]
            or result.get("provider_calls") != 0 or protocol.get("provider_calls") != 0
            or protocol.get("schema") != "dvidia.local-goal-protocol/v1"
            or result.get("schema") != "dvidia.local-goal-results/v1"):
        raise PublicationError("Only a completed, inactive run of the exact frozen protocol can publish.")
    native_seal = json.loads(payload["integrity.json"])
    native_files = native_seal.get("files")
    if (native_seal.get("schema") != "dvidia.local-goal-integrity/v1" or not isinstance(native_files, dict)
            or not {"cases.json", "protocol.json"} <= native_files.keys()):
        raise PublicationError("Native integrity must include the case catalog and protocol.")
    if any(name not in payload or record(payload[name]) != expected for name, expected in native_files.items()):
        raise PublicationError("Frozen native evidence changed or is missing.")
    cases = json.loads(payload["cases.json"])["cases"]
    rows = result.get("results")
    if not isinstance(rows, list) or len(rows) != len(cases) or len(cases) != protocol.get("planned_cases"):
        raise PublicationError("Every planned case needs exactly one result.")
    by_id = {row.get("case_id"): row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != {case["id"] for case in cases}:
        raise PublicationError("Duplicate, missing or unplanned result row.")
    repo = Path(__file__).resolve().parents[2]
    expected_sources = {"src/dvidia_training/" + name + ".py" for name in ("local_goal_verifier", "local_goal_benchmark", "local_goal_scenes", "image_goal_scenes", "image_goal_benchmark", "jev_client", "workstation", "robot_profiles", "reference_sim", "asset_library")}
    expected_sources |= {"src/dvidia_training/robot_profiles.lock.json", "src/dvidia_training/asset_catalog.json"}
    sources = protocol.get("source", {}).get("files", {})
    if set(sources) != expected_sources or any(sha256(read(repo / name, MAX_FILE)).hexdigest() != digest for name, digest in sources.items()):
        raise PublicationError("Running benchmark implementation differs from the frozen source hashes.")
    source_root = str(repo / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from dvidia_training.local_goal_verifier import GoalContract, verify_goal
    for case in cases:
        if any(case.get(key) not in native_files for key in ("packet_file", "state_file", "current_image", "goal_image")):
            raise PublicationError("Case refers to evidence outside the frozen inventory.")
        timing = by_id[case["id"]].get("timing_ns")
        if not isinstance(timing, list) or len(timing) != protocol.get("repeats"):
            raise PublicationError("Every case needs its complete prespecified timing repetitions.")
        packet_path = case["packet_file"]
        if packet_path not in native_files or packet_path not in payload:
            raise PublicationError("Observation packet is outside frozen native evidence.")
        contract = GoalContract(**case["contract"])
        expected = verify_goal(json.loads(payload[packet_path]), contract=contract, now_s=case["now_s"])
        if by_id[case["id"]].get("decision") != expected:
            raise PublicationError("Saved verdict differs from independent evaluation of the exact packet.")
    from dvidia_training.local_goal_benchmark import score
    if result.get("metrics") != score(rows, cases):
        raise PublicationError("Metrics differ from independently recomputed cohort scores.")
    from dvidia_training.local_goal_verifier_report import build_report
    with tempfile.TemporaryDirectory(prefix="dvidia-local-verifier-publication-") as temporary:
        expected_viewer = Path(temporary).resolve() / "viewer"
        expected_seal = build_report(export, expected_viewer)
        if json.loads(payload["report-integrity.json"]) != expected_seal:
            raise PublicationError("Viewer inventory differs from independent regeneration.")
        for name in (*expected_seal["files"], "report-integrity.json"):
            if payload.get(name) != (expected_viewer / name).read_bytes():
                raise PublicationError("Viewer differs from independently regenerated evidence and presentation.")
    return payload


def snapshot(output, parent_commit):
    if SHA.fullmatch(parent_commit) is None:
        raise PublicationError("Snapshot requires an exact parent commit.")
    destination = plain(output)
    if destination.exists():
        raise PublicationError("Snapshot output must be fresh.")
    from huggingface_hub import HfApi
    api = HfApi(token=False)
    info = api.repo_info(REPO_ID, repo_type="space")
    checked_remote(info, "space")
    if info.sha != parent_commit:
        raise PublicationError("Space moved before its snapshot.")
    previous = tree(api, parent_commit)
    if any(name == NAMESPACE or name.startswith(NAMESPACE + "/") for name in previous):
        raise PublicationError("Image-goal namespace already exists.")
    if not ROOT_FILES | {".gitattributes"} <= previous.keys():
        raise PublicationError("Existing Space root is incomplete.")
    roots = {name: public_bytes(name, parent_commit, MAX_FILE) for name in ROOT_FILES | {".gitattributes"}}
    if any(not matches_tree(raw, previous[name]) for name, raw in roots.items()):
        raise PublicationError("Root bytes differ from the pinned tree.")
    destination.mkdir(parents=True)
    for name, raw in roots.items():
        (destination / name).write_bytes(raw)
    (destination / "snapshot.json").write_bytes(encoded({"schema": "dvidia.local-goal-parent/v1",
        "repo_id": REPO_ID, "namespace": NAMESPACE, "parent_commit": parent_commit, "tree": previous}))
    return {"parent_commit": parent_commit, "retained_tree_entries": len(previous)}


def updated_roots(parent, source_commit, export):
    count = len(json.loads(export["cases.json"])["cases"])
    description = (f"Inspect {count} local CPU-verifier cases: fresh native scenes, invalid evidence, and internally consistent fabricated readings. "
        "See positive recognition, false completions, abstentions and predicate reasons. "
        "Privileged simulation observations; zero new Jev calls; no robot controller or hardware qualification.")
    index = parent["index.html"].decode()
    marker = '<section aria-labelledby="image-goals-results-title">'
    if index.count(marker) != 1 or 'id="local-goal-verifier-title"' in index:
        raise PublicationError("Expected one measured Jev section and no existing local-verifier section.")
    card = ('<section aria-labelledby="local-goal-verifier-title"><h2 id="local-goal-verifier-title">What does finished require?</h2>'
        f'<p>{description}</p><a href="{NAMESPACE}/index.html">Inspect local goal decisions →</a></section>')
    index = index.replace(marker, card + marker)
    readme = parent["README.md"] + (f'\n\n## Local goal verifier v0.3\n\n'
        f"[Inspect the local verifier]({NAMESPACE}/index.html). {description}\n\n"
        f"[Pinned source](https://github.com/Dvidia-Inference/dvidia-training/tree/{source_commit}) · "
        '[Research](https://research.dvidia.org/papers/local-goal-verifier-v03/)\n').encode()
    provenance = json.loads(parent["provenance.json"])
    if "local_goal_verifier_v03" in provenance:
        raise PublicationError("Local-verifier root provenance already exists.")
    provenance["local_goal_verifier_v03"] = {"path": NAMESPACE, "source_commit": source_commit,
        "integrity": record(export["publication-integrity.json"]), "status": "completed",
        "mode": "local-cpu-verifier", "new_jev_calls": 0, "learned_policy": False, "physical_robot_ready": False}
    return {"index.html": index.encode(), "README.md": readme, "provenance.json": encoded(provenance)}


def stage(export, parent, output, source_commit):
    if SHA.fullmatch(source_commit) is None:
        raise PublicationError("An exact source commit is required.")
    parent, destination, export = plain(parent), plain(output), plain(export)
    if any(destination == source or source in destination.parents or destination in source.parents for source in (parent, export)):
        raise PublicationError("Stage output must be separate from its inputs.")
    baseline = json.loads(read(parent / "snapshot.json", 2 * 1024 * 1024))
    if (baseline.get("repo_id") != REPO_ID or baseline.get("namespace") != NAMESPACE
            or SHA.fullmatch(baseline.get("parent_commit", "")) is None):
        raise PublicationError("Unexpected parent snapshot.")
    previous = baseline["tree"]
    parent_files = {name: read(parent / name, MAX_FILE) for name in ROOT_FILES}
    attributes = read(parent / ".gitattributes", 65536)
    if any(not matches_tree(raw, previous[name]) for name, raw in {**parent_files, ".gitattributes": attributes}.items()):
        raise PublicationError("Pinned parent root bytes changed.")
    release = check_export(export, source_commit)
    payload = {NAMESPACE + "/" + name: raw for name, raw in release.items()}
    payload.update(updated_roots(parent_files, source_commit, release))
    if set(payload) & set(previous) != ROOT_FILES:
        raise PublicationError("Only root pointers may replace existing files.")
    destination.mkdir(parents=True, exist_ok=False)
    for name, raw in payload.items():
        path = destination / "payload" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    (destination / "parent-gitattributes.txt").write_bytes(attributes)
    plan = {"schema": "dvidia.local-goal-publication/v1", "repo_id": REPO_ID, "namespace": NAMESPACE,
        "source_commit": source_commit, "parent_commit": baseline["parent_commit"], "parent_tree": previous,
        "parent_attributes": record(attributes), "files": {name: record(raw) for name, raw in sorted(payload.items())}}
    (destination / "plan.json").write_bytes(encoded(plan))
    check(destination / "plan.json")
    return {"plan": str(destination / "plan.json"), "files": len(payload), "bytes": sum(map(len, payload.values()))}


def check(path):
    path = plain(path)
    plan = json.loads(read(path, 2 * 1024 * 1024))
    if (plan.get("schema") != "dvidia.local-goal-publication/v1" or plan.get("repo_id") != REPO_ID
            or plan.get("namespace") != NAMESPACE
            or any(SHA.fullmatch(plan.get(key, "")) is None for key in ("source_commit", "parent_commit"))):
        raise PublicationError("Unexpected image-goal publication plan.")
    payload = inventory(path.parent / "payload")
    if plan.get("files") != {name: record(raw) for name, raw in payload.items()}:
        raise PublicationError("Payload changed after staging.")
    if (set(payload) & set(plan["parent_tree"]) != ROOT_FILES
            or any(name not in ROOT_FILES and not name.startswith(NAMESPACE + "/") for name in payload)
            or any(name == NAMESPACE or name.startswith(NAMESPACE + "/") for name in plan["parent_tree"])):
        raise PublicationError("Only the fresh namespace and three root pointers may change.")
    attributes = read(path.parent / "parent-gitattributes.txt", 65536)
    if record(attributes) != plan["parent_attributes"] or not matches_tree(attributes, plan["parent_tree"][".gitattributes"]):
        raise PublicationError("Pinned Hub metadata changed.")
    check_export(path.parent / "payload" / NAMESPACE, plan["source_commit"])
    return plan, payload


def checked_qa(gate, payload):
    """Retain an explicitly authorized blocked-browser release as incomplete QA."""
    if gate.get("browser_qa_passed") is True:
        return {"browser_qa_passed": True, "status": "passed"}
    pending = (gate.get("browser_qa_passed") is False
        and gate.get("browser_qa_status") == "blocked-macos-locked"
        and gate.get("static_qa_passed") is True
        and gate.get("publication_authorized_with_browser_qa_pending") is True)
    qa_path = NAMESPACE + "/ui-qa.json"
    if not pending or qa_path not in payload:
        raise PublicationError("Browser QA must pass or remain explicitly authorized and documented as pending.")
    receipt = json.loads(payload[qa_path])
    if (receipt.get("status") != "partial-browser-blocked"
            or receipt.get("viewer_sha256") != record(payload[NAMESPACE + "/index.html"])["sha256"]
            or receipt.get("static_checks", {}).get("unique_element_ids") is not True
            or receipt.get("browser", {}).get("screenshots") != []):
        raise PublicationError("Pending visual QA requires the exact viewer's honest static-check receipt.")
    return {"browser_qa_passed": False, "status": "pending-macos-locked", "static_qa_passed": True,
            "publication_authorized_with_browser_qa_pending": True, "public_receipt": qa_path}


def publish(path, gate_path):
    path = plain(path)
    plan, payload = check(path)
    gate = json.loads(read(gate_path, 1024 * 1024))
    if (gate.get("source_commit") != plan["source_commit"]
            or any(gate.get(key) is not True for key in ("tests_passed", "benchmark_reviewed"))
            or gate.get("integrity_sha256") != plan["files"][NAMESPACE + "/publication-integrity.json"]["sha256"]):
        raise PublicationError("Publication requires source-bound tests and evidence review.")
    qa = checked_qa(gate, payload)
    api, token = client(path.parent / "publisher-cache")
    if any(token.encode() in raw for raw in payload.values()):
        raise PublicationError("Credential occurs in upload bytes.")
    from huggingface_hub.utils import get_session
    session = get_session()
    session.timeout = type(session.timeout)(60., connect=20., read=120., write=60., pool=20.)
    if "Dvidia" not in {org.get("name") for org in api.whoami().get("orgs", [])}:
        raise PublicationError("Publisher is not a member of the intended organization.")
    pending_path = plain(path.parent / "pending.json")
    plan_hash = record(read(path, 2 * 1024 * 1024))["sha256"]
    if pending_path.exists():
        pending = json.loads(read(pending_path))
        if pending.get("plan_sha256") != plan_hash or SHA.fullmatch(pending.get("commit", "")) is None:
            raise PublicationError("Retry receipt belongs to another plan.")
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
            # The pinned SDK respects this classification. Regular files stay
            # on the static origin; never move active entrypoints to LFS/CDN.
            operation._upload_mode = "regular"
            operation._should_ignore = False
            operations.append(operation)
        result = api.create_commit(REPO_ID, repo_type="space", parent_commit=plan["parent_commit"],
            operations=operations, commit_message="Publish local goal verifier diagnostics with separate native, fault and blind-spot cohorts")
        revision = result.oid
        pending_path.write_bytes(encoded({"commit": revision, "plan_sha256": plan_hash}))
    after = tree(api, revision)
    if set(after) != set(plan["parent_tree"]) | set(payload):
        raise PublicationError("Unexpected paths added or removed.")
    unchanged = set(plan["parent_tree"]) - set(payload)
    if any(after[name] != plan["parent_tree"][name] for name in unchanged):
        raise PublicationError("Historical file or Hub metadata changed.")
    if any(after[name].get("lfs_sha256") is not None or not matches_tree(raw, after[name]) for name, raw in payload.items()):
        raise PublicationError("New files must exactly match regular Git bytes.")
    verified_downloads({"repo_id": REPO_ID, "repo_type": "space", "files": [
        {"path": name, **expected} for name, expected in plan["files"].items()]}, revision, False)
    receipt = {"schema": "dvidia.local-goal-hf-publication/v1", "repo_id": REPO_ID, "namespace": NAMESPACE,
        "source_commit": plan["source_commit"], "parent_commit": plan["parent_commit"], "commit": revision,
        "plan_sha256": plan_hash, "verified_public_files": len(payload), "unchanged_historical_tree_entries": len(unchanged),
        "all_new_files_regular_git": True, "qa": qa, "files": plan["files"]}
    (path.parent / "after-tree.json").write_bytes(encoded(after))
    (path.parent / "publication-receipt.json").write_bytes(encoded(receipt))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    snap = sub.add_parser("snapshot"); snap.add_argument("output"); snap.add_argument("--parent-commit", required=True)
    stage_parser = sub.add_parser("stage")
    for name in ("export", "parent", "output"):
        stage_parser.add_argument(name)
    stage_parser.add_argument("--source-commit", required=True)
    check_parser = sub.add_parser("check"); check_parser.add_argument("plan")
    publish_parser = sub.add_parser("publish"); publish_parser.add_argument("plan"); publish_parser.add_argument("gate")
    args = parser.parse_args()
    try:
        if args.action == "snapshot":
            result = snapshot(args.output, args.parent_commit)
        elif args.action == "stage":
            result = stage(args.export, args.parent, args.output, args.source_commit)
        elif args.action == "check":
            plan, payload = check(args.plan)
            result = {"checked_files": len(payload), "source_commit": plan["source_commit"]}
        else:
            result = publish(args.plan, args.gate)
        print(json.dumps(result, indent=2))
    except Exception as error:
        # Exception messages from SDK transports can contain request context.
        print(json.dumps({"status": "stopped", "error_type": type(error).__name__,
                          "http_status": getattr(getattr(error, "response", None), "status_code", None)}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()

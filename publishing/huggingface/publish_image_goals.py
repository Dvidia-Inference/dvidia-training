"""Append a reviewed image-goal report to the existing public simulation Space.

The only mutable existing files are the three root pointers. Historical Git/LFS
entries and attributes must remain identical. Every new file stays in regular
Git so HTML and JavaScript remain on the static Space origin. Credentials use
the existing memory-only macOS Keychain adapter, never a command-line token.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import tempfile

from publish import PublicationError, checked_remote, client, verified_downloads
from publish_prerequisites import (encoded, git_blob, inspect_bytes, matches_tree,
                                  plain, public_bytes, read, record, tree)

REPO_ID = "Dvidia/robot-simulation-lab"
NAMESPACE = "image-goals/v01"
ROOT_FILES = {"index.html", "README.md", "provenance.json"}
SHA = re.compile(r"[a-f0-9]{40}\Z")
MAX_FILE = 4 * 1024 * 1024
MAX_TOTAL = 64 * 1024 * 1024
MAX_FILES = 256
CONDITIONS = {"rgb", "rgb_telemetry"}
PRIVATE_KEYS = {"authorization", "api_key", "apikey", "credential", "credentials", "token", "headers", "raw_response", "response"}


def inventory(directory):
    directory = plain(directory)
    if not directory.is_dir():
        raise PublicationError("Expected a plain report export directory.")
    payload, size = {}, 0
    for path in sorted(directory.rglob("*")):
        plain(path)
        if path.is_dir():
            continue
        name = path.relative_to(directory).as_posix()
        raw = read(path, MAX_FILE)
        if name in {"protocol.sha256", NAMESPACE + "/protocol.sha256"}:
            if re.fullmatch(rb"[a-f0-9]{64}\n", raw) is None:
                raise PublicationError("Expected exactly one newline-terminated SHA-256 digest.")
            inspect_bytes(name + ".txt", raw)
        elif name in {"verify-image-goal-fixtures.py", NAMESPACE + "/verify-image-goal-fixtures.py"}:
            inspect_bytes(name + ".txt", raw)
        else:
            inspect_bytes(name, raw)
        if name.endswith(".json"):
            stack = [json.loads(raw)]
            while stack:
                value = stack.pop()
                if isinstance(value, dict):
                    if any(key.lower() in PRIVATE_KEYS for key in value):
                        raise PublicationError("Remove raw provider responses, headers and credential fields from the public export.")
                    stack.extend(value.values())
                elif isinstance(value, list):
                    stack.extend(value)
        payload[name] = raw
        size += len(raw)
        if len(payload) > MAX_FILES or size > MAX_TOTAL:
            raise PublicationError("Image-goal report exceeds its publication bound.")
    return payload


def check_export(export, source_commit):
    payload = inventory(export)
    required = {"index.html", "report.json", "report-integrity.json", "integrity.json", "publication-integrity.json", "cases.json",
                "protocol.json", "evaluation-plan.json", "results.json", "provenance.json", "LICENSE"}
    if not required <= payload.keys():
        raise PublicationError("Missing image-goal report, protocol, results, source provenance or license.")
    seal = json.loads(payload["publication-integrity.json"])
    if seal.get("files") != {name: record(raw) for name, raw in payload.items() if name != "publication-integrity.json"}:
        raise PublicationError("Export does not match its complete integrity seal.")
    provenance = json.loads(payload["provenance.json"])
    plan = json.loads(payload["evaluation-plan.json"])
    result = json.loads(payload["results.json"])
    if result.get("status") == "running" or result.get("active_trial") is not None:
        raise PublicationError("Resolve the active or interrupted request before publication.")
    public = json.loads(payload["report.json"])
    cases = json.loads(payload["cases.json"]).get("cases", [])
    if (provenance.get("source_commit") != source_commit or plan.get("source_commit") != source_commit
            or plan.get("model") != "jev-1.14.0" or result.get("model") != plan["model"]
            or public.get("model") != plan["model"]
            or result.get("evaluation_plan_sha256") != record(payload["evaluation-plan.json"])["sha256"]):
        raise PublicationError("Pinned source, model and evaluated plan must agree.")
    if not {"cases.json", "protocol.json"} <= plan.get("files", {}).keys():
        raise PublicationError("Frozen fixture inventory is incomplete.")
    for name, expected_file in plan["files"].items():
        if name not in payload or record(payload[name]) != expected_file:
            raise PublicationError("Frozen fixture bytes changed or were omitted.")
    if (not 1 <= len(cases) <= 32 or len({case["id"] for case in cases}) != len(cases)
            or set(plan.get("conditions", [])) != CONDITIONS):
        raise PublicationError("Expected the frozen paired image-goal cohort.")
    expected = {(case["id"], condition) for case in cases for condition in CONDITIONS}
    for rows in (plan.get("trials", []), result.get("results", []), public.get("results", [])):
        identities = [(row["case_id"], row["condition"]) for row in rows]
        if len(identities) != len(set(identities)) or set(identities) != expected:
            raise PublicationError("Every planned case and condition must remain in the report, including unrun trials.")
    public_cases = {case["id"]: case for case in public.get("cases", [])}
    if len(public_cases) != len(cases):
        raise PublicationError("Viewer case inventory differs from the frozen cohort.")
    for case in cases:
        shown = public_cases.get(case["id"], {})
        if (shown.get("ground_truth") != {key: case["physical_truth"][key] for key in ("present", "complete")}
                or shown.get("expected_verdict") != case.get("expected_verdict")
                or shown.get("expected_presence") != case.get("expected_presence")):
            raise PublicationError("Viewer labels differ from frozen physical and observability references.")
        for key in ("current_image", "goal_image"):
            if shown.get(key) not in payload or case.get(key) not in payload or payload[shown[key]] != payload[case[key]]:
                raise PublicationError("Viewer images differ from the evaluated fixture images.")
    if public.get("status") != result.get("status") or provenance.get("provider_status") != result.get("status"):
        raise PublicationError("Provider status must agree across publication artifacts.")
    pending = all(row.get("status") == "not_run" for row in result["results"])
    if result.get("status") == "not_run":
        if not pending or result.get("metrics") is not None:
            raise PublicationError("Unrun Jev evidence must have no measured provider metrics.")
        for row in result["results"]:
            if (row.get("presence") not in (None, "unknown") or row.get("verdict") not in (None, "unknown")
                    or any(row.get(key) is not None for key in ("p_present", "p_complete", "online_total_ms", "estimated_input_cost_usd"))
                    or row.get("attempts", 0) != 0):
                raise PublicationError("Unrun provider trials cannot carry predictions or service measurements.")
        if b"Jev has not been run" not in payload["index.html"]:
            raise PublicationError("Unrun provider status must be visible in the report.")
    elif result.get("status") != "completed" or pending:
        raise PublicationError("Unexpected provider run status.")
    viewer_seal = json.loads(payload["report-integrity.json"])
    for name, expected_file in viewer_seal.get("files", {}).items():
        if name not in payload or record(payload[name]) != expected_file:
            raise PublicationError("Viewer export differs from its original file inventory.")
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from dvidia_training.image_goal_report import build_report
    with tempfile.TemporaryDirectory(prefix="dvidia-image-goal-publication-") as temporary:
        expected_viewer = Path(temporary).resolve() / "viewer"
        expected_inventory = build_report(export, expected_viewer)
        if viewer_seal != expected_inventory:
            raise PublicationError("Viewer inventory is not the independently regenerated report.")
        for name in (*expected_inventory["files"], "report-integrity.json"):
            if payload.get(name) != (expected_viewer / name).read_bytes():
                raise PublicationError("Viewer differs from independently regenerated results and presentation.")
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
    (destination / "snapshot.json").write_bytes(encoded({"schema": "dvidia.image-goal-parent/v1",
        "repo_id": REPO_ID, "namespace": NAMESPACE, "parent_commit": parent_commit, "tree": previous}))
    return {"parent_commit": parent_commit, "retained_tree_entries": len(previous)}


def updated_roots(parent, source_commit, export):
    status = json.loads(export["results.json"])["status"]
    pending = status == "not_run"
    count = len(json.loads(export["cases.json"])["cases"])
    description = (f"Explore {count} frozen PiPER image-goal cases, including absent targets, wrong arrangements and visually identical moving scenes. "
        + ("Native fixture validation is published. Jev accuracy and service latency are not yet measured."
           if pending else "Inspect all paired Jev judgments, failures, abstentions and measured service latency.")
        + " Synthetic diagnostic fixtures; passive evaluation; no robot controller or hardware qualification.")
    index = parent["index.html"].decode()
    marker = '<section aria-labelledby="graspability-title">'
    if index.count(marker) != 1 or 'id="image-goals-title"' in index:
        raise PublicationError("Expected one existing graspability section and no image-goal section.")
    card = ('<section aria-labelledby="image-goals-title"><h2 id="image-goals-title">Does it only look finished?</h2>'
            f'<p>{description}</p><a href="{NAMESPACE}/index.html">Inspect image goals and physical references →</a></section>')
    index = index.replace(marker, card + marker)
    readme = parent["README.md"] + ("\n\n## Image-goal benchmark v0.1\n\n"
        f"[Inspect the image-goal benchmark]({NAMESPACE}/index.html). {description}\n\n"
        f"[Pinned source](https://github.com/Dvidia-Inference/dvidia-training/tree/{source_commit}) · "
        "[Research](https://research.dvidia.org/papers/image-goal-jev-v01/)\n").encode()
    provenance = json.loads(parent["provenance.json"])
    if "image_goals_v01" in provenance:
        raise PublicationError("Image-goal root provenance already exists.")
    provenance["image_goals_v01"] = {"path": NAMESPACE, "source_commit": source_commit,
        "integrity": record(export["publication-integrity.json"]), "provider_status": status,
        "mode": "recorded-passive-evaluation", "learned_policy": False, "physical_robot_ready": False}
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
    plan = {"schema": "dvidia.image-goal-publication/v1", "repo_id": REPO_ID, "namespace": NAMESPACE,
        "source_commit": source_commit, "parent_commit": baseline["parent_commit"], "parent_tree": previous,
        "parent_attributes": record(attributes), "files": {name: record(raw) for name, raw in sorted(payload.items())}}
    (destination / "plan.json").write_bytes(encoded(plan))
    check(destination / "plan.json")
    return {"plan": str(destination / "plan.json"), "files": len(payload), "bytes": sum(map(len, payload.values()))}


def check(path):
    path = plain(path)
    plan = json.loads(read(path, 2 * 1024 * 1024))
    if (plan.get("schema") != "dvidia.image-goal-publication/v1" or plan.get("repo_id") != REPO_ID
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


def publish(path, gate_path):
    path = plain(path)
    plan, payload = check(path)
    gate = json.loads(read(gate_path, 1024 * 1024))
    if (gate.get("source_commit") != plan["source_commit"]
            or any(gate.get(key) is not True for key in ("tests_passed", "browser_qa_passed", "benchmark_reviewed"))
            or gate.get("integrity_sha256") != plan["files"][NAMESPACE + "/publication-integrity.json"]["sha256"]):
        raise PublicationError("Publication requires source-bound test, browser and evidence review.")
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
            operations=operations, commit_message="Publish image-goal diagnostics with explicit provider status and complete evidence")
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
    receipt = {"schema": "dvidia.image-goal-hf-publication/v1", "repo_id": REPO_ID, "namespace": NAMESPACE,
        "source_commit": plan["source_commit"], "parent_commit": plan["parent_commit"], "commit": revision,
        "plan_sha256": plan_hash, "verified_public_files": len(payload), "unchanged_historical_tree_entries": len(unchanged),
        "all_new_files_regular_git": True, "files": plan["files"]}
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

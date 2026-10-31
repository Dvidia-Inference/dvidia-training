"""Guarded publication of one sealed, recorded task-environment kit.

Only publish reads the named macOS Keychain credential. Stage/check/verify use
anonymous reads. The shared Space receives additions under one new namespace;
all existing Git/LFS objects are preserved through exact parent-CAS commits.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1, sha256
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import zipfile

from publish import PublicationError, client


SCHEMA = "dvidia.task-environments-hub-publication/v2"
REPOS = {"dataset": "Dvidia/task-environments-v02", "space": "Dvidia/robot-simulation-lab"}
NAMESPACE = "task-environments/v02/"
OID = re.compile(r"[a-f0-9]{40}\Z")
MAX_FILE = 16*1024*1024
MAX_PLAN = 4*1024*1024
IMPLEMENTATION = (Path(__file__).name, "publish.py", "prepare.py")


def digest(raw):
    return {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}


def plain(path):
    path = Path(path).absolute()
    if any(p.is_symlink() and str(p) not in ("/var", "/tmp") for p in (path, *path.parents)):
        raise PublicationError("Publication paths cannot traverse artifact symlinks.")
    return path.resolve()


def relative(name):
    if not isinstance(name, str):
        raise PublicationError("Expected an artifact-relative path.")
    path = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or path.is_absolute() or str(path) != name
            or any(part in (".", "..", "") or part.startswith(".") and name != ".gitattributes" for part in path.parts)
            or any(ord(char) < 32 for char in name)):
        raise PublicationError("Expected a plain artifact-relative path.")
    return path


def read(path, limit=MAX_FILE):
    path = plain(path)
    if not path.is_file():
        raise PublicationError("Missing publication file.")
    with path.open("rb") as stream:
        raw = stream.read(limit+1)
    if len(raw) > limit:
        raise PublicationError("Publication file exceeds its byte limit.")
    return raw


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False)+"\n").encode()


def load(path):
    return json.loads(read(path, MAX_PLAN))


def atomic(path, value, exclusive=False):
    path = plain(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".writing")
    if exclusive and path.exists():
        raise PublicationError("Publication evidence already exists.")
    with temporary.open("xb") as stream:
        stream.write(encoded(value)); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def implementation():
    return {name: digest(read(Path(__file__).with_name(name))) for name in IMPLEMENTATION}


def scan(raw):
    if re.search(rb"(?:hf_|gh[pousr]_|apikey_)[A-Za-z0-9_]{24,}", raw):
        raise PublicationError("Credential-shaped bytes found in a public payload.")


def files(root):
    root = plain(root)
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise PublicationError("Public payload contains a link or special file.")
        if path.is_file():
            name = path.relative_to(root).as_posix(); relative(name)
            raw = read(path); scan(raw)
            result.append({"path": name, **digest(raw), "representation": "lfs" if name.endswith((".gz", ".zip")) else "git"})
    if not 1 <= len(result) <= 512 or sum(item["bytes"] for item in result) > 64*1024*1024:
        raise PublicationError("Payload inventory exceeds the fixed bounds.")
    return result


def public_client():
    os.environ.update(HF_DEBUG="0", HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                      HF_HUB_DISABLE_PROGRESS_BARS="1")
    logging.disable(logging.CRITICAL)
    from huggingface_hub import HfApi
    return HfApi(endpoint="https://huggingface.co", token=False)


def download(repo, name, kind, revision, cache):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo, name, repo_type=kind, revision=revision, token=False,
        endpoint="https://huggingface.co", cache_dir=str(cache))
    return Path(path).read_bytes()


def tree(api, kind, revision=None):
    info = api.repo_info(REPOS[kind], repo_type=kind, revision=revision, files_metadata=True, token=False)
    if info.private is not False or not OID.fullmatch(info.sha or "") or kind == "space" and getattr(info, "sdk", None) != "static":
        raise PublicationError("Expected the fixed public repository and static Space.")
    result = {}
    for item in info.siblings:
        relative(item.rfilename)
        if (item.rfilename in result or not isinstance(item.size, int) or item.size < 0
                or not OID.fullmatch(item.blob_id or "")
                or item.lfs and not re.fullmatch(r"[a-f0-9]{64}", item.lfs.sha256 or "")):
            raise PublicationError("Public repository metadata lacks exact object identities.")
        result[item.rfilename] = {"bytes": item.size, "blob": item.blob_id,
                                  "lfs_sha256": item.lfs.sha256 if item.lfs else None}
    return {"commit": info.sha, "tree": result}


def compare(actual, expected):
    if set(actual) != set(expected):
        raise PublicationError("Public file roster differs from the reviewed tree.")
    for name, want in expected.items():
        got = actual[name]
        if (got["bytes"] != want["bytes"] or got["lfs_sha256"] != want["lfs_sha256"]
                or want["blob"] is not None and got["blob"] != want["blob"]):
            raise PublicationError("Public Git/LFS object identity differs from the reviewed tree.")


def expected_tree(row, baseline, count=None):
    result = dict(baseline)
    selected = row["files"] if count is None else row["files"][:count]
    for item in selected:
        raw = read(Path(row["root"])/relative(item["path"]))
        if digest(raw) != {key: item[key] for key in ("bytes", "sha256")}:
            raise PublicationError("Staged bytes changed.")
        result[item["path"]] = {"bytes": len(raw), "blob": sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest() if item["representation"] == "git" else None,
                                "lfs_sha256": item["sha256"] if item["representation"] == "lfs" else None}
    return result


EXPECTED_COUNTS = {"native_checks_passed": 12, "native_goal_successes": 6,
    "native_scheduled": 12, "native_valid": 12, "negative_goals_correctly_rejected": 6,
    "negative_scheduled": 6, "positive_goal_successes": 6, "positive_scheduled": 6,
    "prechecks_correctly_rejected": 4, "prechecks_scheduled": 4}


def validate_audit(audit):
    """Reject incomplete/unknown outcomes before any remote write is prepared."""
    cases = audit.get("cases")
    if (audit.get("schema") != "dvidia.task-scene-audit/v2" or audit.get("mismatches") != 0
            or audit.get("counts") != EXPECTED_COUNTS or audit.get("trace_samples") != 13262
            or audit.get("replay_frames") != 1337 or not isinstance(cases, list) or len(cases) != 12
            or [case.get("id") for case in cases] != [f"case-{i:02d}" for i in range(12)]):
        raise PublicationError("Expected the complete independently audited 16-case passive scene study.")
    for index, case in enumerate(cases):
        if (case.get("mismatches") != 0 or type(case.get("goal_success")) is not bool
                or case["goal_success"] != (index < 6)
                or type(case.get("trace_samples")) is not int or case["trace_samples"] <= 0
                or type(case.get("replay_frames")) is not int or case["replay_frames"] <= 0):
            raise PublicationError("Audit contains missing, unknown or inconsistent native outcomes.")
    if (sum(case["trace_samples"] for case in cases) != 13262
            or sum(case["replay_frames"] for case in cases) != 1337):
        raise PublicationError("Per-case audit coverage differs from the complete saved trace roster.")


def validate_study(kit, audit):
    report = json.loads(read(kit/"benchmark/report.json"))
    protocol_raw = read(kit/"benchmark/protocol.json")
    protocol = json.loads(protocol_raw)
    if (report.get("registered_benchmark") is not True or report.get("learned_skill") is not False
            or report.get("physical_robot_ready") is not False
            or {key: report.get(key) for key in EXPECTED_COUNTS} != EXPECTED_COUNTS
            or any(type(report.get(key)) is not int for key in EXPECTED_COUNTS)
            or report.get("protocol_sha256") != sha256(protocol_raw).hexdigest()
            or audit.get("protocol_sha256") != sha256(protocol_raw).hexdigest()
            or protocol.get("schema") != "dvidia.task-scene-benchmark/v2"
            or len(protocol.get("cases", [])) != 16):
        raise PublicationError("The sealed study report differs from its complete audited protocol.")
    results, prechecks = report.get("results"), report.get("prechecks")
    if not isinstance(results, list) or len(results) != 12 or not isinstance(prechecks, list) or len(prechecks) != 4:
        raise PublicationError("The sealed study omits native attempts or invalid-layout prechecks.")
    for index, result in enumerate(results):
        goal, checks = result.get("native_goal", {}), result.get("checks")
        if (result.get("case", {}).get("id") != f"case-{index:02d}" or result.get("valid") is not True
                or result.get("registered_benchmark") is not True or result.get("checks_passed") is not True
                or not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values())
                or result.get("native_warnings") != [] or goal.get("unknown") is not False
                or goal.get("reason") is not None or not goal.get("objects")
                or type(result.get("native_goal_success")) is not bool
                or result["native_goal_success"] != (index < 6)
                or goal.get("success") is not result["native_goal_success"]
                or result.get("expected_goal_success") is not (index < 6)):
            raise PublicationError("The sealed study contains unknown, failed or inconsistent native attempts.")
    for index, precheck in enumerate(prechecks, 12):
        if (precheck.get("id") != f"case-{index:02d}" or precheck.get("rejected") is not True
                or precheck.get("native_run") is not False or precheck.get("expected_precheck_rejection") is not True):
            raise PublicationError("The sealed study contains missing or unknown layout precheck outcomes.")


def stage(kit, output, *, api=None, downloader=None):
    kit, output = plain(kit), plain(output)
    if output.exists():
        raise PublicationError("Use a fresh publication-plan directory.")
    manifest_raw = read(kit/"manifest.json"); manifest = json.loads(manifest_raw)
    actual = {item["path"]: {key: item[key] for key in ("bytes", "sha256")} for item in files(kit)}
    if actual != {**manifest["files"], "manifest.json": digest(manifest_raw)}:
        raise PublicationError("Public kit differs from its sealed complete manifest.")
    audit = json.loads(read(kit/"audit.json"))
    validate_audit(audit)
    validate_study(kit, audit)
    archive = kit.with_suffix(".zip"); receipt = load(archive.with_suffix(".receipt.json"))
    archive_raw = read(archive)
    if receipt["archive"] != {"name": archive.name, "files": len(actual), **digest(archive_raw)} or receipt["manifest"] != digest(manifest_raw):
        raise PublicationError("Public archive differs from its sealed receipt.")
    with zipfile.ZipFile(archive) as zipped:
        if set(zipped.namelist()) != set(actual) or len(zipped.namelist()) != len(actual):
            raise PublicationError("Public archive file membership differs.")
        for name in actual:
            if digest(zipped.read(name)) != actual[name]:
                raise PublicationError("Public archive member differs from the kit.")
    api = api or public_client()
    downloader = downloader or (lambda repo, name, kind, rev: download(repo, name, kind, rev, output.parent/"task-environments-stage-anonymous-cache"))
    from huggingface_hub.errors import RepositoryNotFoundError
    try:
        dataset = tree(api, "dataset")
    except RepositoryNotFoundError:
        dataset = None
    if dataset is not None and set(dataset["tree"])-{".gitattributes"}:
        raise PublicationError("The fixed dataset destination already contains unrelated material.")
    space = tree(api, "space")
    if any(name.startswith(NAMESPACE) or name in ("task-environments", NAMESPACE.rstrip("/")) for name in space["tree"]):
        raise PublicationError("The append-only Space namespace is already occupied.")
    attrs = downloader(REPOS["space"], ".gitattributes", "space", space["commit"])
    expected_attributes = {"bytes": len(attrs), "blob": sha1(b"blob "+str(len(attrs)).encode()+b"\0"+attrs).hexdigest(), "lfs_sha256": None}
    compare({".gitattributes": space["tree"][".gitattributes"]}, {".gitattributes": expected_attributes})
    if b"*.gz filter=lfs diff=lfs merge=lfs -text" not in attrs.splitlines():
        raise PublicationError("Existing Space attributes do not declare the gzip LFS mapping.")
    if tree(api, "space") != space:
        raise PublicationError("Space moved during anonymous baseline capture.")
    output.mkdir(parents=True)
    shutil.copytree(kit, output/"space"/NAMESPACE)
    dataset_files = {"README.md": ("---\nlicense: mit\nlanguage:\n- en\ntags:\n- robotics\n- simulation\n- openusd\n- mujoco\n- agilex-piper\n- dvidia\npretty_name: DVIDIA task environments v0.2\nsize_categories:\n- n<1K\n---\n\n# DVIDIA task environments v0.2\n\n"
        "Reusable local environment recipes for a pinned native PiPER arm, a five-collider open bin, "
        "and up to four original rigid object instances. A rigid shoe surrogate is compound box geometry; "
        "it is not a measured or deformable shoe. All dynamics parameters are authored assumptions.\n\n"
        "The frozen passive benchmark retains 6 positive native scenes, 6 negative completion fixtures, "
        "and 4 invalid-layout prechecks. All 12 native outcomes matched their declarations; all 4 invalid layouts were rejected. "
        "A separate read-only audit reconstructed 13,262 recorded 1 ms state/goal samples and 1,337 saved replay frames with zero consistency mismatches. "
        "These counts measure scene and verifier checks, not a learned robot manipulation success rate.\n\n"
        "[Open the environment studio and recorded benchmark](https://dvidia-robot-simulation-lab.static.hf.space/task-environments/v02/index.html) · "
        "[Source and Terminal tools](https://github.com/Dvidia-Inference/dvidia-training)\n\n"
        "Download `task-environments-v02.zip`, extract it, and serve the directory with `python -m http.server 8000`. "
        "The browser can edit a draft and replay saved native motion. Running new physics requires the local Python runtime and verified PiPER assets; "
        "a static hosted page does not itself run the native simulator. See `guide.md` for installation, local UI/Terminal commands, and supported inputs.\n\n"
        "PiPER holds its native home controls. No grasp policy is trained, no video-to-skill learning or physical-robot transfer is demonstrated, "
        "and the consistency audit does not establish real-world physics or sensor accuracy. "
        "Negative controls explicitly disable gravity in two stationary verifier fixtures and set initial velocity with zero friction in one moving fixture; "
        "all interventions and attempts are retained. Geometry is stored once by content hash.\n\n"
        "Original DVIDIA code and geometry: MIT; pinned PiPER simulation assets: MIT with notices in the archive. "
        "Other dependencies retain their own licenses. DVIDIA research · [@iammrriver](https://x.com/iammrriver) · hello@dvidia.org.\n").encode(),
        ".gitattributes": b"*.zip filter=lfs diff=lfs merge=lfs -text\n",
        "task-environments-v02.zip": archive_raw, "guide.md": read(kit/"docs/task-environments-v02.md"),
        "audit.json": read(kit/"audit.json"), "manifest.json": manifest_raw,
        "LICENSE": read(kit/"LICENSE"), "environment.json": read(kit/"environment.json"),
        "archive-receipt.json": encoded(receipt)}
    (output/"dataset").mkdir()
    for name, raw in dataset_files.items():
        (output/"dataset"/name).write_bytes(raw)
    rows = [{"kind": kind, "repo": REPOS[kind], "root": str(output/kind), "files": files(output/kind),
             "baseline": dataset if kind == "dataset" else space} for kind in REPOS]
    plan = {"schema": SCHEMA, "public": True, "namespace": NAMESPACE, "repositories": rows,
            "implementation": implementation(), "kit_manifest": digest(manifest_raw), "kit_archive": digest(archive_raw),
            "qualified_robot_skill": False, "physical_robot_ready": False}
    atomic(output/"plan.json", plan, exclusive=True)
    checked(output/"plan.json")
    return plan


def checked(path):
    path = plain(path); plan = load(path)
    if (plan.get("schema") != SCHEMA or plan.get("public") is not True or plan.get("namespace") != NAMESPACE
            or plan.get("qualified_robot_skill") is not False or plan.get("physical_robot_ready") is not False
            or plan.get("implementation") != implementation()
            or [row.get("kind") for row in plan.get("repositories", [])] != list(REPOS)):
        raise PublicationError("Plan differs from the fixed reviewed task-environment publication contract.")
    for row in plan["repositories"]:
        kind = row["kind"]
        if row["repo"] != REPOS[kind] or plain(row["root"]) != path.parent/kind or files(row["root"]) != row["files"]:
            raise PublicationError("Publication destination, snapshot or bytes changed.")
        if kind == "space" and (row["baseline"] is None or any(not item["path"].startswith(NAMESPACE) for item in row["files"])):
            raise PublicationError("Space writes must remain inside the new namespace.")
        baseline = row["baseline"]
        if baseline is not None and (not OID.fullmatch(baseline.get("commit", "")) or not isinstance(baseline.get("tree"), dict)):
            raise PublicationError("Staged baseline lacks an exact commit and tree.")
        if kind == "space" and any(name.startswith(NAMESPACE) or name in ("task-environments", NAMESPACE.rstrip("/")) for name in baseline["tree"]):
            raise PublicationError("Append-only namespace must be absent from the reviewed baseline.")
        if kind == "dataset" and baseline is not None and set(baseline["tree"])-{".gitattributes"}:
            raise PublicationError("The reviewed dataset baseline cannot contain unrelated material.")
    return plan


def _publish(path, api, token, operation_cls):
    path = plain(path); plan = checked(path); plan_id = digest(read(path, MAX_PLAN))
    journal = path.parent/"journal.json"
    state = load(journal) if journal.exists() else {"schema": SCHEMA, "plan": plan_id, "repositories": {}}
    if state.get("plan") != plan_id:
        raise PublicationError("Existing durable intent belongs to another plan.")
    if "dvidia" not in {str(org.get("name", "")).lower() for org in api.whoami().get("orgs", [])}:
        raise PublicationError("The publisher is not a member of the fixed Dvidia organization.")
    for row in plan["repositories"]:
        kind = row["kind"]; value = state["repositories"].get(kind)
        if value is None:
            baseline = row["baseline"]
            if baseline is None:
                state["repositories"][kind] = {"creating": True}; atomic(journal, state)
                api.create_repo(REPOS[kind], repo_type=kind, private=False, exist_ok=False)
                baseline = tree(api, kind)
                if set(baseline["tree"])-{".gitattributes"}:
                    raise PublicationError("New dataset contains an unexpected tree.")
            value = {"parent": baseline["commit"], "baseline": baseline["tree"], "completed": [], "pending": None}
            state["repositories"][kind] = value; atomic(journal, state)
        elif value.get("creating"):
            baseline = tree(api, kind)
            if kind != "dataset" or set(baseline["tree"])-{".gitattributes"}:
                raise PublicationError("Uncertain repository creation produced an unexpected destination.")
            value = {"parent": baseline["commit"], "baseline": baseline["tree"], "completed": [], "pending": None}
            state["repositories"][kind] = value; atomic(journal, state)
        batches = [row["files"][start:start+100] for start in range(0, len(row["files"]), 100)]
        if (row["baseline"] is not None and (value["parent"] != row["baseline"]["commit"] or value["baseline"] != row["baseline"]["tree"])
                or len(value["completed"]) > len(batches) or any(not OID.fullmatch(oid or "") for oid in value["completed"])):
            raise PublicationError("Durable journal differs from the reviewed parent or completion roster.")
        for index, batch in enumerate(batches):
            cumulative = sum(len(items) for items in batches[:index+1])
            wanted = expected_tree(row, value["baseline"], cumulative)
            if index < len(value["completed"]):
                continue
            parent = value["completed"][-1] if value["completed"] else value["parent"]
            title = "DVIDIA task environments v02 "+plan_id["sha256"]+" "+kind+" "+str(index)
            if value["pending"] is not None and value["pending"] != {"index": index, "parent": parent, "title": title}:
                raise PublicationError("Pending intent differs from the exact next reviewed commit.")
            head = tree(api, kind)
            if value["pending"] and head["commit"] != parent:
                history = api.list_repo_commits(REPOS[kind], repo_type=kind, revision=head["commit"], token=False)
                if (len(history) < 2 or history[0].commit_id != head["commit"] or history[0].title != title
                        or history[1].commit_id != parent):
                    raise PublicationError("Uncertain write is not the exact intended immediate child.")
                compare(head["tree"], wanted)
                value["completed"].append(head["commit"]); value["pending"] = None; atomic(journal, state)
                continue
            prior = expected_tree(row, value["baseline"], cumulative-len(batch))
            if head["commit"] != parent:
                raise PublicationError("Destination moved before its parent-CAS write.")
            compare(head["tree"], prior)
            operations = []
            for item in batch:
                raw = read(Path(row["root"])/relative(item["path"]))
                if digest(raw) != {key: item[key] for key in ("bytes", "sha256")} or token.encode() in raw:
                    raise PublicationError("Upload bytes changed or contain the publishing credential.")
                operation = operation_cls(path_in_repo=item["path"], path_or_fileobj=raw)
                operation._upload_mode = "lfs" if item["representation"] == "lfs" else "regular"
                operation._should_ignore = False
                operations.append(operation)
            value["pending"] = {"index": index, "parent": parent, "title": title}; atomic(journal, state)
            try:
                committed = api.create_commit(REPOS[kind], repo_type=kind, parent_commit=parent,
                    operations=operations, commit_message=title)
            except Exception:
                raise PublicationError("Commit reply uncertain; preserve the pending journal for reconciliation.") from None
            head = tree(api, kind)
            if head["commit"] != committed.oid:
                raise PublicationError("Repository moved after the guarded commit.")
            compare(head["tree"], wanted); checked(path)
            value["completed"].append(head["commit"]); value["pending"] = None; atomic(journal, state)
            print(json.dumps({"kind": kind, "batch": index, "commit": head["commit"], "files": len(batch)}), flush=True)
        final = tree(api, kind)
        if final["commit"] != value["completed"][-1]:
            raise PublicationError("Completed repository moved.")
        compare(final["tree"], expected_tree(row, value["baseline"]))
    receipt = {"schema": SCHEMA, "plan": plan_id, "repositories": state["repositories"],
               "all_git_lfs_identities_verified": True, "anonymous_full_readback": False, "physical_robot_ready": False}
    atomic(path.parent/"publication-receipt.json", receipt)
    return receipt


def publish(path):
    path = plain(path); checked(path)
    api, token = client(path.parent/"publisher-cache")
    from huggingface_hub import CommitOperationAdd
    return _publish(path, api, token, CommitOperationAdd)


def verify(path):
    path = plain(path); plan = checked(path); receipt = load(path.parent/"publication-receipt.json")
    if receipt["plan"] != digest(read(path, MAX_PLAN)):
        raise PublicationError("Completed receipt belongs to another plan.")
    api = public_client(); proofs = []
    for row in plan["repositories"]:
        kind = row["kind"]; value = receipt["repositories"][kind]; revision = value["completed"][-1]
        head = tree(api, kind)
        if head["commit"] != revision:
            raise PublicationError("Repository moved before anonymous readback.")
        compare(head["tree"], expected_tree(row, value["baseline"]))
        def one(item):
            raw = download(REPOS[kind], item["path"], kind, revision, path.parent/"anonymous-readback-cache")
            if digest(raw) != {key: item[key] for key in ("bytes", "sha256")}:
                raise PublicationError("Anonymous pinned download differs from its staged bytes.")
            return {"path": item["path"], **digest(raw)}
        with ThreadPoolExecutor(max_workers=4) as pool:
            downloaded = list(pool.map(one, row["files"]))
        if tree(api, kind)["commit"] != revision:
            raise PublicationError("Repository moved during anonymous readback.")
        proofs.append({"kind": kind, "repo": REPOS[kind], "commit": revision, "files": downloaded})
    result = {"schema": SCHEMA, "plan": receipt["plan"], "repositories": proofs,
              "anonymous_full_readback": True, "physical_robot_ready": False}
    atomic(path.parent/"anonymous-readback.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("stage", "check", "publish", "verify"))
    parser.add_argument("--kit"); parser.add_argument("--output"); parser.add_argument("--plan")
    args = parser.parse_args(argv)
    try:
        if args.command == "stage":
            result = stage(args.kit, args.output)
        elif args.command == "check":
            result = checked(args.plan)
        elif args.command == "publish":
            result = publish(args.plan)
        else:
            result = verify(args.plan)
        print(json.dumps({"schema": result["schema"], "command": args.command, "completed": True}), flush=True)
        return 0
    except Exception as exc:
        print("Task-environment publication stopped: "+(str(exc) if isinstance(exc, PublicationError) else type(exc).__name__), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

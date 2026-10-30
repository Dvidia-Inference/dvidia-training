"""One reviewed Space-only recovery; the accepted dataset is read-only reused.

The original publisher, plan, journal and uncertain intents stay unchanged.
Only two server-classified MP4 representations change; every payload byte,
existing Space object, root file and credential handling remains preserved.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sys

import publish_piper_action_capture_v01 as old


SCHEMA = "dvidia.piper-action-capture-hub-recovery/v1"
PLAN01 = "8248c72bbbfed36aa315809f15bfaf5c7e128222bc71885b6940a24971e86ba8"
JOURNAL01 = "c97e8bd4fac0aa2c0cd7a81ecdf2d343f177cf708f802e348989a6226c0c55f2"
DATASET = "ccfc321014afc5d85cede6ec23e65c40aa9a733f"
SPACE_PARENT = "aebd46af39c4158bd042c62617b6aecfa5a57861"
CHANGED = tuple(old.NAMESPACE+"media/"+name+".mp4" for name in ("nominal-full", "nominal-repeat"))
MP4 = {"bytes": 213565, "sha256": "fca479dcb3a71b0058a07e42c4fa76700b56825a860b731bb548d042fee0ac7f"}
EVIDENCE = ("plan.json", "journal.json", "uncertain-space-probe02.json", "upload-classification01.json",
            "dataset-anonymous-partial-readback.json")


def implementation():
    return {Path(__file__).name: old.digest(old.read(__file__)),
            **{name: old.digest(old.read(Path(__file__).with_name(name))) for name in
               ("publish_piper_action_capture_v01.py", "publish.py", "prepare.py")}}


def source(previous):
    """Validate the exact old evidence without altering its durable intent."""
    previous = old.plain(previous)
    if old.digest(old.read(previous, old.MAX_PLAN))["sha256"] != PLAN01:
        raise old.PublicationError("Recovery requires the exact previously reviewed plan.")
    plan = old.checked(previous)
    evidence = {name: old.digest(old.read(previous.parent/name, old.MAX_PLAN)) for name in EVIDENCE}
    if evidence["journal.json"]["sha256"] != JOURNAL01:
        raise old.PublicationError("The original uncertain intent changed.")
    journal = old.load(previous.parent/"journal.json")
    if journal.get("schema") != old.SCHEMA or journal.get("plan") != evidence["plan.json"]:
        raise old.PublicationError("Original journal does not bind the old plan.")
    dataset, space = plan["repositories"]
    states = journal.get("repositories", {})
    ds, sp = states.get("dataset", {}), states.get("space", {})
    title = "DVIDIA PiPER action capture v01 "+PLAN01+" space 0"
    if (ds.get("completed") != [DATASET] or ds.get("pending") is not None
            or sp.get("completed") != [] or sp.get("parent") != SPACE_PARENT
            or sp.get("baseline") != space["baseline"]["tree"]
            or sp.get("pending") != {"index": 0, "parent": SPACE_PARENT, "title": title}
            or space["baseline"]["commit"] != SPACE_PARENT or len(sp["baseline"]) != 4122):
        raise old.PublicationError("Recovery refuses a foreign or resolved old intent.")
    probe = old.load(previous.parent/"uncertain-space-probe02.json")
    found = next((r for r in probe.get("repositories", []) if r.get("kind") == "space"), {})
    if (probe.get("remote_mutations") is not False or probe.get("plan") != evidence["plan.json"]
            or probe.get("journal") != evidence["journal.json"] or found.get("commit") != SPACE_PARENT
            or found.get("files") != 4122 or found.get("pending") != sp["pending"]
            or found.get("unchanged_parent") is not True or found.get("exact_unchanged_baseline_tree") is not True
            or found.get("intended_immediate_child") is not False):
        raise old.PublicationError("Uncertainty probe does not prove the exact unchanged Space parent.")
    classifier = old.load(previous.parent/"upload-classification01.json")
    if (classifier.get("plan") != evidence["plan.json"] or classifier.get("journal") != evidence["journal.json"]
            or classifier.get("parent") != SPACE_PARENT or classifier.get("classification_only") is not True
            or classifier.get("remote_mutations") is not False):
        raise old.PublicationError("Upload classification is not bound to the old intent.")
    expected = [{"path": f["path"], "bytes": f["bytes"], "reported_upload_mode": "lfs" if f["path"] in CHANGED else "regular",
                 "should_ignore": False, "staged_representation": f["representation"]} for f in space["files"]]
    if classifier.get("files") != expected or len(expected) != 27:
        raise old.PublicationError("Only the two reviewed MP4 classifications may differ.")
    for f in space["files"]:
        if f["path"] in CHANGED and ({k: f[k] for k in MP4} != MP4 or f["representation"] != "git"):
            raise old.PublicationError("Classified MP4 bytes changed.")
    proof = old.load(previous.parent/"dataset-anonymous-partial-readback.json")
    wanted = [{k: f[k] for k in ("path", "bytes", "sha256")} for f in dataset["files"]]
    if (proof.get("remote_mutations") is not False or proof.get("plan") != evidence["plan.json"]
            or proof.get("dataset") != {"repo": old.REPOS["dataset"], "commit": DATASET, "files": wanted,
                "git_lfs_identities_verified": True, "anonymous_force_download": True}
            or proof.get("space_publication_complete") is not False or len(wanted) != 23):
        raise old.PublicationError("Accepted dataset lacks its complete anonymous pinned byte proof.")
    return plan, journal, evidence


def projection(kind, root):
    rows = old.files(root)
    if kind == "space":
        for item in rows:
            if item["path"] in CHANGED:
                if {key: item[key] for key in MP4} != MP4:
                    raise old.PublicationError("Recovery MP4 bytes changed.")
                item["representation"] = "lfs"
    return rows


def _accepted_dataset(api, original, journal):
    head = old.tree(api, "dataset")
    if head["commit"] != DATASET:
        raise old.PublicationError("Previously accepted dataset moved.")
    old.compare(head["tree"], old.expected_tree(original["repositories"][0], journal["repositories"]["dataset"]["baseline"]))
    return head


def stage(previous, output, *, api=None):
    previous, output = old.plain(previous), old.plain(output)
    if output.exists():
        raise old.PublicationError("Use a fresh recovery directory.")
    original, journal, evidence = source(previous)
    api = api or old.public_client()
    _accepted_dataset(api, original, journal)
    if old.tree(api, "space") != original["repositories"][1]["baseline"]:
        raise old.PublicationError("Space moved before recovery staging.")
    output.mkdir(parents=True)
    rows = []
    for row in original["repositories"]:
        for item in row["files"]:
            raw = old.read(Path(row["root"])/old.relative(item["path"]))
            target = output/row["kind"]/item["path"]
            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(raw)
        rows.append({**row, "root": str(output/row["kind"]), "files": projection(row["kind"], output/row["kind"])})
    plan = {"schema": SCHEMA, "public": True, "namespace": old.NAMESPACE, "previous_plan": str(previous),
            "evidence": evidence, "implementation": implementation(), "repositories": rows,
            "representation_changes": [{"path": name, "from": "git", "to": "lfs", **MP4} for name in CHANGED],
            "dataset_completion": {"status": "reused-previously-accepted", "commit": DATASET, "new_dataset_writes": False},
            "original_pending_intent_retained": journal["repositories"]["space"]["pending"],
            "qualified_robot_skill": False, "physical_robot_ready": False}
    old.atomic(output/"plan.json", plan, exclusive=True)
    checked(output/"plan.json")
    return plan


def checked(path):
    path = old.plain(path); plan = old.load(path)
    if (plan.get("schema") != SCHEMA or plan.get("public") is not True or plan.get("namespace") != old.NAMESPACE
            or plan.get("implementation") != implementation() or plan.get("qualified_robot_skill") is not False
            or plan.get("physical_robot_ready") is not False):
        raise old.PublicationError("Recovery plan or source differs from its reviewed contract.")
    original, journal, evidence = source(plan["previous_plan"])
    if (plan.get("evidence") != evidence
            or plan.get("dataset_completion") != {"status": "reused-previously-accepted", "commit": DATASET, "new_dataset_writes": False}
            or plan.get("original_pending_intent_retained") != journal["repositories"]["space"]["pending"]
            or plan.get("representation_changes") != [{"path": name, "from": "git", "to": "lfs", **MP4} for name in CHANGED]
            or len(plan.get("repositories", [])) != 2):
        raise old.PublicationError("Recovery transition or prior evidence changed.")
    for row, prior in zip(plan["repositories"], original["repositories"]):
        expected = [{**f, "representation": "lfs" if f["path"] in CHANGED else f["representation"]} for f in prior["files"]]
        if (row != {**prior, "root": str(path.parent/prior["kind"]), "files": expected}
                or projection(row["kind"], row["root"]) != expected):
            raise old.PublicationError("Recovery may change only two Space representations, never payloads or destinations.")
    return plan


def _completed(path, plan, state):
    if (state.get("schema") != SCHEMA or state.get("plan") != old.digest(old.read(path, old.MAX_PLAN))
            or state.get("dataset_reused") != plan["dataset_completion"]
            or state.get("parent") != SPACE_PARENT or state.get("pending") is not None
            or not old.OID.fullmatch(state.get("commit") or "")):
        raise old.PublicationError("Recovery completion differs from the fixed accepted transition.")


def _publish(path, api, token, operation_cls):
    """At most one Space commit; no dataset or repository creation methods."""
    path = old.plain(path); plan = checked(path); plan_id = old.digest(old.read(path, old.MAX_PLAN))
    original, original_journal, _ = source(plan["previous_plan"])
    _accepted_dataset(api, original, original_journal)
    if "dvidia" not in {str(x.get("name", "")).lower() for x in api.whoami().get("orgs", [])}:
        raise old.PublicationError("Publisher is not a member of the fixed Dvidia organization.")
    row = plan["repositories"][1]; wanted = old.expected_tree(row, row["baseline"]["tree"])
    journal_path = path.parent/"journal.json"
    state = old.load(journal_path) if journal_path.exists() else {"schema": SCHEMA, "plan": plan_id,
        "dataset_reused": plan["dataset_completion"], "parent": SPACE_PARENT, "pending": None, "commit": None}
    if (state.get("schema") != SCHEMA or state.get("plan") != plan_id
            or state.get("dataset_reused") != plan["dataset_completion"] or state.get("parent") != SPACE_PARENT):
        raise old.PublicationError("Recovery durable intent belongs to another transition.")
    head = old.tree(api, "space")
    title = "DVIDIA PiPER action capture v01 recovery "+plan_id["sha256"]
    intent = {"parent": SPACE_PARENT, "title": title, "files": 27}
    if state.get("commit") is not None:
        _completed(path, plan, state)
        if head["commit"] != state["commit"]:
            raise old.PublicationError("Completed Space recovery moved.")
        old.compare(head["tree"], wanted)
    elif state.get("pending") is not None:
        if state["pending"] != intent:
            raise old.PublicationError("A foreign pending recovery intent cannot be resolved.")
        if head["commit"] == SPACE_PARENT:
            raise old.PublicationError("Recovery reply is pending at its unchanged parent; no automatic write retry.")
        history = api.list_repo_commits(old.REPOS["space"], repo_type="space", revision=head["commit"], token=False)
        if (len(history) < 2 or history[0].commit_id != head["commit"] or history[0].title != title
                or history[1].commit_id != SPACE_PARENT):
            raise old.PublicationError("Recovery is not the exact intended immediate child.")
        old.compare(head["tree"], wanted)
        state.update(commit=head["commit"], pending=None); old.atomic(journal_path, state)
    else:
        if head != row["baseline"]:
            raise old.PublicationError("Space moved before the single recovery parent-CAS commit.")
        operations = []
        for item in row["files"]:
            raw = old.read(Path(row["root"])/item["path"])
            if old.digest(raw) != {k: item[k] for k in ("bytes", "sha256")} or token.encode() in raw:
                raise old.PublicationError("Recovery payload changed or contains the publishing credential.")
            op = operation_cls(path_in_repo=item["path"], path_or_fileobj=raw)
            op._upload_mode = "lfs" if item["representation"] == "lfs" else "regular"
            op._should_ignore = False; operations.append(op)
        state["pending"] = intent; old.atomic(journal_path, state)
        try:
            committed = api.create_commit(old.REPOS["space"], repo_type="space", parent_commit=SPACE_PARENT,
                operations=operations, commit_message=title)
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            old.atomic(path.parent/"uncertain-error.json", {"schema": SCHEMA, "plan": plan_id,
                "exception_class": type(exc).__name__, "status_code": status})
            raise old.PublicationError("Recovery commit reply uncertain; no automatic retry. Safe exception class/status saved.") from None
        head = old.tree(api, "space")
        if head["commit"] != committed.oid:
            raise old.PublicationError("Space moved after the single recovery commit.")
        old.compare(head["tree"], wanted); checked(path)
        state.update(commit=head["commit"], pending=None); old.atomic(journal_path, state)
    _completed(path, plan, state)
    receipt = {**state, "all_git_lfs_identities_verified": True, "anonymous_space_full_readback": False,
               "dataset_completion_reused": True, "physical_robot_ready": False}
    old.atomic(path.parent/"publication-receipt.json", receipt)
    return receipt


def publish(path):
    path = old.plain(path)
    with old._lock(path.parent):
        checked(path)
        api, token = old.client(path.parent/"publisher-cache")
        from huggingface_hub import CommitOperationAdd
        return _publish(path, api, token, CommitOperationAdd)


def verify(path, *, api=None, downloader=None):
    path = old.plain(path); plan = checked(path); journal = old.load(path.parent/"journal.json")
    _completed(path, plan, journal)
    receipt = old.load(path.parent/"publication-receipt.json")
    if receipt != {**journal, "all_git_lfs_identities_verified": True, "anonymous_space_full_readback": False,
                   "dataset_completion_reused": True, "physical_robot_ready": False}:
        raise old.PublicationError("Recovery receipt differs from completed durable intent.")
    original, original_journal, _ = source(plan["previous_plan"])
    api = api or old.public_client(); _accepted_dataset(api, original, original_journal)
    row = plan["repositories"][1]; revision = journal["commit"]
    head = old.tree(api, "space")
    if head["commit"] != revision:
        raise old.PublicationError("Space moved before anonymous recovery readback.")
    old.compare(head["tree"], old.expected_tree(row, row["baseline"]["tree"]))
    def one(item):
        raw = (downloader(old.REPOS["space"], item["path"], "space", revision) if downloader else
               old.download(old.REPOS["space"], item["path"], "space", revision,
                   path.parent/"anonymous-readback-cache", force_download=True))
        if old.digest(raw) != {k: item[k] for k in ("bytes", "sha256")}:
            raise old.PublicationError("Anonymous recovery download differs from reviewed payload bytes.")
        return {"path": item["path"], **old.digest(raw)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        proofs = list(pool.map(one, row["files"]))
    if old.tree(api, "space")["commit"] != revision or old.tree(api, "dataset")["commit"] != DATASET:
        raise old.PublicationError("A public destination moved during recovery readback.")
    result = {"schema": SCHEMA, "plan": journal["plan"], "space": {"repo": old.REPOS["space"], "commit": revision,
        "files": proofs, "anonymous_force_download": True}, "dataset": {**plan["dataset_completion"],
        "readback_evidence": plan["evidence"]["dataset-anonymous-partial-readback.json"]},
        "all_new_50_file_bytes_verified": True, "physical_robot_ready": False}
    old.atomic(path.parent/"anonymous-readback.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("stage", "check", "publish", "verify"))
    parser.add_argument("--previous-plan"); parser.add_argument("--output"); parser.add_argument("--plan")
    args = parser.parse_args(argv)
    try:
        value = (stage(args.previous_plan, args.output) if args.command == "stage" else
                 checked(args.plan) if args.command == "check" else
                 publish(args.plan) if args.command == "publish" else verify(args.plan))
        print(json.dumps({"schema": value["schema"], "command": args.command, "completed": True})); return 0
    except Exception as exc:
        print("Capture recovery stopped: "+(str(exc) if isinstance(exc, old.PublicationError) else type(exc).__name__), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

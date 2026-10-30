"""Read-only acceptance of an exact, disclosed Hub-side LFS attribute append.

The accepted Space commit failed the old plan's unchanged-attributes condition.
This does not rewrite either pending journal or claim the old condition passed.
It verifies that one narrowly specified server deviation and the full payload.
No authenticated client, upload, repository or commit mutation is used.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sys

import recover_piper_action_capture_v01 as recovery


old = recovery.old
SCHEMA = "dvidia.piper-action-capture-hub-reconciliation/v1"
PLAN02 = "0ca2c22ed282b02ae9d1a2c54466dc2643c66dde5c00e302f1b331bfe3951ff4"
JOURNAL02 = "19d033ca6faea0488a0f15f980b91e37125ddbfa5761ff9872077d118e509d4f"
RECOVERY_SOURCE = "2eee440bf8887c67f46340f49156b156b05d54430f41114c060c6a0501a34561"
ACCEPTED = "fad813975694b8688a959bf135babb685b4f2858"
OLD_ATTRIBUTES = {"bytes": 41841, "sha256": "f427576d03cb462ce4fec9686640b293db14878be56389a0cdd4f2e2d67576d0"}
NEW_ATTRIBUTES = {"bytes": 41999, "sha256": "e30ebf01c3e522690b87114ef8a41a70ab6dd14cbb726d5724d0bedf178ca5fe"}
APPEND = ("action-capture/v01/media/nominal-full.mp4 filter=lfs diff=lfs merge=lfs -text\n"
          "action-capture/v01/media/nominal-repeat.mp4 filter=lfs diff=lfs merge=lfs -text\n").encode()


def checked(path):
    path = old.plain(path)
    if (old.digest(old.read(path, old.MAX_PLAN))["sha256"] != PLAN02
            or old.digest(old.read(recovery.__file__))["sha256"] != RECOVERY_SOURCE
            or old.digest(old.read(path.parent/"journal.json", old.MAX_PLAN))["sha256"] != JOURNAL02):
        raise old.PublicationError("Reconciliation requires the exact reviewed recovery source, plan and pending journal.")
    plan = recovery.checked(path); state = old.load(path.parent/"journal.json")
    intent = {"parent": recovery.SPACE_PARENT,
              "title": "DVIDIA PiPER action capture v01 recovery "+PLAN02, "files": 27}
    expected = {"schema": recovery.SCHEMA, "plan": old.digest(old.read(path, old.MAX_PLAN)),
        "dataset_reused": plan["dataset_completion"], "parent": recovery.SPACE_PARENT,
        "pending": intent, "commit": None}
    if state != expected:
        raise old.PublicationError("The stopped recovery intent must remain unchanged and unreconciled.")
    return plan, state


def verify(path, *, api=None, downloader=None):
    path = old.plain(path); plan, state = checked(path)
    output = path.parent/"final-reconciliation-receipt.json"
    if output.exists():
        raise old.PublicationError("The reconciliation evidence already exists; preserve it.")
    source_before = {"plan02": old.digest(old.read(path, old.MAX_PLAN)),
                     "journal02": old.digest(old.read(path.parent/"journal.json", old.MAX_PLAN))}
    original, original_journal, evidence = recovery.source(plan["previous_plan"])
    api = api or old.public_client()
    recovery._accepted_dataset(api, original, original_journal)
    head = old.tree(api, "space")
    if head["commit"] != ACCEPTED:
        raise old.PublicationError("Reconciliation accepts only the one registered public Space commit.")
    history = api.list_repo_commits(old.REPOS["space"], repo_type="space", revision=ACCEPTED, token=False)
    if (len(history) < 2 or history[0].commit_id != ACCEPTED or history[0].title != state["pending"]["title"]
            or history[1].commit_id != recovery.SPACE_PARENT):
        raise old.PublicationError("Accepted commit is not the exact titled immediate child of the reviewed parent.")
    def fetch(name, revision):
        return (downloader(old.REPOS["space"], name, "space", revision) if downloader else
                old.download(old.REPOS["space"], name, "space", revision,
                             path.parent/"anonymous-reconciliation-cache", force_download=True))
    before = fetch(".gitattributes", recovery.SPACE_PARENT)
    after = fetch(".gitattributes", ACCEPTED)
    if old.digest(before) != OLD_ATTRIBUTES or old.digest(after) != NEW_ATTRIBUTES or after != before+APPEND:
        raise old.PublicationError("Hub attributes differ from the exact byte-preserving two-path LFS append.")
    row = plan["repositories"][1]
    expected = old.expected_tree(row, row["baseline"]["tree"])
    expected[".gitattributes"] = {"bytes": len(after),
        "blob": old.sha1(b"blob "+str(len(after)).encode()+b"\0"+after).hexdigest(), "lfs_sha256": None}
    old.compare(head["tree"], expected)
    def one(item):
        raw = fetch(item["path"], ACCEPTED)
        if old.digest(raw) != {key: item[key] for key in ("bytes", "sha256")}:
            raise old.PublicationError("An anonymous pinned Space payload differs from reviewed bytes.")
        return {"path": item["path"], **old.digest(raw)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        proofs = list(pool.map(one, row["files"]))
    if old.tree(api, "space")["commit"] != ACCEPTED or old.tree(api, "dataset")["commit"] != recovery.DATASET:
        raise old.PublicationError("A public destination moved during anonymous reconciliation.")
    checked(path)
    if source_before != {"plan02": old.digest(old.read(path, old.MAX_PLAN)),
                         "journal02": old.digest(old.read(path.parent/"journal.json", old.MAX_PLAN))}:
        raise old.PublicationError("Original stopped intent changed during read-only verification.")
    result = {"schema": SCHEMA, "reconciler_source": old.digest(old.read(__file__)), "inputs": source_before,
        "remote_mutations": False, "authenticated_requests": False,
        "accepted_space": {"repo": old.REPOS["space"], "commit": ACCEPTED, "parent": recovery.SPACE_PARENT,
            "title": state["pending"]["title"], "files": proofs, "anonymous_force_download": True},
        "server_deviation": {"old_baseline_condition_passed": False,
            "description": "Hub appended two exact path-scoped LFS rules to .gitattributes in the accepted commit.",
            "old_attributes": OLD_ATTRIBUTES, "new_attributes": NEW_ATTRIBUTES,
            "appended_utf8": APPEND.decode(), "old_attribute_bytes_preserved_as_prefix": True,
            "other_existing_objects_unchanged": 4121, "all_new_payload_identities_verified": True},
        "dataset": {"repo": old.REPOS["dataset"], "commit": recovery.DATASET,
            "completion": "reused-previously-accepted", "new_dataset_writes": False,
            "files_previously_force_downloaded": 23,
            "pinned_byte_proof": evidence["dataset-anonymous-partial-readback.json"]},
        "original_intents_retained": {"journal01": evidence["journal.json"], "journal02": source_before["journal02"],
            "journal01_pending": plan["original_pending_intent_retained"], "journal02_pending": state["pending"],
            "rewritten": False}, "all_50_file_bytes_verified_at_pinned_revisions": True,
        "qualified_robot_skill": False, "physical_robot_ready": False}
    old.atomic(output, result, exclusive=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "verify")); parser.add_argument("--plan", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "check": checked(args.plan)
        else: verify(args.plan)
        print(json.dumps({"schema": SCHEMA, "command": args.command, "completed": True, "remote_mutations": False})); return 0
    except Exception as exc:
        print("Capture reconciliation stopped: "+(str(exc) if isinstance(exc, old.PublicationError) else type(exc).__name__), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

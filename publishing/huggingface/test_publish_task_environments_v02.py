"""Credential-free publication contract checks against a small in-memory CAS Hub.

Artifacts are explicitly synthetic fixtures; they are never research evidence.
"""
from copy import deepcopy
from hashlib import sha1, sha256
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from huggingface_hub.errors import RepositoryNotFoundError

import publish_task_environments_v02 as pub


def fixture(root):
    root.mkdir()
    cases = [{"id": f"case-{i:02d}"} for i in range(16)]
    protocol = pub.encoded({"schema": "dvidia.task-scene-benchmark/v2", "cases": cases})
    protocol_sha = sha256(protocol).hexdigest()
    traces = [1501]*7+[501, 501, 151, 101, 1501]
    frames = [151]*7+[51, 51, 16, 11, 151]
    audit = {"schema": "dvidia.task-scene-audit/v2", "mismatches": 0,
        "counts": pub.EXPECTED_COUNTS, "protocol_sha256": protocol_sha,
        "trace_samples": 13262, "replay_frames": 1337,
        "scope": "Synthetic test fixture; no research evidence",
        "cases": [{"id": f"case-{i:02d}", "mismatches": 0, "goal_success": i < 6,
                   "trace_samples": traces[i], "replay_frames": frames[i]} for i in range(12)]}
    report = {"registered_benchmark": True, "learned_skill": False,
        "physical_robot_ready": False, **pub.EXPECTED_COUNTS, "protocol_sha256": protocol_sha,
        "results": [{"case": {"id": f"case-{i:02d}"}, "valid": True, "registered_benchmark": True,
            "checks_passed": True, "checks": {"fixture": True}, "native_warnings": [],
            "native_goal": {"unknown": False, "reason": None, "objects": {"block": {}}, "success": i < 6},
            "native_goal_success": i < 6, "expected_goal_success": i < 6} for i in range(12)],
        "prechecks": [{"id": f"case-{i:02d}", "rejected": True, "native_run": False,
                       "expected_precheck_rejection": True} for i in range(12, 16)]}
    payload = {"audit.json": pub.encoded(audit), "benchmark/report.json": pub.encoded(report),
        "benchmark/protocol.json": protocol, "LICENSE": b"MIT synthetic test fixture\n",
        "environment.json": b'{"fixture":true}\n', "docs/task-environments-v02.md": b"Synthetic guide\n",
        "index.html": b"<p>Synthetic recorded fixture</p>\n"}
    for name, raw in payload.items():
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    seal(root)
    return root


def seal(root):
    inventory = {item["path"]: {key: item[key] for key in ("bytes", "sha256")}
                 for item in pub.files(root) if item["path"] != "manifest.json"}
    manifest = pub.encoded({"files": inventory})
    (root/"manifest.json").write_bytes(manifest)
    archive = root.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                zipped.writestr(path.relative_to(root).as_posix(), path.read_bytes())
    receipt = {"archive": {"name": archive.name, "files": len(inventory)+1, **pub.digest(archive.read_bytes())},
               "manifest": pub.digest(manifest)}
    archive.with_suffix(".receipt.json").write_bytes(pub.encoded(receipt))


class Operation:
    def __init__(self, path_in_repo, path_or_fileobj):
        self.path = path_in_repo
        self.raw = path_or_fileobj


class Hub:
    def __init__(self):
        self.counter = 1
        self.heads = {"space": "1"*40}
        self.data = {"space": {".gitattributes": (b"*.gz filter=lfs diff=lfs merge=lfs -text\n", "git"),
            "README.md": (b"Existing shared research entries\n", "git"),
            "index.html": (b"<p>Existing root entry</p>\n", "git"),
            "existing/evidence.gz": (b"old native evidence", "lfs")}}
        self.snapshots = {"1"*40: deepcopy(self.data["space"])}
        self.history = {}
        self.calls = []
        self.created = []
        self.uncertain_once = False

    def repo_info(self, repo, repo_type, revision=None, **kwargs):
        assert repo == pub.REPOS[repo_type]
        if repo_type not in self.heads:
            raise RepositoryNotFoundError("Not found", response=SimpleNamespace(status_code=404, headers={}, request=None))
        oid = revision or self.heads[repo_type]
        contents = self.snapshots[oid]
        siblings = []
        for name, (raw, mode) in contents.items():
            blob = sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
            siblings.append(SimpleNamespace(rfilename=name, size=len(raw), blob_id=blob,
                lfs=SimpleNamespace(sha256=sha256(raw).hexdigest()) if mode == "lfs" else None))
        return SimpleNamespace(private=False, sha=oid, siblings=siblings, sdk="static")

    def whoami(self):
        return {"orgs": [{"name": "Dvidia"}]}

    def create_repo(self, repo, repo_type, private, exist_ok):
        assert repo == pub.REPOS["dataset"] and repo_type == "dataset" and private is False and exist_ok is False
        self.created.append(repo)
        self.heads[repo_type] = "2"*40
        self.data[repo_type] = {".gitattributes": (b"*.zip filter=lfs diff=lfs merge=lfs -text\n", "git")}
        self.snapshots["2"*40] = deepcopy(self.data[repo_type])

    def create_commit(self, repo, repo_type, parent_commit, operations, commit_message):
        assert parent_commit == self.heads[repo_type]
        self.calls.append((repo_type, [op.path for op in operations]))
        for operation in operations:
            self.data[repo_type][operation.path] = (operation.raw, "lfs" if operation._upload_mode == "lfs" else "git")
        self.counter += 1
        oid = f"{self.counter+10:040x}"
        self.history[oid] = (commit_message, parent_commit)
        self.heads[repo_type] = oid
        self.snapshots[oid] = deepcopy(self.data[repo_type])
        if self.uncertain_once:
            self.uncertain_once = False
            raise RuntimeError("Unknown transport reply; never printed")
        return SimpleNamespace(oid=oid)

    def list_repo_commits(self, repo, repo_type, revision, token):
        title, parent = self.history[revision]
        return [SimpleNamespace(commit_id=revision, title=title), SimpleNamespace(commit_id=parent, title="Previous")]

    def download(self, repo, name, kind, revision):
        return self.snapshots[revision][name][0]


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.kit = fixture(self.root/"fixture-kit")
        self.output = self.root/"plan"
        self.api = Hub()
        self.plan = pub.stage(self.kit, self.output, api=self.api, downloader=self.api.download)
        self.path = self.output/"plan.json"

    def tearDown(self):
        self.temporary.cleanup()

    def test_stage_is_anonymous_read_only_with_fixed_destinations_and_namespace(self):
        plan = pub.checked(self.path)
        self.assertEqual(pub.REPOS, {"dataset": "Dvidia/task-environments-v02", "space": "Dvidia/robot-simulation-lab"})
        self.assertEqual(pub.NAMESPACE, "task-environments/v02/")
        self.assertEqual([row["repo"] for row in plan["repositories"]], list(pub.REPOS.values()))
        self.assertEqual(self.api.calls, [])
        self.assertEqual(self.api.created, [])
        self.assertTrue(all(item["path"].startswith(pub.NAMESPACE) for item in plan["repositories"][1]["files"]))

    def test_publish_preserves_all_existing_objects_and_anonymous_readback_matches(self):
        baseline = deepcopy(self.api.data["space"])
        result = pub._publish(self.path, self.api, "test-only-credential", Operation)
        self.assertTrue(result["all_git_lfs_identities_verified"])
        self.assertEqual({name: self.api.data["space"][name] for name in baseline}, baseline)
        self.assertTrue(all(len(paths) <= 100 for _, paths in self.api.calls))
        self.assertTrue(all(path.startswith(pub.NAMESPACE) for kind, paths in self.api.calls if kind == "space" for path in paths))
        with patch.object(pub, "public_client", return_value=self.api), patch.object(pub, "download", side_effect=lambda repo, name, kind, revision, cache: self.api.download(repo, name, kind, revision)):
            proof = pub.verify(self.path)
        self.assertTrue(proof["anonymous_full_readback"])
        self.assertEqual(sum(len(row["files"]) for row in proof["repositories"]), sum(len(row["files"]) for row in self.plan["repositories"]))

    def test_unknown_commit_reply_reconciles_exact_child_without_duplicate_upload(self):
        self.api.uncertain_once = True
        with self.assertRaisesRegex(pub.PublicationError, "reply uncertain"):
            pub._publish(self.path, self.api, "test-only-credential", Operation)
        result = pub._publish(self.path, self.api, "test-only-credential", Operation)
        self.assertEqual(len(self.api.created), 1)
        self.assertEqual(sum(kind == "dataset" for kind, _ in self.api.calls), 1)
        self.assertEqual(len(result["repositories"]["dataset"]["completed"]), 1)

    def test_parent_movement_blocks_shared_space_mutation(self):
        self.api.heads["space"] = "3"*40
        self.api.snapshots["3"*40] = deepcopy(self.api.data["space"])
        with self.assertRaisesRegex(pub.PublicationError, "moved"):
            pub._publish(self.path, self.api, "test-only-credential", Operation)
        self.assertFalse(any(kind == "space" for kind, _ in self.api.calls))

    def test_checked_rejects_changed_bytes_destination_and_preexisting_namespace(self):
        original = self.path.read_bytes()
        for fault in ("bytes", "destination", "namespace"):
            self.path.write_bytes(original)
            plan = pub.load(self.path)
            if fault == "bytes":
                target = self.output/"dataset/guide.md"
                before = target.read_bytes()
                target.write_bytes(before+b"changed")
            elif fault == "destination":
                plan["repositories"][0]["repo"] = "Other/unauthorized"
                self.path.write_bytes(pub.encoded(plan))
            else:
                plan["repositories"][1]["baseline"]["tree"][pub.NAMESPACE+"existing.json"] = {"bytes": 2, "blob": "4"*40, "lfs_sha256": None}
                self.path.write_bytes(pub.encoded(plan))
            with self.subTest(fault=fault), self.assertRaises(pub.PublicationError):
                pub.checked(self.path)
            if fault == "bytes":
                target.write_bytes(before)

    def test_occupied_namespace_refuses_staging(self):
        self.api.data["space"][pub.NAMESPACE+"old.json"] = (b"{}", "git")
        self.api.snapshots[self.api.heads["space"]] = deepcopy(self.api.data["space"])
        with self.assertRaisesRegex(pub.PublicationError, "occupied"):
            pub.stage(self.kit, self.root/"other-plan", api=self.api, downloader=self.api.download)
        self.assertEqual(self.api.calls, [])

    def test_changed_archive_or_manifest_is_rejected_before_remote_reads(self):
        archive = self.kit.with_suffix(".zip")
        archive.write_bytes(archive.read_bytes()+b"changed")
        with patch.object(pub, "public_client", side_effect=AssertionError("Network before rejection")):
            with self.assertRaisesRegex(pub.PublicationError, "archive"):
                pub.stage(self.kit, self.root/"other-plan")
        seal(self.kit)
        (self.kit/"index.html").write_bytes(b"changed but not sealed")
        with patch.object(pub, "public_client", side_effect=AssertionError("Network before rejection")):
            with self.assertRaisesRegex(pub.PublicationError, "manifest"):
                pub.stage(self.kit, self.root/"other-plan")

    def test_resealed_unknown_or_missing_outcomes_are_rejected_before_remote_reads(self):
        original_audit = (self.kit/"audit.json").read_bytes()
        original_report = (self.kit/"benchmark/report.json").read_bytes()
        for fault in ("audit-unknown", "audit-mismatch", "missing-trace", "report-unknown", "precheck-unknown"):
            (self.kit/"audit.json").write_bytes(original_audit)
            (self.kit/"benchmark/report.json").write_bytes(original_report)
            audit = json.loads(original_audit)
            report = json.loads(original_report)
            if fault == "audit-unknown":
                audit["cases"][0]["goal_success"] = None
            elif fault == "audit-mismatch":
                audit["mismatches"] = 1
            elif fault == "missing-trace":
                audit["trace_samples"] -= 1
            elif fault == "report-unknown":
                report["results"][0]["native_goal"]["unknown"] = True
            else:
                report["prechecks"][0]["rejected"] = None
            (self.kit/"audit.json").write_bytes(pub.encoded(audit))
            (self.kit/"benchmark/report.json").write_bytes(pub.encoded(report))
            seal(self.kit)
            with self.subTest(fault=fault), patch.object(pub, "public_client", side_effect=AssertionError("Network before rejection")):
                with self.assertRaises(pub.PublicationError):
                    pub.stage(self.kit, self.root/"other-plan")


if __name__ == "__main__":
    unittest.main()

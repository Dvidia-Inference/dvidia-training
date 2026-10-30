"""No-network tests for the two-path SDK discovery overlay.

The fake Hub records exact Git/LFS object identities and parent-CAS histories.
No credentials are read and no public repositories are contacted.
"""
from copy import deepcopy
from hashlib import sha1, sha256
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import link_research_sdk_v01 as pub


PARENT = "1" * 40
README = b"---\ntitle: DVIDIA Robot Simulation Lab\nsdk: static\nlicense: mit\n---\n\n# Existing research\n\nKeep this old card byte for byte.\n"
INDEX = (b"<!doctype html><html><body><main><h1>Existing lab</h1>"
    b'<section aria-labelledby="workstation-title"><h2 id="workstation-title">Live workstation</h2>'
    b'<a href="robot-chat/index.html">Robot chat</a></section></main></body></html>\n')


class Operation:
    def __init__(self, path_in_repo, path_or_fileobj):
        self.path = path_in_repo
        self.raw = path_or_fileobj


class Hub:
    def __init__(self):
        self.head = PARENT
        self.data = {
            ".gitattributes": (b"*.gz filter=lfs diff=lfs merge=lfs -text\n", "git"),
            "README.md": (README, "git"),
            "index.html": (INDEX, "git"),
            "robot-chat/index.html": (b"<p>Existing unrelated robot chat</p>\n", "git"),
            "robot-chat/config.json": (b'{"model":"existing"}\n', "git"),
            "old-run/poses.json.gz": (b"Existing LFS trace bytes", "lfs"),
            "usd-workstation/v01/index.html": (b"<p>Sealed USD recorded viewer</p>\n", "git"),
        }
        self.snapshots = {self.head: deepcopy(self.data)}
        self.history = {}
        self.calls = []
        self.counter = 12
        self.lost_reply = None
        self.source_contract = None
        self.source_bytes = None
        self.source_reads = []

    def repo_info(self, repo, repo_type, revision=None, **kwargs):
        assert repo == "Dvidia/robot-simulation-lab" and repo_type == "space"
        oid = revision or self.head
        siblings = []
        for name, (raw, mode) in self.snapshots[oid].items():
            blob = sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            siblings.append(SimpleNamespace(rfilename=name, size=len(raw), blob_id=blob,
                lfs=SimpleNamespace(sha256=sha256(raw).hexdigest()) if mode == "lfs" else None))
        return SimpleNamespace(private=False, sha=oid, siblings=siblings, sdk="static")

    def whoami(self):
        return {"orgs": [{"name": "Dvidia"}]}

    def create_commit(self, repo, repo_type, parent_commit, operations, commit_message):
        assert repo == "Dvidia/robot-simulation-lab" and repo_type == "space"
        assert parent_commit == self.head
        self.calls.append({"parent": parent_commit, "paths": [op.path for op in operations], "title": commit_message})
        if self.lost_reply == "unaccepted":
            self.lost_reply = None
            raise RuntimeError("Unaccepted transport reply")
        for operation in operations:
            mode = "lfs" if operation._upload_mode == "lfs" else "git"
            self.data[operation.path] = (operation.raw, mode)
        self.counter += 1
        oid = f"{self.counter:040x}"
        self.history[oid] = (commit_message, parent_commit)
        self.head = oid
        self.snapshots[oid] = deepcopy(self.data)
        if self.lost_reply == "accepted":
            self.lost_reply = None
            raise RuntimeError("Accepted transport reply lost")
        return SimpleNamespace(oid=oid)

    def list_repo_commits(self, repo, repo_type, revision, **kwargs):
        title, parent = self.history[revision]
        return [SimpleNamespace(commit_id=revision, title=title), SimpleNamespace(commit_id=parent, title="Previous")]

    def download(self, repo, name, kind, revision):
        if kind == "dataset":
            self.source_reads.append((repo, name, kind, revision))
            for source, row in self.source_contract.items():
                if source != "sdk" and (repo, name, revision) == (row["repo"], row["path"], row["revision"]):
                    return self.source_bytes[source]
            raise AssertionError("Unexpected or unpinned source download")
        assert repo == "Dvidia/robot-simulation-lab" and kind == "space"
        return self.snapshots[revision][name][0]

    def move(self, *, title="Foreign update", changes=None):
        parent = self.head
        if changes:
            self.data.update(changes)
        self.counter += 1
        self.head = f"{self.counter:040x}"
        self.history[self.head] = (title, parent)
        self.snapshots[self.head] = deepcopy(self.data)


class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.output = Path(self.temporary.name) / "plan"
        self.path = self.output / pub.PLAN
        self.api = Hub()
        facts = {
            "usd": {"mismatches": 0, "counts": {"scheduled": 45, "valid": 45, "passed": 45,
                "normal_scheduled": 36, "normal_passed": 36, "negative_scheduled": 9, "negative_passed": 9}},
            "v04": {"schema": "dvidia.piper-grip-wait-compact-recovery/v1", "scheduled_attempts": 512,
                "completed_scored_attempts": 508, "interrupted_unevaluable_attempts": 4,
                "qualified_robot_skill": False, "physical_robot_ready": False,
                "variants": {name: {"scheduled": 128, "success": success, "interrupted_unevaluable": 1}
                    for name, success in {"guarded": 42, "clock": 32, "fixed": 44, "teacher": 41}.items()},
                "contrasts": [{"comparator": name, "interval": [-0.05, 0.05]} for name in ("clock", "fixed")]},
        }
        self.sources = {name: json.dumps(value).encode() for name, value in facts.items()}
        self.contract = pub.source_contract()
        for name, raw in self.sources.items():
            self.contract[name]["sha256"] = sha256(raw).hexdigest()
        self.api.source_contract = self.contract
        self.api.source_bytes = self.sources
        self.source_patch = patch.object(pub, "source_contract", return_value=self.contract)
        self.source_patch.start()
        self.auth_patch = patch.object(pub, "client", side_effect=AssertionError("Credential read forbidden in injected no-network tests"))
        self.auth_patch.start()

    def tearDown(self):
        self.source_patch.stop()
        self.auth_patch.stop()
        self.temporary.cleanup()

    def stage(self):
        return pub.stage(PARENT, self.output, api=self.api, downloader=self.api.download)

    def publish(self, **kwargs):
        return pub.publish(self.path, api=self.api, downloader=self.api.download, operation_cls=Operation, **kwargs)

    def payload_directory(self):
        matches = [path.parent for path in self.output.rglob("README.md") if pub.MARKER in path.read_bytes()]
        self.assertEqual(len(matches), 1)
        return matches[0]

    def test_staging_is_read_only_and_preserves_metadata_and_old_cards(self):
        self.stage()
        plan, baseline, payload = pub.checked(self.path)
        self.assertEqual(set(payload), {"README.md", "index.html"})
        self.assertTrue(payload["README.md"].startswith(README))
        self.assertEqual(payload["index.html"].split(pub.MARKER)[0], INDEX.split(pub.ANCHOR)[0])
        self.assertTrue(payload["index.html"].endswith(INDEX[INDEX.index(pub.ANCHOR):]))
        self.assertIn(b"robot-chat/index.html", payload["index.html"])
        self.assertEqual(self.api.calls, [])
        self.assertEqual(len(self.api.source_reads), 2)

    def test_marker_duplicates_or_ambiguous_main_are_refused(self):
        for before in (
            {"README.md": README + pub.MARKER, "index.html": INDEX},
            {"README.md": README, "index.html": INDEX.replace(b"</main>", pub.MARKER + b"</main>")},
            {"README.md": README, "index.html": INDEX.replace(b"</main>", b"</main></main>")},
            {"README.md": README, "index.html": INDEX.replace(b"</main>", b"")},
            {"README.md": README, "index.html": INDEX.replace(pub.ANCHOR, pub.ANCHOR + pub.ANCHOR)},
            {"README.md": README, "index.html": INDEX.replace(pub.ANCHOR, b"")},
        ):
            with self.subTest(before=before):
                with self.assertRaises(pub.PublicationError):
                    pub.derive(before)

    def test_publish_changes_only_two_root_paths_and_preserves_all_git_lfs_objects(self):
        baseline = deepcopy(self.api.data)
        self.stage()
        self.publish()
        self.assertEqual(len(self.api.calls), 1)
        self.assertEqual(set(self.api.calls[0]["paths"]), {"README.md", "index.html"})
        for name in set(baseline) - {"README.md", "index.html"}:
            self.assertEqual(self.api.data[name], baseline[name])
        result = pub.verify(self.path, api=self.api, downloader=self.api.download)
        self.assertTrue(result["anonymous_readback"])

    def test_stale_parent_refuses_before_any_write(self):
        self.stage()
        self.api.move()
        with self.assertRaises(pub.PublicationError):
            self.publish()
        self.assertEqual(self.api.calls, [])

    def test_staged_bytes_tampering_is_refused_before_any_write(self):
        self.stage()
        path = self.payload_directory() / "README.md"
        path.write_bytes(path.read_bytes() + b"\nUnreviewed addition.\n")
        with self.assertRaises(pub.PublicationError):
            pub.checked(self.path)
        with self.assertRaises(pub.PublicationError):
            self.publish()
        self.assertEqual(self.api.calls, [])

    def test_injected_third_payload_path_is_refused(self):
        self.stage()
        (self.payload_directory() / "third.json").write_bytes(b"{}\n")
        with self.assertRaises(pub.PublicationError):
            pub.checked(self.path)
        with self.assertRaises(pub.PublicationError):
            self.publish()
        self.assertEqual(self.api.calls, [])

    def test_changed_source_pin_or_source_bytes_are_refused(self):
        self.stage()
        original = self.contract["sdk"]["source_commit"]
        self.contract["sdk"]["source_commit"] = "9" * 40
        with self.assertRaises(pub.PublicationError):
            pub.checked(self.path)
        self.contract["sdk"]["source_commit"] = original
        source = self.output / "sources" / "usd.json"
        source.write_bytes(source.read_bytes() + b" ")
        with self.assertRaises(pub.PublicationError):
            pub.checked(self.path)

    def test_source_hash_alone_cannot_qualify_incorrect_evidence_facts(self):
        changed = json.loads(self.sources["usd"])
        changed["counts"]["passed"] = 44
        self.sources["usd"] = json.dumps(changed).encode()
        self.contract["usd"]["sha256"] = sha256(self.sources["usd"]).hexdigest()
        with self.assertRaises(pub.PublicationError):
            self.stage()
        self.assertEqual(self.api.calls, [])

    def test_accepted_lost_reply_reconciles_without_second_commit(self):
        self.stage()
        self.api.lost_reply = "accepted"
        with self.assertRaises(pub.PublicationError):
            self.publish()
        self.assertTrue((self.output / pub.JOURNAL).is_file())
        self.publish()
        self.assertEqual(len(self.api.calls), 1)

    def test_unaccepted_lost_reply_requires_manual_resume(self):
        self.stage()
        self.api.lost_reply = "unaccepted"
        with self.assertRaises(pub.PublicationError):
            self.publish()
        pending = (self.output / pub.JOURNAL).read_bytes()
        with self.assertRaises(pub.PublicationError):
            self.publish()
        self.assertEqual(len(self.api.calls), 1)
        self.assertEqual((self.output / pub.JOURNAL).read_bytes(), pending)
        result = self.publish(resume_unchanged_parent=True)
        self.assertTrue(result["anonymous_readback"])
        self.assertEqual(len(self.api.calls), 2)
        self.assertEqual(len(self.api.snapshots), 2)
        self.assertEqual((self.output / pub.JOURNAL).read_bytes(), pending)

    def test_foreign_immediate_child_cannot_reconcile_pending_intent(self):
        self.stage()
        self.api.lost_reply = "unaccepted"
        with self.assertRaises(pub.PublicationError):
            self.publish()
        self.api.move()
        with self.assertRaises(pub.PublicationError):
            self.publish()
        self.assertEqual(len(self.api.calls), 1)

    def test_matching_pending_title_with_unrelated_tree_cannot_reconcile(self):
        self.stage()
        self.api.lost_reply = "accepted"
        with self.assertRaises(pub.PublicationError):
            self.publish()
        self.api.data["robot-chat/config.json"] = (b'{"model":"foreign"}\n', "git")
        self.api.snapshots[self.api.head] = deepcopy(self.api.data)
        with self.assertRaises(pub.PublicationError):
            self.publish()
        self.assertEqual(len(self.api.calls), 1)

    def test_anonymous_download_must_match_staged_bytes(self):
        self.stage()
        self.publish()

        def corrupt(repo, name, kind, revision):
            raw = self.api.download(repo, name, kind, revision)
            return raw + b"Changed after download." if name == "README.md" else raw

        with self.assertRaises(pub.PublicationError):
            pub.verify(self.path, api=self.api, downloader=corrupt)

    def test_failed_postwrite_readback_retains_intent_without_success_receipt(self):
        self.stage()

        def corrupt(repo, name, kind, revision):
            raw = self.api.download(repo, name, kind, revision)
            return raw + b"Corrupt response." if name == "index.html" else raw

        with self.assertRaises(pub.PublicationError):
            pub.publish(self.path, api=self.api, downloader=corrupt, operation_cls=Operation)
        self.assertEqual(len(self.api.calls), 1)
        self.assertTrue((self.output / pub.JOURNAL).is_file())
        self.assertFalse((self.output / pub.RECEIPT).exists())
        receipt = self.publish()
        self.assertTrue(receipt["reconciled"])
        self.assertTrue(receipt["anonymous_readback"])
        self.assertEqual(len(self.api.calls), 1)


if __name__ == "__main__":
    unittest.main()

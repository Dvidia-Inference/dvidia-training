"""No-network publication tests using tiny fixtures and an exact Git/LFS Hub."""
from copy import deepcopy
from contextlib import redirect_stdout
from hashlib import sha1, sha256
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from huggingface_hub.errors import RepositoryNotFoundError

import publish_piper_action_capture_v01 as pub


class Operation:
    def __init__(self, path_in_repo, path_or_fileobj):
        self.path = path_in_repo
        self.raw = path_or_fileobj


class Hub:
    def __init__(self):
        self.counter = 12
        self.heads = {"space": "1" * 40}
        self.data = {"space": {
            ".gitattributes": (b"*.gz filter=lfs diff=lfs merge=lfs -text\n", "git"),
            "README.md": (b"Existing public research card.\n", "git"),
            "index.html": (b"<main>Existing simulation entry.</main>\n", "git"),
            "robot-chat/index.html": (b"<p>Unrelated robot chat.</p>\n", "git"),
            "robot-chat/config.json": (b'{"status":"retained"}\n', "git"),
            "prior-run/evidence.gz": (b"Retained prior native evidence.", "lfs"),
        }}
        self.snapshots = {self.heads["space"]: deepcopy(self.data["space"])}
        self.history = {}
        self.calls = []
        self.created = []
        self.uncertain_create_once = False
        self.uncertain_commit_kind = None
        self.unaccepted_commit_kind = None

    def repo_info(self, repo, repo_type, revision=None, **kwargs):
        assert repo == pub.REPOS[repo_type]
        if repo_type not in self.heads:
            raise RepositoryNotFoundError("Not found", response=SimpleNamespace(status_code=404, headers={}, request=None))
        oid = revision or self.heads[repo_type]
        siblings = []
        for name, (raw, mode) in self.snapshots[oid].items():
            blob = sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            siblings.append(SimpleNamespace(rfilename=name, size=len(raw), blob_id=blob,
                lfs=SimpleNamespace(sha256=sha256(raw).hexdigest()) if mode == "lfs" else None))
        return SimpleNamespace(private=False, sha=oid, siblings=siblings, sdk="static")

    def whoami(self):
        return {"orgs": [{"name": "Dvidia"}]}

    def empty_dataset(self):
        self.heads["dataset"] = "2" * 40
        self.data["dataset"] = {".gitattributes": (b"Default empty repository attributes.\n", "git")}
        self.snapshots[self.heads["dataset"]] = deepcopy(self.data["dataset"])

    def create_repo(self, repo, repo_type, private, exist_ok):
        assert repo == pub.REPOS["dataset"] and repo_type == "dataset"
        assert private is False and exist_ok is False and "dataset" not in self.heads
        self.created.append(repo)
        self.empty_dataset()
        if self.uncertain_create_once:
            self.uncertain_create_once = False
            raise RuntimeError("Repository accepted; response lost")

    def create_commit(self, repo, repo_type, parent_commit, operations, commit_message):
        assert repo == pub.REPOS[repo_type] and parent_commit == self.heads[repo_type]
        self.calls.append({"kind": repo_type, "parent": parent_commit,
            "paths": [operation.path for operation in operations], "title": commit_message})
        if self.unaccepted_commit_kind == repo_type:
            self.unaccepted_commit_kind = None
            raise RuntimeError("Commit not accepted; response lost")
        for operation in operations:
            mode = "lfs" if operation._upload_mode == "lfs" else "git"
            self.data[repo_type][operation.path] = (operation.raw, mode)
        self.counter += 1
        oid = f"{self.counter:040x}"
        self.history[oid] = (commit_message, parent_commit)
        self.heads[repo_type] = oid
        self.snapshots[oid] = deepcopy(self.data[repo_type])
        if self.uncertain_commit_kind == repo_type:
            self.uncertain_commit_kind = None
            raise RuntimeError("Commit accepted; response lost")
        return SimpleNamespace(oid=oid)

    def list_repo_commits(self, repo, repo_type, revision, **kwargs):
        title, parent = self.history[revision]
        return [SimpleNamespace(commit_id=revision, title=title), SimpleNamespace(commit_id=parent, title="Previous")]

    def download(self, repo, name, kind, revision):
        assert repo == pub.REPOS[kind]
        return self.snapshots[revision][name][0]

    def move(self, kind, *, title="Foreign update", changes=None):
        parent = self.heads[kind]
        if changes:
            self.data[kind].update(changes)
        self.counter += 1
        oid = f"{self.counter:040x}"
        self.heads[kind] = oid
        self.history[oid] = (title, parent)
        self.snapshots[oid] = deepcopy(self.data[kind])


class CapturePublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "plan"
        self.path = self.output / "plan.json"
        self.api = Hub()
        kit_manifest = b'{"scope":"fixture complete-kit manifest"}\n'
        web_manifest = b'{"scope":"fixture compact display manifest"}\n'
        archive = b"Original archive fixture bytes."
        wheel = b"Original source-preview wheel fixture bytes."
        self.dataset = {
            ".gitattributes": b"*.zip filter=lfs diff=lfs merge=lfs -text\n*.whl filter=lfs diff=lfs merge=lfs -text\n",
            "README.md": b"DVIDIA capture engineering fixture; no trained skill.\n",
            "piper-action-capture-v01.zip": archive,
            "dvidia_training-0.1.3.dev1-py3-none-any.whl": wheel,
            "kit-manifest.json": kit_manifest,
            "web-manifest.json": web_manifest,
            "summary.json": b'{"trained_skill":false,"physical_robot_ready":false}\n',
        }
        self.space = {
            pub.NAMESPACE + "index.html": b"<main>DVIDIA recorded capture fixture.</main>\n",
            pub.NAMESPACE + "viewer-data.json": b'{"physical_robot_ready":false}\n',
            pub.NAMESPACE + "manifest.json": web_manifest,
        }
        self.evidence = {"kit_manifest": pub.digest(kit_manifest), "kit_archive": pub.digest(archive),
            "web_manifest": pub.digest(web_manifest), "preview_wheel": pub.digest(wheel)}
        self.inputs_patch = patch.object(pub, "_inputs", return_value=(self.dataset, self.space, self.evidence))
        self.inputs_patch.start()
        self.auth_patch = patch.object(pub, "client", side_effect=AssertionError("No credential read permitted in no-network tests"))
        self.auth_patch.start()

    def tearDown(self):
        self.inputs_patch.stop()
        self.auth_patch.stop()
        self.temporary.cleanup()

    def stage(self):
        return pub.stage(self.root / "kit", self.root / "web", self.output, api=self.api, downloader=self.api.download)

    def publish(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return pub._publish(self.path, self.api, "unit-test-publishing-credential", Operation, **kwargs)

    def test_stage_is_read_only_and_binds_fixed_destinations_and_namespace(self):
        self.stage()
        plan = pub.checked(self.path)
        self.assertEqual([row["repo"] for row in plan["repositories"]], list(pub.REPOS.values()))
        self.assertTrue(all(row["path"].startswith(pub.NAMESPACE) for row in plan["repositories"][1]["files"]))
        self.assertEqual(self.api.created, [])
        self.assertEqual(self.api.calls, [])
        self.assertFalse(plan["qualified_robot_skill"])
        self.assertFalse(plan["physical_robot_ready"])

    def test_publish_preserves_entire_prior_space_and_verifies_git_lfs_bytes(self):
        baseline = deepcopy(self.api.data["space"])
        self.stage()
        receipt = self.publish()
        self.assertTrue(receipt["all_git_lfs_identities_verified"])
        self.assertFalse(receipt["anonymous_full_readback"])
        self.assertEqual({name: self.api.data["space"][name] for name in baseline}, baseline)
        self.assertEqual([call["kind"] for call in self.api.calls], ["dataset", "space"])
        self.assertTrue(all(path.startswith(pub.NAMESPACE) for call in self.api.calls if call["kind"] == "space" for path in call["paths"]))
        for name in ("piper-action-capture-v01.zip", "dvidia_training-0.1.3.dev1-py3-none-any.whl"):
            raw, mode = self.api.data["dataset"][name]
            self.assertEqual(raw, self.dataset[name])
            self.assertEqual(mode, "lfs")
        proof = pub.verify(self.path, api=self.api, downloader=self.api.download)
        self.assertTrue(proof["anonymous_full_readback"])
        self.assertEqual(sum(len(row["files"]) for row in proof["repositories"]), len(self.dataset) + len(self.space))

    def test_occupied_capture_namespace_refuses_before_any_remote_write(self):
        self.api.data["space"][pub.NAMESPACE + "old.json"] = (b"{}\n", "git")
        self.api.snapshots[self.api.heads["space"]] = deepcopy(self.api.data["space"])
        with self.assertRaises(pub.PublicationError):
            self.stage()
        self.assertEqual(self.api.created, [])
        self.assertEqual(self.api.calls, [])

    def test_verified_empty_public_dataset_initial_commit_is_allowed(self):
        self.api.empty_dataset()
        self.stage()
        self.publish()
        self.assertEqual(self.api.created, [])
        self.assertEqual(self.api.calls[0]["parent"], "2" * 40)

    def test_existing_nonempty_dataset_cannot_be_repurposed(self):
        self.api.empty_dataset()
        self.api.data["dataset"]["unrelated.json"] = (b"{}\n", "git")
        self.api.snapshots[self.api.heads["dataset"]] = deepcopy(self.api.data["dataset"])
        with self.assertRaises(pub.PublicationError):
            self.stage()
        self.assertEqual(self.api.calls, [])

    def test_unknown_accepted_repository_creation_never_creates_twice(self):
        self.stage()
        self.api.uncertain_create_once = True
        with self.assertRaises((pub.PublicationError, RuntimeError)):
            self.publish()
        self.publish()
        self.assertEqual(len(self.api.created), 1)
        self.assertEqual([call["kind"] for call in self.api.calls], ["dataset", "space"])

    def test_unknown_accepted_commit_reply_reconciles_without_duplicate_commit(self):
        self.stage()
        self.api.uncertain_commit_kind = "space"
        with self.assertRaises(pub.PublicationError):
            self.publish()
        self.publish()
        self.assertEqual([call["kind"] for call in self.api.calls], ["dataset", "space"])
        self.assertEqual(len(self.api.created), 1)

    def test_unchanged_parent_pending_intent_requires_explicit_guarded_resume(self):
        self.stage()
        self.api.unaccepted_commit_kind = "space"
        with self.assertRaises(pub.PublicationError):
            self.publish()
        pending = (self.output / "journal.json").read_bytes()
        with self.assertRaises(pub.PublicationError):
            self.publish()
        self.assertEqual([call["kind"] for call in self.api.calls], ["dataset", "space"])
        self.assertEqual((self.output / "journal.json").read_bytes(), pending)
        self.publish(resume_unchanged_parent=True)
        self.assertEqual([call["kind"] for call in self.api.calls], ["dataset", "space", "space"])

    def test_stale_space_parent_cannot_write_into_the_changed_space(self):
        self.stage()
        self.api.move("space", changes={"robot-chat/config.json": (b'{"status":"changed"}\n', "git")})
        with self.assertRaises(pub.PublicationError):
            self.publish()
        self.assertFalse(any(call["kind"] == "space" for call in self.api.calls))

    def test_staged_payload_tamper_or_injected_third_scope_is_refused(self):
        self.stage()
        path = self.output / "dataset" / "README.md"
        raw = path.read_bytes()
        path.write_bytes(raw + b"Unreviewed content.\n")
        with self.assertRaises(pub.PublicationError):
            pub.checked(self.path)
        path.write_bytes(raw)
        (self.output / "space" / "index.html").write_bytes(b"Root overwrite attempt.\n")
        with self.assertRaises(pub.PublicationError):
            pub.checked(self.path)
        self.assertEqual(self.api.calls, [])

    def test_changed_plan_destination_cannot_redirect_publication(self):
        self.stage()
        plan = pub.load(self.path)
        plan["repositories"][0]["repo"] = "Dvidia/unrelated-dataset"
        self.path.write_bytes(pub.encoded(plan))
        with self.assertRaises(pub.PublicationError):
            self.publish()
        self.assertEqual(self.api.calls, [])
        self.assertEqual(self.api.created, [])

    def test_foreign_child_cannot_resolve_an_unknown_pending_write(self):
        self.stage()
        self.api.unaccepted_commit_kind = "space"
        with self.assertRaises(pub.PublicationError):
            self.publish()
        self.api.move("space")
        with self.assertRaises(pub.PublicationError):
            self.publish()
        self.assertEqual([call["kind"] for call in self.api.calls], ["dataset", "space"])

    def test_intended_child_title_cannot_hide_changes_to_existing_space_objects(self):
        self.stage()
        self.api.uncertain_commit_kind = "space"
        with self.assertRaises(pub.PublicationError):
            self.publish()
        self.api.data["space"]["robot-chat/config.json"] = (b'{"status":"unexpected"}\n', "git")
        self.api.snapshots[self.api.heads["space"]] = deepcopy(self.api.data["space"])
        with self.assertRaises(pub.PublicationError):
            self.publish()
        self.assertEqual([call["kind"] for call in self.api.calls], ["dataset", "space"])

    def test_changed_lfs_identity_refuses_anonymous_verification(self):
        self.stage()
        self.publish()
        name = "piper-action-capture-v01.zip"
        raw, mode = self.api.data["dataset"][name]
        self.api.data["dataset"][name] = (b"x" * len(raw), mode)
        self.api.snapshots[self.api.heads["dataset"]] = deepcopy(self.api.data["dataset"])
        with self.assertRaises(pub.PublicationError):
            pub.verify(self.path, api=self.api, downloader=self.api.download)
        self.assertFalse((self.output / "anonymous-readback.json").exists())

    def test_failed_anonymous_readback_never_claims_full_verification(self):
        self.stage()
        self.publish()

        def corrupt(repo, name, kind, revision):
            raw = self.api.download(repo, name, kind, revision)
            return raw + b"Corrupt download." if kind == "space" and name.endswith("index.html") else raw

        with self.assertRaises(pub.PublicationError):
            pub.verify(self.path, api=self.api, downloader=corrupt)
        self.assertFalse((self.output / "anonymous-readback.json").exists())
        self.assertFalse(pub.load(self.output / "publication-receipt.json")["anonymous_full_readback"])


if __name__ == "__main__":
    unittest.main()

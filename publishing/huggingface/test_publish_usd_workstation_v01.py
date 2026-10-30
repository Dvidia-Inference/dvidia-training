"""No-network publication tests with an in-memory parent-CAS Hub."""
from copy import deepcopy
from hashlib import sha1, sha256
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from huggingface_hub.errors import RepositoryNotFoundError

import publish_usd_workstation_v01 as pub


ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT/"runs/usd-workstation-v01-public02-20261030"


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


@unittest.skipUnless(KIT.is_dir(), "Requires the sealed public v0.1 kit; tests perform no network or credential reads")
class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.output = Path(self.temporary.name)/"plan"
        self.api = Hub()
        self.plan = pub.stage(KIT, self.output, api=self.api, downloader=self.api.download)
        self.path = self.output/"plan.json"

    def tearDown(self):
        self.temporary.cleanup()

    def test_staging_uses_exact_kit_without_remote_mutation(self):
        checked = pub.checked(self.path)
        self.assertEqual([row["repo"] for row in checked["repositories"]], list(pub.REPOS.values()))
        self.assertEqual(self.api.calls, [])
        self.assertEqual(self.api.created, [])
        space = checked["repositories"][1]
        self.assertEqual(len(space["files"]), 345)
        self.assertTrue(all(item["path"].startswith(pub.NAMESPACE) for item in space["files"]))

    def test_publish_preserves_every_existing_space_path_and_bounds_each_commit(self):
        baseline = deepcopy(self.api.data["space"])
        result = pub._publish(self.path, self.api, "unit-test-credential", Operation)
        self.assertTrue(result["all_git_lfs_identities_verified"])
        self.assertEqual({name: self.api.data["space"][name] for name in baseline}, baseline)
        self.assertEqual([len(paths) for kind, paths in self.api.calls if kind == "space"], [100, 100, 100, 45])
        self.assertTrue(all(name.startswith(pub.NAMESPACE) for kind, paths in self.api.calls if kind == "space" for name in paths))
        with patch.object(pub, "public_client", return_value=self.api), patch.object(pub, "download", side_effect=lambda repo, name, kind, revision, cache: self.api.download(repo, name, kind, revision)):
            proof = pub.verify(self.path)
        self.assertTrue(proof["anonymous_full_readback"])
        self.assertEqual(sum(len(row["files"]) for row in proof["repositories"]), 354)

    def test_unknown_commit_reply_reconciles_exact_child_without_duplicate_upload(self):
        self.api.uncertain_once = True
        with self.assertRaisesRegex(pub.PublicationError, "reply uncertain"):
            pub._publish(self.path, self.api, "unit-test-credential", Operation)
        result = pub._publish(self.path, self.api, "unit-test-credential", Operation)
        self.assertEqual(len(self.api.created), 1)
        self.assertEqual(sum(kind == "dataset" for kind, _ in self.api.calls), 1)
        self.assertEqual(len(result["repositories"]["dataset"]["completed"]), 1)

    def test_parent_movement_cannot_change_shared_space(self):
        self.api.heads["space"] = "3"*40
        self.api.snapshots["3"*40] = deepcopy(self.api.data["space"])
        with self.assertRaisesRegex(pub.PublicationError, "moved"):
            pub._publish(self.path, self.api, "unit-test-credential", Operation)
        self.assertFalse(any(kind == "space" for kind, _ in self.api.calls))

    def test_destination_or_staged_byte_tampering_is_rejected(self):
        path = self.output/"dataset/guide.md"
        path.write_bytes(path.read_bytes()+b"\nchanged\n")
        with self.assertRaises(pub.PublicationError):
            pub.checked(self.path)

    def test_checked_plan_rejects_overwriting_a_preexisting_namespace(self):
        changed = pub.load(self.path)
        changed["repositories"][1]["baseline"]["tree"][pub.NAMESPACE+"old.json"] = {"bytes": 2, "blob": "4"*40, "lfs_sha256": None}
        self.path.write_bytes(pub.encoded(changed))
        with self.assertRaisesRegex(pub.PublicationError, "absent"):
            pub.checked(self.path)

    def test_existing_namespace_prevents_staging_and_remote_writes(self):
        self.api.data["space"][pub.NAMESPACE+"old.json"] = (b"{}", "git")
        self.api.snapshots[self.api.heads["space"]] = deepcopy(self.api.data["space"])
        with self.assertRaisesRegex(pub.PublicationError, "occupied"):
            pub.stage(KIT, Path(self.temporary.name)/"other-plan", api=self.api, downloader=self.api.download)
        self.assertEqual(self.api.calls, [])


if __name__ == "__main__":
    unittest.main()

"""Small, no-network tests for the explicit Space-only capture recovery."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import recover_piper_action_capture_v01 as pub
from test_publish_piper_action_capture_v01 import Hub, Operation


class RecoveryHub(Hub):
    def create_repo(self, *args, **kwargs):
        raise AssertionError("Recovery must never create a repository")

    def create_commit(self, repo, repo_type, **kwargs):
        if repo_type != "space":
            raise AssertionError("Recovery must never commit to the accepted dataset")
        return super().create_commit(repo, repo_type, **kwargs)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.previous = self.root / "original" / "plan.json"
        self.previous.parent.mkdir()
        self.output = self.root / "recovery"
        self.path = self.output / "plan.json"
        self.api = RecoveryHub()
        self.api.heads["space"] = pub.SPACE_PARENT
        self.api.snapshots[pub.SPACE_PARENT] = deepcopy(self.api.data["space"])
        space_baseline = pub.old.tree(self.api, "space")
        self.api.empty_dataset()
        dataset_baseline = pub.old.tree(self.api, "dataset")
        mp4 = b"Identical original full/repeat display fixture bytes."
        self.mp4_patch = patch.object(pub, "MP4", pub.old.digest(mp4))
        self.mp4_patch.start()
        dataset = {".gitattributes": b"*.zip filter=lfs diff=lfs merge=lfs -text\n",
            "README.md": b"Previously accepted DVIDIA capture fixture.\n",
            "evidence.zip": b"Previously accepted archive bytes.", "preview.whl": b"Previously accepted wheel bytes."}
        dataset.update({f"proof/{index:02}.json": f'{{"proof":{index}}}\n'.encode() for index in range(19)})
        space = {name: mp4 for name in pub.CHANGED}
        space.update({pub.old.NAMESPACE + "media/interrupted-prefix.mp4": b"Short interrupted display.",
            pub.old.NAMESPACE + "index.html": b"<main>Original recorded display.</main>\n",
            pub.old.NAMESPACE + "viewer-data.json": b'{"physical_robot_ready":false}\n',
            pub.old.NAMESPACE + "manifest.json": b'{"scope":"fixture"}\n'})
        space.update({pub.old.NAMESPACE + f"proof/{index:02}.json": f'{{"boundary":{index}}}\n'.encode() for index in range(21)})
        for kind, payload in (("dataset", dataset), ("space", space)):
            for name, raw in payload.items():
                target = self.previous.parent / kind / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
        rows = [{"kind": kind, "repo": pub.old.REPOS[kind], "root": str(self.previous.parent / kind),
            "files": pub.old.files(self.previous.parent / kind), "baseline": None if kind == "dataset" else space_baseline}
            for kind in ("dataset", "space")]
        self.original = {"schema": pub.old.SCHEMA, "public": True, "namespace": pub.old.NAMESPACE,
            "repositories": rows, "implementation": pub.old.implementation(), "qualified_robot_skill": False,
            "physical_robot_ready": False}
        self.previous.write_bytes(pub.old.encoded(self.original))
        self.original_journal = {"schema": pub.old.SCHEMA, "plan": pub.old.digest(self.previous.read_bytes()),
            "repositories": {
                "dataset": {"parent": dataset_baseline["commit"], "baseline": dataset_baseline["tree"],
                    "completed": [pub.DATASET], "pending": None},
                "space": {"parent": pub.SPACE_PARENT, "baseline": space_baseline["tree"], "completed": [],
                    "pending": {"index": 0, "parent": pub.SPACE_PARENT, "title": "Original retained uncertain intent"}},
            }}
        (self.previous.parent / "journal.json").write_bytes(pub.old.encoded(self.original_journal))
        for name in pub.EVIDENCE:
            if name not in ("plan.json", "journal.json"):
                (self.previous.parent / name).write_bytes(b'{"fixture":"prior evidence retained"}\n')
        self.evidence = {name: pub.old.digest((self.previous.parent / name).read_bytes()) for name in pub.EVIDENCE}
        self.original_bytes = self.snapshot_previous()
        self.api.data["dataset"] = {item["path"]: (dataset[item["path"]], item["representation"])
            for item in rows[0]["files"]}
        self.api.heads["dataset"] = pub.DATASET
        self.api.snapshots[pub.DATASET] = deepcopy(self.api.data["dataset"])

        def source_fixture(previous):
            if Path(previous).resolve() != self.previous.resolve():
                raise pub.old.PublicationError("Unexpected original fixture plan")
            return self.original, self.original_journal, self.evidence

        self.source_patch = patch.object(pub, "source", side_effect=source_fixture)
        self.source_patch.start()
        self.auth_patch = patch.object(pub.old, "client", side_effect=AssertionError("No credential reads in recovery tests"))
        self.auth_patch.start()

    def tearDown(self):
        self.auth_patch.stop()
        self.source_patch.stop()
        self.mp4_patch.stop()
        self.temporary.cleanup()

    def snapshot_previous(self):
        return {path.relative_to(self.previous.parent).as_posix(): path.read_bytes()
            for path in self.previous.parent.rglob("*") if path.is_file()}

    def stage(self):
        return pub.stage(self.previous, self.output, api=self.api)

    def publish(self):
        with redirect_stdout(io.StringIO()):
            return pub._publish(self.path, self.api, "unit-test-recovery-credential", Operation)

    def test_stage_changes_only_two_representations_and_retains_original_evidence(self):
        self.stage()
        plan = pub.checked(self.path)
        changed = [item["path"] for item in plan["repositories"][1]["files"] if item["representation"] == "lfs"]
        self.assertEqual(changed, list(pub.CHANGED))
        self.assertEqual(len(plan["repositories"][0]["files"]), 23)
        self.assertEqual(len(plan["repositories"][1]["files"]), 27)
        for row, previous in zip(plan["repositories"], self.original["repositories"]):
            for item in row["files"]:
                self.assertEqual((Path(row["root"]) / item["path"]).read_bytes(),
                    (Path(previous["root"]) / item["path"]).read_bytes())
        self.assertEqual(self.snapshot_previous(), self.original_bytes)
        self.assertEqual(self.api.calls, [])
        self.assertEqual(self.api.created, [])

    def test_publish_is_one_space_commit_and_accepted_dataset_is_read_only(self):
        baseline = deepcopy(self.api.data["space"])
        dataset = deepcopy(self.api.data["dataset"])
        self.stage()
        receipt = self.publish()
        self.assertTrue(receipt["dataset_completion_reused"])
        self.assertEqual([call["kind"] for call in self.api.calls], ["space"])
        self.assertEqual(self.api.calls[0]["parent"], pub.SPACE_PARENT)
        self.assertEqual(self.api.data["dataset"], dataset)
        self.assertEqual({name: self.api.data["space"][name] for name in baseline}, baseline)
        for name in pub.CHANGED:
            self.assertEqual(self.api.data["space"][name][1], "lfs")
        self.assertEqual(self.api.data["space"][pub.old.NAMESPACE + "media/interrupted-prefix.mp4"][1], "git")
        self.assertEqual(self.snapshot_previous(), self.original_bytes)

    def test_already_completed_publish_cannot_duplicate_the_space_commit(self):
        self.stage()
        self.publish()
        self.publish()
        self.assertEqual(len(self.api.calls), 1)

    def test_moved_dataset_or_space_refuses_recovery_staging(self):
        for kind in ("dataset", "space"):
            original = self.api.heads[kind]
            self.api.move(kind)
            with self.subTest(kind=kind), self.assertRaises(pub.old.PublicationError):
                self.stage()
            self.api.heads[kind] = original
        self.assertEqual(self.api.calls, [])

    def test_moved_space_after_stage_refuses_before_any_write(self):
        self.stage()
        self.api.move("space")
        with self.assertRaises(pub.old.PublicationError):
            self.publish()
        self.assertEqual(self.api.calls, [])

    def test_staged_payload_byte_tamper_refuses(self):
        self.stage()
        path = self.output / "space" / pub.CHANGED[0]
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaises(pub.old.PublicationError):
            self.publish()
        self.assertEqual(self.api.calls, [])

    def test_third_representation_change_or_destination_change_refuses(self):
        self.stage()
        original = self.path.read_bytes()
        for change in ("third-representation", "dataset-write", "destination"):
            plan = pub.old.load(self.path)
            if change == "third-representation":
                row = next(item for item in plan["repositories"][1]["files"] if item["path"].endswith("interrupted-prefix.mp4"))
                row["representation"] = "lfs"
            elif change == "dataset-write":
                plan["dataset_completion"]["new_dataset_writes"] = True
            else:
                plan["repositories"][1]["repo"] = "Dvidia/another-space"
            self.path.write_bytes(pub.old.encoded(plan))
            with self.subTest(change=change), self.assertRaises(pub.old.PublicationError):
                pub.checked(self.path)
            self.path.write_bytes(original)

    def test_changed_prior_source_evidence_digest_refuses(self):
        self.stage()
        self.evidence["journal.json"] = {"bytes": 1, "sha256": "0" * 64}
        with self.assertRaises(pub.old.PublicationError):
            self.publish()
        self.assertEqual(self.api.calls, [])

    def test_accepted_unknown_reply_reconciles_without_duplicate_commit(self):
        self.stage()
        self.api.uncertain_commit_kind = "space"
        with self.assertRaises(pub.old.PublicationError):
            self.publish()
        self.publish()
        self.assertEqual(len(self.api.calls), 1)
        self.assertEqual(self.snapshot_previous(), self.original_bytes)

    def test_unchanged_parent_after_unknown_reply_never_retries_automatically(self):
        self.stage()
        self.api.unaccepted_commit_kind = "space"
        with self.assertRaises(pub.old.PublicationError):
            self.publish()
        for _ in range(2):
            with self.assertRaises(pub.old.PublicationError):
                self.publish()
        self.assertEqual(len(self.api.calls), 1)
        self.assertFalse((self.output / "publication-receipt.json").exists())

    def test_foreign_child_cannot_resolve_recovery_intent(self):
        self.stage()
        self.api.unaccepted_commit_kind = "space"
        with self.assertRaises(pub.old.PublicationError):
            self.publish()
        self.api.move("space")
        with self.assertRaises(pub.old.PublicationError):
            self.publish()
        self.assertEqual(len(self.api.calls), 1)

    def test_verification_reads_all_27_space_bytes_and_reuses_dataset_proof(self):
        self.stage()
        self.publish()
        downloads = []

        def download(repo, name, kind, revision):
            downloads.append((repo, name, kind, revision))
            return self.api.download(repo, name, kind, revision)

        result = pub.verify(self.path, api=self.api, downloader=download)
        self.assertTrue(result["all_new_50_file_bytes_verified"])
        self.assertEqual(len(downloads), 27)
        self.assertTrue(all(kind == "space" for _, _, kind, _ in downloads))
        self.assertEqual(result["dataset"]["commit"], pub.DATASET)
        self.assertEqual(self.api.created, [])

    def test_wrong_anonymous_bytes_or_lfs_metadata_cannot_claim_verified_recovery(self):
        self.stage()
        self.publish()

        def corrupt(repo, name, kind, revision):
            raw = self.api.download(repo, name, kind, revision)
            return raw + b"wrong" if name == pub.CHANGED[0] else raw

        with self.assertRaises(pub.old.PublicationError):
            pub.verify(self.path, api=self.api, downloader=corrupt)
        self.assertFalse((self.output / "anonymous-readback.json").exists())
        raw, _ = self.api.data["space"][pub.CHANGED[0]]
        self.api.data["space"][pub.CHANGED[0]] = (raw, "git")
        self.api.snapshots[self.api.heads["space"]] = deepcopy(self.api.data["space"])
        with self.assertRaises(pub.old.PublicationError):
            pub.verify(self.path, api=self.api, downloader=self.api.download)
        self.assertFalse((self.output / "anonymous-readback.json").exists())


if __name__ == "__main__":
    unittest.main()

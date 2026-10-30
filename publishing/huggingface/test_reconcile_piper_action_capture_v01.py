"""No-network tests for the narrow, read-only server-append reconciliation."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import reconcile_piper_action_capture_v01 as pub
import test_recover_piper_action_capture_v01 as recovery_fixture


class ReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = recovery_fixture.RecoveryTests("test_stage_changes_only_two_representations_and_retains_original_evidence")
        self.fixture.setUp()
        self.fixture.stage()
        self.fixture.api.uncertain_commit_kind = "space"
        with self.assertRaises(pub.old.PublicationError):
            self.fixture.publish()
        self.path = self.fixture.path
        self.api = self.fixture.api
        self.output = self.path.parent / "final-reconciliation-receipt.json"
        self.plan = pub.recovery.checked(self.path)
        self.state = pub.old.load(self.path.parent / "journal.json")
        self.before_sources = {name: (self.path.parent / name).read_bytes() for name in ("plan.json", "journal.json")}
        self.before_attributes = self.api.snapshots[pub.recovery.SPACE_PARENT][".gitattributes"][0]
        self.after_attributes = self.before_attributes + pub.APPEND
        self.api.data["space"][".gitattributes"] = (self.after_attributes, "git")
        self.api.heads["space"] = pub.ACCEPTED
        self.api.snapshots[pub.ACCEPTED] = deepcopy(self.api.data["space"])
        self.api.history[pub.ACCEPTED] = (self.state["pending"]["title"], pub.recovery.SPACE_PARENT)
        self.real_checked = pub.checked
        self.patches = [
            patch.object(pub, "checked", return_value=(self.plan, self.state)),
            patch.object(pub, "OLD_ATTRIBUTES", pub.old.digest(self.before_attributes)),
            patch.object(pub, "NEW_ATTRIBUTES", pub.old.digest(self.after_attributes)),
            patch.object(pub.old, "public_client", side_effect=AssertionError("No implicit network clients in tests")),
            patch.object(self.api, "create_repo", side_effect=AssertionError("Read-only reconciler cannot create repositories")),
            patch.object(self.api, "create_commit", side_effect=AssertionError("Read-only reconciler cannot create commits")),
        ]
        for item in self.patches:
            item.start()
        self.downloads = []

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.fixture.tearDown()

    def download(self, repo, name, kind, revision):
        self.downloads.append((name, kind, revision))
        return self.api.download(repo, name, kind, revision)

    def verify(self, downloader=None):
        return pub.verify(self.path, api=self.api, downloader=downloader or self.download)

    def assert_no_receipt(self):
        self.assertFalse(self.output.exists())

    def test_exact_append_verifies_all_27_payloads_without_mutation_or_intent_rewrite(self):
        dataset_before = deepcopy(self.api.data["dataset"])
        calls_before = list(self.api.calls)
        result = self.verify()
        self.assertFalse(result["remote_mutations"])
        self.assertFalse(result["authenticated_requests"])
        self.assertFalse(result["server_deviation"]["old_baseline_condition_passed"])
        self.assertTrue(result["server_deviation"]["old_attribute_bytes_preserved_as_prefix"])
        self.assertEqual(len(result["accepted_space"]["files"]), 27)
        self.assertEqual(len(self.downloads), 29)
        self.assertEqual(self.downloads[:2], [(".gitattributes", "space", pub.recovery.SPACE_PARENT),
            (".gitattributes", "space", pub.ACCEPTED)])
        self.assertEqual(self.api.calls, calls_before)
        self.assertEqual(self.api.data["dataset"], dataset_before)
        self.assertEqual(self.fixture.snapshot_previous(), self.fixture.original_bytes)
        self.assertEqual({name: (self.path.parent / name).read_bytes() for name in self.before_sources}, self.before_sources)

    def test_extra_attribute_rule_refuses_even_with_a_known_replacement_digest(self):
        changed = self.after_attributes + b"another/path.bin filter=lfs diff=lfs merge=lfs -text\n"
        self.api.data["space"][".gitattributes"] = (changed, "git")
        self.api.snapshots[pub.ACCEPTED] = deepcopy(self.api.data["space"])
        with patch.object(pub, "NEW_ATTRIBUTES", pub.old.digest(changed)), self.assertRaises(pub.old.PublicationError):
            self.verify()
        self.assert_no_receipt()

    def test_changed_attribute_prefix_refuses(self):
        changed = b"changed prefix\n" + pub.APPEND
        self.api.data["space"][".gitattributes"] = (changed, "git")
        self.api.snapshots[pub.ACCEPTED] = deepcopy(self.api.data["space"])
        with self.assertRaises(pub.old.PublicationError):
            self.verify()
        self.assert_no_receipt()

    def test_unregistered_public_head_refuses_before_payload_downloads(self):
        self.api.move("space")
        with self.assertRaises(pub.old.PublicationError):
            self.verify()
        self.assertEqual(self.downloads, [])
        self.assert_no_receipt()

    def test_wrong_title_or_non_immediate_parent_refuses(self):
        wanted = (self.state["pending"]["title"], pub.recovery.SPACE_PARENT)
        for history in (("Foreign title", wanted[1]), (wanted[0], "4" * 40)):
            self.api.history[pub.ACCEPTED] = history
            with self.subTest(history=history), self.assertRaises(pub.old.PublicationError):
                self.verify()
        self.assertEqual(self.downloads, [])
        self.assert_no_receipt()

    def test_changed_prior_object_is_not_hidden_by_the_disclosed_attribute_append(self):
        self.api.data["space"]["robot-chat/config.json"] = (b'{"status":"unexpected"}\n', "git")
        self.api.snapshots[pub.ACCEPTED] = deepcopy(self.api.data["space"])
        with self.assertRaises(pub.old.PublicationError):
            self.verify()
        self.assertEqual(len(self.downloads), 2)
        self.assert_no_receipt()

    def test_wrong_anonymous_payload_bytes_cannot_produce_receipt(self):
        def corrupt(repo, name, kind, revision):
            raw = self.download(repo, name, kind, revision)
            return raw + b"wrong" if name == pub.recovery.CHANGED[0] else raw

        with self.assertRaises(pub.old.PublicationError):
            self.verify(corrupt)
        self.assert_no_receipt()

    def test_stopped_intent_change_during_readback_refuses(self):
        changed = False

        def mutate_temporary_intent(repo, name, kind, revision):
            nonlocal changed
            raw = self.download(repo, name, kind, revision)
            if name != ".gitattributes" and not changed:
                changed = True
                path = self.path.parent / "journal.json"
                path.write_bytes(path.read_bytes() + b" ")
            return raw

        with self.assertRaises(pub.old.PublicationError):
            self.verify(mutate_temporary_intent)
        self.assert_no_receipt()

    def test_existing_final_receipt_is_preserved_without_public_reads(self):
        self.output.write_bytes(b"Retained previous reconciliation receipt.\n")
        with patch.object(self.api, "repo_info", side_effect=AssertionError("Existing receipt must stop before public reads")), self.assertRaises(pub.old.PublicationError):
            self.verify()
        self.assertEqual(self.output.read_bytes(), b"Retained previous reconciliation receipt.\n")
        self.assertEqual(self.downloads, [])

    def test_real_checked_refuses_an_unreviewed_plan_before_any_network(self):
        with self.assertRaises(pub.old.PublicationError):
            self.real_checked(self.path)
        self.assert_no_receipt()


if __name__ == "__main__":
    unittest.main()

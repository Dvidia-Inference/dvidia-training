"""Actual export compatibility and fail-closed physical grounding contracts."""
from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
import hashlib
import json
import math
from pathlib import Path
import unittest

from dvidia_training.skillspace import (ADAPTER_ID, ARM_PROFILE, MAX_DOCUMENT_BYTES, GroundedSkill,
                              SkillspaceError, inspect_skillspace, load_skillspace)


EXAMPLE = Path(__file__).resolve().parent / "fixtures" / "arm_place_skillspace.json"


def example():
    return json.loads(EXAMPLE.read_text())


def source_metadata():
    # Exact field vocabulary from the shipped dvidia-cli place_cup example.
    return {"vdia": "0", "skill": "place_cup", "owner": "dvidia", "version": "0.3.0",
            "task": "Set a cup on the mark.", "demos": 100, "accuracy": 88,
            "cameras": ["ego", "room"], "format": "lerobot/v3", "license": "open",
            "residual": {"claimer": 0.1, "contributors": 0.3, "network": 0.6, "book": "not-live"}}


def cup_grounding():
    grounding = example()["simulation_grounding"]
    grounding["task_type"] = "place_cup"
    grounding["language_aliases"] = ["place a cup", "set a cup on the mark"]
    return grounding


class SkillspaceTests(unittest.TestCase):
    def assertRejected(self, document, code="invalid_document", grounding=None):
        with self.assertRaises(SkillspaceError) as context:
            load_skillspace(document, grounding)
        self.assertEqual(context.exception.code, code)
        json.dumps(context.exception.as_dict(), allow_nan=False)

    def test_cli_public_pack_and_current_browser_exports_are_metadata_only(self):
        cli = source_metadata()
        public = {"vdia": "0", "skill": "place_cup", "version": "0.3.0",
                  "task": "Set a cup on the mark.", "rail": "physical", "demos": 1,
                  "format": "lerobot/v3", "codebase_version": "v3.0",
                  "robot_type": "dvidia.wrist_room", "cameras": ["ego", "room"],
                  "residual": {"claimer": 0.5, "contributors": 0.5, "book": "not-live"}}
        browser = {**public, "accuracy": None, "robot_type": "human_video_reference", "residual": None,
                   "earnings_status": "unavailable", "dual": True, "rig": "wrist-room",
                   "info": "meta/info.json", "lerobot": {"dvidia": {"training_ready": False,
                       "robot_ready": False, "robot_actions": "absent", "robot_state": "absent"}}}
        for source in (cli, public, browser):
            with self.subTest(source=source.get("robot_type", "CLI")):
                before = copy.deepcopy(source)
                report = inspect_skillspace(source)
                self.assertEqual(report["skill_id"], "place_cup")
                self.assertEqual(report["readiness"], "grounding_required")
                self.assertFalse(report["source_has_robot_policy"])
                self.assertFalse(report["robot_ready"])
                self.assertIn("object", report["missing_grounding"])
                self.assertRejected(source, "grounding_required")
                self.assertEqual(source, before)

    def test_actual_contribution_brief_shape_preserves_local_permission_boundary(self):
        source = example()
        del source["simulation_grounding"]
        report = inspect_skillspace(source)
        self.assertEqual(report["source_format"], "dvidia-contribution-brief:1")
        self.assertEqual(report["skill_id"], "place_object")
        self.assertEqual(report["permissions"], "local_only_no_public_reuse")
        self.assertEqual(report["brief_id"], source["brief"]["id"])
        self.assertRejected(source, "grounding_required")
        source["permissions"]["publicReuseGranted"] = True
        self.assertRejected(source)

    def test_grounded_example_normalizes_jsonable_immutable_contract(self):
        raw = EXAMPLE.read_bytes()
        skill = load_skillspace(raw)
        self.assertIsInstance(skill, GroundedSkill)
        self.assertEqual(skill.source_sha256, hashlib.sha256(raw).hexdigest())
        self.assertEqual(skill.source_hash_kind, "exact_bytes")
        self.assertEqual(skill.arm_profile, ARM_PROFILE)
        self.assertEqual(skill.adapter_id, ADAPTER_ID)
        self.assertEqual(skill.object_size, (.04, .04, .04))
        result = json.loads(json.dumps(skill.to_dict(), allow_nan=False))
        self.assertEqual(result["object_position"], [.42, -.04, .31])
        self.assertEqual(result["object_representation"], "authored_box_proxy")
        self.assertFalse(result["robot_ready"])
        with self.assertRaises(FrozenInstanceError):
            skill.skill_id = "tie_lace"
        report = inspect_skillspace(raw)
        self.assertTrue(report["simulation_ready"])
        self.assertEqual(report["readiness"], "simulation_adapter_bound")

    def test_explicit_sidecar_binds_adapter_without_rewriting_source(self):
        source = source_metadata()
        original = copy.deepcopy(source)
        grounding = cup_grounding()
        skill = load_skillspace(source, grounding)
        self.assertEqual(skill.skill_id, "place_cup")
        self.assertEqual(skill.provenance, "authored_simulation_adapter")
        self.assertEqual(skill.source_hash_kind, "canonical_json")
        self.assertEqual(source, original)
        changed = copy.deepcopy(grounding)
        changed["object"]["mass"] = .05
        second = load_skillspace(source, changed)
        self.assertEqual(skill.source_sha256, second.source_sha256)
        self.assertNotEqual(skill.grounding_sha256, second.grounding_sha256)
        inline = example()
        self.assertRejected(inline, "ambiguous_grounding", inline["simulation_grounding"])

    def test_unsupported_skills_and_relabeling_fail_explicitly(self):
        for identifier in ("tie_lace", "tie your shoes", "wipe_counter", "unknown", "Place cup nearby"):
            source = source_metadata()
            source["skill"] = identifier
            with self.subTest(identifier=identifier):
                self.assertRejected(source, "unsupported_task", cup_grounding())
                with self.assertRaises(SkillspaceError):
                    inspect_skillspace(source)
        source = source_metadata()
        grounding = cup_grounding()
        grounding["task_type"] = "place_object"
        self.assertRejected(source, "unsupported_task", grounding)
        brief = example()
        brief["brief"]["taskTitle"] = "Tie your shoes"
        self.assertRejected(brief, "unsupported_task")

    def test_nonfinite_numbers_duplicate_keys_and_oversized_input_are_rejected(self):
        for value in (math.nan, math.inf, -math.inf, 10 ** 400):
            source = example()
            source["simulation_grounding"]["object"]["position"][0] = value
            self.assertRejected(source)
        for text in ('{"vdia":"0","vdia":"0"}', '{"number":NaN}', '{"number":1e999}', '[]', 'null'):
            with self.subTest(text=text):
                self.assertRejected(text)
        self.assertRejected(b"\xff")
        self.assertRejected(" " * (MAX_DOCUMENT_BYTES + 1), "document_too_large")

    def test_geometry_frame_profile_and_goal_bounds_fail_closed(self):
        changes = [("frame", "camera"), ("units", "cm"), ("table_height", .3),
                   ("arm_profile", "physical-arm"), ("adapter_id", "video-policy"),
                   ("provenance", "learned_from_video"), ("schema_version", True),
                   ("procedure", ["teleport", "release"])]
        for field, value in changes:
            source = example()
            source["simulation_grounding"][field] = value
            with self.subTest(field=field):
                self.assertRejected(source)
        invalid_objects = [{"size": [.5, .04, .04]}, {"mass": True}, {"friction": .1},
                           {"kind": "cup"}, {"position": [.7, 0, .31]},
                           {"position": [.42, 0, .50]}, {"position": [0, 0]}, {"velocity": [1, 0, 0]}]
        for values in invalid_objects:
            source = example()
            source["simulation_grounding"]["object"].update(values)
            with self.subTest(values=values):
                self.assertRejected(source)
        invalid_targets = [{"position": [.42, -.04, .31]}, {"position": [.54, .25, .31]},
                           {"position": [.60, .20, .31]},
                           {"position_tolerance": .2}, {"dwell_seconds": 0}]
        for values in invalid_targets:
            source = example()
            source["simulation_grounding"]["target"].update(values)
            with self.subTest(values=values):
                self.assertRejected(source)
        source = example()
        source["simulation_grounding"]["workspace"]["max"][0] = 100
        self.assertRejected(source)

        # A vertical-offset token cannot disguise a too-short horizontal task.
        source = example()
        source["simulation_grounding"]["object"]["position"] = [.42, 0, .30]
        source["simulation_grounding"]["target"]["position"] = [.492, 0, .32]
        self.assertRejected(source)

    def test_grounding_missing_unknown_fields_and_ambiguous_aliases_are_rejected(self):
        for mutate in (lambda row: row.pop("object"), lambda row: row.update({"execute": "evil()"}),
                       lambda row: row.update({"language_aliases": []}),
                       lambda row: row.update({"language_aliases": ["place object", "PLACE OBJECT"]})):
            source = example()
            mutate(source["simulation_grounding"])
            self.assertRejected(source)

    def test_brief_ids_timestamps_and_draft_terms_cannot_be_forged(self):
        for section, field, value in (("brief", "id", "bad-id"), ("brief", "version", True),
                                      ("brief", "createdAt", "2026-02-30T00:00:00.000Z"),
                                      ("commercialProposal", "proposedFounderShareBps", 1),
                                      ("brief", "creatorHandle", "invalid handle")):
            source = example()
            source[section][field] = value
            with self.subTest(field=field):
                self.assertRejected(source)

    def test_multiline_brief_text_uses_actual_export_contract(self):
        source = example()
        source["brief"]["goal"] += "\nThis remains authored scene metadata."
        source["brief"]["captureChecklist"][0] += "\nReview it locally."
        self.assertEqual(load_skillspace(source).skill_id, "place_object")


if __name__ == "__main__":
    unittest.main()

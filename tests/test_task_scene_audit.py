"""Independent audit faults, clock discontinuities and optional frozen run readback."""
from copy import deepcopy
import gzip
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

from dvidia_training import task_scene
from dvidia_training import task_scene_audit as audit


def fixture():
    recipe = task_scene.default_recipe()
    recipe["objects"] = [recipe["objects"][0]]
    recipe["objects"][0]["position_m"] = [.65, .2, .015]
    model = {"bodies": [{"id": 0, "name": "world"}, {"id": 8, "name": "link7"}, {"id": 9, "name": "link8"}],
        "geoms": [{"name": "task_table", "body_id": 0, "collision": True, "type": "box"},
            {"name": "task_bin_floor", "body_id": 0, "collision": True, "type": "box"},
            *[{"name": "pad_"+str(i), "body_id": 8+i//2, "collision": True, "type": "box"} for i in range(4)]]}
    state = {"time_s": 0., "objects": {"block": {"position_m": [.65, .2, .015], "quaternion_wxyz": [1., 0., 0., 0.],
        "linear_velocity_m_s": [0., 0., 0.], "angular_velocity_rad_s": [0., 0., 0.],
        "table_contact": False, "bin_floor_contact": True, "robot_contact": False, "pad_contact": False,
        "contact_geom_names": ["task_bin_floor"]}}, "valid": True, "warnings": []}
    return recipe, model, state


class RecomputeTests(unittest.TestCase):
    def test_consecutive_native_time_and_full_dwell_needed(self):
        recipe, model, state = fixture()
        verifier = audit.Recompute(recipe, model)
        for tick in range(251):
            state["time_s"] = tick*.001
            result = verifier.update(state)
            self.assertEqual(result["success"], tick == 250)
        self.assertAlmostEqual(result["settled_s"], .25)

    def test_point_inside_does_not_hide_corner_overhang_or_tilt(self):
        recipe, model, state = fixture()
        state["objects"]["block"]["position_m"][0] += .175
        result = audit.Recompute(recipe, model).update(state)
        self.assertFalse(result["objects"]["block"]["contained"])
        self.assertTrue(audit._weak(state, result, recipe)["body_origin_only"])
        state["objects"]["block"]["position_m"] = [.65, .2, .015]
        state["objects"]["block"]["quaternion_wxyz"] = [math.sqrt(.5), math.sqrt(.5), 0., 0.]
        self.assertFalse(audit.Recompute(recipe, model).update(state)["instantaneous"])

    def test_contact_strings_are_independently_cross_checked(self):
        recipe, model, state = fixture()
        for field, value in (("bin_floor_contact", False), ("robot_contact", True), ("pad_contact", True)):
            changed = deepcopy(state)
            changed["objects"]["block"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Contact labels"):
                audit.Recompute(recipe, model).update(changed)
        state["objects"]["block"]["contact_geom_names"].append("unknown_geom")
        with self.assertRaisesRegex(ValueError, "Unknown native contact"):
            audit.Recompute(recipe, model).update(state)

    def test_contact_speed_release_and_every_instance_are_required(self):
        recipe, model, state = fixture()
        cases = []
        missing = deepcopy(state)
        missing["objects"]["block"].update(bin_floor_contact=False, contact_geom_names=[])
        cases.append((missing, "bin_floor_contact"))
        moving = deepcopy(state)
        moving["objects"]["block"]["linear_velocity_m_s"] = [.1, 0., 0.]
        cases.append((moving, "linear_speed"))
        spinning = deepcopy(state)
        spinning["objects"]["block"]["angular_velocity_rad_s"] = [0., 0., .2]
        cases.append((spinning, "angular_speed"))
        held = deepcopy(state)
        held["objects"]["block"].update(robot_contact=True, pad_contact=True,
            contact_geom_names=["task_bin_floor", "pad_0"])
        cases.append((held, "released"))
        for changed, field in cases:
            with self.subTest(field=field):
                result = audit.Recompute(recipe, model).update(changed)
                self.assertFalse(result["objects"]["block"][field])
                self.assertFalse(result["instantaneous"])
                self.assertFalse(result["unknown"])
        recipe["objects"].append({**recipe["objects"][0], "id": "second"})
        state["objects"]["second"] = deepcopy(state["objects"]["block"])
        state["objects"]["second"]["position_m"][0] = .35
        result = audit.Recompute(recipe, model).update(state)
        self.assertFalse(result["instantaneous"])
        self.assertTrue(audit._weak(state, result, recipe)["any_instance_only"])

    def test_gap_reset_invalid_and_repeated_timestamp_cannot_supply_dwell(self):
        recipe, model, state = fixture()
        verifier = audit.Recompute(recipe, model)
        for _ in range(300):
            result = verifier.update(state)
        self.assertEqual(result["settled_s"], 0.)
        state["time_s"] = .1
        self.assertEqual(verifier.update(state)["reason"], "unobserved_interval")
        state["time_s"] = 0.
        self.assertEqual(verifier.update(state)["reason"], "time_reset")
        state["valid"] = False
        self.assertEqual(verifier.update(state)["reason"], "invalid_native_state")

    def test_rotated_bin_frame_is_independent_of_world_axes(self):
        recipe, model, state = fixture()
        recipe["bin"]["yaw_rad"] = math.pi/2
        state["objects"]["block"]["position_m"] = [.65, .35, .015]
        self.assertTrue(audit.Recompute(recipe, model).update(state)["instantaneous"])
        state["objects"]["block"]["position_m"] = [.80, .2, .015]
        self.assertFalse(audit.Recompute(recipe, model).update(state)["instantaneous"])

    def test_invalid_layouts_have_independent_geometry_reasons(self):
        from dvidia_training import task_scene_benchmark
        expected = {"invalid-overlap": "overlap", "invalid-support": "support", "invalid-fit": "fit", "invalid-duplicate-id": "duplicate"}
        for case in task_scene_benchmark.roster():
            recipe = task_scene_benchmark.recipe_for_case(case)
            with self.subTest(case=case["name"]):
                self.assertEqual(audit._precheck_fault(recipe), expected.get(case["name"]))

    def test_compressed_evidence_checks_unpacked_hash_and_size(self):
        raw = b'example evidence'
        self.assertEqual(audit._unpack(gzip.compress(raw), audit._identity(raw)), raw)
        with self.assertRaisesRegex(ValueError, "identity"):
            audit._unpack(gzip.compress(raw+b'changed'), audit._identity(raw))
        with self.assertRaisesRegex(ValueError, "byte bound"):
            audit._unpack(b'', {"bytes": audit.MAX_FILE+1, "sha256": "0"*64})

    def test_ambiguous_json_and_unsafe_evidence_paths_rejected(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}'):
            with self.assertRaises(ValueError):
                audit._decode(raw)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ("../outside", "/tmp/outside", "x/../outside", "x\\outside"):
                with self.assertRaises(ValueError):
                    audit._read(root, name)
            (root/"real").write_bytes(b'{}')
            (root/"link").symlink_to(root/"real")
            with self.assertRaisesRegex(ValueError, "links"):
                audit._read(root, "link")


@unittest.skipUnless(os.environ.get("DVIDIA_TASK_SCENE_STUDY"), "Set a completed frozen task-scene study for full read-only audit")
class FrozenStudyTests(unittest.TestCase):
    def test_completed_native_observations_and_all_counters_reconstruct(self):
        result = audit.audit(os.environ["DVIDIA_TASK_SCENE_STUDY"])
        self.assertEqual(result["counts"]["native_scheduled"], 12)
        self.assertEqual(result["counts"]["prechecks_correctly_rejected"], 4)
        self.assertEqual(result["mismatches"], 0)
        self.assertGreater(result["trace_samples"], 1000)


if __name__ == "__main__":
    unittest.main()

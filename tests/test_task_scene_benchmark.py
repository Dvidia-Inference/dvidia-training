"""Failure-sensitive ablations and frozen-input/attempt-preservation contracts."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from dvidia_training import task_scene as bridge
from dvidia_training import task_scene_benchmark as bench

HAS_USD = importlib.util.find_spec("pxr") is not None


def state_goal(*, contained=True, support=True, released=True, speed=True, instantaneous=None):
    item = {"contained": contained, "bin_floor_contact": support, "released": released,
            "linear_speed": speed, "angular_speed": True}
    item["instantaneous"] = all(item.values()) if instantaneous is None else instantaneous
    state = {"valid": True, "objects": {"block": {"position_m": [.65, .2, .03]}}}
    goal = {"unknown": False, "objects": {"block": item}, "instantaneous": item["instantaneous"]}
    return state, goal


class ContractTests(unittest.TestCase):
    def test_fixed_roster_separates_physics_positive_negative_and_precheck_cases(self):
        cases = bench.roster()
        self.assertEqual(len(cases), 16)
        self.assertEqual(len({case["id"] for case in cases}), 16)
        self.assertEqual(sum(c["expected_goal_success"] is True for c in cases), 6)
        self.assertEqual(sum(c["expected_goal_success"] is False for c in cases), 6)
        self.assertEqual(sum(c["control"] == "precheck" for c in cases), 4)
        self.assertEqual({c["name"] for c in cases if c["control"] == "gravity_disabled"}, {"rim-overhang", "unsupported"})

    def test_declared_recipe_prechecks_are_rejected_for_intended_geometry_reasons(self):
        reasons = {"invalid-overlap": "overlapping", "invalid-support": "overhangs", "invalid-fit": "cannot fit", "invalid-duplicate-id": "unique"}
        for case in bench.roster():
            recipe = bench.recipe_for_case(case)
            with self.subTest(case=case["name"]):
                if case["control"] == "precheck":
                    with self.assertRaisesRegex(ValueError, reasons[case["name"]]):
                        bridge.validate_recipe(recipe)
                else:
                    self.assertEqual(bridge.validate_recipe(recipe), recipe)

    def test_center_only_can_accept_overhang_while_full_corner_check_rejects_it(self):
        case = next(c for c in bench.roster() if c["name"] == "rim-overhang")
        recipe = bench.recipe_for_case(case)
        obj = recipe["objects"][0]
        self.assertTrue(bench._origin_inside(obj["position_m"], recipe["bin"]))
        self.assertFalse(bridge.containment(obj, obj["position_m"], [1., 0., 0., 0.], recipe["bin"])["contained"])

    def test_point_containment_uses_rotated_bin_coordinates(self):
        recipe = bridge.default_recipe()
        recipe["bin"]["yaw_rad"] = 1.5707963267948966
        self.assertTrue(bench._origin_inside(bench._world(recipe, .17, 0., .03), recipe["bin"]))
        self.assertFalse(bench._origin_inside(bench._world(recipe, 0., .17, .03), recipe["bin"]))

    def test_ablations_expose_omitted_contact_speed_dwell_and_required_instance_checks(self):
        recipe = bridge.default_recipe()
        for fault in ("support", "speed", "released", "contained"):
            kwargs = {fault: False}
            state, goal = state_goal(**kwargs)
            got = bench.ablation_predicates(state, goal, recipe)
            with self.subTest(fault=fault):
                self.assertFalse(got["without_dwell"])
                self.assertEqual(got["without_speed"], fault == "speed")
                self.assertEqual(got["geometry_only"], fault != "contained")
        state, goal = state_goal()
        goal["objects"]["outside"] = {**goal["objects"]["block"], "contained": False, "instantaneous": False}
        state["objects"]["outside"] = {"position_m": [.35, -.2, 0.]}
        goal["instantaneous"] = False
        got = bench.ablation_predicates(state, goal, recipe)
        self.assertTrue(got["any_instance_only"])
        self.assertFalse(got["without_dwell"])
        self.assertFalse(got["body_origin_only"])

    def test_unknown_invalid_native_state_never_passes_an_ablation(self):
        recipe = bridge.default_recipe()
        state, goal = state_goal()
        for fault in ("unknown", "invalid"):
            changed_state, changed_goal = deepcopy(state), deepcopy(goal)
            if fault == "unknown":
                changed_goal["unknown"] = True
            else:
                changed_state["valid"] = False
            self.assertFalse(any(bench.ablation_predicates(changed_state, changed_goal, recipe).values()))

    def test_dwell_restarts_after_failed_predicate(self):
        dwell = bench._Dwell(.05)
        self.assertFalse(dwell.update(0., True))
        self.assertTrue(dwell.update(.05, True))
        self.assertFalse(dwell.update(.06, False))
        self.assertFalse(dwell.update(.07, True))
        self.assertFalse(dwell.update(.10, True))
        self.assertTrue(dwell.update(.12, True))

    def test_content_addressed_geometry_reuses_exact_bytes_and_preserves_tampered_evidence(self):
        import gzip
        from hashlib import sha256
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"geometry"
            raw = b"native geometry"*100
            first = bench._store_geometry(root, raw)
            second = bench._store_geometry(root, raw)
            self.assertEqual(first, second)
            self.assertEqual(len(list(root.iterdir())), 1)
            path = root/(sha256(raw).hexdigest()+".bin.gz")
            self.assertEqual(gzip.decompress(path.read_bytes()), raw)
            self.assertEqual(first["href"], "../../geometry/"+path.name)
            path.write_bytes(b"retained corruption")
            with self.assertRaises(ValueError):
                bench._store_geometry(root, raw)
            self.assertEqual(path.read_bytes(), b"retained corruption")
            bench._store_geometry(root, b"other geometry")
            self.assertEqual(len(list(root.iterdir())), 2)

    def test_json_reader_rejects_duplicate_and_nonfinite_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"input.json"
            for raw in (b'{"goal":true,"goal":false}', b'{"nested":{"x":1,"x":2}}', b'{"value":NaN}', b'{"value":Infinity}'):
                path.write_bytes(raw)
                with self.subTest(raw=raw), self.assertRaises(ValueError):
                    bench._read_json(path)

    def test_exclusive_evidence_writes_do_not_overwrite_failed_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"started.json"
            bench.write(path, {"status": "failed"})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                bench.write(path, {"status": "success"})
            self.assertEqual(path.read_bytes(), before)

    def test_preview_rejects_nonfinite_unbounded_or_non_aligned_duration_before_native_work(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(bridge, "load_recipe", side_effect=AssertionError("Read before duration rejection")):
            for duration in (float("nan"), float("inf"), True, 0., 10.1, .0105):
                with self.subTest(duration=duration), self.assertRaises(ValueError):
                    bench.run_recipe("never-read", "never-read", Path(tmp)/"output", duration)


@unittest.skipUnless(HAS_USD, "Install optional usd-core for real USD input checks")
class FrozenInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.original = Path(cls.temp.name)/"prepared"
        bench.prepare(cls.original)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.temp_case = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_case.name)/"prepared"
        shutil.copytree(self.original, self.root)

    def tearDown(self):
        self.temp_case.cleanup()

    def assert_rejected_before_native(self):
        with patch.object(bridge, "build_scene", side_effect=AssertionError("Native work began before rejection")):
            with self.assertRaises(ValueError):
                bench.run(self.root, "never-read")

    def test_valid_frozen_inputs_pass_read_only_verification(self):
        protocol, seal, inventory = bench._verify_prepared(self.root)
        self.assertEqual(protocol["cases"], bench.roster())
        self.assertEqual(len(inventory["files"]), 40)
        self.assertIn("sha256", seal["protocol"])
        for case in protocol["cases"]:
            if case["control"] != "precheck":
                folder = self.root/"cases"/case["id"]/"input"
                recipe = bridge.load_recipe(folder/"recipe.json")
                parsed = bridge.inspect_stage(folder/"workstation.usda", recipe)
                self.assertEqual(set(parsed["objects"]), {obj["id"] for obj in recipe["objects"]})

    def test_protocol_inventory_case_source_and_usd_changes_are_rejected(self):
        for fault in ("protocol", "inventory", "usd", "source", "extra-input", "nested-input", "extra-case"):
            if fault != "protocol":
                shutil.rmtree(self.root)
                shutil.copytree(self.original, self.root)
            if fault == "protocol":
                path = self.root/"protocol.json"
                data = json.loads(path.read_bytes())
                data["cases"][0]["expected_goal_success"] = False
                path.write_text(json.dumps(data))
            elif fault == "inventory":
                path = self.root/"input-inventory.json"
                data = json.loads(path.read_bytes())
                data["files"].pop(next(iter(data["files"])))
                path.write_text(json.dumps(data))
            elif fault in ("usd", "source"):
                path = self.root/("cases/case-00/input/workstation.usda" if fault == "usd" else "source/task_scene.py")
                path.write_bytes(path.read_bytes()+b"\n# changed\n")
            elif fault == "extra-case":
                (self.root/"cases/case-99").mkdir()
            else:
                path = self.root/("cases/case-00/input/extra.json" if fault == "extra-input" else "cases/case-00/input/nested/extra.json")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
            with self.subTest(fault=fault):
                self.assert_rejected_before_native()

    def test_any_started_case_or_existing_intent_blocks_the_entire_batch(self):
        for name in ("cases/case-15/started.json", "execution-intent.json"):
            if (self.root/"cases/case-15/started.json").exists():
                shutil.rmtree(self.root)
                shutil.copytree(self.original, self.root)
            (self.root/name).write_text("{}")
            self.assert_rejected_before_native()

    def test_prepare_and_preview_refuse_existing_directories(self):
        before = (self.root/"protocol.json").read_bytes()
        with self.assertRaises(ValueError):
            bench.prepare(self.root)
        with self.assertRaises(ValueError):
            bench.run_recipe("never-read", "never-read", self.root)
        self.assertEqual((self.root/"protocol.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

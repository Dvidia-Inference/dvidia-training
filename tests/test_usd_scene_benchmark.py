"""Independent trajectory oracles and benchmark input/attempt preservation."""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from dvidia_training import usd_scene as bridge
from dvidia_training import usd_scene_benchmark as bench


EXAMPLE = Path(__file__).resolve().parents[1]/"examples/usd-workstation-v01/recipe.json"
HAS_USD = importlib.util.find_spec("pxr") is not None


def protocol():
    return {"thresholds": {"drop_early_position_error_m": .001, "drop_final_height_error_m": .001,
        "drop_final_speed_m_s": .02, "normal_slide_final_speed_m_s": .02,
        "sliding_distance_relative_error": .35, "sliding_distance_absolute_floor_m": .002,
        "initial_slide_speed_tolerance_m_s": .005, "negative_slide_final_speed_m_s": .30},
        "duration_s": {"drop": 2., "slide": 1.}, "slide_initial_speed_m_s": .4,
        "timestep_s": .001}


def drop_rows(case):
    height = .02+case["height_m"]
    result = []
    for tick in range(201):
        t = tick/100
        z = height-.5*9.81*t*t
        velocity = -9.81*t
        support = False
        if case["control"] == "gravity_disabled":
            z, velocity = height, 0.
        elif case["control"] == "normal" and z <= .02:
            z, velocity, support = .02, 0., True
        result.append({"t_s": t, "position_m": [.55, .15, z],
            "linear_velocity_m_s": [0., 0., velocity], "angular_velocity_rad_s": [0., 0., 0.],
            "table_contact": support, "robot_contact": False})
    return result


def slide_rows(case):
    result = []
    for tick in range(101):
        t = tick/100
        mu = case["friction"]
        stop = math.inf if mu == 0 else .4/(mu*9.81)
        until = min(t, stop)
        distance = .4*until-.5*mu*9.81*until*until
        result.append({"t_s": t, "position_m": [.35+distance, .25, .02],
            "linear_velocity_m_s": [max(0., .4-mu*9.81*t), 0., 0.],
            "angular_velocity_rad_s": [0., 0., 0.], "table_contact": True, "robot_contact": False})
    return result


class PhysicsOracleTests(unittest.TestCase):
    def test_all_45_scheduled_cases_have_unique_membership_and_keep_controls(self):
        cases = bench.roster()
        self.assertEqual(len(cases), 45)
        self.assertEqual(len({case["id"] for case in cases}), 45)
        self.assertEqual(sum(case["kind"] == "drop" and case["control"] == "normal" for case in cases), 27)
        self.assertEqual(sum(case["kind"] == "slide" and case["control"] == "normal" for case in cases), 9)
        self.assertEqual(sum(case["control"] != "normal" for case in cases), 9)
        self.assertEqual({case["mass_kg"] for case in cases if case["control"] == "normal"}, {.04, .08, .16})
        self.assertEqual({case["friction"] for case in cases if case["control"] == "normal"}, {.1, .5, 1.})

    def test_analytic_drop_and_slide_oracles_pass_every_declared_condition(self):
        for case in bench.roster():
            rows = drop_rows(case) if case["kind"] == "drop" else slide_rows(case)
            result = bench.score(case, rows, protocol())
            with self.subTest(case=case["id"]):
                self.assertTrue(result["valid"])
                self.assertTrue(result["success"], result)

    def test_normal_drop_requires_freefall_support_height_and_stopping(self):
        case = next(case for case in bench.roster() if case["kind"] == "drop" and case["control"] == "normal")
        for name in ("support", "height", "motion", "gravity", "robot_contact"):
            rows = drop_rows(case)
            if name == "support":
                rows[-1]["table_contact"] = False
            elif name == "height":
                rows[-1]["position_m"][2] = .04
            elif name == "motion":
                rows[-1]["linear_velocity_m_s"][0] = .3
            elif name == "gravity":
                rows[5]["position_m"][2] = rows[0]["position_m"][2]
            else:
                rows[100]["robot_contact"] = True
            with self.subTest(fault=name):
                self.assertFalse(bench.score(case, rows, protocol())["success"])

    def test_negative_controls_fail_if_the_disabled_effect_is_not_observable(self):
        cases = {case["control"]: case for case in bench.roster() if case["control"] != "normal"}
        for name, case in cases.items():
            rows = drop_rows(case) if case["kind"] == "drop" else slide_rows(case)
            if name == "table_disabled":
                rows[-1]["position_m"][2], rows[-1]["table_contact"] = .02, True
            elif name == "gravity_disabled":
                rows[-1]["position_m"][2] -= .02
            else:
                rows[-1]["linear_velocity_m_s"] = [0., 0., 0.]
            with self.subTest(control=name):
                self.assertFalse(bench.score(case, rows, protocol())["success"])

    def test_slide_cannot_pass_on_stopping_alone_when_the_stopping_distance_is_wrong(self):
        case = next(case for case in bench.roster() if case["kind"] == "slide" and case["control"] == "normal")
        rows = slide_rows(case)
        rows[-1]["position_m"][0] += .10
        self.assertFalse(bench.score(case, rows, protocol())["success"])

    def test_stationary_object_does_not_count_as_a_successful_high_friction_slide(self):
        cases = [case for case in bench.roster() if case["kind"] == "slide" and case["control"] == "normal"]
        for case in cases:
            for fake_initial_kick in (False, True):
                rows = slide_rows(case)
                for row in rows:
                    row["position_m"][0] = .35
                    row["linear_velocity_m_s"] = [0., 0., 0.]
                if fake_initial_kick:
                    rows[0]["linear_velocity_m_s"][0] = .4
                with self.subTest(case=case["id"], initial_kick=fake_initial_kick):
                    self.assertFalse(bench.score(case, rows, protocol())["success"])

    def test_empty_trajectory_fails_explicitly(self):
        with self.assertRaises(ValueError):
            bench.score(bench.roster()[0], [], protocol())

    def test_nonfinite_trajectory_and_unobserved_time_are_invalid(self):
        case = bench.roster()[0]
        for fault in ("nan_position", "nan_time", "missing_end", "duplicate_time", "missing_rows"):
            rows = drop_rows(case)
            if fault == "nan_position":
                rows[10]["position_m"][1] = float("nan")
            elif fault == "nan_time":
                rows[10]["t_s"] = float("nan")
            elif fault == "missing_end":
                rows.pop()
            elif fault == "duplicate_time":
                rows[10]["t_s"] = rows[9]["t_s"]
            else:
                rows.pop(10)
            with self.subTest(fault=fault):
                self.assertFalse(bench.score(case, rows, protocol())["valid"])

    def test_exclusive_evidence_writes_preserve_existing_attempt_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)/"attempt.json"
            bench.write(target, {"result": "failed"})
            before = target.read_bytes()
            with self.assertRaises(FileExistsError):
                bench.write(target, {"result": "success"})
            self.assertEqual(target.read_bytes(), before)


@unittest.skipUnless(HAS_USD, "Install optional usd-core for real authored-variant checks")
class FrozenUSDInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.original = Path(cls.temporary.name)/"prepared"
        cls.protocol = bench.prepare(EXAMPLE, cls.original)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.temporary_case = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_case.name)/"prepared"
        shutil.copytree(self.original, self.root)

    def tearDown(self):
        self.temporary_case.cleanup()

    def assert_rejected_before_native(self):
        with patch.object(bench.sim, "_native_imports", side_effect=AssertionError("Native work began before input rejection")):
            with self.assertRaises(ValueError):
                bench.run(self.root, "must-not-read")

    def test_every_variant_is_real_usd_authored_with_both_materials_and_mass_inertia(self):
        from pxr import Usd, UsdPhysics
        for case in self.protocol["cases"]:
            recipe_path = self.root/"cases"/case["id"]/"input/recipe.json"
            recipe = bridge.load_recipe(recipe_path)
            path = recipe_path.parent/"workstation.usda"
            parsed = bridge.inspect_stage(path)
            with self.subTest(case=case["id"]):
                self.assertEqual(parsed["stage_sha256"], recipe["stage_sha256"])
                self.assertAlmostEqual(parsed["object"]["mass_kg"], case["mass_kg"], places=7)
                self.assertAlmostEqual(parsed["object"]["sliding_friction"], case["friction"], places=7)
                self.assertEqual(parsed["object"]["sliding_friction"], parsed["table"]["sliding_friction"])
                inertia = case["mass_kg"]*(.04**2+.04**2)/12
                for component in parsed["object"]["diagonal_inertia_kg_m2"]:
                    self.assertAlmostEqual(component, inertia, places=9)
                stage = Usd.Stage.Open(str(path))
                material = UsdPhysics.MaterialAPI(stage.GetPrimAtPath("/World/ObjectMaterial"))
                self.assertEqual(material.GetStaticFrictionAttr().Get(), material.GetDynamicFrictionAttr().Get())

    def test_protocol_threshold_tampering_fails_before_any_attempt(self):
        path = self.root/"protocol.json"
        data = json.loads(path.read_bytes())
        data["thresholds"]["drop_final_height_error_m"] = 99
        path.write_text(json.dumps(data))
        self.assert_rejected_before_native()

    def test_inventory_omission_extra_file_or_changed_usd_fails_before_native(self):
        for fault in ("omitted_inventory", "extra_input", "nested_extra_input", "unknown_case", "changed_usd"):
            if fault != "omitted_inventory":
                shutil.rmtree(self.root)
                shutil.copytree(self.original, self.root)
            if fault == "omitted_inventory":
                path = self.root/"input-inventory.json"
                data = json.loads(path.read_bytes())
                data["files"].pop(next(iter(data["files"])))
                path.write_text(json.dumps(data))
            elif fault == "extra_input":
                (self.root/"cases/case-00/input/extra.usda").write_text("#usda 1.0")
            elif fault == "nested_extra_input":
                directory = self.root/"cases/case-00/input/nested"
                directory.mkdir()
                (directory/"extra.usda").write_text("#usda 1.0")
            elif fault == "unknown_case":
                (self.root/"cases/case-99").mkdir()
            else:
                path = self.root/"cases/case-00/input/workstation.usda"
                path.write_bytes(path.read_bytes()+b"\n# altered\n")
            with self.subTest(fault=fault):
                self.assert_rejected_before_native()

    def test_retained_runtime_source_tamper_is_rejected_before_native(self):
        path = self.root/"source/usd_scene.py"
        path.write_bytes(path.read_bytes()+b"\n# altered\n")
        self.assert_rejected_before_native()

    def test_later_started_case_blocks_the_entire_batch_before_native_work(self):
        path = self.root/"cases/case-44/started.json"
        path.write_text("{}\n")
        self.assert_rejected_before_native()

    def test_prepare_refuses_existing_output_instead_of_overwriting_failed_attempts(self):
        before = (self.root/"protocol.json").read_bytes()
        with self.assertRaises(ValueError):
            bench.prepare(EXAMPLE, self.root)
        self.assertEqual((self.root/"protocol.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

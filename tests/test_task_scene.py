"""Strict multi-object scene boundaries, real USD parsing and native predicates."""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from dvidia_training import task_scene as scene

HAS_USD = importlib.util.find_spec("pxr") is not None
HAS_NATIVE = importlib.util.find_spec("mujoco") is not None
ASSETS = os.environ.get("DVIDIA_ROBOT_PROFILE_ASSETS")


class RecipeTests(unittest.TestCase):
    def setUp(self):
        self.recipe = scene.default_recipe()

    def test_default_is_independent_and_requires_no_optional_package(self):
        other = scene.default_recipe()
        self.recipe["objects"][0]["position_m"][0] = 100
        self.assertEqual(other["objects"][0]["position_m"][0], .35)
        code = ("import sys; sys.modules['pxr']=None; sys.modules['mujoco']=None; "
                "from dvidia_training.task_scene import default_recipe,validate_recipe; "
                "assert validate_recipe(default_recipe())['robot_profile']=='piper'")
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_all_nested_contracts_reject_missing_unknown_and_wrong_locks(self):
        changes = [{"schema": "other"}, {"robot_profile": "ur5e"}, {"catalog_sha256": "0"*64},
                   {"robot_lock_sha256": "0"*64}, {"source_sha256": {}}, {"timestep_s": .002},
                   {"control_s": .01}, {"custom": "run.py"}, {"objects": []}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                scene.validate_recipe({**self.recipe, **change})
        for key in ("table", "bin", "goal"):
            bad = deepcopy(self.recipe)
            bad[key]["unsupported"] = 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                scene.validate_recipe(bad)
        bad = deepcopy(self.recipe)
        bad["objects"][0]["unsupported"] = 1
        with self.assertRaises(ValueError):
            scene.validate_recipe(bad)

    def test_numerics_reject_boolean_nan_inf_and_unbounded_geometry(self):
        for value in (True, math.nan, math.inf, -.1, 3):
            bad = deepcopy(self.recipe)
            bad["objects"][0]["mass_kg"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                scene.validate_recipe(bad)
        for field in ("timestep_s", "control_s"):
            with self.assertRaises(ValueError):
                scene.validate_recipe({**self.recipe, field: True})
        for value in ([.01, .01], [1, 1, 1], [.04, 0, .04]):
            bad = deepcopy(self.recipe)
            bad["objects"][0]["size_m"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                scene.validate_recipe(bad)

    def test_ids_and_asset_classes_are_bounded_and_unique(self):
        for value in ("../outside", "CamelCase", "x"*25, "", True):
            bad = deepcopy(self.recipe)
            bad["objects"][0]["id"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                scene.validate_recipe(bad)
        bad = deepcopy(self.recipe)
        bad["objects"][1]["id"] = "block"
        with self.assertRaisesRegex(ValueError, "unique"):
            scene.validate_recipe(bad)
        bad = deepcopy(self.recipe)
        bad["objects"][0]["kind"] = "external.usd"
        with self.assertRaisesRegex(ValueError, "original"):
            scene.validate_recipe(bad)
        bad["objects"] = [deepcopy(self.recipe["objects"][0]) for _ in range(5)]
        with self.assertRaisesRegex(ValueError, "four"):
            scene.validate_recipe(bad)

    def test_assumptions_cannot_be_relabelled_measured(self):
        for key in ("table", "bin"):
            bad = deepcopy(self.recipe)
            bad[key]["property_evidence"] = "measured"
            with self.assertRaisesRegex(ValueError, "assumptions"):
                scene.validate_recipe(bad)
        bad = deepcopy(self.recipe)
        bad["objects"][0]["property_evidence"] = "calibrated"
        with self.assertRaisesRegex(ValueError, "assumptions"):
            scene.validate_recipe(bad)

    def test_support_plane_bin_floor_overhang_and_start_overlap_fail(self):
        cases = []
        bad = deepcopy(self.recipe); bad["table"]["center_m"][2] = -.03; cases.append(bad)
        bad = deepcopy(self.recipe); bad["bin"]["center_m"][2] = .05; cases.append(bad)
        bad = deepcopy(self.recipe); bad["objects"][0]["position_m"] = [1., 0., .04]; cases.append(bad)
        bad = deepcopy(self.recipe); bad["bin"]["center_m"][0] = 1.; cases.append(bad)
        bad = deepcopy(self.recipe); bad["objects"][1]["position_m"] = [.35, -.2, .04]; cases.append(bad)
        bad = deepcopy(self.recipe); bad["objects"][0]["position_m"] = [.65, .2, 0.]; cases.append(bad)
        bad = deepcopy(self.recipe); bad["objects"][0]["position_m"] = [.65+.18, .2, .04]; cases.append(bad)
        bad = deepcopy(self.recipe); bad["objects"][0]["position_m"][2] = -.001; cases.append(bad)
        for i, bad in enumerate(cases):
            with self.subTest(case=i), self.assertRaises(ValueError):
                scene.validate_recipe(bad)

    def test_oriented_separating_axes_accept_touch_and_reject_actual_overlap(self):
        a = scene._obb([0, 0, 0], [2, .1, 1], math.pi/4)
        b = scene._obb([0, .2, 0], [2, .1, 1], math.pi/4)
        self.assertFalse(scene._overlap(a, b), "AABB overlap alone is insufficient")
        b["center"][1] = .05
        self.assertTrue(scene._overlap(a, b))
        c = scene._obb([0, 0, 1], [2, .1, 1], 0)
        self.assertFalse(scene._overlap(a, c), "Face touching is not initial penetration")

    def test_object_fit_is_not_silently_resized_or_rotated(self):
        bad = deepcopy(self.recipe)
        bad["bin"]["inner_size_m"] = [.15, .12, .16]
        with self.assertRaisesRegex(ValueError, "cannot fit"):
            scene.validate_recipe(bad)
        bad = deepcopy(self.recipe)
        bad["bin"]["inner_size_m"][2] = .05
        with self.assertRaisesRegex(ValueError, "cannot fit"):
            scene.validate_recipe(bad)

    def test_compound_shoe_is_original_three_touching_boxes_with_conserved_volume_mass(self):
        obj = self.recipe["objects"][1]
        pieces = scene.collision_pieces(obj)
        self.assertEqual(len(pieces), 3)
        self.assertAlmostEqual(sum(p["mass_fraction"] for p in pieces), 1)
        for i, piece in enumerate(pieces):
            top = piece["position_m"][2]+piece["size_m"][2]/2
            if i+1 < len(pieces):
                next_ = pieces[i+1]
                self.assertAlmostEqual(top, next_["position_m"][2]-next_["size_m"][2]/2)
        corners = scene.object_corners(obj)
        self.assertEqual(len(corners), 24)
        self.assertAlmostEqual(max(p[2] for p in corners), obj["position_m"][2]+obj["size_m"][2])
        center, inertia = scene._inertia(obj)
        self.assertEqual(center[:2], [0, 0])
        self.assertTrue(all(x > 0 for x in inertia))
        self.assertTrue(all(x < sum(inertia)-x for x in inertia))

    def test_containment_uses_all_corners_not_centre_and_respects_rotated_bin(self):
        obj, bin_ = self.recipe["objects"][0], self.recipe["bin"]
        pose = [.65, .2, .015]
        self.assertTrue(scene.containment(obj, pose, [1, 0, 0, 0], bin_)["contained"])
        pose[0] += .175
        self.assertFalse(scene.containment(obj, pose, [1, 0, 0, 0], bin_)["contained"])
        rotated = deepcopy(bin_); rotated["yaw_rad"] = math.pi/2
        self.assertFalse(scene.containment(obj, pose, [1, 0, 0, 0], rotated)["contained"])
        pose = [.65, .2+.15, .015]
        self.assertTrue(scene.containment(obj, pose, scene._yaw(math.pi/2), rotated)["contained"])

    def test_tilt_top_floor_and_finite_pose_are_checked(self):
        obj, bin_ = self.recipe["objects"][0], self.recipe["bin"]
        self.assertFalse(scene.containment(obj, [.65, .2, .17], [1, 0, 0, 0], bin_)["contained"])
        self.assertFalse(scene.containment(obj, [.65, .2, 0], [1, 0, 0, 0], bin_)["contained"])
        tilt = [math.sqrt(.5), math.sqrt(.5), 0, 0]
        self.assertFalse(scene.containment(obj, [.65, .2, .015], tilt, bin_)["contained"])
        for pos, quat in (([math.nan, 0, 0], [1, 0, 0, 0]), ([0, 0, 0], [.5, 0, 0, 0])):
            with self.assertRaises(ValueError):
                scene.containment(obj, pos, quat, bin_)

    def test_disk_json_rejects_duplicate_and_nonfinite_tokens(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"recipe.json"
            original = json.dumps(self.recipe)
            duplicate = original[:-1]+', "robot_profile": "piper"}'
            for text in (duplicate, original.replace('"timestep_s": 0.001', '"timestep_s": NaN')):
                path.write_text(text)
                with self.assertRaisesRegex(ValueError, "Duplicate|Nonfinite"):
                    scene.load_recipe(path)

    def test_path_bound_and_symlink_fail_before_optional_packages(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"recipe.json"
            p.write_bytes(b"x"*(scene._MAX_BYTES+1))
            with self.assertRaises(ValueError):
                scene.load_recipe(p)
            link = Path(d)/"link.json"; link.symlink_to(p)
            with self.assertRaises(ValueError):
                scene.load_recipe(link)


@unittest.skipUnless(HAS_USD, "Optional usd-core required")
class USDTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)/"scene"
        self.recipe = scene.default_recipe()
        self.authored = scene.write_scene(self.recipe, self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_sdk_roundtrip_exact_box_dimensions_materials_and_explicit_units(self):
        from pxr import Usd, UsdGeom, UsdPhysics
        stage = Usd.Stage.Open(str(self.root/scene.STAGE))
        self.assertEqual(UsdGeom.GetStageUpAxis(stage), "Z")
        self.assertEqual(UsdGeom.GetStageMetersPerUnit(stage), 1)
        self.assertEqual(UsdPhysics.GetStageKilogramsPerUnit(stage), 1)
        result = scene.inspect_stage(self.root/scene.STAGE, self.recipe)
        self.assertEqual(len(result["bin"]), 5)
        self.assertEqual(len(result["objects"]), 2)
        self.assertEqual(result["table"]["size_m"], [1.2, 1., .08])
        self.assertEqual(result["objects"]["shoe"]["pieces"][0]["size_m"], [.24, .09, .0126])
        self.assertAlmostEqual(result["objects"]["block"]["mass_kg"], .04, places=8)
        self.assertEqual(self.authored["files"][scene.STAGE]["sha256"], sha256((self.root/scene.STAGE).read_bytes()).hexdigest())

    def test_source_and_stage_bytes_are_deterministic_and_output_not_overwritten(self):
        other = Path(self.tmp.name)/"second"
        result = scene.write_scene(self.recipe, other)
        self.assertEqual(self.authored, result)
        for name in ("recipe.json", scene.STAGE, "scene-receipt.json"):
            self.assertEqual((self.root/name).read_bytes(), (other/name).read_bytes())
        with self.assertRaisesRegex(ValueError, "exists"):
            scene.write_scene(self.recipe, self.root)
        link = Path(self.tmp.name)/"link"; link.symlink_to(self.root)
        with self.assertRaises(ValueError):
            scene.write_scene(self.recipe, link)

    def test_composition_external_assets_unknown_metadata_and_time_samples_refused(self):
        from pxr import Sdf, Usd
        original = (self.root/scene.STAGE).read_bytes()
        for kind in ("reference", "sublayer", "inherit", "script", "time", "extra", "units"):
            layer = Sdf.Layer.CreateAnonymous()
            layer.ImportFromString(original.decode())
            prim = layer.GetPrimAtPath("/World/Object_block")
            if kind == "reference":
                prim.referenceList.prependedItems = [Sdf.Reference("https://invalid.example/no.usda")]
            elif kind == "sublayer":
                layer.subLayerPaths = ["missing.usda"]
            elif kind == "inherit":
                prim.inheritPathList.prependedItems = [Sdf.Path("/World/Table")]
            elif kind == "script":
                Sdf.AttributeSpec(prim, "script", Sdf.ValueTypeNames.String).default = "bad.py"
            elif kind == "time":
                layer.SetTimeSample(Sdf.Path("/World/Object_block.xformOp:translate"), 1., (0., 0., 0.))
            elif kind == "extra":
                Sdf.CreatePrimInLayer(layer, "/World/Extra")
            else:
                layer.pseudoRoot.SetInfo("metersPerUnit", .01)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                scene._inspect_bytes(layer.ExportToString().encode(), self.recipe)

    def test_physically_edited_stage_cannot_pass_pinned_recipe_even_if_parseable(self):
        from pxr import Gf, Usd
        stage = Usd.Stage.Open(str(self.root/scene.STAGE))
        stage.GetPrimAtPath("/World/Bin_front").GetAttribute("xformOp:scale").Set(Gf.Vec3d(.36, .015, .001))
        stage.GetRootLayer().Save()
        del stage
        with self.assertRaisesRegex(ValueError, "bounded recipe"):
            scene.inspect_stage(self.root/scene.STAGE, self.recipe)

    def test_object_named_table_does_not_overwrite_table_material(self):
        recipe = deepcopy(self.recipe)
        recipe["objects"][0]["id"] = "table"
        other = Path(self.tmp.name)/"names"
        result = scene.write_scene(recipe, other)
        self.assertAlmostEqual(result["usd"]["table"]["sliding_friction"], .8, places=6)
        self.assertAlmostEqual(result["usd"]["objects"]["table"]["sliding_friction"], .5)


@unittest.skipUnless(HAS_USD and HAS_NATIVE and ASSETS, "usd-core, mujoco and DVIDIA_ROBOT_PROFILE_ASSETS required")
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.recipe = scene.default_recipe()
        # Two distinct objects already inside bin, with a real gravity drop.
        cls.recipe["objects"][0]["position_m"] = [.52, .2, .045]
        cls.recipe["objects"][1]["position_m"] = [.69, .2, .045]
        scene.write_scene(cls.recipe, cls.root/"scene")
        cls.model, cls.receipt = scene.build_scene(cls.root/"scene/recipe.json", ASSETS)
        cls.mj, cls.np = scene.sim._native_imports()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_compiled_mechanics_pin_native_arm_and_open_bin(self):
        self.assertTrue(all(self.receipt["compiled_checks"].values()))
        self.assertEqual((self.model.nq, self.model.nv, self.model.nu, self.model.neq), (22, 20, 7, 1))
        self.assertEqual(len([i for i in range(self.model.ngeom) if (self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_GEOM, i) or "").startswith("task_bin_")]), 5)
        self.assertEqual(self.receipt["scope"]["deformation"], False)
        self.assertIn("world frame", self.receipt["mapping"]["state_velocity_frame"])

    def test_initialize_has_no_hidden_step_force_or_robot_command_modification(self):
        data = scene.initialize(self.model, self.receipt)
        self.assertEqual(data.time, 0)
        self.assertFalse(self.np.any(data.xfrc_applied))
        self.assertFalse(self.np.any(data.qfrc_applied))
        self.assertEqual(data.ctrl.shape, (7,))
        for index, position in enumerate((0., 1.57, -1.3485, 0., 0., 0., 0., 0.)):
            self.assertEqual(data.qpos[index], position)
        state = scene.sample_state(self.model, data, self.receipt)
        self.assertTrue(state["valid"])
        self.assertEqual(set(state["objects"]), {"block", "shoe"})
        self.assertFalse(any(x["bin_floor_contact"] for x in state["objects"].values()))

    def test_native_drop_open_bin_support_and_dwell_all_objects(self):
        data = scene.initialize(self.model, self.receipt)
        monitor = scene.GoalMonitor(self.receipt)
        self.assertFalse(monitor.update(self.model, data)["success"])
        for _ in range(1800):
            self.mj.mj_step(self.model, data)
            self.mj.mj_forward(self.model, data)
            result = monitor.update(self.model, data)
        self.assertTrue(result["success"], result)
        self.assertTrue(all(x["bin_floor_contact"] for x in result["objects"].values()))
        self.assertFalse(self.np.any(data.xfrc_applied))
        self.assertFalse(any(w.number for w in data.warning))
        # A physically shifted single object defeats all-object conjunction.
        joint = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_JOINT, "task_object_block_free")
        data.qpos[self.model.jnt_qposadr[joint]] -= .15
        self.mj.mj_forward(self.model, data)
        self.assertFalse(monitor.update(self.model, data)["success"])

    def test_missing_support_fast_motion_and_robot_contact_cannot_complete(self):
        data = scene.initialize(self.model, self.receipt)
        state = scene.sample_state(self.model, data, self.receipt)
        for item in state["objects"].values():
            item["position_m"][2] = .015
            item["bin_floor_contact"] = True
        state["time_s"] = 0.
        for field, value, expected in (("bin_floor_contact", False, "bin_floor_contact"),
                ("robot_contact", True, "released"), ("pad_contact", True, "released"),
                ("linear_velocity_m_s", [.1, 0, 0], "linear_speed"),
                ("angular_velocity_rad_s", [0, 0, 1.], "angular_speed")):
            altered = deepcopy(state)
            altered["objects"]["block"][field] = value
            with self.subTest(field=field), patch.object(scene, "sample_state", return_value=altered):
                result = scene.GoalMonitor(self.receipt).update(self.model, data)
                self.assertFalse(result["objects"]["block"][expected])
                self.assertFalse(result["instantaneous"])
                self.assertFalse(result["unknown"])

    def test_time_gap_reset_and_warning_are_unknown_repeated_time_adds_no_dwell(self):
        data = scene.initialize(self.model, self.receipt)
        state = scene.sample_state(self.model, data, self.receipt)
        for item in state["objects"].values():
            item["position_m"][2] = .015
            item["bin_floor_contact"] = True
        monitor = scene.GoalMonitor(self.receipt)
        with patch.object(scene, "sample_state", return_value=state):
            for _ in range(100):
                self.assertFalse(monitor.update(self.model, data)["success"])
            self.assertEqual(monitor.since, 0)
            state["time_s"] = .1
            result = monitor.update(self.model, data)
            self.assertEqual(result["reason"], "unobserved_interval")
            self.assertTrue(result["unknown"])
            state["time_s"] = 0.
            self.assertEqual(monitor.update(self.model, data)["reason"], "time_reset")
            state["valid"] = False
            self.assertEqual(monitor.update(self.model, data)["reason"], "invalid_native_state")
        monitor.reset()
        self.assertIsNone(monitor.previous)

    def test_velocity_reports_world_com_motion_and_missing_model_is_unknown(self):
        data = scene.initialize(self.model, self.receipt)
        joint = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_JOINT, "task_object_shoe_free")
        dof = int(self.model.jnt_dofadr[joint])
        data.qvel[dof:dof+6] = [0, 0, 0, 0, 1, 0]
        self.mj.mj_forward(self.model, data)
        measured = scene.sample_state(self.model, data, self.receipt)["objects"]["shoe"]
        self.np.testing.assert_allclose(measured["angular_velocity_rad_s"], [0, 1, 0], atol=1e-12)
        center_z = self.receipt["usd"]["objects"]["shoe"]["center_of_mass_m"][2]
        self.np.testing.assert_allclose(measured["linear_velocity_m_s"], [center_z, 0, 0], atol=1e-12)
        baseline = scene.robot_profiles.build_model("piper", ASSETS, timestep=.001)
        foreign_data = self.mj.MjData(baseline)
        self.mj.mj_forward(baseline, foreign_data)
        result = scene.GoalMonitor(self.receipt).update(baseline, foreign_data)
        self.assertTrue(result["unknown"])
        self.assertFalse(result["success"])

    def test_four_distinct_instances_compile_with_one_unmodified_coupled_gripper(self):
        recipe = scene.default_recipe()
        template = deepcopy(recipe["objects"][0])
        recipe["objects"] = []
        for i, (x, y) in enumerate(((.58, .15), (.58, .25), (.72, .15), (.72, .25))):
            obj = deepcopy(template)
            obj.update(id="block_"+str(i), position_m=[x, y, .06])
            recipe["objects"].append(obj)
        path = self.root/"four"
        scene.write_scene(recipe, path)
        model, receipt = scene.build_scene(path/"recipe.json", ASSETS)
        self.assertEqual((model.nq, model.nv, model.nu, model.neq), (36, 32, 7, 1))
        self.assertTrue(all(receipt["compiled_checks"].values()))
        self.assertEqual(len(scene.sample_state(model, scene.initialize(model, receipt), receipt)["objects"]), 4)

    def test_tamper_fails_before_native_asset_read(self):
        path = self.root/"scene/recipe.json"
        original = path.read_bytes()
        try:
            path.write_bytes(original+b" ")
            with self.assertRaisesRegex(ValueError, "immutable receipt"):
                scene.build_scene(path, "must-not-read")
        finally:
            path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()

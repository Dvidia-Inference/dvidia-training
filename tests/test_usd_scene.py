"""Real OpenUSD parsing, rejection boundaries and native property/goal checks."""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from dvidia_training import usd_scene as bridge


EXAMPLE = Path(__file__).resolve().parents[1]/"examples/usd-workstation-v01"
HAS_USD = importlib.util.find_spec("pxr") is not None


class RecipeTests(unittest.TestCase):
    def test_recipe_import_and_validation_do_not_require_native_or_usd(self):
        code = ("import sys; sys.modules['pxr']=None; sys.modules['mujoco']=None; "
                "from dvidia_training.usd_scene import load_recipe; "
                f"assert load_recipe({str(EXAMPLE/'recipe.json')!r})['robot_profile']=='piper'")
        run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_recipe_digest_catalog_bounds_and_unknown_options(self):
        recipe = bridge.load_recipe(EXAMPLE/"recipe.json")
        for changes in ({"schema": "other"}, {"stage": "../../outside.usda"}, {"robot_profile": "ur5e"},
                        {"catalog_sha256": "0"*64}, {"robot_lock_sha256": "0"*64},
                        {"timestep_s": True}, {"timestep_s": .003}, {"custom_script": "run.py"},
                        {"reset": {**recipe["reset"], "seed": True}},
                        {"reset": {**recipe["reset"], "object_xy_jitter_m": [.03, 0]}},
                        {"goal": {**recipe["goal"], "require_table_contact": False}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                bridge.validate_recipe({**recipe, **changes})


@unittest.skipUnless(HAS_USD, "Install optional usd-core to exercise real USD parsing")
class USDStageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        shutil.copy(EXAMPLE/"workstation.usda", self.root/"workstation.usda")
        self.path = self.root/"workstation.usda"

    def tearDown(self):
        self.tmp.cleanup()

    def edit(self, callback):
        from pxr import Usd
        stage = Usd.Stage.Open(str(self.path))
        callback(stage)
        stage.GetRootLayer().Save()
        del stage

    def test_original_geometry_physics_are_read_from_real_sdk(self):
        result = bridge.inspect_stage(self.path)
        self.assertEqual(result["stage_sha256"], sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(result["table"]["half_size_m"], [.6, .4, .04])
        self.assertEqual(result["object"]["half_size_m"], [.02]*3)
        self.assertAlmostEqual(result["object"]["mass_kg"], .04, places=8)
        self.assertAlmostEqual(result["object"]["diagonal_inertia_kg_m2"][0], .04*(.04**2+.04**2)/12, places=10)
        self.assertEqual(result["object"]["sliding_friction"], .5)

    def test_explicit_units_and_axis_are_required_without_implicit_conversion(self):
        for key, value in (("metersPerUnit", .01), ("kilogramsPerUnit", .001), ("upAxis", "Y")):
            self.edit(lambda stage: stage.SetMetadata(key, value))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "metres and Z-up"):
                bridge.inspect_stage(self.path)
            shutil.copy(EXAMPLE/"workstation.usda", self.path)
        self.edit(lambda stage: stage.ClearMetadata("metersPerUnit"))
        with self.assertRaisesRegex(ValueError, "explicitly"):
            bridge.inspect_stage(self.path)

    def test_missing_mass_and_nonphysical_inertia_fail(self):
        self.edit(lambda stage: stage.GetPrimAtPath("/World/Object").RemoveProperty("physics:mass"))
        with self.assertRaisesRegex(ValueError, "missing or unsupported properties"):
            bridge.inspect_stage(self.path)
        shutil.copy(EXAMPLE/"workstation.usda", self.path)
        from pxr import Gf
        self.edit(lambda stage: stage.GetPrimAtPath("/World/Object").GetAttribute("physics:diagonalInertia").Set(Gf.Vec3f(1, .1, .1)))
        with self.assertRaisesRegex(ValueError, "triangle"):
            bridge.inspect_stage(self.path)

    def test_unknown_shape_and_additional_prim_are_rejected(self):
        self.edit(lambda stage: stage.GetPrimAtPath("/World/Object").SetTypeName("Sphere"))
        with self.assertRaisesRegex(ValueError, "Unsupported USD prim"):
            bridge.inspect_stage(self.path)
        shutil.copy(EXAMPLE/"workstation.usda", self.path)
        self.edit(lambda stage: stage.DefinePrim("/World/Extra", "Scope"))
        with self.assertRaisesRegex(ValueError, "Unsupported USD prim"):
            bridge.inspect_stage(self.path)

    def test_reference_payload_sublayer_and_local_composition_are_rejected(self):
        from pxr import Sdf
        # Sdf authors arcs without opening a Usd stage or resolving their targets.
        for kind in ("reference", "payload", "sublayer", "inherit"):
            layer = Sdf.Layer.FindOrOpen(str(self.path))
            layer.Reload()
            prim = layer.GetPrimAtPath("/World/Object")
            if kind == "reference":
                prim.referenceList.prependedItems = [Sdf.Reference("https://invalid.example/unfetched.usda")]
            elif kind == "payload":
                prim.payloadList.prependedItems = [Sdf.Payload("missing.usda")]
            elif kind == "sublayer":
                layer.subLayerPaths = ["missing.usda"]
            else:
                prim.inheritPathList.prependedItems = [Sdf.Path("/World/Table")]
            layer.Save()
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "references|composition|metadata"):
                bridge.inspect_stage(self.path)
            del layer, prim
            shutil.copy(EXAMPLE/"workstation.usda", self.path)

    def test_unsupported_material_modes_and_time_samples_fail(self):
        self.edit(lambda stage: stage.GetPrimAtPath("/World/ObjectMaterial").GetAttribute("physics:staticFriction").Set(.7))
        with self.assertRaisesRegex(ValueError, "equal static/dynamic"):
            bridge.inspect_stage(self.path)
        shutil.copy(EXAMPLE/"workstation.usda", self.path)
        self.edit(lambda stage: stage.GetPrimAtPath("/World/ObjectMaterial").GetAttribute("physics:restitution").Set(.3))
        with self.assertRaisesRegex(ValueError, "zero restitution"):
            bridge.inspect_stage(self.path)
        shutil.copy(EXAMPLE/"workstation.usda", self.path)
        self.edit(lambda stage: stage.GetPrimAtPath("/World/Object").GetAttribute("size").Set(2., 1.))
        with self.assertRaisesRegex(ValueError, "Time-sampled"):
            bridge.inspect_stage(self.path)

    def test_nonpositive_scale_disabled_collision_and_custom_script_are_rejected(self):
        from pxr import Gf, Sdf
        changes = [lambda stage: stage.GetPrimAtPath("/World/Object").GetAttribute("xformOp:scale").Set(Gf.Vec3d(0, .04, .04)),
                   lambda stage: stage.GetPrimAtPath("/World/Object").GetAttribute("physics:collisionEnabled").Set(False),
                   lambda stage: stage.GetPrimAtPath("/World/Object").CreateAttribute("script", Sdf.ValueTypeNames.String).Set("x")]
        for callback in changes:
            self.edit(callback)
            with self.assertRaises(ValueError):
                bridge.inspect_stage(self.path)
            shutil.copy(EXAMPLE/"workstation.usda", self.path)


@unittest.skipUnless(HAS_USD and os.environ.get("DVIDIA_ROBOT_PROFILE_ASSETS"),
                     "Set DVIDIA_ROBOT_PROFILE_ASSETS and install optional usd-core for native checks")
class NativeUSDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.receipt = bridge.build_scene(EXAMPLE/"recipe.json", os.environ["DVIDIA_ROBOT_PROFILE_ASSETS"])
        cls.recipe = bridge.load_recipe(EXAMPLE/"recipe.json")
        cls.mj, cls.np = bridge.sim._native_imports()

    def test_compiled_usd_properties_and_native_robot_limits_are_preserved(self):
        self.assertTrue(all(self.receipt["compiled_checks"].values()))
        body = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, "usd_object")
        self.np.testing.assert_array_equal(self.model.body_ipos[body], [0, 0, 0])
        self.np.testing.assert_array_equal(self.model.body_iquat[body], [1, 0, 0, 0])
        baseline = bridge.robot_profiles.build_model("piper", os.environ["DVIDIA_ROBOT_PROFILE_ASSETS"], timestep=.001)
        for name in ("actuator_ctrlrange", "actuator_forcerange", "actuator_gainprm", "actuator_biasprm", "actuator_trnid"):
            self.np.testing.assert_array_equal(getattr(self.model, name), getattr(baseline, name))
        for name in ("body_mass", "body_inertia", "body_gravcomp"):
            self.np.testing.assert_array_equal(getattr(self.model, name)[:baseline.nbody], getattr(baseline, name))

    def test_seeded_reset_is_repeatable_and_does_not_hide_simulation_steps(self):
        a = bridge.initialize(self.model, self.recipe, 12)
        b = bridge.initialize(self.model, self.recipe, 12)
        c = bridge.initialize(self.model, self.recipe, 13)
        self.np.testing.assert_array_equal(a.qpos, b.qpos)
        self.assertFalse(self.np.array_equal(a.qpos, c.qpos))
        self.assertEqual(a.time, 0)
        self.assertEqual(a.ncon, 0)

    def test_gravity_contact_and_independent_settling_goal_need_actual_native_time(self):
        data = bridge.initialize(self.model, self.recipe)
        monitor = bridge.GoalMonitor(self.recipe)
        self.assertFalse(monitor.update(self.model, data)["success"])
        for _ in range(2000):
            self.mj.mj_step(self.model, data)
            self.mj.mj_forward(self.model, data)
            result = monitor.update(self.model, data)
        self.assertTrue(result["success"], result)
        stationary = bridge.GoalMonitor(self.recipe)
        for _ in range(100):
            repeated = stationary.update(self.model, data)
        self.assertFalse(repeated["success"], "Repeating one observation cannot create settling time")
        for _ in range(10):
            self.mj.mj_step(self.model, data)
        self.mj.mj_forward(self.model, data)
        self.assertFalse(monitor.update(self.model, data)["success"], "Unobserved intervals must clear dwell")
        self.assertFalse(self.np.any(data.xfrc_applied))
        self.assertFalse(self.np.any(data.qfrc_applied))
        self.assertTrue(all(warning.number == 0 for warning in data.warning))
        wrong = deepcopy(self.recipe)
        wrong["goal"]["object_center_m"] = [.8, -.1, .02]
        self.assertFalse(bridge.GoalMonitor(wrong).update(self.model, data)["instantaneous"])

    def test_stage_digest_tampering_fails_before_native_asset_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copy(EXAMPLE/"workstation.usda", root/"workstation.usda")
            recipe = deepcopy(self.recipe)
            recipe["stage_sha256"] = "0"*64
            (root/"recipe.json").write_text(json.dumps(recipe))
            with self.assertRaisesRegex(ValueError, "pinned bytes"):
                bridge.build_scene(root/"recipe.json", "must-not-read")


if __name__ == "__main__":
    unittest.main()

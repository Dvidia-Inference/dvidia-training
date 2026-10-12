"""Asset evidence boundaries, portable loading and real native contact checks."""
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from dvidia_training import asset_library as library
from dvidia_training import reference_sim as sim
from dvidia_training import robot_profiles


class AssetLibraryContractTests(unittest.TestCase):
    def test_catalog_preserves_robot_axes_source_pins_and_distinct_licenses(self):
        result = library.catalog()
        robots = {item["id"]: item for item in result["robots"]}
        self.assertEqual(set(robots), {"ur5e", "piper", "so101"})
        self.assertEqual([robots[key]["arm_axes"] for key in ("ur5e", "piper", "so101")], [6, 6, 5])
        self.assertEqual(robots["ur5e"]["licenses"], {"universal_robots_ur5e": "BSD-3-Clause", "robotiq_2f85": "BSD-2-Clause"})
        self.assertEqual(robots["piper"]["licenses"], {"agilex_piper": "MIT"})
        self.assertEqual(robots["so101"]["licenses"], {"robotstudio_so101": "Apache-2.0"})
        for robot in robots.values():
            self.assertEqual(robot["source"]["commit"], sim.COMMIT)
            self.assertEqual(len(robot["native_command_units"]), robot["expected_dimensions"]["nu"])
            self.assertTrue(robot["compatibility"]["scene_loading"])
            self.assertFalse(robot["compatibility"]["shoe_grasp_qualified"])
            self.assertFalse(robot["compatibility"]["physical_robot_ready"])
        self.assertEqual(result["catalog_sha256"], sha256(library.CATALOG_PATH.read_bytes()).hexdigest())
        result["shoes"].clear()
        self.assertEqual(len(library.catalog()["shoes"]), 6)

    def test_original_shoes_are_paired_bounded_rigid_proxies_with_untested_properties(self):
        shoes = library.catalog()["shoes"]
        self.assertEqual({item["id"] for item in shoes}, {
            family + "-" + side for family in ("trainer", "ankle-boot", "slipper") for side in ("left", "right")})
        self.assertEqual(len({item["immutable_id"] for item in shoes}), 6)
        self.assertEqual(len({item["pair_id"] for item in shoes}), 3)
        for shoe in shoes:
            self.assertEqual(shoe["license"], "MIT")
            self.assertEqual(shoe["source"]["kind"], "original_procedural")
            self.assertTrue(all(prop["evidence"] == "untested" and prop["source"] and prop["units"]
                                for prop in shoe["properties"].values()))
            self.assertEqual(shoe["fidelity"]["deformation"], "unsupported")
            self.assertFalse(shoe["fidelity"]["shoe_grasp_qualified"])
            self.assertEqual(shoe["frame"]["sole_bottom_z_m"], 0)
            self.assertAlmostEqual(sum(piece["mass_fraction"] for piece in shoe["geometry"]), 1.)
            length, width, height = shoe["properties"]["dimensions_m"]["value"]
            for piece in shoe["geometry"]:
                self.assertIn(piece["type"], ("box", "ellipsoid"))
                self.assertTrue(all(0 < size < 1 for size in piece["size_m"]))
                x, y, z = piece["position_m"]
                sx, sy, sz = piece["size_m"]
                self.assertLessEqual(abs(x) + sx, length / 2 + 1e-12)
                self.assertLessEqual(abs(y) + sy, width / 2 + 1e-12)
                self.assertGreaterEqual(z - sz, -1e-12)
                self.assertLessEqual(z + sz, height + 1e-12)

    def test_catalog_and_scene_descriptors_need_no_native_dependency(self):
        code = ("import sys;sys.modules['mujoco']=None;sys.modules['numpy']=None;"
                "from dvidia_training.asset_library import catalog,scene_config;"
                "assert len(catalog()['shoes'])==6;assert scene_config('piper')['timestep']==.002")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_selection_rejects_unbounded_unknown_and_ambiguous_inputs_before_asset_access(self):
        cases = [{"robot": None}, {"robot": "../piper"}, {"robot": []},
                 {"shoe_ids": "trainer-left"}, {"shoe_ids": []}, {"shoe_ids": {}},
                 {"shoe_ids": ["trainer-left"] * 2}, {"shoe_ids": ["trainer-left"] * 7},
                 {"shoe_ids": ["https://example.com/mesh.xml"]}, {"shoe_ids": [True]},
                 {"timestep": True}, {"timestep": float("nan")}, {"timestep": .003},
                 {"timestep": 10 ** 1000}]
        for changes in cases:
            kwargs = {"robot": "piper", **changes}
            with self.subTest(changes=changes), patch.object(library, "_robot_spec") as native:
                with self.assertRaises(ValueError):
                    library.build_scene(assets="must-not-read", **kwargs)
                native.assert_not_called()

    def test_descriptor_rejects_tampering_unknown_options_and_catalog_drift(self):
        valid = library.scene_config("so101", ["slipper-left"], timestep=.001)
        self.assertEqual(json.loads(json.dumps(valid)), valid)
        bad = [None, [], {**valid, "xml": "<mujoco/>"},
               {**valid, "catalog_sha256": "0" * 64}, {**valid, "schema": "other"},
               {**valid, "catalog_version": "next"}, {**valid, "shoe_ids": ("slipper-left",)}]
        missing = dict(valid)
        missing.pop("catalog_sha256")
        bad.append(missing)
        for config in bad:
            with self.subTest(config=config), patch.object(library, "build_scene") as build:
                with self.assertRaises(ValueError):
                    library.load_scene(config, "must-not-read")
                build.assert_not_called()
        with patch.object(library, "build_scene", return_value="model") as build:
            self.assertEqual(library.load_scene(deepcopy(valid), "local-assets"), "model")
            build.assert_called_once_with("so101", "local-assets", shoe_ids=["slipper-left"], timestep=.001)


@unittest.skipUnless(os.environ.get("DVIDIA_ASSET_UR5E_ROOT") and os.environ.get("DVIDIA_ROBOT_PROFILE_ASSETS"),
                     "Set DVIDIA_ASSET_UR5E_ROOT and DVIDIA_ROBOT_PROFILE_ASSETS to pinned asset roots")
class NativeAssetLibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mj, cls.np = sim._native_imports()
        cls.collection = library.catalog()
        cls.shoes = [item["id"] for item in cls.collection["shoes"]]
        cls.models = {}
        cls.data = {}
        cls.warnings = {}
        cls.baselines = {}
        for robot in ("ur5e", "piper", "so101"):
            root = Path(os.environ["DVIDIA_ASSET_UR5E_ROOT"] if robot == "ur5e" else os.environ["DVIDIA_ROBOT_PROFILE_ASSETS"])
            cls.baselines[robot] = (sim.build_model(root, timestep=.002) if robot == "ur5e"
                                    else robot_profiles.build_model(robot, root, timestep=.002))
            cls.warnings[robot] = []
            with sim.native_warnings(cls.warnings[robot]):
                model = library.load_scene(library.scene_config(robot, cls.shoes), root)
                data = library.initialize(model)
                cls.models[robot], cls.data[robot] = model, data
                for _ in range(1000):
                    cls.mj.mj_step(model, data)
                cls.mj.mj_forward(model, data)

    def test_all_native_arm_mechanics_are_preserved_when_shoes_are_added(self):
        for robot, model in self.models.items():
            baseline = self.baselines[robot]
            self.assertEqual((model.nq, model.nv, model.nu, model.neq, model.nmocap),
                             (baseline.nq + 42, baseline.nv + 36, baseline.nu, baseline.neq, 0))
            for key in ("actuator_ctrlrange", "actuator_forcerange", "actuator_gainprm", "actuator_biasprm", "actuator_trnid"):
                self.np.testing.assert_array_equal(getattr(model, key), getattr(baseline, key))
            for key in ("body_mass", "body_inertia", "body_gravcomp"):
                self.np.testing.assert_array_equal(getattr(model, key)[:baseline.nbody], getattr(baseline, key))
            self.assertEqual(self.warnings[robot], [])
            self.assertTrue(all(warning.number == 0 for warning in self.data[robot].warning))

    def test_all_six_objects_settle_under_gravity_with_actual_support_forces(self):
        for robot, model in self.models.items():
            data = self.data[robot]
            self.assertTrue(all(self.np.isfinite(value).all() for value in (data.qpos, data.qvel, data.qacc)))
            self.assertFalse(self.np.any(data.xfrc_applied))
            self.assertFalse(self.np.any(data.qfrc_applied))
            table = self.mj.mj_name2id(model, self.mj.mjtObj.mjOBJ_GEOM, "asset_table")
            for shoe in self.collection["shoes"]:
                name = "shoe_" + shoe["id"].replace("-", "_")
                body = self.mj.mj_name2id(model, self.mj.mjtObj.mjOBJ_BODY, name)
                joint = self.mj.mj_name2id(model, self.mj.mjtObj.mjOBJ_JOINT, name + "_free")
                self.assertEqual(int(model.jnt_type[joint]), int(self.mj.mjtJoint.mjJNT_FREE))
                self.assertAlmostEqual(model.body_mass[body], shoe["properties"]["mass_kg"]["value"], places=12)
                self.assertLess(abs(data.xpos[body, 2]), .0005)
                dof = model.jnt_dofadr[joint]
                self.assertLess(float(self.np.max(self.np.abs(data.qvel[dof:dof + 6]))), .001)
                normal_force = 0.
                for contact_index in range(data.ncon):
                    contact = data.contact[contact_index]
                    first, second = int(contact.geom1), int(contact.geom2)
                    if (first == table and model.geom_bodyid[second] == body) or (second == table and model.geom_bodyid[first] == body):
                        force = self.np.zeros(6)
                        self.mj.mj_contactForce(model, data, contact_index, force)
                        normal_force += force[0]
                self.assertAlmostEqual(normal_force, model.body_mass[body] * 9.81, delta=.01)

    def test_removing_support_causes_free_fall_instead_of_a_hidden_attachment(self):
        for robot, original in self.models.items():
            # Restore the shared model's table after the control; no subsequent
            # test inherits a disabled contact surface or modified solver state.
            table = self.mj.mj_name2id(original, self.mj.mjtObj.mjOBJ_GEOM, "asset_table")
            before = int(original.geom_contype[table]), int(original.geom_conaffinity[table])
            try:
                original.geom_contype[table] = original.geom_conaffinity[table] = 0
                data = library.initialize(original)
                for _ in range(100):
                    self.mj.mj_step(original, data)
                self.mj.mj_forward(original, data)
                for identifier in self.shoes:
                    body = self.mj.mj_name2id(original, self.mj.mjtObj.mjOBJ_BODY, "shoe_" + identifier.replace("-", "_"))
                    self.assertLess(data.xpos[body, 2], -.10)
                self.assertEqual(data.ncon, 0)
            finally:
                original.geom_contype[table], original.geom_conaffinity[table] = before

    def test_initialization_restores_home_commands_without_settling_objects(self):
        for robot, model in self.models.items():
            item = next(entry for entry in self.collection["robots"] if entry["id"] == robot)
            data = library.initialize(model)
            self.assertEqual(data.time, 0)
            self.assertEqual(data.ncon, 0)
            for joint_name, expected in item["home_joint_positions"].items():
                joint = self.mj.mj_name2id(model, self.mj.mjtObj.mjOBJ_JOINT, joint_name)
                self.assertEqual(data.qpos[model.jnt_qposadr[joint]], expected)
            for identifier in self.shoes:
                body = self.mj.mj_name2id(model, self.mj.mjtObj.mjOBJ_BODY, "shoe_" + identifier.replace("-", "_"))
                self.assertEqual(data.xpos[body, 2], .08)
        with self.assertRaises(ValueError):
            library.initialize(self.baselines["piper"])


if __name__ == "__main__":
    unittest.main()

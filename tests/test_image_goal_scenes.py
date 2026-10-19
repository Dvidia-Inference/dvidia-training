"""Physical and observability controls for the frozen image-goal study."""
from copy import deepcopy
import os
import unittest

from dvidia_training import image_goal_scenes as scenes, reference_sim as sim


class ProtocolTests(unittest.TestCase):
    def test_protocol_is_balanced_explicit_and_independent(self):
        protocol = scenes.protocol()
        self.assertEqual(len(protocol["case_ids"]), 32)
        self.assertEqual(len(set(protocol["case_ids"])), 32)
        self.assertEqual(len(protocol["layouts"]), 4)
        self.assertEqual(len(protocol["categories"]), 8)
        self.assertEqual(protocol["synthetic_shoe"]["width_scale"], .5)
        self.assertIn("not statistical independent", protocol["scope"])
        protocol["criteria"]["position_tolerance_m"] = 9
        self.assertEqual(scenes.protocol()["criteria"]["position_tolerance_m"], .025)

    def test_unknown_fixture_is_rejected_before_loading_assets(self):
        with self.assertRaises(ValueError):
            scenes.build_model("/missing", scenes.LAYOUTS[0], "invented")
        with self.assertRaises(ValueError):
            scenes.build_model("/missing", dict(scenes.LAYOUTS[0], xy_m=[9.,9.]))

    def test_visible_fingers_do_not_prove_a_contact_force_threshold(self):
        labels = scenes.expected_labels("unreleased", {"present": True, "complete": False})
        self.assertEqual(labels["expected_verdict"], {"rgb": "unknown", "rgb_telemetry": "incomplete"})
        self.assertEqual(labels["expected_presence"], {"rgb": "present", "rgb_telemetry": "present"})
        for category in ("missing_target", "wrong_object", "wrong_position", "wrong_orientation"):
            labels = scenes.expected_labels(category, {"present": False, "complete": False})
            self.assertEqual(labels["expected_verdict"]["rgb"], "incomplete")
        with self.assertRaises(ValueError):
            scenes.expected_labels("invented", {"present": True, "complete": True})


@unittest.skipUnless(os.environ.get("DVIDIA_ROBOT_PROFILE_ASSETS"), "Set pinned robot assets")
class NativeFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets = os.environ["DVIDIA_ROBOT_PROFILE_ASSETS"]
        cls.mj, cls.np = sim._native_imports()
        arm = scenes._initial_arm(cls.assets)
        cls.states = {(layout["id"], category): scenes._native_state(
            cls.assets, layout, category, arm)
            for layout in scenes.LAYOUTS for category in scenes.CATEGORIES}

    def test_all_categories_have_measured_physical_truth(self):
        for (layout, category), (model,data,truth,inputs,measured,history) in self.states.items():
            with self.subTest(layout=layout, category=category):
                self.assertEqual(truth["complete"], category in ("complete","occluded"))
                self.assertEqual(truth["present"], category not in ("missing_target","wrong_object"))
                self.assertEqual(sum(w.number for w in data.warning), 0)
                self.assertEqual(model.neq, 1)  # Upstream gripper coupling only.
                self.assertFalse(self.np.any(data.xfrc_applied))
                self.assertFalse(self.np.any(data.qfrc_applied))
                self.assertNotIn("complete", inputs)
                self.assertNotIn("instantaneous_conditions_met", inputs)

    def test_unreleased_is_supported_in_goal_with_real_finger_forces(self):
        for layout in scenes.LAYOUTS:
            model,data,truth,inputs,measured,_ = self.states[(layout["id"],"unreleased")]
            with self.subTest(layout=layout["id"]):
                self.assertTrue(truth["supported"])
                self.assertFalse(truth["released"])
                self.assertLess(truth["position_error_m"], .025)
                self.assertLess(truth["yaw_error_rad"], scenes.CRITERIA["yaw_tolerance_rad"])
                self.assertGreater(measured["robot_contact_force_n"], .01)
                self.assertGreater(measured["table_support_force_n"], .01)
                self.assertEqual(model.actuator_gainprm[6,0], 40.)
                obj=self.mj.mj_name2id(model,self.mj.mjtObj.mjOBJ_BODY,scenes.ws.SHOE)
                self.assertAlmostEqual(model.body_mass[obj], .030)
                self.assertEqual(model.body_gravcomp[obj], 0.)
                self.assertEqual(model.jnt_type[model.body_jntadr[obj]], self.mj.mjtJoint.mjJNT_FREE)

    def test_velocity_alias_keeps_pose_but_changes_truth_and_observation_age(self):
        for layout in scenes.LAYOUTS:
            complete=self.states[(layout["id"],"complete")]
            moving=self.states[(layout["id"],"moving_alias")]
            with self.subTest(layout=layout["id"]):
                self.np.testing.assert_array_equal(complete[1].qpos,moving[1].qpos)
                self.assertTrue(complete[2]["complete"])
                self.assertFalse(moving[2]["complete"])
                self.assertAlmostEqual(moving[2]["linear_speed_m_s"], .12)
                self.assertEqual(moving[3]["native_observation_count"], 0)
                self.assertEqual(moving[3]["observation_window_s"], 0.)

    def test_one_native_tick_violation_invalidates_dwell_and_is_in_extrema(self):
        layout=scenes.LAYOUTS[0]
        model,data,truth,inputs,measured,history=self.states[(layout["id"],"complete")]
        baseline=self.mj.MjData(model)
        baseline.qpos[:]=data.qpos
        baseline.qvel[:]=data.qvel
        baseline.ctrl[:]=data.ctrl
        self.mj.mj_forward(model,baseline)
        obj=self.mj.mj_name2id(model,self.mj.mjtObj.mjOBJ_BODY,scenes.ws.SHOE)
        dof=int(model.jnt_dofadr[int(model.body_jntadr[obj])])
        baseline.qvel[dof]=.12
        self.mj.mj_forward(model,baseline)
        actual_violation=scenes._measure(model,baseline,layout)
        changed=deepcopy(history)
        changed[-25]=actual_violation
        changed_truth,changed_inputs,_=scenes.observe(model,data,layout,changed)
        self.assertFalse(changed_truth["complete"])
        self.assertAlmostEqual(changed_truth["physical_conditions_dwell_s"], .048)
        self.assertGreater(changed_inputs["native_window_extrema"]["linear_speed_m_s"]["max"], .1)
        self.assertEqual(changed_inputs["native_observation_count"], 100)

    @unittest.skipUnless(os.environ.get("DVIDIA_RENDER_MUJOCO"), "Enable native graphics explicitly")
    def test_native_images_preserve_alias_and_fully_occlude_target(self):
        for layout in scenes.LAYOUTS:
            images={}
            for category in ("complete","moving_alias","occluded","missing_target","wrong_object"):
                model,data,*_=self.states[(layout["id"],category)]
                images[category],visibility=scenes.render(model,data,layout)
                with self.subTest(layout=layout["id"],category=category):
                    if category in ("occluded","missing_target","wrong_object"):
                        self.assertEqual(visibility["visible_target_pixels"], 0)
                        self.assertIsNone(visibility["target_box_xyxy_normalized"])
                    else:
                        self.assertGreater(visibility["visible_target_pixels"], 500)
            self.assertEqual(images["complete"],images["moving_alias"])


if __name__=="__main__":
    unittest.main()

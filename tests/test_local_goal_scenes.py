"""Frozen mechanics and declared sampling for fresh native verifier fixtures."""
from copy import deepcopy
import os
import unittest

from dvidia_training import local_goal_scenes as scenes


class LayoutTests(unittest.TestCase):
    def test_seeded_layouts_are_deterministic_and_bounded(self):
        layouts=scenes.make_layouts()
        self.assertEqual(layouts,scenes.make_layouts(20261019,8))
        self.assertEqual(len(layouts),8)
        self.assertEqual(len({row["id"] for row in layouts}),8)
        scenes._validate_layouts(layouts)
        self.assertNotEqual(layouts,scenes.make_layouts(20261020,8))
        for seed,count in ((True,8),(1,True),(-1,8),(1,0),(1,65)):
            with self.assertRaises(ValueError):scenes.make_layouts(seed,count)

    def test_bad_layout_rejected_before_asset_access(self):
        layout=scenes.make_layouts()[0]
        for key,value in (("xy_m",[.8,0.]),("yaw_rad",float("nan")),("id","../escape")):
            bad=deepcopy(layout);bad[key]=value
            with self.assertRaises(ValueError):scenes.build_fresh_model("/missing",bad,"complete")
        with self.assertRaises(ValueError):scenes.build_fresh_model("/missing",layout,"invented")


@unittest.skipUnless(os.environ.get("DVIDIA_ROBOT_PROFILE_ASSETS"),"Set pinned native assets")
class NativeSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets=os.environ["DVIDIA_ROBOT_PROFILE_ASSETS"]
        cls.layout=scenes.make_layouts(20261018,1)[0]
        cls.arm=scenes.frozen._initial_arm(cls.assets)

    def test_moving_intervention_keeps_real_history_without_duplicate_time(self):
        complete=scenes._run(self.assets,self.layout,"complete",self.arm)
        moving=scenes._run(self.assets,self.layout,"moving_alias",self.arm)
        self.assertEqual(complete[1].qpos.tolist(),moving[1].qpos.tolist())
        self.assertTrue(complete[2]["complete"])
        self.assertFalse(moving[2]["complete"])
        history=moving[4]
        self.assertEqual(len(history),100)
        self.assertEqual(history[:-1],complete[4][:-1])
        self.assertEqual(history[-1]["t_s"],complete[4][-1]["t_s"])
        self.assertEqual(len({row["t_s"] for row in history}),100)
        self.assertAlmostEqual(history[-1]["linear_speed_m_s"],.12)
        packet=scenes._packet(history)
        self.assertNotIn("physical_truth",packet)
        self.assertNotIn("instantaneous_conditions_met",packet["samples"][-1])
        self.assertEqual(packet["samples"][-1]["position_m"],history[-1]["target_position_m"])

    def test_reset_relocation_preserves_drives_and_free_body(self):
        mj,np=scenes.sim._native_imports()
        old=scenes.frozen.build_model(self.assets,scenes.frozen.LAYOUTS[0],"complete")
        new=scenes.build_fresh_model(self.assets,self.layout,"complete")
        for field in ("actuator_gainprm","actuator_biasprm","actuator_ctrlrange",
                      "actuator_forcerange","body_mass","body_inertia","body_gravcomp"):
            np.testing.assert_array_equal(getattr(new,field),getattr(old,field))
        self.assertEqual(new.neq,old.neq)
        self.assertEqual(new.neq,1)
        obj=mj.mj_name2id(new,mj.mjtObj.mjOBJ_BODY,scenes.ws.SHOE)
        joint=int(new.body_jntadr[obj]);address=int(new.jnt_qposadr[joint])
        self.assertEqual(new.jnt_type[joint],mj.mjtJoint.mjJNT_FREE)
        np.testing.assert_array_equal(new.qpos0[address:address+2],self.layout["xy_m"])


if __name__=="__main__":unittest.main()

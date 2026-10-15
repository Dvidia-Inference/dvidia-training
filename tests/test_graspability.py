"""Native causal controls for the explicit direct-grasp diagnostic."""
import math
import os
from pathlib import Path
import unittest

from dvidia_training import graspability as grasp, reference_sim as sim


class CaseTests(unittest.TestCase):
    def test_invalid_cases_fail_before_asset_access(self):
        original=grasp.cases()[0]
        for key,value in [('object_shape','unknown'),('mass',0),('friction',float('nan')),
                          ('object_x',.9),('object_y',False),('yaw_rad',1)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                grasp.build_model('/missing',dict(original,**{key:value}))
        with self.assertRaises(ValueError):grasp.build_model('/missing',original,'detach')
        with self.assertRaises(ValueError):grasp.run_case('/missing',original,close_gripper=1)

    def test_development_controls_keep_mass_and_geometry_factors_explicit(self):
        rows=grasp.cases()
        self.assertEqual({r['object_shape'] for r in rows},set(grasp.SHAPES))
        self.assertEqual({r['mass'] for r in rows},{.03,.30})
        self.assertEqual(len(rows),6)
        rows[0]['mass']=42
        self.assertEqual(grasp.cases()[0]['mass'],.03)
        with self.assertRaises(ValueError):grasp.cases('heldout')


@unittest.skipUnless(os.environ.get('DVIDIA_ROBOT_PROFILE_ASSETS'),'Set pinned robot assets')
class NativeGraspTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets=Path(os.environ['DVIDIA_ROBOT_PROFILE_ASSETS'])
        cls.mj,cls.np=sim._native_imports()
        cls.light=grasp.cases()[0]
        cls.results={tool:grasp.run_case(cls.assets,cls.light,tool) for tool in grasp.TOOLS}

    def test_tool_ablation_preserves_drives_and_free_object(self):
        mounted=self.results['mounted']['model'];absent=self.results['absent']['model']
        for field in ('actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange','jnt_range'):
            self.np.testing.assert_array_equal(getattr(mounted,field),getattr(absent,field))
        self.assertEqual(mounted.actuator_gainprm[6,0],40)
        for model in (mounted,absent):
            obj=self.mj.mj_name2id(model,self.mj.mjtObj.mjOBJ_BODY,grasp.ws.SHOE)
            self.assertAlmostEqual(model.body_mass[obj],.03)
            self.assertEqual(model.body_gravcomp[obj],0)
            self.assertEqual(model.jnt_type[model.body_jntadr[obj]],self.mj.mjtJoint.mjJNT_FREE)
            self.assertEqual(model.neq,1)
        self.assertGreaterEqual(self.mj.mj_name2id(mounted,self.mj.mjtObj.mjOBJ_BODY,'reach_rake'),0)
        self.assertEqual(self.mj.mj_name2id(absent,self.mj.mjtObj.mjOBJ_BODY,'reach_rake'),-1)

    def test_small_light_object_lifts_but_open_jaw_control_does_not(self):
        from dvidia_training.graspability_protocol import score_result
        for result in self.results.values():
            self.assertTrue(result['success'])
            self.assertTrue(score_result(result)['task_success'])
            self.assertFalse(result['metrics']['object_external_force_used'])
            self.assertEqual(result['metrics']['object_equality_constraints'],0)
        open_jaw=grasp.run_case(self.assets,self.light,'absent',record=False,close_gripper=False)
        self.assertFalse(open_jaw['success'])
        self.assertFalse(score_result(open_jaw)['task_success'])
        self.assertFalse(open_jaw['metrics']['close_gripper'])

    def test_heavier_matched_box_is_not_mislabeled_as_easy_success(self):
        from dvidia_training.graspability_protocol import score_result
        heavy=grasp.run_case(self.assets,dict(self.light,mass=.30),'absent',record=False)
        self.assertFalse(heavy['success'])
        self.assertFalse(score_result(heavy)['task_success'])
        self.assertEqual(heavy['metrics']['gripper_kp_n_per_m'],40)
        self.assertAlmostEqual(heavy['metrics']['object_weight_n'],2.943)
        self.assertEqual(heavy['metrics']['solver_warnings'],0)

    def test_dense_evidence_clearance_and_commands_are_native_and_bounded(self):
        r=self.results['mounted'];trace=r['native_trace'];columns=r['native_trace_columns']
        self.assertEqual(len(trace),3300)
        self.assertEqual(len(trace),r['metrics']['native_steps'])
        self.assertEqual(trace[0][0],.002);self.assertEqual(trace[-1][0],6.6)
        self.assertLessEqual(r['metrics']['max_joint_command_delta_rad'],.00300000001)
        self.assertEqual(r['metrics']['solver_warnings'],0)
        self.assertTrue(all(math.isfinite(v) for row in trace for v in row))
        reference=[e for e in r['events'] if e['type']=='settled_reference']
        self.assertEqual(len(reference),1);self.assertEqual(reference[0]['t'],.4)
        self.assertLess(abs(reference[0]['object_lowest_z_m']),.001)
        self.assertGreater(min(row[columns.index('object_lowest_z_m')] for row in trace[-250:]),.1)
        for frame in r['frames']:
            self.assertEqual(len(frame['body_poses']),r['model'].nbody)
            self.assertAlmostEqual(frame['jaw_opening_m'],frame['jaw_qpos_m'][0]-frame['jaw_qpos_m'][1])


if __name__=='__main__':unittest.main()

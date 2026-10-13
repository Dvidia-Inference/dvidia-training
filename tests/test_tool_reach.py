"""Causal native checks for the out-of-workspace prerequisite pilot."""
import math
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from dvidia_training import tool_reach as reach, reference_sim as sim


class ReachCaseTests(unittest.TestCase):
    def test_frozen_splits_are_disjoint_and_return_independent_values(self):
        dev, held = reach.cases(), reach.cases('heldout')
        self.assertFalse({x['id'] for x in dev} & {x['id'] for x in held})
        dev[0]['shoe_x']=999
        self.assertLess(reach.cases()[0]['shoe_x'], 1.)
        self.assertTrue(all(x['split']=='heldout' for x in held))
        with self.assertRaises(ValueError): reach.cases('test')

    def test_nonfinite_or_unbounded_case_is_rejected_before_loading_assets(self):
        for key,value in [('mass',-1),('tool_length',float('nan')),('shoe_x',2),('friction',True)]:
            case=dict(reach.cases()[0],**{key:value})
            with self.subTest(key=key),self.assertRaises(ValueError):reach.build_model('/missing',case)
        with self.assertRaises(ValueError):reach.run_case('/missing',reach.cases()[0],'oracle')


@unittest.skipUnless(os.environ.get('DVIDIA_ROBOT_PROFILE_ASSETS'), 'Set DVIDIA_ROBOT_PROFILE_ASSETS to pinned assets')
class NativeReachTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets=Path(os.environ['DVIDIA_ROBOT_PROFILE_ASSETS'])
        cls.mj,cls.np=sim._native_imports()
        cls.standard=reach.run_case(cls.assets,reach.cases()[0],'recovery')

    def test_retrieval_requires_tool_contact_and_free_shoe_is_never_attached(self):
        r=self.standard;m=r['model'];shoe=self.mj.mj_name2id(m,self.mj.mjtObj.mjOBJ_BODY,'shoe_trainer_left')
        self.assertEqual(m.jnt_type[m.body_jntadr[shoe]],self.mj.mjtJoint.mjJNT_FREE)
        self.assertEqual(m.body_gravcomp[shoe],0)
        self.assertEqual(m.neq,1)  # Original gripper coupling, not a shoe constraint.
        self.assertTrue(r['metrics']['retrieval_success'])
        self.assertGreater(r['metrics']['tool_shoe_contact_ticks'],0)
        self.assertGreater(r['metrics']['shoe_displacement_x_m'],.20)
        self.assertFalse(r['metrics']['shoe_external_force_used'])
        self.assertEqual(r['metrics']['shoe_equality_constraints'],0)
        disabled=reach.build_model(self.assets,reach.cases()[0])
        tool=self.mj.mj_name2id(disabled,self.mj.mjtObj.mjOBJ_BODY,'reach_rake')
        disabled.geom_contype[disabled.geom_bodyid==tool]=0
        disabled.geom_conaffinity[disabled.geom_bodyid==tool]=0
        with patch.object(reach,'build_model',return_value=disabled):
            control=reach.run_case(self.assets,reach.cases()[0],'recovery',False)
        self.assertFalse(control['metrics']['retrieval_success'])
        self.assertLess(abs(control['metrics']['shoe_displacement_x_m']),.001)

    def test_no_tool_control_fails_and_short_tool_aborts_without_claiming_success(self):
        baseline=reach.run_case(self.assets,reach.cases()[0],'baseline',False)
        self.assertFalse(baseline['success'])
        self.assertFalse(baseline['metrics']['retrieval_success'])
        self.assertTrue(baseline['metrics']['no_tool_baseline'])
        short=next(c for c in reach.cases() if c['id']=='reach-dev-short')
        result=reach.run_case(self.assets,short,'recovery',False)
        self.assertFalse(result['success']);self.assertEqual(result['metrics']['outcome'],'deferred_insufficient_reach')
        self.assertTrue(any(e.get('reason')=='tool_cannot_reach_far_edge' for e in result['events']))
        self.assertEqual(result['metrics']['tool_shoe_contact_ticks'],0)

    def test_actual_miss_is_detected_before_retry_and_recovers_retrieval(self):
        case=next(c for c in reach.cases() if c['id']=='reach-dev-miss-retry')
        r=reach.run_case(self.assets,case,'recovery',False)
        checks=[e for e in r['events'] if e['type']=='retrieval_check']
        self.assertEqual([e['retrieved'] for e in checks],[False,True])
        self.assertEqual(r['metrics']['recovery_count'],1)
        self.assertTrue(r['metrics']['retrieval_success'])
        self.assertFalse(r['success'])  # Retrieval does not imply retained lift.

    def test_lift_metric_requires_contact_retention_not_height_alone(self):
        self.assertFalse(self.standard['success'])
        self.assertFalse(self.standard['metrics']['retained_lift_success'])
        self.assertGreater(self.standard['metrics']['peak_lift_m'],.005)
        heavy=next(c for c in reach.cases() if c['id']=='reach-dev-heavy')
        # Retain the observed false-positive as a regression: the original low
        # lift tilted the shoe while the toe still received table support.
        with patch.object(reach,'LIFT_HEIGHT',.20):
            low=reach.run_case(self.assets,heavy,'recovery')
        self.assertFalse(low['success'])
        self.assertGreater(low['metrics']['peak_lift_m'],.07)
        self.assertTrue(any(row[-1]>.01 for row in low['native_trace'][-250:]))
        r=reach.run_case(self.assets,heavy,'recovery')
        self.assertTrue(r['success'])
        for row in r['native_trace'][-250:]:
            self.assertGreater(row[7],.035)
            self.assertGreater(min(row[3:5]),.01)
            self.assertLess(row[1],.01)
            self.assertLess(row[-1],.01)

    def test_native_state_finite_commands_bounded_and_replay_deterministic(self):
        r=self.standard
        self.assertEqual(r['metrics']['solver_warnings'],0)
        self.assertLessEqual(r['metrics']['max_joint_command_delta_rad'],.00300000001)
        self.assertEqual(r['metrics']['physics_timestep_s'],.002)
        self.assertEqual(len(r['native_trace']),r['metrics']['native_steps'])
        self.assertAlmostEqual(max(row[2] for row in r['native_trace']),r['metrics']['peak_contact_force_n'])
        self.assertEqual(r['metrics']['policy_interval_s'],.02)
        for frame in r['frames']:
            self.assertTrue(all(math.isfinite(v) for row in frame['body_poses'] for v in row))
            self.assertEqual(len(frame['body_poses']),r['model'].nbody)
        again=reach.run_case(self.assets,reach.cases()[0],'recovery')
        self.assertEqual(r['events'],again['events'])
        self.assertEqual(r['frames'],again['frames'])

if __name__=='__main__': unittest.main()

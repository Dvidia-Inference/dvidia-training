"""Frozen inputs, native passive-hinge dynamics and failure/recovery evidence."""
import math
import os
from pathlib import Path
import unittest

from dvidia_training import hinged_shoebox as box, workstation as ws, reference_sim as sim


class HingedBoxProtocolTests(unittest.TestCase):
    def test_cases_are_independent_frozen_and_split_specific(self):
        development=box.cases();heldout=box.cases('heldout')
        self.assertEqual(len(development),4)
        self.assertEqual(len(heldout),4)
        self.assertFalse({c['id'] for c in development}&{c['id'] for c in heldout})
        development[0]['lid_mass_kg']=99
        self.assertEqual(box.cases()[0]['lid_mass_kg'],.12)
        for invalid in ('test','all',None):
            with self.assertRaises(ValueError):box.cases(invalid)
        for invalid in (None,{},development[0],{**box.cases()[0],'extra':True}):
            with self.assertRaises(ValueError):box._validate_case(invalid)

    def test_guard_uses_motion_and_robot_volume_not_case_name(self):
        self.assertTrue(box.risk_observation(math.radians(75),-.1,box.START))
        self.assertFalse(box.risk_observation(math.radians(75),.1,box.START))
        self.assertFalse(box.risk_observation(math.radians(118),-.1,box.START))
        self.assertFalse(box.risk_observation(math.radians(75),-.1,[.36,-.40,.2]))
        self.assertFalse(box.risk_observation(math.radians(75),-.1,[.70,-.20,.2]))
        self.assertFalse(box.risk_observation(math.radians(75),-.1,[.36,-.20,.50]))

    def test_policy_rejects_unknown_before_loading_assets(self):
        with self.assertRaises(ValueError):
            box.run_case('/missing',box.cases()[0],policy='learned')
        with self.assertRaises(ValueError):
            box.run_case('/missing',box.cases()[0],record='yes')


@unittest.skipUnless(os.environ.get('DVIDIA_ROBOT_PROFILE_ASSETS'),'Set DVIDIA_ROBOT_PROFILE_ASSETS to pinned PiPER assets')
class NativeHingedBoxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets=Path(os.environ['DVIDIA_ROBOT_PROFILE_ASSETS'])
        cls.mj,cls.np=sim._native_imports()
        cls.results={}
        for case in box.cases():
            for policy in ('baseline','recovery'):
                cls.results[(case['id'],policy)]=box.run_case(cls.assets,case,policy)

    def test_exact_robot_actuators_and_unactuated_native_hinge(self):
        baseline=ws.build_model(self.assets)
        model=self.results[('open-light','baseline')]['model']
        self.assertEqual((model.nq,model.nv,model.nu,model.neq,model.nmocap),(16,15,7,1,0))
        for key in ('actuator_ctrlrange','actuator_forcerange','actuator_gainprm','actuator_biasprm','actuator_trnid'):
            self.np.testing.assert_array_equal(getattr(model,key),getattr(baseline,key))
        hinge=model.joint(box.HINGE)
        self.assertEqual(int(model.jnt_type[hinge.id]),int(self.mj.mjtJoint.mjJNT_HINGE))
        self.np.testing.assert_allclose(model.jnt_range[hinge.id],[0,math.radians(120)])
        self.assertNotIn(hinge.id,model.actuator_trnid[:,0])
        self.assertEqual(float(model.body_gravcomp[model.body(box.LID).id]),0.)

    def test_gravity_closes_lid_and_guard_counts_safe_abort_not_task_success(self):
        baseline=self.results[('falling-light','baseline')]
        recovery=self.results[('falling-light','recovery')]
        self.assertTrue(baseline['valid'] and recovery['valid'])
        self.assertTrue(baseline['metrics']['lid_robot_collision'])
        self.assertTrue(any(e['type']=='hazardous_contact' for e in baseline['events']))
        self.assertFalse(recovery['success'])
        self.assertTrue(recovery['metrics']['safe_abort'])
        self.assertTrue(recovery['metrics']['help_requested'])
        self.assertFalse(recovery['metrics']['lid_robot_collision'])
        self.assertLess(recovery['frames'][-1]['lid_angle_rad'],math.radians(12))
        self.assertEqual(recovery['metrics']['outcome'],'safe_abort')
        self.assertLessEqual(recovery['metrics']['first_response_latency_s'],box.CONTROL_DT)

    def test_open_box_completes_and_transient_can_resume_without_false_success(self):
        for policy in ('baseline','recovery'):
            result=self.results[('open-light',policy)]
            self.assertTrue(result['success'])
            self.assertFalse(result['metrics']['help_requested'])
        result=self.results[('transient-closing','recovery')]
        self.assertTrue(result['success'])
        self.assertEqual(result['metrics']['resumed_count'],1)
        kinds=[e['type'] for e in result['events']]
        self.assertLess(kinds.index('withdrawal_started'),kinds.index('access_resumed'))
        self.assertLess(kinds.index('access_resumed'),kinds.index('inspection_waypoint_reached'))
        self.assertEqual(result['metrics']['outcome'],'recovered')
        # Baseline also succeeds here: our conservative interruption is costly,
        # not a task-success improvement. This deliberate tradeoff stays visible.
        self.assertTrue(self.results[('transient-closing','baseline')]['success'])

    def test_every_native_step_is_observed_and_peak_matches_dense_trace(self):
        result=self.results[('falling-heavy','baseline')]
        trace=self.np.asarray(result['native_trace'])
        self.assertEqual(len(trace),int(box.DURATION/box.TIMESTEP)+1)
        self.np.testing.assert_allclose(self.np.diff(trace[:,0]),box.TIMESTEP,atol=1e-12)
        self.assertAlmostEqual(float(trace[:,3].max()),result['metrics']['peak_lid_robot_normal_force_n'])
        self.assertTrue(self.np.isfinite(trace).all())
        self.assertEqual(len(result['frames']),int(box.DURATION/box.CONTROL_DT)+1)
        for frame in result['frames']:
            self.assertTrue(self.np.isfinite(frame['body_poses']).all())
            self.assertEqual(len(frame['body_poses']),result['model'].nbody)
            for key,size in (('qpos_rad_m',16),('qvel_rad_m_s',15),('issued_ctrl',7),('requested_joint_target',7)):
                self.assertEqual(len(frame[key]),size)
                self.assertTrue(self.np.isfinite(frame[key]).all())
            self.assertAlmostEqual(frame['qpos_rad_m'][-1],frame['lid_angle_rad'])
            self.assertAlmostEqual(frame['qvel_rad_m_s'][-1],frame['lid_velocity_rad_s'])
        # This contact force peak falls between 20 ms frames; dense observation
        # protects precisely the failure information sparse episode tapes lose.
        self.assertGreater(result['metrics']['peak_lid_robot_normal_force_n'],max(f['lid_robot_force_n'] for f in result['frames']))
        self.assertAlmostEqual(result['metrics']['peak_lid_robot_normal_force_n'],max(f['native_window_peak_force_n'] for f in result['frames']))


if __name__=='__main__':unittest.main()

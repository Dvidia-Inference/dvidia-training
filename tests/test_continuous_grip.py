"""Native regression checks for authored continuous grip, on unchanged physics."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from dvidia_training import continuous_grip as grip, tool_reach as v01


class ContractTests(unittest.TestCase):
    def test_baseline_is_original_mounted_rake_not_no_tool_control(self):
        receipt={'task':'tool-reach','policy':'recovery','metrics':{},'success':False}
        case=v01.cases()[0]
        with patch.object(v01,'run_case',return_value=receipt) as run:
            actual=grip.run_case('/assets',case,'baseline',False)
        run.assert_called_once_with('/assets',case,policy='recovery',record=False)
        self.assertEqual(actual['policy'],'baseline')
        self.assertIn('not original no-tool',actual['metrics']['comparison_scope'])

    def test_new_holdout_owned_by_independent_protocol(self):
        with self.assertRaises(ValueError):grip.cases('heldout')
        self.assertTrue(all(c['split']=='regression' for c in grip.cases('regression')))
        with self.assertRaises(ValueError):grip.run_case('/assets',v01.cases()[0],'oracle')

    def test_quaternion_is_normalized_and_does_not_modify_workstation_orientation(self):
        import math
        old=list(grip.ws.ORIENTATION)
        for yaw,pitch,roll in ((0,150,0),(.4,140,0),(-.2,150,55)):
            q=grip._quaternion(yaw,pitch,roll)
            self.assertAlmostEqual(sum(x*x for x in q),1.)
        self.assertEqual(old,list(grip.ws.ORIENTATION))
        self.assertTrue(all(math.isclose(x,y,abs_tol=1e-6) for x,y in zip(old,grip._quaternion(0))))


@unittest.skipUnless(os.environ.get('DVIDIA_ROBOT_PROFILE_ASSETS'),'Pinned native assets required')
class NativeContinuousGripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets=Path(os.environ['DVIDIA_ROBOT_PROFILE_ASSETS'])
        cls.heavy=next(c for c in v01.cases() if c['id']=='reach-dev-heavy')
        cls.result=grip.run_case(cls.assets,cls.heavy,'recovery')

    def test_physics_identical_to_original_mounted_tool(self):
        import numpy as np
        original=v01.build_model(self.assets,self.heavy,'recovery');current=self.result['model']
        for name in ('body_mass','body_inertia','geom_friction','geom_size','geom_pos','actuator_gainprm','actuator_biasprm',
                     'actuator_forcerange','actuator_ctrlrange','eq_data','jnt_range'):
            np.testing.assert_array_equal(getattr(current,name),getattr(original,name),err_msg=name)
        self.assertEqual(current.neq,original.neq)
        self.assertEqual(current.nmocap,0)
        self.assertFalse(self.result['metrics']['shoe_external_force_used'])

    def test_known_retention_survives_continuous_checkpoint_and_strict_scoring(self):
        result=self.result
        self.assertTrue(result['valid']);self.assertTrue(result['success'])
        self.assertEqual(result['metrics']['retention_claim_scope'],'strict')
        checkpoint=next(e for e in result['events'] if e['type']=='test_lift_check')
        self.assertTrue(checkpoint['passed']);self.assertGreaterEqual(checkpoint['tcp_rise_m'],.04)
        self.assertFalse(any(e['type']=='grip_loss_detected' for e in result['events']))
        for row in result['native_trace'][-250:]:
            self.assertGreater(row[7],.035);self.assertGreater(min(row[3:5]),.01)
            self.assertLess(row[1],.01);self.assertLess(row[8],.01)
            self.assertLess(row[2]-sum(row[3:5])-row[1],.01)
        self.assertAlmostEqual(result['metrics']['simulation_seconds'],12.6)

    def test_failed_first_grip_lowered_on_real_support_with_bounded_commands(self):
        result=grip.run_case(self.assets,v01.cases()[0],'recovery',True)
        events=result['events'];loss=next(e for e in events if e['type']=='grip_loss_detected')
        recovery=next(e for e in events if e['type']=='grip_recovery_started')
        issued=next(e for e in events if e['type']=='recovery_control_issued')
        self.assertGreaterEqual(recovery['t'],loss['t']);self.assertGreaterEqual(issued['t'],recovery['t'])
        self.assertEqual(issued['requested_at_s'],recovery['t'])
        for event in events:
            if event['type']=='phase' and event.get('name')=='open_on_support':
                row=min(result['native_trace'],key=lambda r:abs(r[0]-event['t']))
                self.assertGreater(row[8],.01)
        self.assertLessEqual(result['metrics']['max_joint_command_delta_rad'],.0030000001)
        self.assertLessEqual(result['metrics']['grasp_attempts'],2)
        self.assertLessEqual(result['metrics']['simulation_seconds'],grip.MAX_SECONDS+.002)
        self.assertFalse(result['metrics']['shoe_external_force_used'])
        self.assertEqual(len(result['native_trace']),result['metrics']['native_steps'])
        self.assertTrue(all(len(f['issued_ctrl'])==7 for f in result['frames']))

    def test_cleanup_time_is_reserved_before_a_new_attempt(self):
        episode=grip.Episode(self.assets,self.heavy,False)
        episode.data.time=grip.MAX_SECONDS-grip.ATTEMPT_RESERVE_SECONDS+.01
        self.assertFalse(episode.grasp())
        self.assertEqual(episode.steps,0)
        self.assertEqual(episode.grasp_attempts,0)
        self.assertEqual(episode.reason,'bounded_episode_budget')


if __name__=='__main__':unittest.main()

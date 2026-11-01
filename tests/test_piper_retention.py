"""Synthetic retention-controller boundaries; no native physics or hardware."""
from copy import deepcopy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dvidia_training import piper_retention as hold
from dvidia_training import piper_task


class Environment:
    def __init__(self):
        self.config = SimpleNamespace(control_dt=.02, timeout_s=20., tool_orientation_wxyz=[1.,0.,0.,0.],
                                      goal_position_tolerance_m=.025)
        self.model = SimpleNamespace(actuator_ctrlrange=[[-2.,2.]]*6+[[0.,.035]])
        self.requests, self.raw = [], None
        self.observation = None

    def pose_action(self, position, quaternion, jaw):
        self.requests.append((list(position),list(quaternion),jaw))
        # A nonzero moving-goal servo offset becomes zero only at the exact
        # admission pose, making issued and measured holds meaningfully differ.
        offset = .03 if math.dist(position,self.observation['tcp_position_m']) > 1e-10 else 0.
        return {'joint_targets_rad': list(self.raw) if self.raw is not None else [q+offset for q in self.observation['joint_position_rad']],
                'gripper_position_m': .012}


class Teacher:
    def __init__(self, env):
        self.env, self.phase, self.issued_phase, self.ticks = env, 'close', 'close', 35
        self.held_offset, self.grasp_position, self.calls = None, [.3,0.,.082], []

    def __call__(self, observation):
        if self.phase not in ('approach','descend','close'):
            raise AssertionError('Full transport must not be dispatched')
        self.calls.append(self.phase)
        self.issued_phase = self.phase
        self.phase, self.ticks = 'lift', 0
        return {'joint_targets_rad': [.05]*6, 'gripper_position_m': .012}


def observation(t, *, rise=None, force=.4, supported=False, both=True, slip=0., joints=.02, tcp_offset=0.):
    rise = max(0.,min(.01,.02*(t-.02))) if rise is None else rise
    return {'t_seconds':t,'joint_position_rad':[joints]*6,'gripper_joint_position_m':[.02,-.02],
            'tcp_position_m':[.3,0.,.082+rise+tcp_offset],'tcp_quaternion_wxyz':[1.,0.,0.,0.],
            'object_position_m':[.3+slip,0.,.08+rise+tcp_offset], 'target_position_m':[.35,.08,.08],
            'grasp_contact_counts':{'left':int(both),'right':int(both)},
            'pad_normal_forces_n':{'left':force,'right':force},'table_object_contact_count':int(supported),
            'unexpected_contact_count':0,'native_actuator_controls':[0.]*6+[.027]}


def make(variant):
    env = Environment()
    with patch.object(piper_task,'PiperTaskPolicy',Teacher):value=hold.RetentionPolicy(env,variant)
    return value,env


def call(value,env,obs):
    env.observation = obs
    return value(obs)


def admitted(value,env):
    for index in range(40):
        action=call(value,env,observation(index*.02))
        if value.admitted:return action
    raise AssertionError('Bounded synthetic fixture did not admit')


class RetentionTests(unittest.TestCase):
    def test_contract_has_fixed_pretrial_limits_and_explicit_scope(self):
        contract=hold.retention_contract()
        self.assertEqual(contract['variants'],['measured_joint','issued_joint','cartesian'])
        self.assertEqual(contract['limits']['joint_correction_ceiling_rad'],.03)
        self.assertEqual(contract['limits']['joint_correction_rate_rad_s'],.10)
        self.assertFalse(contract['physical_robot_ready'])
        for variant in ('evidence',True,None):
            with self.assertRaises(ValueError):make(variant)

    def test_prefix_is_exactly_equal_until_admission_and_no_full_teacher_dispatch(self):
        values=[make(variant) for variant in hold.VARIANTS]
        for index in range(40):
            results=[call(value,env,observation(index*.02)) for value,env in values]
            if any(value.admitted for value,_ in values):
                self.assertTrue(all(value.admitted for value,_ in values));break
            self.assertEqual(results[0],results[1]);self.assertEqual(results[0],results[2])
        self.assertTrue(all(value.teacher.calls==['close'] for value,_ in values))
        for value,_ in values:
            self.assertEqual(value.admission_time_s,values[0][0].admission_time_s)

    def test_measured_and_issued_freeze_distinct_actual_targets_without_jaw_change(self):
        for variant in ('measured_joint','issued_joint'):
            value,env=make(variant)
            action=admitted(value,env)
            expected=([.02]*6 if variant=='measured_joint' else value.hold_reference['issued_joint_targets_rad'])
            self.assertEqual(action['joint_targets_rad'],expected)
            self.assertEqual(action['gripper_position_m'],.012)
            later=call(value,env,observation(value.last_observed_s+.02,joints=.04))
            self.assertEqual(later,action)
            if variant=='issued_joint':
                self.assertNotEqual(action['joint_targets_rad'],[.02]*6)

    def test_cartesian_admission_is_bumpless_then_correction_has_fixed_bias_and_rate(self):
        value,env=make('cartesian');action=admitted(value,env)
        reference=value.hold_reference
        self.assertEqual(action['joint_targets_rad'],reference['issued_joint_targets_rad'])
        self.assertEqual(action['gripper_position_m'],.012)
        env.raw=[.03]*6
        corrected=call(value,env,observation(value.last_observed_s+.02))
        self.assertEqual(corrected['joint_targets_rad'],[baseline+.002 for baseline in reference['issued_joint_targets_rad']])
        self.assertEqual(corrected['gripper_position_m'],.012)
        record=value.hold_records[-1]
        self.assertEqual(record['correction_rate_limited_axes'],list(range(6)))
        self.assertEqual(record['correction_rad'],[.002]*6)
        self.assertEqual(record['reference_position_m'],reference['position_m'])
        self.assertEqual(record['fixed_bias_rad'],reference['fixed_bias_rad'])

    def test_original_loss_and_force_ceiling_terminally_refuse_without_requalification(self):
        for mutation,reason in (({'both':False},'bilateral_contact_or_force_loss'),
                                ({'supported':True},'support_returned'),
                                ({'force':10.0001},'declared_force_ceiling_exceeded'),
                                ({'slip':.0031},'relative_slip')):
            for variant in hold.VARIANTS:
                value,env=make(variant);admitted(value,env)
                action=call(value,env,observation(value.last_observed_s+.02,joints=.04,**mutation))
                self.assertFalse(value.admitted)
                self.assertEqual(value.abort_reason,reason)
                self.assertEqual(action,{'joint_targets_rad':[.04]*6,'gripper_position_m':.012})
                again=call(value,env,observation(value.last_observed_s+.02))
                self.assertEqual(again,action)
                self.assertFalse(value.admitted)

    def test_force_ceiling_also_stops_acquisition_before_teacher(self):
        value,env=make('cartesian')
        action=call(value,env,observation(0.,force=10.001))
        self.assertEqual(value.teacher.calls,[])
        self.assertEqual(action,{'joint_targets_rad':[.02]*6,'gripper_position_m':.027})
        self.assertEqual(value.abort_reason,'declared_force_ceiling_exceeded')

    def test_terminal_records_do_not_relabel_old_ik_or_rate_flags_as_current(self):
        value,env=make('cartesian');admitted(value,env)
        env.raw=[.03]*6
        call(value,env,observation(value.last_observed_s+.02))
        self.assertTrue(value.hold_records[-1]['correction_rate_limited'])
        call(value,env,observation(value.last_observed_s+.02,force=10.001))
        for record in (value.hold_records[-1],):
            self.assertIsNone(record['raw_pose_joint_targets_rad'])
            self.assertIsNone(record['desired_correction_rad'])
            self.assertFalse(record['correction_rate_limited'])
        call(value,env,observation(value.last_observed_s+.02))
        self.assertIsNone(value.hold_records[-1]['raw_pose_joint_targets_rad'])
        self.assertFalse(value.hold_records[-1]['correction_rate_limited'])

    def test_position_orientation_correction_and_initial_bias_exhaustion_refuse(self):
        for bound in ('position','orientation','correction','initial_bias'):
            value,env=make('cartesian')
            action=admitted(value,env) if bound!='initial_bias' else None
            if bound=='initial_bias':
                for index in range(40):
                    if value.qualification_since is not None and index*.02-value.qualification_since >= .1-1e-9:
                        env.raw=[-.2]*6
                    action=call(value,env,observation(index*.02))
                    if value.aborted:break
                expected='initial_bias_exhausted'
            else:
                obs=observation(value.last_observed_s+.02)
                if bound=='position':
                    obs=observation(obs['t_seconds'],tcp_offset=.0041)
                    expected='tcp_position_error_exhausted'
                elif bound=='orientation':
                    angle=.051
                    obs['tcp_quaternion_wxyz']=[math.cos(angle/2),math.sin(angle/2),0.,0.]
                    expected='tcp_orientation_error_exhausted'
                else:
                    env.raw=[.051]*6
                    expected='joint_correction_exhausted'
                action=call(value,env,obs)
            with self.subTest(bound=bound):
                self.assertFalse(value.admitted)
                self.assertEqual(value.abort_reason,expected)
                self.assertEqual(action['joint_targets_rad'],[.02]*6)
                self.assertEqual(action['gripper_position_m'],.012)

    def test_native_margin_exhaustion_refuses_instead_of_clipping(self):
        value,env=make('cartesian');admitted(value,env)
        value.joint_limits=[[.001,.041]]*6
        env.raw=[.022]*6
        action=call(value,env,observation(value.last_observed_s+.02))
        self.assertEqual(value.abort_reason,'native_joint_margin_exhausted')
        self.assertFalse(value.admitted)
        self.assertEqual(action['joint_targets_rad'],[.02]*6)

    def test_invalid_measurement_refuses_without_new_command_or_untrusted_hold(self):
        for bad in ({'joint_position_rad':[float('nan')]*6},{'t_seconds':.01}):
            value,env=make('cartesian');admitted(value,env)
            obs=observation(value.last_observed_s+.02);obs.update(bad)
            before=len(value.requested_controls)
            with self.assertRaises(ValueError):call(value,env,obs)
            self.assertEqual(len(value.requested_controls),before)
            self.assertFalse(value.admitted)
            self.assertEqual(value.status,'invalid_measurement_refused')
            self.assertFalse(value.input_refusals[-1]['command_issued'])
            with self.assertRaises(ValueError):call(value,env,observation(value.last_observed_s+.02))

    def test_final_observation_checks_force_and_error_without_another_solve_or_command(self):
        for mutation,reason in (({'force':10.001},'declared_force_ceiling_exceeded'),
                                ({'tcp_offset':.0041},'tcp_position_error_exhausted'),
                                ({},None)):
            value,env=make('cartesian');admitted(value,env)
            requests,commands=len(env.requests),len(value.requested_controls)
            result=value.observe_final(observation(value.last_observed_s+.02,**mutation))
            self.assertEqual(len(env.requests),requests)
            self.assertEqual(len(value.requested_controls),commands)
            self.assertEqual(result['abort_reason'],reason)
            self.assertEqual(result['admitted'],reason is None)
            self.assertTrue(result['retention']['hold_records'][-1]['final_observation'])
            self.assertFalse(result['retention']['hold_records'][-1]['command_issued'])

    def test_summary_records_are_detached(self):
        value,env=make('cartesian');admitted(value,env)
        summary=value.summary()
        self.assertEqual(summary['variant'],'cartesian')
        self.assertEqual(summary['acquisition_variant'],'evidence')
        summary['retention']['hold_reference']['position_m'][0]=99.
        summary['retention']['hold_records'][0]['joint_targets_rad'][0]=99.
        self.assertNotEqual(value.hold_reference['position_m'][0],99.)
        self.assertNotEqual(value.hold_records[0]['joint_targets_rad'][0],99.)


if __name__=='__main__':unittest.main()

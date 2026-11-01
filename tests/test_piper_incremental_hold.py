"""Pure causal controller tests using a simple Cartesian joint kinematic model."""
from copy import deepcopy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dvidia_training import piper_incremental_hold as incremental
from dvidia_training import piper_retention as retention
from dvidia_training import piper_task


class KinematicEnvironment:
    """One metre per radian for translation axes; no simulator is constructed."""

    def __init__(self):
        self.config = SimpleNamespace(control_dt=.02, timeout_s=20.,
                                      tool_orientation_wxyz=[1., 0., 0., 0.],
                                      goal_position_tolerance_m=.025)
        self.model = SimpleNamespace(actuator_ctrlrange=[[-2., 2.]]*6+[[0., .035]])
        self.observation, self.raw_override = None, None
        self.requests = []

    def pose_action(self, position, quaternion, jaw):
        self.requests.append((list(position), list(quaternion), jaw))
        if isinstance(self.raw_override, Exception):
            raise self.raw_override
        if self.raw_override is not None:
            return {'joint_targets_rad': deepcopy(self.raw_override), 'gripper_position_m': .012}
        targets = list(self.observation['joint_position_rad'])
        # Exact inverse of x(q + dq) = x(q) + dq for the first three joints.
        for axis in range(3):
            targets[axis] += position[axis]-self.observation['tcp_position_m'][axis]
        desired_roll = 2*math.atan2(quaternion[1], quaternion[0])
        measured_quat = self.observation['tcp_quaternion_wxyz']
        targets[3] += desired_roll-2*math.atan2(measured_quat[1], measured_quat[0])
        # Reproduce the real pose solver's absolute target margin. Outside it,
        # even zero pose error can produce raw target minus measured q != 0.
        targets = [max(low+.02, min(high-.02, target))
                   for target, (low, high) in zip(targets, self.model.actuator_ctrlrange[:6])]
        return {'joint_targets_rad': targets, 'gripper_position_m': .012}


class Teacher:
    def __init__(self, env):
        self.env, self.phase, self.issued_phase, self.ticks = env, 'close', 'close', 35
        self.held_offset, self.grasp_position, self.calls = None, [.3, 0., .082], []

    def __call__(self, observation):
        if self.phase not in ('approach', 'descend', 'close'):
            raise AssertionError('Full transport must not be dispatched')
        self.calls.append(self.phase)
        self.issued_phase = self.phase
        self.phase, self.ticks = 'lift', 0
        return {'joint_targets_rad': [.05]*6, 'gripper_position_m': .012}


def observation(t, *, rise=None, displacement=0., joint_drift=0., force=.4,
                both=True, supported=False, slip=0., roll=0.):
    rise = max(0., min(.01, .02*(t-.02))) if rise is None else rise
    return {
        't_seconds': t, 'joint_position_rad': [.02+joint_drift]+[.02]*5,
        'gripper_joint_position_m': [.02, -.02],
        'tcp_position_m': [.3+displacement, 0., .082+rise],
        'tcp_quaternion_wxyz': [math.cos(roll/2), math.sin(roll/2), 0., 0.],
        'object_position_m': [.3+displacement+slip, 0., .08+rise],
        'target_position_m': [.35, .08, .08],
        'grasp_contact_counts': {'left': int(both), 'right': int(both)},
        'pad_normal_forces_n': {'left': force, 'right': force},
        'table_object_contact_count': int(supported), 'unexpected_contact_count': 0,
        'native_actuator_controls': [0.]*6+[.027],
    }


def make(variant='incremental'):
    env = KinematicEnvironment()
    with patch.object(piper_task, 'PiperTaskPolicy', Teacher):
        value = (incremental.IncrementalHoldPolicy(env) if variant == 'incremental'
                 else retention.RetentionPolicy(env, variant))
    return value, env


def call(value, env, obs):
    env.observation = obs
    return value(obs)


def admit(value, env):
    for index in range(40):
        action = call(value, env, observation(index*.02))
        if value.admitted:
            return action
    raise AssertionError('Kinematic fixture did not admit')


def next_observation(value, **kwargs):
    return observation(value.last_observed_s+.02,
                       rise=value.hold_reference['position_m'][2]-.082, **kwargs)


class IncrementalHoldTests(unittest.TestCase):
    def assertVectorAlmostEqual(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, places=12)

    def test_contract_fixes_one_candidate_gain_and_original_limits(self):
        contract = incremental.incremental_hold_contract()
        self.assertEqual(contract['variants'], ['incremental'])
        self.assertEqual(contract['update_gain_per_boundary'], 1.)
        self.assertEqual(contract['limits'], retention.retention_contract()['limits'])
        self.assertEqual(contract['control_dt_s'], .02)
        self.assertFalse(contract['physical_robot_ready'])
        self.assertEqual(retention.VARIANTS, ('measured_joint', 'issued_joint', 'cartesian'))

    def test_common_acquisition_prefix_and_exact_bumpless_admission(self):
        policies = [make(variant) for variant in (*retention.VARIANTS, 'incremental')]
        previous = None
        for tick in range(40):
            actions = [call(value, env, observation(tick*.02)) for value, env in policies]
            if any(value.admitted for value, _ in policies):
                self.assertTrue(all(value.admitted for value, _ in policies))
                self.assertEqual(actions[-1], previous[-1])
                self.assertEqual(actions[-1], actions[1])
                break
            self.assertTrue(all(action == actions[0] for action in actions))
            previous = actions
        else:
            self.fail('All variants must admit')
        value = policies[-1][0]
        self.assertEqual(value.correction, [0.]*6)
        self.assertEqual(value.teacher.calls, ['close'])
        self.assertEqual(value.hold_records[-1]['variant'], 'incremental')
        self.assertFalse(value.hold_records[-1]['correction_update_accepted'])
        self.assertEqual(value.events[-1]['diagnostic']['decision'], 'bounded_incremental_hold')

    def test_kinematic_error_produces_opposing_signed_target_correction(self):
        for displacement in (-.001, .001):
            with self.subTest(displacement=displacement):
                value, env = make()
                first = admit(value, env)
                action = call(value, env, next_observation(value, displacement=displacement,
                                                         joint_drift=displacement))
                correction = action['joint_targets_rad'][0]-first['joint_targets_rad'][0]
                self.assertLess(displacement*correction, 0.)
                self.assertAlmostEqual(correction, -displacement)
                # The kinematic predicted displacement after that joint change
                # reaches the frozen reference in this idealized model.
                self.assertAlmostEqual(displacement+correction, 0.)

    def test_repeated_drift_accumulates_even_when_absolute_pose_target_cancels(self):
        value, env = make()
        first = admit(value, env)
        raw_at_admission = value.hold_reference['raw_pose_joint_targets_rad']
        for index in range(1, 5):
            action = call(value, env, next_observation(value, displacement=.001,
                                                     joint_drift=.001))
            record = value.hold_records[-1]
            self.assertVectorAlmostEqual(record['raw_pose_joint_targets_rad'], raw_at_admission)
            self.assertAlmostEqual(record['pose_error_increment_rad'][0], -.001)
            self.assertAlmostEqual(value.correction[0], -.001*index)
            self.assertAlmostEqual(action['joint_targets_rad'][0], first['joint_targets_rad'][0]-.001*index)

    def test_zero_error_keeps_existing_accumulation_despite_new_measured_joints(self):
        value, env = make()
        admit(value, env)
        before = call(value, env, next_observation(value, displacement=.001))
        accumulated = list(value.correction)
        for drift in (.005, -.008, .013):
            action = call(value, env, next_observation(value, joint_drift=drift))
            self.assertEqual(action, before)
            self.assertEqual(value.correction, accumulated)
            self.assertEqual(value.hold_records[-1]['pose_error_increment_rad'], [0.]*6)

    def test_opposite_error_unwinds_accumulation_and_pose_and_jaw_stay_frozen(self):
        value, env = make()
        first = admit(value, env)
        reference = deepcopy(value.hold_reference)
        call(value, env, next_observation(value, displacement=.001))
        action = call(value, env, next_observation(value, displacement=-.001))
        self.assertVectorAlmostEqual(value.correction, [0.]*6)
        self.assertVectorAlmostEqual(action['joint_targets_rad'], first['joint_targets_rad'])
        self.assertEqual(action['gripper_position_m'], first['gripper_position_m'])
        self.assertEqual(value.hold_reference, reference)
        self.assertEqual(env.requests[-1], (reference['position_m'], reference['quaternion_wxyz'], .012))

    def test_slew_is_per_joint_and_unlimited_proposal_is_explicit(self):
        value, env = make()
        first = admit(value, env)
        env.raw_override = [.03, .01, .021, .019, .02, .02]
        action = call(value, env, next_observation(value))
        self.assertVectorAlmostEqual(value.correction, [.002, -.002, .001, -.001, 0., 0.])
        record = value.hold_records[-1]
        self.assertVectorAlmostEqual(record['proposed_cumulative_correction_rad'], [.01, -.01, .001, -.001, 0., 0.])
        self.assertVectorAlmostEqual(record['applied_cumulative_correction_rad'], value.correction)
        self.assertEqual(record['correction_rate_limited_axes'], [0, 1])
        self.assertTrue(record['correction_update_accepted'])
        self.assertVectorAlmostEqual(action['joint_targets_rad'],
                                     [a+b for a, b in zip(first['joint_targets_rad'], value.correction)])

    def test_total_bound_refuses_unlimited_proposal_before_slew_and_preserves_state(self):
        for direction in (-1., 1.):
            with self.subTest(direction=direction):
                value, env = make()
                admit(value, env)
                env.raw_override = [.02+direction*.002]+[.02]*5
                for _ in range(15):
                    call(value, env, next_observation(value))
                    self.assertTrue(value.admitted)
                previous = list(value.correction)
                self.assertAlmostEqual(previous[0], direction*.03)
                action = call(value, env, next_observation(value))
                self.assertEqual(value.abort_reason, 'joint_correction_exhausted')
                self.assertEqual(value.correction, previous)
                self.assertEqual(action['joint_targets_rad'], [.02]*6)
                record = value.hold_records[-1]
                self.assertAlmostEqual(record['proposed_cumulative_correction_rad'][0], direction*.032)
                self.assertEqual(record['accumulated_correction_rad'], previous)
                self.assertIsNone(record['applied_cumulative_correction_rad'])
                self.assertFalse(record['correction_update_accepted'])
                self.assertEqual(value.events[-1]['refused_bounds'], 'joint_correction_exhausted')
                calls = len(env.requests)
                again = call(value, env, next_observation(value))
                self.assertEqual(again, action)
                self.assertEqual(len(env.requests), calls)
                self.assertFalse(value.admitted)
                self.assertIsNone(value.hold_records[-1]['pose_error_increment_rad'])
        value, env = make()
        admit(value, env)
        env.raw_override = [.051]+[.02]*5
        call(value, env, next_observation(value))
        self.assertEqual(value.abort_reason, 'joint_correction_exhausted')
        self.assertEqual(value.correction, [0.]*6)

    def test_original_grip_force_pose_and_margin_refusals_never_integrate(self):
        cases = [({'both': False}, 'bilateral_contact_or_force_loss'),
                 ({'supported': True}, 'support_returned'),
                 ({'slip': .0031}, 'relative_slip'),
                 ({'force': 10.0001}, 'declared_force_ceiling_exceeded'),
                 ({'displacement': .0041}, 'tcp_position_error_exhausted'),
                 ({'roll': .051}, 'tcp_orientation_error_exhausted')]
        for kwargs, reason in cases:
            with self.subTest(reason=reason):
                value, env = make()
                admit(value, env)
                call(value, env, next_observation(value, displacement=.001))
                previous, requests = list(value.correction), len(env.requests)
                action = call(value, env, next_observation(value, **kwargs))
                self.assertEqual(value.abort_reason, reason)
                self.assertEqual(value.correction, previous)
                self.assertEqual(len(env.requests), requests)
                self.assertEqual(action['gripper_position_m'], .012)
                self.assertFalse(value.admitted)
        value, env = make()
        admit(value, env)
        call(value, env, next_observation(value, displacement=.001))
        previous = list(value.correction)
        # Joint 0 already sits on this margin. The next negative correction
        # must refuse, even though the cumulative correction remains bounded.
        low = value.stop_targets[0]-retention.JOINT_MARGIN_RAD
        value.joint_limits[0] = [low, 2.]
        call(value, env, next_observation(value, displacement=.001))
        self.assertEqual(value.abort_reason, 'native_joint_margin_exhausted')
        self.assertEqual(value.correction, previous)

    def test_admission_bias_and_margin_are_still_checked(self):
        for bound in ('bias', 'margin'):
            value, env = make()
            for tick in range(40):
                if value.qualification_since is not None and tick*.02-value.qualification_since >= .1-1e-9:
                    if bound == 'bias':
                        env.raw_override = [-.2]*6
                    else:
                        value.joint_limits[0] = [0., 2.]
                        value.last_requested['joint_targets_rad'][0] = .01
                call(value, env, observation(tick*.02))
                if value.aborted:
                    break
            self.assertEqual(value.abort_reason, 'initial_bias_exhausted' if bound == 'bias'
                             else 'native_joint_margin_exhausted')
            self.assertFalse(value.ever_admitted)
            self.assertEqual(value.correction, [0.]*6)

    def test_clipped_zero_error_solver_is_refused_at_measured_margin_on_admission(self):
        value, env = make()
        for tick in range(40):
            obs = observation(tick*.02)
            admitting = (value.qualification_since is not None
                         and tick*.02-value.qualification_since >= .1-1e-9)
            if admitting:
                low = value.joint_limits[0][0]
                obs['joint_position_rad'][0] = low+.01
                value.last_requested['joint_targets_rad'][0] = low+.02
                env.observation = obs
                raw = env.pose_action(obs['tcp_position_m'], obs['tcp_quaternion_wxyz'], .012)
                self.assertAlmostEqual(raw['joint_targets_rad'][0]-obs['joint_position_rad'][0], .01)
                calls = len(env.requests)
            call(value, env, obs)
            if value.aborted:
                self.assertTrue(admitting)
                self.assertEqual(len(env.requests), calls)
                break
        self.assertEqual(value.abort_reason, 'measured_joint_margin_exhausted')
        self.assertFalse(value.ever_admitted)
        self.assertEqual(value.correction, [0.]*6)
        self.assertEqual(value.hold_records[-1]['refused_bounds'], 'measured_joint_margin_exhausted')

    def test_measured_margin_checks_update_and_final_without_solving_or_integrating(self):
        for final in (False, True):
            with self.subTest(final=final):
                value, env = make()
                admit(value, env)
                call(value, env, next_observation(value, displacement=.001))
                previous, solves, commands = list(value.correction), len(env.requests), len(value.requested_controls)
                obs = next_observation(value)
                obs['joint_position_rad'][0] = value.joint_limits[0][0]+.01
                env.raw_override = AssertionError('Measured margin must be checked before IK')
                if final:
                    value.observe_final(obs)
                else:
                    call(value, env, obs)
                self.assertEqual(value.abort_reason, 'measured_joint_margin_exhausted')
                self.assertEqual(value.correction, previous)
                self.assertEqual(len(env.requests), solves)
                self.assertEqual(len(value.requested_controls), commands+int(not final))
                self.assertFalse(value.hold_records[-1]['correction_update_accepted'])

    def test_malformed_solver_refuses_without_integrating(self):
        for bad in ([float('nan')]*6, [0.]*5, 'bad', ValueError('solver')):
            value, env = make()
            admit(value, env)
            call(value, env, next_observation(value, displacement=.001))
            previous = list(value.correction)
            env.raw_override = bad
            call(value, env, next_observation(value))
            self.assertEqual(value.abort_reason, 'pose_solver_refused')
            self.assertEqual(value.correction, previous)
            self.assertFalse(value.admitted)

    def test_invalid_or_noncausal_input_refuses_without_command_or_integration(self):
        for invalid in ({'joint_position_rad': [float('nan')]*6},
                        {'t_seconds': .01}, {'tcp_quaternion_wxyz': [1., 1., 0., 0.]},
                        {'pad_normal_forces_n': {'left': -.1, 'right': .4}}):
            value, env = make()
            admit(value, env)
            call(value, env, next_observation(value, displacement=.001))
            before = (len(value.requested_controls), len(env.requests), list(value.correction))
            obs = next_observation(value)
            obs.update(invalid)
            with self.assertRaises(ValueError):
                call(value, env, obs)
            self.assertEqual((len(value.requested_controls), len(env.requests), value.correction), before)
            self.assertEqual(value.status, 'invalid_measurement_refused')
            self.assertFalse(value.input_refusals[-1]['command_issued'])
            with self.assertRaises(ValueError):
                call(value, env, next_observation(value))

    def test_final_observation_checks_evidence_without_solve_command_or_integration(self):
        for kwargs, reason in (({}, None), ({'displacement': .0041}, 'tcp_position_error_exhausted'),
                               ({'force': 10.001}, 'declared_force_ceiling_exceeded'),
                               ({'both': False}, 'bilateral_contact_or_force_loss'),
                               ({'roll': .051}, 'tcp_orientation_error_exhausted')):
            with self.subTest(reason=reason):
                value, env = make()
                admit(value, env)
                call(value, env, next_observation(value, displacement=.001))
                before = (len(value.requested_controls), len(env.requests), list(value.correction))
                env.raw_override = AssertionError('Final observation must never solve')
                result = value.observe_final(next_observation(value, **kwargs))
                self.assertEqual((len(value.requested_controls), len(env.requests), value.correction), before)
                self.assertEqual(result['abort_reason'], reason)
                record = result['retention']['hold_records'][-1]
                self.assertTrue(record['final_observation'])
                self.assertFalse(record['command_issued'])
                self.assertFalse(record['correction_update_accepted'])
                self.assertIsNone(record['pose_error_increment_rad'])
                self.assertIsNone(record['joint_targets_rad'])
                self.assertIsNone(record['applied_cumulative_correction_rad'])
                self.assertEqual(result['retention']['final_controller_status'], value.status)

    def test_summary_is_detached_and_every_event_has_truthful_candidate_identity(self):
        value, env = make()
        admit(value, env)
        call(value, env, next_observation(value, displacement=.001))
        summary = value.summary()
        self.assertEqual(summary['schema'], incremental.SCHEMA)
        self.assertEqual(summary['variant'], 'incremental')
        self.assertEqual(summary['contract']['variants'], ['incremental'])
        self.assertEqual(summary['acquisition_variant'], 'evidence')
        for event in summary['events']:
            self.assertEqual(event['variant'], 'incremental')
            self.assertEqual(event['retention_variant'], 'incremental')
            self.assertEqual(event['acquisition_variant'], 'evidence')
        summary['retention']['hold_reference']['position_m'][0] = 99.
        summary['retention']['hold_records'][-1]['accumulated_correction_rad'][0] = 99.
        summary['retention']['accumulated_correction_rad'][0] = 99.
        self.assertNotEqual(value.hold_reference['position_m'][0], 99.)
        self.assertNotEqual(value.correction[0], 99.)
        self.assertNotEqual(value.hold_records[-1]['accumulated_correction_rad'][0], 99.)


if __name__ == '__main__':
    unittest.main()

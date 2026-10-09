"""Model identification, causal projection and bounded feedback behavior."""
from dataclasses import asdict, replace
import json
import math
import unittest

import numpy as np

from dvidia_training.motor_memory import (AppliedTarget, CalibrationTransition,
    ControllerConfig, ExpectedContract, MotorController, MotorMemory,
    MotorObservation, estimate_state, fit_memory)


def memory():
    return MotorMemory(100., 10., .2, 20., 0., .1, 100)


def calibration():
    rng = np.random.default_rng(814)
    rows = []
    for _ in range(400):
        position = float(rng.uniform(-.1, .3))
        velocity = float(rng.uniform(-.5, .5))
        target = position+float(rng.uniform(-.15, .15))
        acceleration = max(-8., min(8., 100.*(target-position)-10.*velocity+.2))
        rows.append(CalibrationTransition(position, velocity, target, velocity+.001*acceleration, .001))
    return rows


class MemoryFitTests(unittest.TestCase):
    def test_identifies_excited_saturated_position_servo(self):
        result = fit_memory(calibration())
        self.assertAlmostEqual(result.stiffness_s2, 100., places=9)
        self.assertAlmostEqual(result.damping_s, 10., places=9)
        self.assertAlmostEqual(result.bias_m_s2, .2, places=9)
        self.assertEqual(result.sample_count, 400)
        self.assertGreaterEqual(result.max_acceleration_m_s2, 8.)
        self.assertLess(result.acceleration_rmse_m_s2, .2)
        self.assertGreaterEqual(result.acceleration_error_bound_m_s2, result.acceleration_rmse_m_s2)
        self.assertEqual(result.fit_scope, 'privileged_native_calibration_transitions')

    def test_json_roundtrip_preserves_authored_semantics(self):
        contract = ExpectedContract(holding_required=True, expected_contact='bilateral_pad_contact')
        result = fit_memory([asdict(r) for r in calibration()], expected_contract=contract)
        restored = MotorMemory.from_dict(json.loads(json.dumps(result.to_dict(), allow_nan=False)))
        self.assertEqual(restored, result)
        self.assertEqual(restored.expected_contract, contract)
        self.assertEqual(fit_memory(calibration()).expected_contract.holding_required, False)

    def test_acceleration_is_saturated_and_finite(self):
        fitted = fit_memory(calibration())
        self.assertEqual(fitted.acceleration(0., 0., 100.), fitted.max_acceleration_m_s2)
        self.assertEqual(fitted.acceleration(0., 0., -100.), -fitted.max_acceleration_m_s2)
        with self.assertRaises(ValueError):
            fitted.acceleration(math.nan, 0., 0.)

    def test_bad_unexcited_or_unstable_training_rejected(self):
        for rows in (None, [], calibration()[:11], [CalibrationTransition(0., 0., 0., 0., .01)]*30,
                     [CalibrationTransition(0., 0., .1, .1, .01)]*30):
            with self.subTest(rows=str(rows)[:30]), self.assertRaises(ValueError):
                fit_memory(rows)
        with self.assertRaises(ValueError):
            fit_memory([{'position_m': 0., 'hidden_goal': 1.}]*20)
        unstable = [replace(r, next_velocity_m_s=r.velocity_m_s-(r.next_velocity_m_s-r.velocity_m_s)) for r in calibration()]
        with self.assertRaises(ValueError):
            fit_memory(unstable)

    def test_metadata_and_nonfinite_artifacts_rejected(self):
        value = memory().to_dict()
        for update in ({'sample_count': True}, {'stiffness_s2': math.inf}, {'damping_s': -1.},
                       {'schema_version': 'unversioned'}, {'fit_scope': 'evaluation'},
                       {'uncertainty_scope': 'guaranteed'}, {'acceleration_rmse_m_s2': 1.}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                MotorMemory.from_dict(value | update)
        for invalid in ({k: v for k, v in value.items() if k != 'bias_m_s2'}, value | {'unknown': 1}):
            with self.assertRaises(ValueError):
                MotorMemory.from_dict(invalid)
        for factory in (lambda: CalibrationTransition(0., 0., 0., 0., 0.),
                        lambda: ExpectedContract(holding_required=1),
                        lambda: ExpectedContract(expected_contact=''),
                        lambda: fit_memory(calibration(), expected_contract=False)):
            with self.assertRaises(ValueError):
                factory()


class EstimatorTests(unittest.TestCase):
    def test_delayed_sample_projects_with_acknowledged_changes(self):
        observation = MotorObservation(0., .04, 0., 0., .0001)
        forward = (AppliedTarget(0., .1),)
        reversed_history = (AppliedTarget(0., .1), AppliedTarget(.02, -.1))
        predicted = estimate_state(.04, observation, memory(), forward)
        reversed_state = estimate_state(.04, observation, memory(), reversed_history)
        self.assertGreater(predicted.position_m, observation.position_m)
        self.assertGreater(predicted.velocity_m_s, reversed_state.velocity_m_s)
        self.assertGreater(predicted.position_m, reversed_state.position_m)
        self.assertAlmostEqual(predicted.age_s, .04)
        self.assertGreater(predicted.uncertainty_m, observation.position_uncertainty_m)

    def test_command_change_boundary_not_backdated(self):
        observation = MotorObservation(0., .01, 0., 0.)
        unchanged = estimate_state(.01, observation, memory(), [AppliedTarget(0., .1)], step_s=.01)
        changed_at_now = estimate_state(.01, observation, memory(),
            [AppliedTarget(0., .1), AppliedTarget(.01, -.1)], step_s=.01)
        self.assertEqual(unchanged, changed_at_now)
        changed_halfway = estimate_state(.01, observation, memory(),
            [AppliedTarget(0., .1), AppliedTarget(.005, -.1)], step_s=.01)
        self.assertLess(changed_halfway.velocity_m_s, unchanged.velocity_m_s)

    def test_new_observation_corrects_unexpected_motion(self):
        initial = MotorObservation(0., .04, 0., 0.)
        history = [AppliedTarget(0., .1)]
        before = estimate_state(.04, initial, memory(), history)
        measured = MotorObservation(.04, .04, -.02, -.1)
        after = estimate_state(.04, measured, memory(), history)
        self.assertGreater(before.position_m, 0.)
        self.assertEqual(after.position_m, -.02)
        self.assertEqual(after.velocity_m_s, -.1)
        self.assertEqual(after.age_s, 0.)

    def test_history_anchor_duplicates_and_prior_irrelevance(self):
        observation = MotorObservation(.02, .03, 0., 0.)
        state = estimate_state(.03, observation, memory(), [AppliedTarget(0., -.1), AppliedTarget(.02, .1)])
        identical = estimate_state(.03, observation, memory(),
            [AppliedTarget(0., .5), AppliedTarget(.02, 0.), AppliedTarget(.02, .1)])
        self.assertEqual(state, identical)

    def test_noncausal_stale_and_missing_projection_rejected(self):
        observation = MotorObservation(0., .01, 0., 0.)
        for now, obs, history, kwargs in (
            (0., observation, [AppliedTarget(0., 0.)], {}),
            (.2, observation, [AppliedTarget(0., 0.)], {}),
            (.01, observation, [], {}),
            (.01, observation, [AppliedTarget(.01, .1)], {}),
            (.01, observation, [AppliedTarget(0., 0.), AppliedTarget(.02, .1)], {}),
            (.01, observation, [AppliedTarget(.005, 0.), AppliedTarget(0., 0.)], {}),
            (.01, observation, [AppliedTarget(0., 0.)], {'step_s': 1e-8}),
        ):
            with self.subTest(now=now, history=history), self.assertRaises(ValueError):
                estimate_state(now, obs, memory(), history, **kwargs)

    def test_observation_validation_rejects_zero_missingness_substitution(self):
        for factory in (lambda: MotorObservation(.1, 0., 0., 0.),
                        lambda: MotorObservation(0., 0., math.nan, 0.),
                        lambda: MotorObservation(0., 0., 0., 0., -1.),
                        lambda: MotorObservation(True, 0., 0., 0.),
                        lambda: AppliedTarget(0., math.inf)):
            with self.assertRaises(ValueError):
                factory()


class ControllerTests(unittest.TestCase):
    def setup_controller(self, mode='adaptive_predictive', **kwargs):
        controller = MotorController(memory(), mode, ControllerConfig(**kwargs))
        history = [AppliedTarget(0., 0.)]
        controller.update(0., MotorObservation(0., 0., 0., 0.), .3, history)
        return controller, history

    def test_all_modes_slew_bounded_and_respond_to_public_goal_reversal(self):
        for mode in MotorController.MODES:
            with self.subTest(mode=mode):
                controller, history = self.setup_controller(mode)
                forward = controller.update(.01, MotorObservation(.01, .01, 0., 0.), .3, history)
                reverse = controller.update(.02, MotorObservation(.02, .02, 0., 0.), -.1, history)
                self.assertGreater(forward.target_m, 0.)
                self.assertLess(reverse.target_m, forward.target_m)
                self.assertLessEqual(abs(forward.target_m), .16*.01+1e-12)
                self.assertLessEqual(abs(reverse.target_m-forward.target_m), .16*.01+1e-12)

    def test_slow_baseline_is_half_the_matched_speed_cap(self):
        slow, history = self.setup_controller('slow_feedback', max_speed_m_s=.24)
        fast, _ = self.setup_controller('fast_fixed', max_speed_m_s=.24)
        observation = MotorObservation(.01, .01, 0., 0.)
        a, b = slow.update(.01, observation, .3, history), fast.update(.01, observation, .3, history)
        self.assertAlmostEqual(a.target_m, .0012)
        self.assertAlmostEqual(b.target_m, .0024)

    def test_uncertainty_and_delay_reduce_adaptive_approach_speed(self):
        certain, history = self.setup_controller(max_speed_m_s=.24)
        uncertain, _ = self.setup_controller(max_speed_m_s=.24)
        delayed, _ = self.setup_controller(max_speed_m_s=.24, command_delay_s=.2)
        clear = certain.update(.01, MotorObservation(.01, .01, .25, 0.), .3, history)
        noisy = uncertain.update(.01, MotorObservation(.01, .01, .25, 0., .01), .3, history)
        lagged = delayed.update(.01, MotorObservation(.01, .01, .25, 0.), .3, history)
        self.assertLess(abs(noisy.desired_speed_m_s), abs(clear.desired_speed_m_s))
        self.assertLess(abs(lagged.desired_speed_m_s), abs(clear.desired_speed_m_s))

    def test_adaptive_only_projects_state_but_all_share_observations(self):
        observation = MotorObservation(0., .04, 0., .2)
        history = [AppliedTarget(0., .03)]
        fast = MotorController(memory(), 'fast_fixed').update(.04, observation, .3, history)
        adaptive = MotorController(memory()).update(.04, observation, .3, history)
        self.assertEqual(fast.estimated_state.position_m, 0.)
        self.assertGreater(adaptive.estimated_state.position_m, 0.)
        self.assertEqual(fast.estimated_state.age_s, adaptive.estimated_state.age_s)

    def test_reference_does_not_run_arbitrarily_ahead_of_observed_position(self):
        controller, history = self.setup_controller('fast_fixed', max_speed_m_s=.24, max_target_lead_m=.01)
        previous = 0.
        for index in range(1, 51):
            now = index*.01
            output = controller.update(now, MotorObservation(now, now, 0., 0.), .3, history)
            self.assertFalse(output.halt)
            self.assertLessEqual(output.target_m, .01+1e-12)
            self.assertGreaterEqual(output.target_m, previous)
            previous = output.target_m

    def test_missing_stale_future_outside_and_missed_deadline_halt_without_target(self):
        for now, observation, goal in ((.01, None, .3),
            (.16, MotorObservation(0., .01, 0., 0.), .3),
            (.01, MotorObservation(.02, .02, 0., 0.), .3),
            (.01, MotorObservation(.01, .01, 0., 0.), 1.),
            (.06, MotorObservation(.06, .06, 0., 0.), .3),
            (.01, MotorObservation(.01, .01, -.5, 0.), .3)):
            controller, history = self.setup_controller()
            output = controller.update(now, observation, goal, history)
            self.assertTrue(output.halt)
            self.assertIsNone(output.target_m)
            self.assertEqual(output.desired_speed_m_s, 0.)

    def test_repeated_goal_feedback_corrects_a_disturbance(self):
        controller, history = self.setup_controller('fast_fixed')
        for index in range(1, 30):
            now = index*.01
            controller.update(now, MotorObservation(now, now, .001*index, .1), .3, history)
        output = controller.update(.30, MotorObservation(.30, .30, .05, 0.), -.05, history)
        self.assertFalse(output.halt)
        self.assertLess(output.desired_speed_m_s, 0.)

    def test_halt_latches_until_explicit_reset_and_fresh_anchor(self):
        controller, history = self.setup_controller()
        first = controller.update(.01, None, .3, history)
        self.assertTrue(first.halt)
        observation = MotorObservation(.02, .02, 0., 0.)
        self.assertEqual(controller.update(.02, observation, .3, history), first)
        controller.reset()
        resumed = controller.update(.02, observation, .3, history)
        self.assertFalse(resumed.halt)
        self.assertEqual(resumed.target_m, 0.)

    def test_near_goal_and_unknown_mode_or_invalid_config(self):
        controller = MotorController(memory())
        result = controller.update(0., MotorObservation(0., 0., .3, 0.), .3, [AppliedTarget(0., .3)])
        self.assertEqual(result.desired_speed_m_s, 0.)
        for factory in (lambda: MotorController(memory(), 'replay'),
                        lambda: ControllerConfig(braking_deceleration_m_s2=0.),
                        lambda: ControllerConfig(slow_speed_factor=1.1),
                        lambda: ControllerConfig(workspace_min_m=.4, workspace_max_m=.1)):
            with self.assertRaises(ValueError):
                factory()


if __name__ == '__main__':
    unittest.main()

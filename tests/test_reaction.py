"""Causal sensing, finite prediction and fail-closed reaction checks."""
from dataclasses import replace
import math
import unittest

from dvidia_training.reaction import (ContactSample, GripSample, MotionSample,
    ObstacleSphere, ReactionConfig, ReactionMonitor, RobotSphere, SensorConfig,
    SensorStream, Signal)


def motion(position=(1., 0., 0.), obstacle_velocity=(0., 0., 0.), robot_velocity=(0., 0., 0.)):
    return MotionSample((RobotSphere((0., 0., 0.), robot_velocity, .05),),
                        (ObstacleSphere(position, obstacle_velocity, .05),))


def observations(now=0., scene=None, force=0., grip=None):
    return {'motion': Signal(now, scene or motion()),
            'contact': Signal(now, ContactSample(force)),
            'grip': None if grip is None else Signal(now, grip)}


class SensorStreamTests(unittest.TestCase):
    def test_transport_preserves_acquisition_time_and_age(self):
        stream = SensorStream(SensorConfig(period_s=.02, delivery_delay_s=.03))
        self.assertIsNone(stream.update(0., ContactSample(1.)))
        self.assertIsNone(stream.update(.02, ContactSample(2.)))
        delivered = stream.update(.03, ContactSample(3.))
        self.assertEqual(delivered, Signal(0., ContactSample(1.)))
        self.assertAlmostEqual(.03-delivered.acquired_s, .03)
        self.assertEqual(stream.update(.05, ContactSample(5.)), Signal(.02, ContactSample(2.)))
        self.assertEqual(stream.sample_count, 3)

    def test_delivered_dropout_clears_held_value_without_fabricated_zero(self):
        stream = SensorStream(SensorConfig(period_s=.1))
        self.assertEqual(stream.update(0., ContactSample(5.)).value.unexpected_force_n, 5.)
        self.assertEqual(stream.update(.05, None).value.unexpected_force_n, 5.)
        self.assertIsNone(stream.update(.1, None))
        self.assertIsNone(stream.latest)
        always_dropped = SensorStream(SensorConfig(dropout_probability=1.))
        self.assertIsNone(always_dropped.update(0., GripSample(True, .01)))

    def test_seeded_faults_reproducible_independent_of_intermediate_polls(self):
        config = SensorConfig(period_s=.1, force_noise_n=2., dropout_probability=.3)
        sparse, dense = SensorStream(config, seed=43), SensorStream(config, seed=43)
        results = []
        for index in range(20):
            now = index*.1
            value = ContactSample(5.)
            a, b = sparse.update(now, value), dense.update(now, value)
            self.assertEqual(a, b)
            results.append(a)
            dense.update(now+.01, value)
        self.assertTrue(any(value is None for value in results))
        self.assertTrue(any(value is not None for value in results))
        self.assertEqual(sparse.sample_count, dense.sample_count)

    def test_noise_bounded_per_axis_and_magnitude(self):
        config = SensorConfig(position_noise_m=.01, velocity_noise_m_s=.02,
                              force_noise_n=2., slip_noise_m_s=.003)
        scene = motion(obstacle_velocity=(-.2, 0., 0.))
        stream = SensorStream(config, seed=82)
        noisy = stream.update(0., scene).value
        for old, new in zip(scene.robot_spheres+scene.obstacle_spheres,
                            noisy.robot_spheres+noisy.obstacle_spheres):
            self.assertTrue(all(abs(a-b) <= .01 for a, b in zip(old.center_m, new.center_m)))
            self.assertTrue(all(abs(a-b) <= .02 for a, b in zip(old.velocity_m_s, new.velocity_m_s)))
            self.assertEqual(old.radius_m, new.radius_m)
        self.assertLessEqual(abs(stream.update(.02, ContactSample(10.)).value.unexpected_force_n-10.), 2.)
        self.assertLessEqual(abs(stream.update(.04, GripSample(True, .01)).value.slip_speed_m_s-.01), .003)
        self.assertEqual(stream.latest.value.holding, True)

    def test_late_poll_does_not_backdate_or_synthesize_acquisitions(self):
        stream = SensorStream(SensorConfig(period_s=.01))
        stream.update(0., ContactSample(1.))
        self.assertEqual(stream.update(1., ContactSample(2.)).acquired_s, 1.)
        self.assertEqual(stream.sample_count, 2)
        stream.update(1., ContactSample(3.))
        self.assertEqual(stream.sample_count, 2)


class ReactionMonitorTests(unittest.TestCase):
    def test_approaching_obstacle_stops_before_contact(self):
        decision = ReactionMonitor().update(0., observations(scene=motion((.4, 0., 0.), (-1., 0., 0.))))
        self.assertTrue(decision.stop)
        self.assertEqual(decision.reason, 'predicted_collision_moving')
        self.assertAlmostEqual(decision.minimum_predicted_clearance_m, -.1)
        self.assertEqual((decision.signal_name, decision.signal_acquired_s, decision.detected_s), ('motion', 0., 0.))

    def test_robot_approaching_static_obstacle_is_positive(self):
        decision = ReactionMonitor().update(0., observations(scene=motion((.4, 0., 0.), robot_velocity=(1., 0., 0.))))
        self.assertTrue(decision.stop)
        self.assertEqual(decision.reason, 'predicted_collision_static')

    def test_receding_and_out_of_horizon_motion_do_not_stop(self):
        config = ReactionConfig(horizon_s=.2, command_delay_s=0., margin_m=0.)
        for position, velocity in (((.4, 0., 0.), (1., 0., 0.)), ((1., 0., 0.), (-1., 0., 0.)),
                                   ((.4, .3, 0.), (-1., 0., 0.))):
            with self.subTest(position=position, velocity=velocity):
                self.assertFalse(ReactionMonitor(config).update(0., observations(scene=motion(position, velocity))).stop)

    def test_sample_age_propagates_relative_positions_to_now(self):
        config = ReactionConfig(horizon_s=.05, command_delay_s=0., margin_m=0., max_sensor_age_s=.2)
        scene = motion((.22, 0., 0.), (-1., 0., 0.))
        monitor = ReactionMonitor(config)
        self.assertFalse(monitor.update(0., observations(scene=scene)).stop)
        obs = observations(.1)
        obs['motion'] = Signal(0., scene)
        decision = monitor.update(.1, obs)
        self.assertTrue(decision.stop)
        self.assertAlmostEqual(decision.minimum_predicted_clearance_m, -.03)

    def test_braking_assumption_and_command_delay_increase_margin(self):
        scene = motion((.5, 0., 0.), robot_velocity=(1., 0., 0.))
        fast = ReactionConfig(horizon_s=.01, margin_m=0., command_delay_s=0., assumed_deceleration_m_s2=100.)
        slow = replace(fast, assumed_deceleration_m_s2=1.)
        delayed = replace(fast, command_delay_s=.4)
        self.assertFalse(ReactionMonitor(fast).update(0., observations(scene=scene)).stop)
        self.assertTrue(ReactionMonitor(slow).update(0., observations(scene=scene)).stop)
        self.assertTrue(ReactionMonitor(delayed).update(0., observations(scene=scene)).stop)

    def test_force_and_holding_slip_have_causal_triggers(self):
        force = ReactionMonitor().update(.2, observations(.2, force=10.1))
        self.assertEqual(force.reason, 'unexpected_contact')
        self.assertEqual(force.signal_name, 'contact')
        self.assertEqual(force.detected_s, .2)
        slip = ReactionMonitor().update(0., observations(grip=GripSample(True, .02)))
        self.assertEqual(slip.reason, 'grip_slip')
        self.assertFalse(ReactionMonitor().update(0., observations(grip=GripSample(False, .02))).stop)

    def test_missing_and_stale_required_signals_stop(self):
        for name in ('motion', 'contact'):
            obs = observations()
            obs[name] = None
            self.assertEqual(ReactionMonitor().update(0., obs).reason, f'missing_{name}')
            obs = observations(.2)
            obs[name] = observations()[name]
            self.assertEqual(ReactionMonitor().update(.2, obs).reason, f'stale_{name}')

    def test_delayed_contact_detects_at_delivery_not_acquisition(self):
        stream = SensorStream(SensorConfig(period_s=.01, delivery_delay_s=.03))
        monitor = ReactionMonitor(ReactionConfig(require_motion=False, require_contact=False))
        for now in (0., .01, .02):
            decision = monitor.update(now, {'contact': stream.update(now, ContactSample(20.))})
            self.assertFalse(decision.stop)
        decision = monitor.update(.03, {'contact': stream.update(.03, ContactSample(20.))})
        self.assertEqual(decision.reason, 'unexpected_contact')
        self.assertEqual((decision.signal_acquired_s, decision.detected_s), (0., .03))

    def test_grip_required_while_holding_and_stale_optional_signals_ignored(self):
        monitor = ReactionMonitor()
        self.assertFalse(monitor.update(0., observations(grip=GripSample(True, 0.))).stop)
        self.assertEqual(monitor.update(.02, observations(.02)).reason, 'missing_grip')
        self.assertFalse(ReactionMonitor(ReactionConfig(require_motion=False, require_contact=False))
                         .update(.5, {'contact': Signal(0., ContactSample(100.))}).stop)
        self.assertEqual(ReactionMonitor().update(0., observations(), holding=True).reason, 'missing_grip')

    def test_stale_release_report_cannot_clear_holding_obligation(self):
        monitor = ReactionMonitor()
        self.assertFalse(monitor.update(0., observations(grip=GripSample(True, 0.))).stop)
        obs = observations(.3)
        obs['grip'] = Signal(.01, GripSample(False, 0.))
        self.assertEqual(monitor.update(.3, obs).reason, 'stale_grip')

    def test_expected_retention_obligation_is_independent_of_measured_holding(self):
        monitor = ReactionMonitor()
        lost = monitor.update(0., observations(grip=GripSample(False, 0.)), holding=True)
        self.assertEqual(lost.reason, 'grip_lost')
        self.assertEqual(lost.signal_name, 'grip')
        released = ReactionMonitor().update(0., observations(grip=GripSample(True, .02)), holding=False)
        self.assertFalse(released.stop)

    def test_stop_latches_after_obstacle_disappears_until_reset(self):
        monitor = ReactionMonitor()
        first = monitor.update(0., observations(force=20.))
        self.assertEqual(monitor.update(.1, observations(.1)), first)
        monitor.reset()
        self.assertFalse(monitor.update(0., observations()).stop)

    def test_no_obstacles_has_no_invented_clearance(self):
        scene = MotionSample((RobotSphere((0., 0., 0.), (0., 0., 0.), .05),), ())
        decision = ReactionMonitor().update(0., observations(scene=scene))
        self.assertFalse(decision.stop)
        self.assertIsNone(decision.minimum_predicted_clearance_m)


class ValidationTests(unittest.TestCase):
    def test_invalid_configuration_and_payloads_rejected(self):
        factories = [lambda: SensorConfig(period_s=0.), lambda: SensorConfig(dropout_probability=1.1),
                     lambda: ReactionConfig(assumed_deceleration_m_s2=0.), lambda: ReactionConfig(require_motion=1),
                     lambda: SensorConfig(force_noise_n=math.inf), lambda: ContactSample(-1.),
                     lambda: ContactSample(10**1000), lambda: MotionSample(None, ()),
                     lambda: GripSample(1, 0.), lambda: Signal(-1., ContactSample(0.)),
                     lambda: Signal(0., 0.), lambda: RobotSphere((0., 0.), (0., 0., 0.), .1),
                     lambda: RobotSphere((0., math.nan, 0.), (0., 0., 0.), .1),
                     lambda: RobotSphere((0., 0., 0.), (0., 0., 0.), 0.), lambda: MotionSample((), ())]
        for factory in factories:
            with self.subTest(factory=factory), self.assertRaises(ValueError):
                factory()

    def test_future_regressing_and_wrong_modality_observations_rejected(self):
        monitor = ReactionMonitor()
        for obs in ({'motion': Signal(.1, motion())}, {'contact': Signal(0., motion())}, {'truth': None}):
            with self.subTest(obs=obs), self.assertRaises(ValueError):
                monitor.update(0., obs)
        monitor.update(.1, observations(.1))
        with self.assertRaises(ValueError):
            monitor.update(.2, observations(0.))
        with self.assertRaises(ValueError):
            monitor.update(.05, observations(.05))
        stream = SensorStream()
        stream.update(.1, ContactSample(0.))
        with self.assertRaises(ValueError):
            stream.update(0., ContactSample(0.))


    def test_latched_stop_still_rejects_regressing_sensor_history(self):
        monitor = ReactionMonitor()
        monitor.update(0., {'motion': None, 'contact': None})
        original = monitor.update(.1, observations(.1))
        self.assertTrue(original.stop)
        self.assertEqual(monitor.update(.2, observations(.2)), original)
        with self.assertRaises(ValueError):
            monitor.update(.3, observations(.15))


if __name__ == '__main__':
    unittest.main()

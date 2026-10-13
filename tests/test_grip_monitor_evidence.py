"""Synthetic observation-clock tests; these are not physical robot safety tests."""
from types import SimpleNamespace
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from dvidia_training.continuous_grip import Episode
from dvidia_training import tool_reach


def guard_fixture():
    # Exercise the real guard method against a deterministic observation stream.
    # No body motion, native contact or sensor-latency claim comes from this stub.
    episode=Episode.__new__(Episode)
    episode.data=SimpleNamespace(time=0.)
    episode.events=[];episode.monitor=True;episode.loss_since=None
    episode.first_loss=None;episode.reason=None;episode.force=[0.,2.,1.,1.,0.]
    episode.trace=[];episode.slip_since=None
    return episode


class GripObservationClockTests(unittest.TestCase):
    def test_two_millisecond_dropout_between_decision_ticks_emits_no_sampled_alarm(self):
        episode=guard_fixture()
        self.assertTrue(episode.guard())
        # Native contact vanishes between observations, then returns before20ms.
        episode.data.time=.008;episode.force[2]=0.;episode.first_loss=.008
        episode.data.time=.010;episode.force[2]=1.;episode.first_loss=None
        episode.data.time=.020
        self.assertTrue(episode.guard())
        self.assertEqual(episode.events,[])
        self.assertIsNone(episode.loss_since)

    def test_three_bad_decision_samples_are_not_claimed_as_continuous_native_loss(self):
        episode=guard_fixture();episode.force[2]=0.;episode.first_loss=0.
        self.assertTrue(episode.guard())
        # Bilateral contact briefly recovers between sampled decisions.
        episode.data.time=.008;episode.force[2]=1.;episode.first_loss=None
        episode.data.time=.018;episode.force[2]=0.;episode.first_loss=.018
        episode.data.time=.020
        self.assertTrue(episode.guard())
        episode.data.time=.028;episode.force[2]=1.;episode.first_loss=None
        episode.data.time=.038;episode.force[2]=0.;episode.first_loss=.038
        episode.data.time=.040
        self.assertFalse(episode.guard())
        alarm=episode.events[-1]
        self.assertEqual(alarm['type'],'grip_loss_detected')
        self.assertAlmostEqual(alarm['t'],.040)
        self.assertAlmostEqual(alarm['first_native_loss_s'],.038)
        self.assertAlmostEqual(alarm['t']-alarm['first_native_loss_s'],.002)

    def test_observed_contact_recovery_resets_debounce(self):
        episode=guard_fixture();episode.force[2]=0.;episode.first_loss=0.
        self.assertTrue(episode.guard())
        episode.data.time=.020;episode.force[2]=1.;episode.first_loss=None
        self.assertTrue(episode.guard());self.assertIsNone(episode.loss_since)
        episode.data.time=.040;episode.force[2]=0.;episode.first_loss=.038
        self.assertTrue(episode.guard())
        self.assertEqual(episode.events,[])
        self.assertAlmostEqual(episode.loss_since,.040)

    def test_upward_load_transfer_and_common_downward_motion_do_not_trigger_slip(self):
        for shoe_delta,tcp_delta in ((.060,.110),(-.020,-.020)):
            episode=guard_fixture()
            episode.trace=[[i*.002,0.,2.,1.,1.,.4,-.1,.10+i/100*shoe_delta,0.,.4,-.1,.15+i/100*tcp_delta] for i in range(101)]
            for t in (.20,.22,.24):
                episode.data.time=t;self.assertTrue(episode.guard())
            self.assertEqual(episode.events,[])

    def test_relative_fall_requires_three_sampled_observations(self):
        episode=guard_fixture()
        episode.trace=[[i*.002,0.,2.,1.,1.,.4,-.1,.10-i/100*.020,0.,.4,-.1,.15+i/100*.020] for i in range(101)]
        episode.data.time=.20;self.assertTrue(episode.guard())
        episode.data.time=.22;self.assertTrue(episode.guard())
        episode.data.time=.24;self.assertFalse(episode.guard())
        alarm=episode.events[-1]
        self.assertEqual(alarm['reason'],'relative_downward_slip')
        self.assertIsNone(alarm['first_native_loss_s'])
        self.assertAlmostEqual(alarm['relative_drop_m'],.040)

    def test_lift_to_hold_keeps_pending_sampled_loss(self):
        episode=guard_fixture();episode.data.time=.040;episode.steps=20
        episode.force[2]=0.;episode.loss_since=0.;episode.first_loss=.038
        episode.target=SimpleNamespace(tolist=lambda:[.4,-.1,.28])
        def unexpected_step():self.fail('A due loss alarm must stop the next hold step')
        episode.tick=unexpected_step
        self.assertFalse(episode.hold('retention_hold',.5,0.,monitor=True))
        alarm=episode.events[-1]
        self.assertEqual(alarm['type'],'grip_loss_detected')
        self.assertEqual(alarm['t'],.040)
        self.assertEqual(alarm['first_native_loss_s'],.038)


@unittest.skipUnless(os.environ.get('DVIDIA_ROBOT_PROFILE_ASSETS'),'Pinned native PiPER assets required')
class NativeRecoveryPacketTests(unittest.TestCase):
    def test_recovery_event_matches_control_packet_given_to_native_step(self):
        # An old development scene probes issuance only, not successful recovery.
        episode=Episode(Path(os.environ['DVIDIA_ROBOT_PROFILE_ASSETS']),tool_reach.cases('development')[0],False)
        for _ in range(6):episode.tick()
        packets=[];native_step=episode.mj.mj_step
        def capture(model,data):
            packets.append((round(float(data.time),6),data.ctrl.copy()))
            return native_step(model,data)
        with patch.object(episode.mj,'mj_step',side_effect=capture):
            episode.controlled_lower('independent_issuance_probe')
        requests=[e for e in episode.events if e['type']=='grip_recovery_started']
        issued=[e for e in episode.events if e['type']=='recovery_control_issued']
        self.assertEqual(len(requests),1);self.assertEqual(len(issued),1)
        packet=issued[0]
        self.assertEqual(packet['requested_at_s'],requests[0]['t'])
        self.assertEqual(packet['t'],packets[0][0])
        episode.np.testing.assert_array_equal(packet['issued_ctrl'],packets[0][1])
        self.assertGreaterEqual(packet['t'],requests[0]['t'])
        self.assertLessEqual(packet['t']-requests[0]['t'],.002+1e-8)
        self.assertIn('pre-step',packet['time_convention'])


if __name__=='__main__':unittest.main()

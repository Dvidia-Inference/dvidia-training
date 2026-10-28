"""Handcrafted causal numeric evidence; no policy, fitting or physics calls."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from publishing import audit_piper_grip_clock_v03 as audit
from tests.test_piper_feedback_model import base_model


CONFIG = dict(gate_offset_s=.16, minimum_pad_force_n=.20, secure_dwell_s=.10,
              maximum_relative_drift_m=.002, acquisition_only=True)


def state(t, *, force=1., contacts=1, angle=0.):
    quaternion = [float(np.cos(angle/2)), 0., float(np.sin(angle/2)), 0.]
    tcp = np.asarray([.299, -.03, .16])
    offset = np.asarray([-.08*np.sin(angle), 0., -.08*np.cos(angle)])
    return dict(t_seconds=t, object_position_m=(tcp+offset).tolist(), tcp_position_m=tcp.tolist(),
                tcp_quaternion_wxyz=quaternion, target_position_m=[.35, .08, .08],
                joint_position_rad=[0., 1.4, -1.3, 0., 0., 0.],
                object_linear_velocity_m_s=[0., 0., 0.], gripper_joint_position_m=[.0175, -.0175],
                grasp_contact_counts=dict(left=contacts, right=contacts),
                pad_normal_forces_n=dict(left=force, right=force), table_object_contact_count=1)


def native(rows=100):
    return dict(t_seconds=np.arange(1, rows+1)*.001,
                object_position_m=np.tile([.299, -.03, .08], (rows, 1)),
                tcp_position_m=np.tile([.299, -.03, .16], (rows, 1)),
                tcp_quaternion_wxyz=np.tile([1., 0., 0., 0.], (rows, 1)),
                grasp_contact_counts=np.ones((rows, 2)), pad_normal_forces_n=np.ones((rows, 2)),
                table_object_contact_count=np.ones(rows))


def synthetic_base():
    base = base_model()
    weights = np.asarray(base['weights'])
    centers = np.asarray(base['time_centers_s'])
    weights[:, 0, :, 6] = np.where((centers >= .2) & (centers < 1.2), 0., 1.)
    base['weights'] = weights.tolist()
    return base


def physical_jaw(base, t):
    # Independent one-dimensional fixture formula, with only spatial constant
    # coefficient active. No runtime/auditor raw-reference function is used.
    centers = np.asarray(base['time_centers_s'])
    logits = -.5*((t-centers)/base['time_width_s'])**2
    weights = np.exp(logits-np.max(logits)); weights /= weights.sum()
    return float(.035*np.dot(weights, np.asarray(base['weights'])[0, 0, :, 6]))


def synthetic_trace(variant='fixed'):
    base = synthetic_base()
    states = [state(i*.02) for i in range(7)]
    actions = []
    snapshot = np.r_[0., 0., -1.6, (.35-.299)/.2, (.08+.03)/.2, 0.,
                     0., 0., 0., .5, -.5, np.log(2)/3, np.log(2)/3, .5, .5, .5]
    for i in range(6):
        t, jaw = i*.02, physical_jaw(base, i*.02)
        feature = np.r_[.299/.5, -.03/.5, .35/.5, .08/.5, t/20., jaw/.035,
                        min(t, .04)/.12, min(t, .12)/.12, snapshot, snapshot, snapshot, t/20., 0.]
        if variant == 'blind_clock':
            feature[8:56] = 0.
        decision = dict(physical_time_s=t, progress_before_s=t, progress_after_s=(i+1)*.02,
                        increment_s=.02, features=feature.tolist(), nominal_physical_jaw_m=jaw,
                        advance_probability=1., frozen_jaw_target_m=jaw, jaw_mode='coordinated_reference',
                        decision_origin='learned_blind_clock' if variant == 'blind_clock' else
                                        'learned_grip_clock' if variant == 'clock' else 'fixed_unit_progress')
        actions.append(dict(t_seconds=t, joint_targets_rad=[0., 1.4, -1.3, 0., 0., 0.],
                            gripper_position_m=jaw, clock_decision=decision))
    return base, dict(policy_kind=variant, initial_object_position_m=[.299, -.03, .08],
                      trajectory=states, actions=actions, config=dict(min_lift_m=.06, min_lift_dwell_s=.1))


class IndependentClockAuditTests(unittest.TestCase):
    def test_tool_local_offset_is_invariant_under_rigid_rotation(self):
        for degrees in (0., 1., 5., 90.):
            observation = state(0., angle=np.deg2rad(degrees))
            np.testing.assert_allclose(audit.local_offset(observation['object_position_m'],
                observation['tcp_position_m'], observation['tcp_quaternion_wxyz']), [0., 0., -.08], atol=1e-16)
        with self.assertRaises(ValueError):
            audit.local_offset([0., 0., 0.], [0., 0., 0.], [2., 0., 0., 0.])

    def test_six_causal_observations_establish_sampled_readiness_and_latch_supported_grip(self):
        measurement = audit.CausalMeasurements([.299, -.03, .08], [.35, .08, .08], CONFIG)
        for i in range(6):
            _, result = measurement.observe(state(i*.02, angle=np.deg2rad(i)), .035, 0., 0, 1000)
            self.assertEqual(result['secure'], i == 5)
            self.assertEqual(result['teacher_advance'], i == 5)
        self.assertTrue(result['supported'])
        self.assertTrue(result['acquisition_released'])
        self.assertAlmostEqual(result['drift_m'], 0., places=12)
        _, result = measurement.observe(state(.12, force=0., contacts=0), .035, .02, 0, 1000)
        self.assertFalse(result['secure'])
        self.assertTrue(result['teacher_advance'])

    def test_contact_force_or_drift_breaks_readiness_without_future_observations(self):
        for fault in ('force', 'contact', 'drift'):
            with self.subTest(fault=fault):
                measurement = audit.CausalMeasurements([.299, -.03, .08], [.35, .08, .08], CONFIG)
                for i in range(6):
                    observation = state(i*.02)
                    if i == 5:
                        if fault == 'force': observation['pad_normal_forces_n']['right'] = .199
                        elif fault == 'contact': observation['grasp_contact_counts']['left'] = 0
                        else: observation['tcp_position_m'][0] += .003
                    _, result = measurement.observe(observation, .035, 0., 0, 1000)
                self.assertFalse(result['secure'])
                self.assertFalse(result['teacher_advance'])

    def test_history_lags_and_elapsed_context_are_reconstructed_independently(self):
        measurement = audit.CausalMeasurements([.299, -.03, .08], [.35, .08, .08], CONFIG)
        for i in range(8):
            feature, _ = measurement.observe(state(i*.02, contacts=i), .0175, 0., 0, 1000)
        # Left contact slot in each16-component snapshot is13.
        self.assertEqual(feature[8+13], 3.5)
        self.assertEqual(feature[8+16+13], 2.5)
        self.assertEqual(feature[8+32+13], .5)
        self.assertAlmostEqual(feature[4], .14/20.)
        self.assertEqual(feature[56], 0.)
        self.assertAlmostEqual(feature[57], .14/20.)
        with self.assertRaises(ValueError):
            measurement.observe(state(.14), .0175, 0., 0, 1000)

    def test_native_window_distinguishes_sampled_ready_from_a_hidden_1ms_force_loss(self):
        arrays = native()
        good = audit.native_readiness_window(arrays, .1, CONFIG, np.asarray([0., 0., -.08]))
        self.assertTrue(good['secure'])
        self.assertEqual(good['supported_rows'], 100)
        arrays['pad_normal_forces_n'][54, 0] = 0.
        bad = audit.native_readiness_window(arrays, .1, CONFIG, np.asarray([0., 0., -.08]))
        self.assertTrue(bad['complete'])
        self.assertFalse(bad['secure'])
        self.assertEqual(bad['force_or_contact_insecure_rows'], 1)
        missing = native(99)
        self.assertFalse(audit.native_readiness_window(missing, .1, CONFIG, np.asarray([0., 0., -.08]))['complete'])
        duplicate = native()
        duplicate['t_seconds'][54] = duplicate['t_seconds'][53]
        self.assertFalse(audit.native_readiness_window(duplicate, .1, CONFIG, np.asarray([0., 0., -.08]))['complete'])

    def test_raw_reference_and_native_caps_are_numeric_not_policy_calls(self):
        base = synthetic_base()
        reference = audit.References(base, [.299, -.03, .08], [.35, .08, .08])
        for t in (0., .2, .4, 1.2, 20.):
            measured = state(0.)
            measured['joint_position_rad'][0] = 1.
            targets, jaw = reference.action(measured, t)
            self.assertAlmostEqual(targets[0], .94)
            np.testing.assert_allclose(targets[1:], [1.4, -1.3, 0., 0., 0.], atol=1e-12)
            self.assertAlmostEqual(jaw, physical_jaw(base, t), places=12)
        with self.assertRaises(ValueError): reference.raw(.021)
        with self.assertRaises(ValueError): reference.raw(20.02)

    def test_trace_audit_catches_feature_progress_and_command_tampering(self):
        base, episode = synthetic_trace()
        result = audit.inspect_trace(episode, native(), base, {}, None, CONFIG, 'coordinated_reference')
        self.assertEqual(result['feature_mismatches'], 0)
        self.assertEqual(result['progress_mismatches'], 0)
        self.assertEqual(result['command_mismatches'], 0)
        self.assertEqual(result['clock_metadata_mismatches'], 0)
        self.assertEqual(result['sampled_secure_observations'], 1)
        self.assertEqual(result['first_sampled_secure_s'], .1)
        modified = deepcopy(episode)
        modified['actions'][0]['clock_decision']['features'][4] = .5  # Future elapsed time.
        modified['actions'][1]['clock_decision']['progress_after_s'] = .10
        modified['actions'][2]['joint_targets_rad'][0] = .001
        modified['actions'][3]['clock_decision']['jaw_mode'] = 'physical_adapter'
        result = audit.inspect_trace(modified, native(), base, {}, None, CONFIG, 'coordinated_reference')
        self.assertEqual(result['feature_mismatches'], 1)
        self.assertGreaterEqual(result['progress_mismatches'], 1)
        self.assertEqual(result['command_mismatches'], 1)
        self.assertEqual(result['clock_metadata_mismatches'], 1)

    def test_blind_tree_decisions_use_only_masked_clock_features(self):
        base, episode = synthetic_trace('blind_clock')
        document = dict(nodes=[dict(feature=8, threshold=.1, left=1, right=2),
                               dict(feature=-1, advance_probability=1.), dict(feature=-1, advance_probability=0.)])
        result = audit.inspect_trace(episode, native(), base, {}, document, CONFIG, 'coordinated_reference')
        self.assertEqual(result['decision_mismatches'], 0)
        self.assertEqual(result['feature_mismatches'], 0)
        episode['actions'][0]['clock_decision']['features'][8] = .5
        result = audit.inspect_trace(episode, native(), base, {}, document, CONFIG, 'coordinated_reference')
        self.assertEqual(result['feature_mismatches'], 1)
        self.assertEqual(result['decision_mismatches'], 0)  # Rebuilt inputs determine the independent decision.
        with self.assertRaises(ValueError):
            audit.tree_probability({'nodes': [dict(feature=0, threshold=0., left=0, right=0)]}, np.zeros(58))

    def test_audit_cannot_write_inside_the_study_or_overwrite_an_existing_output(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)/'study'; root.mkdir()
            with self.assertRaises(ValueError): audit.audit(root, root/'audit.json')
            output = Path(temporary)/'exists.json'; output.write_text('{}')
            with self.assertRaises(ValueError): audit.audit(root, output)
            self.assertEqual(list(root.iterdir()), [])

    def test_attempt_inventory_and_hashes_are_checked_without_opening_a_simulator(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            case = dict(id='case-a')
            binding = dict(case=case, variant='fixed', collect_labels=False,
                           protocol_sha256='a'*64, model_raw_sha256=None)
            (folder/'started.json').write_text(json.dumps(binding))
            for name in ('episode.json.gz', 'native.npz', 'native.json', 'score.json', 'post-grip.json'):
                (folder/name).write_bytes(b'fixture')
            files = {p.name: audit._hash(p) for p in folder.iterdir()}
            (folder/'receipt.json').write_text(json.dumps(dict(binding=binding, case_id='case-a', variant='fixed', files=files)))
            self.assertEqual(audit._verify_attempt(folder, case, 'fixed', 'a'*64, None, False), binding)
            (folder/'native.npz').write_bytes(b'changed')
            with self.assertRaises(ValueError): audit._verify_attempt(folder, case, 'fixed', 'a'*64, None, False)


if __name__ == '__main__':
    unittest.main()

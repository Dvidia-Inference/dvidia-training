"""Passive telemetry preserves physics and makes sampling limitations explicit."""
from copy import deepcopy
from hashlib import sha256
import gzip
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dvidia_training.arm_env import ArmConfig, ArmEnv
from dvidia_training.native_contact_audit import (
    AuditedArmEnv, TickRecorder, assert_equal, canonical, compare_sample,
    contact_classification, planned_inputs, read_frozen_input, run_audit,
    sampling_misses,
)


def row(time, *, left=1, right=1, table=0, lift=.06, action=0, phase='lift'):
    return {'time': time, 'phase': phase, 'action_index': action,
            'pads': {'left': {'scored_count': left}, 'right': {'scored_count': right}},
            'table': {'scored_count': table}, 'object_lift_m': lift}


class NativeContactAuditTests(unittest.TestCase):
    def test_force_threshold_and_geometry_are_different_evidence(self):
        self.assertEqual(contact_classification([.011], -.00001), 'scored_contact')
        # Several weak contacts can carry a summed force above 0.01 N while
        # no individual contact passes the unchanged original scoring gate.
        self.assertEqual(contact_classification([.008] * 4, -.00001), 'positive_force_below_threshold')
        self.assertEqual(contact_classification([.01], 0.), 'positive_force_below_threshold')
        self.assertEqual(contact_classification([0.], -.00001), 'candidate_without_positive_force')
        self.assertEqual(contact_classification([], .002), 'geometrically_separated')
        self.assertEqual(contact_classification([], -.00001), 'no_scored_contact_unclassified')

    def test_duplicates_cannot_inflate_native_dwell(self):
        recorder = TickRecorder(.001)
        first = row(.001)
        self.assertTrue(recorder.append(first))
        self.assertFalse(recorder.append(row(.001)))
        self.assertEqual(len(recorder.rows), 1)
        self.assertEqual(recorder.dwell, .001)
        with self.assertRaisesRegex(ValueError, 'skip'):
            recorder.append(row(.003))
        with self.assertRaisesRegex(ValueError, 'reverse'):
            recorder.append(row(0.))

    def test_native_dropout_resets_streak_even_between_control_samples(self):
        recorder = TickRecorder(.001)
        for tick in range(1, 100):
            recorder.append(row(tick * .001))
        self.assertFalse(recorder.lift_seen)
        interrupted = row(.100, left=0)
        recorder.append(interrupted)
        self.assertEqual(interrupted['retained_lift_failure_reasons'], ['left_contact'])
        self.assertAlmostEqual(interrupted['interrupted_dwell_seconds'], .099)
        self.assertEqual(recorder.dwell, 0.)
        for tick in range(101, 201):
            recorder.append(row(tick * .001))
        self.assertTrue(recorder.lift_seen)
        recorder.append(row(.201, table=1, lift=.02))
        self.assertEqual(recorder.rows[-1]['retained_lift_failure_reasons'], ['table_support', 'lift_height'])
        self.assertTrue(recorder.lift_seen)  # original task latches a completed lift

    def test_sampling_miss_requires_both_observed_boundaries(self):
        rows = [row(tick * .001, action=(tick - 1) // 20,
                    left=0 if tick == 30 else 1) for tick in range(1, 61)]
        report = sampling_misses(rows)
        self.assertEqual(report['eligible_bilateral_boundary_intervals'], 2)
        self.assertEqual(report['intervals_hiding_native_contact_loss'], 1)
        self.assertEqual(report['intervals'][0]['missing_native_ticks'], 1)
        rows[39]['pads']['right']['scored_count'] = 0
        self.assertEqual(sampling_misses(rows)['intervals_hiding_native_contact_loss'], 0)

    def test_sampling_scope_does_not_treat_closure_as_loaded_motion(self):
        rows = [row(tick * .001, action=(tick - 1) // 20,
                    left=0 if tick == 30 else 1, phase='close') for tick in range(1, 41)]
        self.assertEqual(sampling_misses(rows)['intervals_hiding_native_contact_loss'], 1)
        self.assertEqual(sampling_misses(rows, ('lift', 'transfer'))['eligible_bilateral_boundary_intervals'], 0)

    def test_reproduction_rejects_intermediate_state_or_scoring_change(self):
        obs = {'object_position': [.1, .2, .3], 'jaw_actuator_force': [.3, .3]}
        info = {'simulation_time': .02, 'target_distance': .2, 'dwell_elapsed': 0.,
                'lift_dwell_elapsed': .01, 'lift_seen': False}
        expected = {**obs, **{key: value for key, value in info.items() if key != 'simulation_time'}, 'time': .02}
        compare_sample(obs, info, expected, 1)
        for key in ('object_position', 'jaw_actuator_force', 'lift_seen', 'lift_dwell_elapsed'):
            altered = deepcopy(expected)
            altered[key] = [9., 9., 9.] if isinstance(altered[key], list) else True
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                compare_sample(obs, info, altered, 1)
        del expected['jaw_actuator_force']
        with self.assertRaisesRegex(ValueError, 'Missing frozen physical'):
            compare_sample(obs, info, expected, 1)

    def test_native_probe_preserves_every_state_and_deduplicates_reads(self):
        config = ArmConfig(scene_jitter=0.)
        original, audited = ArmEnv(config), AuditedArmEnv(config)
        obs, info = original.reset(seed=317)
        measured, measured_info = audited.reset(seed=317)
        assert_equal(measured, obs, 'probe reset observation')
        assert_equal(measured_info, info, 'probe reset info')
        self.assertEqual(audited.recorder.rows, [])
        # Includes closing commands and target ramping, rather than only a
        # frozen zero-action observation; all state comparisons are exact.
        for index in range(4):
            action = {'joint_targets': [float(x + .005 * index) for x in obs['joint_position']],
                      'gripper_width': .06 - .01 * index}
            audited.action_context = {'action_index': index, 'phase': 'approach',
                                      'phase_command_started_at': 0., 'held_offset_m': None,
                                      'requested_gripper_width_m': action['gripper_width']}
            before = original.step(action)
            after = audited.step(action)
            assert_equal(after, before, 'probe complete step')
            obs = before[0]
            for _ in range(3):
                audited._contact_state()
                audited._observation()
                audited._info()
            self.assertEqual(len(audited.recorder.rows), (index + 1) * 20)
            assert_equal(audited.recorder.dwell, after[4]['lift_dwell_elapsed'], 'probe dwell')
            np.testing.assert_array_equal(audited.data.qpos, original.data.qpos)
            np.testing.assert_array_equal(audited.data.qvel, original.data.qvel)
        self.assertEqual(audited.recorder.rows[0]['native_tick'], 1)
        self.assertEqual(audited.recorder.rows[-1]['native_tick'], 80)

    def test_portable_frozen_input_rejects_tampering_and_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'inputs').mkdir()
            raw = b'{"exact original bytes": true}\n'
            packed = gzip.compress(raw, mtime=0)
            target = root / 'inputs' / 'sample.json.gz'
            target.write_bytes(packed)
            plan = {'id': 'sample', 'input_path': 'inputs/sample.json.gz',
                    'input_sha256': sha256(raw).hexdigest(), 'compressed_input_sha256': sha256(packed).hexdigest()}
            self.assertEqual(read_frozen_input(root, plan), raw)
            with self.assertRaisesRegex(ValueError, 'local episode ID'):
                read_frozen_input(root, {**plan, 'input_path': '../escape'})
            with self.assertRaisesRegex(ValueError, 'decompressed input'):
                read_frozen_input(root, {**plan, 'input_sha256': '0' * 64})
            target.write_bytes(packed + b'changed')
            with self.assertRaisesRegex(ValueError, 'compressed input'):
                read_frozen_input(root, plan)

    def test_scope_is_seven_fixed_tapes_and_seven_paired_controls(self):
        plans = planned_inputs('smoothing', 'compliance')
        self.assertEqual(len(plans), 14)
        self.assertEqual(len({identity for identity, _, _ in plans}), 14)
        self.assertEqual(sum(role == 'recorded_open_jaw_control' for _, _, role in plans), 7)
        self.assertEqual(sum('scene14' in identity for identity, _, _ in plans), 2)

    def test_completed_output_is_never_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'fresh directory'):
                run_audit('unused', 'unused', directory)


if __name__ == '__main__':
    unittest.main()

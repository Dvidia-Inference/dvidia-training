"""Checks for explicit interventions, unchanged baselines and causal telemetry."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from dvidia_training.arm_env import ArmConfig
from dvidia_training.arm_runner import rollout
from dvidia_training.grip_diagnostic import (ARMS, GoalProxy, _same_trajectory,
                                           rollout_diagnostic, summarize_episode, validate_arm)
from dvidia_training.grip_followup import ARMS as FOLLOWUP_ARMS, paired_factorial


class FakeEnvironment:
    def joint_targets_for_pose(self, goal):
        return np.asarray(goal).tolist()


class DiagnosticTests(unittest.TestCase):
    def proxy(self, phase, cap=.0025):
        proxy = GoalProxy(FakeEnvironment(), loaded_goal_cap_m=cap)
        proxy.observation = {'end_effector_position': [0., 0., 0.]}
        proxy.policy = SimpleNamespace(_action_phase=phase)
        return proxy

    def test_cap_changes_only_loaded_action_phases(self):
        for phase in ('lift', 'transfer', 'place'):
            proxy = self.proxy(phase)
            np.testing.assert_allclose(proxy.joint_targets_for_pose([.03, .04, 0.]), [.0015, .002, 0.])
            self.assertEqual(proxy.intervention_calls, 1)
        for phase in ('approach', 'descend', 'close', 'recover_wait', 'recover_up', 'release', 'retreat', 'verify'):
            proxy = self.proxy(phase)
            self.assertEqual(proxy.joint_targets_for_pose([.03, .04, 0.]), [.03, .04, 0.])
            self.assertEqual(proxy.intervention_calls, 0)

    def test_head_and_teacher_receive_same_intervened_goal(self):
        teacher = self.proxy('transfer')
        student = self.proxy('transfer')
        student.head = SimpleNamespace(predict=lambda observation, goal: list(goal))
        self.assertEqual(teacher.joint_targets_for_pose([0., 0., .02]),
                         student.joint_targets_for_pose([0., 0., .02]))

    def test_noop_cap_and_near_goal_are_unchanged(self):
        self.assertEqual(self.proxy('transfer', None).joint_targets_for_pose([0., .1, 0.]), [0., .1, 0.])
        self.assertEqual(self.proxy('transfer').joint_targets_for_pose([0., .001, 0.]), [0., .001, 0.])

    def test_invalid_caps_and_unplanned_factor_combinations_rejected(self):
        for cap in (True, 0, float('nan'), .02, 'slow'):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                self.proxy('transfer', cap)
        modified = deepcopy(ARMS[1])
        modified['horizon'] = 30.
        with self.assertRaises(ValueError):
            validate_arm(modified)

    def test_frozen_arms_each_change_one_factor(self):
        baseline = ARMS[0]
        for arm in ARMS[1:]:
            self.assertEqual(sum(arm[key] != baseline[key] for key in ('scene_overrides', 'horizon', 'loaded_goal_cap_m')), 1)
            self.assertEqual(validate_arm(arm), arm)
        copied = validate_arm(ARMS[1])
        copied['scene_overrides']['jaw_force_limit'] = 8
        self.assertEqual(ARMS[1]['scene_overrides']['jaw_force_limit'], .6)

    def test_noop_native_harness_reproduces_existing_authored_rollout(self):
        config = replace(ArmConfig(scene_jitter=0), horizon=.4)
        original = rollout(config, 317)
        diagnostic = rollout_diagnostic(config, 317)
        self.assertTrue(_same_trajectory(diagnostic, original))
        self.assertEqual(diagnostic['diagnostic_intervention_calls'], 0)
        changed = deepcopy(diagnostic)
        changed['actions'][0]['action']['gripper_width'] = .03
        self.assertFalse(_same_trajectory(changed, original))

    def test_retention_trigger_uses_pre_action_contacts_not_opened_jaws(self):
        config = replace(ArmConfig(scene_jitter=0), horizon=.4)
        episode = rollout_diagnostic(config, 317)
        frame = deepcopy(episode['trace'][-1])
        frame['stage'] = 'recover_wait'
        frame['grasp_contacts'] = {'left': 0, 'right': 1}
        frame['pre_action'].update(stage='transfer', next_stage='recover_wait',
                                  contacts={'left': 4, 'right': 4}, relative_offset_error_m=.0123)
        episode['trace'] = [frame]
        result = summarize_episode(episode)
        self.assertEqual(result['retention_transitions'][0]['contacts'], {'left': 4, 'right': 4})
        self.assertEqual(result['loaded_pre_action_bilateral_samples'], 1)
        self.assertIsNone(result['first_loaded_contact_or_offset_loss'])
        self.assertAlmostEqual(result['first_pre_action_retention_threshold_breach']['relative_offset_error_m'], .0123)

    def test_open_jaw_control_replays_arm_commands_without_supervisor(self):
        config = replace(ArmConfig(scene_jitter=0), horizon=.4)
        original = rollout_diagnostic(config, 317)
        control = rollout_diagnostic(config, 317, action_tape=original['actions'])
        self.assertEqual(len(original['actions']), len(control['actions']))
        for before, after in zip(original['actions'], control['actions']):
            self.assertEqual(before['action']['joint_targets'], after['action']['joint_targets'])
            self.assertEqual(after['action']['gripper_width'], .08)
        self.assertTrue(all(frame.get('pre_action') is None for frame in control['trace']))

    def test_followup_keeps_factorial_axes_and_exploration_explicit(self):
        cells = {(arm['scene_overrides'].get('jaw_force_limit', .3), arm['horizon']) for arm in FOLLOWUP_ARMS[:4]}
        self.assertEqual(cells, {(.3, 18.), (.6, 18.), (.3, 24.), (.6, 24.)})
        exploration = FOLLOWUP_ARMS[4]
        self.assertEqual(exploration['scene_overrides'], {})
        self.assertEqual(exploration['horizon'], 18.)
        self.assertEqual(exploration['loaded_goal_cap_m'], .004)

    def test_missing_factorial_cell_or_duplicate_is_rejected_and_errors_retained(self):
        rows = [{'arm_id': arm['id'], 'policy': policy, 'status': 'error', 'control_status': 'error'}
                for arm in FOLLOWUP_ARMS for policy in ('student', 'teacher')]
        summary = paired_factorial(rows)
        self.assertEqual(summary['student'][FOLLOWUP_ARMS[0]['id']]['status'], 'error')
        self.assertIsNone(summary['student'][FOLLOWUP_ARMS[0]['id']]['control_success'])
        for invalid in (rows[:-1], rows[:-1] + [rows[0]]):
            with self.assertRaises(ValueError):
                paired_factorial(invalid)


if __name__ == '__main__':
    unittest.main()

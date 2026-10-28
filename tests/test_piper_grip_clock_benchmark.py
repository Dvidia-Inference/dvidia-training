"""Synthetic runner contracts only: no fixtures claim native task accuracy."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dvidia_training import piper_grip_clock_benchmark as bench
from dvidia_training import piper_grip_clock as clock
from dvidia_training import piper_grip_clock_model as model
from dvidia_training import piper_grip_adapter as grip
from dvidia_training import piper_imitation as core
from tests.test_piper_feedback_model import base_model, observation


RUNTIME = {'synthetic-clock-runtime.py': 'a'*64}
NATIVE_SCORE = dict(valid=True, success=False, reason='synthetic_unsuccessful_attempt')
ANALYSIS = dict(valid=True, eligible=False, scheduled=False, activated=False, effective=False,
                interruption_condition=True, nonactivation_reason='eligibility_not_reached')
SCORE = dict(NATIVE_SCORE, post_grip_analysis=ANALYSIS)


def replace(path, value):
    Path(path).write_bytes(core.canonical(value))


def case(name, x, condition='nominal'):
    return dict(id=name, start_xy=[x, -.03], goal_xy=[.35, .08], seed=271000, condition=condition)


def synthetic_feature(t, progress):
    value = np.zeros(model.FEATURE_COUNT)
    value[model.ELAPSED_INDEX] = t/20.
    value[-2:] = progress/20., max(0., t-progress)/20.
    value[model.LIVE_START:model.LIVE_STOP] = .125
    return value


class FakePolicy:
    def __init__(self, *args, **kwargs):
        self.calls, self.progress = 0, 0.
        self.last_decision = None
        self.blind = kwargs.get('blind', False)

    def __call__(self, obs):
        increment = .02 if self.calls % 2 else 0.
        features = synthetic_feature(obs['t_seconds'], self.progress)
        if self.blind:
            features[model.LIVE_START:model.LIVE_STOP] = 0.
        self.last_decision = dict(physical_time_s=obs['t_seconds'], progress_before_s=self.progress,
            progress_after_s=self.progress+increment, increment_s=increment,
            features=features.tolist(), decision_origin='authored_causal_grip_teacher')
        self.progress += increment
        self.calls += 1
        return dict(joint_targets_rad=list(obs['joint_position_rad']), gripper_position_m=.0175)

    def diagnostics(self):
        return dict(synthetic_fixture=True, calls=self.calls)


class FakeEnv:
    instances = []

    def __init__(self, assets, config, **parameters):
        self.config, self.parameters = config, parameters
        self.steps, self.done = 0, False
        self.initial_object_position = np.asarray(config.object_position_m)
        self.model = SimpleNamespace(actuator_gainprm=np.zeros((7, 1)), actuator_forcerange=np.zeros((7, 2)))
        self.contact_pair_native_ticks = {}
        self.instances.append(self)

    def state(self):
        value = observation(self.steps*.02)
        value.update(object_position_m=list(self.initial_object_position),
                     issued_joint_targets_rad=None, issued_gripper_position_m=None, command=None)
        return value

    def reset(self, seed):
        return self.state(), {}

    def step(self, action):
        self.steps += 1
        self.done = self.steps >= 1000
        # Deliberately report success early: the runner must retain 20 seconds.
        return self.state(), dict(valid=True, success=True, reason='synthetic', simulation_time=self.steps*.02)

    def native_arrays(self):
        return {'synthetic_ticks': np.arange(self.steps, dtype=float)}

    def native_metadata(self):
        return {'post_grip_intervention': {'parameters': self.parameters}}


class GripClockRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = base_model()
        cls.jaw = grip.train(dict(features=np.zeros((2, grip.FEATURE_COUNT)),
            targets=np.zeros(2), groups=['synthetic']*2), base_sha256=core.digest(cls.base), min_leaf=2)
        cls.models = {}
        for blind in (False, True):
            cls.models[blind] = model.train(dict(features=np.zeros((2, model.FEATURE_COUNT)),
                targets=np.zeros(2), groups=['synthetic']*2), base_sha256=core.digest(cls.base),
                jaw_sha256=core.digest(cls.jaw), blind=blind, min_leaf=2)

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        bench.write(self.root/'base-model.json', self.base)
        bench.write(self.root/'jaw-model.json', self.jaw)
        self.paths = {}
        for blind, document in self.models.items():
            self.paths[blind] = self.root/('blind.json' if blind else 'clock.json')
            bench.write(self.paths[blind], document)
        self.protocol = dict(cases=dict(train=[case('train', .285)],
            development=[case('development', .290, 'post_grip')], confirmation=[case('confirmation', .286)]),
            teacher_config=asdict(clock.TeacherConfig()), post_grip_parameters=dict(delay_s=.02))
        bench.write(self.root/'protocol.json', self.protocol)
        for name, value in [('BASE_SHA', bench.sha(self.root/'base-model.json')),
                            ('JAW_SHA', bench.sha(self.root/'jaw-model.json'))]:
            handle = patch.object(bench, name, value)
            handle.start(); self.addCleanup(handle.stop)
        handle = patch.object(bench, 'runtime_hashes', return_value=deepcopy(RUNTIME))
        handle.start(); self.addCleanup(handle.stop)

    def fake_rollout(self, assets, selected, base, jaw, *, variant, clock_document=None,
                     teacher_config=None, intervention_parameters=None, collect_labels=False):
        actions = []
        if collect_labels:
            actions = [dict(t_seconds=0., joint_targets_rad=[0.]*6, gripper_position_m=0.,
                clock_decision=dict(features=synthetic_feature(0., 0.).tolist(), increment_s=0.,
                                    physical_time_s=0., progress_before_s=0., progress_after_s=0.,
                                    decision_origin='authored_causal_grip_teacher'))]
        episode = dict(schema=bench.EPISODE_SCHEMA, config=bench._canonical(asdict(bench.config(selected))),
            case_id=selected['id'], seed=selected['seed'], condition=selected['condition'], policy_kind=variant,
            base_model_sha256=core.digest(base), jaw_model_sha256=core.digest(jaw),
            jaw_mode=bench.jaw_contract(base, jaw)[0],
            model_sha256=core.digest(clock_document) if clock_document else None,
            teacher_config=teacher_config, post_grip_parameters=intervention_parameters,
            implementation_sha256=deepcopy(RUNTIME), actions=actions, trajectory=[], valid=False, success=False)
        metadata = dict(post_grip_intervention=dict(parameters=dict(
            interruption=selected['condition'] == 'post_grip', **intervention_parameters)))
        samples = bench._samples_from_episode(episode, selected['id']) if collect_labels else {}
        return episode, {'synthetic': np.asarray([1.])}, metadata, samples

    def test_literal_split_labels_variant_and_frozen_identity_guards(self):
        development = self.protocol['cases']['development'][0]
        binding = bench._request_binding(self.root, development, 'fixed')
        self.assertEqual(binding['split'], 'development')
        self.assertEqual(binding['post_grip_parameters']['delay_s'], .02)
        self.assertEqual(binding['runtime_sha256'], RUNTIME)
        changed = deepcopy(development); changed['seed'] += 1
        failures = [(changed, 'fixed', None, False), (development, 'teacher', None, True),
                    (development, 'fixed', None, True), (development, 'clock', None, False),
                    (development, 'fixed', self.paths[False], False),
                    (development, 'blind_clock', self.paths[False], False),
                    (self.protocol['cases']['confirmation'][0], 'fixed', None, False)]
        for selected, variant, path, labels in failures:
            with self.subTest(variant=variant, labels=labels), self.assertRaises(ValueError):
                bench._request_binding(self.root, selected, variant, path, labels)
        teacher = bench._request_binding(self.root, self.protocol['cases']['train'][0], 'teacher', collect_labels=True)
        self.assertTrue(teacher['collect_labels'])
        confirmation = self.protocol['cases']['confirmation'][0]
        frozen = dict(protocol_sha256=bench.sha(self.root/'protocol.json'), runtime_sha256=deepcopy(RUNTIME),
                      selected_sha256=bench.sha(self.paths[False]))
        bench.write(self.root/'frozen.json', frozen)
        self.assertEqual(bench._request_binding(self.root, confirmation, 'clock', self.paths[False])['split'], 'confirmation')
        frozen['runtime_sha256'] = {'changed.py': 'b'*64}
        replace(self.root/'frozen.json', frozen)
        with self.assertRaisesRegex(ValueError, 'Frozen confirmation'):
            bench._request_binding(self.root, confirmation, 'fixed')

    def test_cross_split_xy_and_duplicate_case_ids_are_rejected(self):
        selected = self.protocol['cases']['development'][0]
        for mutate in ('xy', 'id'):
            protocol = deepcopy(self.protocol)
            protocol['cases']['confirmation'][0][{'xy': 'start_xy', 'id': 'id'}[mutate]] = selected[{'xy': 'start_xy', 'id': 'id'}[mutate]]
            replace(self.root/'protocol.json', protocol)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                bench._request_binding(self.root, selected, 'fixed')

    def test_coordinated_jaw_manifest_is_bound_and_cannot_use_a_v1_clock(self):
        selected = self.protocol['cases']['development'][0]
        jaw = clock.reference_jaw_manifest(self.base)
        replace(self.root/'jaw-model.json', jaw)
        protocol = deepcopy(self.protocol)
        protocol.update(jaw_mode='coordinated_reference',
                        jaw_model_raw_sha256=bench.sha(self.root/'jaw-model.json'))
        replace(self.root/'protocol.json', protocol)
        binding = bench._request_binding(self.root, selected, 'fixed')
        self.assertEqual(binding['jaw_mode'], 'coordinated_reference')
        self.assertEqual(binding['jaw_model_sha256'], core.digest(jaw))
        self.assertEqual(bench.jaw_contract(self.base, jaw), ('coordinated_reference', jaw))
        v1 = model.train(dict(features=np.zeros((2, 58)), targets=np.zeros(2), groups=['synthetic']*2),
                         base_sha256=core.digest(self.base), jaw_sha256=core.digest(jaw), min_leaf=2)
        replace(self.paths[False], v1)
        with self.assertRaisesRegex(ValueError, 'schema differs'):
            bench._request_binding(self.root, selected, 'clock', self.paths[False])
        protocol['jaw_model_raw_sha256'] = 'b'*64
        replace(self.root/'protocol.json', protocol)
        with self.assertRaisesRegex(ValueError, 'Frozen jaw bytes'):
            bench._request_binding(self.root, selected, 'fixed')

    def test_immutable_complete_attempt_reuse_and_interrupted_or_unbound_refusal(self):
        selected = self.protocol['cases']['development'][0]
        folder = self.root/'attempt'
        with patch.object(bench, 'rollout', side_effect=self.fake_rollout), patch.object(
                bench, '_score', return_value=(deepcopy(SCORE), deepcopy(ANALYSIS))):
            score = bench.execute(self.root, 'unused', selected, 'fixed', folder)
        self.assertFalse(score['success'])
        self.assertEqual(bench.read(folder/'receipt.json')['binding']['case']['condition'], 'post_grip')
        self.assertFalse(score['post_grip_analysis']['activated'])
        original = (folder/'episode.json.gz').read_bytes()
        with patch.object(bench, 'rollout', side_effect=AssertionError('must not rerun physics')), patch.object(
                bench, '_score', return_value=(deepcopy(SCORE), deepcopy(ANALYSIS))):
            self.assertEqual(bench.execute(self.root, 'unused', selected, 'fixed', folder), SCORE)
            with self.assertRaises(ValueError):
                bench.execute(self.root, 'unused', selected, 'teacher', folder)
            interrupted = self.root/'interrupted'
            bench.write(interrupted/'started.json', {'retained': True})
            with self.assertRaisesRegex(ValueError, 'automatic rerun refused'):
                bench.execute(self.root, 'unused', selected, 'fixed', interrupted)
            unbound = self.root/'unbound'; unbound.mkdir()
            (unbound/'unknown.txt').write_text('retain')
            with self.assertRaisesRegex(ValueError, 'automatic rerun refused'):
                bench.execute(self.root, 'unused', selected, 'fixed', unbound)
        self.assertEqual((folder/'episode.json.gz').read_bytes(), original)
        (folder/'native.npz').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'completed attempt changed'):
            bench.execute(self.root, 'unused', selected, 'fixed', folder)

    def test_all_teacher_labels_retained_even_when_scored_unsuccessful(self):
        selected = self.protocol['cases']['train'][0]
        folder = self.root/'teacher-attempt'
        with patch.object(bench, 'rollout', side_effect=self.fake_rollout), patch.object(
                bench, '_score', return_value=(deepcopy(SCORE), deepcopy(ANALYSIS))):
            self.assertFalse(bench.execute(self.root, 'unused', selected, 'teacher', folder, collect_labels=True)['success'])
            samples = bench._load_npz(folder/'samples.npz')
            self.assertEqual(samples['features'].shape, (1, 58))
            self.assertEqual(samples['targets'].tolist(), [0.])
            self.assertEqual(samples['weights'].tolist(), [20.])
            self.assertEqual(samples['groups'].tolist(), [selected['id']])
            self.assertEqual(bench.execute(self.root, 'unused', selected, 'teacher', folder, collect_labels=True), SCORE)

    def test_request_source_or_protocol_changes_during_run_leave_unfinished_marker(self):
        selected = self.protocol['cases']['development'][0]
        folder = self.root/'changing-attempt'
        def changed(*args, **kwargs):
            result = self.fake_rollout(*args, **kwargs)
            protocol = deepcopy(self.protocol); protocol['changed_during_run'] = True
            replace(self.root/'protocol.json', protocol)
            return result
        with patch.object(bench, 'rollout', side_effect=changed), self.assertRaisesRegex(ValueError, 'changed during execution'):
            bench.execute(self.root, 'unused', selected, 'fixed', folder)
        self.assertTrue((folder/'started.json').exists())
        self.assertFalse((folder/'score.json').exists())

    def physics_stubs(self):
        return (patch('dvidia_training.piper_grip_clock_env.PostGripPiperTaskEnv', FakeEnv),
                patch('dvidia_training.robot_profiles.verify_assets', return_value={'synthetic_fixture': True}),
                patch('dvidia_training.reference_sim._native_imports', return_value=(SimpleNamespace(__version__='synthetic'), np)),
                patch('dvidia_training.reference_sim.model_digest', return_value='synthetic-model'))

    def test_every_student_variant_runs_full_duration_without_teacher_access(self):
        selected = self.protocol['cases']['development'][0]
        for variant in ('fixed', 'clock', 'blind_clock'):
            with (self.subTest(variant=variant),
                self.physics_stubs()[0],
                self.physics_stubs()[1], self.physics_stubs()[2], self.physics_stubs()[3],
                patch.object(clock, 'TeacherClock', side_effect=AssertionError('learner cannot access teacher')),
                patch.object(clock, 'FixedClock', FakePolicy), patch.object(clock, 'LearnedClock', FakePolicy)):
                episode, arrays, metadata, samples = bench.rollout('unused', selected, self.base, self.jaw,
                    variant=variant, clock_document=self.models[variant == 'blind_clock'] if variant != 'fixed' else None)
            self.assertIsNone(episode['error'], episode)
            self.assertEqual(len(episode['actions']), 1000)
            self.assertEqual(len(episode['trajectory']), 1001)
            self.assertEqual(episode['simulated_seconds'], 20.)
            self.assertEqual(samples, {})
            self.assertTrue(metadata['post_grip_intervention']['parameters']['interruption'])
            self.assertEqual(episode['timing']['inference']['samples'], 1000)
            self.assertEqual(episode['timing']['control_step']['samples'], 1000)
            for state, record in zip(episode['trajectory'][:-1], episode['actions']):
                self.assertEqual(state['command'], record)
                self.assertEqual(state['issued_joint_targets_rad'], record['joint_targets_rad'])
                self.assertEqual(state['issued_gripper_position_m'], record['gripper_position_m'])
            terminal = episode['trajectory'][-1]
            self.assertIsNone(terminal['command'])
            self.assertIsNone(terminal['issued_joint_targets_rad'])
            self.assertIsNone(terminal['issued_gripper_position_m'])
            self.assertEqual(episode['actions'][0]['clock_decision']['features'][8], 0. if variant == 'blind_clock' else .125)

    def test_teacher_progress_labels_are_exact_causal_full_and_blind_pairs(self):
        selected = self.protocol['cases']['train'][0]
        with (self.physics_stubs()[0], self.physics_stubs()[1], self.physics_stubs()[2], self.physics_stubs()[3],
              patch.object(clock, 'TeacherClock', FakePolicy)):
            episode, _, _, samples = bench.rollout('unused', selected, self.base, self.jaw,
                                                   variant='teacher', collect_labels=True)
        self.assertIsNone(episode['error'], episode)
        self.assertEqual(samples['features'].shape, (1000, 58))
        np.testing.assert_array_equal(samples['targets'], np.tile([0., .02], 500))
        np.testing.assert_array_equal(samples['weights'], np.tile([20., 1.], 500))
        np.testing.assert_array_equal(samples['blind_features'][:, 8:56], np.zeros((1000, 48)))
        np.testing.assert_array_equal(samples['features'][:, :8], samples['blind_features'][:, :8])
        np.testing.assert_array_equal(samples['features'][:, 56:], samples['blind_features'][:, 56:])
        self.assertEqual(set(samples['groups'].tolist()), {selected['id']})
        self.assertTrue(all(value.dtype.kind != 'O' for value in samples.values()))

    def test_invalid_intervention_audit_cannot_leave_a_successful_score(self):
        with (patch('dvidia_training.piper_feedback_score.score_feedback_episode', return_value=dict(valid=True, success=True, reason='success')),
              patch('dvidia_training.piper_grip_clock_env.analyze_post_grip', return_value=dict(valid=False, mismatches=['synthetic']))):
            score, _ = bench._score({'condition': 'post_grip'}, {}, {}, [.35, .08])
        self.assertFalse(score['valid'])
        self.assertFalse(score['success'])
        self.assertEqual(score['native_score_reason'], 'success')
        self.assertEqual(score['reason'], 'post_grip_intervention_mismatch')

    def test_teacher_label_clock_alignment_and_blind_mask_tampering_are_rejected(self):
        action = dict(t_seconds=0., clock_decision=dict(
            physical_time_s=0., progress_before_s=0., progress_after_s=.02, increment_s=.02,
            features=synthetic_feature(0., 0.).tolist(), decision_origin='authored_causal_grip_teacher'))
        for field, value in [('physical_time_s', .02), ('progress_before_s', .02),
                             ('progress_after_s', .04), ('increment_s', .01)]:
            changed = deepcopy(action); changed['clock_decision'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                bench._samples_from_episode({'actions': [changed]}, 'synthetic')
        valid = bench._samples_from_episode({'actions': [action]}, 'synthetic')
        valid['blind_features'][0, 8] = 1.
        with self.assertRaisesRegex(ValueError, 'live/history slots'):
            bench._sample_arrays(valid, 'synthetic')


if __name__ == '__main__':
    unittest.main()

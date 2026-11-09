"""Publication-card checks use fabricated sealed receipts, never native physics."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import render_camera_action_card as cards
from dvidia_training import camera_action_study as study


class CardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / 'study'
        self.fixed = self.base / 'fixed.json'
        study.write(self.fixed, {'fixture': 'fixed comparator'})
        with patch.object(study, 'runtime_hashes', return_value={'fixture.py': '0' * 64}), \
             patch('dvidia_training.piper_motor_primitive.validate_model', side_effect=lambda value: value):
            study.prepare(self.root, fixed_model=self.fixed)
        protocol = study.read(self.root / 'protocol.json')
        rows = [{**deepcopy(a), 'status': 'unattempted', 'complete': False, 'strict_success': False,
            'score': {}, 'path': None, 'receipt_sha256': None} for a in protocol['rosters']['diagnostic']]
        receipt = dict(schema=study.SCHEMA + '/stage', stage='diagnostic', status='blocked',
            reason='Synthetic fixture: capture budget unavailable.', scheduled=4, strict_successes=0,
            protocol_sha256=study.file_hash(self.root / 'protocol.json'), rows=rows)
        study.write(self.root / 'stages/diagnostic.started.json', {'fixture': True})
        study._seal_stage(self.root, 'diagnostic', receipt)
        study.write(self.root / 'report.json', study.report_study(self.root))
        self.summary = self.base / 'summary.json'
        study.write(self.summary, cards.recompute_summary(self.root))

    def render(self, **kwargs):
        return cards.render(self.root, self.base / 'cards', summary=self.summary,
            source_commit='a' * 40, **kwargs)

    def test_blocked_real_seals_render_honest_budget_missing_metrics_and_local_receipt(self):
        result = self.render()
        text = (self.base / 'cards/README.md').read_text()
        self.assertEqual(result['qualification'], 'research_candidate')
        self.assertIn('license: mit\ntask_categories:\n- robotics', text)
        self.assertIn('Research candidate — blocked', text)
        self.assertIn('452 attempts', text)
        self.assertIn('0 / 4', text)
        self.assertIn('No completed model fit is sealed before the blocker', text)
        self.assertIn('No full-study independent audit was supplied', text)
        self.assertIn('Not applicable', text)
        self.assertIn('not reported', text)
        self.assertIn('https://github.com/Dvidia-Inference/dvidia-training/tree/' + 'a' * 40, text)
        self.assertIn('shasum -a256 -c SHA256SUMS', text)
        self.assertIn('dvidia-train robot inspect', text)
        self.assertNotIn('`summary.json`', text)
        self.assertNotIn(str(self.root), text)
        receipt = study.read(self.base / 'cards/card-receipt.json')
        self.assertFalse(receipt['full_study_audited'])
        self.assertFalse(receipt['qualified'])
        self.assertFalse(receipt['physics_executed'])
        self.assertTrue(receipt['summary_reconstructed'])
        self.assertEqual(receipt['files']['README.md'], cards.evidence.identity(self.base / 'cards/README.md'))
        self.assertEqual((self.base / 'cards/summary.json').read_bytes(), self.summary.read_bytes())
        self.assertEqual(receipt['source_inputs']['summary.json'], cards.evidence.identity(self.base / 'cards/summary.json'))

    def test_fabricated_success_or_latency_summary_is_refused_before_output(self):
        value = study.read(self.summary)
        value['controllers'][0]['strict_successes'] = 4
        self.summary.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'fresh read-only reconstruction'):
            self.render()
        self.assertFalse((self.base / 'cards').exists())
        value = cards.recompute_summary(self.root)
        value['controllers'][0]['timing']['full_boundary_wall_seconds']['p95'] = .00001
        self.summary.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'fresh read-only reconstruction'):
            self.render()

    def test_retained_summary_growth_is_refused_before_writing_excess_bytes(self):
        original = self.summary.read_bytes()
        identity = cards.evidence.identity
        def grow_after_identity(path, **kwargs):
            record = identity(path, **kwargs)
            if Path(path) == self.summary:
                self.summary.write_bytes(original + b' ' * (1024 * 1024))
            return record
        with patch.object(cards.evidence, 'identity', side_effect=grow_after_identity), \
                self.assertRaisesRegex(ValueError, 'Summary grew'):
            self.render()
        self.assertLessEqual((self.base / 'cards/summary.json').stat().st_size, len(original))

    def test_running_worker_and_unsealed_partial_study_are_not_publishable(self):
        (self.root / 'worker.lock').write_text('{}')
        with patch.object(study, 'inspect') as inspect, self.assertRaisesRegex(ValueError, 'worker'):
            self.render()
        inspect.assert_not_called()
        (self.root / 'worker.lock').unlink()
        state = study.inspect(self.root)
        state['status'] = 'ready'
        with patch.object(study, 'inspect', return_value=state), self.assertRaisesRegex(ValueError, 'explicitly blocked'):
            self.render()

    def test_complete_state_requires_full_study_audit_even_when_gates_fail(self):
        state = study.inspect(self.root)
        state['status'] = 'complete'
        state['report']['status'] = 'complete'
        (self.root / 'report.json').write_bytes(study.canonical(state['report']))
        with patch.object(study, 'inspect', return_value=state), self.assertRaisesRegex(ValueError, 'full-study audit'):
            self.render()

    def test_mismatched_report_or_stage_seal_cannot_render(self):
        report_path = self.root / 'report.json'
        report = study.read(report_path)
        report['qualified'] = True
        report_path.write_bytes(study.canonical(report))
        with self.assertRaisesRegex(ValueError, 'Saved report differs'):
            self.render()
        report_path.write_bytes(study.canonical(study.report_study(self.root)))
        stage_path = self.root / 'stages/diagnostic.json'
        stage_path.write_bytes(stage_path.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'receipt identity changed'):
            self.render()

    def test_output_is_fresh_and_outside_study(self):
        self.render()
        with self.assertRaisesRegex(ValueError, 'fresh cards'):
            self.render()
        with self.assertRaisesRegex(ValueError, 'outside the study'):
            cards.render(self.root, self.root / 'cards', summary=self.summary, source_commit='a' * 40)

    def test_hardware_requires_whitelisted_public_fields(self):
        hardware = self.base / 'hardware.json'
        study.write(hardware, {'hostname': 'private'})
        with self.assertRaisesRegex(ValueError, 'public field whitelist'):
            self.render(hardware=hardware)

    def test_supplied_audit_failure_cannot_be_ignored_on_a_blocked_study(self):
        audit = self.base / 'audit.json'
        study.write(audit, dict(schema='dvidia.camera-action-study-audit/v1', passed=False, errors=['fixture']))
        with self.assertRaisesRegex(ValueError, 'Full-study audit does not pass'):
            self.render(study_audit=audit)
        self.assertFalse((self.base / 'cards').exists())

    def test_static_methodology_cannot_describe_different_registered_capture(self):
        protocol = study.read(self.root / 'protocol.json')
        protocol['capture']['camera_hz'] = 20
        with patch.object(cards.evidence, 'read', side_effect=lambda path: protocol if Path(path).name == 'protocol.json'
                          else study.read(path)), self.assertRaisesRegex(ValueError, 'methodology'):
            self.render()

    def test_blocked_study_cannot_silently_omit_interrupted_fit_prefix(self):
        prefix = self.root / 'models/initial/full'
        prefix.mkdir(parents=True)
        study.write(prefix / 'started.json', {'stage': 'initial', 'variant': 'full'})
        with self.assertRaisesRegex(ValueError, 'Unsealed or orphan model'):
            self.render()
        (prefix / 'started.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Unsealed or orphan model'):
            self.render()

    def test_model_header_alone_cannot_claim_a_trained_network(self):
        models = {}
        for variant in ('full', 'blind'):
            prefix = self.root / 'models/initial' / variant
            study.write(prefix / 'model.json', dict(schema='dvidia.camera-action-mlp/v1',
                hidden_units=[64, 64], blind=variant == 'blind', physical_robot_ready=False))
            record = dict(path=f'models/initial/{variant}/model.json',
                model_sha256=study.file_hash(prefix / 'model.json'), model_bytes=(prefix / 'model.json').stat().st_size,
                training_episodes=[], development_episodes=[])
            study.write(prefix / 'receipt.json', record)
            study.write(prefix / 'started.json', {'fixture': True})
            models[variant] = record
        stage = dict(stage='train-initial', status='complete', models=models,
            protocol_sha256=study.file_hash(self.root / 'protocol.json'))
        study.write(self.root / 'stages/train-initial.started.json', {'fixture': True})
        study._seal_stage(self.root, 'train-initial', stage)
        self.summary.write_bytes(study.canonical(cards.recompute_summary(self.root)))
        with self.assertRaisesRegex(ValueError, 'Unsupported camera-action model schema'):
            self.render()

    def test_phase_coverage_requires_bound_audit_and_reconstruction(self):
        audit = self.base / 'diagnostic-audit.json'
        study.write(audit, {'fixture': 'diagnostic proof'})
        phase = dict(schema='dvidia.camera-action-phase-coverage/v1',
            protocol_sha256=study.file_hash(self.root / 'protocol.json'),
            diagnostic_audit_sha256=study.file_hash(audit), manipulation_available=0,
            manipulation_decisions=749, manipulation_phases=['close', 'lift', 'transfer', 'place', 'release'], cases=4)
        phase_path = self.base / 'phase.json'
        study.write(phase_path, phase)
        with self.assertRaisesRegex(ValueError, 'independently audited diagnostic'):
            self.render(phase_coverage=phase_path)
        with patch.object(cards, 'recompute_phase', return_value=phase) as recompute:
            self.render(phase_coverage=phase_path, diagnostic_audit=audit)
        recompute.assert_called_once_with(self.root, audit)
        text = (self.base / 'cards/README.md').read_text()
        self.assertIn('0 / 749 visual estimates', text)
        self.assertIn('diagnostic subset, not student confirmation', text)
        phase['manipulation_available'] = 749
        phase_path.write_bytes(study.canonical(phase))
        with patch.object(cards, 'recompute_phase', return_value={**phase, 'manipulation_available': 0}), \
             self.assertRaisesRegex(ValueError, 'Diagnostic phase coverage differs'):
            cards.render(self.root, self.base / 'cards-other', summary=self.summary, source_commit='a' * 40,
                phase_coverage=phase_path, diagnostic_audit=audit)

    def test_complete_failed_gates_and_model_identities_are_visible(self):
        data = cards._inputs(self.root, summary=self.summary, source_commit='a' * 40)
        data['report'].update(status='complete', gates={'at_least_58_of_64': False,
            'no_critical_contact_increase': True}, selected_stage='initial')
        training = dict(training_wall_seconds=12.5, shared_export_wall_seconds=3.0,
            process_peak_rss_bytes=125000000, memory_scope='Process high-water RSS including export and previous fit.',
            training_episodes=['training-0'], development_episodes=['dev-0'])
        provenance = dict(stage='initial', sample_count=1000, train_groups=['train-0'],
            dataset_sha256='1' * 64, provenance_sha256='2' * 64, model_source_sha256='3' * 64)
        data['models'] = [dict(stage='initial', variant=variant, selected=True,
            path=f'models/initial/{variant}/model.json', identity={'bytes': 471234, 'sha256': digit * 64},
            training=training, model={'training': provenance}) for variant, digit in [('full', '4'), ('blind', '5')]]
        text = cards.dataset_card(data)
        self.assertIn('Research candidate — qualification gates failed', text)
        self.assertIn('at_least_58_of_64 | FAIL', text)
        self.assertIn('no_critical_contact_increase | PASS', text)
        self.assertIn('4' * 64, text)
        self.assertIn('5' * 64, text)
        self.assertIn('models/initial/full/model.json', text)
        self.assertIn('simulation candidate', text)
        self.assertIn('[selected full-feedback model](models/initial/full/model.json)', text)
        self.assertIn("pip install '/path/to/downloaded-wheel.whl[arm]'", text)
        self.assertIn("printf '%s  model.json\\n' '" + '4'*64 + "' | shasum -a256 -c -", text)
        self.assertIn('The raw episode corpus and source archive are not needed', text)
        self.assertLess(text.index('## Try the selected simulation candidate'), text.index('## Reproduce the complete study'))
        self.assertEqual(text.count('dvidia-arm install --model'), 1)
        model = cards.model_card(data, data['models'][1])
        self.assertIn('matched blind baseline', model)
        self.assertIn('live and historical sensor-feedback tensor channels are masked', model)
        self.assertIn('12.5', model)
        self.assertIn('1' * 64, model)
        self.assertNotIn('This is the selected full-feedback model', model)
        data['report'].update(qualified=True, qualification='bounded_simulation_skill')
        for record in (data['models'][1], {**data['models'][0], 'selected': False}):
            text = cards.model_card(data, record)
            self.assertIn('no separately awarded skill qualification', text)
            self.assertNotIn('Bounded simulation skill: all registered gates passed', text)

    def test_text_values_cannot_add_html_or_new_table_rows(self):
        self.assertEqual(cards._cell('<script>|`\nnext'), '&lt;script&gt;\\|&#96; next')
        data = cards._inputs(self.root, summary=self.summary, source_commit='a' * 40)
        data['receipts']['diagnostic']['reason'] = '<img src=x>|\nnew row'
        text = cards.dataset_card(data)
        self.assertNotIn('<img', text)

    def test_equal_gate_results_render_identically_after_json_key_reordering(self):
        data = cards._inputs(self.root, summary=self.summary, source_commit='a' * 40)
        data['report'].update(status='complete', gates={
            'all_registered_attempts_accounted': True,
            'at_least_58_of_64': False,
            'positive_paired_comparisons': False,
            'positive_common_admitted_comparisons': False,
            'no_critical_contact_increase': True,
            'all_negative_controls_correct': True,
            'no_privileged_features_or_teacher_intervention': True})
        original = cards.dataset_card(data).encode()
        reordered = deepcopy(data)
        reordered['report']['gates'] = dict(reversed(list(data['report']['gates'].items())))
        canonical = deepcopy(data)
        canonical['report'] = json.loads(json.dumps(data['report'], sort_keys=True))
        self.assertEqual(data['report'], reordered['report'])
        self.assertEqual(data['report'], canonical['report'])
        self.assertEqual(original, cards.dataset_card(reordered).encode())
        self.assertEqual(original, cards.dataset_card(canonical).encode())


if __name__ == '__main__':
    unittest.main()

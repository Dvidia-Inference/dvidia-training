"""Capture declarations cannot fabricate independence, training or qualification."""
from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from dvidia_training.coverage import (CoverageError, MAX_CASES, MAX_EVIDENCE,
                                     MAX_PLAN_BYTES, example_plan, main,
                                     summarize_plan, validate_plan)


def plan_with(*evidence):
    plan = example_plan()
    plan['cases'] = [plan['cases'][0]]
    plan['cases'][0]['required_groups'] = 3
    plan['evidence'] = list(evidence)
    return plan


def attempt(identity, **changes):
    row = copy.deepcopy(example_plan()['evidence'][0])
    row.update(id=identity, media_sha256=f'{int(identity[-1]):064x}',
               source_recording_id='recording-' + identity,
               session_id='session-' + identity, object_instance_id='object-' + identity)
    row.update(changes)
    return row


class CoverageTests(unittest.TestCase):
    def test_example_is_fabricated_and_summarizes_planning_only(self):
        summary = summarize_plan(example_plan())
        self.assertEqual(summary['coverage'], {'cells_met': 2, 'cells_total': 6,
                                             'accepted_groups': 4, 'submitted_attempts': 6,
                                             'accepted_attempts': 4, 'excluded_attempts': 2,
                                             'all_cells_met': False})
        self.assertEqual(summary['next_capture']['case_id'], 'recovery')
        self.assertFalse(summary['robot_skill_acquired'])
        self.assertFalse(summary['physical_robot_ready'])
        self.assertFalse(summary['benchmark']['executed'])
        self.assertFalse(summary['benchmark']['qualified'])
        self.assertFalse(summary['recognition']['payments_active'])
        self.assertFalse(summary['recognition']['awards_issued'])
        self.assertTrue(any('fabricated' in message for message in summary['warnings']))
        self.assertTrue(any('not scientifically sufficient' in message for message in summary['warnings']))
        self.assertEqual(summary['outcomes']['recovery'], {'submitted': 1, 'accepted': 0})

    def test_aliases_and_cuts_do_not_manufacture_distinct_groups(self):
        rows = [attempt('e1'), attempt('e2'), attempt('e3')]
        rows[1]['media_sha256'] = rows[0]['media_sha256']
        rows[2]['source_recording_id'] = rows[1]['source_recording_id']
        summary = summarize_plan(plan_with(*rows))
        self.assertEqual(summary['coverage']['accepted_groups'], 1)
        self.assertEqual(summary['cells'][0]['accepted_groups'], 1)
        self.assertEqual(summary['cells'][0]['gap'], 2)
        self.assertFalse(summary['coverage']['all_cells_met'])
        self.assertEqual(summary['recognition']['accepted_contributions'], 1)
        self.assertEqual(summary['recognition']['eligible_evidence_records'], 3)

    def test_conflicting_reviewed_outcomes_for_identical_media_cannot_be_cherry_picked(self):
        first, second = attempt('e1'), attempt('e2', outcome='failure')
        second['media_sha256'] = first['media_sha256']
        summary = summarize_plan(plan_with(first, second))
        self.assertEqual(summary['coverage']['accepted_groups'], 0)
        self.assertTrue(all('conflicting accepted human outcome' in text for text in summary['exclusions']))
        second['review_state'] = 'proposed'
        second['review_origin'] = 'ai'
        summary = summarize_plan(plan_with(first, second))
        self.assertEqual(summary['coverage']['accepted_groups'], 1)

    def test_transitive_recording_session_object_connections_are_one_group(self):
        rows = [attempt('e1'), attempt('e2'), attempt('e3'), attempt('e4')]
        rows[1]['source_recording_id'] = rows[0]['source_recording_id']
        rows[2]['session_id'] = rows[1]['session_id']
        rows[3]['object_instance_id'] = rows[2]['object_instance_id']
        summary = summarize_plan(plan_with(*rows))
        self.assertEqual(summary['coverage']['accepted_groups'], 1)
        self.assertEqual(summary['cells'][0]['accepted_groups'], 1)

    def test_unchanged_text_across_different_identity_names_does_not_connect_fields(self):
        first, second = attempt('e1'), attempt('e2')
        first['session_id'] = second['object_instance_id']
        summary = summarize_plan(plan_with(first, second))
        self.assertEqual(summary['coverage']['accepted_groups'], 2)

    def test_unknown_lineage_never_satisfies_a_quota(self):
        for field in ('source_recording_id', 'session_id', 'object_instance_id'):
            with self.subTest(field=field):
                summary = summarize_plan(plan_with(attempt('e1', **{field: None})))
                self.assertEqual(summary['coverage']['accepted_groups'], 0)
                self.assertEqual(summary['coverage']['excluded_attempts'], 1)
                self.assertIn('unknown recording, session or object lineage', summary['exclusions'][0])

    def test_partial_proposed_ai_rejected_rights_and_stale_evidence_excluded(self):
        variations = [dict(complete=False), dict(review_state='proposed'), dict(review_origin='ai'),
                      dict(review_origin='unknown'), dict(review_state='rejected'),
                      dict(review_state='needs_changes'), dict(review_state='stale'),
                      dict(rights_status='unknown'), dict(rights_status='denied'),
                      dict(plan_revision=2), dict(split='unassigned'), dict(outcome='unknown'),
                      dict(outcome='failure'), dict(case_id='old-case')]
        for changes in variations:
            with self.subTest(changes=changes):
                summary = summarize_plan(plan_with(attempt('e1', **changes)))
                self.assertEqual(summary['coverage']['accepted_groups'], 0)
                self.assertEqual(summary['coverage']['excluded_attempts'], 1)
                self.assertIsNotNone(summary['next_capture'])

    def test_confirmation_reservation_excludes_entire_connected_group(self):
        first, reserved, bridge = attempt('e1'), attempt('e2', split='confirmation'), attempt('e3')
        bridge['session_id'] = first['session_id']
        bridge['object_instance_id'] = reserved['object_instance_id']
        summary = summarize_plan(plan_with(first, reserved, bridge))
        self.assertEqual(summary['coverage']['accepted_groups'], 0)
        self.assertEqual(summary['coverage']['excluded_attempts'], 3)
        self.assertTrue(all('reserved for confirmation' in text for text in summary['exclusions']))

    def test_train_development_conflict_excludes_group_even_if_labels_accepted(self):
        first, second = attempt('e1'), attempt('e2', split='dev')
        second['session_id'] = first['session_id']
        summary = summarize_plan(plan_with(first, second))
        self.assertEqual(summary['coverage']['accepted_groups'], 0)
        self.assertTrue(all('crosses train and development' in text for text in summary['exclusions']))

    def test_pending_or_stale_alias_cannot_hide_a_confirmation_reservation(self):
        first, holdout = attempt('e1'), attempt('e2', split='confirmation', review_state='proposed', plan_revision=7)
        holdout['media_sha256'] = first['media_sha256']
        summary = summarize_plan(plan_with(first, holdout))
        self.assertEqual(summary['coverage']['accepted_groups'], 0)

    def test_met_targets_do_not_claim_data_sufficiency_or_robot_qualification(self):
        plan = plan_with(attempt('e1'), attempt('e2'), attempt('e3'))
        summary = summarize_plan(plan)
        self.assertTrue(summary['coverage']['all_cells_met'])
        self.assertEqual(summary['cells'][0]['status'], 'met')
        self.assertIsNone(summary['next_capture'])
        self.assertFalse(summary['robot_skill_acquired'])
        self.assertFalse(summary['benchmark']['qualified'])
        self.assertTrue(any('does not establish training readiness' in text for text in summary['warnings']))

    def test_priority_ties_are_stable_and_unknown_cases_require_definition(self):
        plan = example_plan()
        plan['evidence'] = []
        plan['cases'][0].update(priority=10, id='aaa', outcome='unknown')
        summary = summarize_plan(plan)
        self.assertEqual(summary['next_capture']['case_id'], 'aaa')
        self.assertIn('Define this case', summary['next_capture']['why'])
        self.assertTrue(any('unknown intended outcome' in text for text in summary['warnings']))

    def test_benchmark_is_mandatory_and_passed_or_qualified_cannot_be_declared(self):
        mutations = [lambda plan: plan.pop('benchmark'),
                     lambda plan: plan['benchmark'].update(status='passed'),
                     lambda plan: plan['benchmark'].update(executed=True),
                     lambda plan: plan['benchmark'].update(qualified=True),
                     lambda plan: plan.update(robot_skill_acquired=True),
                     lambda plan: plan['benchmark'].update(environment=''),
                     lambda plan: plan['benchmark'].update(success_predicate=''),
                     lambda plan: plan['benchmark'].update(failure_cases=[])]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                plan = example_plan()
                mutate(plan)
                with self.assertRaises(CoverageError):
                    validate_plan(plan)
        plan = example_plan()
        plan['benchmark'].update(status='blocked', environment='', reason='No compatible shoe adapter exists.',
                                 success_predicate='', failure_cases=[])
        summary = summarize_plan(plan)
        self.assertEqual(summary['benchmark']['status'], 'blocked')
        self.assertIn('shoe adapter', summary['benchmark']['reason'])
        plan['benchmark']['reason'] = ''
        with self.assertRaises(CoverageError):
            validate_plan(plan)

    def test_strict_fields_types_bounds_and_unique_ids(self):
        mutations = [lambda plan: plan.update(schema_version=True),
                     lambda plan: plan.update(plan_revision=True),
                     lambda plan: plan['cases'][0].update(required_groups=True),
                     lambda plan: plan['cases'][0].update(priority=11),
                     lambda plan: plan['cases'][0].update(brief=''),
                     lambda plan: plan['skill'].update(goal='\x00'),
                     lambda plan: plan['evidence'][0].update(complete=1),
                     lambda plan: plan['evidence'][0].update(media_sha256='A'*64),
                     lambda plan: plan['evidence'][0].update(session_id=''),
                     lambda plan: plan['evidence'][0].update(arbitrary_path='/tmp/file'),
                     lambda plan: plan['cases'].append(copy.deepcopy(plan['cases'][0])),
                     lambda plan: plan['evidence'].append(copy.deepcopy(plan['evidence'][0])),
                     lambda plan: plan.update(cases=[copy.deepcopy(plan['cases'][0])] * (MAX_CASES + 1)),
                     lambda plan: plan.update(evidence=[copy.deepcopy(plan['evidence'][0])] * (MAX_EVIDENCE + 1))]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                plan = example_plan()
                mutate(plan)
                with self.assertRaises(CoverageError):
                    validate_plan(plan)

    def test_json_rejects_duplicates_nonfinite_unicode_and_oversized_input(self):
        for raw in ('{"format":1,"format":2}', '{"x":NaN}', '{"x":1e999}', b'\xff',
                    ' ' * (MAX_PLAN_BYTES + 1), {'x': float('nan')}, {'x': '\ud800'}):
            with self.subTest(raw=str(raw)[:40]):
                with self.assertRaises(CoverageError):
                    validate_plan(raw)

    def test_normalization_is_detached_and_summary_hash_is_deterministic(self):
        original = example_plan()
        normalized = validate_plan(original)
        normalized['benchmark']['failure_cases'].append('Changed elsewhere')
        self.assertNotIn('Changed elsewhere', original['benchmark']['failure_cases'])
        raw = json.dumps(original, sort_keys=True).encode()
        self.assertEqual(summarize_plan(original), summarize_plan(raw))

    def test_cli_example_inspect_bounded_read_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'plan.json'
            summary_path = Path(directory) / 'summary.json'
            self.assertEqual(main(['example', '--output', str(path)]), 0)
            self.assertEqual(main(['inspect', str(path), '--output', str(summary_path)]), 0)
            summary = json.loads(summary_path.read_text())
            self.assertFalse(summary['benchmark']['executed'])
            original = path.read_bytes()
            error = io.StringIO()
            with contextlib.redirect_stderr(error):
                self.assertEqual(main(['example', '--output', str(path)]), 1)
            self.assertEqual(path.read_bytes(), original)
            self.assertIn('Coverage planning failed', error.getvalue())
            path.write_bytes(b' ' * (MAX_PLAN_BYTES + 1))
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(['inspect', str(path)]), 1)


if __name__ == '__main__':
    unittest.main()

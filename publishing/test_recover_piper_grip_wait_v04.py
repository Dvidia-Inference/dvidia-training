"""Interruption bookkeeping and no-retry contracts, without native simulation."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from publishing import recover_piper_grip_wait_v04 as recovery


def fixture(root):
    cases=[]
    for xy in range(32):
        for goal in range(2):
            for condition in ('nominal','post_grip'):
                cases.append(dict(id=f'confirmation-xy{xy:02}-g{goal}-{condition}',start_xy=[xy*.001,0.],goal_xy=[goal,0.],condition=condition))
    protocol=dict(cases=dict(confirmation=cases))
    records=[]
    for i,(case,variant,folder) in enumerate(recovery.requests(root,protocol)):
        interrupted=case['id']==recovery.INCOMPLETE_CASE
        state='interrupted' if interrupted else 'complete' if i<72 else 'never_started'
        records.append(dict(case_id=case['id'],variant=variant,status=state,relative_path=folder.relative_to(root).as_posix()))
        if state!='never_started':
            folder.mkdir(parents=True)
            (folder/('receipt.json' if state=='complete' else 'started.json')).write_text('{}')
    return protocol,dict(attempts=records,config=dict(maximum_wait_s=.20),original_inventory=recovery.inventory(root))


def outcome_rows():
    rows=[]
    for xy in range(32):
        for goal in range(2):
            for condition in ('nominal','post_grip'):
                case=f'confirmation-xy{xy:02}-g{goal}-{condition}'
                for variant in recovery.bench.VARIANTS:
                    missing=case==recovery.INCOMPLETE_CASE
                    success=not missing and (xy%3!=0 if variant=='guarded' else xy%3==0)
                    rows.append(dict(case_id=case,start_xy=[xy*.001,0.],condition=condition,variant=variant,
                        interrupted_unevaluable=missing,success=success,raw_strict_success=success,valid=not missing,
                        aborted=False,acquisition_fail=not success,retention_fail=False,critical_contact=False,
                        session='original' if xy<4 else 'resumed',timing=dict(inference_samples_s=[.001],control_step_samples_s=[.003],wall_seconds=1.)))
    return rows


class RecoveryTests(unittest.TestCase):
    def test_jobs_include_only_never_started_436_and_never_complete_or_partial(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);protocol,plan=fixture(root)
            jobs=recovery.never_started_jobs(root,'assets',protocol,plan)
            self.assertEqual(len(jobs),436)
            self.assertFalse(any(job[4].exists() for job in jobs))
            self.assertFalse(any(job[2]['id']==recovery.INCOMPLETE_CASE for job in jobs))
            self.assertTrue(all(job[5]==dict(maximum_wait_s=.20) for job in jobs))

    def test_started_after_plan_is_never_retried_even_when_empty(self):
        for empty in (False,True):
            with self.subTest(empty=empty),TemporaryDirectory() as temp:
                root=Path(temp);protocol,plan=fixture(root)
                folder=recovery.never_started_jobs(root,'assets',protocol,plan)[0][4]
                folder.mkdir(parents=True)
                if not empty:(folder/'started.json').write_text('{}')
                with self.assertRaisesRegex(ValueError,'never be retried'):
                    recovery.never_started_jobs(root,'assets',protocol,plan)

    def test_original_complete_and_partial_bytes_are_preserved_exactly(self):
        for target in ('receipt.json','started.json'):
            with self.subTest(target=target),TemporaryDirectory() as temp:
                root=Path(temp);protocol,plan=fixture(root)
                path=next(root.rglob(target));path.write_text('{"tampered":true}')
                with self.assertRaisesRegex(ValueError,'Original evidence changed'):
                    recovery.check_original_inventory(root,plan)

    def test_only_predeclared_new_attempt_subtrees_may_be_added(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);protocol,plan=fixture(root)
            folder=recovery.never_started_jobs(root,'assets',protocol,plan)[0][4]
            folder.mkdir(parents=True);(folder/'started.json').write_text('{}')
            recovery.check_original_inventory(root,plan)
            with self.assertRaisesRegex(ValueError,'Unplanned evidence'):
                recovery.check_original_inventory(root,plan,exact=True)
            (root/'report.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Unplanned evidence'):
                recovery.check_original_inventory(root,plan)

    def test_symlinks_cannot_hide_changed_original_evidence(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);protocol,plan=fixture(root)
            path=next(root.rglob('started.json'));source=root/'external';source.write_bytes(path.read_bytes())
            path.unlink();path.symlink_to(source)
            with self.assertRaisesRegex(ValueError,'symlinks'):
                recovery.check_original_inventory(root,plan)

    def test_scheduled_denominator_512_retains_four_unknowns_not_measured_failures(self):
        rows=outcome_rows();totals=recovery.aggregate(rows)
        self.assertEqual(sum(v['scheduled'] for v in totals.values()),512)
        self.assertEqual(sum(v['completed_scored'] for v in totals.values()),508)
        self.assertEqual(sum(v['interrupted_unevaluable'] for v in totals.values()),4)
        self.assertEqual(sum(v['valid'] for v in totals.values()),508)
        self.assertTrue(all(v['by_condition']['nominal']['completed_scored']==63 for v in totals.values()))
        self.assertTrue(all(v['by_condition']['post_grip']['completed_scored']==64 for v in totals.values()))
        for variant,value in totals.items():
            self.assertEqual(value['success'],sum(r['success'] for r in rows if r['variant']==variant))

    def test_duplicate_and_missing_roster_rows_fail_aggregation(self):
        rows=outcome_rows()
        for bad in (rows[:-1],[*rows[:-1],deepcopy(rows[0])]):
            with self.assertRaisesRegex(ValueError,'512 unique'):
                recovery.aggregate(bad)

    def test_primary_all32_and_sensitivity31_groups_are_distinct(self):
        rows=outcome_rows()
        primary=recovery.contrasts(rows)
        sensitivity=recovery.contrasts(rows,drop_incomplete=True)
        self.assertTrue(all(c['independent_groups']==32 and len(c['group_differences'])==32 for c in primary))
        self.assertTrue(all(c['independent_groups']==31 and len(c['group_differences'])==31 for c in sensitivity))
        self.assertTrue(all(c['degrees_of_freedom']==30 for c in sensitivity))
        self.assertNotEqual(primary[0]['mean_difference'],sensitivity[0]['mean_difference'])
        self.assertAlmostEqual(recovery._critical(31),recovery.bench.old_cli.T975_DF31,places=9)

    def test_read_only_audit_requires_receipt_and_cannot_start_physics(self):
        with TemporaryDirectory() as temp:
            folder=Path(temp);request=('confirmation','wait200',{},'guarded',folder,{})
            with patch.object(recovery.bench,'execute',side_effect=AssertionError('No physics')):
                with self.assertRaisesRegex(ValueError,'never starts physics'):
                    recovery._read_only_job(request)

    def test_independent_summary_disagreement_is_fatal(self):
        with TemporaryDirectory() as temp:
            folder=Path(temp);(folder/'receipt.json').write_text('{}')
            request=('confirmation','wait200',{},'guarded',folder,{})
            result=dict(valid=True,raw_strict_success=True,success=True,aborted=False)
            with patch.object(recovery,'_AUDIT_CONTEXT',dict(root=folder)),patch.object(
                    recovery.independent,'_CONTEXT',{}),patch.object(recovery.independent,'_attempt',return_value=result),patch.object(
                    recovery.bench,'execute',return_value={**result,'success':False}):
                with self.assertRaisesRegex(ValueError,'summary differ'):
                    recovery._read_only_job(request)


if __name__=='__main__':unittest.main()

"""Pure saved-evidence counterexamples: no physics or learned inference."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location('development_failures',
    Path(__file__).with_name('development_failure_summary.py'))
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def trajectory(closure=60):
    steps=[]
    for i in range(1000):
        action=dict(joint_targets_rad=[0.]*6,gripper_position_m=.01 if closure is not None and i>=closure else .035)
        steps.append(dict(before=dict(timestamp_seconds=i*.02), after=dict(timestamp_seconds=(i+1)*.02),
            command=dict(issued_at_seconds=i*.02,issued=action),applied=dict(interval_start_seconds=i*.02,
                interval_end_seconds=(i+1)*.02,rate_limited=False,ramped_targets=deepcopy(action))))
    arrays=dict(arm_actuator_controls_rad=np.zeros((20000,6)),normal_rate_limited_jaw_target_m=np.repeat(
        [s['command']['issued']['gripper_position_m'] for s in steps],20),
        tcp_position_m=np.tile([.133,.0,.15],(20000,1)),object_position_m=np.zeros((20000,3)),
        gripper_joint_position_m=np.tile([.0346,-.0346],(20000,1)),
        unexpected_contact_count=np.zeros(20000),grasp_contact_counts=np.zeros((20000,2)))
    score=dict(unexpected_contact_native_ticks=0,grasp_seen=False,lift_seen=False,max_object_lift_m=.002)
    return steps,arrays,score


def measured(**changes):
    value=dict(control_intervals=1000,rate_limited_intervals=977,critical_contact_ticks=100,
        critical_contact_episode=True,bilateral_contact_ticks=0,strict_grasp_seen=False,lift_seen=False,
        maximum_object_lift_m=.002,no_closure_command=False,closure_command_beyond_20mm_xy=True,
        first_closure_command=dict(time_seconds=1.2,geometry_available=True,tcp_object_xy_distance_m=.133))
    value.update(changes);return value


def row(metrics=None,status='complete',success=False):
    return dict(id='case0',metrics=metrics,status=status,strict_success=success)


class MeasurementTests(unittest.TestCase):
    def test_closure_uses_preceding_native_tick_and_single_finger_command(self):
        steps,arrays,score=trajectory();arrays['tcp_position_m'][1199]=[.133,.0,.15]
        arrays['tcp_position_m'][1200]=[.001,.0,.0]
        result=module.measure(steps,arrays,score)
        event=result['first_closure_command']
        self.assertEqual(event['time_seconds'],1.2)
        self.assertAlmostEqual(event['tcp_object_xy_distance_m'],.133)
        self.assertEqual(event['issued_single_finger_target_m'],.01)
        self.assertEqual(event['actual_finger_positions_m'],[.0346,-.0346])
        self.assertTrue(result['closure_command_beyond_20mm_xy'])

    def test_time_zero_geometry_and_no_closure_are_distinct(self):
        result=module.measure(*trajectory(0))
        self.assertIsNone(result['closure_command_beyond_20mm_xy'])
        self.assertFalse(result['first_closure_command']['geometry_available'])
        self.assertFalse(result['no_closure_command'])
        result=module.measure(*trajectory(None))
        self.assertTrue(result['no_closure_command'])
        self.assertIsNone(result['first_closure_command'])
        self.assertFalse(result['closure_command_beyond_20mm_xy'])

    def test_clock_flags_and_native_binding_tamper_refused(self):
        for change in ('decision','end','flag','joint','jaw','critical'):
            with self.subTest(change=change):
                steps,arrays,score=trajectory()
                if change=='decision':steps[60]['before']['timestamp_seconds']+=.001
                if change=='end':steps[60]['applied']['interval_end_seconds']+=.001
                if change=='flag':steps[60]['applied']['rate_limited']=True
                if change=='joint':arrays['arm_actuator_controls_rad'][1219,0]=.5
                if change=='jaw':arrays['normal_rate_limited_jaw_target_m'][1219]=.02
                if change=='critical':arrays['unexpected_contact_count'][10]=1
                with self.assertRaises((ValueError,AssertionError)):module.measure(steps,arrays,score)


class DenominatorTests(unittest.TestCase):
    def test_failures_refusals_and_conditional_grasp_denominators(self):
        rows=[row(measured()),row(measured(strict_grasp_seen=True,bilateral_contact_ticks=40)),
            row(measured(strict_grasp_seen=True,lift_seen=True),success=True),row(status='refused'),row(status='unknown')]
        value=module.aggregate(rows)
        self.assertEqual(value['scheduled'],5)
        self.assertEqual(value['strict_successes'],1)
        self.assertEqual(value['no_strict_grasp'],dict(events=1,known=3,missing=2))
        self.assertEqual(value['grasp_without_lift'],dict(events=1,known=2,not_applicable=1,missing=2))
        self.assertEqual(value['control_intervals_observed'],3000)
        self.assertEqual(value['rate_limited_intervals_observed'],2931)

    def test_missing_pair_never_becomes_zero_difference(self):
        a=row(measured());b=row(status='refused');c=module.compare(a,b)
        self.assertFalse(c['both_complete_metrics'])
        self.assertIsNone(c['closure_time_difference_seconds'])
        a['teacher_comparison']=c
        value=module.aggregate([a])['matched_teacher']
        self.assertEqual(value['closure_earlier'],dict(events=0,known=0,missing=1))
        self.assertIsNone(value['rate_limited_interval_difference']['median'])

    def test_no_closure_pair_is_not_claimed_late_or_early(self):
        a=row(measured(first_closure_command=None,no_closure_command=True));b=row(measured())
        value=module.compare(a,b)
        self.assertTrue(value['both_complete_metrics'])
        self.assertIsNone(value['closes_before_teacher'])
        self.assertEqual(value['rate_limited_interval_difference'],0)


class RegistrationTests(unittest.TestCase):
    def test_summary_waits_for_both_seals_before_reading_evidence(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(module,'load') as loader:
            root=Path(tmp);(root/'stages').mkdir()
            (root/'stages/development-initial.integrity.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Both development'):module.summarize(root)
            loader.assert_not_called()

    def test_both_exact_cohorts_required_without_dropped_or_duplicate_cases(self):
        cases=[dict(id=f'case{i}',seed=i) for i in range(16)]
        protocol=dict(cases=dict(development=cases))
        stages={name:dict(rows=[dict(case=case,controller=c) for case in cases for c in ('student','blind')])
            for name in module.DEVELOPMENT}
        self.assertEqual(len(module.development_cases(protocol,stages)),16)
        for change in ('drop','duplicate','case','controller'):
            value=deepcopy(stages);rows=value['development-corrected']['rows']
            if change=='drop':rows.pop()
            if change=='duplicate':rows[1]=deepcopy(rows[0])
            if change=='case':rows[0]['case']['seed']=999
            if change=='controller':rows[0]['controller']='teacher'
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'coverage'):
                module.development_cases(protocol,value)

    def test_unknown_prefix_and_unattempted_keep_denominator_without_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);attempt=dict(id='a',case={},controller='blind',status='unknown',complete=False,strict_success=False)
            (root/'episodes/development-corrected/a').mkdir(parents=True)
            with patch.object(module,'load') as loader:
                value=module.episode(root,'development-corrected',attempt,'model','tracker',{})
                self.assertIsNone(value['metrics']);loader.assert_not_called()
            attempt=dict(attempt,id='b',status='unattempted')
            self.assertIsNone(module.episode(root,'development-corrected',attempt,'model','tracker',{})['metrics'])
            attempt['strict_success']=True
            with self.assertRaisesRegex(ValueError,'Unknown/unattempted'):module.episode(root,'development-corrected',attempt,'model','tracker',{})

    def test_stage_seal_registration_roster_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'stages').mkdir();name='development-initial';ph='a'*64
            attempts=[dict(id=f'a{i}',case=dict(id=f'c{i}'),controller='student',negative_control=None) for i in range(2)]
            protocol=dict(source_sha256={'test.py':'b'*64},rosters={name:attempts})
            receipt=dict(schema='dvidia.camera-action-study/v1/stage',stage=name,protocol_sha256=ph,status='complete',
                rows=deepcopy(attempts),scheduled=2,model_bindings={'student':'c'*64})
            started=dict(schema='dvidia.camera-action-study/v1/stage-start',stage=name,protocol_sha256=ph,
                source_sha256=protocol['source_sha256'],model_bindings=receipt['model_bindings'])
            def save():
                path=root/'stages'/f'{name}.json';path.write_text(json.dumps(receipt))
                (path.with_name(name+'.integrity.json')).write_text(json.dumps(dict(sha256=module.audit.identity(path)['sha256'],protocol_sha256=ph)))
                (path.with_name(name+'.started.json')).write_text(json.dumps(started))
            save();self.assertEqual(module.stage(root,name,protocol,ph,{})['scheduled'],2)
            receipt['rows'].reverse();save()
            with self.assertRaisesRegex(ValueError,'roster'):module.stage(root,name,protocol,ph,{})
            receipt['rows'].reverse();started['source_sha256']={};save()
            with self.assertRaisesRegex(ValueError,'registration'):module.stage(root,name,protocol,ph,{})

    def test_training_receipt_uses_distinct_schema_and_binds_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'stages').mkdir();name='train-corrected';ph='a'*64
            protocol=dict(source_sha256={'test.py':'b'*64})
            receipt=dict(schema='dvidia.camera-action-study/v1/training',stage=name,protocol_sha256=ph,status='complete',models={})
            started=dict(schema='dvidia.camera-action-study/v1/stage-start',stage=name,protocol_sha256=ph,
                source_sha256=protocol['source_sha256'],model_bindings=None)
            def save():
                path=root/'stages'/f'{name}.json';path.write_text(json.dumps(receipt))
                path.with_name(name+'.integrity.json').write_text(json.dumps(dict(sha256=module.audit.identity(path)['sha256'],protocol_sha256=ph)))
                path.with_name(name+'.started.json').write_text(json.dumps(started))
            save();self.assertEqual(module.stage(root,name,protocol,ph,{})['models'],{})
            receipt['schema']='dvidia.camera-action-study/v1/stage';save()
            with self.assertRaisesRegex(ValueError,'registration'):module.stage(root,name,protocol,ph,{})
            receipt['schema']='dvidia.camera-action-study/v1/training';started['schema']='wrong';save()
            with self.assertRaisesRegex(ValueError,'registration'):module.stage(root,name,protocol,ph,{})


class RefusalEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
        self.name='development-corrected';self.model='b'*64;self.tracker='c'*64
        self.attempt=dict(id='corrected-case0-blind',case=dict(id='case0',group_id='group0',goal_position_m=[.35,.08,.08],seed=7),
            controller='blind',negative_control=None,status='refused',complete=False,strict_success=False,recorded_complete_intervals=0,
            policy_artifact_sha256=self.model,score=dict(valid=False,success=False,reason='native_evidence_unavailable'))
        self.relative='episodes/'+self.name+'/'+self.attempt['id'];self.directory=self.root/self.relative;self.directory.mkdir(parents=True)
        self.attempt['path']=self.relative;case={**self.attempt['case'],'policy_artifact_sha256':self.model}
        self.receipt=dict(schema='dvidia.camera-action-rollout/v2',case=case,mode='student',negative_control=None,
            status='refused',strict_success=False,score=self.attempt['score'],tracker_sha256=self.tracker,teacher_queries=0,
            student_calls=0,recorded_complete_intervals=0,no_privileged_features_or_teacher_intervention=True)
        self.started=dict(schema=self.receipt['schema'],case=case,mode='student',negative_control=None,
            config=dict(duration_s=20.,seed=7,width=320,height=240,camera_every_controls=2))
        self.save()

    def save(self):
        (self.directory/'started.json').write_text(json.dumps(self.started))
        (self.directory/'rollout.json').write_text(json.dumps(self.receipt))
        files={name:module.audit.identity(self.directory/name) for name in ('started.json','rollout.json')}
        path=self.directory/'integrity.json';path.write_text(json.dumps(dict(schema='dvidia.camera-action-rollout/v2/integrity',files=files)))
        self.attempt.update(integrity_sha256=module.audit.identity(path)['sha256'],receipt_sha256=files['rollout.json']['sha256'])

    def check(self):return module.episode(self.root,self.name,self.attempt,self.model,self.tracker,{})

    def test_blind_is_student_control_and_refusal_is_missing_not_zero(self):
        result=self.check();self.assertEqual(result['controller'],'blind')
        self.assertIsNone(result['metrics']);self.assertFalse(result['native_score_valid']);self.assertEqual(result['teacher_queries'],0)

    def test_teacher_intervention_and_tracker_drift_refused(self):
        self.receipt['teacher_queries']=1;self.save()
        with self.assertRaisesRegex(ValueError,'teacher query'):self.check()
        self.receipt['teacher_queries']=0;self.receipt['tracker_sha256']='d'*64;self.save()
        with self.assertRaisesRegex(ValueError,'tracker'):self.check()

    def test_changed_bound_bytes_and_model_identity_refused(self):
        self.receipt['student_calls']=1
        (self.directory/'rollout.json').write_text(json.dumps(self.receipt))
        with self.assertRaisesRegex(ValueError,'Sealed evidence changed'):self.check()
        self.save();self.attempt['policy_artifact_sha256']='d'*64
        with self.assertRaisesRegex(ValueError,'model binding'):self.check()


if __name__=='__main__':unittest.main()

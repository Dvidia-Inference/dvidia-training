"""Independent scoring boundary tests; fresh cases are inspected, never simulated."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from dvidia_training import grip_protocol as grip, tool_reach


def fixture(ticks=600):
    rows=[[round(i*.002,6),0.,1.,.6,.4,.40,-.10,.10,0.] for i in range(1,ticks+1)]
    return {'valid':True,'success':True,'case':{'id':'synthetic-audit','shoe_x':.72,'tool_length':.35},
            'metrics':{'simulation_seconds':ticks*.002,'native_steps':ticks,
                       'retained_lift_success':True,'retrieval_success':True},
            'events':[],'native_trace_columns':list(grip._REQUIRED),'native_trace':rows}


class GripProtocolTests(unittest.TestCase):
    def test_seal_fresh_pairs_and_prior_cases_are_regression(self):
        self.assertEqual(sha256(grip.FIXTURE.read_bytes()).hexdigest(),grip.FIXTURE_SHA256)
        fresh=grip.protocol('heldout');regression=grip.protocol('regression')
        self.assertEqual(len(fresh['attempts']),20)
        self.assertEqual(len(regression['attempts']),18)
        prior={row['case']['id'] for row in regression['attempts']}
        self.assertFalse(prior & {row['case']['id'] for row in fresh['attempts']})
        for row in fresh['attempts']:tool_reach._validate_case(row['case'])
        for case in {row['case']['id'] for row in fresh['attempts']}:
            self.assertEqual({r['policy'] for r in fresh['attempts'] if r['case']['id']==case},{'baseline','recovery'})
        self.assertEqual(sum(r['case']['shoe_x']+.165-r['case']['tool_length']>.54 for r in fresh['attempts']),4)
        fresh['criteria']['final_retention_s']=9
        self.assertEqual(grip.protocol()['criteria']['final_retention_s'],.5)
        with self.assertRaises(ValueError):grip.protocol('test')

    def test_seal_rejects_changed_fixture(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'changed.json';path.write_bytes(grip.FIXTURE.read_bytes()+b' ')
            with patch.object(grip,'FIXTURE',path),self.assertRaises(ValueError):grip.protocol()

    def test_valid_continuous_grip_and_exact_final_window(self):
        score=grip.audit_result(fixture())
        self.assertTrue(score['task_success'])
        self.assertEqual(score['strict_passing_ticks'],250)
        self.assertEqual(score['final_window_ticks'],250)
        self.assertAlmostEqual(score['final_window_start_exclusive_s'],.7)
        self.assertFalse(score['legacy_only_claim'])

    def test_any_single_final_tick_support_loss_or_height_gap_fails(self):
        # Exposes failures that averages and sparse 20ms frames can hide.
        for column,value in [('world_shoe_force_n',.01),('tool_shoe_force_n',.01),
                             ('left_finger_force_n',.01),('right_finger_force_n',.01),('shoe_z_m',.035)]:
            result=fixture();row=result['native_trace'][-127]
            row[grip._REQUIRED.index(column)]=value
            if column=='tool_shoe_force_n':row[2]+=.01
            if column=='left_finger_force_n':row[2]=row[3]+row[4]
            if column=='right_finger_force_n':row[2]=row[3]+row[4]
            with self.subTest(column=column):
                self.assertFalse(grip.score_result(result)['task_success'])
                with self.assertRaises(ValueError):grip.audit_result(result)

    def test_legacy_pass_with_other_robot_support_stays_valid_but_fails_uniform_strict_score(self):
        result=fixture();result['native_trace'][-127][2]=1.1
        score=grip.audit_result(result)
        self.assertTrue(score['legacy_retention_pass'])
        self.assertTrue(score['legacy_only_claim'])
        self.assertTrue(score['valid'])
        self.assertFalse(score['task_success'])
        self.assertEqual(score['other_robot_supported_ticks'],1)
        result['metrics']['retention_claim_scope']='strict'
        with self.assertRaises(ValueError):grip.audit_result(result)

    def test_dense_time_coverage_cannot_be_truncated_or_duplicated(self):
        for mutation in ('drop','duplicate','truncate','nonfinite','force_mismatch'):
            result=fixture()
            if mutation=='drop':result['native_trace'].pop(150)
            elif mutation=='duplicate':result['native_trace'][150][0]=result['native_trace'][149][0]
            elif mutation=='truncate':result['native_trace'].pop()
            elif mutation=='nonfinite':result['native_trace'][150][2]=float('nan')
            else:result['native_trace'][150][2]=0.
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):grip.score_result(result)

    def test_loss_detection_command_and_requalification_are_separate(self):
        result=fixture()
        for row in result['native_trace'][150:180]:row[3]=0.;row[2]=row[4]
        result['events']=[{'type':'grip_loss_detected','t':.32},{'type':'grip_recovery_started','t':.34},
                          {'type':'recovery_control_issued','t':.342,'requested_at_s':.34,
                           'issued_ctrl':[0.]*7,'requested_joint_target':[0.]*7}]
        score=grip.audit_result(result)
        self.assertTrue(score['task_success'])
        self.assertEqual(score['sustained_loss_count'],1)
        self.assertEqual(score['requalifications_after_command'],1)
        loss=score['loss_episodes'][0]
        self.assertAlmostEqual(loss['onset_s'],.302)
        self.assertAlmostEqual(loss['detection_latency_s'],.018)
        self.assertAlmostEqual(loss['request_latency_s'],.038)
        self.assertAlmostEqual(loss['command_issue_latency_s'],.040)
        self.assertAlmostEqual(loss['requalified_s'],.46)
        result['events'].pop();score=grip.score_result(result)
        self.assertEqual(score['recovery_request_count'],1)
        self.assertEqual(score['recovery_command_count'],0)
        self.assertEqual(score['requalifications_after_command'],0)
        result['events']=[];score=grip.score_result(result)
        self.assertEqual(score['sustained_losses_without_detection'],1)
        self.assertEqual(score['requalifications_after_command'],0)

    def test_recovery_packet_requires_native_clock_matching_request_and_finite_vector(self):
        result=fixture();request={'type':'grip_recovery_started','t':.34}
        packet={'type':'recovery_control_issued','t':.342,'requested_at_s':.34,
                'issued_ctrl':[0.]*7,'requested_joint_target':[0.]*7}
        for mutation in ('missing_request','duplicate','off_clock','before_request','bad_vector'):
            result['events']=[deepcopy(request),deepcopy(packet)]
            if mutation=='missing_request':result['events'].pop(0)
            elif mutation=='duplicate':result['events'].append(deepcopy(packet))
            elif mutation=='off_clock':result['events'][1]['t']=.343
            elif mutation=='before_request':result['events'][1]['t']=.338
            else:result['events'][1]['issued_ctrl'][0]=float('nan')
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):grip.score_result(result)

    def test_appropriate_deferral_is_not_task_success(self):
        result=fixture();result['case']['tool_length']=.18
        for row in result['native_trace']:row[2:5]=[0.,0.,0.];row[7]=0.;row[8]=2.;row[5]=.72
        result['events']=[{'type':'help_requested','t':.4,'reason':'tool_cannot_reach_far_edge'}]
        result['success']=False;result['metrics'].update(retained_lift_success=False,retrieval_success=False)
        score=grip.audit_result(result)
        self.assertTrue(score['appropriate_deferral'])
        self.assertFalse(score['task_success'])
        self.assertFalse(score['deferral_counts_as_task_success'])


if __name__=='__main__':unittest.main()

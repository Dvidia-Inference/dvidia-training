"""Posthoc receipt/terminal-component tests using fabricated numeric fixtures."""
from copy import deepcopy
from contextlib import redirect_stdout, redirect_stderr
from hashlib import sha256
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

PATH=Path(__file__).resolve().parents[1]/'benchmarks/camera-action-v01/summarize_results.py'
SPEC=importlib.util.spec_from_file_location('camera_action_posthoc',PATH)
summary=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(summary)


def raw_json(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);raw=raw_json(value);path.write_bytes(raw);return raw

def identity(raw):return dict(bytes=len(raw),sha256=sha256(raw).hexdigest())


def native_fixture(*,ticks=20000):
    position=np.tile([.35,.08,.08],(ticks,1))
    arrays=dict(t_seconds=np.arange(1,ticks+1)*.001,native_tick=np.arange(1,ticks+1),
        grasp_contact_counts=np.zeros((ticks,2)),gripper_joint_position_m=np.tile([.03,-.03],(ticks,1)),
        object_position_m=position,tcp_position_m=position+np.array([0.,0.,.05]),
        object_linear_velocity_m_s=np.zeros((ticks,3)),object_angular_velocity_rad_s=np.zeros((ticks,3)),
        table_object_contact_count=np.ones(ticks))
    metadata=dict(schema='dvidia.piper-feedback-native/v1',native_timestep_s=.001,rows=ticks,side_order=['left','right'])
    episode=dict(config=dict(timestep=.001,timeout_s=ticks*.001,goal_dwell_s=.25,
        retreat_clearance_m=.10,goal_position_tolerance_m=.025,target_position_m=[.35,.08,.08]))
    return arrays,metadata,episode


def study_fixture(root):
    case=dict(id='held-out-0',group_id='held-out-0',goal_id=0)
    roster=[dict(id='attempt-0',controller='student',case=case),dict(id='attempt-1',controller='student',case={**case,'id':'held-out-1'})]
    protocol=dict(rosters={'confirmation':roster})
    protocol_raw=write(root/'protocol.json',protocol);protocol_hash=summary.digest(protocol_raw)
    write(root/'registration.json',dict(protocol_sha256=protocol_hash))
    score=dict(valid=True,success=False,native_recomputed=dict(grasp_seen=True,lift_seen=True,
        carry_complete=True,retention_lost=False,unexpected_contact_native_ticks=5),terminal_only=dict(success=False,dwell_s=0.))
    perception=dict(available=600,missing=400,decisions=1000,longest_missing_seconds=7.2)
    receipt=dict(status='complete',recorded_complete_intervals=1000,score=score,perception=perception,
        timing=dict(wall_seconds=30.,boundaries=[dict(policy_inference_wall_seconds=.001,
            full_boundary_wall_seconds=.01),dict(policy_inference_wall_seconds=.003,full_boundary_wall_seconds=.03)]))
    episode_root=root/'episodes'/'attempt-0';episode_root.mkdir(parents=True)
    receipt_raw=write(episode_root/'rollout.json',receipt)
    context_raw=write(episode_root/'context.json',dict(metadata=dict(runtime=dict(system='fixture',machine='no robot'))))
    arrays,metadata,episode=native_fixture()
    metadata_raw=write(episode_root/'native_metadata.json',metadata)
    episode_raw=write(episode_root/'native_episode.json',episode)
    stream=io.BytesIO();np.savez_compressed(stream,**arrays);native_raw=stream.getvalue();(episode_root/'native.npz').write_bytes(native_raw)
    inventory=dict(files={name:identity(raw) for name,raw in (
        ('rollout.json',receipt_raw),('context.json',context_raw),('native.npz',native_raw),
        ('native_metadata.json',metadata_raw),('native_episode.json',episode_raw))})
    inventory_raw=write(episode_root/'integrity.json',inventory)
    row={**roster[0], 'path':'episodes/attempt-0','receipt_sha256':summary.digest(receipt_raw),
        'integrity_sha256':summary.digest(inventory_raw),'status':'complete','complete':True,'strict_success':False,
        'score':score,'perception':perception}
    refused={**roster[1],'status':'refused','complete':False,'strict_success':False,'score':{'valid':False,'success':False},
             'receipt_sha256':None,'path':None}
    stage=dict(stage='confirmation',status='complete',protocol_sha256=protocol_hash,scheduled=2,
               strict_successes=0,rows=[row,refused])
    stage_raw=write(root/'stages/confirmation.json',stage)
    write(root/'stages/confirmation.integrity.json',dict(sha256=summary.digest(stage_raw),protocol_sha256=protocol_hash))
    return stage


class NativeTerminalTests(unittest.TestCase):
    def test_released_and_settled_but_not_retreated_are_independent(self):
        arrays,metadata,episode=native_fixture(ticks=1000)
        result=summary.native_terminal_components(arrays,metadata,episode)['predicates']
        self.assertTrue(result['release']['final_predicate']);self.assertTrue(result['release']['dwell_met'])
        self.assertTrue(result['settling']['dwell_met']);self.assertFalse(result['retreat']['final_predicate'])
        self.assertFalse(result['joint_terminal']['dwell_met'])
        self.assertEqual(result['retreat']['final_consecutive_dwell_seconds'],0.)

    def test_dwell_counts_only_final_consecutive_ticks_at_exact_threshold(self):
        arrays,metadata,episode=native_fixture(ticks=1000)
        arrays['tcp_position_m'][-250:,2]=.18
        result=summary.native_terminal_components(arrays,metadata,episode)['predicates']
        self.assertEqual(result['retreat']['final_consecutive_dwell_seconds'],.25)
        self.assertTrue(result['retreat']['dwell_met']);self.assertTrue(result['joint_terminal']['dwell_met'])
        arrays['tcp_position_m'][-250,2]=.179
        result=summary.native_terminal_components(arrays,metadata,episode)['predicates']
        self.assertAlmostEqual(result['retreat']['final_consecutive_dwell_seconds'],.249)
        self.assertTrue(result['retreat']['final_predicate']);self.assertFalse(result['retreat']['dwell_met'])

    def test_actual_jaw_contacts_speed_and_support_are_distinct(self):
        arrays,metadata,episode=native_fixture(ticks=1000)
        arrays['tcp_position_m'][:,2]=.2
        arrays['object_linear_velocity_m_s'][-1,0]=.035
        result=summary.native_terminal_components(arrays,metadata,episode)['predicates']
        self.assertTrue(result['release']['dwell_met']);self.assertFalse(result['settling']['final_predicate'])
        self.assertTrue(result['retreat']['dwell_met'])
        arrays['object_linear_velocity_m_s'][-1,0]=0.
        arrays['gripper_joint_position_m'][-1,0]=.028
        result=summary.native_terminal_components(arrays,metadata,episode)['predicates']
        self.assertFalse(result['release']['final_predicate']);self.assertTrue(result['settling']['dwell_met'])
        arrays['gripper_joint_position_m'][-1,0]=.03;arrays['grasp_contact_counts'][-1,0]=1
        self.assertFalse(summary.native_terminal_components(arrays,metadata,episode)['predicates']['release']['final_predicate'])
        arrays['grasp_contact_counts'][-1,0]=0;arrays['table_object_contact_count'][-1]=0
        result=summary.native_terminal_components(arrays,metadata,episode)['predicates']
        self.assertTrue(result['release']['dwell_met']);self.assertFalse(result['joint_terminal']['final_predicate'])

    def test_missing_nonfinite_and_incomplete_native_evidence_is_refused(self):
        arrays,metadata,episode=native_fixture(ticks=1000)
        bad=deepcopy(arrays);del bad['object_linear_velocity_m_s']
        with self.assertRaisesRegex(ValueError,'Missing or malformed'):
            summary.native_terminal_components(bad,metadata,episode)
        bad=deepcopy(arrays);bad['object_position_m'][0,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Missing or malformed'):
            summary.native_terminal_components(bad,metadata,episode)
        bad=deepcopy(arrays);bad['native_tick'][-1]=500
        with self.assertRaisesRegex(ValueError,'time/tick'):
            summary.native_terminal_components(bad,metadata,episode)


class PosthocSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'study';self.root.mkdir();self.stage=study_fixture(self.root)

    def test_real_seal_readback_preserves_all_attempt_and_conditional_denominators(self):
        result=summary.summarize(self.root);row=result['controllers'][0]
        self.assertEqual(row['scheduled'],2);self.assertEqual(row['complete'],1)
        self.assertEqual(row['native_assessable'],1);self.assertEqual(row['strict_successes'],0)
        self.assertEqual(row['components']['release']['conditional_assessed'],1)
        self.assertEqual(row['components']['release']['conditional_failed'],0)
        self.assertEqual(row['components']['retreat']['conditional_failed'],1)
        self.assertEqual(row['components']['release']['unassessed'],1)
        self.assertEqual(row['critical_contacts']['assessed_attempts'],1)
        self.assertEqual(row['critical_contacts']['unassessed_attempts'],1)
        self.assertEqual(row['critical_contacts']['attempts_with_contacts'],1)
        self.assertEqual(row['critical_contacts']['native_ticks'],5)
        self.assertEqual(row['perception']['available_fraction'],.6)
        self.assertEqual(row['perception']['episodes_unknown'],1)
        self.assertEqual(row['terminal_components_by_attempt'][0]['source_sha256']['native.npz'],
            summary.digest((self.root/'episodes/attempt-0/native.npz').read_bytes()))
        self.assertIsNone(result['hardware'][0]['cpu_model'])
        self.assertTrue(any(x['status']=='pending_receipt' for x in result['stages']))
        text=summary.markdown(result)
        self.assertIn('0 / 2',text);self.assertIn('posthoc',text);self.assertIn('not physical sensor-to-actuator',text)

    def test_missing_metrics_stay_unassessed_not_guessed_from_overall_failure(self):
        row=dict(score={'valid':True,'success':False,'native_recomputed':{}},strict_success=False)
        result=summary.component_counts([row],2,summary.COMPONENTS[0])
        self.assertEqual(result['assessed'],0);self.assertEqual(result['failed'],0);self.assertEqual(result['unassessed'],2)
        result=summary.component_counts([row],2,summary.COMPONENTS[5])
        self.assertEqual(result['conditional_assessed'],0);self.assertEqual(result['prerequisite_unknown'],2)

    def test_latency_pools_boundary_samples_instead_of_episode_percentiles(self):
        roster=[dict(id='a'),dict(id='b')]
        rows=[dict(id=key,controller='student',score={}) for key in ('a','b')]
        episodes={'a':({'timing':{'boundaries':[{'full_boundary_wall_seconds':1.}]}},None),
                  'b':({'timing':{'boundaries':[{'full_boundary_wall_seconds':.01}]*19}},None)}
        record=summary._controller('test','student',roster,rows,episodes)['timing']['full_boundary_wall_seconds']
        self.assertEqual(record['samples'],20);self.assertAlmostEqual(record['p95'],.0595)
        self.assertFalse(record['deadline_claim'])

    def test_modified_native_bytes_or_stage_receipts_are_rejected(self):
        path=self.root/'episodes/attempt-0/native.npz';path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'Native numeric evidence differs'):
            summary.summarize(self.root)
        stage_path=self.root/'stages/confirmation.json';stage_path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'Stage seal'):
            summary.summarize(self.root)

    def test_resealed_missing_or_replaced_attempt_id_is_rejected(self):
        for replacement in (None, 'unregistered-attempt'):
            with self.subTest(replacement=replacement):
                stage=deepcopy(self.stage)
                if replacement is None:
                    stage['rows'].pop()
                else:
                    stage['rows'][-1]['id']=replacement
                raw=write(self.root/'stages/confirmation.json',stage)
                write(self.root/'stages/confirmation.integrity.json',dict(sha256=summary.digest(raw),
                    protocol_sha256=stage['protocol_sha256']))
                with self.assertRaisesRegex(ValueError,'ID set differs'):
                    summary.summarize(self.root)

    def test_resealed_negative_control_change_is_rejected(self):
        stage=deepcopy(self.stage);stage['rows'][-1]['negative_control']='open_gripper'
        raw=write(self.root/'stages/confirmation.json',stage)
        write(self.root/'stages/confirmation.integrity.json',dict(sha256=summary.digest(raw),
            protocol_sha256=stage['protocol_sha256']))
        with self.assertRaisesRegex(ValueError,'registered roster'):
            summary.summarize(self.root)

    def test_unattempted_rows_remain_explicit_and_count_in_denominator(self):
        stage=deepcopy(self.stage);stage['status']='blocked'
        stage['rows'][-1]=dict(id=stage['rows'][-1]['id'],controller='student',
            case=stage['rows'][-1]['case'],status='unattempted',complete=False,strict_success=False)
        raw=write(self.root/'stages/confirmation.json',stage)
        write(self.root/'stages/confirmation.integrity.json',dict(sha256=summary.digest(raw),
            protocol_sha256=stage['protocol_sha256']))
        result=summary.summarize(self.root)['controllers'][0]
        self.assertEqual(result['scheduled'],2);self.assertEqual(result['retained_rows'],2)
        self.assertEqual(result['statuses']['unattempted'],1)
        self.assertEqual(result['components']['retreat']['unassessed'],1)

    def test_cli_writes_fresh_posthoc_files_outside_frozen_study(self):
        output=Path(self.temp.name)/'report'
        with redirect_stdout(io.StringIO()):
            summary.main(['--study',str(self.root),'--output',str(output)])
        self.assertTrue((output/'summary.json').is_file());self.assertTrue((output/'summary.md').is_file())
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            summary.main(['--study',str(self.root),'--output',str(output)])
        self.assertFalse((self.root/'summary.json').exists())


if __name__=='__main__':unittest.main()

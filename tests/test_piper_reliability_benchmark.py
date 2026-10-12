"""v0.5 failure denominators, probe evidence and native causal replay."""
from copy import deepcopy
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dvidia_training import piper_reliability_benchmark as bench
from tests import test_piper_recovery_benchmark as recovery_fixture
from tests import test_piper_task_benchmark as original


def plan():
    p=json.loads(json.dumps(bench.make_protocol(seed=35)))
    for scene in p['scenes']:
        scene['config']['timeout_s']=.1
    return p


def fixture_episode(protocol,scene,control,mode,tape=None,invalid=False):
    mapped='baseline' if mode=='baseline' else 'recovery'
    raw=recovery_fixture.fixture_episode(protocol,scene,control,mapped,tape,invalid)
    if mode in ('reacquire','reacquire_grasp'):
        raw['grip_guard']=bench._inactive(mode)
        raw['grip_guard'].update(active=control=='placement',diagnostic_count=len(raw['actions']) if control=='placement' else 0)
        raw['policy_implementation_sha256']={n:protocol['implementation_sha256'][n] for n in bench._policy_sources(mode)}
        for state in raw['trajectory']:
            if state.get('guard_diagnostic') is not None:
                state['guard_diagnostic'].update(second_probe_count=0,second_probe_started_s=None,
                    low_settled_dwell_s=0.,second_probe_qualification_dwell_s=0.,second_probe_requested_rise_m=0.,
                    second_probe_relative_drift_m=None,second_probe_origin_tcp_m=None,second_probe_orientation_wxyz=None,
                    second_probe_anchor_local_m=None,second_probe_measured_tcp_rise_m=None,second_probe_goal_m=None)
    raw['metrics']['lift_seen']=False
    for state in raw['trajectory']:
        state['native_lift_seen']=False
    return raw


def run_fixture(root,invalid_reference=False):
    protocol,calls=plan(),[]
    def run(_assets,config,**options):
        assert json.loads((root/'protocol.json').read_bytes())==protocol
        calls.append(options)
        scene=next(s for s in protocol['scenes'] if s['seed']==options['seed'])
        return fixture_episode(protocol,scene,options['control'],options['policy_mode'],options.get('arm_tape'),
            invalid=invalid_reference and options['control']=='placement' and options['policy_mode']=='recovery')
    with patch.object(bench,'make_protocol',return_value=protocol), \
         patch.object(bench.robot_profiles,'verify_assets',return_value=deepcopy(bench.base._pinned_assets()[0])), \
         patch.object(bench.robot_profiles,'_payload',return_value=({},original.BLOBS)), \
         patch.object(bench.task,'run_episode',side_effect=run), \
         patch.object(bench,'audit_policy'):
        report=bench.run_benchmark('unused',root)
    return report,calls


def reseal(root):
    records={}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.name=='integrity.json': continue
        data,name=path.read_bytes(),path.relative_to(root).as_posix()
        record={'bytes':len(data),'sha256':sha256(data).hexdigest()}
        if name.endswith('.gz'):
            decoded=bench.decode_attempt(data)
            record.update(encoding='gzip',decoded_bytes=len(decoded),decoded_sha256=sha256(decoded).hexdigest())
        records[name]=record
    (root/'integrity.json').write_bytes(bench._json_bytes({'schema':bench.INTEGRITY_SCHEMA,'files':records}))


def mutate(root,report,index,callback):
    row=report['attempts'][index]
    raw=json.loads(bench.decode_attempt((root/row['result_file']).read_bytes()))
    callback(raw)
    raw['recovery_evaluation']=bench.evaluate_episode(raw)
    (root/row['result_file']).write_bytes(gzip.compress(bench._json_bytes(raw),mtime=0))
    report['attempts'][index]={**bench._summary(raw),'result_file':row['result_file']}
    report['summary']=bench.aggregate(report['attempts'],report['scenes'])
    report['paired_comparisons']=bench.paired_comparisons(report['attempts'],report['scenes'])
    report['valid_attempts']=sum(r['valid'] for r in report['attempts'])
    report['quality_gates']=bench.quality_gates(report['attempts'],report['scenes'])
    report['execution_metrics']=bench.execution_metrics(report['attempts'])
    (root/'report.json').write_bytes(bench._json_bytes(report));reseal(root)


class ReliabilityBenchmarkTests(unittest.TestCase):
    def setUp(self):
        expected,_=bench.base._pinned_assets()
        notices={name:sha256(value).hexdigest() for name,value in original.BLOBS.items()}
        p=patch.object(bench.base,'_pinned_assets',return_value=(expected,notices))
        p.start();self.addCleanup(p.stop)

    def inspect(self,root,report):
        with patch.object(bench,'make_protocol',return_value=report['protocol']):
            return bench.load_reliability_bundle(root)

    def test_fixed_three_mode_plan_keeps_own_controls_and_original_task(self):
        p=bench.make_protocol()
        self.assertEqual(p,bench.make_protocol())
        self.assertEqual(bench.MODES,('recovery','reacquire','reacquire_grasp'))
        self.assertEqual(p['planned_attempts'],84)
        self.assertEqual(len(p['scenes']),12)
        self.assertEqual(p['variants'],list(bench.VARIANTS))
        self.assertEqual(p['contract'],bench.task.task_contract())
        self.assertEqual(bench._variant('reacquire_open_jaw'),('replay_open_jaw','reacquire'))

    def test_complete_bundle_and_raw_policy_inputs_are_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'run';report,calls=run_fixture(root)
            checked,files=self.inspect(root,report)
            self.assertEqual(report,checked)
            self.assertEqual((report['attempted'],report['valid_attempts'],len(files)),(84,84,90))
            self.assertTrue(all(c['record_step_timings'] and c['record_policy_inputs'] for c in calls))
            for placed,opened in ((0,1),(2,3),(4,5)):
                raw=json.loads(bench.decode_attempt(files[report['attempts'][placed]['result_file']]))
                self.assertEqual(calls[opened]['arm_tape'],raw['arm_tape'])

    def test_invalid_reference_keeps_failed_control_and_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'run';report,_=run_fixture(root,True)
            self.assertEqual((report['attempted'],report['valid_attempts']),(84,60))
            self.assertFalse(report['attempts'][1]['valid'])
            self.assertTrue(report['attempts'][2]['valid'])
            self.inspect(root,report)

    def test_resealed_source_model_tape_lift_input_and_probe_invention_reject(self):
        def tape(raw):
            raw['arm_tape'][0][0]+=.001;raw['actions'][0]['joint_targets_rad'][0]+=.001
            raw['trajectory'][0]['command']['joint_targets_rad'][0]+=.001
            raw['trajectory'][0]['issued_joint_targets_rad'][0]+=.001
        changes=((2,lambda r:r.update(compiled_model_sha256='b'*64)),(3,tape),
                 (2,lambda r:r['trajectory'][0].pop('native_lift_seen')),
                 (2,lambda r:r['trajectory'][1].update(native_lift_seen=True)),
                 (2,lambda r:r['grip_guard'].update(second_probe_count=1)),
                 (2,lambda r:r['policy_implementation_sha256'].update(piper_reacquisition_py='0'*64)))
        for index,callback in changes:
            with self.subTest(index=index),tempfile.TemporaryDirectory() as folder:
                root=Path(folder)/'run';report,_=run_fixture(root);mutate(root,report,index,callback)
                with self.assertRaises(ValueError): self.inspect(root,report)

    def test_changed_seeded_scene_and_missing_attempt_reject(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'run';report,_=run_fixture(root)
            with self.assertRaisesRegex(ValueError,'seeded fixed'): bench.load_reliability_bundle(root)
            report['attempts'].pop()
            (root/'report.json').write_bytes(bench._json_bytes(report));reseal(root)
            with self.assertRaises(ValueError): self.inspect(root,report)

    def test_comparison_keeps_gains_and_losses_outside_speed_subset(self):
        scenes=[{'id':'a'},{'id':'b'},{'id':'c'}];rows=[]
        for scene in scenes:
            for mode in bench.MODES:
                success=scene['id']=='a' or scene['id']=='b' and mode=='recovery' or scene['id']=='c' and mode=='reacquire'
                rows.append({'scene_id':scene['id'],'variant':mode+'_placement','valid':True,'success':success,
                    'recovery_evaluation':{'completion_time_s':10. if mode=='recovery' else 9.,'abort_time_s':None}})
        pair=bench.paired_comparisons(rows,scenes)[0]
        self.assertEqual((pair['common_successes'],pair['gained_successes'],pair['lost_successes']),(1,1,1))
        self.assertEqual(pair['common_success_mean_delta_s'],-1.)
        self.assertIsNone(pair['pairs'][1]['completion_delta_s'])


class NativeReliabilityAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets=os.environ.get('DVIDIA_ROBOT_PROFILE_ASSETS')
        if not cls.assets:
            raise unittest.SkipTest('Pinned native assets not configured')
        cls.protocol=bench.make_protocol(seed=941827)
        cls.raws=[]
        for name,mode in (('friction080-mass060-layout00','reacquire'),('friction015-mass040-layout00','reacquire'),
                          ('friction055-mass060-layout00','reacquire_grasp')):
            s=next(s for s in cls.protocol['scenes'] if s['id']==name)
            raw=bench.task.run_episode(cls.assets,bench.task.PiperTaskConfig(**s['config']),seed=s['seed'],
                policy_mode=mode,record_step_timings=True,record_policy_inputs=True)
            raw.update(variant=mode+'_placement',attempt_wall_seconds=raw['timing']['episode_wall_seconds'])
            raw['recovery_evaluation']=bench.evaluate_episode(raw)
            cls.raws.append(raw)
        if not cls.raws[0]['success'] or cls.raws[0]['grip_guard']['second_probe_resume_count']!=1:
            raise AssertionError('Native fixture did not exercise successful second lift')
        if not cls.raws[1]['valid'] or cls.raws[1]['success']:
            raise AssertionError('Native fixture did not preserve failed acquisition')
        if not cls.raws[2]['valid'] or not cls.raws[2]['success']:
            raise AssertionError('Native fixture did not complete alternate-grasp gain')

    def test_successful_second_lift_and_failed_acquisition_pass_both_audits(self):
        for raw in self.raws:
            bench._policy_inputs(raw)
            bench._reliability(raw,self.protocol)
            bench.audit_policy(raw,self.assets,self.protocol)

    def test_forged_probe_target_or_qualification_rejects_portably(self):
        for change in ('target','event','dwell'):
            raw=deepcopy(self.raws[0])
            if change=='target':
                state=next(s for s in raw['trajectory'] if s.get('guard_diagnostic',{}).get('second_probe_goal_m'))
                state['guard_diagnostic']['second_probe_goal_m'][0]+=.001
            elif change=='event':
                raw['grip_guard']['events']=[e for e in raw['grip_guard']['events'] if e['type']!='second_probe_started']
            else:
                state=next(s for s in raw['trajectory'] if s.get('guard_diagnostic',{}).get('decision')=='second_probe_qualified_resume')
                state['guard_diagnostic']['second_probe_qualification_dwell_s']+=.02
            raw['recovery_evaluation']=bench.evaluate_episode(raw)
            with self.subTest(change=change),self.assertRaises(ValueError): bench._reliability(raw,self.protocol)

    def test_joint_target_forgery_rejected_by_native_policy_audit(self):
        raw=deepcopy(self.raws[0])
        state=next(s for s in raw['trajectory'] if s.get('command',{}).get('phase')=='memory_second_probe')
        index=round(state['t_seconds']/raw['config']['control_dt'])
        raw['actions'][index]['joint_targets_rad'][0]+=.001
        with self.assertRaisesRegex(ValueError,'command differs'): bench.audit_policy(raw,self.assets,self.protocol)

    def test_forged_tcp_pose_rejected_by_native_forward_kinematics(self):
        raw=deepcopy(self.raws[0]);raw['trajectory'][0]['tcp_position_m'][0]+=.001
        with self.assertRaisesRegex(ValueError,'TCP pose differs'): bench.audit_policy(raw,self.assets,self.protocol)

    def test_audit_requires_exact_frozen_sources(self):
        protocol=deepcopy(self.protocol);protocol['implementation_sha256']['piper_reacquisition.py']='a'*64
        with self.assertRaisesRegex(ValueError,'exact frozen source'): bench.audit_policy(self.raws[0],self.assets,protocol)


if __name__=='__main__': unittest.main()

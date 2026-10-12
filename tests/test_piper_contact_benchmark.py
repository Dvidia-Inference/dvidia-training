"""Contact-history paired denominators, causal evidence and bounded receipts."""
from copy import deepcopy
import gzip
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dvidia_training import piper_contact_benchmark as bench
from tests import test_piper_task_benchmark as original


def plan():
    protocol=json.loads(json.dumps(bench.make_protocol(seed=35)))
    for scene in protocol['scenes']:
        scene['config']['timeout_s']=.1
    return protocol


def fixture_episode(protocol,scene,control,mode,tape=None,invalid=False):
    subset=deepcopy(protocol); subset['scenes']=[scene]
    raw=original.episode(subset,control,valid=not invalid,tape=tape)
    from dvidia_training import piper_grip_guard, piper_contact_memory
    raw['grip_guard']=(piper_grip_guard.inactive_summary(mode) if mode=='baseline' else piper_contact_memory.inactive_summary(mode))
    names=('piper_grip_guard.py',) if mode=='baseline' else ('piper_grip_guard.py','piper_contact_memory.py')
    raw['policy_implementation_sha256']={name:protocol['implementation_sha256'][name] for name in names}
    if mode!='baseline' and control=='placement':
        raw['grip_guard'].update(active=True,diagnostic_count=len(raw['actions']))
        for state in raw['trajectory'][:-1]:
            state['guard_diagnostic']={**bench._history_values([],protocol['contact_contract']['parameters']),
                't_seconds':state['t_seconds'],'phase':'approach','bilateral_contact':False,'table_supported':True,
                'memory_stage':'observe','slip_trend_streak':0,'persistent_loss_streak':0,
                'settled_dwell_seconds':0.,'joint_delta_scale':1.,'unscaled_joint_targets_rad':None,
                'resume_local_offset_m':None,'motion_phase':'approach','object_height_delta_m':None}
    return raw


def run_fixture(root,invalid_baseline=False):
    protocol,calls=plan(),[]
    def run(_assets,config,**options):
        assert json.loads((root/'protocol.json').read_bytes())==protocol
        calls.append(options)
        scene=next(s for s in protocol['scenes'] if s['seed']==options['seed'])
        return fixture_episode(protocol,scene,options['control'],options['policy_mode'],options.get('arm_tape'),
            invalid=invalid_baseline and options['control']=='placement' and options['policy_mode']=='baseline')
    with patch.object(bench,'make_protocol',return_value=protocol), \
         patch.object(bench.robot_profiles,'verify_assets',return_value=deepcopy(bench.base._pinned_assets()[0])), \
         patch.object(bench.robot_profiles,'_payload',return_value=({},original.BLOBS)), \
         patch.object(bench.task,'run_episode',side_effect=run):
        report=bench.run_benchmark('unused',root)
    return report,calls


def reseal(root):
    records={}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.name=='integrity.json':
            continue
        data,name=path.read_bytes(),path.relative_to(root).as_posix()
        record={'bytes':len(data),'sha256':sha256(data).hexdigest()}
        if name.endswith('.gz'):
            decoded=bench.decode_attempt(data)
            record.update(encoding='gzip',decoded_bytes=len(decoded),decoded_sha256=sha256(decoded).hexdigest())
        records[name]=record
    (root/'integrity.json').write_bytes(bench._json_bytes({'schema':bench.INTEGRITY_SCHEMA,'files':records}))


def mutate(root,report,index,callback,recompute=True):
    row=report['attempts'][index]
    raw=json.loads(bench.decode_attempt((root/row['result_file']).read_bytes()))
    callback(raw)
    if recompute:
        raw['contact_evaluation']=bench.evaluate_episode(raw)
    (root/row['result_file']).write_bytes(gzip.compress(bench._json_bytes(raw),mtime=0))
    report['attempts'][index]={**bench._summary(raw),'result_file':row['result_file']}
    report['summary']=bench.aggregate(report['attempts'],report['scenes'])
    report['valid_attempts']=sum(r['valid'] for r in report['attempts'])
    (root/'report.json').write_bytes(bench._json_bytes(report))
    reseal(root)


class ContactBenchmarkTests(unittest.TestCase):
    def setUp(self):
        expected,_=bench.base._pinned_assets()
        notices={name:sha256(value).hexdigest() for name,value in original.BLOBS.items()}
        patcher=patch.object(bench.base,'_pinned_assets',return_value=(expected,notices))
        patcher.start(); self.addCleanup(patcher.stop)

    def test_preregistered_cartesian_scene_plan_and_shared_layouts(self):
        p=bench.make_protocol()
        self.assertEqual(p,bench.make_protocol())
        self.assertEqual(p['planned_attempts'],84)
        self.assertEqual(len(p['scenes']),12)
        self.assertEqual(p['variants'],list(bench.VARIANTS))
        self.assertEqual({(s['friction'],s['mass_kg'],s['layout_index']) for s in p['scenes']},
                         {(f,m,i) for f in bench.FRICTIONS for m in bench.MASSES for i in (0,1)})
        for index in (0,1):
            scenes=[s for s in p['scenes'] if s['layout_index']==index]
            self.assertTrue(all(s['config']['object_position_m']==scenes[0]['config']['object_position_m'] and
                s['config']['target_position_m']==scenes[0]['config']['target_position_m'] for s in scenes))
        self.assertEqual(p['contact_contract']['acquisition'],p['acquisition_contract'])
        self.assertEqual(p['contact_contract']['parameters']['maximum_reassessments'],2)
        self.assertNotEqual(p['scenes'],bench.make_protocol(seed=813408)['scenes'])
        for options in ({'seed':True},{'seed':999995},{'cohort':'test'}):
            with self.subTest(options=options),self.assertRaises(ValueError): bench.make_protocol(**options)

    def test_complete_84_attempt_bundle_three_own_tapes_and_exact_gzip(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'run'; report,calls=run_fixture(root)
            self.assertEqual((report['attempted'],report['valid_attempts']),(84,84))
            for a,b in ((0,1),(2,3),(4,5)):
                raw=json.loads(bench.decode_attempt((root/report['attempts'][a]['result_file']).read_bytes()))
                self.assertEqual(calls[b]['arm_tape'],raw['arm_tape'])
            checked,files=bench.load_contact_bundle(root)
            self.assertEqual(checked,report); self.assertEqual(len(files),90)
            self.assertEqual(files[bench.base.LICENSE_PATH],original.LICENSE)
            for name,record in json.loads(files['integrity.json'])['files'].items():
                if name.endswith('.gz'):
                    self.assertEqual(files[name],gzip.compress(bench.decode_attempt(files[name],record),mtime=0))
            self.assertEqual(len(report['summary']),42)
            self.assertTrue(all(r['attempted']==2 and r['no_qualification_decision_attempts']==2 for r in report['summary']))
            self.assertIn('no qualification decision',(root/'report.html').read_text())
            with self.assertRaises(FileExistsError): run_fixture(root)

    def test_unavailable_baseline_tapes_preserve_other_policies_and_denominator(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'run'; report,calls=run_fixture(root,invalid_baseline=True)
            self.assertEqual((report['attempted'],report['valid_attempts'],len(calls)),(84,60,72))
            self.assertFalse(report['attempts'][1]['valid'])
            self.assertTrue(all(report['attempts'][i]['valid'] for i in (2,3,4,5,6)))
            self.assertIsNone(report['attempts'][0]['contact_evaluation']['following']['following_opportunity'])
            self.assertEqual(report['attempts'][1]['grip_guard']['schema'],'dvidia.piper-grip-guard/v1')

    def test_resealed_causal_history_stage_and_intervention_evidence_reject(self):
        def fake_hold(raw):
            guard=raw['grip_guard']; state=raw['trajectory'][0]
            guard['reassessment_count']=1
            guard['events']=[{'type':'reassessment_started','t_seconds':0.,'phase':'memory_hold','reason':'fixture',
                'retry_index':0,'diagnostic':deepcopy(state['guard_diagnostic'])}]
        callbacks=(lambda r:r['trajectory'][0]['guard_diagnostic'].update(history_rate_m_s=.5),
                   lambda r:r['trajectory'][0]['guard_diagnostic'].update(history_count=1),
                   lambda r:r['trajectory'][0]['guard_diagnostic'].update(memory_stage='hold'),
                   lambda r:r['trajectory'][0]['guard_diagnostic'].update(history_displacement_m=10**1000),fake_hold)
        for callback in callbacks:
            with self.subTest(callback=callback),tempfile.TemporaryDirectory() as folder:
                root=Path(folder)/'run'; report,_=run_fixture(root)
                mutate(root,report,4,callback)
                with self.assertRaises(ValueError): bench.load_contact_bundle(root)

    def test_resealed_probe_only_cannot_claim_history_events_or_new_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'run'; report,_=run_fixture(root)
            mutate(root,report,2,lambda r:r['grip_guard'].update(resume_count=1))
            with self.assertRaisesRegex(ValueError,'count'): bench.load_contact_bundle(root)

    def test_resealed_matched_tape_hold_and_native_model_disagreements_reject(self):
        def move(raw):
            for state,action,tape in zip(raw['trajectory'],raw['actions'],raw['arm_tape']):
                action['joint_targets_rad'][0]+=.1; tape[0]+=.1
                state['issued_joint_targets_rad'][0]+=.1; state['command']['joint_targets_rad'][0]+=.1
        for index,callback in ((3,move),(5,move),(6,move),(4,lambda r:r.update(compiled_model_sha256='b'*64))):
            with self.subTest(index=index),tempfile.TemporaryDirectory() as folder:
                root=Path(folder)/'run'; report,_=run_fixture(root); mutate(root,report,index,callback)
                with self.assertRaises(ValueError): bench.load_contact_bundle(root)

    def test_invalid_partial_warning_retained_but_cadence_not_forged(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'run'; report,_=run_fixture(root)
            def partial(raw):
                raw['valid']=raw['metrics']['valid']=False
                raw.update(reason='solver_warning',warnings=['fixture warning'],simulated_seconds=.033)
                raw['actions'],raw['arm_tape'],raw['trajectory']=raw['actions'][:2],raw['arm_tape'][:2],raw['trajectory'][:3]
                raw['trajectory'][-1].update(t_seconds=.033,command=None,issued_joint_targets_rad=None,issued_gripper_position_m=None)
            mutate(root,report,6,partial)
            self.assertEqual(bench.inspect_contact_bundle(root)['valid_attempts'],83)
            mutate(root,report,6,lambda r:r['trajectory'][1].update(t_seconds=.013))
            with self.assertRaisesRegex(ValueError,'cadence'): bench.load_contact_bundle(root)

    def test_resealed_evaluation_and_missing_attempts_reject(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'run'; report,_=run_fixture(root)
            mutate(root,report,4,lambda r:r['contact_evaluation'].update(resumed_completed=True),recompute=False)
            with self.assertRaisesRegex(ValueError,'evaluation'): bench.load_contact_bundle(root)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'run'; report,_=run_fixture(root)
            report['attempts'].pop(); report['attempted']=83
            (root/'report.json').write_bytes(bench._json_bytes(report)); reseal(root)
            with self.assertRaisesRegex(ValueError,'planned paired'): bench.load_contact_bundle(root)

    def test_bounded_decoder_rejects_trailing_member_and_huge_diagnostic_cleanly(self):
        raw=b'{"fixture":true}'; encoded=gzip.compress(raw,mtime=0)
        self.assertEqual(bench.decode_attempt(encoded),raw)
        for bad in (encoded+encoded,encoded[:-1],b'notgzip'):
            with self.assertRaises(ValueError): bench.decode_attempt(bad)
        self.assertFalse(bench._near(10**1000,1.0))


class ActualMotionEvaluationTests(unittest.TestCase):
    def raw(self):
        p=plan(); raw=fixture_episode(p,p['scenes'][0],'placement','history')
        raw['config']=p['scenes'][0]['config']; raw['config']['timeout_s']=1.
        first=raw['trajectory'][0]
        raw['trajectory']=[deepcopy(first) for _ in range(51)]
        for i,state in enumerate(raw['trajectory']):
            state['t_seconds']=i*.02; state['command']={'phase':'memory_hold'}
        raw['grip_guard']['events']=[{'type':'reassessment_started','t_seconds':.2,'reason':'fixture'}]
        return raw

    def test_actual_native_motion_after_hold_and_absent_loss_opportunity(self):
        raw=self.raw()
        for i,state in enumerate(raw['trajectory']):
            state['tcp_position_m'][2]+=.001*i; state['object_position_m'][2]+=.0005*i
        result=bench.evaluate_episode(raw); hold=result['interventions'][0]
        self.assertTrue(hold['after']['complete'])
        self.assertAlmostEqual(hold['after']['tcp_path_m'],.005)
        self.assertAlmostEqual(hold['after']['tcp_mean_speed_m_s'],.05)
        self.assertAlmostEqual(hold['after']['object_mean_speed_m_s'],.025)
        self.assertTrue(hold['sampled_following_loss_comparison']['no_preceding_sampled_loss_opportunity'])
        raw['trajectory']=raw['trajectory'][:12]
        self.assertFalse(bench.evaluate_episode(raw)['interventions'][0]['after']['complete'])

    def test_resume_not_task_completion_and_low_motion_requires_loaded_rise(self):
        raw=self.raw(); raw['grip_guard']['events']=[{'type':'carry_resumed','t_seconds':.2,'reason':'fixture'}]
        raw['valid'],raw['success']=True,False
        result=bench.evaluate_episode(raw)
        self.assertTrue(result['resumed_not_completed']); self.assertFalse(result['resumed_completed'])
        self.assertFalse(result['interventions'][0]['recorded_loaded_unsupported_window'])
        for state in raw['trajectory'][1:]:
            state['table_object_contact_count']=0
            state['grasp_contact_counts']={'left':1,'right':1}; state['pad_normal_forces_n']={'left':1.,'right':1.}
            state['object_position_m'][2]+=.01
        self.assertTrue(bench.evaluate_episode(raw)['interventions'][0]['recorded_low_tcp_motion_window'])
        raw['success']=True
        self.assertTrue(bench.evaluate_episode(raw)['resumed_completed'])


if __name__=='__main__': unittest.main()

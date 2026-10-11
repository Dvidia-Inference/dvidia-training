"""Paired controls, causal guard evidence and bounded compressed receipts."""
from copy import deepcopy
import gzip
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dvidia_training import piper_grip_benchmark as bench
from tests import test_piper_task_benchmark as original


def plan():
    value = json.loads(json.dumps(bench.make_protocol(seed=35)))
    for scene in value['scenes']:
        scene['config']['timeout_s'] = .1
    return value


def fixture_episode(protocol, scene, control, mode, tape=None, invalid=False):
    subset = deepcopy(protocol)
    subset['scenes'] = [scene]
    raw = original.episode(subset,control,valid=not invalid,tape=tape)
    raw['grip_guard'] = {'schema':'dvidia.piper-grip-guard/v1','mode':mode,'active':mode=='guarded' and control=='placement',
                         'contract':protocol['guard_contract'],'events':[],'retry_count':0,'acquisition_qualified':False,
                         'acquisition_qualification_count':0,'abort_reason':None,'aborted':False,'diagnostic_count':0}
    raw['policy_implementation_sha256'] = {'piper_grip_guard.py':protocol['implementation_sha256']['piper_grip_guard.py']}
    if raw['grip_guard']['active']:
        for state in raw['trajectory'][:-1]:
            state['guard_diagnostic'] = {'t_seconds':state['t_seconds'],'bilateral_contact':False,'table_supported':True}
        raw['grip_guard']['diagnostic_count'] = len(raw['actions'])
    return raw


def run_fixture(root, invalid_baseline=False):
    protocol, calls = plan(), []
    def run(_assets, config, **options):
        assert json.loads((root/'protocol.json').read_bytes()) == protocol
        calls.append(options)
        scene = next(s for s in protocol['scenes'] if s['seed']==options['seed'])
        return fixture_episode(protocol,scene,options['control'],options['policy_mode'],options.get('arm_tape'),
                               invalid=invalid_baseline and options['policy_mode']=='baseline' and options['control']=='placement')
    with patch.object(bench,'make_protocol',return_value=protocol), \
            patch.object(bench.robot_profiles,'verify_assets',return_value=deepcopy(bench.base._pinned_assets()[0])), \
            patch.object(bench.robot_profiles,'_payload',return_value=({},original.BLOBS)), \
            patch.object(bench.task,'run_episode',side_effect=run):
        report = bench.run_benchmark('unused',root)
    return report,calls


def reseal(root):
    records = {}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.name=='integrity.json':
            continue
        data, name = path.read_bytes(), path.relative_to(root).as_posix()
        record = {'bytes':len(data),'sha256':sha256(data).hexdigest()}
        if name.endswith('.gz'):
            decoded = bench.decode_attempt(data)
            record.update(encoding='gzip',decoded_bytes=len(decoded),decoded_sha256=sha256(decoded).hexdigest())
        records[name] = record
    (root/'integrity.json').write_bytes(bench._json_bytes({'schema':bench.INTEGRITY_SCHEMA,'files':records}))


def mutate(root, report, index, callback):
    row = report['attempts'][index]
    raw = json.loads(bench.decode_attempt((root/row['result_file']).read_bytes()))
    callback(raw)
    raw['grip_evaluation'] = bench.evaluate_episode(raw)
    (root/row['result_file']).write_bytes(gzip.compress(bench._json_bytes(raw),mtime=0))
    report['attempts'][index] = {**bench._summary(raw),'result_file':row['result_file']}
    report['summary'] = bench.aggregate(report['attempts'],report['scenes'])
    report['valid_attempts'] = sum(r['valid'] for r in report['attempts'])
    (root/'report.json').write_bytes(bench._json_bytes(report))
    reseal(root)


class GripTests(unittest.TestCase):
    def setUp(self):
        expected,_ = bench.base._pinned_assets()
        notices = {name:sha256(value).hexdigest() for name,value in original.BLOBS.items()}
        patcher = patch.object(bench.base,'_pinned_assets',return_value=(expected,notices))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_frozen_paired_plan_shares_layouts_and_keeps_all_variants(self):
        protocol = bench.make_protocol()
        self.assertEqual(protocol,bench.make_protocol())
        self.assertEqual(protocol['planned_attempts'],30)
        self.assertEqual(protocol['variants'],list(bench.VARIANTS))
        self.assertEqual([s['friction'] for s in protocol['scenes']],[.8,.8,.55,.55,.15,.15])
        self.assertEqual(protocol['guard_contract']['parameters']['max_retries'],1)
        for layout in (0,1):
            matching = [s for s in protocol['scenes'] if s['layout_index']==layout]
            self.assertTrue(all(s['config']['object_position_m']==matching[0]['config']['object_position_m'] for s in matching))
            self.assertTrue(all(s['config']['target_position_m']==matching[0]['config']['target_position_m'] for s in matching))
        self.assertNotEqual(protocol['scenes'],bench.make_protocol(seed=712048)['scenes'])
        for options in ({'seed':True},{'seed':999995},{'cohort':'test'}):
            with self.subTest(options=options),self.assertRaises(ValueError):
                bench.make_protocol(**options)

    def test_complete_gzip_bundle_exact_bytes_and_paired_control_tapes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'run'
            report,calls = run_fixture(root)
            self.assertEqual((report['attempted'],report['valid_attempts']),(30,30))
            self.assertEqual(calls[1]['arm_tape'],json.loads(bench.decode_attempt((root/report['attempts'][0]['result_file']).read_bytes()))['arm_tape'])
            self.assertEqual(calls[3]['arm_tape'],json.loads(bench.decode_attempt((root/report['attempts'][2]['result_file']).read_bytes()))['arm_tape'])
            checked,files = bench.load_grip_bundle(root)
            self.assertEqual(checked,report)
            self.assertEqual(len(files),36)
            self.assertEqual(files[bench.base.LICENSE_PATH],original.LICENSE)
            seal = json.loads(files['integrity.json'])
            for name,record in seal['files'].items():
                if name.endswith('.gz'):
                    raw = bench.decode_attempt(files[name],record)
                    self.assertEqual(files[name],gzip.compress(raw,mtime=0))
            self.assertTrue(all(row['no_qualification_decision_attempts']==2 for row in report['summary']))
            self.assertIn('no qualification decision',(root/'report.html').read_text())
            with self.assertRaises(FileExistsError):
                run_fixture(root)

    def test_invalid_baseline_does_not_remove_guarded_or_dependent_attempts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'run'
            report,calls = run_fixture(root,invalid_baseline=True)
            self.assertEqual(report['attempted'],30)
            self.assertEqual(len(calls),24)
            self.assertEqual(report['valid_attempts'],18)
            self.assertFalse(report['attempts'][1]['valid'])
            self.assertTrue(report['attempts'][2]['valid'])
            self.assertTrue(report['attempts'][3]['valid'])
            self.assertIsNone(report['attempts'][0]['grip_evaluation']['following_opportunity'])

    def test_resealed_wrong_own_tape_hold_and_native_mechanics_fail(self):
        def wrong_tape(raw):
            for state,action,tape in zip(raw['trajectory'],raw['actions'],raw['arm_tape']):
                action['joint_targets_rad'][0] += .1
                tape[0] += .1
                state['issued_joint_targets_rad'][0] += .1
                state['command']['joint_targets_rad'][0] += .1
        for index,callback in ((3,wrong_tape),(4,wrong_tape),(2,lambda raw:raw.update(compiled_model_sha256='b'*64))):
            with self.subTest(index=index),tempfile.TemporaryDirectory() as folder:
                root = Path(folder)/'run'
                report,_ = run_fixture(root)
                mutate(root,report,index,callback)
                with self.assertRaises(ValueError):
                    bench.load_grip_bundle(root)

    def test_resealed_guard_causal_hold_recovery_and_unqualified_carry_fail(self):
        def fake_alarm(raw):
            state = raw['trajectory'][0]
            raw['grip_guard']['events'] = [{'type':'alarm','t_seconds':0.,'phase':'guard_stop','reason':'fixture',
                                          'retry_index':0,'diagnostic':deepcopy(state['guard_diagnostic'])}]
        def fake_retry(raw):
            state = raw['trajectory'][0]
            state['table_object_contact_count']=0
            state['guard_diagnostic']['table_supported']=False
            raw['grip_guard']['retry_count']=1
            raw['grip_guard']['events']=[{'type':'retry_started','t_seconds':0.,'phase':'guard_stop','reason':'fixture',
                                        'retry_index':1,'diagnostic':deepcopy(state['guard_diagnostic'])}]
        def early_carry(raw):
            raw['actions'][0]['phase']=raw['trajectory'][0]['command']['phase']='transfer'
        def future_event(raw):
            fake_alarm(raw)
            raw['grip_guard']['events'][0]['t_seconds']=.01
        def fake_qualification(raw):
            guard=raw['grip_guard']
            guard['acquisition_qualified']=True
            guard['acquisition_qualification_count']=1
            guard['events']=[{'type':'probe_started','t_seconds':0.,'phase':'guard_probe','reason':'fixture','retry_index':0,
                              'diagnostic':deepcopy(raw['trajectory'][0]['guard_diagnostic'])},
                             {'type':'acquisition_qualified','t_seconds':.08,'phase':'guard_probe','reason':'fixture','retry_index':0,
                              'diagnostic':deepcopy(raw['trajectory'][4]['guard_diagnostic'])}]
            for event in guard['events']:
                event['diagnostic'].update(object_height_delta_m=.02,qualification_dwell_seconds=.1)
        for callback in (fake_alarm,fake_retry,early_carry,future_event,fake_qualification):
            with self.subTest(callback=callback.__name__),tempfile.TemporaryDirectory() as folder:
                root = Path(folder)/'run'
                report,_ = run_fixture(root)
                mutate(root,report,2,callback)
                with self.assertRaises(ValueError):
                    bench.load_grip_bundle(root)

    def test_resealed_arbitrary_guard_thresholds_and_partial_warning_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'run'
            report,_=run_fixture(root)
            protocol=json.loads((root/'protocol.json').read_bytes())
            protocol['guard_contract']['parameters']['probe_min_object_rise_m']=0.
            (root/'protocol.json').write_bytes(bench._json_bytes(protocol))
            report['protocol']=protocol
            report['protocol_sha256']=sha256((root/'protocol.json').read_bytes()).hexdigest()
            (root/'report.json').write_bytes(bench._json_bytes(report))
            reseal(root)
            with self.assertRaisesRegex(ValueError,'protocol/units'):
                bench.load_grip_bundle(root)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'run'
            report,_=run_fixture(root)
            def partial(raw):
                raw['valid']=raw['metrics']['valid']=False
                raw['reason'],raw['warnings'],raw['simulated_seconds']='solver_warning',['fixture warning'],.033
                raw['actions'],raw['arm_tape'],raw['trajectory']=raw['actions'][:2],raw['arm_tape'][:2],raw['trajectory'][:3]
                raw['trajectory'][-1].update(t_seconds=.033,command=None,issued_joint_targets_rad=None,issued_gripper_position_m=None)
            mutate(root,report,4,partial)
            checked=bench.inspect_grip_bundle(root)
            self.assertEqual((checked['attempted'],checked['valid_attempts']),(30,29))
            mutate(root,report,4,lambda raw:raw['trajectory'][1].update(t_seconds=.013))
            with self.assertRaisesRegex(ValueError,'cadence'):
                bench.load_grip_bundle(root)

    def test_resealed_evaluation_and_dropped_denominator_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'run'
            report,_ = run_fixture(root)
            row = report['attempts'][2]
            raw = json.loads(bench.decode_attempt((root/row['result_file']).read_bytes()))
            raw['grip_evaluation']['following_opportunity']=True
            (root/row['result_file']).write_bytes(gzip.compress(bench._json_bytes(raw),mtime=0))
            report['attempts'][2]={**bench._summary(raw),'result_file':row['result_file']}
            report['summary']=bench.aggregate(report['attempts'],report['scenes'])
            (root/'report.json').write_bytes(bench._json_bytes(report))
            reseal(root)
            with self.assertRaisesRegex(ValueError,'sampled evaluation'):
                bench.load_grip_bundle(root)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'run'
            report,_ = run_fixture(root)
            report['attempts'].pop()
            report['attempted']=29
            (root/'report.json').write_bytes(bench._json_bytes(report))
            reseal(root)
            with self.assertRaisesRegex(ValueError,'planned paired'):
                bench.load_grip_bundle(root)

    def test_resealed_inactive_acquisition_flag_cannot_invent_a_decision(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'run'
            report,_=run_fixture(root)
            mutate(root,report,1,lambda raw:raw['grip_guard'].update(acquisition_qualified=True))
            with self.assertRaisesRegex(ValueError,'final decision'):
                bench.load_grip_bundle(root)

    def test_single_member_raw_hash_and_decompression_bounds(self):
        raw = b'{"raw":"fixture"}'
        encoded = gzip.compress(raw,mtime=0)
        self.assertEqual(bench.decode_attempt(encoded),raw)
        for data in (encoded[:-1],encoded+encoded,b'not gzip'):
            with self.subTest(data=data[:10]),self.assertRaises(ValueError):
                bench.decode_attempt(data)
        with patch.object(bench,'MAX_FILE_BYTES',100),self.assertRaisesRegex(ValueError,'byte bound'):
            bench.decode_attempt(gzip.compress(b'a'*1000,mtime=0))
        with self.assertRaisesRegex(ValueError,'digest/size'):
            bench.decode_attempt(encoded,{'encoding':'gzip','decoded_bytes':len(raw),'decoded_sha256':'0'*64})


class IndependentEvaluationTests(unittest.TestCase):
    def raw(self):
        protocol = plan()
        raw = fixture_episode(protocol,protocol['scenes'][0],'placement','guarded')
        raw['config']=deepcopy(protocol['scenes'][0]['config'])
        raw['config']['timeout_s']=1.
        first = deepcopy(raw['trajectory'][0])
        raw['trajectory']=[deepcopy(first) for _ in range(51)]
        for i,state in enumerate(raw['trajectory']):
            state['t_seconds']=i*.02
            state['command']={'phase':'guard_probe'}
        raw['grip_guard']['events']=[{'type':'acquisition_qualified','t_seconds':0.,'reason':'fixture'}]
        return raw

    def test_following_truth_is_future_retention_not_a_controller_flag(self):
        raw = self.raw()
        self.assertFalse(bench.evaluate_episode(raw)['qualification_decisions'][0]['following_observed'])
        for state in raw['trajectory'][1:]:
            state['object_position_m'][2]+=.02
            state['tcp_position_m'][2]+=.02
            state['table_object_contact_count']=0
        result = bench.evaluate_episode(raw)
        self.assertTrue(result['qualification_decisions'][0]['following_observed'])
        self.assertTrue(result['following_opportunity'])
        raw['trajectory']=raw['trajectory'][:2]
        self.assertIsNone(bench.evaluate_episode(raw)['qualification_decisions'][0]['following_observed'])

    def test_post_alarm_motion_includes_hold_inertia_and_empty_delay_opportunity(self):
        raw = self.raw()
        raw['grip_guard']['events']=[{'type':'alarm','t_seconds':.2,'reason':'fixture'}]
        for index,state in enumerate(raw['trajectory']):
            state['tcp_position_m'][2]+=.001*index
        alarm = bench.evaluate_episode(raw)['alarms'][0]
        self.assertTrue(alarm['post_alarm_window_complete'])
        self.assertAlmostEqual(alarm['tcp_path_m'],.005)
        self.assertAlmostEqual(alarm['tcp_rise_m'],.005)
        self.assertIsNone(alarm['sampled_loss_to_alarm_delay_s'])
        self.assertTrue(alarm['no_preceding_sampled_loss_opportunity'])


if __name__=='__main__':
    unittest.main()

"""v0.4 paired denominators, causal recovery decisions and raw clock samples."""
from copy import deepcopy
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dvidia_training import piper_recovery_benchmark as bench
from dvidia_training import piper_contact_recovery as recovery
from tests import test_piper_contact_benchmark as contact_fixture
from tests import test_piper_task_benchmark as original


def plan():
    protocol=json.loads(json.dumps(bench.make_protocol(seed=35)))
    for scene in protocol['scenes']:
        scene['config']['timeout_s']=.1
    return protocol


def fixture_episode(protocol,scene,control,mode,tape=None,invalid=False):
    raw=contact_fixture.fixture_episode(protocol,scene,control,'history' if mode in recovery.MODES else mode,tape,invalid)
    if mode in recovery.MODES:
        value=raw['grip_guard']
        value.update(recovery.inactive_summary(mode),active=control=='placement',diagnostic_count=len(raw['actions']) if control=='placement' else 0)
        raw['policy_implementation_sha256']['piper_contact_recovery.py']=protocol['implementation_sha256']['piper_contact_recovery.py']
        for state in raw['trajectory']:
            if state.get('guard_diagnostic') is not None:
                state['guard_diagnostic'].update(bench._acquisition_values([],protocol['recovery_contract']['parameters']),
                    restoration_stable_dwell_seconds=0.,restoration_progress_seconds=0.,restoration_started=False,
                    restoration_complete=False,recovery_scale=1.)
    values=[.001]*len(raw['actions'])
    raw['timing']['control_step_samples_seconds']=values
    if values: raw['timing']['control_step_seconds']=bench.task.sim._summary(values)
    return raw


def run_fixture(root,invalid_history=False):
    protocol,calls=plan(),[]
    def run(_assets,config,**options):
        assert json.loads((root/'protocol.json').read_bytes())==protocol
        calls.append(options)
        scene=next(s for s in protocol['scenes'] if s['seed']==options['seed'])
        return fixture_episode(protocol,scene,options['control'],options['policy_mode'],options.get('arm_tape'),
            invalid=invalid_history and options['control']=='placement' and options['policy_mode']=='history')
    with patch.object(bench,'make_protocol',return_value=protocol), \
         patch.object(bench.robot_profiles,'verify_assets',return_value=deepcopy(bench.base._pinned_assets()[0])), \
         patch.object(bench.robot_profiles,'_payload',return_value=({},original.BLOBS)), \
         patch.object(bench.task,'run_episode',side_effect=run):
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


def mutate(root,report,index,callback,recompute=True):
    row=report['attempts'][index]
    raw=json.loads(bench.decode_attempt((root/row['result_file']).read_bytes()))
    callback(raw)
    if recompute: raw['recovery_evaluation']=bench.evaluate_episode(raw)
    (root/row['result_file']).write_bytes(gzip.compress(bench._json_bytes(raw),mtime=0))
    report['attempts'][index]={**bench._summary(raw),'result_file':row['result_file']}
    report['summary']=bench.aggregate(report['attempts'],report['scenes'])
    report['paired_comparisons']=bench.paired_comparisons(report['attempts'],report['scenes'])
    report['valid_attempts']=sum(r['valid'] for r in report['attempts'])
    (root/'report.json').write_bytes(bench._json_bytes(report));reseal(root)


def inspect_fixture(root,report):
    with patch.object(bench,'make_protocol',return_value=report['protocol']):
        return bench.load_recovery_bundle(root)


class RecoveryBenchmarkTests(unittest.TestCase):
    def setUp(self):
        expected,_=bench.base._pinned_assets()
        notices={name:sha256(value).hexdigest() for name,value in original.BLOBS.items()}
        patcher=patch.object(bench.base,'_pinned_assets',return_value=(expected,notices))
        patcher.start();self.addCleanup(patcher.stop)

    def test_fixed_plan_is_matched_and_distinguishes_acquisition_ablation(self):
        p=bench.make_protocol()
        self.assertEqual(p,bench.make_protocol())
        self.assertEqual(p['planned_attempts'],84);self.assertEqual(len(p['scenes']),12)
        self.assertEqual(p['variants'],list(bench.VARIANTS))
        self.assertEqual(p['recovery_contract']['prior_contact_contract'],p['contact_contract'])
        self.assertEqual(bench._variant('recovery_carry_placement'),('placement','recovery_carry'))
        self.assertNotEqual(p['scenes'],bench.make_protocol(seed=813408)['scenes'])

    def test_complete_receipt_has_own_controls_and_raw_timing(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'run';report,calls=run_fixture(root)
            checked,files=inspect_fixture(root,report)
            self.assertEqual(checked,report);self.assertEqual(len(files),90)
            self.assertEqual((report['attempted'],report['valid_attempts']),(84,84))
            self.assertTrue(all(c['record_step_timings'] is True for c in calls))
            for placed,opened in ((0,1),(2,3),(4,5)):
                raw=json.loads(bench.decode_attempt(files[report['attempts'][placed]['result_file']]))
                self.assertEqual(calls[opened]['arm_tape'],raw['arm_tape'])

    def test_invalid_history_preserves_dependent_failure_and_other_modes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'run';report,calls=run_fixture(root,True)
            self.assertEqual(report['attempted'],84);self.assertEqual(report['valid_attempts'],60)
            self.assertFalse(report['attempts'][1]['valid'])
            self.assertTrue(report['attempts'][2]['valid'])
            inspect_fixture(root,report)

    def test_resealed_native_start_velocity_and_own_tape_changes_reject(self):
        def velocity(raw): raw['trajectory'][0]['qvel'][-1]=.2
        def tape(raw):
            raw['arm_tape'][0][0]+=.001;raw['actions'][0]['joint_targets_rad'][0]+=.001
            raw['trajectory'][0]['command']['joint_targets_rad'][0]+=.001
            raw['trajectory'][0]['issued_joint_targets_rad'][0]+=.001
        for index,callback in ((4,velocity),(3,tape),(4,lambda r:r.update(compiled_model_sha256='b'*64))):
            with self.subTest(index=index),tempfile.TemporaryDirectory() as folder:
                root=Path(folder)/'run';report,_=run_fixture(root);mutate(root,report,index,callback)
                with self.assertRaises(ValueError): inspect_fixture(root,report)

    def test_resealed_invented_acquisition_window_or_restoration_reject(self):
        callbacks=(lambda r:r['trajectory'][0]['guard_diagnostic'].update(acquisition_stable_closed_supported=True),
                   lambda r:r['grip_guard'].update(restoration_started_count=1),
                   lambda r:r['grip_guard'].update(unchanged_retry_refused=True),
                   lambda r:r['trajectory'][0]['guard_diagnostic'].update(history_span_seconds=.1))
        for callback in callbacks:
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder)/'run';report,_=run_fixture(root);mutate(root,report,4,callback)
                with self.assertRaises(ValueError): inspect_fixture(root,report)

    def test_resealed_evaluation_summary_and_missing_latency_reject(self):
        callbacks=(lambda r:r['recovery_evaluation'].update(completion_time_s=1.),
                   lambda r:r['timing'].pop('control_step_samples_seconds'),
                   lambda r:r['timing']['control_step_samples_seconds'].__setitem__(0,.5))
        for callback in callbacks:
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder)/'run';report,_=run_fixture(root);mutate(root,report,4,callback,recompute=False)
                with self.assertRaises(ValueError): inspect_fixture(root,report)

    def test_resealed_completion_bounds_and_consistent_but_impossible_clock_reject(self):
        def clocks(raw):
            values=[1.]*len(raw['actions'])
            raw['timing'].update(control_step_samples_seconds=values,
                control_step_seconds=bench.task.sim._summary(values),missed_control_boundaries=len(values))
        callbacks=[lambda r,value=value:r['metrics'].update(first_success_time_seconds=value) for value in (True,-1.,21.)]+[clocks]
        for callback in callbacks:
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder)/'run';report,_=run_fixture(root);mutate(root,report,4,callback)
                with self.assertRaises(ValueError): inspect_fixture(root,report)

    def test_shortened_or_changed_seeded_scenes_do_not_pass_real_inspector(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'run';report,_=run_fixture(root)
            with self.assertRaisesRegex(ValueError,'seeded fixed'): bench.load_recovery_bundle(root)


class RecoveryMetricsTests(unittest.TestCase):
    def test_paired_speed_uses_common_success_and_keeps_outcome_regressions(self):
        scenes=[{'id':'a'},{'id':'b'},{'id':'c'}];rows=[]
        for scene in scenes:
            for mode in bench.MODES:
                success=(scene['id']=='a' or scene['id']=='b' and mode=='history' or scene['id']=='c' and mode!='history')
                rows.append({'scene_id':scene['id'],'variant':mode+'_placement','valid':True,'success':success,
                    'recovery_evaluation':{'completion_time_s':10. if mode=='history' else 8.,'abort_time_s':None}})
        result=bench.paired_comparisons(rows,scenes)[0]
        self.assertEqual((result['common_successes'],result['gained_successes'],result['lost_successes']),(1,1,1))
        self.assertEqual(result['common_success_mean_delta_s'],-2.)
        self.assertIsNone(result['pairs'][1]['completion_delta_s'])

    def test_hold_without_resume_remains_unresolved_with_actual_motion(self):
        raw=contact_fixture.ActualMotionEvaluationTests().raw()
        raw['grip_guard']['events']=[{'type':'reassessment_started','t_seconds':.2,'reason':'fixture'}]
        raw['simulated_seconds']=1.
        result=bench.evaluate_episode(raw)
        self.assertEqual(result['holds'][0]['outcome'],'unresolved')
        self.assertAlmostEqual(result['holds'][0]['duration_s'],.8)
        self.assertTrue(result['holds'][0]['actual_motion']['complete'])
        self.assertIsNone(result['completion_time_s'])


class NativeRecoveryInspectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assets=os.environ.get('DVIDIA_ROBOT_PROFILE_ASSETS')
        if not assets:
            raise unittest.SkipTest('Pinned native assets not configured')
        cls.protocol=bench.make_protocol(seed=813407)
        scene=next(s for s in cls.protocol['scenes'] if s['id']=='friction055-mass040-layout00')
        cls.raw=bench.task.run_episode(assets,bench.task.PiperTaskConfig(**scene['config']),
            seed=scene['seed'],policy_mode='recovery',record_step_timings=True)
        cls.raw.update(variant='recovery_placement',attempt_wall_seconds=1.)
        cls.raw['recovery_evaluation']=bench.evaluate_episode(cls.raw)
        if not cls.raw['valid'] or not cls.raw['success']:
            raise AssertionError('Native recovery regression fixture did not complete')
        if not any(e['type']=='full_speed_restored' for e in cls.raw['grip_guard']['events']):
            raise AssertionError('Native fixture did not exercise speed restoration')

    def test_real_restoration_and_raw_timing_validate(self):
        bench._recovery(self.raw,self.protocol)
        bench._timing_samples(self.raw)
        bench._completion_timestamp(self.raw)

    def test_forged_scale_and_missing_transition_cannot_pass_old_gate_projection(self):
        for kind in ('scale','event','event_diagnostic'):
            raw=deepcopy(self.raw)
            if kind=='scale':
                state=next(s for s in raw['trajectory'] if s.get('guard_diagnostic',{}).get('recovery_scale',0.)>.45
                    and s.get('guard_diagnostic',{}).get('unscaled_joint_targets_rad') is not None
                    and s['command']['phase']!='memory_slowdown')
                state['guard_diagnostic']['restoration_progress_seconds']+=.02
            elif kind=='event_diagnostic':
                event=next(e for e in raw['grip_guard']['events'] if e['type']=='full_speed_restored')
                event['diagnostic']['restoration_progress_seconds']+=.02
            else:
                raw['grip_guard']['events']=[e for e in raw['grip_guard']['events'] if e['type']!='full_speed_restored']
                raw['grip_guard']['full_speed_restored_count']=0
            raw['recovery_evaluation']=bench.evaluate_episode(raw)
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                bench._recovery(raw,self.protocol)

    def test_completion_timestamp_requires_corroborating_dwell_but_survives_later_failure(self):
        for value in (None,True,-1.,float('nan'),21.,1.):
            raw=deepcopy(self.raw);raw['metrics']['first_success_time_seconds']=value
            with self.subTest(value=value),self.assertRaises(ValueError): bench._completion_timestamp(raw)
        raw=deepcopy(self.raw);raw['success']=False
        bench._completion_timestamp(raw)
        self.assertIsNone(bench.evaluate_episode(raw)['completion_time_s'])


if __name__=='__main__': unittest.main()

"""Runner contracts with fake renderer/estimator only; no native simulation."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import io
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dvidia_training import piper_rgbd_benchmark as bench


def with_signals(result):
    result['age_seconds']=result['decision_at_seconds']-result['captured_at_seconds']
    result['signals']={}
    for name in bench.SIGNALS:
        if result['status']=='available':
            value=result[name]
            result['signals'][name]=dict(status='available',captured_at_seconds=result['captured_at_seconds'],
                available_at_seconds=result['available_at_seconds'],age_seconds=result['age_seconds'],
                value=value if isinstance(value,list) else [value])
        else:
            result['signals'][name]=dict(status='missing',captured_at_seconds=None,
                available_at_seconds=None,age_seconds=None,value=None)
    return result


def missing_result(rgb, depth, calibration, **timing):
    return with_signals(dict(status='missing',reason='fake_test_abstention',**timing,
                **{name:None for name in bench.SIGNALS}))


class FakeSession:
    failures = set()
    closes = []

    def __init__(self, assets, config):
        self.config=config; self.renderer_warnings=[]
        self.env=SimpleNamespace(data=SimpleNamespace(qpos=np.zeros(7)),block_geom=0,
            model=SimpleNamespace(jnt_qposadr=np.array([0]),geom_rgba=np.array([[.75,.3,.16,1.]])))
        self.mj=SimpleNamespace(mjtObj=SimpleNamespace(mjOBJ_JOINT=0),mj_name2id=lambda *args:0)

    def open(self):
        if self.config.width in self.failures: raise RuntimeError('fake context failure')
        return {'binding':{}},None,{'assets':{},'runtime':{},'camera':{}}

    def close(self): self.closes.append(self.config.width)


def fake_binding(session, header, metadata):
    return dict(resolution=session.config.width,calibration={'id':'fake-calibration'},
                object_half_size_m=[.02]*3,object_rgba=[.75,.3,.16,1.])


def fake_render(session,item,group,folder,baseline,address,rgba,binding):
    rgb=np.zeros((16,16,3),dtype=np.uint8);depth=np.ones((16,16),dtype=np.float32)
    bench._record(folder/'rgb.png',bench._png(rgb));bench._npy(folder/'rgb.npy',rgb);bench._npy(folder/'depth.npy',depth)
    timing=dict(frame_index=item['frame_index'],captured_at_seconds=.2,available_at_seconds=.2,decision_at_seconds=.2)
    bench._json(folder/'input.json',dict(calibration=binding['calibration'],timing=timing))
    bench._json(folder/'truth.json',dict(position_m=[.3,0,.1],quaternion_wxyz=[1,0,0,0],yaw_rad=0.,tilt_rad=0.,present=item['variant']!='absent'))


class RunnerContracts(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'study';bench.prepare(self.root)
        self.protocol=bench._load(self.root/'protocol.json')
        # Two independent groups, ten variants, two resolutions: 20 IDs per split.
        self.protocol['groups']=[self.protocol['groups'][0],self.protocol['groups'][8]]
        self.write(self.root/'protocol.json',self.protocol)
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(bench,'_NativeSession',FakeSession))
        self.stack.enter_context(patch.object(bench,'_renderer_binding',fake_binding))
        self.stack.enter_context(patch.object(bench,'_render_fixture',fake_render))
        self.stack.enter_context(patch.object(bench,'runtime',lambda:{'runtime':'test'}))
        self.stack.enter_context(patch.object(bench,'hashes',lambda:{'source':'test'}))
        FakeSession.failures=set();FakeSession.closes=[]

    @staticmethod
    def write(path,value): path.write_text(json.dumps(value,allow_nan=False)+'\n')

    def render(self,split='development'): return bench.render(self.root,'unused',split)

    def estimate(self,split='development',estimator=missing_result):
        with patch('dvidia_training.piper_rgbd_tracker.estimate_pose',estimator):
            return bench.estimate(self.root,split)

    def score(self,split='development'): return bench.score(self.root,split)

    def ready(self): self.render();self.estimate();self.score()

    def test_predeclared_schedule_preserves_fixture_failure(self):
        calls=[]
        def fail_one(session,item,*args):
            calls.append(item['id'])
            if item['frame_index']==1: raise ValueError('fixture failed')
            return fake_render(session,item,*args)
        with patch.object(bench,'_render_fixture',fail_one): result=self.render()
        self.assertEqual(result,dict(scheduled=20,rendered=19,split='development',pipeline_complete=False))
        self.assertEqual(len(calls),20)
        manifest=bench._load(self.root/'development/manifest.json')
        self.assertEqual(len(manifest['records']),20)
        self.assertEqual(manifest['records'][1]['render_status'],'failed')
        self.assertTrue((self.root/'development'/manifest['records'][1]['id']/'started.json').exists())
        self.estimate();self.score()
        report=bench._load(self.root/'development/report.json')
        self.assertEqual(report['scheduled'],20)
        self.assertEqual(sum(r['status']=='pipeline_failure' for r in report['rows']),1)
        self.assertFalse(report['gates'][0]['checks']['all_scheduled_fixtures_complete'])

    def test_setup_failure_keeps_all_ids_and_other_resolution(self):
        FakeSession.failures={160};self.assertEqual(self.render()['rendered'],10)
        rows=bench._load(self.root/'development/manifest.json')['records']
        self.assertEqual([r['render_status'] for r in rows[:10]],['unattempted']*10)
        self.assertEqual([r['render_status'] for r in rows[10:]],['rendered']*10)
        self.assertEqual(FakeSession.closes,[160,320])

    def test_interrupt_never_retries_remaining_fixtures(self):
        with patch.object(bench,'_render_fixture',side_effect=KeyboardInterrupt):
            result=self.render()
        rows=bench._load(self.root/'development/manifest.json')['records']
        self.assertEqual(result['rendered'],0)
        self.assertEqual(rows[0]['render_status'],'failed')
        self.assertTrue(all(r['render_status']=='unattempted' for r in rows[1:]))
        self.assertEqual(FakeSession.closes,[160])

    def test_estimator_exception_preserved_and_reinvocation_refused(self):
        self.render();calls=[]
        def estimator(*args,**kwargs):
            calls.append(kwargs['frame_index'])
            if kwargs['frame_index']==2:raise RuntimeError('failure')
            return missing_result(*args,**kwargs)
        result=self.estimate(estimator=estimator)
        self.assertEqual(result['estimated'],19);self.assertEqual(len(calls),20)
        self.score()
        self.assertEqual(sum(s['pipeline_failures'] for s in bench._load(self.root/'development/report.json')['summary']),1)
        with self.assertRaises(FileExistsError):self.estimate()

    def test_truth_is_never_read_by_estimate(self):
        self.render();original_load=bench._load;opened=[]
        def spy(path):
            opened.append(Path(path).name)
            if Path(path).name=='truth.json':raise AssertionError('leaked truth')
            return original_load(path)
        with patch.object(bench,'_load',spy):self.estimate()
        self.assertNotIn('truth.json',opened)

    def test_missing_duplicate_and_unknown_prediction_ids_refused(self):
        self.render();self.estimate();path=self.root/'development/predictions.json';original=bench._load(path)
        alternatives=[original['predictions'][:-1], original['predictions']+[original['predictions'][0]],
                      [dict(original['predictions'][0],id='unknown')]+original['predictions'][1:]]
        for alternative in alternatives:
            self.write(path,dict(original,predictions=alternative))
            with self.assertRaisesRegex(ValueError,'every scheduled ID'):self.score()

    def test_changed_truth_refuses_offline_score(self):
        self.render();self.estimate();item=bench._load(self.root/'development/manifest.json')['records'][0]
        path=self.root/'development'/item['id']/'truth.json';truth=bench._load(path);truth['position_m'][0]+=1
        self.write(path,truth)
        with self.assertRaisesRegex(ValueError,'artifact changed'):self.score()

    def test_changed_image_is_not_reestimated(self):
        self.render();item=bench._load(self.root/'development/manifest.json')['records'][0]
        path=self.root/'development'/item['id']/'rgb.npy';path.write_bytes(b'altered')
        result=self.estimate();self.assertEqual(result['estimated'],19)
        with self.assertRaisesRegex(ValueError,'artifact changed'):self.score()

    def test_changed_prediction_receipt_refuses_score(self):
        self.render();self.estimate();pred=bench._load(self.root/'development/predictions.json')['predictions'][0]
        (self.root/'development/estimates'/(pred['id']+'.json')).write_text('{}')
        with self.assertRaisesRegex(ValueError,'journal changed'):self.score()

    def test_freeze_checks_protocol_runtime_and_calibration(self):
        self.ready();bench.freeze(self.root)
        with patch.object(bench,'runtime',lambda:{'runtime':'different'}):
            with self.assertRaisesRegex(ValueError,'differs from freeze'):self.render('confirmation')
        protocol=deepcopy(self.protocol);protocol['seed']=999;self.write(self.root/'protocol.json',protocol)
        with self.assertRaisesRegex(ValueError,'differs from freeze'):self.render('confirmation')
        self.write(self.root/'protocol.json',self.protocol)
        with patch.object(bench,'_renderer_binding',lambda *args:dict(fake_binding(*args),calibration={'id':'changed'})):
            result=self.render('confirmation')
        self.assertEqual(result['rendered'],0)
        with self.assertRaisesRegex(ValueError,'rendering differs'):self.estimate('confirmation')

    def test_frozen_confirmation_and_descriptive_gate(self):
        self.ready();bench.freeze(self.root);self.render('confirmation');self.estimate('confirmation');self.score('confirmation')
        report=bench._load(self.root/'confirmation/report.json')
        self.assertEqual(report['scheduled'],20)
        self.assertTrue(all(g['checks']['no_negative_false_available'] for g in report['gates']))
        self.assertTrue(all(not g['passed'] for g in report['gates']))
        self.assertFalse(report['physical_robot_ready'])
        with self.assertRaisesRegex(ValueError,'after confirmation'):bench.freeze(self.root)

    def test_false_available_prevents_gate_even_when_clear_passes(self):
        self.render()
        def available(*args,**timing):
            return with_signals(dict(status='available',reason='fake',base_frame='robot_base',**timing,
                object_position_estimate_m=[.3,0,.1],object_quaternion_estimate_wxyz=[1.,0,0,0],
                object_position_uncertainty_m=.001,object_orientation_uncertainty_rad=.01))
        self.estimate(estimator=available);self.score()
        report=bench._load(self.root/'development/report.json')
        self.assertTrue(all(g['checks']['clear_position'] for g in report['gates']))
        self.assertTrue(all(g['checks']['clear_coverage'] for g in report['gates']))
        self.assertTrue(all(not g['passed'] for g in report['gates']))
        clear=next(s for s in report['summary'] if s['variant']=='clear')
        self.assertEqual(clear['position_error_within_heuristic_scale'],1)

    def test_upright_native_roundoff_keeps_yaw_denominator(self):
        def fixture(session,item,group,folder,*args):
            fake_render(session,item,group,folder,*args)
            truth=bench._load(folder/'truth.json');truth['tilt_rad']=1.49e-8
            self.write(folder/'truth.json',truth)
        with patch.object(bench,'_render_fixture',fixture):self.render()
        def available(*args,**timing):
            if timing['frame_index']%10: return missing_result(*args,**timing)
            return with_signals(dict(status='available',reason='fake',base_frame='robot_base',**timing,
                object_position_estimate_m=[.3,0,.1],object_quaternion_estimate_wxyz=[1.,0,0,0],
                object_position_uncertainty_m=.001,object_orientation_uncertainty_rad=.01))
        self.estimate(estimator=available);self.score()
        report=bench._load(self.root/'development/report.json')
        self.assertTrue(all(s['yaw_evaluated']==s['available'] for s in report['summary'] if s['variant']=='clear'))
        self.assertTrue(all(g['checks']['clear_pose_metrics_complete'] for g in report['gates']))
        self.assertTrue(all(g['passed'] for g in report['gates']))

    def test_unscorable_available_clear_pose_fails_gate(self):
        def fixture(session,item,group,folder,*args):
            fake_render(session,item,group,folder,*args)
            truth=bench._load(folder/'truth.json');truth['tilt_rad']=2e-6
            self.write(folder/'truth.json',truth)
        with patch.object(bench,'_render_fixture',fixture):self.render()
        def available(*args,**timing):
            if timing['frame_index']%10: return missing_result(*args,**timing)
            return with_signals(dict(status='available',reason='fake',base_frame='robot_base',**timing,
                object_position_estimate_m=[.3,0,.1],object_quaternion_estimate_wxyz=[1.,0,0,0],
                object_position_uncertainty_m=.001,object_orientation_uncertainty_rad=.01))
        self.estimate(estimator=available);self.score()
        report=bench._load(self.root/'development/report.json')
        self.assertTrue(all(not g['checks']['clear_pose_metrics_complete'] for g in report['gates']))
        self.assertTrue(all(not g['passed'] for g in report['gates']))

    def test_freeze_rejects_predictions_from_older_candidate(self):
        self.ready()
        with patch.object(bench,'hashes',lambda:{'source':'new-candidate'}):
            with self.assertRaisesRegex(ValueError,'predictions do not bind the current'):bench.freeze(self.root)
        self.assertFalse((self.root/'freeze.json').exists())

    def test_freeze_rejects_report_from_other_predictions_or_scorer(self):
        self.ready();path=self.root/'development/report.json';original=bench._load(path)
        for bad in (dict(original,predictions_sha256='0'*64),dict(original,scorer_source_hashes={'source':'older'})):
            self.write(path,bad)
            with self.assertRaisesRegex(ValueError,'report does not bind'):bench.freeze(self.root)
        self.assertFalse((self.root/'freeze.json').exists())

    def test_confirmation_score_rejects_changed_current_code(self):
        self.ready();bench.freeze(self.root);self.render('confirmation');self.estimate('confirmation')
        with patch.object(bench,'hashes',lambda:{'source':'edited-after-freeze'}):
            with self.assertRaisesRegex(ValueError,'differs from freeze'):self.score('confirmation')
        self.assertFalse((self.root/'confirmation/report.json').exists())

    def test_nested_observation_contract_rejects_replacement_estimator(self):
        timing=dict(frame_index=0,captured_at_seconds=.2,available_at_seconds=.2,decision_at_seconds=.2)
        valid=with_signals(dict(status='available',reason='fake',base_frame='robot_base',**timing,
            object_position_estimate_m=[.3,0,.1],object_quaternion_estimate_wxyz=[1.,0,0,0],
            object_position_uncertainty_m=.001,object_orientation_uncertainty_rad=.01))
        bench._valid_result(valid,timing)
        for field,value in (('value',[99.,0.,0.]),('status','missing'),('captured_at_seconds',.1),
                            ('available_at_seconds',.3),('age_seconds',.1),('age_seconds',False)):
            bad=deepcopy(valid);bad['signals'][bench.SIGNALS[0]][field]=value
            with self.assertRaisesRegex(ValueError,'Nested observation'):bench._valid_result(bad,timing)
        bad=deepcopy(valid);bad['age_seconds']=.1
        with self.assertRaisesRegex(ValueError,'Top-level estimator age'):bench._valid_result(bad,timing)
        missing=missing_result(None,None,None,**timing);missing['signals'][bench.SIGNALS[0]]['value']=[0.,0.,0.]
        with self.assertRaisesRegex(ValueError,'missing/null'):bench._valid_result(missing,timing)
        self.render()
        def replacement(*args,**kwargs):
            result=missing_result(*args,**kwargs);result.pop('signals');return result
        result=self.estimate(estimator=replacement)
        self.assertFalse(result['pipeline_complete']);self.assertEqual(result['estimated'],0)

    def test_cli_pipeline_failures_exit_nonzero_with_receipts_retained(self):
        FakeSession.failures={160}
        with redirect_stdout(io.StringIO()) as output:
            code=bench.main(['render','--output',str(self.root),'--assets','unused'])
        self.assertEqual(code,1)
        self.assertEqual(json.loads(output.getvalue())['scheduled'],20)
        self.assertTrue((self.root/'development/manifest.json').exists())
        with patch('dvidia_training.piper_rgbd_tracker.estimate_pose',missing_result):
            with redirect_stdout(io.StringIO()) as output:
                code=bench.main(['estimate','--output',str(self.root)])
        self.assertEqual(code,1)
        self.assertEqual(json.loads(output.getvalue())['scheduled'],20)
        self.assertTrue((self.root/'development/predictions.json').exists())

    def test_cli_valid_abstentions_exit_zero(self):
        self.render()
        with patch('dvidia_training.piper_rgbd_tracker.estimate_pose',missing_result):
            with redirect_stdout(io.StringIO()):
                code=bench.main(['estimate','--output',str(self.root)])
        self.assertEqual(code,0)

    def test_custom_geom_kept_in_depth_segmentation(self):
        geom0=SimpleNamespace(segid=7);geom1=SimpleNamespace(segid=-1)
        scene=SimpleNamespace(ngeom=1,maxgeom=2,geoms=[geom0,geom1])
        session=SimpleNamespace(renderer=SimpleNamespace(scene=scene),
            mj=SimpleNamespace(mjtGeom=SimpleNamespace(mjGEOM_BOX=6),mjv_initGeom=lambda *args:None))
        self.assertEqual(bench._visual_box(session,[0,0,0],[.02]*3,np.eye(3),[1]*4),8)
        self.assertEqual(scene.ngeom,2);self.assertEqual(geom1.segid,8)


if __name__=='__main__':unittest.main()

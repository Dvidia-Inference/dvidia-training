"""Saved-data audit tests with a fabricated 320-row study, no native imports."""
from contextlib import redirect_stdout
from hashlib import sha256
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np


MODULE = Path(__file__).resolve().parents[1]/'benchmarks/piper-rgbd-tracker-v01/independent_audit.py'
spec = importlib.util.spec_from_file_location('rgbd_independent_audit', MODULE)
audit_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit_module)


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audit_module.canonical(document)+b'\n')


def identity(path):
    body = path.read_bytes(); return dict(bytes=len(body), sha256=sha256(body).hexdigest())


def fixture(root):
    """Entirely authored data double; it does not measure a physical estimator."""
    root.mkdir(); folder = root/'confirmation'; folder.mkdir(); source = root/'source'; source.mkdir()
    (source/'tracker.py').write_text('# frozen audit test double\n')
    sources = {'tracker.py': identity(source/'tracker.py')['sha256']}; runtime = {'test_double': True}
    bindings = {'160': {'test_double': True}, '320': {'test_double': True}}
    groups = [dict(id=f'pose{i:02}', split='development' if i < 8 else 'confirmation',
                   position_m=[0.,0.,.1], yaw_rad=0.) for i in range(24)]
    variants = list(audit_module.VARIANTS); negatives = variants[2:]
    limits = dict(clear_coverage_min=.9, clear_position_p95_m_max=.01,
        clear_yaw_p95_rad_max=.17453292519943295, negative_false_available_max=0)
    protocol = dict(groups=groups, variants=variants, resolutions=[[160,120],[320,240]],
        expected_available=['clear'], challenge_positive=['partial_occlusion'], expected_abstain=negatives,
        provisional_gate=limits)
    write(root/'protocol.json', protocol); protocol_hash = identity(root/'protocol.json')['sha256']
    dev = {}
    for name in ('manifest.json','predictions.json','report.json'):
        write(root/'development'/name, {'test_double': True}); dev[name] = identity(root/'development'/name)
    freeze = dict(protocol_sha256=protocol_hash, development_evidence=dev, sources=sources,
                  runtime=runtime, renderer_bindings=bindings)
    write(root/'freeze.json',freeze)
    schedule=[]; items=[]; predictions=[]; rows=[]
    for width,height in ((160,120),(320,240)):
        calibration = dict(base_frame='robot_base',base_from_camera=[1.,0.,0.,0., 0.,-1.,0.,0., 0.,0.,-1.,1., 0.,0.,0.,1.],
            intrinsics=dict(width=width,height=height,fx=100.,fy=100.,cx=width/2,cy=height/2))
        for group in groups[8:]:
            for variant in variants:
                entry=dict(id=f'{group["id"]}-{width}-{variant}', group_id=group['id'],variant=variant,
                    width=width,height=height,frame_index=len(schedule)); schedule.append(entry)
                path=folder/entry['id'];path.mkdir()
                timing=dict(frame_index=entry['frame_index'],captured_at_seconds=.2,
                    available_at_seconds=.201 if variant=='future' else .2,
                    decision_at_seconds=.301 if variant=='stale' else .2)
                write(path/'input.json',dict(calibration=calibration,timing=timing))
                write(path/'truth.json',dict(position_m=[0.,0.,.1],quaternion_wxyz=[1.,0.,0.,0.],yaw_rad=0.,
                    tilt_rad=.3 if variant in ('tilt15','tilt30') else 0.,present=variant!='absent'))
                (path/'rgb.npy').write_bytes(b'authored RGB hash fixture; not consumed by auditor')
                (path/'rgb.png').write_bytes(b'authored PNG hash fixture')
                if variant in ('clear','occluded'):
                    np.save(path/'depth.npy',np.full((height,width),.838 if variant=='occluded' else .88,dtype=np.float32),allow_pickle=False)
                else: np.save(path/'depth.npy',np.ones((1,1),dtype=np.float32),allow_pickle=False)
                write(path/'started.json',dict(**entry,status='started'))
                item=dict(**entry,render_status='rendered',error=None,files={p.name:identity(p) for p in path.iterdir()},
                          render_wall_seconds=.01,renderer_warnings=[])
                write(path/'receipt.json',item);items.append(dict(**item,receipt_sha256=identity(path/'receipt.json')['sha256']))
                is_available=variant=='clear';age=None if variant=='future' else timing['decision_at_seconds']-.2
                values=([.001,0.,.1],[1.,0.,0.,0.],.002,.01) if is_available else (None,)*4
                result=dict(status='available' if is_available else 'missing',reason='authored_test_double',
                    **timing,age_seconds=age,base_frame='robot_base',depth_accuracy_verified=False,signals={})
                for name,value in zip(audit_module.SIGNALS,values):
                    result[name]=value
                    result['signals'][name]=(dict(status='available',value=value if isinstance(value,list) else [value],
                        captured_at_seconds=.2,available_at_seconds=.2,age_seconds=0.) if is_available else
                        dict(status='missing',value=None,captured_at_seconds=None,available_at_seconds=None,age_seconds=None))
                pred=dict(id=entry['id'],estimate_status='complete',result=result,error=None,wall_seconds=.001,cpu_seconds=.0008,
                    pipeline_wall_seconds=.002,inputs={name:item['files'][name] for name in audit_module.INPUTS})
                estimate_path=folder/'estimates'/(entry['id']+'.json');write(estimate_path,pred)
                predictions.append(dict(**pred,receipt_sha256=identity(estimate_path)['sha256']))
                rows.append(dict(**items[-1],estimate_status='complete',available=is_available,
                    position_error_m=.001 if is_available else None,yaw_error_rad=0. if is_available else None,
                    position_heuristic_scale_m=.002 if is_available else None,orientation_heuristic_scale_rad=.01 if is_available else None,
                    position_error_within_heuristic_scale=True if is_available else None,
                    yaw_error_within_heuristic_scale=True if is_available else None,wall_seconds=.001,cpu_seconds=.0008))
    write(folder/'scheduled.json',dict(records=schedule,protocol_sha256=protocol_hash,split='confirmation'))
    manifest=dict(records=items,source_hashes=sources,source_stable=True,runtime=runtime,renderer_bindings=bindings,
        protocol_sha256=protocol_hash,scheduled_sha256=identity(folder/'scheduled.json')['sha256'],split='confirmation',session_errors=[])
    write(folder/'manifest.json',manifest)
    write(folder/'predictions.json',dict(predictions=predictions,source_hashes=sources,runtime=runtime,
        manifest_sha256=identity(folder/'manifest.json')['sha256'],protocol_sha256=protocol_hash,split='confirmation',
        freeze_sha256=identity(root/'freeze.json')['sha256']))
    summaries=[];gates=[]
    for width in (160,320):
        for variant in variants:
            available=variant=='clear'
            summaries.append(dict(width=width,variant=variant,scheduled=16,available=16 if available else 0,pipeline_failures=0,
                position_evaluated=16 if available else 0,yaw_evaluated=16 if available else 0,
                position_p50_m=.001 if available else None,position_p95_m=.001 if available else None,
                yaw_p95_rad=0. if available else None,wall_p50_s=.001,wall_p95_s=.001,cpu_p95_s=.0008,
                heuristic_comparisons=16 if available else 0,yaw_heuristic_comparisons=16 if available else 0,
                position_error_within_heuristic_scale=16 if available else 0,yaw_error_within_heuristic_scale=16 if available else 0))
        checks={name:True for name in ('all_scheduled_fixtures_complete','clear_pose_metrics_complete','clear_coverage',
            'clear_position','clear_yaw','no_negative_false_available')}
        gates.append(dict(width=width,passed=True,checks=checks,clear_scheduled=16,clear_coverage=1.,negative_scheduled=128,negative_false_available=0))
    write(folder/'report.json',dict(rows=rows,summary=summaries,gates=gates,scheduled=320,
        predictions_sha256=identity(folder/'predictions.json')['sha256'],manifest_sha256=identity(folder/'manifest.json')['sha256'],
        physical_robot_ready=False,depth_accuracy_verified=False))
    return source


class IndependentAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'study';self.source=fixture(self.root)

    def audit(self): return audit_module.audit_study(self.root,self.source)

    def test_valid_full_denominator_depth_and_metrics(self):
        result=self.audit()
        self.assertTrue(result['passed'],result['mismatches'])
        self.assertEqual(result['processed_confirmation_ids'],320)
        self.assertEqual(len(result['occluder_depth_checks']),32)
        self.assertTrue(all(row['passed'] for row in result['occluder_depth_checks']))
        self.assertFalse(result['simulator_executed']);self.assertFalse(result['estimator_executed'])
        self.assertTrue(result['source_bytes_verified'])

    def test_missing_artifact_fails_closed(self):
        (self.root/'confirmation/pose08-160-clear/rgb.npy').unlink()
        result=self.audit();self.assertFalse(result['passed'])
        self.assertTrue(any('Missing/nonregular' in x for x in result['mismatches']))

    def test_changed_raw_or_frozen_source_bytes_are_detected(self):
        (self.source/'tracker.py').write_text('# altered\n')
        result=self.audit();self.assertFalse(result['passed'])
        self.assertIn('source.tracker.py',result['mismatches'])

    def test_wrong_summary_metric_fails_even_with_untouched_raw_data(self):
        path=self.root/'confirmation/report.json';data=json.loads(path.read_text())
        data['summary'][0]['position_p95_m']=0.;write(path,data)
        result=self.audit();self.assertFalse(result['passed'])
        self.assertTrue(any('position_p95_m' in x for x in result['mismatches']))

    def test_missing_prediction_denominator_is_refused(self):
        path=self.root/'confirmation/predictions.json';data=json.loads(path.read_text())
        data['predictions'].pop();write(path,data)
        result=self.audit();self.assertFalse(result['passed'])
        self.assertTrue(any('missing/duplicate/unknown' in x for x in result['mismatches']))

    def test_depth_card_check_uses_saved_depth_not_presence_label(self):
        path=self.root/'confirmation/pose08-160-occluded/depth.npy'
        np.save(path,np.full((120,160),.88,dtype=np.float32),allow_pickle=False)
        # Call the independent geometric check directly: a forged receipt does
        # not make a card that is RGB-only appear in the saved depth image.
        audit=audit_module.Audit(self.root)
        row=audit_module._occluder_check(audit,self.root/'confirmation','pose08',160)
        self.assertFalse(row['passed']);self.assertEqual(len(audit.errors),1)

    def test_nested_signal_inconsistency_is_caught(self):
        path=self.root/'confirmation/estimates/pose08-160-clear.json'
        pred=json.loads(path.read_text());result=pred['result'];result['signals'][audit_module.SIGNALS[0]]['value'][0]=1.
        audit=audit_module.Audit(self.root)
        timing={k:result[k] for k in ('frame_index','captured_at_seconds','available_at_seconds','decision_at_seconds')}
        audit_module._signal_contract(audit,result,timing,'test')
        self.assertTrue(any('signals.object_position' in x for x in audit.errors))

    def test_cli_writes_fresh_result_hash_and_refuses_overwrite(self):
        path=Path(self.temp.name)/'audit.json'
        with redirect_stdout(io.StringIO()) as output:
            code=audit_module.main(['--study',str(self.root),'--source-root',str(self.source),'--output',str(path)])
        self.assertEqual(code,0);response=json.loads(output.getvalue())
        self.assertEqual(response['audit_sha256'],identity(path)['sha256'])
        with self.assertRaises(FileExistsError):
            audit_module.main(['--study',str(self.root),'--output',str(path)])


if __name__=='__main__':unittest.main()

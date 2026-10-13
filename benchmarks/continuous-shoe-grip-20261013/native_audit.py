"""Audit the frozen cohort from its source checkout; compile models without stepping physics.

Install the checkout's ``.[arm]`` dependencies and provision its pinned PiPER
assets first. Paths supplied on the command line are relative to the current
working directory. The imported package must be the frozen source checkout,
for example by setting PYTHONPATH=src when running from that checkout.
"""
from collections import Counter
from hashlib import sha256
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import numpy as np
from dvidia_training import grip_protocol, reference_sim, tool_reach

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input',type=Path,default=Path('runs/grip-v02/frozen/lab'),
                    help='Frozen heldout export directory (default: %(default)s)')
parser.add_argument('--assets',type=Path,default=Path('runs/robot-assets'),
                    help='Pinned PiPER asset directory from dvidia-robot-sim fetch (default: %(default)s)')
parser.add_argument('--output',type=Path,default=Path('runs/grip-v02-publication/qa'),
                    help='Audit output directory (default: %(default)s)')
args=parser.parse_args()
ROOT=args.input
OUT=args.output
ASSETS=args.assets
SOURCE=Path(grip_protocol.__file__).resolve().parent
COMMIT='af8933994b4626e28a0554abc782ae9d22cff06c'
def read(path):return json.loads(path.read_text())
def check(ok,message):
    if not ok:raise AssertionError(message)
def digest(path):return sha256(path.read_bytes()).hexdigest()
def close(a,b,tol=1e-9):return bool(np.allclose(a,b,rtol=0,atol=tol))
def stats(values):
    return {'count':len(values),'min':min(values),'median':float(np.median(values)),
            'max':max(values)} if values else {'count':0,'min':None,'median':None,'max':None}

report=read(ROOT/'benchmark.json');plan=read(ROOT/'protocol.json');integrity=read(ROOT/'integrity.json')
check(report['source']['commit']==plan['source_commit']==COMMIT,'Wrong frozen source')
check(report['split']==plan['split']=='heldout','Wrong cohort split')
check(digest(ROOT/'protocol.json')==report['source']['protocol_sha256'],'Protocol binding mismatch')
check(plan['fixture_sha256']==grip_protocol.FIXTURE_SHA256==digest(grip_protocol.FIXTURE),'Fresh seal mismatch')
check(plan['attempts']==grip_protocol.protocol('heldout')['attempts'],'Cohort attempt inventory changed')
source_checks={}
for name,expected in plan['implementation_sha256'].items():
    committed=subprocess.check_output(['git','-C',str(SOURCE),'show',COMMIT+':src/dvidia_training/'+name])
    check(sha256(committed).hexdigest()==expected,'Committed implementation mismatch: '+name)
    check(digest(SOURCE/name)==expected,'Working implementation mismatch: '+name)
    source_checks[name]=expected
files={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and p.name!='integrity.json'}
check(files==set(integrity['files']),'Integrity inventory does not cover every export')
for name,expected in integrity['files'].items():
    raw=(ROOT/name).read_bytes()
    check(len(raw)==expected['bytes'] and sha256(raw).hexdigest()==expected['sha256'],'File checksum mismatch: '+name)

decoded={}
def binary(meta):
    if meta['href'] not in decoded:
        raw=(ROOT/meta['href']).read_bytes();check(len(raw)==meta['compressed_bytes'],'Compressed size mismatch')
        value=gzip.decompress(raw)
        check(len(value)==meta['bytes'] and sha256(value).hexdigest()==meta['sha256'],'Binary content checksum mismatch')
        decoded[meta['href']]=value
    return decoded[meta['href']]

trials={};models={};audited=[];loss_latencies=[];recovery_packets=[];request_reason=Counter();frame_count=0;trace_count=0
index={r['id']:r for r in report['attempts']}
check(set(index)=={a['id'] for a in plan['attempts']} and len(index)==20,'Attempt set mismatch')
for attempt in plan['attempts']:
    row=read(ROOT/'trials'/str(attempt['id']+'.json'));trials[row['id']]=row
    check(row['case']==attempt['case'] and row['policy']==attempt['policy'],'Trial case or policy mismatch')
    check({k:v for k,v in row.items() if k!='native_trace'}==index[row['id']],'Index differs from full receipt')
    # Independently recompute all retained-grip and event predicates from complete native traces.
    score=grip_protocol.audit_result({**row,'success':row['controller_reported_success']})
    check(all(row['independent_score'][k]==v for k,v in score.items()),'Stored independent score differs')
    check(row['success']==score['task_success'] and row['valid'],'Invalid attempt or false success')
    trace=np.asarray(row['native_trace']);trace_count+=len(trace);times=np.round(trace[:,0],6)
    bytime={round(float(t),6):i for i,t in enumerate(times)}
    model=read(ROOT/row['replay']['model']);motion=read(ROOT/row['replay']['motion']);models[row['id']]=model
    check(model['source']==motion['source'],'Model/motion source mismatch')
    check(model['source']['compiled_model_sha256']==row['independent_score']['compiled_model_sha256'],'Model hash binding mismatch')
    binary(model['geometry_binary']);raw=binary(motion['motion_binary'])
    poses=np.frombuffer(raw,dtype='<f4').reshape(motion['frame_count'],motion['body_count'],7)
    check(np.isfinite(poses).all(),'Nonfinite native replay pose')
    check(close(np.linalg.norm(poses[:,:,3:],axis=2),1.,.001),'Invalid replay quaternions')
    check(len(motion['rows'])==len(motion['times'])==len(poses),'Frame counts mismatch')
    shoe=next(i for i,b in enumerate(model['bodies']) if b['name']=='shoe_trainer_left')
    frame_bytime={round(float(f['t']),6):f for f in motion['rows']}
    for i,frame in enumerate(motion['rows']):
        check(close(frame['t'],motion['times'][i]),'Motion timestamp mismatch')
        native=trace[bytime[round(float(frame['t']),6)]]
        check(close(frame['shoe_position'],native[5:8]),'Replay row differs from native shoe position')
        check(close(poses[i,shoe,:3],native[5:8],1e-7),'Replay binary differs from native shoe position')
        check(close(frame['finger_normal_forces_n'],native[3:5]),'Replay finger forces differ')
        check(close(frame['world_shoe_force_n'],native[8]),'Replay world force differs')
        check(close(frame['qpos_rad_m'][8:11],native[5:8]),'Shoe free-joint coordinates differ')
    frame_count+=len(motion['rows'])
    events=row['events'];signals=[e for e in events if e['type']=='grip_loss_detected']
    for event in signals:
        if event['reason']!='bilateral_contact_lost':continue
        onset=event['first_native_loss_s'];end=event['t'];start_index=bytime[onset];end_index=bytime[end]
        # Verify the latest uninterrupted native bilateral-contact gap used by the event.
        bad=np.min(trace[start_index:end_index+1,3:5],axis=1)<=.01
        check(bool(np.all(bad)),'Claimed native contact-loss segment contains contact recovery')
        check(start_index==0 or min(trace[start_index-1,3:5])>.01,'Claimed onset is not latest native loss transition')
        sampled=[round(end-d,6) for d in (.04,.02,0.)]
        check(all(min(trace[bytime[t],3:5])<=.01 for t in sampled),'Alarm lacks three bad decision samples')
        requests=[e for e in events if e['type']=='grip_recovery_started' and e['t']==end]
        packets=[e for e in events if e['type']=='recovery_control_issued' and e['requested_at_s']==end]
        check(len(requests)==len(packets)==1,'Contact alarm does not have one request and native packet')
        packet=packets[0]
        loss_latencies.append({'attempt':row['id'],'reason':event['reason'],'latest_native_loss_s':onset,
            'detector_s':end,'request_s':requests[0]['t'],'packet_s':packet['t'],
            'native_loss_to_detector_ms':round((end-onset)*1000,6),
            'native_loss_to_packet_ms':round((packet['t']-onset)*1000,6),
            'detector_to_packet_ms':round((packet['t']-end)*1000,6),
            'trace_verified':True,'three_failed_decision_samples_verified':True})
    for event in (e for e in events if e['type']=='grip_recovery_started'):
        request_reason[event['reason']]+=1
        packet=next(e for e in events if e['type']=='recovery_control_issued' and e['requested_at_s']==event['t'])
        previous=frame_bytime[packet['t']];following=frame_bytime[round(packet['t']+.02,6)]
        ctrl=np.asarray(packet['issued_ctrl']);desired=np.asarray(packet['requested_joint_target']);caps=np.asarray([.003]*6+[.0001])
        check(close(ctrl,np.asarray(previous['issued_ctrl'])+np.clip(desired-previous['issued_ctrl'],-caps,caps)),'Packet differs from one native bounded-control update')
        check(close(following['requested_joint_target'],desired),'Next recorded target differs from issued request')
        expected=ctrl.copy()
        for _ in range(9):expected+=np.clip(desired-expected,-caps,caps)
        check(close(following['issued_ctrl'],expected),'Following recorded control differs from native ramp')
        open_event=next((e for e in events if e['type']=='phase' and e.get('name')=='open_on_support' and event['t']<e['t']<=event['t']+1.),None)
        supported=bool(open_event and trace[bytime[open_event['t']],8]>.01)
        check(supported,'Recovery opened jaws without recorded world support')
        recovery_packets.append({'attempt':row['id'],'reason':event['reason'],'request_s':event['t'],'packet_s':packet['t'],
            'recorded_controls_match_native_ramp':True,'open_on_support_s':open_event['t'],
            'world_support_at_open_n':float(trace[bytime[open_event['t']],8]),
            'strict_recovered_completion':bool(score['task_success'])})
    audited.append({'id':row['id'],'valid':True,'strict_success':score['task_success'],'trace_ticks':len(trace),
        'final_strict_ticks':score['strict_passing_ticks'],'appropriate_deferral':score['appropriate_deferral'],
        'strict_condition_losses':score['loss_episodes'],'controller_loss_events':len(signals),
        'compiled_model_sha256':model['source']['compiled_model_sha256']})

pairs=[];end_check_intervals=[];strict_gaps=[]
for case_id in dict.fromkeys(a['case']['id'] for a in plan['attempts']):
    a=trials[case_id+'-baseline'];b=trials[case_id+'-recovery'];ma=models[a['id']];mb=models[b['id']]
    check(ma==mb,'Paired exported models differ')
    # This constructs the frozen mechanics only. No mj_step or policy execution occurs.
    compiled=tool_reach.build_model(ASSETS,a['case'],'recovery');model_hash=reference_sim.model_digest(compiled)
    check(model_hash==a['independent_score']['compiled_model_sha256']==b['independent_score']['compiled_model_sha256'],'Recompiled frozen model differs')
    av=np.asarray(a['native_trace']);bv=np.asarray(b['native_trace']);n=min(len(av),len(bv))
    different=np.flatnonzero(np.max(abs(av[:n,:9]-bv[:n,:9]),axis=1)>1e-9)
    first_divergence=float(av[different[0],0]) if len(different) else None
    requests=[e for e in b['events'] if e['type']=='grip_recovery_started'];first_request=requests[0]['t'] if requests else None
    check(first_divergence is None or first_request is not None and first_divergence>first_request,'Paired dynamics differ before intervention')
    pairs.append({'case_id':case_id,'identical_exported_model':True,'compiled_model_verified':True,
                  'baseline_success':a['success'],'recovery_success':b['success'],
                  'first_trace_divergence_s':first_divergence,'first_recovery_request_s':first_request,
                  'shared_preintervention_native_trace':True})
    first_loss=next((e for e in b['events'] if e['type']=='grip_loss_detected'),None)
    check_event=next((e for e in a['events'] if e['type']=='lift_check'),None)
    if first_loss and first_loss['reason']=='bilateral_contact_lost' and check_event and not a['success']:
        onset=first_loss['first_native_loss_s']
        end_check_intervals.append({'case_id':case_id,'shared_preintervention_loss_origin_s':onset,
            'v02_detector_s':first_loss['t'],'v01_scheduled_terminal_check_s':check_event['t'],
            'loss_to_v02_detector_ms':round((first_loss['t']-onset)*1000,6),
            'loss_to_v01_scheduled_check_ms':round((check_event['t']-onset)*1000,6),
            'not_baseline_detector_or_physical_response_latency':True})
    for loss in b['independent_score']['loss_episodes']:
        end=loss['requalified_s'] or bv[-1,0];window=bv[(bv[:,0]>=loss['onset_s'])&(bv[:,0]<=end)]
        channels={'height':window[:,7]<=.035,'left_finger':window[:,3]<=.01,'right_finger':window[:,4]<=.01,
                  'world_support':window[:,8]>=.01,'rake_support':window[:,1]>=.01,
                  'other_robot_support':window[:,2]-window[:,1]-window[:,3]-window[:,4]>=.01}
        strict_gaps.append({'case_id':case_id,'onset_s':loss['onset_s'],'requalified_s':loss['requalified_s'],
            'failed_ticks_by_condition':{k:int(sum(v)) for k,v in channels.items()},
            'detected':loss['detected_s'] is not None,'scope':'Scoring-condition gap, not automatically a bilateral-slip event or missed safety alarm.'})

summary={'schema':'dvidia.continuous-shoe-grip-native-audit/v1','source_commit':COMMIT,
    'input_root':str(ROOT),'protocol_sha256':digest(ROOT/'protocol.json'),'integrity_sha256':digest(ROOT/'integrity.json'),
    'fresh_fixture_sha256':grip_protocol.FIXTURE_SHA256,'fresh_simulations_executed_during_audit':0,
    'compiled_model_checks_without_physics_steps':len(pairs),'implementation_files_verified':source_checks,
    'export_files_verified':len(files),'binary_payloads_verified':len(decoded),'native_trace_ticks_verified':trace_count,
    'replay_frames_verified':frame_count,'valid_attempts':len(audited),'total_attempts':20,
    'strict_successes':{p:sum(a['strict_success'] for a in audited if a['id'].endswith('-'+p)) for p in ('baseline','recovery')},
    'successful_cases':{p:[a['id'].removesuffix('-'+p) for a in audited if a['id'].endswith('-'+p) and a['strict_success']] for p in ('baseline','recovery')},
    'paired_gains':sum(not p['baseline_success'] and p['recovery_success'] for p in pairs),
    'paired_losses':sum(p['baseline_success'] and not p['recovery_success'] for p in pairs),
    'appropriate_deferrals':{p:sum(a['appropriate_deferral'] for a in audited if a['id'].endswith('-'+p)) for p in ('baseline','recovery')},
    'bilateral_loss_to_detector_ms':stats([r['native_loss_to_detector_ms'] for r in loss_latencies]),
    'bilateral_loss_to_packet_ms':stats([r['native_loss_to_packet_ms'] for r in loss_latencies]),
    'detector_to_packet_ms':stats([r['detector_to_packet_ms'] for r in loss_latencies]),
    'recovery_requests_by_reason':dict(request_reason),'issued_recovery_packets':len(recovery_packets),
    'completed_after_recovery':sum(r['strict_recovered_completion'] for r in recovery_packets),
    'latency_scope':'Nine contact-loss observations use the latest uninterrupted bilateral-contact gap before a sampled alarm. Three failed 20 ms decision samples can include unobserved native recoveries between samples. Same-timestamp command issue means a pre-step control packet, not zero physical response time. No camera/transport/hardware latency is modeled. Acquisition failures have no prior acquired-loss origin and are excluded.',
    'baseline_timing_scope':'The four paired intervals share identical native trajectories through the v0.2 intervention. Baseline intervals end at a scheduled terminal task check, not a contact-loss detector. These intervals are not a physical stopping, recovery-speed or sensor-latency comparison.',
    'comparison_scope':'Ten deterministic declared parameter cases, one execution per policy. The same two cases pass both policies; no observed completion improvement. Rigid synthetic shoes and privileged contact/state observations; no hardware safety or learned-policy evidence.',
    'attempts':audited,'pairs':pairs,'contact_loss_events':loss_latencies,'recovery_packets':recovery_packets,
    'strict_condition_gaps':strict_gaps,'paired_scheduled_check_intervals':end_check_intervals}
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'native-audit.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:summary[k] for k in ('valid_attempts','strict_successes','paired_gains','paired_losses','bilateral_loss_to_detector_ms','issued_recovery_packets','native_trace_ticks_verified','replay_frames_verified')}))

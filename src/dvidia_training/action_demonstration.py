"""Synchronized, data-only PiPER action demonstrations; no robot/simulator calls.

Reuses episode_tape's JSON/time-evidence conventions without changing its fixed
authored-arm schema. PiPER's seventh actuator is a single-finger slide target,
not the older tape/transition model's total jaw gap. Validation is consistency,
not proof of camera accuracy, correction quality, physical safety or skill.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import math
import os
from pathlib import Path,PurePosixPath
import stat
import tempfile

from .episode_tape import canonical,digest,_decode,_keys,_number,_integer,_identifier,_sha,_vector
from .embodiment_calibration import _errors

SCHEMA='dvidia.action-demonstration/v1'
MAX_BYTES=8*1024*1024
MAX_STEPS=1000
MAX_FRAME_BYTES=16*1024*1024
MAX_FRAME_TOTAL=512*1024*1024
LOCK=Path(__file__).with_name('robot_profiles.lock.json')
DT=.02
MAX_AGE=.10
CLOCK_TOL=1e-9
SIGNALS={
    'joint_position_rad':dict(size=6,units='rad',frame='joint_coordinates',origin='simulated_encoder'),
    'joint_velocity_rad_s':dict(size=6,units='rad/s',frame='joint_coordinates',origin='simulated_encoder'),
    'gripper_joint_position_m':dict(size=2,units='m',frame='joint7_joint8_signed_coordinates',origin='simulated_encoder'),
    'gripper_joint_velocity_m_s':dict(size=2,units='m/s',frame='joint7_joint8_signed_coordinates',origin='simulated_encoder'),
    'object_position_estimate_m':dict(size=3,units='m',frame='robot_base',origin='declared_rgbd_tracker'),
    'object_quaternion_estimate_wxyz':dict(size=4,units='unit_quaternion_wxyz',frame='robot_base',origin='declared_rgbd_tracker'),
    'object_position_uncertainty_m':dict(size=1,units='m',frame='robot_base',origin='declared_rgbd_tracker'),
    'object_orientation_uncertainty_rad':dict(size=1,units='rad',frame='robot_base',origin='declared_rgbd_tracker'),
    'tcp_position_estimate_m':dict(size=3,units='m',frame='robot_base',origin='declared_encoder_fk'),
    'tcp_quaternion_estimate_wxyz':dict(size=4,units='unit_quaternion_wxyz',frame='robot_base',origin='declared_encoder_fk'),
}
HEADER={'schema','scope','physical_robot_ready','provenance','binding','profile','limits','calibration','clock','signal_contract','goal_position_m'}
SHA_FIELDS={'profile_lock_sha256','profile_record_sha256','model_sha256','scene_sha256','configuration_sha256',
            'limits_sha256','calibration_sha256','reference_policy_sha256'}


def _bounded(value,label,low=-1e6,high=1e6):
    _number(value,label)
    if not low<=value<=high:raise ValueError(label+': value exceeds contract bounds.')
    return value


def _vec(value,size,label,low=-1e6,high=1e6):
    _vector(value,size,label)
    for item in value:_bounded(item,label,low,high)
    return value


def _time(value,label,low=0,high=20):
    # Preserve native timestamps, including sub-nanosecond accumulated-step
    # roundoff. Do not quantize, clamp or repair the original evidence.
    return _bounded(value,label,low-CLOCK_TOL,high+CLOCK_TOL)


def _relative(value):
    if type(value) is not str:raise ValueError('Frame references require a bounded relative path.')
    p=PurePosixPath(value)
    if (not value or len(value)>240 or str(p)!=value or p.is_absolute() or '\\' in value or ':' in value
            or any(part in ('','..','.') or part.startswith('.') for part in p.parts)
            or any(ord(c)<32 for c in value)):
        raise ValueError('Frame references require a bounded relative path.')
    return p


def _json(document):
    if isinstance(document,(bytes,str)) and len(document)>MAX_BYTES:raise ValueError('Demonstration exceeds 8 MiB.')
    result=_decode(document)
    if len(canonical(result))>MAX_BYTES:raise ValueError('Demonstration exceeds 8 MiB.')
    return result


def profile_contract():
    """Read the existing packaged mechanical order; no assets or native imports."""
    body=LOCK.read_bytes();lock=json.loads(body);profile=next(p for p in lock['profiles'] if p['id']=='piper')
    contract={key:profile[key] for key in ('id','arm_joint_names','gripper_joint_names','actuator_names','native_command_units','native_command_range','gripper_command','tool_frame_body')}
    return contract,sha256(body).hexdigest(),digest(profile)


def _header(header):
    _keys(header,HEADER,'header')
    if header['schema']!=SCHEMA or header['scope']!='simulation-only' or header['physical_robot_ready'] is not False:
        raise ValueError('Only the new simulation-only PiPER demonstration contract is supported.')
    source=header['provenance']
    _keys(source,{'episode_id','source_recording_id','session_id','source_group','kind','producer_revision','replay_of_sha256'},'provenance')
    if source['kind']=='video_only':raise ValueError('Video-only evidence cannot supply synchronized actuator targets.')
    if source['kind'] not in ('authored_fixture','simulation_teleoperation','simulation_authored_controller'):
        raise ValueError('Unsupported source; hardware/device clocks require a separate capture contract.')
    for name in ('episode_id','source_recording_id','session_id','source_group','producer_revision'):_identifier(source[name],name)
    if source['replay_of_sha256'] is not None:_sha(source['replay_of_sha256'],'replay_of_sha256')
    expected,lock_sha,record_sha=profile_contract()
    if canonical(header['profile'])!=canonical(expected):raise ValueError('PiPER ordered native channels or finger-slide units changed.')
    binding=header['binding'];_keys(binding,SHA_FIELDS,'binding')
    for name,value in binding.items():_sha(value,name)
    if binding['profile_lock_sha256']!=lock_sha or binding['profile_record_sha256']!=record_sha:
        raise ValueError('Packaged profile identity differs from the declared binding.')
    limits=header['limits'];_keys(limits,{'joint_limits_rad','gripper_position_limits_m','joint_target_rate_rad_s','jaw_target_rate_m_s'},'limits')
    if type(limits['joint_limits_rad']) is not list or len(limits['joint_limits_rad'])!=6:raise ValueError('Six ordered native joint ranges are required.')
    for pair,native in zip(limits['joint_limits_rad'],expected['native_command_range'][:6]):
        _vec(pair,2,'joint limits',-10,10)
        if not pair[0]<pair[1]:raise ValueError('Joint limit lower endpoint must precede upper endpoint.')
        if pair[0]<native[0] or pair[1]>native[1]:raise ValueError('Declared joint limits may tighten, never widen, the bound PiPER native ranges.')
    _vec(limits['gripper_position_limits_m'],2,'finger-slide limits',0,.035)
    if limits['gripper_position_limits_m']!=[0.,.035]:raise ValueError('PiPER command is a 0–35 mm finger-slide coordinate, not total gap.')
    _bounded(limits['joint_target_rate_rad_s'],'joint reference rate',.01,10)
    _bounded(limits['jaw_target_rate_m_s'],'jaw reference rate',.001,1.)
    if digest(limits)!=binding['limits_sha256']:raise ValueError('Actuator-limit declaration hash changed.')
    calibration=header['calibration']
    _keys(calibration,{'id','base_frame','tool_frame','camera_frame','intrinsics','base_from_camera','depth_units','valid_from_seconds','valid_until_seconds'},'calibration')
    _identifier(calibration['id'],'calibration id')
    if (calibration['base_frame']!='robot_base' or calibration['tool_frame']!='link6'
            or calibration['camera_frame']!='rgbd_optical' or calibration['depth_units']!='m'):
        raise ValueError('Explicit robot-base/link6/camera frames and metre depth are required.')
    intrinsics=calibration['intrinsics'];_keys(intrinsics,{'width','height','fx','fy','cx','cy'},'intrinsics')
    for name in ('width','height'):
        _integer(intrinsics[name],name,minimum=1)
        if intrinsics[name]>8192:raise ValueError('Camera dimensions exceed the capture contract.')
    for name in ('fx','fy'):_bounded(intrinsics[name],name,1e-6,1e6)
    _bounded(intrinsics['cx'],'cx',0,intrinsics['width']);_bounded(intrinsics['cy'],'cy',0,intrinsics['height'])
    transform=_vec(calibration['base_from_camera'],16,'base_from_camera',-100,100)
    if transform[12:]!=[0,0,0,1]:raise ValueError('Camera extrinsics require a homogeneous transform.')
    rotation=[transform[0:3],transform[4:7],transform[8:11]]
    for i in range(3):
        for j in range(3):
            if not math.isclose(sum(a*b for a,b in zip(rotation[i],rotation[j])),int(i==j),abs_tol=1e-6,rel_tol=0):
                raise ValueError('Camera rotation must be orthonormal.')
    a,b,c=rotation
    determinant=a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
    if not math.isclose(determinant,1.,abs_tol=1e-6,rel_tol=0):raise ValueError('Camera rotation cannot reflect axes.')
    start=_bounded(calibration['valid_from_seconds'],'calibration start',0,20)
    end=_bounded(calibration['valid_until_seconds'],'calibration end',0,20)
    if start!=0 or end<=start:raise ValueError('Calibration must cover the declared zero-time recording origin.')
    if digest(calibration)!=binding['calibration_sha256']:raise ValueError('Camera calibration hash changed.')
    if canonical(header['clock'])!=canonical(dict(kind='shared_simulation_elapsed_seconds',origin='producer_zero',control_dt_seconds=DT,maximum_sensor_age_seconds=MAX_AGE)):
        raise ValueError('Shared simulation clock is required; device clocks are not automatically aligned.')
    if canonical(header['signal_contract'])!=canonical(SIGNALS):raise ValueError('Signal dimensions, frames, units or origins changed.')
    _vec(header['goal_position_m'],3,'goal',-10,10)


def _availability(sample,timestamp,label):
    status=sample['status']
    if status=='missing':
        if any(sample[key] is not None for key in ('captured_at_seconds','available_at_seconds','age_seconds','value')):
            raise ValueError('Missing readings must retain null values and times, never synthetic zeros.')
        return None
    if status not in ('available','stale'):raise ValueError('Unknown reading status.')
    captured=_time(sample['captured_at_seconds'],label+' captured')
    available=_time(sample['available_at_seconds'],label+' available')
    age=_time(sample['age_seconds'],label+' age')
    if (captured>available+1e-9 or available>timestamp+1e-9
            or not math.isclose(timestamp-captured,age,abs_tol=1e-9,rel_tol=0)):
        raise ValueError('Capture and arrival times must causally precede the observation.')
    if (status=='available')!=(age<=MAX_AGE+1e-9):raise ValueError('Freshness flag disagrees with acquisition age.')
    return captured


def _snapshot(snapshot,calibration,previous=None):
    _keys(snapshot,{'timestamp_seconds','signals','camera','vision_frame_index'},'snapshot')
    t=_time(snapshot['timestamp_seconds'],'snapshot time')
    if not calibration['valid_from_seconds']-CLOCK_TOL<=t<=calibration['valid_until_seconds']+CLOCK_TOL:
        raise ValueError('Observation falls outside the bound calibration validity interval.')
    _keys(snapshot['signals'],SIGNALS,'signals')
    for name,contract in SIGNALS.items():
        sample=snapshot['signals'][name]
        _keys(sample,{'status','captured_at_seconds','available_at_seconds','age_seconds','value'},name)
        captured=_availability(sample,t,name)
        if captured is not None:
            value=_vec(sample['value'],contract['size'],name)
            if 'uncertainty' in name and value[0]<0:raise ValueError('Declared estimate uncertainty cannot be negative.')
            if contract['units']=='unit_quaternion_wxyz' and not math.isclose(math.hypot(*value),1.,abs_tol=1e-6,rel_tol=0):
                raise ValueError('Observed quaternion estimate must be normalized.')
            if previous is not None:
                old_sample=previous['signals'][name];old=old_sample['captured_at_seconds']
                if old is not None and captured<old-1e-9:raise ValueError('Sensor acquisition timestamps cannot move backwards.')
                if old==captured and (sample['value']!=old_sample['value'] or sample['available_at_seconds']!=old_sample['available_at_seconds']):
                    raise ValueError('A reused acquisition timestamp must retain its value and arrival time.')
    camera=snapshot['camera']
    _keys(camera,{'status','captured_at_seconds','available_at_seconds','age_seconds','value'},'camera')
    captured=_availability(camera,t,'camera')
    if captured is not None:
        frame=camera['value'];_keys(frame,{'index','exposure_start_seconds','exposure_end_seconds','rgb','depth'},'RGB-D frame')
        _integer(frame['index'],'frame index')
        if frame['index']>10**9:raise ValueError('Camera frame index exceeds the capture contract.')
        exposure_start=_time(frame['exposure_start_seconds'],'exposure start')
        exposure_end=_time(frame['exposure_end_seconds'],'exposure end')
        if exposure_start>exposure_end or exposure_end!=captured:raise ValueError('Frame capture must equal exposure end.')
        for name in ('rgb','depth'):
            artifact=frame[name];_keys(artifact,{'path','bytes','sha256'},'frame artifact')
            _relative(artifact['path']);_sha(artifact['sha256'],'frame SHA256')
            _integer(artifact['bytes'],'frame bytes',minimum=1)
            if artifact['bytes']>MAX_FRAME_BYTES:raise ValueError('Frame artifact exceeds 16 MiB.')
        if previous is not None and previous['camera']['value'] is not None:
            old=previous['camera']['value']
            if frame['index']<old['index'] or captured<previous['camera']['captured_at_seconds']-1e-9:
                raise ValueError('Camera indices or capture timestamps cannot move backwards.')
            if frame['index']==old['index'] and frame!=old:
                raise ValueError('A reused camera frame index must retain exact exposure and artifact identities.')
            if frame['index']==old['index'] and camera['available_at_seconds']!=previous['camera']['available_at_seconds']:
                raise ValueError('A reused frame retains its original arrival time.')
            if frame['index']>old['index'] and captured<=previous['camera']['captured_at_seconds']:
                raise ValueError('A new frame index requires a later exposure end.')
        if snapshot['vision_frame_index']!=frame['index'] or type(snapshot['vision_frame_index']) is not int:
            raise ValueError('Tracker estimates must bind the declared RGB-D frame index.')
        for name,contract in SIGNALS.items():
            sample=snapshot['signals'][name]
            if contract['origin']=='declared_rgbd_tracker' and sample['status']!='missing':
                if sample['captured_at_seconds']!=captured or sample['available_at_seconds']<camera['available_at_seconds']:
                    raise ValueError('Tracker acquisition and availability must follow its bound camera frame.')
    elif snapshot['vision_frame_index'] is not None:
        raise ValueError('A missing camera must have a null tracker frame binding.')
    return t


def _acquisition_order(snapshot,known):
    """Retain last known identities across missing gaps, without filling data."""
    result=dict(known)
    samples=dict(snapshot['signals'],camera=snapshot['camera'])
    for name,sample in samples.items():
        if sample['status']=='missing':continue
        old=known.get(name)
        if old is not None:
            captured=sample['captured_at_seconds'];previous=old['captured_at_seconds']
            if captured<previous-CLOCK_TOL:raise ValueError('Sensor acquisition order cannot move backwards, including across missing gaps.')
            if captured==previous and (sample['value']!=old['value'] or sample['available_at_seconds']!=old['available_at_seconds']):
                raise ValueError('A reused acquisition retains its identity and arrival, including across missing gaps.')
            if name=='camera':
                frame=sample['value'];prior=old['value']
                if frame['index']<prior['index']:raise ValueError('Camera frame indices cannot move backwards across missing gaps.')
                if frame['index']==prior['index'] and (frame!=prior or sample['available_at_seconds']!=old['available_at_seconds']):
                    raise ValueError('A reused camera frame retains its exact identity and arrival across missing gaps.')
                if frame['index']>prior['index'] and captured<=previous:raise ValueError('A new camera index requires a later exposure, including across missing gaps.')
        result[name]=sample
    return result


def _action(value,header,label):
    _keys(value,{'joint_targets_rad','gripper_position_m'},label)
    _vec(value['joint_targets_rad'],6,label,-10,10)
    if any(not low<=x<=high for x,(low,high) in zip(value['joint_targets_rad'],header['limits']['joint_limits_rad'])):
        raise ValueError('Actuator reference exceeds the bound native joint limits.')
    _bounded(value['gripper_position_m'],label+' finger slide',0,.035)


def _step(step,index,previous,header,previous_ramped):
    _keys(step,{'index','before','reference','command','applied','after'},'transition')
    if type(step['index']) is not int or step['index']!=index or step['before']!=previous:
        raise ValueError('Transitions must be contiguous and preserve the preceding observed state.')
    start=_snapshot(step['before'],header['calibration'])
    end=_snapshot(step['after'],header['calibration'],previous)
    if not math.isclose(end-start,DT,abs_tol=1e-9,rel_tol=0):raise ValueError('Only complete, ordered 20 ms transitions are retained.')
    reference=step['reference']
    if reference is not None:
        _keys(reference,{'computed_at_seconds','progress_seconds','action'},'reference')
        _time(reference['computed_at_seconds'],'reference computation',0,start)
        _time(reference['progress_seconds'],'reference progress');_action(reference['action'],header,'reference action')
    command=step['command']
    _keys(command,{'issued_at_seconds','requested','issued','authority','operator_correction','issue_reason'},'command')
    if command['issued_at_seconds']!=start or type(command['issued_at_seconds']) not in (int,float):
        raise ValueError('Command issue time must equal the causal before-state boundary.')
    _action(command['requested'],header,'requested');_action(command['issued'],header,'issued')
    authority={'authored_fixture':'authored_fixture','simulation_teleoperation':'operator','simulation_authored_controller':'authored_controller'}[header['provenance']['kind']]
    if command['authority']!=authority or type(command['operator_correction']) is not bool:
        raise ValueError('Command authority must match the declared producer kind.')
    if command['operator_correction'] and authority!='operator':raise ValueError('Only declared operator commands can claim operator correction.')
    if command['issue_reason'] not in ('unchanged','producer_limiter','operator_override'):
        raise ValueError('Issued command requires an explicit supported modification reason.')
    if command['issue_reason']=='unchanged' and command['requested']!=command['issued']:
        raise ValueError('An unchanged issued command must equal the requested intent.')
    if command['issue_reason']=='operator_override' and authority!='operator':
        raise ValueError('Only declared operator commands can claim an operator override.')
    applied=step['applied']
    _keys(applied,{'interval_start_seconds','interval_end_seconds','ramped_targets','native_position_controls','rate_limited','scope'},'applied')
    _time(applied['interval_start_seconds'],'application start')
    _time(applied['interval_end_seconds'],'application end')
    if applied['interval_start_seconds']!=start or applied['interval_end_seconds']!=end:
        raise ValueError('Applied-command interval must exactly match the observed transition.')
    if applied['scope']!='final_native_tick_only' or type(applied['rate_limited']) is not bool:
        raise ValueError('Applied controls must explicitly describe the final native tick only.')
    _action(applied['ramped_targets'],header,'ramped targets')
    # Endpoint evidence can falsify a declared slew cap, but cannot certify
    # every intermediate native tick or compensated final control.
    current=applied['ramped_targets'];limits=header['limits'];elapsed=end-start
    if (any(abs(a-b)>limits['joint_target_rate_rad_s']*elapsed+1e-9 for a,b in zip(current['joint_targets_rad'],previous_ramped['joint_targets_rad']))
            or abs(current['gripper_position_m']-previous_ramped['gripper_position_m'])>limits['jaw_target_rate_m_s']*elapsed+1e-9):
        raise ValueError('Ramped endpoint change exceeds the declared target rate bound.')
    controls=_vec(applied['native_position_controls'],7,'native position controls',-10,10)
    _action(dict(joint_targets_rad=controls[:6],gripper_position_m=controls[6]),header,'native controls')
    # Native controls may include compensation and are never equated with
    # ramped targets or measured q. Intra-step caps/contact cannot be audited
    # from endpoints alone; the limits hash binds a producer declaration.
    return step['after']


def validate_demonstration(document,*,require_targets=False):
    tape=_json(document)
    _keys(tape,HEADER|{'initial','initial_applied_targets','steps','outcome'},'demonstration')
    header={key:tape[key] for key in HEADER};_header(header)
    if _snapshot(tape['initial'],header['calibration'])!=0:raise ValueError('Recording must start at its declared zero origin.')
    _action(tape['initial_applied_targets'],header,'initial applied targets')
    if type(tape['steps']) is not list or not 0<=len(tape['steps'])<=MAX_STEPS:raise ValueError('Recording permits at most 1000 explicit transitions.')
    previous=tape['initial'];ramped=tape['initial_applied_targets'];known=_acquisition_order(previous,{})
    for index,step in enumerate(tape['steps']):
        previous=_step(step,index,previous,header,ramped)
        known=_acquisition_order(previous,known);ramped=step['applied']['ramped_targets']
    outcome=tape['outcome'];_keys(outcome,{'status','task_result','reason','end_seconds','scorer_sha256','evidence_sha256'},'outcome')
    if outcome['status'] not in ('complete','interrupted') or outcome['task_result'] not in ('unverified','success','failure','unknown'):
        raise ValueError('Complete/interrupted status and task outcome must remain separate.')
    if outcome['status']=='interrupted' and outcome['task_result']!='unknown':raise ValueError('Interrupted recordings have unknown task outcomes.')
    if outcome['status']=='complete' and not tape['steps']:raise ValueError('A complete recording requires at least one complete transition.')
    if outcome['end_seconds']!=previous['timestamp_seconds'] or type(outcome['end_seconds']) not in (int,float):raise ValueError('Outcome time must equal the last complete observation.')
    _identifier(outcome['reason'],'outcome reason')
    for name in ('scorer_sha256','evidence_sha256'):
        if outcome[name] is not None:_sha(outcome[name],name)
    if outcome['task_result'] in ('success','failure') and (outcome['scorer_sha256'] is None or outcome['evidence_sha256'] is None):
        raise ValueError('Declared task labels require separate scorer and evidence identities.')
    if require_targets:_require_pairs(tape)
    return tape


def _require_pairs(tape,*,allow_fixture=False):
    if tape['outcome']['status']!='complete':raise ValueError('Interrupted/partial recordings cannot supply complete learning pairs.')
    if tape['provenance']['kind']=='authored_fixture' and not allow_fixture:
        raise ValueError('Authored fixture values are not collected motor demonstrations.')
    for snapshot in [tape['initial']]+[s['after'] for s in tape['steps']]:
        if any(s['status']!='available' for s in snapshot['signals'].values()) or snapshot['camera']['status']!='available':
            raise ValueError('Missing or stale required signals/frames refuse learning and replay targets.')
    if any(step['reference'] is None for step in tape['steps']):raise ValueError('Residual targets require a causal pre-action reference.')


class Recorder:
    """In-memory data recorder. The caller supplies evidence; no actions execute."""
    def __init__(self,header,initial,initial_applied_targets):
        self.header=_json(header);_header(self.header)
        self.initial=_json(initial);_snapshot(self.initial,self.header['calibration'])
        if self.initial['timestamp_seconds']!=0:raise ValueError('Recorder initial time must be zero.')
        self.initial_applied_targets=_json(initial_applied_targets);_action(self.initial_applied_targets,self.header,'initial applied targets')
        self.steps=[];self.finished=False;self._known=_acquisition_order(self.initial,{})

    def append(self,step):
        if self.finished or len(self.steps)>=MAX_STEPS:raise ValueError('Recorder is finished or full.')
        step=_json(step);previous=self.steps[-1]['after'] if self.steps else self.initial
        ramped=self.steps[-1]['applied']['ramped_targets'] if self.steps else self.initial_applied_targets
        _step(step,len(self.steps),previous,self.header,ramped)
        known=_acquisition_order(step['after'],self._known)
        self.steps.append(step);self._known=known

    def finish(self,*,status='complete',task_result='unverified',reason='producer_finished',scorer_sha256=None,evidence_sha256=None):
        if self.finished:raise ValueError('Recorder was already finished.')
        end=self.steps[-1]['after']['timestamp_seconds'] if self.steps else self.initial['timestamp_seconds']
        tape=dict(self.header,initial=self.initial,initial_applied_targets=self.initial_applied_targets,steps=self.steps,
            outcome=dict(status=status,task_result=task_result,reason=reason,end_seconds=end,scorer_sha256=scorer_sha256,evidence_sha256=evidence_sha256))
        result=validate_demonstration(tape);self.finished=True;return result


def _ordinary_bytes(path,maximum):
    path=Path(path).absolute()
    if any(p.is_symlink() and str(p) not in ('/var','/tmp') for p in (path,*path.parents)):
        raise ValueError('Data files cannot traverse artifact symlinks.')
    before=path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size>maximum:raise ValueError('Expected a bounded regular non-symlink data file.')
    descriptor=os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0))
    with os.fdopen(descriptor,'rb') as stream:
        opened=os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or (before.st_dev,before.st_ino)!=(opened.st_dev,opened.st_ino):raise ValueError('Data file identity changed during opening.')
        body=stream.read(maximum+1)
    if len(body)>maximum:raise ValueError('Data file exceeds the contract limit.')
    return body


def load_demonstration(path):return validate_demonstration(_ordinary_bytes(path,MAX_BYTES))


def write_demonstration(path,document):
    tape=validate_demonstration(document);body=canonical(tape)+b'\n'
    if len(body)>MAX_BYTES:raise ValueError('Demonstration exceeds 8 MiB.')
    path=Path(path).absolute();temporary=None
    if any(p.is_symlink() and str(p) not in ('/var','/tmp') for p in (path,*path.parents)):
        raise ValueError('Demonstration output cannot traverse artifact symlinks.')
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.action-tape-',delete=False) as stream:
            temporary=Path(stream.name);stream.write(body);stream.flush();os.fsync(stream.fileno())
        os.link(temporary,path)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
    return dict(bytes=len(body),sha256=sha256(body).hexdigest())


def verify_frame_artifacts(tape,root):
    root=Path(root).absolute()
    if any(p.is_symlink() and str(p) not in ('/var','/tmp') for p in (root,*root.parents)):raise ValueError('Frame root cannot traverse symlinks.')
    inventory={}
    for snapshot in [tape['initial']]+[s['after'] for s in tape['steps']]:
        if snapshot['camera']['value'] is None:continue
        for name in ('rgb','depth'):
            artifact=snapshot['camera']['value'][name];path=root/str(_relative(artifact['path']))
            if any(p.is_symlink() for p in (path,*path.parents) if root==p or root in p.parents):raise ValueError('Frame files cannot traverse symlinks.')
            if artifact['path'] in inventory:
                if inventory[artifact['path']]!=artifact:raise ValueError('A frame path cannot identify different bytes.')
                continue
            body=_ordinary_bytes(path,MAX_FRAME_BYTES)
            if len(body)!=artifact['bytes'] or sha256(body).hexdigest()!=artifact['sha256']:raise ValueError('Frame artifact byte identity differs.')
            inventory[artifact['path']]=artifact
            if sum(value['bytes'] for value in inventory.values())>MAX_FRAME_TOTAL:raise ValueError('Frame readback exceeds 512 MiB.')
    return dict(unique_frame_files=len(inventory),bytes=sum(x['bytes'] for x in inventory.values()),
        byte_hashes_verified=True,images_decoded=False,depth_accuracy_verified=False)


def inspect_demonstration(document):
    tape=validate_demonstration(document);reasons=[]
    try:_require_pairs(tape)
    except ValueError as error:reasons.append(str(error))
    snapshots=[tape['initial']]+[s['after'] for s in tape['steps']]
    return dict(schema=SCHEMA+'/inspection',content_sha256=digest(tape),source_group=tape['provenance']['source_group'],
        kind=tape['provenance']['kind'],control_steps=len(tape['steps']),outcome=tape['outcome'],
        motor_command_targets_present=bool(tape['steps']),causal_learning_pairs_declared_available=not reasons,
        learning_refusal_reasons=reasons,frame_artifacts_verified=False,
        ramped_endpoint_rate_bounds_consistent=True,intermediate_rate_bounds_verified=False,
        operator_correction_steps=sum(s['command']['operator_correction'] for s in tape['steps']),
        missing_or_stale_snapshots=sum(any(x['status']!='available' for x in s['signals'].values()) or s['camera']['status']!='available' for s in snapshots),
        source_authenticated=False,corrective_demonstration_quality_verified=False,physical_robot_ready=False,
        scope='Data consistency only; no physics replay, image decoding, training, correction collection or hardware command execution.')


def learning_pairs(document,*,artifacts,allow_fixture=False):
    tape=validate_demonstration(document);_require_pairs(tape,allow_fixture=allow_fixture)
    proof=verify_frame_artifacts(tape,artifacts)
    rows=[]
    for step in tape['steps']:
        reference=step['reference']['action'];issued=step['command']['issued']
        residual=[a-b for a,b in zip(issued['joint_targets_rad'],reference['joint_targets_rad'])]
        residual.append(issued['gripper_position_m']-reference['gripper_position_m'])
        rows.append(dict(index=step['index'],before=deepcopy(step['before']),reference=deepcopy(step['reference']),
            goal_position_m=tape['goal_position_m'],issued_action_target=deepcopy(issued),residual_action_target=residual,
            source_group=tape['provenance']['source_group'],session_id=tape['provenance']['session_id']))
    return dict(schema=SCHEMA+'/learning-pairs',source_sha256=digest(tape),rows=rows,artifact_readback=proof,
        input_fields=['before','reference','goal_position_m'],target_fields=['issued_action_target','residual_action_target'],
        residual_action_order=deepcopy(tape['profile']['actuator_names']),residual_action_units=deepcopy(tape['profile']['native_command_units']),
        source_context={key:deepcopy(tape[key]) for key in HEADER},
        excluded_inputs=['requested_teacher_intent','current_issued_target','applied_final_controls','after_state','operator_correction','task_outcome','selection_metadata','source_context'],
        target_definition='Producer-issued native position references; applied controls and measured states are distinct evidence.',
        selection_metadata=dict(provenance=deepcopy(tape['provenance']),outcome=deepcopy(tape['outcome']),
            operator_correction_steps=sum(s['command']['operator_correction'] for s in tape['steps'])),
        selection_scope='All complete causal motor labels, including failure/unverified episodes. Not selected successful corrections; consumer must separately review source, task evidence and operator segments. Selection metadata is excluded from inputs.',
        authored_fixture=tape['provenance']['kind']=='authored_fixture',source_authenticated=False,
        corrective_demonstration_quality_verified=False,physical_robot_ready=False)


def replay_check(reference,replayed):
    """Compare two producer recordings. Never replay commands into an environment."""
    left=validate_demonstration(reference);right=validate_demonstration(replayed)
    _require_pairs(left,allow_fixture=True);_require_pairs(right,allow_fixture=True)
    if (left['binding']!=right['binding'] or left['goal_position_m']!=right['goal_position_m']
            or right['provenance']['replay_of_sha256']!=digest(left) or len(left['steps'])!=len(right['steps'])
            or left['initial_applied_targets']!=right['initial_applied_targets']):
        raise ValueError('Replay records require exact source, profile, scene, calibration and schedule bindings.')
    joint_errors=[];jaw_errors=[];applied_joint_errors=[];applied_jaw_errors=[]
    snapshots=[(left['initial'],right['initial'])]
    for a,b in zip(left['steps'],right['steps']):
        if a['command']['issued']!=b['command']['issued'] or a['command']['issued_at_seconds']!=b['command']['issued_at_seconds']:
            raise ValueError('Replay issued commands or shared timestamps differ.')
        snapshots.append((a['after'],b['after']))
        applied_joint_errors.extend(y-x for x,y in zip(a['applied']['native_position_controls'][:6],b['applied']['native_position_controls'][:6]))
        applied_jaw_errors.append(b['applied']['native_position_controls'][6]-a['applied']['native_position_controls'][6])
    for a,b in snapshots:
        for name,target in (('joint_position_rad',joint_errors),('gripper_joint_position_m',jaw_errors)):
            target.extend(y-x for x,y in zip(a['signals'][name]['value'],b['signals'][name]['value']))
    return dict(schema=SCHEMA+'/replay-check',reference_sha256=digest(left),replayed_sha256=digest(right),
        transitions=len(left['steps']),command_schedule_matches=True,initial_ramped_targets_match=True,joint_error_rad=_errors(joint_errors),
        measured_finger_error_m=_errors(jaw_errors),applied_joint_reference_error_rad=_errors(applied_joint_errors),
        applied_finger_reference_error_m=_errors(applied_jaw_errors),
        frame_artifacts_verified=False,physics_executed=False,robot_commands_sent=False,physical_robot_ready=False,
        scope='Read-only producer-record comparison, not dynamics replay or a skill/contact/safety verdict. Arm and finger errors retain separate units.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('validate','inspect','targets','replay-check'))
    parser.add_argument('recording');parser.add_argument('--replayed');parser.add_argument('--artifacts')
    args=parser.parse_args(argv)
    try:
        tape=load_demonstration(args.recording)
        if args.command=='targets':
            if not args.artifacts:raise ValueError('Learning targets require explicit local frame artifact readback.')
            result=learning_pairs(tape,artifacts=args.artifacts)
        elif args.command=='replay-check':
            if not args.replayed:raise ValueError('Replay-check requires a second recorded file; no physics is executed.')
            result=replay_check(tape,load_demonstration(args.replayed))
        else:result=inspect_demonstration(tape)
        print(json.dumps(result,indent=2,allow_nan=False));return 0
    except (ValueError,OSError,KeyError,TypeError,OverflowError) as error:
        print(json.dumps(dict(status='refused',reason=str(error),physics_executed=False,robot_commands_sent=False)));return 1


if __name__=='__main__':raise SystemExit(main())

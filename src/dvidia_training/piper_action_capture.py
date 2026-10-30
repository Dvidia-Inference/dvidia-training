"""Bounded native PiPER motor/RGB-D capture, never a learned or hardware skill.

The authored controller reads privileged simulator state. Object-tracker
channels remain missing: rendered pixels are not an implemented pose tracker.
Only complete 20 ms intervals enter the action-demonstration contract.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import platform
import struct
import tempfile
import time
import uuid
import zlib

from . import action_demonstration as tape


SCHEMA = 'dvidia.piper-action-capture/v1'
SCOPE = ('Native simulation motor and rendered RGB-D evidence from an authored '
         'controller. No object tracker, operator corrections, learned policy, '
         'hardware commands, task qualification or human-video motor recovery.')
SOURCES = ('piper_action_capture.py', 'action_demonstration.py', 'piper_task.py',
           'reference_task.py', 'reference_sim.py', 'robot_profiles.py', 'robot_profiles.lock.json')


@dataclass(frozen=True)
class CaptureConfig:
    duration_s: float = .2
    seed: int = 0
    mode: str = 'authored_task'
    width: int = 160
    height: int = 120
    interrupt_after: int | None = None

    def __post_init__(self):
        if (type(self.duration_s) not in (int, float) or not .1 <= self.duration_s <= 20.
                or not math.isfinite(self.duration_s)
                or not math.isclose(self.duration_s/tape.DT, round(self.duration_s/tape.DT), abs_tol=1e-9, rel_tol=0)):
            raise ValueError('Capture duration must be 0.1–20 seconds of complete 20 ms intervals.')
        if type(self.seed) is not int or not 0 <= self.seed <= 999999:raise ValueError('Seed must be an integer in [0,999999].')
        if self.mode not in ('hold', 'authored_task'):raise ValueError('Mode must be hold or authored_task; no learned or operator source is provided.')
        for name, low, high in (('width', 64, 320), ('height', 64, 240)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:raise ValueError(name+' is outside the bounded renderer dimensions.')
        intervals=round(self.duration_s/tape.DT)
        if self.interrupt_after is not None and (type(self.interrupt_after) is not int or not 0<=self.interrupt_after<intervals):
            raise ValueError('Engineering interruption must precede a scheduled interval.')
        # Conservative PNG/raw-depth allocation estimate; no hidden adaptive
        # resolution reduction or missing frames at the readback budget.
        if (self.width*self.height*7+4096)*(intervals+1)>tape.MAX_FRAME_TOTAL:
            raise ValueError('Configured RGB-D capture exceeds the 512 MiB frame readback budget.')


def _record(path, body):
    """Exclusive, flushed, atomic data creation; never overwrite an attempt."""
    path = Path(path); temporary = None
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.capture-', delete=False) as stream:
            temporary = Path(stream.name); stream.write(body); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
    return dict(path=path.name, bytes=len(body), sha256=sha256(body).hexdigest())


def _json(path, document):return _record(path, tape.canonical(document)+b'\n')


def _missing():return dict(status='missing', captured_at_seconds=None, available_at_seconds=None, age_seconds=None, value=None)


def _sample(value, seconds):
    return dict(status='available', captured_at_seconds=seconds, available_at_seconds=seconds, age_seconds=0., value=value)


def _png(rgb):
    height, width, _ = rgb.shape
    def chunk(kind, body):
        return struct.pack('>I',len(body))+kind+body+struct.pack('>I',zlib.crc32(kind+body)&0xffffffff)
    raw=b''.join(b'\0'+row.tobytes() for row in rgb)
    return (b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,2,0,0,0))
            +chunk(b'IDAT',zlib.compress(raw,6))+chunk(b'IEND',b''))


def optical_calibration(camera, *, base_position, base_rotation, width, height, duration_s):
    """Actual symmetric GL frustum → right/down/forward optical coordinates.

    Pixel coordinates refer to pixel centres (column+.5,row+.5). The camera is
    perspective, centred and mono; other projections are refused, not guessed.
    """
    # This pure numerical helper imports no simulator and sends no commands.
    def vector(value):return [float(x) for x in value]
    def normalize(v):
        n=math.hypot(*v)
        if not math.isfinite(n) or n<1e-9:raise ValueError('Degenerate camera basis.')
        return [x/n for x in v]
    def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
    forward=normalize(vector(camera.forward));right=normalize(cross(forward,vector(camera.up)))
    down=normalize(cross(forward,right))
    near=float(camera.frustum_near);top=float(camera.frustum_top);bottom=float(camera.frustum_bottom)
    if (camera.orthographic or not 0<near or not bottom<top
            or abs(float(camera.frustum_center))>1e-7 or abs(top+bottom)>1e-7):
        raise ValueError('Capture requires a centred mono perspective frustum.')
    focal=height*near/(top-bottom)
    rotation=[vector(row) for row in base_rotation];origin=vector(base_position);position=vector(camera.pos)
    # base_rotation is world_from_base. Optical axes are columns of world_from_optical.
    axes=[right,down,forward]
    transform=[]
    for axis in range(3):
        transform.extend(sum(rotation[k][axis]*column[k] for k in range(3)) for column in axes)
        transform.append(sum(rotation[k][axis]*(position[k]-origin[k]) for k in range(3)))
    transform.extend([0.,0.,0.,1.])
    return dict(id='native-fixed-perspective-v1',base_frame='robot_base',tool_frame='link6',camera_frame='rgbd_optical',
        intrinsics=dict(width=width,height=height,fx=focal,fy=focal,cx=width/2.,cy=height/2.),
        base_from_camera=transform,depth_units='m',valid_from_seconds=0.,valid_until_seconds=float(duration_s))


class _NativeSession:
    """Lazy optional native dependencies; the original task code stays intact."""
    def __init__(self, assets, config):
        self.assets=assets;self.config=config;self.renderer=None;self.env=None
        self.physics_attempted=False;self.initialized=False;self.renderer_warnings=[]

    def open(self):
        from . import piper_task as task, reference_sim as sim, robot_profiles
        self.mj,self.np=sim._native_imports();self.sim=sim
        self.task_config=task.PiperTaskConfig(timeout_s=self.config.duration_s)
        self.assets_metadata=robot_profiles.verify_assets('piper',task._asset_root(self.assets))
        self.env=task.PiperTaskEnv(self.assets,self.task_config);self.physics_attempted=True
        self.observation,self.info=self.env.reset(self.config.seed);self.initialized=True
        self.native_origin=float(self.env.data.time)
        self.base=self.mj.mj_name2id(self.env.model,self.mj.mjtObj.mjOBJ_BODY,'base_link')
        self.tool=self.mj.mj_name2id(self.env.model,self.mj.mjtObj.mjOBJ_BODY,'link6')
        if min(self.base,self.tool)<=0 or self.env.model.body_jntnum[self.base]:raise ValueError('Expected a fixed native base_link and link6 tool frame.')
        profile,lock_sha,record_sha=tape.profile_contract()
        if self.env.model.actuator_ctrlrange.tolist()!=profile['native_command_range']:raise ValueError('Compiled native command ranges differ from the packaged profile.')
        expected_joints=profile['arm_joint_names']+profile['gripper_joint_names']
        if ([self.mj.mj_id2name(self.env.model,self.mj.mjtObj.mjOBJ_ACTUATOR,i) for i in range(7)]!=profile['actuator_names']
                or [self.mj.mj_id2name(self.env.model,self.mj.mjtObj.mjOBJ_JOINT,i) for i in range(8)]!=expected_joints
                or self.env.model.jnt_qposadr[:8].tolist()!=list(range(8))
                or self.env.model.jnt_dofadr[:8].tolist()!=list(range(8))):
            raise ValueError('Compiled encoder/actuator channel ordering differs from the packaged PiPER profile.')
        visual=self.env.model.vis.global_;visual.orthographic=0;visual.fovy=45.;visual.ipd=0.
        visual.offwidth=max(visual.offwidth,self.config.width);visual.offheight=max(visual.offheight,self.config.height)
        self.camera=self.mj.MjvCamera();self.camera.lookat[:]=[.30,0.,.12]
        self.camera.distance=1.;self.camera.azimuth=135.;self.camera.elevation=-45.
        # Select the pinned classic renderer explicitly; its returned rows are
        # top-first and depth is linear metric optical-Z, not a raw z buffer.
        from mujoco.rendering.classic.renderer import Renderer
        with sim.native_warnings(self.renderer_warnings):self.renderer=Renderer(self.env.model,self.config.height,self.config.width)
        if getattr(self.renderer,'_gl_context',None) is None:raise RuntimeError('Classic renderer context unavailable; refusing to guess image row orientation.')
        with sim.native_warnings(self.renderer_warnings):self.renderer.update_scene(self.env.data,camera=self.camera)
        self.calibration=self._calibration()
        self.policy=task.PiperTaskPolicy(self.env)
        limits=dict(joint_limits_rad=deepcopy(profile['native_command_range'][:6]),
            gripper_position_limits_m=deepcopy(profile['native_command_range'][6]),
            joint_target_rate_rad_s=self.task_config.joint_target_rate_rad_s,jaw_target_rate_m_s=self.task_config.jaw_target_rate_m_s)
        source=Path(__file__).parent
        self.sources={name:sha256((source/name).read_bytes()).hexdigest() for name in SOURCES}
        self.configuration=dict(capture=asdict(self.config),task=asdict(self.task_config),
            camera=dict(lookat=[.30,0.,.12],distance=1.,azimuth_deg=135.,elevation_deg=-45.,fovy_deg=45.,ipd_m=0.,projection='perspective'),
            implementation_sha256=self.sources)
        unique=uuid.uuid4().hex
        self.header=dict(schema=tape.SCHEMA,scope='simulation-only',physical_robot_ready=False,
            provenance=dict(episode_id='native-'+unique,source_recording_id='native-source-'+unique,
                session_id='native-session-'+unique,source_group='piper-xy-'+tape.digest(list(self.task_config.object_position_m[:2]))[:20],
                kind='simulation_authored_controller',producer_revision=self.sources['piper_action_capture.py'],replay_of_sha256=None),
            binding=dict(profile_lock_sha256=lock_sha,profile_record_sha256=record_sha,
                model_sha256=sim.model_digest(self.env.model),scene_sha256=tape.digest(dict(task=asdict(self.task_config),assets=self.assets_metadata)),
                configuration_sha256=tape.digest(self.configuration),limits_sha256=tape.digest(limits),
                calibration_sha256=tape.digest(self.calibration),reference_policy_sha256=self.sources['piper_task.py' if self.config.mode=='authored_task' else 'piper_action_capture.py']),
            profile=profile,limits=limits,calibration=self.calibration,
            clock=dict(kind='shared_simulation_elapsed_seconds',origin='producer_zero',control_dt_seconds=tape.DT,maximum_sensor_age_seconds=tape.MAX_AGE),
            signal_contract=deepcopy(tape.SIGNALS),goal_position_m=list(self.task_config.target_position_m))
        self.metadata=dict(runtime=dict(mujoco=self.mj.__version__,numpy=self.np.__version__,python=platform.python_version(),system=platform.system(),machine=platform.machine()),
            configuration=self.configuration,assets=self.assets_metadata,native_time_origin_seconds=self.native_origin,
            camera=dict(renderer='mujoco.rendering.classic.renderer.Renderer',
                gl_context=type(getattr(self.renderer,'_gl_context',None)).__module__,
                exposure='Instantaneous synthetic exposure at a paused simulator boundary; no rolling shutter or physical delivery latency.',
                pixel_origin='top-left; sample centre at (column+.5,row+.5)',optical_axes='x right, y down, z forward',
                rgb='lossless PNG, uint8 RGB, H×W×3',depth='NPY v1+, little-endian float32 H×W, metres along optical z; no pickle; far-plane background retained',
                calibration_scope='Exact configured synthetic renderer/base transform; not hardware or depth-accuracy calibration.',
                depth_accuracy_verified=False,renderer_warnings_location='capture.json native_status.renderer_warnings'),
            policy_observation_origin='privileged_native_simulator_state',tracker_implemented=False,operator_corrections_collected=False)
        return deepcopy(self.header),self.ramped(),deepcopy(self.metadata)

    def _calibration(self):
        return optical_calibration(self.renderer.scene.camera[0],base_position=self.env.data.xpos[self.base],
            base_rotation=self.env.data.xmat[self.base].reshape(3,3),width=self.config.width,height=self.config.height,duration_s=self.config.duration_s)

    def ramped(self):
        return dict(joint_targets_rad=self.env.command_targets.tolist(),gripper_position_m=float(self.env.command_gripper))

    def sample(self,index):
        began=time.perf_counter();seconds=float(self.env.elapsed);native_time=float(self.env.data.time)
        if not math.isclose(native_time-self.native_origin,seconds,abs_tol=tape.CLOCK_TOL,rel_tol=0):raise RuntimeError('Native data time and declared episode elapsed clock differ.')
        self.renderer.disable_depth_rendering()
        with self.sim.native_warnings(self.renderer_warnings):self.renderer.update_scene(self.env.data,camera=self.camera)
        actual=self._calibration()
        if any(abs(a-b)>1e-6 for a,b in zip(actual['base_from_camera'],self.calibration['base_from_camera'])) or actual['intrinsics']!=self.calibration['intrinsics']:
            raise RuntimeError('Fixed rendered camera calibration changed.')
        with self.sim.native_warnings(self.renderer_warnings):
            rgb=self.renderer.render().copy()
            try:
                self.renderer.enable_depth_rendering();depth=self.renderer.render().copy()
            finally:self.renderer.disable_depth_rendering()
        if (rgb.shape!=(self.config.height,self.config.width,3) or rgb.dtype!=self.np.uint8
                or depth.shape!=(self.config.height,self.config.width) or depth.dtype!=self.np.float32
                or not self.np.isfinite(depth).all() or not (depth>0).all()):raise RuntimeError('Renderer did not supply finite aligned RGB and metric depth arrays.')
        if native_time!=float(self.env.data.time) or seconds!=float(self.env.elapsed):raise RuntimeError('Physics advanced during paired RGB-D capture.')
        depth_buffer=io.BytesIO();self.np.save(depth_buffer,depth.astype('<f4',copy=False),allow_pickle=False)
        frames={f'frames/rgb-{index:04}.png':_png(rgb),f'frames/depth-{index:04}.npy':depth_buffer.getvalue()}
        artifacts={name:dict(path=path,bytes=len(frames[path]),sha256=sha256(frames[path]).hexdigest())
            for name,path in (('rgb',f'frames/rgb-{index:04}.png'),('depth',f'frames/depth-{index:04}.npy'))}
        frame=dict(index=index,exposure_start_seconds=seconds,exposure_end_seconds=seconds,**artifacts)
        signals={name:_missing() for name in tape.SIGNALS}
        base_rotation=self.env.data.xmat[self.base].reshape(3,3)
        tool_rotation=base_rotation.T@self.env.data.xmat[self.tool].reshape(3,3)
        quat=self.np.empty(4);self.mj.mju_mat2Quat(quat,tool_rotation.reshape(9))
        values=dict(joint_position_rad=self.env.data.qpos[:6].tolist(),joint_velocity_rad_s=self.env.data.qvel[:6].tolist(),
            gripper_joint_position_m=self.env.data.qpos[6:8].tolist(),gripper_joint_velocity_m_s=self.env.data.qvel[6:8].tolist(),
            tcp_position_estimate_m=(base_rotation.T@(self.env.data.xpos[self.tool]-self.env.data.xpos[self.base])).tolist(),
            tcp_quaternion_estimate_wxyz=quat.tolist())
        for name,value in values.items():signals[name]=_sample(value,seconds)
        snapshot=dict(timestamp_seconds=seconds,signals=signals,camera=_sample(frame,seconds),vision_frame_index=index)
        diagnostic=dict(scope='Privileged native diagnostics, excluded from primary sensor learning inputs.',
            elapsed_seconds=seconds,native_data_time_seconds=native_time,native_info=deepcopy(self.info),
            policy_observation=deepcopy(self.observation))
        return snapshot,frames,diagnostic,dict(render_encode_wall_seconds=time.perf_counter()-began,
            depth_min_m=float(depth.min()),depth_max_m=float(depth.max()),renderer_warnings=list(self.renderer_warnings))

    def decide(self):
        if self.config.mode=='authored_task':action=self.policy(deepcopy(self.observation))
        else:action=deepcopy(self.initial_hold)
        return deepcopy(action),dict(scope='Privileged authored-controller decision input; not RGB-D tracking.',
            observation=deepcopy(self.observation),phase=self.policy.issued_phase if self.config.mode=='authored_task' else 'hold')

    def advance(self,action):
        before=float(self.env.elapsed)
        self.observation,self.info=self.env.step(action)
        if (self.observation is None or not self.info['valid'] or self.env.warnings
                or not math.isclose(self.env.elapsed-before,tape.DT,abs_tol=tape.CLOCK_TOL,rel_tol=0)):
            raise RuntimeError('Native interval is invalid, incomplete or warned; preserve only the preceding complete prefix.')
        return dict(interval_start_seconds=before,interval_end_seconds=float(self.env.elapsed),
            ramped_targets=self.ramped(),native_position_controls=self.env.data.ctrl.tolist(),
            rate_limited=self.ramped()!=action,scope='final_native_tick_only')

    def status(self):
        return dict(physics_attempted=self.physics_attempted,native_initialized=self.initialized,
            actual_native_elapsed_seconds=float(self.env.elapsed) if self.initialized else None,
            actual_native_data_time_seconds=float(self.env.data.time) if self.initialized else None,
            native_info=deepcopy(self.info) if self.initialized else None,
            renderer_warnings=list(self.renderer_warnings),depth_accuracy_verified=False)

    def close(self):
        if self.renderer is not None:self.renderer.close()


def capture(assets,output,*,config=None):
    """Create a fresh local capture; exceptions seal a refusal/unknown prefix.

    Started decision files precede native commands. Hard termination can leave
    an unsealed directory; it is never treated as complete or auto-retried.
    """
    config=CaptureConfig() if config is None else config
    if type(config) is not CaptureConfig:raise TypeError('Expected CaptureConfig.')
    root=Path(output).absolute()
    if any(p.is_symlink() and str(p) not in ('/var','/tmp') for p in (root,*root.parents)):raise ValueError('Capture output cannot traverse symlinks.')
    root.mkdir(parents=True,exist_ok=False)
    _json(root/'started.json',dict(schema=SCHEMA,status='started',config=asdict(config),scope=SCOPE,physical_robot_ready=False))
    session=_NativeSession(assets,config);recorder=None;recording=None;metadata=None;error=None
    started=time.perf_counter();timings=[];started_intervals=0;frame_files=0;reason='capture_budget_complete'
    try:
        header,initial_targets,metadata=session.open()
        snapshot,frames,diagnostic,timing=session.sample(0)
        for path,body in frames.items():_record(root/path,body);frame_files+=1
        _json(root/'native/0000.json',diagnostic);timings.append(timing)
        recorder=tape.Recorder(header,snapshot,initial_targets)
        session.initial_hold=deepcopy(initial_targets)
        _json(root/'context.json',dict(header=header,initial_applied_targets=initial_targets,initial=snapshot,metadata=metadata))
        for index in range(round(config.duration_s/tape.DT)):
            if config.interrupt_after is not None and index>=config.interrupt_after:
                raise KeyboardInterrupt('Declared engineering interruption before the next native action.')
            boundary=time.perf_counter();action,decision=session.decide();seconds=snapshot['timestamp_seconds']
            command=dict(issued_at_seconds=seconds,requested=deepcopy(action),issued=deepcopy(action),
                authority='authored_controller',operator_correction=False,issue_reason='unchanged')
            reference=dict(computed_at_seconds=seconds,progress_seconds=seconds,action=deepcopy(action))
            # Validate intent before journalling or calling the native step.
            tape._action(action,header,'native authored decision')
            _json(root/f'decisions/{index:04}.json',dict(index=index,before_seconds=seconds,reference=reference,command=command,privileged_decision=decision))
            started_intervals+=1;applied=session.advance(action)
            after,frames,diagnostic,timing=session.sample(index+1)
            for path,body in frames.items():_record(root/path,body);frame_files+=1
            step=dict(index=index,before=snapshot,reference=reference,command=command,applied=applied,after=after)
            recorder.append(step)
            _json(root/f'transitions/{index:04}.json',step);_json(root/f'native/{index+1:04}.json',diagnostic)
            timing['full_boundary_wall_seconds']=time.perf_counter()-boundary;timings.append(timing);snapshot=after
    except (Exception,KeyboardInterrupt) as exc:
        reason='producer_interrupted' if isinstance(exc,KeyboardInterrupt) else 'producer_error'
        message=exc.strerror if isinstance(exc,OSError) and exc.strerror else str(exc)
        for path,label in ((str(Path(assets).absolute()),'<asset-root>'),(str(root),'<capture-root>'),(str(Path(__file__).parent),'<package-source>')):
            message=message.replace(path,label)
        error=dict(type=type(exc).__name__,message=message[:500])
    finally:
        native_status=session.status()
        try:session.close()
        except Exception as exc:
            if error is None:error=dict(type=type(exc).__name__,message=str(exc)[:500]);reason='renderer_close_error'
    status='complete' if error is None else ('interrupted' if recorder is not None else 'refused')
    if recorder is not None:
        recording=recorder.finish(status='complete' if status=='complete' else 'interrupted',
            task_result='unverified' if status=='complete' else 'unknown',reason=reason)
        tape.write_demonstration(root/'recording.json',recording)
    report=dict(schema=SCHEMA,status=status,scope=SCOPE,physical_robot_ready=False,
        config=asdict(config),requested_intervals=round(config.duration_s/tape.DT),started_intervals=started_intervals,
        recorded_complete_intervals=len(recording['steps']) if recording else 0,
        recorded_complete_seconds=recording['outcome']['end_seconds'] if recording else 0.,
        error=error,reason=reason,native_status=native_status,metadata=metadata,frame_files=frame_files,
        inspection=tape.inspect_demonstration(recording) if recording else None,
        recording_sha256=sha256((root/'recording.json').read_bytes()).hexdigest() if recording else None,
        timing=dict(wall_seconds=time.perf_counter()-started,frames=timings,
            scope='Shared-host wall acquisition/render/serialization computation; simulator pauses during capture. Not physical delivery or real-time control latency.'),
        strict_learning_targets_available=False,tracker_implemented=False,operator_corrections_collected=False,
        task_success_verified=False,uncommitted_native_intervals=started_intervals-(len(recording['steps']) if recording else 0))
    _json(root/'capture.json',report)
    files={str(path.relative_to(root)):dict(bytes=path.stat().st_size,sha256=sha256(path.read_bytes()).hexdigest()) for path in sorted(root.rglob('*')) if path.is_file()}
    _json(root/'integrity.json',dict(schema=SCHEMA+'/integrity',files=files))
    return report


def inspect_capture(directory):
    """Offline complete-file readback; never advance or initialize a simulator."""
    root=Path(directory);manifest=tape._json(tape._ordinary_bytes(root/'integrity.json',tape.MAX_BYTES))
    tape._keys(manifest,{'schema','files'},'capture integrity')
    if manifest.get('schema')!=SCHEMA+'/integrity' or type(manifest.get('files')) is not dict:raise ValueError('Unsealed or invalid capture integrity manifest.')
    if not 2<=len(manifest['files'])<=6000:raise ValueError('Capture file inventory is outside the bounded format.')
    observed={str(path.relative_to(root)) for path in root.rglob('*') if path.is_file() and path!=root/'integrity.json'}
    if observed!=set(manifest['files']):raise ValueError('Capture file inventory changed.')
    for name,record in manifest['files'].items():
        tape._keys(record,{'bytes','sha256'},'capture file identity');tape._sha(record['sha256'],'capture file hash')
        tape._integer(record['bytes'],'capture file bytes',minimum=1)
        path=root/str(tape._relative(name));body=tape._ordinary_bytes(path,tape.MAX_FRAME_BYTES)
        if len(body)!=record['bytes'] or sha256(body).hexdigest()!=record['sha256']:raise ValueError('Capture file byte identity differs.')
    report=tape._json(tape._ordinary_bytes(root/'capture.json',tape.MAX_BYTES))
    if report.get('schema')!=SCHEMA or report.get('status') not in ('complete','interrupted','refused'):raise ValueError('Unknown capture receipt.')
    result=dict(schema=SCHEMA+'/inspection',status=report['status'],files_verified=len(manifest['files']),
        recorded_complete_intervals=report['recorded_complete_intervals'],started_intervals=report['started_intervals'],
        native_status=report['native_status'],strict_learning_targets_available=False,source_authenticated=False,
        tracker_implemented=False,operator_corrections_collected=False,physics_executed=False,robot_commands_sent=False,
        physical_robot_ready=False,scope='Read-only local capture byte/contract verification; not native replay, task scoring, tracker validation or hardware qualification.')
    if report['status']!='refused':
        body=tape._ordinary_bytes(root/'recording.json',tape.MAX_BYTES);recording=tape.validate_demonstration(body)
        if sha256(body).hexdigest()!=report['recording_sha256']:raise ValueError('Recording identity differs from capture receipt.')
        expected='complete' if report['status']=='complete' else 'interrupted'
        if (recording['outcome']['status']!=expected or len(recording['steps'])!=report['recorded_complete_intervals']
                or report['recorded_complete_seconds']!=recording['outcome']['end_seconds']
                or report['started_intervals']-len(recording['steps'])!=report['uncommitted_native_intervals']
                or (expected=='complete' and len(recording['steps'])!=report['requested_intervals'])):
            raise ValueError('Capture completeness or interruption counts disagree.')
        result['recording']=tape.inspect_demonstration(recording)
        result['frames']=tape.verify_frame_artifacts(recording,root)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    run=commands.add_parser('run');run.add_argument('--assets',required=True);run.add_argument('--output',required=True)
    run.add_argument('--duration',type=float,default=.2);run.add_argument('--seed',type=int,default=0)
    run.add_argument('--mode',choices=('hold','authored_task'),default='authored_task')
    run.add_argument('--width',type=int,default=160);run.add_argument('--height',type=int,default=120)
    run.add_argument('--interrupt-after',type=int,help='Engineering negative control: stop before the next interval after this many complete intervals.')
    check=commands.add_parser('inspect');check.add_argument('directory');args=parser.parse_args(argv)
    try:
        result=(inspect_capture(args.directory) if args.command=='inspect' else capture(args.assets,args.output,
            config=CaptureConfig(args.duration,args.seed,args.mode,args.width,args.height,args.interrupt_after)))
        print(json.dumps(result,indent=2,allow_nan=False));return 0 if result['status']=='complete' else 1
    except (ValueError,OSError,TypeError,RuntimeError) as exc:
        print(json.dumps(dict(status='refused',reason=str(exc),robot_commands_sent=False,physical_robot_ready=False)));return 1


if __name__=='__main__':raise SystemExit(main())

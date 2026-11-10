"""Local, externally controlled PiPER block recording; no hardware commands.

Godot is a view/controller. The camera-action native environment owns 1 ms
physics, 50 Hz commands and 25 Hz RGB-D. This protocol is separate from the
2 ms shoe workstation and never relabels its recordings.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import socket
import time

from . import asset_preview, camera_action_capture as camera, godot_bridge as wire
from . import reference_replay, reference_sim

SCHEMA = 'dvidia.godot-demonstration/v1'
DEFAULT_PORT = 8431
BOUNDS = ((.10, .50), (-.25, .25), (.06, .55))
DEFAULT_CASE = dict(id='operator-block-001', group_id='operator-start-001', seed=0,
                    object_position_m=[.30, -.03, .082], goal_position_m=[.35, .08, .08])
SCOPE = ('Simulation demonstration capture only. Fixed-orientation XYZ plus single-finger '
         'target; 1 ms MuJoCo physics, 50 Hz controls, 25 Hz native RGB-D. '
         'No hand tracking, trained policy, physical commands or hardware qualification.')
TIMING_SCOPE = 'Paused-simulator computation and capture; not physical reaction latency.'
MAX_EPISODES = 16
RESERVE_BYTES = 10 * 1024**3


def _case(path):
    return camera._case(DEFAULT_CASE if path is None else json.loads(Path(path).read_text()))


def _state(native, target):
    """Privileged viewer state. It is never an input to the pose controller."""
    env, mj, np = native.env, native.mj, native.np
    quaternion = np.empty(4)
    mj.mju_mat2Quat(quaternion, env.data.site_xmat[env.tcp])
    return dict(time=float(env.elapsed), native_time=float(env.data.time),
        body_positions=env.data.xpos.tolist(), body_quaternions=env.data.xquat.tolist(),
        qpos=env.data.qpos.tolist(), ctrl=env.data.ctrl.tolist(),
        tcp_position=env.data.site_xpos[env.tcp].tolist(), tcp_quaternion=quaternion.tolist(),
        object_position=env.data.xpos[env.block].tolist(),
        position_error_m=float(np.linalg.norm(env.data.site_xpos[env.tcp]-target)),
        scope='Viewer state includes native truth; excluded from learning inputs.')


def prepare(assets, output, *, case=None, duration_s=20.):
    case = camera._case(DEFAULT_CASE if case is None else case)
    config = camera.RolloutConfig(duration_s=duration_s, seed=case['seed'])
    destination = Path(output).absolute()
    if destination.exists() or destination.is_symlink() or any(p.is_symlink() for p in destination.parents):
        raise ValueError('Prepared scene requires a fresh ordinary directory.')
    provenance, notices = asset_preview._licenses('piper', assets)
    native = camera._Session(assets, config, case)
    try:
        header, _, metadata = native.open()
        model = native.env.model
        digest = reference_sim.model_digest(model)
        scene, geometry = asset_preview._scene(model, dict(robot='piper', assets=provenance,
            mujoco=native.mj.__version__, compiled_model_sha256=digest))
        home = native.env.data.site_xpos[native.env.tcp].tolist()
        if any(not low <= value <= high for value, (low, high) in zip(home, BOUNDS)):
            raise ValueError('Initial TCP lies outside the declared operator envelope.')
        scene['scope'] = SCOPE
        scene['bridge'] = dict(schema=SCHEMA, model_sha256=digest, home_target=home,
            bounds=[list(p) for p in BOUNDS], gripper_initial=.035, physics_dt=.001,
            control_dt=.02, fixed_orientation_wxyz=list(native.task_config.tool_orientation_wxyz),
            case=case, duration_s=duration_s, calibration_sha256=header['binding']['calibration_sha256'],
            scope=SCOPE)
        scene['geometry_binary'] = dict(href='geometry.bin', compression='none',
            bytes=len(geometry), sha256=sha256(geometry).hexdigest())
        files = {'scene.json': wire._json(scene), 'geometry.bin': geometry,
            'LICENSE': reference_replay.MIT_LICENSE.encode(), **notices}
        files['README.md'] = ('# DVIDIA demonstration scene\n\n'+SCOPE+'\n\n'
            'Prepared using the native block task reset/settling, not a zero-physics export. '
            'Training RGB-D comes from the native calibrated camera, not the Godot viewport. '
            'RGB-D calibration is simulated; it does not qualify physical transfer. '
            'The upstream PiPER notices are in licenses/. No model weights are included.\n').encode()
        seal = dict(schema=wire.INTEGRITY_SCHEMA, model_sha256=digest,
            files={name:dict(bytes=len(body), sha256=sha256(body).hexdigest()) for name, body in files.items()})
        files['scene-integrity.json'] = wire._json(seal)
        if (any(len(body)>asset_preview.MAX_FILE_BYTES for body in files.values())
                or sum(map(len,files.values()))>asset_preview.MAX_TOTAL_BYTES):
            raise ValueError('Prepared geometry exceeds the existing export budget.')
        destination.mkdir(parents=True, exist_ok=False)
        for name, body in files.items():
            path = destination/name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(body)
        return dict(schema=SCHEMA, output=str(destination), model_sha256=digest,
            bytes=sum(map(len,files.values())), native_reset_performed=True,
            task_control_steps=0, physical_robot_ready=False, runtime=metadata['runtime'])
    finally:
        native.close()


class Session:
    """One connection, bounded new episodes; no command loop without a client."""
    def __init__(self, factory, controller, *, model_sha256, home_target, max_episodes=4,
                 state_reader=_state):
        if type(max_episodes) is not int or not 1 <= max_episodes <= MAX_EPISODES:
            raise ValueError('Episode budget must be in [1,16].')
        self.factory, self.controller, self.state_reader = factory, controller, state_reader
        self.model_sha256, self.home_target = model_sha256, list(home_target)
        self.max_episodes, self.episodes = max_episodes, 0
        self.expected_sequence, self.ready, self.faulted = 1, False, False
        self.status, self.capture, self.last_report = 'ready', None, None
        self.target, self.jaw = list(home_target), .035
        self.steps, self.max_steps = 0, 0
        self.controller_sources = {name:sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in ('godot_demonstration_bridge.py','piper_encoder_kinematics.py')}
        self._new_capture()

    def _new_capture(self):
        if self.episodes >= self.max_episodes:
            raise wire.ProtocolError('Episode budget exhausted; start a new explicitly bounded session.', 'episode_budget')
        capture = self.factory(self.episodes)
        try:
            opened = capture.open()
            if opened['header']['binding']['model_sha256'] != self.model_sha256:
                raise ValueError('Prepared and recording model digests differ.')
        except BaseException:
            capture.finish(reason='initialization_failed')
            raise
        self.capture = capture
        self.episodes += 1
        self.max_steps = round(capture.config.duration_s/.02)
        self.steps, self.status, self.last_report = 0, 'ready', None
        self.target, self.jaw = list(self.home_target), .035
        self.last_state = self.state_reader(capture.session, self.target)

    def recording(self):
        return dict(status=self.status, steps=self.steps, max_steps=self.max_steps,
            source_kind=self.capture.source_kind, output_directory=str(self.capture.root),
            episode=self.episodes, max_episodes=self.max_episodes,
            physical_robot_ready=False, source_authenticated=False)

    def error(self, exc):
        return dict(type='error', code=exc.code, message=str(exc), expected_sequence=self.expected_sequence)

    def _finish(self, reason):
        report = self.capture.finish(reason=reason)
        self.last_report = report
        if (report.get('sealed') is not True or report.get('error') is not None
                or report.get('native_evidence_saved') is not True):
            raise RuntimeError('Capture could not seal complete native evidence; inspect retained files.')
        return report

    def process(self, payload):
        try:
            return self.handle(wire.decode_request(payload))
        except wire.ProtocolError as exc:
            return self.error(exc)

    def handle(self, request):
        if self.faulted:
            raise wire.ProtocolError('Session faulted; retained evidence may include an incomplete interval.', 'session_fault')
        kind = request.get('type')
        if not self.ready:
            wire._fields(request, ('type','schema','model_sha256'))
            if kind != 'hello' or request['schema'] != SCHEMA or request['model_sha256'] != self.model_sha256:
                raise wire.ProtocolError('Prepared model and demonstration protocol must match.', 'handshake_required')
            self.ready = True
            return dict(type='ready', schema=SCHEMA, model_sha256=self.model_sha256,
                physics_dt=.001, control_dt=.02, state=self.last_state, recording=self.recording(), scope=SCOPE)
        if kind == 'ping':
            wire._fields(request, ('type',))
            return dict(type='pong', model_sha256=self.model_sha256, expected_sequence=self.expected_sequence)
        if kind not in ('begin','step','pause','resume','finish','reset'):
            raise wire.ProtocolError('Unknown demonstration command.')
        wire._fields(request, ('type','sequence','target','jaw') if kind=='step' else ('type','sequence'))
        sequence = request['sequence']
        if type(sequence) is not int or sequence != self.expected_sequence or sequence > wire.MAX_SEQUENCE:
            raise wire.ProtocolError('Expected the next command sequence.', 'sequence_mismatch')
        allowed = {'begin':('ready',),'step':('recording',),'pause':('recording',),
                   'resume':('paused',),'finish':('ready','recording','paused','finished'),
                   'reset':('ready','recording','paused','finished')}
        if self.status not in allowed[kind]:
            raise wire.ProtocolError('Command does not match recording status.', 'recording_status')
        if kind=='reset' and self.episodes >= self.max_episodes:
            raise wire.ProtocolError('Episode budget exhausted.', 'episode_budget')
        began = time.perf_counter()
        target, jaw, action = self.target, self.jaw, None
        if kind=='step':
            if type(request['target']) is not list or len(request['target']) != 3:
                raise wire.ProtocolError('Target requires three coordinates.')
            target = [wire._number(x,*bounds,'Target') for x,bounds in zip(request['target'],BOUNDS)]
            jaw = wire._number(request['jaw'],0.,.035,'Single-finger target')
            # Only measured encoder state is provided to this isolated FK/IK
            # controller; never the viewer's native object or contact truth.
            try:
                joints = self.capture.context()['snapshot']['signals']['joint_position_rad']['value']
                action = self.controller(joints,target,jaw)
            except ValueError as exc:
                raise wire.ProtocolError('Encoder-to-target mapping refused: '+str(exc),'control_refused') from exc
        try:
            if kind=='begin':
                self.capture.record_event('begin')
                self.status='recording'
            elif kind=='pause':
                self.capture.record_event('pause')
                self.status='paused'
            elif kind=='resume':
                self.capture.record_event('resume')
                self.status='recording'
            elif kind=='step':
                result = self.capture.step(action, input_event=dict(schema=SCHEMA,
                    sequence=sequence, target_world_m=target, gripper_single_finger_m=jaw,
                    input='fixed_orientation_cartesian', controller='encoder_only_dls',
                    controller_sources_sha256=self.controller_sources))
                self.steps += 1
                self.target,self.jaw = target,jaw
                self.last_state = self.state_reader(self.capture.session,target)
                if result['done']:
                    self._finish('capture_budget_complete')
                    self.status='finished'
            elif kind=='finish':
                if self.status!='finished':
                    self._finish('operator_stopped')
                    self.status='finished'
            elif kind=='reset':
                if self.status!='finished':
                    self.capture.record_event('reset')
                    self._finish('operator_reset')
                self._new_capture()
        except BaseException as exc:
            self.faulted=True
            try:
                self.capture.finish(reason='session_fault')
            except Exception:
                pass  # Original prefix remains recoverable; never retry a command.
            if isinstance(exc,(KeyboardInterrupt,SystemExit)):
                raise
            raise wire.ProtocolError('Recording failed; inspect the retained prefix before starting a new episode.', 'recording_fault') from exc
        self.expected_sequence += 1
        return dict(type='state',sequence=sequence,model_sha256=self.model_sha256,
            state=self.last_state,recording=self.recording(),
            step_compute_wall_seconds=time.perf_counter()-began,timing_scope=TIMING_SCOPE)

    def disconnect(self):
        if self.status!='finished':
            try:
                self.capture.record_event('disconnect')
                self._finish('client_disconnected')
                self.status='finished'
            except Exception:
                self.faulted=True
                self.status='faulted'
                self.capture.session.close()


def serve(assets, scene, output, *, port=DEFAULT_PORT, duration_s=None,
          source_kind='simulation_human_input', max_episodes=4):
    from .operator_capture import OperatorCapture, SOURCE_KINDS
    from .piper_encoder_kinematics import PiperEncoderKinematics
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError('Port must be an integer in [1024,65535].')
    if source_kind not in SOURCE_KINDS:
        raise ValueError('Unknown capture source kind.')
    if type(max_episodes) is not int or not 1 <= max_episodes <= MAX_EPISODES:
        raise ValueError('Episode budget must be in [1,16].')
    scene = Path(scene)
    seal = json.loads((scene/'scene-integrity.json').read_text())
    body = (scene/'scene.json').read_bytes()
    if seal.get('files',{}).get('scene.json') != dict(bytes=len(body),sha256=sha256(body).hexdigest()):
        raise ValueError('Prepared scene identity differs.')
    bridge = json.loads(body)['bridge']
    if bridge['schema'] != SCHEMA or bridge['physics_dt'] != .001 or bridge['control_dt'] != .02:
        raise ValueError('Unsupported prepared scene protocol or clocks.')
    duration_s = bridge['duration_s'] if duration_s is None else duration_s
    if duration_s != bridge['duration_s']:
        raise ValueError('Duration must match prepared calibration; prepare a fresh scene for a different duration.')
    config = camera.RolloutConfig(duration_s=duration_s, seed=bridge['case']['seed'])
    root=Path(output).absolute()
    if root.exists() or root.is_symlink() or any(p.is_symlink() for p in root.parents):
        raise ValueError('Session output must be a fresh ordinary directory.')
    parent=root.parent
    while not parent.exists(): parent=parent.parent
    upper=(round(duration_s/.04)+1)*(320*240*7+4096)+32*1024**2
    if shutil.disk_usage(parent).free < RESERVE_BYTES+max_episodes*upper:
        raise ValueError('Capture capacity guard: preserve 10 GiB plus the declared episode budget.')
    kinematics=PiperEncoderKinematics(assets)
    def controller(joints,target,jaw):
        return kinematics.pose_action(joints,target,bridge['fixed_orientation_wxyz'],jaw)
    root.mkdir(parents=True,exist_ok=False)
    def factory(index):
        case=deepcopy(bridge['case']);case['id'] += '-episode-'+str(index+1)
        return OperatorCapture(assets,root/f'episode-{index+1:04}',case=case,
            duration_s=duration_s,source_kind=source_kind)
    session=None
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        listener.bind((wire.HOST,port));listener.listen(1)
        print(json.dumps(dict(schema=SCHEMA,status='listening',host=wire.HOST,port=port,
            output=str(root),source_kind=source_kind,max_episodes=max_episodes)),flush=True)
        try:
            connection,_=listener.accept()
            with connection:
                connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
                connection.settimeout(wire.IDLE_TIMEOUT_SECONDS)
                session=Session(factory,controller,model_sha256=bridge['model_sha256'],
                    home_target=bridge['home_target'],max_episodes=max_episodes)
                wire.handle_connection(connection,session)
        finally:
            if session is not None: session.disconnect()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    pre=sub.add_parser('prepare');pre.add_argument('--assets',required=True,type=Path)
    pre.add_argument('--output',required=True,type=Path);pre.add_argument('--case',type=Path)
    pre.add_argument('--duration',type=float,default=20.)
    run=sub.add_parser('serve');run.add_argument('--assets',required=True,type=Path)
    run.add_argument('--scene',required=True,type=Path);run.add_argument('--output',required=True,type=Path)
    run.add_argument('--port',type=int,default=DEFAULT_PORT)
    run.add_argument('--source-kind',choices=('simulation_human_input','simulation_scripted_input'),default='simulation_human_input')
    run.add_argument('--max-episodes',type=int,default=4)
    inspect=sub.add_parser('inspect');inspect.add_argument('directory',type=Path)
    recover=sub.add_parser('recover');recover.add_argument('directory',type=Path)
    args=parser.parse_args(argv)
    try:
        if args.command=='prepare':
            result=prepare(args.assets,args.output,case=_case(args.case),duration_s=args.duration)
        elif args.command=='serve':
            serve(args.assets,args.scene,args.output,port=args.port,source_kind=args.source_kind,max_episodes=args.max_episodes)
            return 0
        else:
            from .operator_capture import inspect_capture,recover_capture
            result=(inspect_capture if args.command=='inspect' else recover_capture)(args.directory)
        print(json.dumps(result,indent=2))
    except KeyboardInterrupt:
        return 0
    except (ValueError,OSError,RuntimeError) as exc:
        parser.exit(2,str(exc)+'\n')
    return 0


if __name__=='__main__':
    raise SystemExit(main())

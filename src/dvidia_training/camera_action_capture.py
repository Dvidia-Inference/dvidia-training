"""Causal camera/action rollout v2 for the bounded simulated PiPER task.

Rendered observations, issued commands, privileged teacher suggestions, and
native scoring are distinct artifacts. This is an offline simulation recorder;
it makes no real-time delivery or physical-hardware qualification claim.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import random
import time

from . import action_demonstration as tape
from . import piper_action_capture as base
from . import piper_rgbd_tracker as tracker

SCHEMA = 'dvidia.camera-action-rollout/v2'
PERCEPTION_SCHEMA = 'dvidia.camera-action-perception/v1'
TEACHER_SCHEMA = 'dvidia.camera-action-teacher-query/v1'
MAX_ROLLOUT_BYTES = 64 * 1024 * 1024
MODES = ('teacher', 'student', 'mixed', 'fixed')
NEGATIVES = (None, 'initial_missing', 'initial_stale', 'altered_calibration', 'open_gripper')
SCOPE = ('Offline native simulation with authored teacher demonstrations and '
         'current-frame RGB-D estimates. No hardware commands, physical delivery '
         'latency, human-video motor recovery or general object tracking.')


@dataclass(frozen=True)
class RolloutConfig:
    duration_s: float = 20.
    seed: int = 0
    mode: str = 'authored_task'
    width: int = 320
    height: int = 240
    camera_every_controls: int = 2
    interrupt_after: int | None = None

    def __post_init__(self):
        value = self.duration_s
        if (type(value) not in (int, float) or not math.isfinite(value) or not .1 <= value <= 20.
                or not math.isclose(value / tape.DT, round(value / tape.DT), abs_tol=1e-9, rel_tol=0)):
            raise ValueError('Duration requires complete 20 ms intervals in [0.1,20] seconds.')
        if type(self.seed) is not int or not 0 <= self.seed <= 999999:
            raise ValueError('Seed must be an integer in [0,999999].')
        if (self.mode != 'authored_task' or type(self.width) is not int or type(self.height) is not int
                or self.width != 320 or self.height != 240
                or type(self.camera_every_controls) is not int or self.camera_every_controls != 2):
            raise ValueError('The registered camera contract is 320x240 RGB-D at 25 Hz.')
        count = round(value / tape.DT)
        if self.interrupt_after is not None and (type(self.interrupt_after) is not int
                or not 0 <= self.interrupt_after < count):
            raise ValueError('Interruption must precede a scheduled interval.')
        pairs = count // self.camera_every_controls + 1
        if (self.width * self.height * 7 + 4096) * pairs > tape.MAX_FRAME_TOTAL:
            raise ValueError('Unique camera frames exceed the existing 512 MiB frame budget.')


def _case(case):
    if type(case) is not dict:
        raise ValueError('A registered case dictionary is required.')
    result = deepcopy(case)
    for name in ('id', 'group_id'):
        if not isinstance(case.get(name), str) or not case[name] or len(case[name]) > 200:
            raise ValueError('Case requires bounded nonempty '+name)
    if type(case.get('seed')) is not int or not 0 <= case['seed'] <= 999999:
        raise ValueError('Case requires a bounded seed.')
    from .piper_task import PiperTaskConfig
    config = PiperTaskConfig(object_position_m=case.get('object_position_m'),
                             target_position_m=case.get('goal_position_m'))
    result['object_position_m'] = list(config.object_position_m)
    result['goal_position_m'] = list(config.target_position_m)
    return result


def _identity(path):
    path = Path(path)
    return dict(bytes=path.stat().st_size, sha256=sha256(path.read_bytes()).hexdigest())


def _artifact(root, relative, document):
    identity = base._json(root / relative, document)
    identity['path'] = relative
    return identity


def _sample_at(sample, seconds):
    result = deepcopy(sample)
    if result['status'] == 'available':
        result['age_seconds'] = max(0., seconds - result['captured_at_seconds'])
    return result


def _estimate_at(estimate, seconds):
    """Reuse one captured frame, retaining its original acquisition timestamp."""
    result = deepcopy(estimate)
    result['decision_at_seconds'] = seconds
    if result.get('captured_at_seconds') is not None:
        result['age_seconds'] = max(0., seconds - result['captured_at_seconds'])
    for name, sample in result.get('signals', {}).items():
        result['signals'][name] = _sample_at(sample, seconds)
    if result.get('status') == 'available' and result['age_seconds'] > .1 + 1e-12:
        result['status'], result['reason'] = 'stale', 'source_frame_too_old'
        for name in tracker.SIGNALS:
            result[name] = None
            result['signals'][name] = base._missing()
    return result


class _Session(base._NativeSession):
    def __init__(self, assets, config, case):
        super().__init__(assets, config)
        self.case = case

    def _task_configuration(self, task):
        return task.PiperTaskConfig(timeout_s=self.config.duration_s,
            object_position_m=self.case['object_position_m'],
            target_position_m=self.case['goal_position_m'])

    def _environment(self, task):
        from .piper_feedback_env import FeedbackPiperTaskEnv
        return FeedbackPiperTaskEnv(self.assets, self.task_config)

    def open(self):
        header, targets, metadata = super().open()
        rotation = self.env.data.xmat[self.base].reshape(3, 3)
        transform = self.np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = self.env.data.xpos[self.base]
        if (not self.np.isfinite(transform).all()
                or not self.np.allclose(rotation.T @ rotation, self.np.eye(3), atol=1e-8, rtol=0)
                or not math.isclose(float(self.np.linalg.det(rotation)), 1., abs_tol=1e-8, rel_tol=0)):
            raise ValueError('Native robot base transform is not finite and rigid.')
        metadata['world_from_robot_base'] = transform.reshape(16).tolist()
        metadata['world_from_robot_base_sha256'] = tape.digest(metadata['world_from_robot_base'])
        metadata['base_transform_scope'] = 'Fixed native world/base diagnostic transform; excluded from policy inputs.'
        return header, targets, metadata

    def measured(self, previous_camera):
        """50 Hz encoders/FK and the explicitly old 25 Hz camera frame."""
        seconds = float(self.env.elapsed)
        if not math.isclose(float(self.env.data.time)-self.native_origin, seconds,
                            abs_tol=tape.CLOCK_TOL, rel_tol=0):
            raise RuntimeError('Native and declared clocks differ.')
        rotation = self.env.data.xmat[self.base].reshape(3, 3)
        tool_rotation = rotation.T @ self.env.data.xmat[self.tool].reshape(3, 3)
        quaternion = self.np.empty(4)
        self.mj.mju_mat2Quat(quaternion, tool_rotation.reshape(9))
        values = dict(joint_position_rad=self.env.data.qpos[:6].tolist(),
            joint_velocity_rad_s=self.env.data.qvel[:6].tolist(),
            gripper_joint_position_m=self.env.data.qpos[6:8].tolist(),
            gripper_joint_velocity_m_s=self.env.data.qvel[6:8].tolist(),
            tcp_position_estimate_m=(rotation.T @ (self.env.data.xpos[self.tool]-self.env.data.xpos[self.base])).tolist(),
            tcp_quaternion_estimate_wxyz=quaternion.tolist())
        signals = {name: base._missing() for name in tape.SIGNALS}
        signals.update({name: base._sample(value, seconds) for name, value in values.items()})
        snapshot = dict(timestamp_seconds=seconds, signals=signals,
            camera=_sample_at(previous_camera, seconds), vision_frame_index=previous_camera['value']['index'])
        diagnostic = dict(scope='Privileged native diagnostics; excluded from policy inputs.',
            elapsed_seconds=seconds, native_data_time_seconds=float(self.env.data.time),
            native_info=deepcopy(self.info), policy_observation=deepcopy(self.observation))
        return snapshot, {}, diagnostic, dict(render_encode_wall_seconds=0.)

    def native_evidence(self):
        return self.env.native_arrays(), self.env.native_metadata()

    def anchor_error(self, estimate):
        if estimate.get('status') != 'available':
            return None
        position = self.env.data.xmat[self.base].reshape(3, 3).T @ (
            self.env.data.xpos[self.env.block]-self.env.data.xpos[self.base])
        return float(self.np.linalg.norm(position-estimate['object_position_estimate_m']))


def _initial_negative(snapshot, rgb, depth, calibration, negative):
    """Faults affect only the perception input; no hidden object pose changes."""
    local = deepcopy(calibration)
    captured = snapshot['camera']['captured_at_seconds']
    available = snapshot['camera']['available_at_seconds']
    if negative == 'initial_missing':
        rgb = rgb.copy(); rgb[:] = 0
    elif negative == 'initial_stale':
        captured = available = -.2
    elif negative == 'altered_calibration':
        local['base_from_camera'][0] *= 1.1
    return rgb, depth, local, captured, available


def _fresh_estimate(session, snapshot, calibration, negative=None):
    rgb, depth = session.latest_rgbd
    rgb, depth, actual, captured, available = _initial_negative(snapshot, rgb, depth, calibration, negative)
    began = time.perf_counter()
    estimate = tracker.estimate_pose(rgb, depth, actual,
        frame_index=snapshot['vision_frame_index'], captured_at_seconds=captured,
        available_at_seconds=available, decision_at_seconds=snapshot['timestamp_seconds'])
    return estimate, dict(estimator_wall_seconds=time.perf_counter()-began,
                         input_calibration_sha256=tape.digest(actual), negative_control=negative)


def _native_episode(session, states, actions):
    info = deepcopy(session.info)
    return dict(schema='dvidia.piper-task-episode/v1', config=asdict(session.task_config),
        initial_object_position_m=deepcopy(states[0]['object_position_m']) if states else None,
        trajectory=states, actions=actions, valid=bool(info.get('valid')),
        success=bool(info.get('success')), reason=info.get('reason'),
        metrics={**info, 'contact_pair_native_ticks': deepcopy(session.env.contact_pair_native_ticks)},
        physical_robot_ready=False, simulated_seconds=float(session.env.elapsed))


def _score(root, goal):
    import numpy as np
    from .piper_feedback_score import score_feedback_episode
    if not (root / 'native_episode.json').exists():
        return dict(valid=False, success=False, reason='native_not_initialized')
    episode = json.loads((root / 'native_episode.json').read_text())
    metadata = json.loads((root / 'native_metadata.json').read_text())
    with np.load(root / 'native.npz', allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    return score_feedback_episode(episode, arrays, metadata, goal[:2])


def run_rollout(assets, output, *, case, mode, policy=None, negative_control=None,
                duration_s=20., interrupt_after=None):
    """Run a new attempt; failures seal the exact complete prefix, never retry.

    ``policy(context)`` receives only ``snapshot``, ``estimate``,
    ``previous_applied`` and ``goal_position_m``. It returns the existing seven
    target action dictionary. Teacher queries follow the actual trajectory and
    are called once per boundary only in teacher/mixed modes.
    """
    case = _case(case)
    if mode not in MODES or negative_control not in NEGATIVES:
        raise ValueError('Unknown controller mode or negative control.')
    if mode != 'teacher' and not callable(policy):
        raise ValueError('Student, mixed and fixed modes require a policy callback.')
    config = RolloutConfig(duration_s=duration_s, seed=case['seed'], interrupt_after=interrupt_after)
    root = Path(output).absolute()
    if any(p.is_symlink() and str(p) not in ('/var', '/tmp') for p in (root, *root.parents)):
        raise ValueError('Output cannot traverse symlinks.')
    root.mkdir(parents=True, exist_ok=False)
    tracker_hash = sha256(Path(tracker.__file__).read_bytes()).hexdigest()
    base._json(root / 'started.json', dict(schema=SCHEMA, status='started', case=case,
        mode=mode, config=asdict(config), negative_control=negative_control, scope=SCOPE))
    session = _Session(assets, config, case)
    started = time.perf_counter()
    steps, states, actions, timings, estimates = [], [], [], [], []
    header = initial = metadata = initial_targets = None
    error = None; reason = 'capture_budget_complete'; started_intervals = 0
    frame_bytes = 0; camera_pairs = 0; anchor_available = False; anchor_error = None
    native_status = {}; native_saved = False; rng = random.Random(case['seed'])
    teacher_calls = student_calls = 0
    try:
        header, initial_targets, metadata = session.open()
        metadata = deepcopy(metadata)
        metadata.update(tracker_implemented=True,
            policy_observation_origin='teacher: privileged native state; student/fixed: whitelist of measured signals and RGB-D estimates',
            rendered_camera_cadence_seconds=.04, measured_control_cadence_seconds=.02)
        header = deepcopy(header); header['schema'] = SCHEMA
        header['provenance']['kind'] = 'simulation_'+mode+'_controller'
        header['provenance']['source_group'] = case['group_id']
        header['provenance']['case_id'] = case['id']
        header['goal_position_m'] = deepcopy(case['goal_position_m'])
        header['tracker_sha256'] = tracker_hash
        header['camera_cadence_seconds'] = .04
        header['control_mode'] = mode
        snapshot, frames, diagnostic, timing = session.sample(0)
        initial = deepcopy(snapshot)
        current_estimate, estimate_timing = _fresh_estimate(session, snapshot, header['calibration'], negative_control)
        timing.update(estimate_timing); timings.append(timing)
        anchor_available = current_estimate['status'] == 'available'
        anchor_error = session.anchor_error(current_estimate)
        for path, body in frames.items():
            frame_bytes += len(body); base._record(root / path, body)
        camera_pairs += 1
        if frame_bytes > tape.MAX_FRAME_TOTAL:
            raise ValueError('Actual unique frame bytes exceed the 512 MiB limit.')
        base._json(root / 'context.json', dict(header=header, initial=snapshot,
            initial_applied_targets=initial_targets, metadata=metadata, case=case))
        base._json(root / 'native/0000.json', diagnostic)
        states.append(deepcopy(diagnostic['policy_observation']))
        previous_applied = deepcopy(initial_targets)
        for index in range(round(duration_s / tape.DT)):
            if config.interrupt_after is not None and index >= config.interrupt_after:
                raise KeyboardInterrupt('Declared engineering interruption.')
            boundary = time.perf_counter()
            seconds = snapshot['timestamp_seconds']
            estimate = _estimate_at(current_estimate, seconds)
            perception = dict(schema=PERCEPTION_SCHEMA, index=index,
                decision_at_seconds=seconds, snapshot_sha256=tape.digest(snapshot),
                frame=deepcopy(snapshot['camera']['value']),
                calibration_sha256=header['binding']['calibration_sha256'],
                tracker_sha256=tracker_hash, estimate=estimate,
                input_calibration_sha256=estimate_timing['input_calibration_sha256'],
                negative_control=estimate_timing['negative_control'])
            perception_ref = _artifact(root, f'perception/{index:04}.json', perception)
            estimates.append(estimate)
            if index == 0 and not anchor_available and mode != 'teacher':
                reason = 'initial_visual_anchor_unavailable'
                break
            teacher_action = None; teacher_ref = None
            if mode in ('teacher', 'mixed'):
                teacher_calls += 1; teacher_action, teacher_decision = session.decide()
                tape._action(teacher_action, header, 'queried teacher targets')
                teacher_ref = _artifact(root, f'teacher/{index:04}.json', dict(
                    schema=TEACHER_SCHEMA, index=index, snapshot_sha256=tape.digest(snapshot),
                    computed_at_seconds=seconds, action=teacher_action,
                    origin='privileged_simulator_teacher', phase=teacher_decision.get('phase')))
            if mode == 'teacher':
                requested, authority = deepcopy(teacher_action), 'teacher'
            else:
                context = dict(snapshot=deepcopy(snapshot), estimate=deepcopy(estimate),
                    previous_applied=deepcopy(previous_applied), goal_position_m=deepcopy(case['goal_position_m']))
                inference = time.perf_counter(); student_calls += 1; student_action = policy(context)
                inference_wall = time.perf_counter()-inference
                tape._action(student_action, header, 'policy targets')
                if mode == 'mixed' and rng.random() < .5:
                    requested, authority = deepcopy(teacher_action), 'teacher'
                else:
                    requested, authority = deepcopy(student_action), ('fixed' if mode == 'fixed' else 'student')
            issued = deepcopy(requested)
            if negative_control == 'open_gripper':
                issued['gripper_position_m'] = .035
            tape._action(issued, header, 'issued targets')
            command = dict(issued_at_seconds=seconds, requested=requested, issued=issued,
                authority=authority, operator_correction=False,
                issue_reason='open_gripper_negative_control' if issued != requested else 'unchanged')
            base._json(root / f'decisions/{index:04}.json', dict(index=index, command=command,
                perception=perception_ref, teacher_query=teacher_ref))
            started_intervals += 1
            applied = session.advance(issued)
            next_index = index + 1
            if next_index % config.camera_every_controls == 0:
                after, frames, diagnostic, timing = session.sample(next_index // config.camera_every_controls)
                for path, body in frames.items():
                    frame_bytes += len(body); base._record(root / path, body)
                camera_pairs += 1
                if frame_bytes > tape.MAX_FRAME_TOTAL:
                    raise ValueError('Actual unique frame bytes exceed the 512 MiB limit.')
                current_estimate, estimate_timing = _fresh_estimate(session, after, header['calibration'])
                timing.update(estimate_timing)
            else:
                after, frames, diagnostic, timing = session.measured(snapshot['camera'])
            step = dict(index=index, before=snapshot, command=command, applied=applied,
                after=after, perception=perception_ref, teacher_query=teacher_ref)
            base._json(root / f'transitions/{index:04}.json', step)
            base._json(root / f'native/{next_index:04}.json', diagnostic)
            steps.append(step)
            action_row = dict(t_seconds=seconds, **issued, phase=authority)
            states[-1].update(issued_joint_targets_rad=issued['joint_targets_rad'],
                issued_gripper_position_m=issued['gripper_position_m'], command=action_row)
            actions.append(action_row); states.append(deepcopy(diagnostic['policy_observation']))
            timing['full_boundary_wall_seconds'] = time.perf_counter()-boundary
            timing['policy_inference_wall_seconds'] = 0. if mode == 'teacher' else inference_wall
            timings.append(timing)
            snapshot = after; previous_applied = deepcopy(applied['ramped_targets'])
    except (Exception, KeyboardInterrupt) as exc:
        reason = 'producer_interrupted' if isinstance(exc, KeyboardInterrupt) else 'producer_error'
        error = dict(type=type(exc).__name__, message=str(exc)[:500])
    finally:
        try:
            native_status = session.status()
            if header is not None:
                import numpy as np
                arrays, native_metadata = session.native_evidence()
                buffer = io.BytesIO(); np.savez_compressed(buffer, **arrays)
                base._record(root / 'native.npz', buffer.getvalue())
                base._json(root / 'native_metadata.json', native_metadata)
                base._json(root / 'native_episode.json', _native_episode(session, states, actions))
                native_saved = True
        except Exception as exc:
            if error is None:
                error = dict(type=type(exc).__name__, message=str(exc)[:500]); reason = 'native_evidence_error'
        try:
            session.close()
        except Exception as exc:
            if error is None:
                error = dict(type=type(exc).__name__, message=str(exc)[:500]); reason = 'renderer_close_error'
    complete = error is None and len(steps) == round(duration_s / tape.DT)
    status = 'complete' if complete else 'interrupted' if steps else 'refused'
    if header is not None:
        recording = {**header, 'initial': initial, 'initial_applied_targets': initial_targets,
            'steps': steps, 'outcome': dict(status=status, reason=reason,
                end_seconds=steps[-1]['after']['timestamp_seconds'] if steps else 0.,
                task_result='unverified' if complete else 'unknown')}
        base._json(root / 'recording.json', recording)
    score = _score(root, case['goal_position_m']) if native_saved else dict(valid=False, success=False, reason='native_evidence_unavailable')
    missing = [estimate['status'] != 'available' for estimate in estimates]
    longest = streak = 0
    for absent in missing:
        streak = streak + 1 if absent else 0; longest = max(longest, streak)
    critical_ticks = (native_status.get('native_info', {}).get('unexpected_contact_native_ticks')
                      if native_status.get('native_info') else None)
    report = dict(schema=SCHEMA, status=status, reason=reason, error=error, case=case, mode=mode,
        scope=SCOPE, physical_robot_ready=False, negative_control=negative_control,
        requested_intervals=round(duration_s / tape.DT), started_intervals=started_intervals,
        recorded_complete_intervals=len(steps), uncommitted_native_intervals=started_intervals-len(steps),
        camera_pairs=camera_pairs, frame_bytes=frame_bytes,
        initial_anchor_available=anchor_available, initial_anchor_error_m=anchor_error,
        initial_anchor_binding_valid=bool(header is not None and anchor_available and negative_control not in ('initial_stale','altered_calibration')),
        teacher_queries=teacher_calls, student_calls=student_calls,
        no_privileged_features_or_teacher_intervention=bool(mode in ('student', 'fixed') and teacher_calls == 0),
        policy_boundary_scope='Whitelisted deep-copied callback inputs; no process sandbox or external callback code attestation.',
        score=score, strict_success=bool(complete and score.get('success')),
        critical_contacts=critical_ticks, critical_contact_ticks=critical_ticks,
        native_status=native_status, tracker_sha256=tracker_hash,
        perception=dict(decisions=len(estimates), available=sum(not value for value in missing),
            missing=sum(missing), longest_missing_seconds=longest*tape.DT),
        timing=dict(wall_seconds=time.perf_counter()-started, boundaries=timings,
            scope='Offline paused-simulator acquisition and computation. Synthetic capture/availability times; no physical transport or end-to-end real-time latency.'))
    base._json(root / 'rollout.json', report)
    files = {str(path.relative_to(root)): _identity(path) for path in sorted(root.rglob('*')) if path.is_file()}
    base._json(root / 'integrity.json', dict(schema=SCHEMA+'/integrity', files=files))
    return report


def _read(root, relative, maximum=MAX_ROLLOUT_BYTES):
    return json.loads(tape._ordinary_bytes(root / str(tape._relative(relative)), maximum))


def inspect_rollout(directory):
    """Read-only byte, source-frame, authority and independent-score checks."""
    root = Path(directory)
    manifest = _read(root, 'integrity.json')
    if manifest.get('schema') != SCHEMA+'/integrity' or type(manifest.get('files')) is not dict:
        raise ValueError('Unsealed or unknown rollout manifest.')
    observed = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name != 'integrity.json'}
    if observed != set(manifest['files']):
        raise ValueError('Rollout inventory differs.')
    for name, expected in manifest['files'].items():
        body = tape._ordinary_bytes(root / str(tape._relative(name)), MAX_ROLLOUT_BYTES)
        if expected != dict(bytes=len(body), sha256=sha256(body).hexdigest()):
            raise ValueError('Rollout byte identity differs: '+name)
    report = _read(root, 'rollout.json')
    if report.get('schema') != SCHEMA or report.get('status') not in ('complete', 'interrupted', 'refused'):
        raise ValueError('Unknown rollout receipt.')
    if 'recording.json' in manifest['files']:
        recording = _read(root, 'recording.json')
        if recording.get('schema') != SCHEMA or len(recording.get('steps', [])) != report['recorded_complete_intervals']:
            raise ValueError('Recording schema or completeness differs.')
        previous = recording['initial']
        native_episode = _read(root, 'native_episode.json') if 'native_episode.json' in manifest['files'] else None
        native_arrays = None
        if native_episode is not None and 'actions' in native_episode:
            import numpy as np
            with np.load(root / 'native.npz', allow_pickle=False) as archive:
                native_arrays = {name: archive[name] for name in ('arm_actuator_controls_rad',
                    'applied_jaw_target_m', 'normal_rate_limited_jaw_target_m')}
        tape._snapshot(previous, recording['calibration'])
        if tape.digest(recording['calibration']) != recording['binding']['calibration_sha256']:
            raise ValueError('Bound calibration differs.')
        for index, step in enumerate(recording['steps']):
            if step['index'] != index or step['before'] != previous:
                raise ValueError('Recording transition prefix differs.')
            seconds = index*tape.DT
            if not math.isclose(step['before']['timestamp_seconds'], seconds, abs_tol=tape.CLOCK_TOL, rel_tol=0):
                raise ValueError('Decision cadence differs.')
            if not math.isclose(step['after']['timestamp_seconds'], seconds+tape.DT, abs_tol=tape.CLOCK_TOL, rel_tol=0):
                raise ValueError('Applied interval is incomplete.')
            tape._snapshot(step['after'], recording['calibration'], previous)
            if any(step['before']['signals'][name]['status'] != 'missing' for name in tracker.SIGNALS):
                raise ValueError('Raw sensor snapshots cannot substitute tracker or native object truth.')
            frame = step['before']['camera']['value']
            if frame['index'] != index // 2:
                raise ValueError('Camera cadence differs from 25 Hz.')
            for channel in ('rgb', 'depth'):
                reference = frame[channel]
                if manifest['files'].get(reference['path']) != {key: reference[key] for key in ('bytes', 'sha256')}:
                    raise ValueError('Camera source byte binding differs.')
            for field in ('requested', 'issued'):
                tape._action(step['command'][field], recording, field)
            expected_issued = deepcopy(step['command']['requested'])
            if report['negative_control'] == 'open_gripper':
                expected_issued['gripper_position_m'] = .035
            if step['command']['issued'] != expected_issued:
                raise ValueError('Actual issued command differs from declared intervention.')
            if native_episode is not None and 'actions' in native_episode:
                actual = native_episode['actions'][index]
                if ({name: actual[name] for name in ('joint_targets_rad', 'gripper_position_m')}
                        != step['command']['issued']):
                    raise ValueError('Issued command differs from native action evidence.')
                native_state = native_episode['trajectory'][index]
                for signal in ('joint_position_rad', 'joint_velocity_rad_s', 'gripper_joint_position_m'):
                    if step['before']['signals'][signal]['value'] != native_state[signal]:
                        raise ValueError('Measured inputs differ from native encoder evidence.')
                boundary = (index+1)*20-1
                controls = [*native_arrays['arm_actuator_controls_rad'][boundary].tolist(),
                    float(native_arrays['applied_jaw_target_m'][boundary])]
                ramped = dict(joint_targets_rad=controls[:6],
                    gripper_position_m=float(native_arrays['normal_rate_limited_jaw_target_m'][boundary]))
                if step['applied']['native_position_controls'] != controls or step['applied']['ramped_targets'] != ramped:
                    raise ValueError('Previous-applied learning inputs differ from native servo references.')
            for field in ('perception', 'teacher_query'):
                reference = step[field]
                if reference is None:
                    if field == 'perception' or report['mode'] in ('teacher', 'mixed'):
                        raise ValueError('Required sidecar missing.')
                    continue
                identity = manifest['files'].get(reference['path'])
                if identity != {key: reference[key] for key in ('bytes', 'sha256')}:
                    raise ValueError('Sidecar binding differs.')
                sidecar = _read(root, reference['path'])
                if sidecar['index'] != index or sidecar['snapshot_sha256'] != tape.digest(step['before']):
                    raise ValueError('Sidecar source snapshot differs.')
                if field == 'perception':
                    if (sidecar['tracker_sha256'] != recording['tracker_sha256']
                            or sidecar['calibration_sha256'] != recording['binding']['calibration_sha256']
                            or sidecar['frame'] != step['before']['camera']['value']):
                        raise ValueError('Perception frame/tracker/calibration binding differs.')
                    if sidecar['estimate']['decision_at_seconds'] != step['before']['timestamp_seconds']:
                        raise ValueError('Perception decision time differs.')
                    estimate = sidecar['estimate']
                    if estimate['frame_index'] != frame['index']:
                        raise ValueError('Estimated source frame index differs.')
                    if estimate['status'] == 'available':
                        camera = step['before']['camera']
                        if (estimate['captured_at_seconds'] != camera['captured_at_seconds']
                                or estimate['available_at_seconds'] != camera['available_at_seconds']
                                or estimate['age_seconds'] != camera['age_seconds']):
                            raise ValueError('Estimated source frame acquisition timing differs.')
                        for name in tracker.SIGNALS:
                            signal = estimate['signals'][name]
                            if (signal['status'] != 'available' or any(signal[key] != estimate[key] for key in
                                    ('captured_at_seconds', 'available_at_seconds', 'age_seconds'))):
                                raise ValueError('Estimated source signal timing differs.')
                            expected = estimate[name]
                            if name.endswith('_uncertainty_m') or name.endswith('_uncertainty_rad'):
                                expected = [expected]
                            if signal['value'] != expected:
                                raise ValueError('Estimated source signal value differs.')
                    elif any(estimate[name] is not None or estimate['signals'][name]['value'] is not None
                             for name in tracker.SIGNALS):
                        raise ValueError('Missing estimates must not carry substituted values.')
                elif step['command']['authority'] == 'teacher' and sidecar['action'] != step['command']['requested']:
                    raise ValueError('Teacher authority differs from its query.')
            authority = step['command']['authority']
            allowed = {'teacher': {'teacher'}, 'student': {'student'}, 'mixed': {'teacher', 'student'}, 'fixed': {'fixed'}}[report['mode']]
            if authority not in allowed or step['command']['operator_correction'] is not False:
                raise ValueError('Invalid control authority.')
            previous = step['after']
        if report['status'] == 'complete' and len(recording['steps']) != report['requested_intervals']:
            raise ValueError('Complete receipt omits scheduled intervals.')
    recomputed = _score(root, report['case']['goal_position_m']) if 'native_episode.json' in manifest['files'] else dict(valid=False, success=False, reason='native_evidence_unavailable')
    if recomputed != report['score']:
        raise ValueError('Independent task score differs.')
    if report['strict_success'] != bool(report['status'] == 'complete' and recomputed.get('success')):
        raise ValueError('Reported strict success differs from complete native scoring.')
    return dict(schema=SCHEMA+'/inspection', status=report['status'], files_verified=len(manifest['files']),
        source_authenticated=False, physics_executed=False, physical_robot_ready=False,
        score=recomputed, strict_success=report['strict_success'],
        recorded_complete_intervals=report['recorded_complete_intervals'])


def load_rollout(directory):
    inspect_rollout(directory)
    return _read(Path(directory), 'recording.json')

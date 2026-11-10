"""Stateful external-command recording for the bounded native PiPER task.

The public ``session`` is available for scene/model display binding only. The
caller supplies every seven-target command; this module never calls a teacher.
Human input is a producer assertion, not authenticated human provenance.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import io
import math
import os
from pathlib import Path
import time

from . import action_demonstration as tape
from . import camera_action_capture as camera
from . import piper_action_capture as base
from . import piper_rgbd_tracker as tracker

SCHEMA = 'dvidia.operator-capture/v1'
PERCEPTION_SCHEMA = SCHEMA + '/perception'
JOURNAL_SCHEMA = SCHEMA + '/journal'
SOURCE_KINDS = ('simulation_human_input', 'simulation_scripted_input')
MAX_DOCUMENT_BYTES = 64 * 1024 * 1024
MAX_JOURNAL_BYTES = 16 * 1024 * 1024
MAX_EVENTS = 10000
SOURCE_FILES = ('operator_capture.py', 'camera_action_capture.py', 'piper_action_capture.py',
                'action_demonstration.py', 'piper_rgbd_tracker.py', 'piper_feedback_env.py',
                'piper_feedback_score.py', 'piper_task.py', 'piper_imitation_score.py',
                'piper_encoder_kinematics.py')
SCOPE = ('Native simulation with externally issued targets, 1 ms physics, '
         '50 Hz controls and native 25 Hz 320x240 RGB-D. Input provenance is '
         'a producer assertion; no human authentication or hardware qualification.')


def _read(root, name, maximum=MAX_DOCUMENT_BYTES):
    return tape._decode(tape._ordinary_bytes(root / str(tape._relative(name)), maximum))


def _reference(root, name, document):
    result = base._json(root / name, document)
    result['path'] = name
    return result


def _identity(root, name):
    body = tape._ordinary_bytes(root / str(tape._relative(name)), MAX_DOCUMENT_BYTES)
    return dict(bytes=len(body), sha256=sha256(body).hexdigest())


def _artifact_name(root, path):
    """Serialize local inventory paths using the platform-independent tape contract."""
    name = path.relative_to(root).as_posix()
    tape._relative(name)
    return name


def _detail(value):
    if value is not None and type(value) is not dict:
        raise ValueError('Input/event metadata must be a JSON object or null.')
    body = tape.canonical(value)
    if len(body) > 4096:
        raise ValueError('Input/event metadata exceeds 4096 bytes.')
    return tape._decode(body)


def _header(header):
    if (header.get('schema') != SCHEMA or header.get('control_mode') != 'operator'
            or header.get('provenance', {}).get('kind') not in SOURCE_KINDS
            or header.get('source_authenticated') is not False
            or header.get('human_authenticated') is not False):
        raise ValueError('Invalid operator source provenance.')
    # Validate unchanged native contracts using the strict v1 header validator.
    # This local adapter is never serialized or used as a supervision label.
    native = {name: deepcopy(header[name]) for name in tape.HEADER}
    native['schema'] = tape.SCHEMA
    native['provenance'].pop('case_id', None)
    native['provenance']['kind'] = 'simulation_teleoperation'
    tape._header(native)
    if header.get('camera_cadence_seconds') != .04:
        raise ValueError('Operator camera cadence must be 25 Hz.')
    tape._sha(header.get('tracker_sha256'), 'tracker hash')
    sources = header.get('operator_implementation_sha256')
    if type(sources) is not dict or set(sources) != set(SOURCE_FILES):
        raise ValueError('Operator implementation source identities are required.')
    for name, value in sources.items():
        tape._sha(value, name)
    if (header['provenance']['producer_revision'] != sources['operator_capture.py']
            or header['tracker_sha256'] != sources['piper_rgbd_tracker.py']):
        raise ValueError('Operator/tracker source identity differs from implementation receipt.')


def _snapshot(snapshot, header, index, previous=None):
    tape._snapshot(snapshot, header['calibration'], previous)
    if not math.isclose(snapshot['timestamp_seconds'], index*tape.DT,
                        abs_tol=tape.CLOCK_TOL, rel_tol=0):
        raise ValueError('Operator boundary must be a complete 20 ms interval.')
    if any(snapshot['signals'][name]['status'] != 'missing' for name in tracker.SIGNALS):
        raise ValueError('Raw snapshots cannot contain privileged object poses or tracker substitutions.')
    frame = snapshot['camera']['value']
    if frame is None or frame['index'] != index//2:
        raise ValueError('Camera frame cadence differs from 25 Hz.')
    if not math.isclose(snapshot['camera']['captured_at_seconds'], (index//2)*.04,
                        abs_tol=tape.CLOCK_TOL, rel_tol=0):
        raise ValueError('Camera acquisition time differs from native cadence.')


def _command(command, header, snapshot):
    tape._keys(command, {'issued_at_seconds', 'requested', 'issued', 'authority',
                        'operator_correction', 'issue_reason'}, 'operator command')
    if (command['authority'] != header['provenance']['kind']
            or command['operator_correction'] is not False or command['issue_reason'] != 'unchanged'
            or command['requested'] != command['issued']
            or command['issued_at_seconds'] != snapshot['timestamp_seconds']):
        raise ValueError('Operator issued command authority differs.')
    tape._action(command['issued'], header, 'issued command')


def _perception(root, reference, snapshot, header, index):
    expected = {key: reference[key] for key in ('bytes', 'sha256')}
    if _identity(root, reference['path']) != expected:
        raise ValueError('Perception sidecar byte identity differs.')
    item = _read(root, reference['path'])
    if (item.get('schema') != PERCEPTION_SCHEMA or item.get('index') != index
            or item.get('snapshot_sha256') != tape.digest(snapshot)
            or item.get('frame') != snapshot['camera']['value']
            or item.get('tracker_sha256') != header['tracker_sha256']
            or item.get('calibration_sha256') != header['binding']['calibration_sha256']
            or item.get('input_calibration_sha256') != header['binding']['calibration_sha256']):
        raise ValueError('Perception source snapshot/frame/calibration binding differs.')
    estimate = item['estimate']; sensed = snapshot['camera']
    if (estimate.get('decision_at_seconds') != snapshot['timestamp_seconds']
            or estimate.get('frame_index') != snapshot['vision_frame_index']
            or estimate.get('status') not in ('available', 'missing', 'stale', 'invalid', 'refused')):
        raise ValueError('Perception decision time or source frame differs.')
    if estimate['status'] == 'available':
        for field in ('captured_at_seconds', 'available_at_seconds', 'age_seconds'):
            if estimate.get(field) != sensed[field]:
                raise ValueError('Perception acquisition timing differs.')
        for name in tracker.SIGNALS:
            signal = estimate['signals'][name]
            tape._availability(signal, snapshot['timestamp_seconds'], name)
            expected_value = estimate[name]
            if 'uncertainty' in name:
                expected_value = [expected_value]
            if (signal['status'] != 'available' or signal['value'] != expected_value
                    or any(signal[field] != estimate[field] for field in
                           ('captured_at_seconds', 'available_at_seconds', 'age_seconds'))):
                raise ValueError('Perception signal value or timing differs.')
            tape._vec(signal['value'], tape.SIGNALS[name]['size'], name)
            if 'uncertainty' in name and signal['value'][0] < 0:
                raise ValueError('Perception uncertainty cannot be negative.')
            if 'quaternion' in name and not math.isclose(math.hypot(*signal['value']), 1., abs_tol=1e-6):
                raise ValueError('Perception quaternion must be normalized.')
    elif any(estimate.get(name) is not None or estimate.get('signals', {}).get(name) != base._missing()
             for name in tracker.SIGNALS):
        raise ValueError('Unavailable perception must preserve explicit missing masks.')
    return item


class OperatorCapture:
    """One exclusive attempt. ``step`` commits exactly one caller-issued interval.

    ``context`` returns only sensor/perception data, preceding applied targets,
    and the declared goal. ``session`` is privileged display access and is not
    part of that context. Call ``finish`` on stop/disconnect, including errors.
    A failed step seals its committed prefix and re-raises; it never retries.
    """
    def __init__(self, assets, output, *, case, duration_s=20.,
                 source_kind='simulation_scripted_input'):
        self.case = camera._case(case)
        if source_kind not in SOURCE_KINDS:
            raise ValueError('Choose simulation_human_input or simulation_scripted_input.')
        self.config = camera.RolloutConfig(duration_s=duration_s, seed=self.case['seed'])
        self.root = Path(output).absolute()
        if any(item.is_symlink() and str(item) not in ('/var', '/tmp')
               for item in (self.root, *self.root.parents)):
            raise ValueError('Output cannot traverse symlinks.')
        self.source_kind = source_kind
        self.session = camera._Session(assets, self.config, self.case)
        self.header = self.initial = self.snapshot = self.estimate = None
        self.previous_applied = self.initial_targets = None
        self.steps, self.states, self.actions = [], [], []
        self.started_intervals = self.frame_bytes = self.camera_pairs = 0
        self._journal = None; self._sequence = 0; self._last_hash = None
        self._opened = self._sealed = False; self._report = None
        self._started_wall = time.monotonic(); self._error = None
        self._estimate_timing = {}; self._timings = []

    @property
    def done(self):
        return self._sealed or len(self.steps) >= round(self.config.duration_s/tape.DT)

    def _append(self, kind, **payload):
        # Reserve one bounded row so event flooding can still seal a refusal.
        if self._sequence >= MAX_EVENTS-int(kind != 'finished'):
            raise ValueError('Operator journal exceeds its bounded event count.')
        record = dict(schema=JOURNAL_SCHEMA, sequence=self._sequence,
            previous_sha256=self._last_hash, kind=kind,
            simulation_seconds=self.snapshot['timestamp_seconds'] if self.snapshot else 0.,
            monotonic_wall_seconds=time.monotonic()-self._started_wall, **payload)
        body = tape.canonical(record)+b'\n'
        if self._journal.tell()+len(body) > MAX_JOURNAL_BYTES-(1024 if kind != 'finished' else 0):
            raise ValueError('Operator journal exceeds its byte budget.')
        self._journal.write(body); self._journal.flush(); os.fsync(self._journal.fileno())
        self._sequence += 1; self._last_hash = sha256(body).hexdigest()
        return record

    def record_event(self, name, detail=None):
        if not self._opened or self._sealed:
            raise RuntimeError('Events require an open, unfinished capture.')
        if type(name) is not str or not name or len(name) > 80:
            raise ValueError('Event name must contain 1 to 80 characters.')
        return deepcopy(self._append('event', name=name, detail=_detail(detail)))

    def _save_frames(self, frames):
        additional = sum(len(body) for body in frames.values())
        if self.frame_bytes+additional > tape.MAX_FRAME_TOTAL:
            raise ValueError('Unique native RGB-D frames exceed 512 MiB.')
        for name, body in frames.items():
            tape._relative(name)
            if not name.startswith('frames/') or len(body) > tape.MAX_FRAME_BYTES:
                raise ValueError('Native frame path or size is invalid.')
            base._record(self.root/name, body); self.frame_bytes += len(body)
        if frames:
            self.camera_pairs += 1

    def _save_perception(self, index):
        estimate = camera._estimate_at(self.estimate, self.snapshot['timestamp_seconds'])
        return _reference(self.root, f'perception/{index:04}.json', dict(
            schema=PERCEPTION_SCHEMA, index=index, snapshot_sha256=tape.digest(self.snapshot),
            frame=deepcopy(self.snapshot['camera']['value']), estimate=estimate,
            tracker_sha256=self.header['tracker_sha256'],
            calibration_sha256=self.header['binding']['calibration_sha256'],
            input_calibration_sha256=self._estimate_timing['input_calibration_sha256']))

    def open(self):
        if self._opened or self._sealed:
            raise RuntimeError('A capture can only be opened once.')
        self.root.mkdir(parents=True, exist_ok=False)
        base._json(self.root/'started.json', dict(schema=SCHEMA, status='started',
            case=self.case, config=asdict(self.config), source_kind=self.source_kind, scope=SCOPE))
        self._journal = (self.root/'commands.jsonl').open('xb')
        self._opened = True
        try:
            native_header, targets, metadata = self.session.open()
            tape._header(native_header)
            self.header = deepcopy(native_header)
            self.header.update(schema=SCHEMA, control_mode='operator', camera_cadence_seconds=.04,
                source_authenticated=False, human_authenticated=False,
                tracker_sha256=sha256(Path(tracker.__file__).read_bytes()).hexdigest())
            self.header['provenance'].update(kind=self.source_kind, source_group=self.case['group_id'],
                                             case_id=self.case['id'])
            source_hashes = {name: sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                             for name in SOURCE_FILES}
            self.header['provenance']['producer_revision'] = source_hashes['operator_capture.py']
            self.header['operator_implementation_sha256'] = source_hashes
            metadata = deepcopy(metadata)
            metadata.update(operator_implementation_sha256=source_hashes,
                tracker_implemented=True, operator_corrections_collected=False,
                policy_observation_origin='Privileged diagnostics only; excluded from operator context.',
                command_origin='External input supplied to OperatorCapture.step; no teacher queries.')
            self.header['goal_position_m'] = deepcopy(self.case['goal_position_m'])
            self.initial_targets = self.previous_applied = deepcopy(targets)
            self.snapshot, frames, diagnostic, timing = self.session.sample(0)
            _snapshot(self.snapshot, self.header, 0)
            self.initial = deepcopy(self.snapshot)
            self.estimate, self._estimate_timing = camera._fresh_estimate(
                self.session, self.snapshot, self.header['calibration'])
            self._save_frames(frames)
            self._perception_ref = self._save_perception(0)
            _perception(self.root, self._perception_ref, self.snapshot, self.header, 0)
            base._json(self.root/'native/0000.json', diagnostic)
            self.states.append(deepcopy(diagnostic['policy_observation']))
            base._json(self.root/'context.json', dict(header=self.header, initial=self.initial,
                initial_applied_targets=self.initial_targets, metadata=metadata, case=self.case,
                initial_perception=self._perception_ref))
            self._timings.append({**timing, **self._estimate_timing})
            self.record_event('opened')
            return dict(context=self.context(), header=deepcopy(self.header),
                        snapshot=deepcopy(self.snapshot), estimate=deepcopy(self.estimate))
        except (Exception, KeyboardInterrupt) as exc:
            self._error = dict(type=type(exc).__name__, message=str(exc)[:500])
            self.finish('open_error')
            raise

    def context(self):
        if self.snapshot is None or self.estimate is None:
            raise RuntimeError('Capture has no initialized sensor boundary.')
        return dict(snapshot=deepcopy(self.snapshot),
            estimate=camera._estimate_at(self.estimate, self.snapshot['timestamp_seconds']),
            previous_applied=deepcopy(self.previous_applied),
            goal_position_m=deepcopy(self.case['goal_position_m']))

    def step(self, action, *, input_event=None):
        if not self._opened or self.done:
            raise RuntimeError('Capture is not open or has reached its interval budget.')
        tape._action(action, self.header, 'operator-issued targets')
        action = deepcopy(action); input_event = _detail(input_event)
        index = len(self.steps); before = deepcopy(self.snapshot)
        command = dict(issued_at_seconds=before['timestamp_seconds'], requested=deepcopy(action),
            issued=deepcopy(action), authority=self.source_kind, operator_correction=False,
            issue_reason='unchanged')
        try:
            decision = dict(index=index, command=command, perception=self._perception_ref,
                            input_event=input_event, snapshot_sha256=tape.digest(before))
            decision_ref = _reference(self.root, f'decisions/{index:04}.json', decision)
            self._append('intent', index=index, decision=decision_ref)
            self.started_intervals += 1
            applied = self.session.advance(action)
            next_index = index+1
            if next_index % 2 == 0:
                after, frames, diagnostic, timing = self.session.sample(next_index//2)
                estimate, estimate_timing = camera._fresh_estimate(
                    self.session, after, self.header['calibration'])
            else:
                after, frames, diagnostic, timing = self.session.measured(before['camera'])
                estimate, estimate_timing = self.estimate, self._estimate_timing
            _snapshot(after, self.header, next_index, before)
            self._save_frames(frames)
            tape._action(applied['ramped_targets'], self.header, 'applied targets')
            if (applied['interval_start_seconds'] != before['timestamp_seconds']
                    or applied['interval_end_seconds'] != after['timestamp_seconds']):
                raise ValueError('Native applied interval differs from the observed 20 ms boundary.')
            step = dict(index=index, before=before, command=command, applied=deepcopy(applied),
                after=deepcopy(after), perception=deepcopy(self._perception_ref),
                input_event=input_event, teacher_query=None)
            native_ref = _reference(self.root, f'native/{next_index:04}.json', diagnostic)
            transition_ref = _reference(self.root, f'transitions/{index:04}.json', step)
            # Outcome is the commit marker. A transition without this marker is
            # retained as evidence, but is never included in a recovered prefix.
            self._append('outcome', index=index, status='committed',
                         transition=transition_ref, native=native_ref)
            self.steps.append(step)
            action_row = dict(t_seconds=before['timestamp_seconds'], **action, phase=self.source_kind)
            self.states[-1].update(issued_joint_targets_rad=action['joint_targets_rad'],
                issued_gripper_position_m=action['gripper_position_m'], command=action_row)
            self.actions.append(action_row); self.states.append(deepcopy(diagnostic['policy_observation']))
            self.snapshot = deepcopy(after); self.estimate = estimate; self._estimate_timing = estimate_timing
            self.previous_applied = deepcopy(applied['ramped_targets'])
            self._timings.append(timing)
            self._perception_ref = self._save_perception(next_index)
            return dict(snapshot=deepcopy(self.snapshot), estimate=self.context()['estimate'],
                        applied=deepcopy(applied), done=self.done, context=self.context())
        except (Exception, KeyboardInterrupt) as exc:
            self._error = dict(type=type(exc).__name__, message=str(exc)[:500])
            self.finish('producer_interrupted' if isinstance(exc, KeyboardInterrupt) else 'producer_error')
            raise

    def _reconcile(self):
        """Rebuild counters and native control trace from durable commit markers.

        An interrupt can occur after fsync and before *any* Python assignment,
        including the journal sequence/hash cursor. Memory is never the source
        of truth for finalization.
        """
        self._journal.flush(); os.fsync(self._journal.fileno())
        _, context, steps, intents, _, finished = _prefix(self.root)
        entries, _ = _journal(self.root)
        body = tape._ordinary_bytes(self.root/'commands.jsonl', MAX_JOURNAL_BYTES)
        self._sequence = len(entries)
        self._last_hash = sha256(body.splitlines(keepends=True)[-1]).hexdigest() if entries else None
        self._journal.seek(0, os.SEEK_END)
        self.started_intervals = len(intents)
        self.steps = deepcopy(steps)
        self.states, self.actions = [], []
        if context is None:
            # A sampled frame in memory does not establish an initialized
            # recording. Without context there is no bound native initial state.
            self.initial = self.snapshot = self.estimate = None
            self.initial_targets = self.previous_applied = None
        else:
            self.header = deepcopy(context['header']); self.initial = deepcopy(context['initial'])
            self.initial_targets = deepcopy(context['initial_applied_targets'])
            self.states.append(deepcopy(_read(self.root, 'native/0000.json')['policy_observation']))
            outcomes = [entry for entry in entries if entry['kind'] == 'outcome']
            for step, outcome in zip(steps, outcomes):
                action = deepcopy(step['command']['issued'])
                row = dict(t_seconds=step['before']['timestamp_seconds'], **action, phase=self.source_kind)
                self.states[-1].update(issued_joint_targets_rad=action['joint_targets_rad'],
                    issued_gripper_position_m=action['gripper_position_m'], command=row)
                self.actions.append(row)
                self.states.append(deepcopy(_bound_document(self.root, outcome['native'])['policy_observation']))
            self.snapshot = deepcopy(steps[-1]['after'] if steps else self.initial)
            self.previous_applied = deepcopy(steps[-1]['applied']['ramped_targets'] if steps else self.initial_targets)
            perception_path = f'perception/{len(steps):04}.json'
            if (self.root/perception_path).exists():
                self.estimate = _read(self.root, perception_path)['estimate']
            else:
                sensed = self.snapshot['camera']
                self.estimate = tracker._empty(self.snapshot['vision_frame_index'],
                    sensed['captured_at_seconds'], sensed['available_at_seconds'], self.snapshot['timestamp_seconds'])
                self.estimate['reason'] = 'terminal_perception_not_committed'
        frames = [path for path in (self.root/'frames').glob('*') if path.is_file()]
        self.frame_bytes = sum(path.stat().st_size for path in frames)
        rgb = {path.name.removeprefix('rgb-').split('.',1)[0] for path in frames if path.name.startswith('rgb-')}
        depth = {path.name.removeprefix('depth-').split('.',1)[0] for path in frames if path.name.startswith('depth-')}
        self.camera_pairs = len(rgb & depth)
        return finished

    def _leave_unsealed(self, exc, reason):
        """Close resources and retain an explicit recovery requirement on disk."""
        failure = dict(type=type(exc).__name__, message=str(exc)[:500], stage='finish_reconciliation')
        try:
            self.session.close()
        except (Exception, KeyboardInterrupt) as close_error:
            failure['renderer_close_error'] = str(close_error)[:500]
        try:
            self._journal.close()
        except (Exception, KeyboardInterrupt) as close_error:
            failure['journal_close_error'] = str(close_error)[:500]
        report = dict(schema=SCHEMA, status='interrupted', sealed=False,
            reason='journal_reconciliation_failed', requested_finish_reason=reason,
            error=self._error, finish_error=failure, source_kind=self.source_kind, case=self.case,
            recorded_complete_intervals=None, started_intervals=None,
            uncommitted_native_intervals=None, recovery_required=True,
            source_authenticated=False, human_authenticated=False, physical_robot_ready=False,
            teacher_queries=0, score=_no_score(), strict_success=False)
        try:
            base._json(self.root/'finish_error.json', report)
        except (Exception, KeyboardInterrupt) as write_error:
            report['finish_error']['receipt_write_error'] = str(write_error)[:500]
        self._sealed = True; self._report = report
        return deepcopy(report)

    def finish(self, reason='operator_stopped'):
        if self._sealed:
            return deepcopy(self._report)
        if not self._opened:
            self.session.close()
            self._report = dict(schema=SCHEMA, status='refused', sealed=False,
                reason='capture_not_opened', source_kind=self.source_kind, case=self.case,
                requested_intervals=round(self.config.duration_s/tape.DT), started_intervals=0,
                recorded_complete_intervals=0, uncommitted_native_intervals=0,
                source_authenticated=False, human_authenticated=False, physical_robot_ready=False,
                teacher_queries=0, score=_no_score(), strict_success=False)
            self._sealed = True
            return deepcopy(self._report)
        if type(reason) is not str or not reason or len(reason) > 200:
            raise ValueError('Finish reason must contain 1 to 200 characters.')
        try:
            already_finished = self._reconcile()
        except (Exception, KeyboardInterrupt) as exc:
            # Never seal an internally inconsistent journal/recording pair.
            # Recovery remains read-only and may inspect a valid flushed prefix.
            return self._leave_unsealed(exc, reason)
        native_status = {}; native_saved = False
        try:
            native_status = self.session.status()
            if self.initial is not None:
                import numpy as np
                arrays, metadata = self.session.native_evidence()
                buffer = io.BytesIO(); np.savez_compressed(buffer, **arrays)
                base._record(self.root/'native.npz', buffer.getvalue())
                base._json(self.root/'native_metadata.json', metadata)
                base._json(self.root/'native_episode.json', camera._native_episode(self.session, self.states, self.actions))
                native_saved = True
        except Exception as exc:
            self._error = self._error or dict(type=type(exc).__name__, message=str(exc)[:500])
            reason = 'native_evidence_error' if self._error['message'] == str(exc)[:500] else reason
        finally:
            try:
                self.session.close()
            except Exception as exc:
                self._error = self._error or dict(type=type(exc).__name__, message=str(exc)[:500])
        initialized = (self.root/'context.json').exists()
        complete = initialized and self._error is None and len(self.steps) == round(self.config.duration_s/tape.DT)
        status = 'complete' if complete else 'interrupted' if initialized else 'refused'
        if complete:
            reason = 'capture_budget_complete'
        try:
            score = camera._score(self.root, self.case['goal_position_m']) if native_saved else _no_score()
        except Exception as exc:
            score = dict(valid=False, success=False, reason='native_score_error', error=str(exc)[:500])
        if initialized:
            recording = dict(**self.header, initial=self.initial,
                initial_applied_targets=self.initial_targets, steps=self.steps,
                outcome=dict(status=status, reason=reason,
                    end_seconds=self.steps[-1]['after']['timestamp_seconds'] if self.steps else 0.,
                    task_result='unverified' if complete else 'unknown'))
            base._json(self.root/'recording.json', recording)
        if not already_finished:
            self._append('finished', reason=reason, status=status)
        self._journal.close()
        report = dict(schema=SCHEMA, status=status, reason=reason, sealed=True,
            error=self._error, source_kind=self.source_kind, case=self.case, scope=SCOPE,
            source_authenticated=False, human_authenticated=False, physical_robot_ready=False,
            teacher_queries=0, requested_intervals=round(self.config.duration_s/tape.DT),
            started_intervals=self.started_intervals, recorded_complete_intervals=len(self.steps),
            uncommitted_native_intervals=self.started_intervals-len(self.steps),
            uncommitted_interval_scope='Intent without a commit may have advanced zero, partial, or full native ticks.',
            camera_pairs=self.camera_pairs, frame_bytes=self.frame_bytes,
            initial_anchor_available=bool(initialized and
                _read(self.root, 'perception/0000.json')['estimate']['status'] == 'available'),
            native_status=native_status, native_evidence_saved=native_saved,
            score=score, strict_success=bool(complete and score.get('success')),
            timing=dict(monotonic_wall_seconds=time.monotonic()-self._started_wall,
                        scope='Paused-simulation acquisition; no wall-clock delivery guarantee.'))
        base._json(self.root/'capture.json', report)
        names = [_artifact_name(self.root, path) for path in sorted(self.root.rglob('*')) if path.is_file()]
        files = {name: _identity(self.root, name) for name in names}
        base._json(self.root/'integrity.json', dict(schema=SCHEMA+'/integrity', files=files))
        self._sealed = True; self._report = report
        return deepcopy(report)


def _no_score():
    return dict(valid=False, success=False, reason='native_evidence_unavailable')


def _journal(root, *, allow_truncated=False):
    body = tape._ordinary_bytes(root/'commands.jsonl', MAX_JOURNAL_BYTES)
    lines = body.splitlines(keepends=True); records = []; last_hash = None
    tail_bytes = 0
    for line in lines:
        if not line.endswith(b'\n'):
            if not allow_truncated:
                raise ValueError('Command journal has an unflushed final record.')
            tail_bytes = len(line); break
        item = tape._decode(line)
        if (item.get('schema') != JOURNAL_SCHEMA or item.get('sequence') != len(records)
                or item.get('previous_sha256') != last_hash or len(records) >= MAX_EVENTS):
            raise ValueError('Command journal hash chain or order differs.')
        tape._bounded(item.get('monotonic_wall_seconds'), 'journal wall time', 0, 1e9)
        tape._time(item.get('simulation_seconds'), 'journal simulation time')
        if records and item['monotonic_wall_seconds'] < records[-1]['monotonic_wall_seconds']:
            raise ValueError('Journal wall time cannot run backwards.')
        last_hash = sha256(line).hexdigest(); records.append(item)
    return records, tail_bytes


def _bound_document(root, reference):
    if _identity(root, reference['path']) != {key: reference[key] for key in ('bytes', 'sha256')}:
        raise ValueError('Journal artifact byte identity differs.')
    return _read(root, reference['path'])


def _prefix(root, *, allow_truncated=False):
    started = _read(root, 'started.json')
    if started.get('schema') != SCHEMA or started.get('source_kind') not in SOURCE_KINDS:
        raise ValueError('Unknown operator recording source.')
    config = camera.RolloutConfig(**started['config'])
    entries, tail_bytes = _journal(root, allow_truncated=allow_truncated)
    context = _read(root, 'context.json') if (root/'context.json').exists() else None
    if context is not None:
        _header(context['header'])
        tape._action(context['initial_applied_targets'], context['header'], 'initial applied targets')
    steps = []; intents = []; pending = None; finished = False
    for entry in entries:
        if finished:
            raise ValueError('Journal continues after finish.')
        kind = entry['kind']
        if kind == 'intent':
            if pending is not None or entry['index'] != len(steps) or context is None:
                raise ValueError('Command intents must follow the committed prefix once.')
            pending = _bound_document(root, entry['decision'])
            if pending['index'] != len(steps):
                raise ValueError('Decision index differs from command intent.')
            boundary = steps[-1]['after'] if steps else context['initial']
            _command(pending['command'], context['header'], boundary)
            _detail(pending['input_event'])
            if pending['snapshot_sha256'] != tape.digest(boundary):
                raise ValueError('Pending intent differs from its causal sensor boundary.')
            _perception(root, pending['perception'], boundary, context['header'], len(steps))
            intents.append(pending)
        elif kind == 'outcome':
            if pending is None or entry['index'] != len(steps) or entry.get('status') != 'committed':
                raise ValueError('Command outcome has no matching pending intent.')
            step = _bound_document(root, entry['transition'])
            _bound_document(root, entry['native'])
            if (step['index'] != len(steps) or step['command'] != pending['command']
                    or step['perception'] != pending['perception']
                    or step['input_event'] != pending['input_event']
                    or tape.digest(step['before']) != pending['snapshot_sha256']):
                raise ValueError('Committed transition differs from recorded command intent.')
            steps.append(step); pending = None
        elif kind == 'event':
            if type(entry.get('name')) is not str or not 1 <= len(entry['name']) <= 80:
                raise ValueError('Invalid operator event name.')
            _detail(entry.get('detail'))
        elif kind == 'finished':
            finished = True
        else:
            raise ValueError('Unknown command journal event.')
    if len(intents) > round(config.duration_s/tape.DT):
        raise ValueError('Operator capture exceeds its scheduled interval budget.')
    if context is not None:
        header = context['header']; _header(header)
        if (header['provenance']['kind'] != started['source_kind']
                or header['provenance']['source_group'] != started['case']['group_id']
                or header['provenance']['case_id'] != started['case']['id']):
            raise ValueError('Operator provenance differs from its started attempt.')
        previous = context['initial']; _snapshot(previous, header, 0)
        _perception(root, context['initial_perception'], previous, header, 0)
        for index, step in enumerate(steps):
            if step['before'] != previous or step.get('teacher_query') is not None:
                raise ValueError('Operator prefix continuity or teacher separation differs.')
            _snapshot(step['after'], header, index+1, previous)
            command = step['command']
            _command(command, header, step['before'])
            applied = step['applied']
            tape._action(applied['ramped_targets'], header, 'applied targets')
            controls = tape._vec(applied['native_position_controls'], 7, 'native controls', -10, 10)
            tape._action(dict(joint_targets_rad=controls[:6], gripper_position_m=controls[6]), header, 'native controls')
            if (applied['interval_start_seconds'] != step['before']['timestamp_seconds']
                    or applied['interval_end_seconds'] != step['after']['timestamp_seconds']
                    or applied['scope'] != 'final_native_tick_only' or type(applied['rate_limited']) is not bool):
                raise ValueError('Applied interval differs from the committed boundary.')
            _perception(root, step['perception'], step['before'], header, index)
            previous = step['after']
        # Both the initial and final snapshots bind their native frame bytes.
        for snapshot in [context['initial'], *[step['after'] for step in steps]]:
            for channel in ('rgb', 'depth'):
                reference = snapshot['camera']['value'][channel]
                if _identity(root, reference['path']) != {key: reference[key] for key in ('bytes', 'sha256')}:
                    raise ValueError('Camera source byte binding differs.')
    return started, context, steps, intents, tail_bytes, finished


def recover_capture(directory):
    """Read-only recovery of an unsealed committed prefix; never resume or seal.

    An outstanding intent cannot establish how many native ticks ran. Even a
    full scheduled prefix remains interrupted until an ordinary finish sealed
    its native evidence. Recovered rows cannot be passed to the exporter.
    """
    root = Path(directory)
    if (root/'integrity.json').exists():
        raise ValueError('Capture is already sealed; inspect or load it normally.')
    started, context, steps, intents, tail, _ = _prefix(root, allow_truncated=True)
    report = dict(schema=SCHEMA+'/recovery', status='interrupted', sealed=False,
        reason='process_interrupted', source_kind=started['source_kind'],
        recorded_complete_intervals=len(steps), started_intervals=len(intents),
        uncommitted_native_intervals=len(intents)-len(steps), truncated_journal_bytes=tail,
        native_progress_of_uncommitted_interval='unknown', strict_success=False,
        score=_no_score(), source_authenticated=False, physical_robot_ready=False)
    recording = None if context is None else dict(**context['header'], initial=context['initial'],
        initial_applied_targets=context['initial_applied_targets'], steps=steps,
        outcome=dict(status='interrupted', reason='process_interrupted', task_result='unknown',
                     end_seconds=steps[-1]['after']['timestamp_seconds'] if steps else 0.))
    return dict(report=report, recording=recording)


def _native_bindings(root, steps, report):
    """Bind emitted targets, encoder inputs, and applied references to native evidence."""
    if not report['native_evidence_saved']:
        return
    episode = _read(root, 'native_episode.json')
    if 'actions' not in episode:
        if report['native_status'].get('native_initialized'):
            raise ValueError('Initialized native recording lacks native action evidence.')
        return  # Explicit non-physics fixtures cannot establish strict success.
    import numpy as np
    import zipfile
    with zipfile.ZipFile(root/'native.npz') as archive:
        if sum(item.file_size for item in archive.infolist()) > MAX_DOCUMENT_BYTES:
            raise ValueError('Expanded native arrays exceed the bounded evidence budget.')
    actions, states = episode['actions'], episode['trajectory']
    if len(actions) != len(steps) or len(states) != len(steps)+1:
        raise ValueError('Native action/state trace differs from the committed prefix.')
    with np.load(root/'native.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in ('arm_actuator_controls_rad',
            'applied_jaw_target_m', 'normal_rate_limited_jaw_target_m',
            'joint_position_rad', 'gripper_joint_position_m', 'requested_jaw_target_m')}
    widths = {'arm_actuator_controls_rad':6, 'joint_position_rad':6,
              'gripper_joint_position_m':2}
    for name, values in arrays.items():
        expected_tail = (widths[name],) if name in widths else ()
        if (values.dtype.kind not in 'fiu' or values.shape[1:] != expected_tail
                or not len(steps)*20 <= len(values) <= report['requested_intervals']*20
                or not np.isfinite(values).all()):
            raise ValueError('Native array shape or finite-value contract differs: '+name)
    for index, step in enumerate(steps):
        if {name: actions[index][name] for name in ('joint_targets_rad', 'gripper_position_m')} != step['command']['issued']:
            raise ValueError('Issued command differs from native action evidence.')
        for position, snapshot in ((index, step['before']), (index+1, step['after'])):
            for name in ('joint_position_rad', 'joint_velocity_rad_s', 'gripper_joint_position_m'):
                if snapshot['signals'][name]['value'] != states[position][name]:
                    raise ValueError('Measured input differs from native encoder evidence.')
        boundary = (index+1)*20-1
        controls = [*arrays['arm_actuator_controls_rad'][boundary].tolist(),
                    float(arrays['applied_jaw_target_m'][boundary])]
        ramped = dict(joint_targets_rad=controls[:6],
                      gripper_position_m=float(arrays['normal_rate_limited_jaw_target_m'][boundary]))
        if step['applied']['native_position_controls'] != controls or step['applied']['ramped_targets'] != ramped:
            raise ValueError('Applied inputs differ from native servo reference evidence.')
        for name in ('joint_position_rad', 'gripper_joint_position_m'):
            if not np.allclose(arrays[name][boundary], step['after']['signals'][name]['value'],
                               atol=2e-7, rtol=0):
                raise ValueError('After-state encoders differ from native tick evidence.')
        if not np.allclose(arrays['requested_jaw_target_m'][index*20:(index+1)*20],
                           step['command']['issued']['gripper_position_m'], atol=1e-12, rtol=0):
            raise ValueError('Issued jaw target differs from native tick evidence.')


def inspect_capture(directory):
    """Verify the sealed journal, causal rows, byte bindings and independent score."""
    root = Path(directory); manifest = _read(root, 'integrity.json')
    if manifest.get('schema') != SCHEMA+'/integrity' or type(manifest.get('files')) is not dict:
        raise ValueError('An operator recording requires a sealed integrity manifest.')
    observed = {_artifact_name(root, path) for path in root.rglob('*')
                if path.is_file() and path != root/'integrity.json'}
    if observed != set(manifest['files']):
        raise ValueError('Operator recording inventory differs.')
    for name, identity in manifest['files'].items():
        if _identity(root, name) != identity:
            raise ValueError('Operator recording byte identity differs: '+name)
    frame_identities = [identity for name, identity in manifest['files'].items() if name.startswith('frames/')]
    frame_bytes = sum(identity['bytes'] for identity in frame_identities)
    if frame_bytes > tape.MAX_FRAME_TOTAL or any(identity['bytes'] > tape.MAX_FRAME_BYTES for identity in frame_identities):
        raise ValueError('Native RGB-D frame evidence exceeds its byte budget.')
    started, context, steps, intents, _, finished = _prefix(root)
    report = _read(root, 'capture.json')
    entries, _ = _journal(root)
    if (report.get('schema') != SCHEMA or report.get('sealed') is not True or not finished
            or report.get('status') not in ('complete', 'interrupted', 'refused')
            or report.get('source_kind') != started['source_kind']
            or report.get('teacher_queries') != 0 or report.get('source_authenticated') is not False
            or report.get('human_authenticated') is not False
            or report.get('physical_robot_ready') is not False
            or report.get('case') != started['case']
            or report.get('requested_intervals') != round(started['config']['duration_s']/tape.DT)
            or report.get('frame_bytes') != frame_bytes
            or report.get('recorded_complete_intervals') != len(steps)
            or report.get('started_intervals') != len(intents)
            or report.get('uncommitted_native_intervals') != len(intents)-len(steps)):
        raise ValueError('Operator report disagrees with the command journal.')
    if entries[-1].get('status') != report['status'] or entries[-1].get('reason') != report['reason']:
        raise ValueError('Operator report differs from the journal finish record.')
    if context is not None:
        recording = _read(root, 'recording.json')
        if (any(recording.get(key) != value for key, value in context['header'].items())
                or recording.get('initial') != context['initial']
                or recording.get('initial_applied_targets') != context['initial_applied_targets']
                or recording.get('steps') != steps
                or recording['outcome']['status'] != report['status']):
            raise ValueError('Sealed recording differs from its committed prefix.')
    if report['status'] == 'complete' and (len(steps) != round(started['config']['duration_s']/tape.DT)
                                          or len(intents) != len(steps) or report.get('error') is not None):
        raise ValueError('Complete recording omits scheduled intervals or contains an error.')
    _native_bindings(root, steps, report)
    score = camera._score(root, started['case']['goal_position_m']) if report['native_evidence_saved'] else _no_score()
    if score != report['score'] or report['strict_success'] != bool(report['status'] == 'complete' and score.get('success')):
        raise ValueError('Independent native score differs from the sealed report.')
    return dict(schema=SCHEMA+'/inspection', status=report['status'], sealed=True,
        files_verified=len(manifest['files']), recorded_complete_intervals=len(steps),
        source_authenticated=False, human_authenticated=False, physical_robot_ready=False,
        score=score, strict_success=report['strict_success'])


def load_capture(directory):
    inspect_capture(directory)
    return _read(Path(directory), 'recording.json')

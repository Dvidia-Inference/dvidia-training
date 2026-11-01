"""Local Training Studio bridge for registered camera/action studies.

It shares the footage server's one worker and lock. A browser never supplies
Python code, arbitrary commands or hardware endpoints. Stage order comes from
the same verified study inspection used by the Terminal workflow.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import time
import uuid

SCHEMA = 'dvidia.camera-action-studio/v1'


def _worker_lock(study):
    """Read only the study's small lock record; never infer liveness or unlink it."""
    path = Path(study)/'worker.lock'
    if not path.exists() and not path.is_symlink():
        return dict(present=False)
    result = dict(present=True)
    try:
        if not path.is_symlink() and path.is_file() and path.stat().st_size <= 1024:
            value = json.loads(path.read_text())
            if type(value) is dict and isinstance(value.get('operation'), str):
                result['operation'] = value['operation'][:80]
    except (OSError, ValueError):
        pass
    return result


def _local_path(value, label):
    if not isinstance(value, str) or not value.strip() or len(value) > 8192 or any(ord(c) < 32 for c in value):
        raise ValueError('Supply a local '+label+' folder.')
    if '://' in value:
        raise ValueError('Use local folders; the robot study does not download inputs.')
    path = Path(value).expanduser().absolute()
    if any(p.is_symlink() and str(p) not in ('/tmp', '/var') for p in (path, *path.parents)):
        raise ValueError('The '+label+' folder cannot traverse symlinks.')
    if not path.is_dir():
        raise ValueError('The '+label+' folder does not exist on this server.')
    return path.resolve()


def check_study(study, assets):
    from . import camera_action_study, robot_profiles
    root = _local_path(study, 'prepared study')
    asset_root = _local_path(assets, 'verified robot assets')
    state = camera_action_study.inspect(root)
    camera_action_study._protocol(root, runtime=True)
    metadata = robot_profiles.verify_assets('piper', asset_root)
    binding = dict(study=str(root), assets=str(asset_root), protocol_sha256=state['protocol_sha256'],
        assets_sha256=sha256(json.dumps(metadata, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest())
    stage = state.get('report', {}).get('selected_stage')
    model = root/'models'/str(stage)/'full'/'model.json' if stage in ('initial', 'corrected') else None
    install_available = bool(state['status'] == 'complete' and model is not None and model.is_file() and not model.is_symlink())
    worker_lock = _worker_lock(root)
    return dict(schema=SCHEMA, binding=binding, inspection=state, install_available=install_available,
        selected_model=str(model) if install_available else None, study_locked=worker_lock['present'],
        study_worker_lock=worker_lock, physical_robot_ready=False)


def _write_progress(output, status, message, **extra):
    output = Path(output)
    value = dict(schema=SCHEMA, status=status, message=message, **extra)
    target = output/'robot-progress.json'
    temporary = output/'robot-progress.tmp'
    temporary.write_text(json.dumps(value, allow_nan=False))
    temporary.replace(target)


def _record(output, value):
    with (Path(output)/'robot-receipt.json').open('x') as stream:
        json.dump(value, stream, allow_nan=False)


def run_worker(kind, study, assets, output, expected_binding=None, installation=None):
    """Worker-only orchestration. Each registered stage is attempted at most once."""
    from . import camera_action_study
    steps = []; started = time.perf_counter()
    try:
        _write_progress(output, 'checking', 'Verifying the registered study and robot assets.')
        checked = check_study(study, assets)
        if expected_binding is not None and checked['binding'] != expected_binding:
            raise ValueError('The checked study, protocol or assets changed. Check the inputs again.')
        if kind != 'check' and _worker_lock(study)['present']:
            raise ValueError('This study has an existing worker lock. Wait for its Terminal run, then check the inputs again.')
        if kind == 'check':
            status = 'blocked' if checked['inspection']['status'] == 'blocked' else 'checked'
        elif kind == 'run':
            for _ in range(len(camera_action_study.STAGES)+1):
                state = checked['inspection']
                if state['status'] in ('blocked', 'complete'):
                    break
                actions = state.get('next_actions', [])
                if len(actions) != 1:
                    raise ValueError('The study did not declare exactly one next stage.')
                action = actions[0]
                command, stage = action['command'], action['stage']
                if command not in ('collect', 'train', 'evaluate'):
                    raise ValueError('Unknown registered study stage.')
                key = (command, stage)
                if key in steps:
                    raise ValueError('The study requested a repeated stage; no automatic retry is allowed.')
                _write_progress(output, 'running', 'Running '+command+' / '+stage+'.', command=command, stage=stage)
                steps.append(key)
                if command == 'train':
                    camera_action_study.train(study, stage=stage)
                else:
                    getattr(camera_action_study, command)(study, assets, stage=stage)
                checked = check_study(study, assets)
                if checked['binding'] != expected_binding:
                    raise ValueError('The bound study protocol or assets changed during execution.')
            else:
                raise ValueError('Registered study stage limit exceeded.')
            status = checked['inspection']['status']
        elif kind == 'install':
            if not checked['install_available']:
                raise ValueError('Complete the benchmark before installing its simulation candidate.')
            from .arm_skill_install import install_model, inspect_installation
            destination = Path(installation)
            if destination.exists() or destination.is_relative_to(Path(study)):
                raise ValueError('Installation must use a fresh directory outside the study.')
            install_model(checked['selected_model'], 'piper', assets, destination)
            installation_record = inspect_installation(destination, assets)
            checked['installation'] = dict(path=str(destination), record=installation_record)
            status = 'installed'
        else:
            raise ValueError('Unknown robot studio action.')
        result = dict(**checked, status=status, executed_stages=[list(step) for step in steps],
            elapsed_seconds=time.perf_counter()-started)
        _write_progress(output, status, 'Simulation study '+status+'.')
        _record(output, result)
        return result
    except Exception as exc:
        result = dict(schema=SCHEMA, status='failed', error=str(exc)[:2000],
            executed_stages=[list(step) for step in steps], elapsed_seconds=time.perf_counter()-started,
            install_available=False, physical_robot_ready=False)
        _write_progress(output, 'failed', str(exc)[:600])
        _record(output, result)
        raise


class CameraActionStudio:
    """One robot job, protected by the parent server's shared lock and pool."""
    def __init__(self, server, *, worker=None, checker=None):
        self.server = server
        self.worker = worker or run_worker
        self.checker = checker or check_study
        self.job = None; self.checked = None
        self.external_lock_seen = False

    def refresh(self):
        job = self.job
        if job is None or job['status'] != 'running' or not job['future'].done():
            return
        try:
            value = job['future'].result()
            if type(value) is not dict or value.get('status') not in ('checked', 'blocked', 'complete', 'installed'):
                raise ValueError('The robot worker did not return a verified study state.')
            verified = self.checker(job['study'], job['assets'])
            if verified['binding'] != value['binding']:
                raise ValueError('The study or assets changed after the worker completed.')
            if value['status'] == 'blocked' or verified['inspection']['status'] == 'blocked':
                self.checked = None
                value = {**verified, 'status': 'blocked', 'install_available': False}
            else:
                self.checked = verified
                value.update(verified)
                if job['kind'] == 'check':
                    self.external_lock_seen = _worker_lock(job['study'])['present']
            job.update(status=value['status'], summary=value, error=None, finished=time.monotonic())
        except Exception as exc:
            self.checked = None
            job.update(status='failed', summary=None, error=str(exc)[:2000], finished=time.monotonic())

    def busy(self):
        self.refresh()
        return self.job is not None and self.job['status'] == 'running'

    def submit(self, kind, study, assets):
        with self.server.lock:
            self.server.refresh(); self.refresh()
            if self.busy() or (self.server.job is not None and self.server.job['status'] == 'running'):
                raise RuntimeError('One CPU job is already running. Wait for its outcome.')
            try:
                study = str(_local_path(study, 'prepared study')); assets = str(_local_path(assets, 'robot assets'))
                binding = None
                if kind in ('run', 'install'):
                    if self.checked is None:
                        raise ValueError('Check the paired demonstrations and assets before continuing.')
                    verified = self.checker(study, assets)
                    if self.external_lock_seen or _worker_lock(study)['present']:
                        raise ValueError('Study running in Terminal or awaiting a fresh check. Recheck after its worker lock clears.')
                    if verified['binding'] != self.checked['binding']:
                        raise ValueError('The checked protocol or assets changed. Check the inputs again.')
                    if kind == 'run' and verified['inspection']['status'] != 'ready':
                        raise ValueError('This study has no runnable next stage.')
                    if kind == 'install' and not verified['install_available']:
                        raise ValueError('No completed simulation candidate is available to install.')
                    binding = verified['binding']
                elif kind == 'check':
                    self.checked = None
                    self.external_lock_seen = False
                else:
                    raise ValueError('Unknown robot studio action.')
            except (OSError, ValueError, RuntimeError):
                self.checked = None
                self.job = None
                raise
            identity = uuid.uuid4().hex
            output = self.server.directory/'robot-jobs'/identity
            output.mkdir(parents=True, exist_ok=False)
            installation = str(self.server.directory/'installations'/identity) if kind == 'install' else None
            future = self.server.pool.submit(self.worker, kind, study, assets, str(output), binding, installation)
            self.job = dict(id=identity, kind=kind, output=output, study=study, assets=assets,
                status='running', future=future, summary=None, error=None,
                started=time.monotonic(), finished=None)
            return dict(status='running', job_id=identity, kind=kind)

    def state(self):
        with self.server.lock:
            self.server.refresh(); self.refresh()
            if self.checked is not None and not (self.job and self.job['status'] == 'running'):
                try:
                    protocol = Path(self.checked['binding']['study'])/'protocol.json'
                    if protocol.is_symlink() or sha256(protocol.read_bytes()).hexdigest() != self.checked['binding']['protocol_sha256']:
                        raise ValueError('The checked protocol changed. Check the inputs again.')
                except (OSError, ValueError) as exc:
                    self.checked = None
                    if self.job is not None:
                        self.job.update(status='failed', summary=None, error=str(exc)[:2000], finished=time.monotonic())
            footage_busy = self.server.job is not None and self.server.job['status'] == 'running'
            result = dict(schema=SCHEMA, status='idle', busy=footage_busy, job=None,
                summary=None, error=None, checked=False, can_run=False, can_install=False,
                study_locked=False, recheck_required=False, study_worker_lock=dict(present=False),
                physical_robot_ready=False)
            if self.job is not None:
                job = self.job
                progress = None
                path = job['output']/'robot-progress.json'
                if path.is_file() and path.stat().st_size <= 8192:
                    try:
                        progress = json.loads(path.read_text())
                    except (ValueError, OSError):
                        pass
                result.update(status=job['status'], busy=footage_busy or job['status']=='running',
                    summary=job['summary'], error=job['error'], job=dict(id=job['id'],kind=job['kind'],
                        elapsed_seconds=(job['finished'] or time.monotonic())-job['started'], progress=progress,
                        study=job['study'], assets=job['assets']))
            if self.checked is not None:
                worker_lock = _worker_lock(self.checked['binding']['study'])
                if worker_lock['present'] and not (self.job and self.job['status']=='running'):
                    self.external_lock_seen = True
                result.update(checked=True,
                    study_locked=worker_lock['present'], study_worker_lock=worker_lock,
                    recheck_required=self.external_lock_seen,
                    can_run=not result['busy'] and not self.external_lock_seen and not worker_lock['present']
                        and self.checked['inspection']['status']=='ready',
                    can_install=not result['busy'] and not self.external_lock_seen and not worker_lock['present']
                        and self.checked['install_available'])
            return result

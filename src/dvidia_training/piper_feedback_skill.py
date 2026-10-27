"""Run a downloaded PiPER feedback skill once in local native simulation.

Both policies are finite data-only JSON. The frozen base supplies nominal
motion; a seven-command residual or jaw-only adapter reads privileged simulator
measurements. No network, expert, IK fallback, user fault injection or physical
robot interface is used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from .piper_imitation import MAX_MODEL_BYTES, digest
from .piper_motor_primitive import GOALS, START_BOUNDS, validate_model as validate_base
from .piper_feedback_model import validate_model as validate_feedback
from .reference_replay import _read


SCHEMA = 'dvidia.piper-feedback-skill-install/v0.2'
SCOPE = ('Simulation-only learned PiPER grip correction over the exact frozen v0.1 nominal model. '
         'Synthetic authored-expert labels and privileged native simulator measurements; '
         'uncalibrated rigid-block contact, no camera learning, physical robot connection or hardware qualification.')
NATIVE_BYTE_LIMIT = 128*1024*1024


def _point(value, name, limits):
    if type(value) not in (int, float) or not limits[0] <= value <= limits[1] or not math.isfinite(value):
        raise ValueError(f'{name} must be finite and inside [{limits[0]}, {limits[1]}] metres.')
    return float(value)


def _error(exc):
    return {'type': type(exc).__name__, 'message': str(exc)[:1000]}


def _inside(path, parent):
    return path == parent or parent in path.parents


def _check_output(path, sources, assets):
    # The final directory is reserved with mkdir(exist_ok=False), after all
    # input validation. Resolve parent aliases for collision checks; ordinary
    # system aliases such as macOS /tmp remain usable output locations.
    if path.exists() or path.is_symlink():
        raise ValueError('output must be a fresh directory; previous attempts are preserved.')
    destination = path.absolute().resolve(strict=False)
    if _inside(destination, assets.resolve()) or any(_inside(source.resolve(), destination) for source in sources):
        raise ValueError('output must be separate from the read-only models and pinned asset directory.')


def _native_bytes(arrays):
    if type(arrays) is not dict:
        raise ValueError('Native evidence must be a numeric array mapping.')
    total = 0
    for name, value in arrays.items():
        if (type(name) is not str or not name or not name.replace('_', '').isalnum()
                or not isinstance(value, np.ndarray) or value.dtype.kind not in 'fiu'
                or not np.isfinite(value).all()):
            raise ValueError('Native evidence requires named finite numeric arrays; object/pickle arrays are rejected.')
        total += value.nbytes
        if total > NATIVE_BYTE_LIMIT:
            raise ValueError('Native evidence exceeds the bounded local archive size.')
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def run_feedback_skill(base_model, model, assets, output, *, start_x=.30, start_y=-.03, goal=0):
    """Install exact JSON bytes and preserve one bounded, default-nominal attempt.

    Returns final ``install.json``. Input errors raise before an output is
    reserved. Once reserved, runtime failures become recorded invalid attempts.
    Every output file is created exclusively; interrupted folders are retained
    and a subsequent invocation must choose a new destination.
    """
    from .piper_feedback_benchmark import BASE_SHA, config, rollout, runtime_hashes
    from .piper_feedback_score import SCHEMA as SCORE_SCHEMA, score_feedback_episode
    from .piper_imitation_benchmark import _atomic_create, save_episode, sha, write
    from .piper_task import _asset_root
    from .robot_profiles import verify_assets
    start_x = _point(start_x, 'start_x', START_BOUNDS[0])
    start_y = _point(start_y, 'start_y', START_BOUNDS[1])
    if type(goal) is not int or goal not in (0, 1):
        raise ValueError('goal must be integer 0 or 1, selecting one trained destination.')
    base_source, feedback_source = Path(base_model), Path(model)
    for source in (base_source, feedback_source):
        if source.suffix.lower() != '.json':
            raise ValueError('Both downloaded models must be bounded data-only .json files.')
    base_raw, feedback_raw = _read(base_source, MAX_MODEL_BYTES), _read(feedback_source, MAX_MODEL_BYTES)
    base = validate_base(base_raw)
    decoded = json.loads(feedback_raw)
    if isinstance(decoded, dict) and decoded.get('schema') == 'dvidia.piper-learned-grip-adapter/v1':
        from .piper_grip_adapter import validate_model as validate_grip
        from .piper_learned_grip_benchmark import rollout, runtime_hashes
        feedback = validate_grip(feedback_raw)
        architecture = 'learned_absolute_jaw_over_frozen_six_joint_path'
    else:
        feedback = validate_feedback(feedback_raw)
        architecture = 'learned_seven_command_residual'
    base_raw_sha, feedback_raw_sha = sha256(base_raw).hexdigest(), sha256(feedback_raw).hexdigest()
    if base_raw_sha != BASE_SHA or feedback['base_sha256'] != digest(base):
        raise ValueError('Feedback skill requires its exact frozen v0.1 base model.')
    asset_root, destination = _asset_root(assets), Path(output)
    if asset_root.is_symlink() or not asset_root.is_dir():
        raise ValueError('assets must be an existing pinned native asset directory.')
    _check_output(destination, (base_source, feedback_source), asset_root)
    provenance = verify_assets('piper', asset_root)
    variant = 'blind' if feedback['blind'] else 'feedback'
    case = dict(id=f'installed-feedback-goal{goal}-nominal', start_xy=[start_x, start_y],
                goal_xy=list(GOALS[goal]), seed=271517, condition='nominal', pulses=[])
    config_document = json.loads(json.dumps(asdict(config(case)), allow_nan=False))
    runtime = runtime_hashes()
    request = dict(schema=SCHEMA, scope=SCOPE, physical_robot_ready=False,
                   architecture=architecture,
                   policy_kind=variant, observation_origin='privileged_native_simulator_state',
                   supervision_origin='synthetic_authored_expert', teacher_available=False,
                   teacher_fallback_calls=0, user_fault_injection=False, case=case, config=config_document,
                   base_model_file='base-model.json', feedback_model_file='feedback-model.json',
                   base_model_raw_sha256=base_raw_sha, feedback_model_raw_sha256=feedback_raw_sha,
                   base_model_sha256=digest(base), feedback_model_sha256=digest(feedback),
                   runtime_sha256=runtime, runner_sha256=sha(Path(__file__)),
                   asset_provenance=provenance,
                   trained_scope=dict(block_mass_kg=.04, start_bounds_xy_m=[list(b) for b in START_BOUNDS],
                                      known_goals_xy_m=[list(g) for g in GOALS]),
                   created_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   timing_scope='Local CPU simulation and evidence recording; excludes camera, communication and physical actuator latency.')
    destination.mkdir(parents=True, exist_ok=False)
    _atomic_create(destination/'base-model.json', base_raw)
    _atomic_create(destination/'feedback-model.json', feedback_raw)
    write(destination/'started.json', request)
    began = time.perf_counter()
    error = None
    try:
        episode, arrays, metadata, samples = rollout(asset_root, case, base,
            variant=variant, feedback_document=feedback, collect_labels=False)
        if any(samples.get(key) for key in ('features', 'blind_features', 'targets', 'groups', 'weights')):
            raise ValueError('An installed student attempt unexpectedly collected expert labels.')
    except Exception as exc:
        error = _error(exc)
        episode = dict(schema='dvidia.piper-feedback-episode/v0.2', case_id=case['id'], seed=case['seed'],
                       condition='nominal', pulses=[], policy_kind=variant, config=config_document,
                       base_model_sha256=digest(base), model_sha256=digest(feedback),
                       implementation_sha256=runtime, physical_robot_ready=False,
                       observation_origin='privileged_native_simulator_state', valid=False, success=False,
                       reason='exception', error=error, trajectory=[], actions=[], metrics={})
        arrays, metadata = {}, {'rows': 0, 'pulses': [], 'recording_error': error}
    try:
        native_raw = _native_bytes(arrays)
    except (ValueError, TypeError, OverflowError) as exc:
        error = _error(exc)
        # Never emit object arrays or a pickle-bearing archive. The original
        # episode and the archive validation failure remain visible evidence.
        arrays = {}
        native_raw = _native_bytes(arrays)
        metadata = {**metadata, 'archive_error': error}
    try:
        score = score_feedback_episode(episode, arrays, metadata, list(GOALS[goal]))
    except Exception as exc:
        error = _error(exc)
        score = dict(schema=SCORE_SCHEMA, valid=False, success=False, reason='scoring_exception', error=error)
    binding_error = None
    try:
        installed_ok = (_read(destination/'base-model.json', MAX_MODEL_BYTES) == base_raw
                        and _read(destination/'feedback-model.json', MAX_MODEL_BYTES) == feedback_raw)
    except (ValueError, OSError):
        installed_ok = False
    if not installed_ok:
        binding_error = 'installed_model_changed'
    elif (episode.get('base_model_sha256') != digest(base)
          or episode.get('model_sha256') != digest(feedback)
          or episode.get('implementation_sha256') != runtime or runtime_hashes() != runtime):
        binding_error = 'runtime_model_or_source_digest_mismatch'
    elif (episode.get('case_id') != case['id'] or episode.get('seed') != case['seed']
          or episode.get('policy_kind') != variant or episode.get('condition') != 'nominal'
          or episode.get('pulses') != [] or episode.get('config') != config_document
          or metadata.get('pulses') != []):
        binding_error = 'runtime_case_or_configuration_mismatch'
    elif episode.get('valid') is True and episode.get('asset_provenance') != provenance:
        binding_error = 'runtime_asset_provenance_mismatch'
    if error is not None or binding_error is not None:
        score = {**score, 'valid': False, 'success': False,
                 'reason': binding_error or score.get('reason', 'recording_exception')}
    save_episode(destination/'episode.json.gz', episode)
    _atomic_create(destination/'native.npz', native_raw)
    write(destination/'native.json', metadata)
    write(destination/'score.json', score)
    names = ('base-model.json', 'feedback-model.json', 'started.json', 'episode.json.gz',
             'native.npz', 'native.json', 'score.json')
    valid = score.get('valid') is True and episode.get('valid') is True
    manifest = dict(request, status='completed', valid=valid,
                    success=bool(valid and score.get('success') is True),
                    reason=score.get('reason', 'unknown'), error=error or episode.get('error'),
                    files={name: sha(destination/name) for name in names},
                    compiled_model_sha256=episode.get('compiled_model_sha256'),
                    attempt_wall_seconds=time.perf_counter()-began)
    write(destination/'install.json', manifest)
    write(destination/'receipt.json', dict(schema=SCHEMA, files={**manifest['files'],
        'install.json': sha(destination/'install.json')}, valid=manifest['valid'], success=manifest['success']))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        'Scope: fixed 40 g rigid block; two trained destinations and a 30×30 mm starting region. '
        'No hardware connection. Exit 0: strict independently reconstructed success; '
        '1: valid retained failure; 2: invalid input or invalid recorded attempt.'))
    parser.add_argument('--base-model', type=Path, required=True, help='Exact frozen v0.1 base-model JSON.')
    parser.add_argument('--model', type=Path, required=True, help='Bound residual, jaw-only adapter or matched blind model JSON.')
    parser.add_argument('--assets', type=Path, required=True, help='Pinned native asset root containing agilex_piper/.')
    parser.add_argument('--start-x', type=float, default=.30, help='Initial x in [0.284,0.314] metres.')
    parser.add_argument('--start-y', type=float, default=-.03, help='Initial y in [-0.045,-0.015] metres.')
    parser.add_argument('--goal', type=int, choices=(0, 1), default=0, help='Trained destination index 0 or 1.')
    parser.add_argument('--output', type=Path, required=True, help='Fresh output directory; never overwritten.')
    args = parser.parse_args(argv)
    try:
        manifest = run_feedback_skill(args.base_model, args.model, args.assets, args.output,
                                      start_x=args.start_x, start_y=args.start_y, goal=args.goal)
    except (ValueError, OSError, TypeError) as exc:
        print(f'PiPER feedback input rejected: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(dict(valid=manifest['valid'], success=manifest['success'], reason=manifest['reason'],
                         manifest=str(args.output/'install.json'), physical_robot_ready=False), indent=2))
    return 0 if manifest['success'] else 1 if manifest['valid'] else 2


if __name__ == '__main__':
    raise SystemExit(main())

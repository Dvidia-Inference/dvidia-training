"""Run a downloaded data-only PiPER model in local native simulation.

The installed model supplies all six arm targets and the single jaw target.
This command has no teacher or fallback and does not connect to robot hardware.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys
import time

from .piper_imitation import MAX_MODEL_BYTES, validate_model
from .reference_replay import _read


SCHEMA = 'dvidia.piper-learned-skill-install/v1'
KNOWN_GOALS = ((.35, .08), (.40, -.08))
SCOPE = ('Simulation-only installation of a learned PiPER policy distilled from synthetic '
         'authored-controller demonstrations. Privileged native simulator measurements; '
         'uncalibrated contacts, no camera learning or physical robot qualification.')
TRAINED_SCOPE = {'mass_kg': .04, 'start_x_m': [.284, .314], 'start_y_m': [-.045, -.015],
                'known_goals_xy_m': [list(point) for point in KNOWN_GOALS]}


def _point(value, name, low, high):
    if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value):
        raise ValueError(f'{name} must be finite and inside [{low}, {high}] metres.')
    return float(value)


def _write(path, value, *, exclusive=True):
    raw = (json.dumps(value, indent=2, allow_nan=False)+'\n').encode('utf8')
    with path.open('xb' if exclusive else 'wb') as stream:
        stream.write(raw)


def _error(exc):
    return {'type': type(exc).__name__, 'message': str(exc)[:1000]}


def run_learned_skill(model, assets, output, *, start_x=.30, start_y=-.03, goal=0):
    """Reserve a fresh installation, then run and independently score one episode.

    Returns the saved install.json manifest. Any native/inference/scoring failure
    is retained as a failed attempt; validation errors raise before execution.
    The source JSON is copied byte-for-byte and its SHA256 binds the installed
    model used by the rollout. Start coordinates are restricted to the training
    envelope, with two known destinations and the fixed 40 g block.
    """
    from .piper_imitation_benchmark import config, rollout, save_episode
    from .piper_imitation_score import score_episode
    from .piper_task import _asset_root
    start_x = _point(start_x, 'start_x', .284, .314)
    start_y = _point(start_y, 'start_y', -.045, -.015)
    if type(goal) is not int or goal not in (0, 1):
        raise ValueError('goal must be integer 0 or 1, selecting a known trained destination.')
    source, asset_root, destination = Path(model), _asset_root(assets), Path(output)
    if (re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,119}\.json', source.name) is None
            or '..' in source.name):
        raise ValueError('Model filename must be a safe ASCII .json name, such as model.json.')
    raw = _read(source, MAX_MODEL_BYTES)
    candidate = validate_model(raw)
    if asset_root.is_symlink() or not asset_root.is_dir():
        raise ValueError('assets must point to an existing pinned native asset directory.')
    if destination.exists() or destination.is_symlink():
        raise ValueError('output must be a fresh directory; previous attempts are preserved.')
    case = {'id': f'installed-known-goal{goal}', 'start_xy': [start_x, start_y],
            'goal_xy': list(KNOWN_GOALS[goal]), 'seed': 261517}
    task_config = config(case)
    config_document = json.loads(json.dumps(asdict(task_config), allow_nan=False))
    model_sha = sha256(raw).hexdigest()
    manifest = {'schema': SCHEMA, 'scope': SCOPE, 'status': 'running', 'success': False,
                'physical_robot_ready': False, 'policy_kind': 'learned', 'policy_learned': True,
                'teacher_available': False, 'teacher_fallback_calls': 0,
                'observation_origin': 'privileged_native_simulator_state',
                'model_sha256': model_sha, 'installed_model_sha256': model_sha,
                'model_file': 'model.json', 'model_schema': candidate['schema'],
                'feature_contract': candidate['feature_contract'],
                'trained_scope': json.loads(json.dumps(TRAINED_SCOPE)), 'start_xy_m': [start_x, start_y],
                'goal_index': goal, 'goal_xy_m': list(KNOWN_GOALS[goal]), 'config': config_document,
                'episode_file': 'episode.json.gz', 'score_file': 'score.json',
                'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'timing_scope': 'Local CPU simulation compute; excludes sensing, communications and hardware actuation.'}
    destination.mkdir(parents=True, exist_ok=False)
    installed = destination/'model.json'
    with installed.open('xb') as stream:
        stream.write(raw)
    _write(destination/'install.json', manifest)
    began = time.perf_counter()
    try:
        episode = rollout(asset_root, case, installed, variant='learned')
    except Exception as exc:
        episode = {'schema': 'dvidia.piper-learned-episode/v1', 'config': config_document,
                   'valid': False, 'success': False, 'reason': 'exception', 'error': _error(exc),
                   'trajectory': [], 'actions': [], 'metrics': {}, 'timing': {},
                   'policy_kind': 'learned', 'policy_learned': True, 'physical_robot_ready': False,
                   'model_sha256': model_sha}
    try:
        score = score_episode(episode, list(KNOWN_GOALS[goal]))
    except Exception as exc:
        score = {'success': False, 'reason': 'scoring_exception', 'error': _error(exc)}
    try:
        installed_unchanged = sha256(_read(installed, MAX_MODEL_BYTES)).hexdigest() == model_sha
    except (ValueError, OSError):
        installed_unchanged = False
    if not installed_unchanged:
        score = {**score, 'success': False, 'reason': 'installed_model_changed'}
    elif episode.get('model_sha256') != model_sha:
        score = {**score, 'success': False, 'reason': 'runtime_model_digest_mismatch'}
    save_episode(destination/'episode.json.gz', episode)
    _write(destination/'score.json', score)
    manifest.update(status='completed', success=score.get('success') is True, reason=score.get('reason', 'unknown'),
                    episode_sha256=sha256((destination/'episode.json.gz').read_bytes()).hexdigest(),
                    score_sha256=sha256((destination/'score.json').read_bytes()).hexdigest(),
                    native_valid=episode.get('valid') is True, error=episode.get('error'),
                    asset_provenance=episode.get('asset_provenance'),
                    compiled_model_sha256=episode.get('compiled_model_sha256'),
                    attempt_wall_seconds=time.perf_counter()-began)
    _write(destination/'install.json', manifest, exclusive=False)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        'Scope: a 40 g rigid block, x=0.284–0.314 m, y=−0.045–−0.015 m; '
        'known goals 0=(0.35,0.08) m and 1=(0.40,−0.08) m. The native environment verifies pinned assets. '
        'Exit status: 0 independent success, 1 failed recorded attempt, 2 invalid input. '
        'All control/physics runs locally in native simulation; no physical robot is connected.'))
    parser.add_argument('--model', type=Path, required=True, help='Downloaded finite data-only model JSON (at most 8 MiB).')
    parser.add_argument('--assets', type=Path, required=True, help='Pinned native asset root containing agilex_piper/.')
    parser.add_argument('--start-x', type=float, default=.30, help='Block start x in metres within [0.284,0.314] (default: 0.30).')
    parser.add_argument('--start-y', type=float, default=-.03, help='Block start y in metres within [−0.045,−0.015] (default: −0.03).')
    parser.add_argument('--goal', type=int, choices=(0, 1), default=0, help='Known goal index, 0 or 1 (default: 0).')
    parser.add_argument('--output', type=Path, required=True, help='Fresh directory for installed model, native trace, score and install manifest.')
    args = parser.parse_args(argv)
    try:
        manifest = run_learned_skill(args.model, args.assets, args.output,
                                     start_x=args.start_x, start_y=args.start_y, goal=args.goal)
    except (ValueError, OSError, TypeError) as exc:
        print(f'PiPER model input rejected: {exc}', file=sys.stderr)
        return 2
    print(json.dumps({'success': manifest['success'], 'reason': manifest['reason'],
                      'manifest': str(args.output/'install.json'), 'physical_robot_ready': False}, indent=2))
    return 0 if manifest['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

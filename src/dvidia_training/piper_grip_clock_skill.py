"""Install a data-only grip clock and run one nominal PiPER simulation attempt.

No expert labels, disturbance, network access or physical robot connection are
used. The selected model changes timing of all seven frozen references; it does
not learn a new grasp or path. Failed and interrupted attempts are preserved.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .piper_feedback_skill import _point, _check_output
from .piper_motor_primitive import GOALS, START_BOUNDS
from .piper_imitation import MAX_MODEL_BYTES
from .reference_replay import _read


SCHEMA = 'dvidia.piper-grip-clock-install/v0.3'


def run_clock_skill(base_model, jaw_model, model, assets, output, *, start_x=.30, start_y=-.03, goal=0):
    from . import piper_grip_clock_benchmark as runner
    from .piper_grip_clock_model import validate_model
    from .piper_imitation_benchmark import _atomic_create, read, write, sha
    from .piper_task import _asset_root
    from .robot_profiles import verify_assets
    start_x = _point(start_x, 'start_x', START_BOUNDS[0])
    start_y = _point(start_y, 'start_y', START_BOUNDS[1])
    if type(goal) is not int or goal not in (0, 1):
        raise ValueError('goal must select known destination 0 or 1.')
    paths = tuple(Path(p) for p in (base_model, jaw_model, model))
    if any(p.suffix.lower() != '.json' for p in paths):
        raise ValueError('All three model inputs must be bounded data-only JSON.')
    raw = tuple(_read(p, MAX_MODEL_BYTES) for p in paths)
    documents = tuple(json.loads(value) for value in raw)
    clock_document = validate_model(documents[2])
    variant = 'blind_clock' if clock_document['blind'] else 'clock'
    base, jaw, clock = runner._documents(*documents, variant)
    if sha256(raw[0]).hexdigest() != runner.BASE_SHA:
        raise ValueError('The grip clock requires the exact frozen base bytes.')
    mode, _ = runner.jaw_contract(base, jaw)
    if mode != 'coordinated_reference':
        raise ValueError('This installer supports the explicit coordinated-reference v0.3 architecture.')
    asset_root, destination = _asset_root(assets), Path(output)
    if asset_root.is_symlink() or not asset_root.is_dir():
        raise ValueError('assets must be a pinned native asset directory.')
    _check_output(destination, paths, asset_root)
    provenance = verify_assets('piper', asset_root)
    runtime = runner.runtime_hashes()
    case = dict(id=f'installed-clock-goal{goal}-nominal', start_xy=[start_x, start_y],
                goal_xy=list(GOALS[goal]), seed=273517, condition='nominal')
    request = dict(schema=SCHEMA, variant=variant, jaw_mode=mode, case=case,
        physical_robot_ready=False, teacher_available=False, teacher_fallback_calls=0,
        user_fault_injection=False, observation_origin='privileged_native_simulator_state',
        runtime_sha256=runtime, installer_sha256=sha(__file__), asset_provenance=provenance,
        scope='One local native rigid-block simulation attempt; no camera, video training or hardware transfer.')
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in zip(('base-model.json', 'jaw-model.json', 'clock-model.json'), raw):
        _atomic_create(destination/name, content)
    write(destination/'install-started.json', request)
    write(destination/'protocol.json', dict(schema=runner.SCHEMA,
        purpose='single-installed-nominal-attempt', cases=dict(train=[], development=[case], confirmation=[]),
        jaw_mode=mode, jaw_model_raw_sha256=sha(destination/'jaw-model.json')))
    # execute preserves a started marker and refuses to retry interrupted work.
    score = runner.execute(destination, asset_root, case, variant, destination/'attempt',
                           model_path=destination/'clock-model.json', collect_labels=False)
    if runner.runtime_hashes() != runtime:
        raise ValueError('Runtime changed during the installed attempt; evidence retained.')
    receipt = read(destination/'attempt'/'receipt.json')
    manifest = dict(request, status='completed', valid=score['valid'], success=score['success'],
        reason=score['reason'], attempt_receipt_sha256=sha(destination/'attempt'/'receipt.json'),
        files={name: sha(destination/name) for name in
               ('base-model.json', 'jaw-model.json', 'clock-model.json', 'protocol.json', 'install-started.json')},
        attempt_files=receipt['files'])
    write(destination/'install.json', manifest)
    write(destination/'receipt.json', dict(schema=SCHEMA, install_sha256=sha(destination/'install.json'),
        files=manifest['files'], attempt_receipt_sha256=manifest['attempt_receipt_sha256']))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base-model', 'jaw-model', 'model', 'assets', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--start-x', type=float, default=.30)
    parser.add_argument('--start-y', type=float, default=-.03)
    parser.add_argument('--goal', type=int, choices=(0, 1), default=0)
    args = parser.parse_args(argv)
    try:
        result = run_clock_skill(args.base_model, args.jaw_model, args.model, args.assets, args.output,
                                 start_x=args.start_x, start_y=args.start_y, goal=args.goal)
    except (ValueError, TypeError, OSError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, f'Grip-clock attempt could not complete: {type(exc).__name__}: {str(exc)[:180]}\n')
    print(json.dumps({key: result[key] for key in ('schema', 'variant', 'valid', 'success', 'reason')}))
    return 0 if result['valid'] and result['success'] else 1 if result['valid'] else 2


if __name__ == '__main__':
    raise SystemExit(main())

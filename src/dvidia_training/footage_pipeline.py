"""Offline Skillspace footage intake, measured learning and data-only export.

A visual model is not a robot policy. Movement supervision and an explicit
supported task contract are required before a simulation capsule can be built.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import zipfile


LIMITATIONS = [
    'Raw footage trains a small visual prediction model; it does not recover robot actions or prove task success.',
    'Movement training requires timestamped, hash-bound state/action supervision with the supported feature contract.',
    'Movement alignment evidence is declared by its supplier; this tool does not independently attest its accuracy.',
    'An optional capsule uses an authored placement and gripper supervisor, privileged simulator state and one original arm profile.',
    'No physical robot is qualified by this pipeline. Simulation qualification is a separate local test.',
    'Short clips count as complete demonstrations only when their supplier explicitly labels them complete.',
]


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')


def _write(path, value):
    path.write_bytes(_canonical(value) + b'\n')


def _json(path, limit=8 * 1024 * 1024):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError('Expected a bounded local JSON file, without a symbolic link.')
    def pairs(items):
        row = {}
        for key, value in items:
            if key in row:
                raise ValueError('Duplicate JSON field.')
            row[key] = value
        return row
    def finite(token):
        raise ValueError('Nonfinite JSON value.')
    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=finite)
        _canonical(value)
        return value
    except (UnicodeError, RecursionError) as exc:
        raise ValueError('Expected finite UTF-8 JSON.') from exc


def _within(root, relative):
    if type(relative) is not str or not relative or '\\' in relative:
        raise ValueError('Artifact paths must be relative local paths.')
    path = Path(relative)
    if path.is_absolute() or any(part in ('.', '..') for part in path.parts):
        raise ValueError('Artifact path escapes its directory.')
    candidate = root / path
    for parent in [candidate, *candidate.parents]:
        if parent == root.parent:
            break
        if parent.is_symlink():
            raise ValueError('Symbolic links are not supported for training artifacts.')
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError('Artifact path escapes its directory.')
    return candidate


def _fresh(output):
    output = Path(output).absolute()
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('The output directory may not contain symbolic links.')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Use a new or empty output directory; prior runs are never overwritten.')
    output.mkdir(parents=True, exist_ok=True)
    return output


def _counts(dataset):
    episodes = dataset['episodes']
    counts = {name: sum(row['split'] == name for row in episodes) for name in ('train', 'dev', 'test')}
    return {**counts, 'episodes': len(episodes), 'groups': len({row['group_id'] for row in episodes}),
            'total_seconds': sum(row['duration_seconds'] for row in episodes),
            'declared_complete': sum(row.get('complete') is True for row in episodes)}


def _prepared_copy(source, destination):
    """Copy only hashed dataset media/JSON, never other files in a supplied folder."""
    dataset = _json(source)
    if type(dataset) is not dict or dataset.get('kind') != 'dvidia.footage-dataset' or dataset.get('schema_version') != 1:
        raise ValueError('Unsupported prepared dataset.')
    rows = dataset.get('episodes')
    if type(rows) is not list or not 3 <= len(rows) <= 1200:
        raise ValueError('A prepared dataset requires 3–1200 footage records.')
    files = {}
    total = 0
    source_manifest = dataset.get('source_manifest_path')
    original_manifest = _within(source.parent, source_manifest)
    _json(original_manifest, limit=2 * 1024 * 1024)
    manifest_digest = sha256(original_manifest.read_bytes()).hexdigest()
    if manifest_digest != dataset.get('source_manifest_sha256'):
        raise ValueError('Prepared original source manifest is missing or differs from its hash-bound metadata.')
    files[source_manifest] = manifest_digest
    for row in rows:
        if type(row) is not dict:
            raise ValueError('Invalid prepared footage record.')
        for path_key, hash_key, limit in (('media_path', 'media_sha256', 128 * 1024 * 1024),
                                          ('actions_path', 'actions_sha256', 8 * 1024 * 1024)):
            if row.get(path_key) is None:
                if path_key == 'media_path':
                    raise ValueError('Prepared footage is missing media.')
                continue
            relative = row[path_key]
            original = _within(source.parent, relative)
            if not original.is_file() or original.stat().st_size > limit:
                raise ValueError('Prepared media or sidecar is missing or too large.')
            digest = sha256(original.read_bytes()).hexdigest()
            if digest != row.get(hash_key):
                raise ValueError('Prepared media or sidecar digest mismatch.')
            if relative in files and files[relative] != digest:
                raise ValueError('Conflicting prepared artifact hashes.')
            if relative not in files:
                total += original.stat().st_size
                if total > 500 * 1024 * 1024:
                    raise ValueError('Prepared footage exceeds the 500 MiB intake limit.')
                files[relative] = digest
    destination.mkdir()
    for relative in files:
        target = _within(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_within(source.parent, relative), target)
    _write(destination / 'dataset.json', dataset)
    return dataset


def _prepare(source, output, *, offline, seed):
    from .footage_data import prepare, probe_video, _groups
    from .footage_train import _load_dataset
    local = Path(source) if not str(source).startswith('https://') else None
    if local is not None and local.is_file() and local.suffix.lower() == '.json':
        document = _json(local)
        if type(document) is dict and document.get('kind') == 'dvidia.footage-dataset':
            _prepared_copy(local, output)
        else:
            prepare(source, output, offline=offline, seed=seed)
    else:
        prepare(source, output, offline=offline, seed=seed)
    dataset, _, paths, sidecars = _load_dataset(output / 'dataset.json')
    recalculated = deepcopy(dataset['episodes'])
    groups = _groups(recalculated)
    for original, checked in zip(dataset['episodes'], recalculated):
        if original['group_id'] != checked['group_id']:
            raise ValueError('Prepared source-group identities do not match connected recording lineage.')
        measured = probe_video(paths[original['id']])
        for key in ('duration_seconds', 'start_seconds', 'width', 'height', 'fps'):
            if original[key] != measured[key]:
                raise ValueError('Prepared video timing or dimensions do not match the actual media.')
    for group in groups:
        group['split'] = next(row['split'] for row in dataset['episodes'] if row['group_id'] == group['id'])
    dataset['groups'] = groups
    movement_rows = {name: [sample for row in dataset['episodes'] if row['split'] == name and row['id'] in sidecars
                            for sample in sidecars[row['id']]['samples']] for name in ('train', 'dev', 'test')}
    meaningful = {sha256(_canonical({key: sample[key] for key in ('context', 'error', 'delta')})).hexdigest()
                  for sample in movement_rows['train'] if sum(value * value for value in sample['delta']) > 1e-12}
    movement_ready = (all(movement_rows.values()) and 32 <= len(movement_rows['train']) <= 100000
                      and sum(map(len, movement_rows.values())) <= 100000 and len(meaningful) >= 32
                      and sum(row['split'] == 'train' and row['id'] in sidecars for row in dataset['episodes']) <= 64)
    dataset['capability'] = {'visual_ready': True, 'movement_ready': False,
                             'action_sidecars_present': len(sidecars),
                             'movement_training_ready': bool(movement_ready)}
    _write(output / 'dataset.json', dataset)
    return dataset


def prepare_run(source, output, *, offline=True, seed=17):
    """Audit footage and fix group splits without fitting a model."""
    output = _fresh(output)
    started = time.perf_counter()
    dataset = _prepare(source, output / 'dataset', offline=offline, seed=seed)
    summary = {'schema_version': 1, 'kind': 'dvidia.footage-intake', 'status': 'prepared',
               'task': dataset['task'], 'counts': _counts(dataset),
               'capability': dataset['capability'], 'lineage_warnings': dataset.get('lineage_warnings', []),
               'dataset_path': str(output / 'dataset' / 'dataset.json'),
               'elapsed_seconds': time.perf_counter() - started}
    _write(output / 'intake.json', summary)
    return summary


def _receipt(dataset):
    fields = ('id', 'media_sha256', 'duration_seconds', 'start_seconds', 'width', 'height',
              'fps', 'group_id', 'split', 'complete', 'actions_sha256')
    return {'schema_version': 1, 'kind': 'dvidia.footage-dataset-receipt', 'task': dataset['task'],
            'source_manifest_sha256': dataset.get('source_manifest_sha256'),
            'lineage_warnings': dataset.get('lineage_warnings', []), 'counts': _counts(dataset),
            'episodes': [{key: row[key] for key in fields if key in row} for row in dataset['episodes']],
            'media_included': False, 'note': 'Receipt only. Source footage remains in the local prepared dataset.'}


def _artifact(path):
    return {'bytes': path.stat().st_size, 'sha256': sha256(path.read_bytes()).hexdigest()}


def _export(output, names):
    manifest = {'schema_version': 1, 'kind': 'dvidia.training-result-manifest',
                'scope': 'local-training-result', 'physical_robot_ready': False,
                'files': {name: _artifact(_within(output, name)) for name in sorted(names)}}
    _write(output / 'export-manifest.json', manifest)
    with zipfile.ZipFile(output / 'training-result.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in [*sorted(names), 'export-manifest.json']:
            archive.write(_within(output, name), name)
    return manifest


def run(source, output, *, offline=True, seed=17, frames_per_clip=12,
        resolution=16, latent_dim=8, task_source=None, grounding=None, progress=None):
    """Run actual CPU fits on footage. Never promote a visual model to a robot skill."""
    from .footage_train import train, train_movement
    output = _fresh(output)
    started = time.perf_counter()
    def stage(name, message):
        if progress is not None:
            progress({'stage': name, 'message': message})
    stage('intake', 'Checking footage, exact hashes and independent source groups.')
    dataset = _prepare(source, output / 'dataset', offline=offline, seed=seed)
    stage('training', 'Learning a visual model on training clips; selecting settings on development clips.')
    visual = train(output / 'dataset' / 'dataset.json', output / 'training', seed=seed,
                   frames_per_clip=frames_per_clip, resolution=resolution, latent_dim=latent_dim)
    movement = None
    if any(row.get('actions_path') is not None for row in dataset['episodes']):
        stage('training', 'Validating aligned movement supervision and fitting a simulation movement candidate.')
        movement = train_movement(output / 'dataset' / 'dataset.json', output / 'movement', seed=seed)
    if task_source is not None:
        if movement is None:
            raise ValueError('A task capsule requires validated movement supervision; visual footage alone is insufficient.')
        from .arm_runner import authored_grounding
        from .capsule_installer import capsule_runtime
        from .skill_capsule import build_capsule, encode_capsule
        source_file = Path(task_source)
        if source_file.is_symlink() or source_file.stat().st_size > 128 * 1024:
            raise ValueError('Task source must be a bounded local JSON file.')
        raw = source_file.read_bytes()
        resolved_grounding = _json(Path(grounding)) if grounding is not None else (
            None if 'simulation_grounding' in json.loads(raw) else authored_grounding(raw))
        candidate = build_capsule(raw, resolved_grounding, _json(output / 'movement' / 'movement_head.json'),
                                  runtime=capsule_runtime())
        (output / 'candidate.skill-capsule.json').write_bytes(encode_capsule(candidate))
    stage('export', 'Writing model weights, held-out measurements and a data-only download.')
    _write(output / 'dataset-receipt.json', _receipt(dataset))
    names = ['dataset-receipt.json']
    for directory in ('training', 'movement'):
        parent = output / directory
        if parent.exists():
            for path in sorted(parent.iterdir()):
                if path.is_symlink() or not path.is_file() or path.suffix not in ('.json', '.npz'):
                    raise ValueError('Training outputs must be ordinary JSON or numeric NPZ data files.')
                if path.stat().st_size > 64 * 1024 * 1024:
                    raise ValueError('Training output exceeds the 64 MiB artifact limit.')
                names.append(path.relative_to(output).as_posix())
    if (output / 'candidate.skill-capsule.json').exists():
        names.append('candidate.skill-capsule.json')
    summary = {'schema_version': 1, 'kind': 'dvidia.footage-training-run', 'status': 'complete',
               'task': dataset['task'], 'counts': _counts(dataset), 'dataset_path': 'dataset/dataset.json',
               'capability': {'visual_trained': True, 'movement_candidate_trained': movement is not None,
                              'robot_skill_acquired': False, 'physical_robot_ready': False,
                              'simulation_capsule_exported': (output / 'candidate.skill-capsule.json').exists()},
               'visual': visual, 'movement': movement, 'seed': seed, 'offline': offline,
               'elapsed_seconds': time.perf_counter() - started,
               'timing_scope': 'Intake, frame decoding and learning; final receipt hashes, run JSON, ZIP creation and integrity verification excluded.',
               'environment': {'python': platform.python_version(), 'platform': platform.system(),
                               'machine': platform.machine(), 'gpu_required': False},
               'export_file': 'training-result.zip', 'artifacts': {name: _artifact(output / name) for name in names},
               'limitations': LIMITATIONS}
    _write(output / 'run.json', summary)
    _export(output, [*names, 'run.json'])
    stage('complete', 'Training finished. Inspect the held-out result before using the model.')
    return summary


def inspect_run(path):
    """Verify every exported artifact; corruption does not remain a successful run."""
    output = Path(path).absolute()
    summary = _json(output / 'run.json')
    if (type(summary) is not dict or summary.get('kind') != 'dvidia.footage-training-run'
            or summary.get('schema_version') != 1 or summary.get('status') != 'complete'
            or type(summary.get('capability')) is not dict
            or summary['capability'].get('physical_robot_ready') is not False
            or summary['capability'].get('robot_skill_acquired') is not False):
        raise ValueError('No completed footage training result exists here.')
    manifest = _json(output / 'export-manifest.json')
    if (type(manifest) is not dict or manifest.get('kind') != 'dvidia.training-result-manifest'
            or manifest.get('physical_robot_ready') is not False or type(manifest.get('files')) is not dict
            or 'run.json' not in manifest['files'] or not 3 <= len(manifest['files']) <= 20):
        raise ValueError('Invalid training export manifest.')
    expected_files = set(summary.get('artifacts', {})) | {'run.json'}
    permitted = {'dataset-receipt.json', 'run.json', 'training/model.json', 'training/report.json',
                 'training/visual_model.npz', 'movement/movement_head.json', 'movement/report.json',
                 'candidate.skill-capsule.json'}
    if set(manifest['files']) != expected_files or not expected_files <= permitted:
        raise ValueError('Unknown or missing training artifact in export.')
    total_bytes = 0
    for name, receipt in manifest['files'].items():
        artifact = _within(output, name)
        if (type(receipt) is not dict or set(receipt) != {'bytes', 'sha256'} or not artifact.is_file()
                or type(receipt['bytes']) is not int or type(receipt['sha256']) is not str
                or artifact.stat().st_size > 64 * 1024 * 1024 or _artifact(artifact) != receipt):
            raise ValueError('Training result integrity check failed.')
        total_bytes += artifact.stat().st_size
    if total_bytes > 128 * 1024 * 1024:
        raise ValueError('Training export exceeds the 128 MiB expanded limit.')
    from .footage_train import load_visual_model
    load_visual_model(output / 'training' / 'model.json')
    if summary['capability'].get('movement_candidate_trained'):
        from .arm_distill import validate_model
        validate_model(_json(output / 'movement' / 'movement_head.json'))
    if summary['capability'].get('simulation_capsule_exported'):
        from .capsule_installer import capsule_runtime
        from .skill_capsule import inspect_capsule
        inspect_capsule((output / 'candidate.skill-capsule.json').read_bytes(), expected_runtime=capsule_runtime())
    archive_path = _within(output, summary.get('export_file'))
    if not archive_path.is_file() or archive_path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError('Training export download is missing or exceeds its limit.')
    with zipfile.ZipFile(archive_path) as archive:
        expected = {*manifest['files'], 'export-manifest.json'}
        if len(archive.namelist()) != len(expected) or set(archive.namelist()) != expected:
            raise ValueError('Unexpected files in the training download.')
        for name in expected:
            entry = archive.getinfo(name)
            if entry.file_size > 64 * 1024 * 1024 or archive.read(name) != _within(output, name).read_bytes():
                raise ValueError('Training download differs from its verified local artifacts.')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'run'):
        action = commands.add_parser(name)
        action.add_argument('source', help='Skillspace folder, ZIP, manifest, prepared dataset or public DVIDIA URL.')
        action.add_argument('--output', type=Path, required=True)
        action.add_argument('--allow-online', action='store_true', help='Allow explicit public footage downloads during intake.')
        action.add_argument('--seed', type=int, default=17)
        if name == 'run':
            action.add_argument('--frames-per-clip', type=int, default=12)
            action.add_argument('--resolution', type=int, default=16)
            action.add_argument('--latent-dim', type=int, default=8)
            action.add_argument('--task-source', type=Path, help='Explicit supported placement task contract for a movement capsule.')
            action.add_argument('--grounding', type=Path)
    check = commands.add_parser('inspect')
    check.add_argument('directory', type=Path)
    example = commands.add_parser('example', help='Create explicitly synthetic encoded footage for trying the pipeline.')
    example.add_argument('--output', type=Path, required=True)
    example.add_argument('--clips', type=int, default=10)
    example.add_argument('--native-arm', action='store_true', help='Collect actual native simulation telemetry and synchronized schematic videos.')
    args = parser.parse_args()
    try:
        if args.command == 'inspect':
            result = inspect_run(args.directory)
        elif args.command == 'example':
            from .footage_example import create_example
            result = create_example(args.output, clips=args.clips, native_arm=args.native_arm)
        else:
            kwargs = {'offline': not args.allow_online, 'seed': args.seed}
            if args.command == 'prepare':
                result = prepare_run(args.source, args.output, **kwargs)
            else:
                result = run(args.source, args.output, **kwargs, frames_per_clip=args.frames_per_clip,
                             resolution=args.resolution, latent_dim=args.latent_dim,
                             task_source=args.task_source, grounding=args.grounding)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, RuntimeError) as exc:
        print('Training stopped: ' + str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == '__main__':
    main()

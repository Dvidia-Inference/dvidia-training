"""Bounded CPU learning from local RGB footage, with optional aligned motor data.

The visual lane learns pixels, not calibrated geometry or a robot policy.  Motor
sidecars are a separate, hash-bound declaration; this module does not attest the
alignment or run closed-loop physics.  Neither lane establishes physical skill.
"""
from __future__ import annotations

from hashlib import sha256
import io
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
import time
import zipfile

import numpy as np

MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_SIDECAR_BYTES = 8 * 1024 * 1024
MAX_TOTAL_SIDECAR_BYTES = 64 * 1024 * 1024
MAX_MEDIA_BYTES = 128 * 1024 * 1024
MAX_TOTAL_MEDIA_BYTES = 500 * 1024 * 1024
MAX_ARRAY_ELEMENTS = 8_000_000
MAX_NPZ_BYTES = 32 * 1024 * 1024
RIDGE_GRID = (1e-6, 1e-4, 1e-2, 1.)
SPLITS = ('train', 'dev', 'test')
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.webm', '.m4v', '.avi', '.mkv'}
VIDEO_FORMATS = 'mov,matroska,webm,avi'
FEATURE_CONTRACT = 'q6_object_minus_tcp3_target_minus_tcp3__capped_goal_delta3_upright_orientation_delta3_v1'


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise ValueError('Expected finite JSON data.') from exc


def digest(value):
    return sha256(canonical(value)).hexdigest()


def _json(raw, maximum=MAX_JSON_BYTES):
    if not raw or len(raw) > maximum:
        raise ValueError('JSON exceeds its bounded size or is empty.')
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError('Duplicate JSON field.')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('Nonfinite JSON value.')
    try:
        document = json.loads(raw.decode('utf8'), object_pairs_hook=pairs, parse_constant=constant)
        canonical(document)
        return document
    except (UnicodeError, RecursionError) as exc:
        raise ValueError('Expected bounded finite UTF-8 JSON.') from exc


def _read(path, maximum):
    if not path.is_file() or path.stat().st_size > maximum:
        raise ValueError('Artifact is missing or exceeds its size limit.')
    with path.open('rb') as handle:
        result = handle.read(maximum + 1)
    if len(result) > maximum:
        raise ValueError('Artifact exceeds its size limit.')
    return result


def _text(value, label, maximum=120):
    if (type(value) is not str or not value.strip() or len(value) > maximum
            or any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in value)):
        raise ValueError(label + ' must be bounded nonempty text.')
    return value


def _sha(value, label):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(label + ' must be a lowercase SHA256.')
    return value


def _number(value, label, lower, upper):
    if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(label + ' is outside its finite numeric envelope.')
    return value


def _safe_path(root, name):
    _text(name, 'Artifact path', 1000)
    relative = PurePosixPath(name)
    if (relative.is_absolute() or '\\' in name or ':' in name
            or any(part in ('', '.', '..') for part in name.split('/'))):
        raise ValueError('Artifact paths must be relative and remain inside the dataset.')
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError('Symlink artifacts are not accepted.')
    try:
        current.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError('Artifact escaped the dataset directory.') from exc
    if not current.is_file():
        raise ValueError('Dataset artifact is missing.')
    return current


def _unsymlinked(path):
    # An explicitly provided directory or file cannot hide a symlink in a parent.
    path = Path(path).absolute()
    system_aliases = {Path('/tmp'): Path('/private/tmp'), Path('/var'): Path('/private/var')}
    for part in (path, *path.parents):
        if part.is_symlink() and not (part in system_aliases and part.resolve() == system_aliases[part]):
            raise ValueError('Symlink inputs or outputs are not accepted.')
    return path.resolve()


def _load_dataset(dataset_path):
    path = _unsymlinked(dataset_path)
    raw = _read(path, MAX_JSON_BYTES)
    dataset = _json(raw)
    if (type(dataset) is not dict or dataset.get('kind') != 'dvidia.footage-dataset'
            or type(dataset.get('schema_version')) is not int or dataset['schema_version'] != 1):
        raise ValueError('Expected a schema-1 dvidia.footage-dataset.')
    _text(dataset.get('task'), 'Task', 500)
    _sha(dataset.get('source_manifest_sha256'), 'Source manifest hash')
    if type(dataset.get('capability')) is not dict:
        raise ValueError('Dataset capability must be a JSON object.')
    episodes = dataset.get('episodes')
    if type(episodes) is not list or not 3 <= len(episodes) <= 1200:
        raise ValueError('Use 3–1200 episodes with all three source-group splits.')
    required = {'id', 'media_path', 'media_sha256', 'duration_seconds', 'start_seconds',
                'width', 'height', 'fps', 'source_recording_id', 'session_id', 'shoe_pair_id',
                'complete', 'task', 'group_id', 'split'}
    optional = {'media_bytes', 'codec', 'actions_path', 'actions_sha256', 'actions_bytes', 'actions_validation'}
    seen, identity_splits, paths, sidecars, total = set(), {}, {}, {}, 0
    movement_splits = {}
    action_bytes, action_samples = 0, 0
    for episode in episodes:
        if type(episode) is not dict or not required <= set(episode) or set(episode) - required - optional:
            raise ValueError('Episode fields are missing or unsupported.')
        identifier = _text(episode['id'], 'Episode ID', 80)
        if identifier in seen:
            raise ValueError('Episode IDs must be distinct.')
        seen.add(identifier)
        _text(episode['task'], 'Episode task', 500)
        if episode['task'] != dataset['task']:
            raise ValueError('Every episode must describe the dataset task.')
        if type(episode['complete']) is not bool:
            raise ValueError('Complete must be an explicit boolean declaration.')
        if episode['split'] not in SPLITS:
            raise ValueError('Each episode needs train, dev or test split.')
        _text(episode['group_id'], 'Group ID')
        _sha(episode['media_sha256'], 'Media hash')
        _number(episode['duration_seconds'], 'Duration', .05, 600)
        _number(episode['start_seconds'], 'Stream start', 0, 86400)
        _number(episode['fps'], 'Frame rate', .1, 1000)
        if any(type(episode[key]) is not int or not 1 <= episode[key] <= 16384 for key in ('width', 'height')):
            raise ValueError('Frame dimensions exceed the supported envelope.')
        for key in ('group_id', 'media_sha256', 'source_recording_id', 'session_id', 'shoe_pair_id'):
            value = episode[key]
            if value is None and key not in ('group_id', 'media_sha256'):
                continue
            _text(value, key)
            identity = (key, value)
            if identity in identity_splits and identity_splits[identity] != episode['split']:
                raise ValueError('Source group, recording/session/shoe identity or media hash crosses splits.')
            identity_splits[identity] = episode['split']
        media = _safe_path(path.parent, episode['media_path'])
        if media.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError('Media must use a supported ordinary video-container extension.')
        size = media.stat().st_size
        total += size
        if not 0 < size <= MAX_MEDIA_BYTES or total > MAX_TOTAL_MEDIA_BYTES:
            raise ValueError('Dataset media exceeds bounded size limits.')
        with media.open('rb') as handle:
            media_hash = sha256()
            while block := handle.read(1024 * 1024):
                media_hash.update(block)
        if media_hash.hexdigest() != episode['media_sha256']:
            raise ValueError('Media SHA256 mismatch.')
        if 'media_bytes' in episode and (type(episode['media_bytes']) is not int or episode['media_bytes'] != size):
            raise ValueError('Media byte count mismatch.')
        paths[identifier] = media
        action_path, action_hash = episode.get('actions_path'), episode.get('actions_sha256')
        if (action_path is None) != (action_hash is None):
            raise ValueError('Action sidecar path and hash must be supplied together.')
        if action_path is not None:
            _sha(action_hash, 'Action hash')
            raw_action = _read(_safe_path(path.parent, action_path), MAX_SIDECAR_BYTES)
            action_bytes += len(raw_action)
            if action_bytes > MAX_TOTAL_SIDECAR_BYTES:
                raise ValueError('Total action-sidecar JSON exceeds 64 MiB.')
            if sha256(raw_action).hexdigest() != action_hash:
                raise ValueError('Action sidecar SHA256 mismatch.')
            sidecar = validate_movement_sidecar(_json(raw_action, MAX_SIDECAR_BYTES), episode)
            action_samples += len(sidecar['samples'])
            if action_samples > 100000:
                raise ValueError('Total aligned movement samples exceed 100000.')
            # Different video encodings, IDs or timestamps cannot turn one motor
            # trajectory into independent train and evaluation demonstrations.
            trajectory = [{key: row[key] for key in ('context', 'error', 'delta')}
                          for row in sidecar['samples']]
            bindings = (sidecar['provenance']['validation']['samples_sha256'], digest(trajectory))
            for binding in bindings:
                if binding in movement_splits and movement_splits[binding] != episode['split']:
                    raise ValueError('Duplicate movement trajectory crosses train/dev/test splits.')
                movement_splits[binding] = episode['split']
            sidecars[identifier] = sidecar
    if {row['split'] for row in episodes} != set(SPLITS):
        raise ValueError('Training requires nonempty train, dev and test splits.')
    return dataset, sha256(raw).hexdigest(), paths, sidecars


def validate_dataset(dataset_path: Path):
    """Return the dataset after validating media, lineage and advertised actions."""
    dataset, _, _, _ = _load_dataset(dataset_path)
    return dataset


def _vector(value, length, limit, label):
    if (type(value) is not list or len(value) != length
            or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > limit for v in value)):
        raise ValueError(label + ' must have finite bounded numerical dimensions.')
    return np.asarray(value, dtype=np.float64)


def validate_movement_sidecar(document, episode):
    """Validate a declared alignment record; this is not third-party attestation."""
    fields = {'kind', 'schema_version', 'feature_contract', 'media_sha256', 'samples', 'provenance'}
    if type(document) is not dict or set(document) != fields:
        raise ValueError('Movement sidecar fields must match the strict data-only schema.')
    if (document['kind'] != 'dvidia.aligned-movement-demonstration'
            or type(document['schema_version']) is not int or document['schema_version'] != 1
            or document['feature_contract'] != FEATURE_CONTRACT
            or document['media_sha256'] != episode['media_sha256']):
        raise ValueError('Movement sidecar contract, version or media binding mismatch.')
    rows = document['samples']
    if type(rows) is not list or not 2 <= len(rows) <= 10000:
        raise ValueError('Use 2–10000 aligned samples per clip.')
    previous = -1.
    for row in rows:
        if type(row) is not dict or set(row) != {'timestamp_seconds', 'context', 'error', 'delta'}:
            raise ValueError('Movement samples require timestamp/context12/error6/delta6 only.')
        timestamp = _number(row['timestamp_seconds'], 'Sample timestamp', 0, episode['duration_seconds'])
        if timestamp <= previous:
            raise ValueError('Movement timestamps must be strictly increasing and clip-relative.')
        previous = timestamp
        _vector(row['context'], 12, 20., 'Context')
        error = _vector(row['error'], 6, 1.00001, 'Error')
        if np.linalg.norm(error[:3]) > 1.00001 or np.linalg.norm(error[3:]) > 1.00001:
            raise ValueError('Pose errors must respect the normalized translation/orientation caps.')
        _vector(row['delta'], 6, .06001, 'Joint delta')
    provenance = document['provenance']
    if type(provenance) is not dict or set(provenance) != {'source_kind', 'validation'}:
        raise ValueError('Movement provenance requires source_kind and a validation record.')
    if provenance['source_kind'] not in ('robot-teleoperation', 'validated-video-action-bridge', 'native-simulation-telemetry'):
        raise ValueError('Movement source must be teleoperation, a validated bridge or native simulation telemetry.')
    record = provenance['validation']
    expected = {'kind': 'dvidia.movement-alignment-validation', 'schema_version': 1,
                'status': 'validated', 'scope': 'simulation-only', 'media_sha256': episode['media_sha256'],
                'samples_sha256': digest(rows), 'feature_contract': FEATURE_CONTRACT}
    if type(record) is not dict or set(record) != set(expected) | {'method', 'evidence_sha256'}:
        raise ValueError('A strict hash-bound movement alignment validation record is required.')
    if any(type(record[k]) is not type(v) or record[k] != v for k, v in expected.items()):
        raise ValueError('Movement validation record has incorrect sample/media/feature bindings.')
    _text(record['method'], 'Alignment validation method', 1000)
    _sha(record['evidence_sha256'], 'Alignment evidence hash')
    return document


def _extract_frames(media, episode, count, resolution):
    # start_seconds is stream metadata, not an instruction to trim that offset.
    rate = count / episode['duration_seconds']
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1', '-filter_threads', '1',
               '-protocol_whitelist', 'file,pipe', '-format_whitelist', VIDEO_FORMATS,
               '-i', str(media), '-an', '-sn', '-dn',
               '-t', str(episode['duration_seconds']), '-vf',
               f'fps={rate:.12g},scale={resolution}:{resolution}:flags=area',
               '-frames:v', str(count), '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError('ffmpeg must decode each local clip within 30 seconds.') from exc
    expected = count * resolution * resolution * 3
    if result.returncode or len(result.stdout) != expected:
        raise ValueError('Video did not decode the required number of RGB frames.')
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(count, -1).astype(np.float32) / 255.


def _pca(frames, dimensions, seed):
    mean = frames.mean(axis=0, dtype=np.float64)
    centered = frames.astype(np.float64) - mean
    components = min(dimensions, len(frames) - 1, frames.shape[1])
    # Randomized range finding keeps the CPU work bounded for hundreds of clips.
    width = min(components + 8, *centered.shape)
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(centered @ rng.normal(size=(centered.shape[1], width)), mode='reduced')
    for _ in range(2):
        q, _ = np.linalg.qr(centered @ (centered.T @ q), mode='reduced')
    _, singular, basis = np.linalg.svd(q.T @ centered, full_matrices=False)
    return mean, basis[:components], singular[:components]


def _pairs(frames, mean, basis):
    pairs = []
    for episode, pixels in frames:
        latent = (pixels - mean) @ basis.T
        pairs.append((episode, pixels[:-1], pixels[1:], latent[:-1], latent[1:]))
    return pairs


def _fit_transition(pairs, regularization):
    x = np.concatenate([row[3] for row in pairs])
    y = np.concatenate([row[4] for row in pairs])
    x = np.column_stack((x, np.ones(len(x))))
    penalty = np.eye(x.shape[1]) * regularization
    penalty[-1, -1] = 0.
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def _visual_metrics(pairs, mean, basis, transition):
    metrics = []
    for episode, current, target, latent, target_latent in pairs:
        prediction_latent = np.column_stack((latent, np.ones(len(latent)))) @ transition
        prediction = np.clip(prediction_latent @ basis + mean, 0., 1.)
        metrics.append({'episode_id': episode['id'], 'group_id': episode['group_id'],
                        'transitions': len(target), 'learned_rgb_mse': float(np.mean((prediction-target)**2)),
                        'persistence_rgb_mse': float(np.mean((current-target)**2)),
                        'train_mean_rgb_mse': float(np.mean((mean-target)**2)),
                        'learned_latent_mse': float(np.mean((prediction_latent-target_latent)**2))})
    count = sum(row['transitions'] for row in metrics)
    aggregate = {key: sum(row[key] * row['transitions'] for row in metrics) / count
                 for key in ('learned_rgb_mse', 'persistence_rgb_mse', 'train_mean_rgb_mse', 'learned_latent_mse')}
    return {'episodes': len(metrics), 'transitions': count, **aggregate, 'per_episode': metrics}


def _output_path(output):
    path = _unsymlinked(output)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError('Training output must be a new or empty directory; frozen reports are not overwritten.')
    return path


def _write_outputs(output, json_files, npz_arrays=None):
    # Complete all validation and fitting first. A staged rename avoids partial output.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.footage-train-', dir=output.parent) as name:
        staged = Path(name) / 'result'
        staged.mkdir()
        if npz_arrays is not None:
            np.savez_compressed(staged/'visual_model.npz', **npz_arrays)
            npz_hash = sha256((staged/'visual_model.npz').read_bytes()).hexdigest()
            json_files['model.json']['arrays_sha256'] = npz_hash
            json_files['report.json']['visual_model_arrays_sha256'] = npz_hash
        for filename, document in json_files.items():
            (staged/filename).write_bytes(canonical(document)+b'\n')
        if output.exists():
            output.rmdir()  # Already checked empty; concurrent writes make this fail.
        staged.rename(output)


def train(dataset_path: Path, output: Path, *, seed=17, frames_per_clip=12, resolution=16, latent_dim=8):
    """Fit train-only PCA/ridge; dev selects ridge and sealed test is scored once."""
    if type(seed) is not int or not 0 <= seed <= 999999:
        raise ValueError('Use a seed in [0, 999999].')
    for value, lower, upper, label in ((frames_per_clip, 4, 32, 'Frames per clip'),
                                      (resolution, 8, 32, 'Resolution'), (latent_dim, 1, 32, 'Latent dimension')):
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(label + ' is outside the CPU envelope.')
    output = _output_path(output)
    before = time.perf_counter()
    dataset, dataset_hash, paths, sidecars = _load_dataset(dataset_path)
    if len(dataset['episodes']) * frames_per_clip * resolution * resolution * 3 > MAX_ARRAY_ELEMENTS:
        raise ValueError('Decoded RGB arrays exceed eight million numerical elements; reduce resolution/frames.')
    frames = {split: [] for split in SPLITS}
    for episode in dataset['episodes']:
        if episode['fps'] * episode['duration_seconds'] < frames_per_clip - .5:
            raise ValueError('Clip has fewer original frames than the requested training sample count.')
        frames[episode['split']].append((episode, _extract_frames(paths[episode['id']], episode, frames_per_clip, resolution)))
    mean, basis, singular = _pca(np.concatenate([row[1] for row in frames['train']]), latent_dim, seed)
    pairs = {split: _pairs(frames[split], mean, basis) for split in SPLITS}
    choices = []
    for ridge in RIDGE_GRID:
        transition = _fit_transition(pairs['train'], ridge)
        dev = _visual_metrics(pairs['dev'], mean, basis, transition)
        choices.append((dev['learned_rgb_mse'], ridge, transition))
    _, ridge, transition = min(choices, key=lambda item: (item[0], item[1]))
    arrays = {'mean': mean.astype(np.float32), 'basis': basis.astype(np.float32),
              'transition': transition.astype(np.float32)}
    if not all(np.isfinite(array).all() and np.max(np.abs(array)) <= 1e6 for array in arrays.values()):
        raise ValueError('Trained numerical model is outside its finite float32 envelope.')
    # Metrics use precisely the float32 arrays that are shipped, promoted for arithmetic.
    mean, basis, transition = (arrays[key].astype(np.float64) for key in ('mean', 'basis', 'transition'))
    counts = {split: {'episodes': len(frames[split]), 'complete_declared': sum(row[0]['complete'] for row in frames[split]),
                      'groups': len({row[0]['group_id'] for row in frames[split]}),
                      'frames': len(frames[split])*frames_per_clip} for split in SPLITS}
    model = {'kind': 'dvidia.cpu-visual-transition', 'schema_version': 1, 'scope': 'visual-only',
             'physical_robot_ready': False, 'dataset_sha256': dataset_hash,
             'source_manifest_sha256': dataset['source_manifest_sha256'],
             'arrays_path': 'visual_model.npz', 'arrays_sha256': None,
             'arrays': {key: {'dtype': 'float32', 'shape': list(array.shape)} for key, array in arrays.items()},
             'training': {'algorithm': 'train-only-randomized-pca-affine-ridge', 'seed': seed,
                          'frames_per_clip': frames_per_clip, 'resolution': resolution,
                          'latent_dim': len(basis), 'regularization': ridge, 'ridge_grid': list(RIDGE_GRID),
                          'train_episode_ids': [row[0]['id'] for row in frames['train']],
                          'selection_episode_ids': [row[0]['id'] for row in frames['dev']]}}
    report = {'kind': 'dvidia.footage-training-report', 'schema_version': 1, 'scope': 'visual-only',
              'physical_robot_ready': False, 'robot_policy_trained': False,
              'capability': {'visual_model_trained': True, 'movement_candidate_trained': False},
              'dataset_sha256': dataset_hash, 'source_manifest_sha256': dataset['source_manifest_sha256'],
              'counts': counts, 'parameters': model['training'], 'selection': [
                  {'regularization': row[1], 'dev_rgb_mse': row[0]} for row in choices],
              'dev': _visual_metrics(pairs['dev'], mean, basis, transition),
              'test': _visual_metrics(pairs['test'], mean, basis, transition),
              'train_pca_singular_values': singular.tolist(),
              'visual_model_arrays_sha256': None, 'wall_seconds': time.perf_counter()-before,
              'action_sidecars_validated': len(sidecars),
              'limitations': ['RGB MSE uses normalized uncalibrated pixels; it is not geometry, task success or robot competence.',
                  'Frames/transitions within clips are correlated; complete/group counts are declarations, not independent performance trials.',
                  'PCA and transition weights use train only; dev selects the declared ridge; test is scored after selection.',
                  'Lineage and action-alignment hashes establish declared consistency, not source authenticity or independent validation.',
                  'Motor sidecars, if present, require the separate train_movement call and closed-loop qualification.']}
    _write_outputs(output, {'model.json': model, 'report.json': report}, arrays)
    return report


def load_visual_model(model_path: Path):
    """Load only bounded numeric NPZ data with its hash and exact array schema."""
    path = _unsymlinked(model_path)
    model = _json(_read(path, MAX_JSON_BYTES))
    fields = {'kind', 'schema_version', 'scope', 'physical_robot_ready', 'dataset_sha256',
              'source_manifest_sha256', 'arrays_path', 'arrays_sha256', 'arrays', 'training'}
    if (type(model) is not dict or set(model) != fields or model['kind'] != 'dvidia.cpu-visual-transition'
            or type(model['schema_version']) is not int or model['schema_version'] != 1
            or model['scope'] != 'visual-only' or model['physical_robot_ready'] is not False
            or model['arrays_path'] != 'visual_model.npz'):
        raise ValueError('Unsupported visual model schema or scope.')
    for key in ('arrays_sha256', 'dataset_sha256', 'source_manifest_sha256'):
        _sha(model[key], key)
    training = model['training']
    training_fields = {'algorithm', 'seed', 'frames_per_clip', 'resolution', 'latent_dim',
                       'regularization', 'ridge_grid', 'train_episode_ids', 'selection_episode_ids'}
    if (type(training) is not dict or set(training) != training_fields
            or training['algorithm'] != 'train-only-randomized-pca-affine-ridge'
            or type(training['seed']) is not int or not 0 <= training['seed'] <= 999999
            or training['ridge_grid'] != list(RIDGE_GRID)
            or type(training['regularization']) not in (int, float) or training['regularization'] not in RIDGE_GRID):
        raise ValueError('Invalid visual model training metadata.')
    for key, low, high in (('frames_per_clip', 4, 32), ('resolution', 8, 32), ('latent_dim', 1, 32)):
        if type(training[key]) is not int or not low <= training[key] <= high:
            raise ValueError('Visual model training dimensions are invalid.')
    for key in ('train_episode_ids', 'selection_episode_ids'):
        values = training[key]
        if type(values) is not list or not 1 <= len(values) <= 1200:
            raise ValueError('Visual training IDs must be bounded lists.')
        for value in values:
            _text(value, 'Training episode ID', 80)
        if len(set(values)) != len(values):
            raise ValueError('Visual training IDs must be distinct.')
    if set(training['train_episode_ids']) & set(training['selection_episode_ids']):
        raise ValueError('Visual training and selection IDs overlap.')
    raw = _read(_safe_path(path.parent, model['arrays_path']), MAX_NPZ_BYTES)
    if sha256(raw).hexdigest() != model['arrays_sha256']:
        raise ValueError('Visual array SHA256 mismatch.')
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            infos = archive.infolist()
            if ({entry.filename for entry in infos} != {'mean.npy', 'basis.npy', 'transition.npy'}
                    or len(infos) != 3 or sum(entry.file_size for entry in infos) > MAX_NPZ_BYTES
                    or any(entry.flag_bits & 1 for entry in infos)):
                raise ValueError('Visual NPZ must contain only three bounded numerical arrays.')
            shapes = {'mean': (3*training['resolution']**2,),
                      'basis': (training['latent_dim'], 3*training['resolution']**2),
                      'transition': (training['latent_dim']+1, training['latent_dim'])}
            # Check headers before np.load can allocate arrays from hostile shapes.
            for entry in infos:
                with archive.open(entry) as handle:
                    version = np.lib.format.read_magic(handle)
                    if version == (1, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
                    elif version == (2, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
                    else:
                        raise ValueError('Unsupported visual NPY header.')
                    key = entry.filename[:-4]
                    if shape != shapes[key] or dtype != np.dtype('float32') or fortran:
                        raise ValueError('Visual array header has invalid dimensions or data type.')
                    if entry.file_size != handle.tell() + math.prod(shape)*dtype.itemsize:
                        raise ValueError('Visual array payload size mismatch.')
        with np.load(io.BytesIO(raw), allow_pickle=False) as package:
            arrays = {key: package[key].copy() for key in ('mean', 'basis', 'transition')}
    except (OSError, KeyError, zipfile.BadZipFile, TypeError) as exc:
        raise ValueError('Visual model arrays could not be safely loaded.') from exc
    mean, basis, transition = (arrays[key] for key in ('mean', 'basis', 'transition'))
    if (mean.ndim != 1 or not 192 <= len(mean) <= 3072 or basis.ndim != 2
            or not 1 <= len(basis) <= 32 or basis.shape[1] != len(mean)
            or transition.shape != (len(basis)+1, len(basis)) or type(model['arrays']) is not dict
            or set(model['arrays']) != set(arrays)):
        raise ValueError('Visual model numerical dimensions are invalid.')
    for key, array in arrays.items():
        if (array.dtype != np.dtype('float32') or not np.isfinite(array).all() or np.max(np.abs(array)) > 1e6
                or model['arrays'][key] != {'dtype': 'float32', 'shape': list(array.shape)}):
            raise ValueError('Visual model array schema or finite float32 range mismatch.')
    return model, arrays


def validate_saved_model(output: Path):
    """Return JSON metadata after checking the visual artifact's complete payload."""
    model, _ = load_visual_model(Path(output)/'model.json')
    return model


def _movement_predictions(model, rows):
    from .arm_env import JOINT_LIMITS
    context = np.asarray([row['context'] for row in rows], dtype=np.float64)
    errors = np.asarray([row['error'] for row in rows], dtype=np.float64)
    centers = np.asarray(model['centers'], dtype=np.float64)
    normalized = (context-np.asarray(model['context_mean'])) / np.asarray(model['context_scale'])
    logits = -np.sum((normalized[:, None, :]-centers)**2, axis=-1)/(2*model['kernel_width']**2)
    logits -= logits.max(axis=1, keepdims=True)
    kernel = np.exp(logits)
    kernel /= kernel.sum(axis=1, keepdims=True)
    design = (kernel[:, :, None]*errors[:, None, :]).reshape(len(rows), -1)
    delta = np.clip(design @ np.asarray(model['weights']).reshape(-1, 6), -.06, .06)
    limits = np.asarray(JOINT_LIMITS)
    return np.clip(context[:, :6]+delta, limits[:, 0]+.02, limits[:, 1]-.02)-context[:, :6]


def train_movement(dataset_path: Path, output: Path, *, seed=17):
    """Fit a simulation candidate from aligned actions, never from RGB alone."""
    from .arm_entrypoints import require_arm
    require_arm()
    from .arm_distill import train as train_head, validate_model
    if type(seed) is not int or not 0 <= seed <= 999999:
        raise ValueError('Use a seed in [0, 999999].')
    output = _output_path(output)
    dataset, dataset_hash, _, sidecars = _load_dataset(dataset_path)
    rows, episodes = {split: [] for split in SPLITS}, {split: [] for split in SPLITS}
    provenance = []
    for episode in dataset['episodes']:
        if episode['id'] not in sidecars:
            continue
        sidecar = sidecars[episode['id']]
        episodes[episode['split']].append(episode['id'])
        rows[episode['split']].extend({'context': row['context'], 'error': row['error'], 'delta': row['delta'],
                                      'case_id': episode['id']} for row in sidecar['samples'])
        provenance.append({'episode_id': episode['id'], 'split': episode['split'],
                           'actions_sha256': episode['actions_sha256'], **sidecar['provenance']})
    if not all(rows[split] for split in SPLITS) or not 32 <= len(rows['train']) <= 100000 or len(episodes['train']) > 64:
        raise ValueError('Movement requires aligned train/dev/test episodes, 32–100000 train samples and at most 64 train cases.')
    if sum(len(value) for value in rows.values()) > 100000:
        raise ValueError('Total movement samples exceed 100000.')
    meaningful = {digest({key: row[key] for key in ('context', 'error', 'delta')})
                  for row in rows['train'] if np.linalg.norm(row['delta']) > 1e-6}
    if len(meaningful) < 32:
        raise ValueError('Movement training needs at least 32 distinct nonzero action examples.')
    training_data = {'kind': 'dvidia.footage-aligned-movement-training', 'schema_version': 1,
                     'feature_contract': FEATURE_CONTRACT, 'source_dataset_sha256': dataset_hash,
                     'scene_ids': episodes['train'], 'samples': rows['train'],
                     'source_sidecars': [row for row in provenance if row['split'] == 'train']}
    target = {split: np.asarray([row['delta'] for row in rows[split]]) for split in SPLITS}
    candidates = []
    for regularization in RIDGE_GRID:
        model = train_head(training_data, centers=8, regularization=regularization, seed=seed)
        dev_mse = float(np.mean((_movement_predictions(model, rows['dev'])-target['dev'])**2))
        candidates.append((dev_mse, regularization, model))
    _, _, model = min(candidates, key=lambda item: (item[0], item[1]))
    model = validate_model(model)
    mean_delta = target['train'].mean(axis=0)
    def metrics(split):
        predictions = _movement_predictions(model, rows[split])
        return {'episodes': len(episodes[split]), 'samples': len(rows[split]),
                'learned_joint_delta_mse': float(np.mean((predictions-target[split])**2)),
                'zero_delta_mse': float(np.mean(target[split]**2)),
                'train_mean_delta_mse': float(np.mean((mean_delta-target[split])**2))}
    report = {'kind': 'dvidia.footage-movement-training-report', 'schema_version': 1,
              'scope': 'simulation-only', 'status': 'candidate', 'physical_robot_ready': False,
              'capability': {'movement_candidate_trained': True, 'closed_loop_qualified': False},
              'dataset_sha256': dataset_hash, 'movement_head_sha256': digest(model),
              'train_episode_ids': episodes['train'], 'training': model['training'],
              'counts': {split: {'episodes': len(episodes[split]), 'samples': len(rows[split])} for split in SPLITS},
              'selection': [{'regularization': row[1], 'dev_joint_delta_mse': row[0]} for row in candidates],
              'dev': metrics('dev'), 'test': metrics('test'), 'alignment_records': provenance,
              'limitations': ['Actions are learned from aligned numerical supervision, not inferred from pixels.',
                  'Alignment records and external evidence hashes are declared consistency, not independently attested validation.',
                  'Native simulation telemetry is simulation-only; it is not human-video calibration or physical robot evidence.',
                  'Joint-delta MSE is offline imitation error; native closed-loop scene qualification is required separately.',
                  'The learned head retains an authored phase/contact/gripper supervisor and privileged simulator features.']}
    _write_outputs(output, {'movement_head.json': model, 'report.json': report})
    return report

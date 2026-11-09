"""Exact causal servo references and feasible labels beside a pinned v0.1 cache.

This additive Python API verifies source metadata, not raw RGB-D/native replay.
It does not change learner inputs, fit a model, or execute a policy or simulator.
Caller pins must be kept independently. Preserve source files immutably and use
exclusive write access while constructing a fresh derived artifact.
"""
from __future__ import annotations

from hashlib import sha256
import io
import math
from pathlib import Path

import numpy as np

from . import camera_action_cache as cache
from . import camera_action_model as model

SCHEMA = 'dvidia.camera-action-reference-labels/v1'
FILES = {'manifest.json', 'previous-references.npy', 'feasible-targets.npy'}
MAX_SAMPLES = 128_000
SCOPE = ('Derived training labels and exact prior normal servo references; metadata bytes verified, '
         'not raw recording revalidation, source authentication, inference, physics or qualification.')
SOURCE_PINS = {
    'piper_task.py': 'e66a44a8df4e7a8ee42e4f3866f889f8797cdfb8e2688a146e09b75d2c76664f',
    'piper_imitation.py': '849f66512c58441a240cd20230fcc58691234ddd254c5bbcbfd76abc21b14868',
    'camera_action_model.py': 'e9a9c4d1d0f16fc7c91cac3ca2e6560d0532370ab78a2af5a4ec84425c23932f',
    'camera_action_export.py': '2c4e30587605edc0342150a93430a70ed0559c3ee3eda17d93c5670831370b02',
    'piper_action_capture.py': 'da037b2198cf7d2f3a9aa40fc19a4c0f5cc375a22c5dd75606e2c1856f6efc7f',
    'camera_action_capture.py': '399f512a08087c72bfa37e382157a2250f413e423974b77b1bd3c0a911ff0a1e',
    'robot_profiles.lock.json': '4b2ee2ca675aceb2f826983cb3082a3e08e9aa0ff419b23078abcff08ebcc6fe',
    'camera_action_cache.py': '9191f2eb9f96bd09d4f711c31966a583bd938853a874e6018f8ddebfb0bc00d8',
}
CONTRACT = dict(reference_kind='previous_applied_normal_servo_reference',
    reset_source='initial_applied_targets', later_source='steps[i-1].applied.ramped_targets',
    joint_order=[f'joint{i}' for i in range(1, 7)], action_units=['rad']*6 + ['single_finger_m'],
    limits_native=[list(x) for x in model.core.JOINT_LIMITS] + [[0., .035]],
    timestep_s=.001, control_dt_s=.02, microsteps=20, joint_rate_rad_s=1.5, jaw_rate_m_s=.05,
    endpoint_algorithm='20 original float64 reference updates, not closed-form projection',
    clock_tolerance_s=1e-9, numerical_equivalence_tolerance_native=1e-12,
    actual_pulse_overridden_controls_are_not_references=True,
    limiter_ast_sha256='a5356fb9e1bd8b1dc6e6b2b14d657fbe625e018ba8d1e5e668ba4d11191f7921')


def source_identity():
    """Bind this helper, unchanged parent implementation and fixed limiter source."""
    package = Path(__file__).parent
    actual = {name: sha256(cache._bytes(package/name, 8*1024**2)).hexdigest() for name in SOURCE_PINS}
    if actual != SOURCE_PINS:
        raise ValueError('Reference export requires the exact reviewed frozen source files.')
    return dict(source_sha256=actual, parent_source=cache.source_identity(), contract=CONTRACT,
                helper_sha256=sha256(cache._bytes(Path(__file__), 8*1024**2)).hexdigest())


def source_digest():
    return cache._digest(source_identity())


def _matrix(value, name):
    if (not isinstance(value, np.ndarray) or value.dtype.str != '<f8' or value.ndim != 2
            or value.shape[1:] != (7,) or not 1 <= len(value) <= MAX_SAMPLES
            or not value.flags.c_contiguous or not np.isfinite(value).all()):
        raise ValueError(name + ' requires a bounded finite C-order float64 [N,7] array.')
    bounds = np.asarray(CONTRACT['limits_native'], dtype=np.float64)
    if np.any(value < bounds[:, 0]) or np.any(value > bounds[:, 1]):
        raise ValueError(name + ' exceeds original joint/single-finger bounds.')
    return value


def _reference_updates(previous_references, targets):
    references = _matrix(previous_references, 'Previous references').copy()
    targets = _matrix(targets, 'Targets')
    if targets.shape != references.shape:
        raise ValueError('Previous references and targets must have identical shapes.')
    # Same float64 subtraction, clipping and addition order as the two pinned
    # native reference statements. The jaw is its NORMAL single-finger state,
    # never pulse-overridden data.ctrl, measured q or a reference reconstructed
    # from float32 features. This is only numerical reference evolution.
    for _ in range(20):
        references[:, :6] += np.clip(targets[:, :6]-references[:, :6], -1.5*.001, 1.5*.001)
        references[:, 6] += np.clip(targets[:, 6]-references[:, 6], -.05*.001, .05*.001)
        yield references.copy()


def feasible_targets(previous_references, targets):
    """Return the original 20-update endpoints, without modifying either input."""
    for endpoint in _reference_updates(previous_references, targets):
        pass
    return endpoint


def _pins(expected_parent_manifest_sha256, expected_parent_source_sha256,
          expected_protocol_sha256, expected_stage, expected_split, expected_source_sha256):
    if expected_split != 'train' or expected_stage not in ('demonstrations', 'corrections'):
        raise ValueError('Reference labels require an explicit training stage; no development or confirmation.')
    return dict(parent_manifest_sha256=cache._hash(expected_parent_manifest_sha256, 'parent manifest'),
                parent_source_sha256=cache._hash(expected_parent_source_sha256, 'parent source'),
                protocol_sha256=cache._hash(expected_protocol_sha256, 'protocol'),
                stage=expected_stage, split=expected_split,
                source_sha256=cache._hash(expected_source_sha256, 'reference source'))


def _parent(directory, pins):
    directory = cache._path(directory)
    dataset = cache.load_cache(directory,
        expected_manifest_sha256=pins['parent_manifest_sha256'],
        expected_protocol_sha256=pins['protocol_sha256'], expected_split=pins['split'],
        expected_source_sha256=pins['parent_source_sha256'])
    manifest, digest = cache._json(directory/'manifest.json')
    if (digest != pins['parent_manifest_sha256'] or manifest['blind'] is not False
            or manifest['binding']['stage'] != pins['stage']):
        raise ValueError('Parent must retain the pinned full-feedback training stage.')
    return dataset, manifest


def _time(value, expected):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not math.isclose(value, expected, rel_tol=0., abs_tol=1e-9)):
        raise ValueError('Reference row timestamps must preserve complete causal 20 ms intervals.')


def _metadata(root, episode):
    path = episode['path']
    expected_path = f'episodes/{episode["label_stage"]}/{episode["attempt_id"]}'
    if path != expected_path or Path(path).is_absolute() or '..' in Path(path).parts:
        raise ValueError('Unregistered episode metadata path.')
    bodies = {name: cache._bytes(root/path/name, 64*1024**2)
              for name in ('recording.json', 'integrity.json', 'rollout.json')}
    if {name: sha256(body).hexdigest() for name, body in bodies.items()} != episode['metadata_sha256']:
        raise ValueError('Recording metadata differs from the independently pinned parent cache.')
    return model.core._decode(bodies['recording.json'])


def _references(recording, source, episode, parent_targets):
    """Only reset/preceding normal references become output; queries stay labels."""
    label = episode['label_source']
    mode = 'teacher' if label == 'issued' else 'mixed'
    steps = recording.get('steps')
    if (recording.get('schema') != 'dvidia.camera-action-rollout/v2'
            or recording.get('control_mode') != mode or not isinstance(steps, list)
            or len(steps) != source['sample_count'] or len(steps) != len(parent_targets)
            or recording.get('provenance', {}).get('source_group') != source['source_group']
            or recording.get('provenance', {}).get('case_id') != source['case_id']):
        raise ValueError('Recording identity, mode or sample count differs from the parent.')
    previous = model.action_vector(recording['initial_applied_targets'])
    references, issued, ramped, authorities, query_hashes = [], [], [], [], []
    prior_after = recording['initial']
    for i, step in enumerate(steps):
        if type(step.get('index')) is not int or step['index'] != i or step['before'] != prior_after:
            raise ValueError('Reference row indices and preceding snapshot chain must be preserved.')
        start, end = step['before']['timestamp_seconds'], step['after']['timestamp_seconds']
        _time(start, i*.02); _time(end, (i+1)*.02)
        _time(step['command']['issued_at_seconds'], start)
        _time(step['applied']['interval_start_seconds'], start)
        _time(step['applied']['interval_end_seconds'], end)
        if step['applied']['scope'] != 'final_native_tick_only':
            raise ValueError('References require the original final normal native-tick scope.')
        authority = step['command']['authority']
        if authority not in (('teacher',) if mode == 'teacher' else ('teacher', 'student')):
            raise ValueError('Recorded command authority differs from the parent label stage.')
        issued_target = model.action_vector(step['command']['issued'])
        if label == 'issued' and not np.array_equal(issued_target, parent_targets[i]):
            raise ValueError('Teacher-issued labels differ from the pinned parent targets.')
        if label == 'teacher_query':
            query = step.get('teacher_query')
            if not isinstance(query, dict):
                raise ValueError('Mixed supervision requires a separately bound teacher query.')
            query_hashes.append(cache._hash(query.get('sha256'), 'teacher query'))
        references.append(previous.copy())
        issued.append(issued_target)
        previous = model.action_vector(step['applied']['ramped_targets'])
        ramped.append(previous.copy()); authorities.append(authority)
        prior_after = step['after']
    if (cache._digest(authorities) != source['actual_authority_sha256']
            or cache._digest(query_hashes) != source['teacher_query_references_sha256']):
        raise ValueError('Teacher-query or actual-control authority identities changed.')
    references = np.asarray(references, dtype='<f8')
    expected_ramped = feasible_targets(references, np.asarray(issued, dtype='<f8'))
    if not np.array_equal(expected_ramped, np.asarray(ramped, dtype='<f8')):
        raise ValueError('Normal ramped metadata does not follow the exact original 20 updates.')
    return references


def _episode_rows(parent):
    rows, offset = [], 0
    for episode, source in zip(parent['episodes'], parent['provenance']['episodes']):
        count = source['sample_count']
        rows.append(dict(**episode, label_stage=parent['binding']['stage'],
                         offset=offset, sample_count=count, recording_sha256=source['recording_sha256']))
        offset += count
    if offset != parent['sample_count']:
        raise ValueError('Parent episode offsets do not cover every row exactly.')
    return rows


def _identity(pins):
    identity = source_identity()
    if cache._digest(identity) != pins['source_sha256']:
        raise ValueError('Reference helper/source differs from independent caller expectation.')
    return identity


def build_reference_export(study_root, parent_cache, output, *,
        expected_parent_manifest_sha256, expected_parent_source_sha256,
        expected_protocol_sha256, expected_stage, expected_split, expected_source_sha256):
    """Seal exact prior references and endpoints using pinned metadata and labels.

    Parent cache validation checks derived bytes. Here only recording/integrity/
    rollout and study registration/stage metadata are re-read. No raw frames,
    native trace, teacher query body, model weights or simulation are loaded.
    """
    pins = _pins(expected_parent_manifest_sha256, expected_parent_source_sha256,
                 expected_protocol_sha256, expected_stage, expected_split, expected_source_sha256)
    identity = _identity(pins)
    root, parent_path, out = map(cache._path, (study_root, parent_cache, output))
    if (out == root or root in out.parents or out == parent_path or parent_path in out.parents
            or out.exists() or not out.parent.is_dir()):
        raise ValueError('Use a fresh output directory outside the immutable study and parent cache.')
    if (root/'worker.lock').exists():
        raise ValueError('Do not export from an actively written study.')
    dataset, parent = _parent(parent_path, pins)
    episodes = _episode_rows(parent)
    rows, binding = cache._selection(root, pins['protocol_sha256'], pins['split'], pins['stage'],
                                    [row['attempt_id'] for row in episodes])
    if binding != parent['binding'] or [r['path'] for r in rows] != [r['path'] for r in episodes]:
        raise ValueError('Registered training stage differs from the pinned parent cache.')
    cache._reserve(out.parent, parent['sample_count']*112 + 2*1024**2)
    references = []
    for episode, source in zip(episodes, parent['provenance']['episodes']):
        recording = _metadata(root, episode)
        first, count = episode['offset'], episode['sample_count']
        references.append(_references(recording, source, episode, dataset['targets'][first:first+count]))
    references = np.ascontiguousarray(np.concatenate(references), dtype='<f8')
    endpoints = feasible_targets(references, dataset['targets'])
    for episode in episodes:
        _metadata(root, episode)  # source bytes must still match before sealing
    after_rows, after_binding = cache._selection(root, pins['protocol_sha256'], pins['split'], pins['stage'],
                                                [row['attempt_id'] for row in episodes])
    if after_rows != rows or after_binding != binding or _identity(pins) != identity:
        raise ValueError('Source/stage changed while deriving reference labels.')
    # Revalidate the independently pinned parent snapshot as well; no raw export.
    _, after_parent = _parent(parent_path, pins)
    if after_parent != parent:
        raise ValueError('Parent cache changed during reference export.')
    bodies, arrays = {}, {}
    for name, value in (('previous-references.npy', references), ('feasible-targets.npy', endpoints)):
        stream = io.BytesIO(); np.save(stream, value, allow_pickle=False); body = stream.getvalue()
        bodies[name] = body
        arrays[name] = dict(bytes=len(body), sha256=sha256(body).hexdigest(), shape=list(value.shape), dtype='<f8')
    manifest = dict(schema=SCHEMA, scope=SCOPE, raw_source_reverified=False, binding=pins,
        source_identity=identity, contract=CONTRACT, sample_count=len(references), episodes=episodes,
        parent_dataset_sha256=parent['dataset_sha256'], parent_provenance_sha256=cache._digest(dataset['provenance']),
        parent_target_sha256=parent['arrays']['targets.npy']['sha256'], arrays=arrays)
    body = model.canonical(manifest) + b'\n'
    cache._reserve(out.parent, sum(map(len, bodies.values()))+len(body))
    out.mkdir(mode=0o700)
    for name, content in bodies.items():
        cache._write(out/name, content)
    cache._write(out/'manifest.json', body)
    out.chmod(0o555)
    return dict(**pins, manifest_sha256=sha256(body).hexdigest(), sample_count=len(references),
                parent_dataset_sha256=parent['dataset_sha256'], raw_source_reverified=False)


def load_reference_export(directory, parent_cache, *, expected_manifest_sha256,
        expected_parent_manifest_sha256, expected_parent_source_sha256,
        expected_protocol_sha256, expected_stage, expected_split, expected_source_sha256):
    """Validate derived arrays and their pinned parent; no source recording read.

    Returned parent inputs remain unchanged. `feasible_targets` are supervision,
    never decoder input. Runtime/head integration is not supplied by this API.
    """
    pins = _pins(expected_parent_manifest_sha256, expected_parent_source_sha256,
                 expected_protocol_sha256, expected_stage, expected_split, expected_source_sha256)
    identity = _identity(pins)
    root = cache._path(directory)
    if not root.is_dir() or {p.name for p in root.iterdir()} != FILES:
        raise ValueError('Reference export must contain exactly its two arrays and sealed manifest.')
    manifest, actual = cache._json(root/'manifest.json')
    if actual != cache._hash(expected_manifest_sha256, 'reference manifest'):
        raise ValueError('Reference manifest differs from the independently retained caller digest.')
    fields = {'schema', 'scope', 'raw_source_reverified', 'binding', 'source_identity', 'contract',
              'sample_count', 'episodes', 'parent_dataset_sha256', 'parent_provenance_sha256',
              'parent_target_sha256', 'arrays'}
    if (set(manifest) != fields or manifest['schema'] != SCHEMA or manifest['scope'] != SCOPE
            or manifest['raw_source_reverified'] is not False or manifest['binding'] != pins
            or manifest['source_identity'] != identity or manifest['contract'] != CONTRACT):
        raise ValueError('Reference manifest has an unknown source, binding or contract.')
    dataset, parent = _parent(parent_cache, pins)
    n = parent['sample_count']
    if (type(manifest['sample_count']) is not int or manifest['sample_count'] != n
            or manifest['episodes'] != _episode_rows(parent)
            or manifest['parent_dataset_sha256'] != parent['dataset_sha256']
            or manifest['parent_target_sha256'] != parent['arrays']['targets.npy']['sha256']
            or manifest['parent_provenance_sha256'] != cache._digest(dataset['provenance'])
            or set(manifest['arrays']) != FILES-{'manifest.json'}):
        raise ValueError('Reference rows, labels or provenance differ from the pinned parent.')
    arrays = []
    for name in ('previous-references.npy', 'feasible-targets.npy'):
        ref = manifest['arrays'][name]
        if set(ref) != {'bytes', 'sha256', 'shape', 'dtype'} or ref['shape'] != [n, 7] or ref['dtype'] != '<f8':
            raise ValueError('Reference array declaration must retain its exact float64 shape.')
        body = cache._bytes(root/name, n*7*8+4096)
        if ref['bytes'] != len(body) or ref['sha256'] != sha256(body).hexdigest():
            raise ValueError('Reference array bytes were changed, reordered or truncated.')
        stream = io.BytesIO(body)
        value = np.load(stream, allow_pickle=False, max_header_size=4096)
        _matrix(value, name)
        if value.shape != (n, 7) or stream.tell() != len(body):
            raise ValueError('Reference array length or trailing bytes differ.')
        value.flags.writeable = False; arrays.append(value)
    if not np.array_equal(feasible_targets(arrays[0], dataset['targets']), arrays[1]):
        raise ValueError('Feasible labels differ from the original 20-update rule.')
    return dict(schema=SCHEMA, previous_references=arrays[0], feasible_targets=arrays[1],
                parent_dataset=dataset, binding=pins, raw_source_reverified=False)

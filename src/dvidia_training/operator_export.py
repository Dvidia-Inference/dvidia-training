"""Explicitly reviewed external commands as direct camera-action supervision.

Selections name original control-row indices. History is computed causally from
all preceding sensor rows, then selected; gaps never reset or fabricate history.
Failed attempts remain usable with an explicit review reason. No labels are
inferred from native outcomes, teacher suggestions, or zero residuals.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path

import numpy as np

from . import camera_action_model as model
from . import operator_capture as capture

STORAGE_SCHEMA = 'dvidia.operator-dataset/v1'
MAX_ARRAY_BYTES = 16 * 1024 * 1024


def _selection(count, reviewed_indices, reviewed_segments):
    if (reviewed_indices is None) == (reviewed_segments is None):
        raise ValueError('Choose exactly one explicit reviewed row-index or segment selection.')
    if reviewed_indices is not None:
        if type(reviewed_indices) not in (list, tuple):
            raise ValueError('Reviewed indices must be an explicit list of integers.')
        selected = list(reviewed_indices)
    else:
        if type(reviewed_segments) not in (list, tuple) or not reviewed_segments:
            raise ValueError('Reviewed segments must be an explicit list of [start, stop) pairs.')
        selected = []
        for segment in reviewed_segments:
            if (type(segment) not in (list, tuple) or len(segment) != 2
                    or any(type(value) is not int for value in segment)
                    or not 0 <= segment[0] < segment[1] <= count):
                raise ValueError('Reviewed segments use bounded, half-open [start, stop) row indices.')
            selected.extend(range(*segment))
    if not selected or any(type(index) is not int or not 0 <= index < count for index in selected):
        raise ValueError('Every reviewed index must identify a committed control row.')
    if len(set(selected)) != len(selected):
        raise ValueError('Duplicate reviewed indices or overlapping segments are forbidden.')
    return sorted(selected)


def export_episode(directory, *, reviewed_indices=None, reviewed_segments=None,
                   review_reason=None, blind=False):
    """Export sealed, explicitly selected rows with exact issued seven-target labels.

    ``reviewed_segments`` uses half-open ``[start, stop)`` index pairs. The
    nonempty review reason is retained even for unsuccessful/interrupted runs.
    This records the caller's review assertion, not authenticated human review.
    """
    if type(review_reason) is not str or not review_reason.strip() or len(review_reason) > 2000:
        raise ValueError('An explicit nonempty review reason is required for every export.')
    if type(blind) is not bool:
        raise ValueError('blind must be boolean.')
    root = Path(directory)
    recording = capture.load_capture(root)
    recording_body = capture.tape._ordinary_bytes(root/'recording.json', capture.MAX_DOCUMENT_BYTES)
    if recording_body.strip() != model.canonical(recording):
        raise ValueError('Recording changed after operator verification.')
    if (recording.get('schema') != capture.SCHEMA or recording.get('control_mode') != 'operator'
            or recording.get('provenance', {}).get('kind') not in capture.SOURCE_KINDS):
        raise ValueError('Reviewed operator exports require operator source provenance.')
    selected = _selection(len(recording['steps']), reviewed_indices, reviewed_segments)
    selected_set = set(selected)
    encoder = model.CameraActionFeatures(blind=blind)
    previous_applied = recording['initial_applied_targets']
    features, targets, perception_hashes, authorities = [], [], [], []
    for index, step in enumerate(recording['steps']):
        if index > selected[-1]:
            break
        item = capture._perception(root, step['perception'], step['before'], recording, index)
        feature = encoder(dict(snapshot=step['before'], estimate=item['estimate'],
            previous_applied=previous_applied, goal_position_m=recording['goal_position_m']))
        if index in selected_set:
            if (step['command']['authority'] != recording['provenance']['kind']
                    or step.get('teacher_query') is not None):
                raise ValueError('Operator labels cannot impersonate student or teacher authority.')
            features.append(feature)
            targets.append(model.action_vector(step['command']['issued']))
            perception_hashes.append(step['perception']['sha256'])
            authorities.append(step['command']['authority'])
        previous_applied = step['applied']['ramped_targets']
    report = capture._read(root, 'capture.json')
    group = recording['provenance']['source_group']
    source = dict(source_group=group, case_id=recording['provenance']['case_id'],
        source_kind=recording['provenance']['kind'], source_authenticated=False,
        recording_sha256=sha256(recording_body).hexdigest(),
        integrity_sha256=sha256((root/'integrity.json').read_bytes()).hexdigest(),
        calibration_sha256=recording['binding']['calibration_sha256'],
        tracker_sha256=recording['tracker_sha256'], label_source='operator_issued',
        sample_count=len(selected), reviewed_indices=selected, review_reason=review_reason.strip(),
        review_authenticated=False, attempt_status=report['status'],
        attempt_strict_success=report['strict_success'],
        perception_references_sha256=model.digest(perception_hashes),
        actual_authority_sha256=model.digest(authorities))
    sources = [source]
    provenance = dict(feature_contract_sha256=model.FEATURE_CONTRACT_SHA256, blind=blind,
        episodes=sources, source_identity_sha256=model.digest(sources),
        export_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        review_selection_sha256=model.digest(dict(indices=selected, reason=review_reason.strip())),
        label_definition='absolute targets actually issued by the declared external operator source')
    dataset = dict(schema=model.DATASET_SCHEMA, features=np.asarray(features, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float64), groups=[group]*len(selected), provenance=provenance)
    model._dataset(dataset, blind=blind)
    return dataset


def _validated_dataset(dataset):
    if type(dataset) is not dict:
        raise ValueError('A reviewed operator dataset dictionary is required.')
    provenance = dataset.get('provenance', {})
    if type(provenance.get('blind')) is not bool:
        raise ValueError('Operator dataset requires explicit feedback masking provenance.')
    x, y, groups, provenance, _ = model._dataset(dataset, blind=provenance['blind'])
    sources = provenance.get('episodes')
    if (provenance.get('feature_contract_sha256') != model.FEATURE_CONTRACT_SHA256
            or type(sources) is not list or len(sources) != 1
            or provenance.get('source_identity_sha256') != model.digest(sources)):
        raise ValueError('Operator dataset requires one bound reviewed source episode.')
    source = sources[0]
    indices = source.get('reviewed_indices')
    if (source.get('source_kind') not in capture.SOURCE_KINDS
            or source.get('source_authenticated') is not False
            or source.get('review_authenticated') is not False
            or source.get('label_source') != 'operator_issued'
            or type(source.get('review_reason')) is not str or not source['review_reason'].strip()
            or type(indices) is not list or len(indices) != len(groups)
            or indices != _selection(1000, indices, None)
            or source.get('sample_count') != len(groups)
            or any(group != source.get('source_group') for group in groups)):
        raise ValueError('Operator dataset review, authority, or source-group provenance differs.')
    if provenance.get('review_selection_sha256') != model.digest(
            dict(indices=indices, reason=source['review_reason'])):
        raise ValueError('Operator review selection binding differs.')
    return x, y, groups, provenance


def save_dataset(dataset, output):
    """Save one reviewed export to a fresh directory, with exclusive array files."""
    x, y, groups, provenance = _validated_dataset(dataset)
    root = Path(output).absolute()
    if any(item.is_symlink() and str(item) not in ('/var', '/tmp')
           for item in (root, *root.parents)):
        raise ValueError('Dataset output cannot traverse symlinks.')
    buffers = {}
    for name, values in (('features', np.asarray(x, dtype=np.float32)),
                         ('targets', np.asarray(y, dtype=np.float64))):
        buffer = io.BytesIO(); np.save(buffer, values, allow_pickle=False)
        body = buffer.getvalue()
        if len(body) > MAX_ARRAY_BYTES:
            raise ValueError('Reviewed dataset array exceeds its bounded storage budget.')
        buffers[name] = body
    root.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, body in buffers.items():
        filename = name+'.npy'
        files[name] = capture.base._record(root/filename, body)
    document = dict(storage_schema=STORAGE_SCHEMA, schema=model.DATASET_SCHEMA,
                    groups=groups, provenance=provenance, files=files)
    capture.base._json(root/'dataset.json', document)
    return root/'dataset.json'


def load_dataset(directory):
    """Read bounded non-pickle arrays and verify reviewed source provenance."""
    root = Path(directory)
    if {path.name for path in root.iterdir()} != {'dataset.json','features.npy','targets.npy'}:
        raise ValueError('Reviewed dataset inventory differs.')
    document = capture._read(root, 'dataset.json', maximum=2*1024*1024)
    if (set(document) != {'storage_schema','schema','groups','provenance','files'}
            or document['storage_schema'] != STORAGE_SCHEMA or document['schema'] != model.DATASET_SCHEMA
            or type(document['groups']) is not list or not 1 <= len(document['groups']) <= 1000
            or type(document['files']) is not dict or set(document['files']) != {'features','targets'}):
        raise ValueError('Unknown or unbounded reviewed operator dataset.')
    arrays = {}
    for name, width in (('features', model.FEATURE_COUNT), ('targets', 7)):
        reference = document['files'][name]
        if type(reference) is not dict or set(reference) != {'path','bytes','sha256'} or reference['path'] != name+'.npy':
            raise ValueError('Dataset arrays require their exact ordinary relative paths.')
        body = capture.tape._ordinary_bytes(root/reference['path'], MAX_ARRAY_BYTES)
        if reference['bytes'] != len(body) or reference['sha256'] != sha256(body).hexdigest():
            raise ValueError('Dataset array byte identity differs.')
        stream = io.BytesIO(body)
        version = np.lib.format.read_magic(stream)
        reader = {(1,0):np.lib.format.read_array_header_1_0,
                  (2,0):np.lib.format.read_array_header_2_0}.get(version)
        if reader is None:
            raise ValueError('Unsupported reviewed array format.')
        shape, fortran, dtype = reader(stream)
        if (shape != (len(document['groups']), width) or fortran or dtype.kind != 'f'
                or dtype.itemsize not in (4,8) or dtype.hasobject
                or stream.tell()+int(np.prod(shape))*dtype.itemsize != len(body)):
            raise ValueError('Dataset array shape, numeric dtype, or payload size differs.')
        arrays[name] = np.load(io.BytesIO(body), allow_pickle=False)
    dataset = dict(schema=document['schema'], **arrays,
                   groups=document['groups'], provenance=document['provenance'])
    _validated_dataset(dataset)
    return dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description='Export explicitly reviewed native operator rows.')
    parser.add_argument('--capture', required=True, help='Sealed operator capture directory')
    parser.add_argument('--review', required=True, help='JSON with indices OR segments, plus reason')
    parser.add_argument('--output', required=True, help='Fresh output dataset directory')
    parser.add_argument('--blind', action='store_true', help='Mask live feedback using the existing feature contract')
    args = parser.parse_args(argv)
    review = capture.tape._decode(capture.tape._ordinary_bytes(Path(args.review), 64*1024))
    if type(review) is not dict or set(review) not in ({'indices','reason'}, {'segments','reason'}):
        parser.error('Review JSON requires exactly indices OR segments, plus reason.')
    dataset = export_episode(args.capture, reviewed_indices=review.get('indices'),
        reviewed_segments=review.get('segments'), review_reason=review['reason'], blind=args.blind)
    result = save_dataset(dataset, args.output)
    print(json.dumps(dict(dataset=str(result), reviewed_rows=len(dataset['groups']),
                          source_authenticated=False, training_started=False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

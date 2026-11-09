"""Explicit, content-bound cache for verified camera-action learning exports.

Build calls the unchanged strict exporter. A cache hit validates cached bytes,
not the raw recordings: keep the returned pins outside the cache in a trusted run
record. This optional module does not change v0.1 study orchestration or fitting.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile

import numpy as np

from . import camera_action_export as export
from . import camera_action_model as model
from . import camera_action_study as study

SCHEMA = 'dvidia.camera-action-export-cache/v1'
MAX_SAMPLES = 128_000
RESERVE_BYTES = 30 * 1024**3
FILES = {'manifest.json', 'features.npy', 'targets.npy'}
SCOPE = ('Cached derived learning arrays, not raw recording revalidation, source authentication, '
         'training, native simulation or hardware qualification.')


def _digest(value):
    return sha256(model.canonical(value)).hexdigest()


def _hash(value, name):
    return model._hash(value, name)


def _path(value):
    path = Path(os.path.abspath(value))
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Cache and source paths must not traverse symlinks; use canonical /private/tmp on macOS.')
    return path


def _bytes(path, maximum):
    path = _path(path)
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not 0 < before.st_size <= maximum:
            raise ValueError('Expected one bounded ordinary file without hard links.')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            body = stream.read(maximum + 1)
        after = os.fstat(fd)
        if len(body) != before.st_size or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('File changed while being read.')
        return body
    finally:
        os.close(fd)


def _json(path, maximum=16 * 1024**2):
    body = _bytes(path, maximum)
    return model.core._decode(body), sha256(body).hexdigest()


def source_identity():
    """Exact implementation identity, including all existing registered sources."""
    return dict(runtime=study.runtime_hashes(), feature_contract_sha256=model.FEATURE_CONTRACT_SHA256,
                cache_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())


def source_digest():
    return _digest(source_identity())


def _reserve(parent, extra):
    if shutil.disk_usage(parent).free < RESERVE_BYTES + extra:
        raise ValueError('Cache refused: preserve at least 30 GiB free disk after the bounded write.')


def _write(path, body):
    # A fresh cache directory has no usable identity until manifest.json is
    # atomically linked last. Interrupted directories stay visibly unsealed.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.pending-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(body); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    # Windows refuses unlinking a read-only hard link. Remove the writable
    # temporary name first, then seal the complete, atomically linked target.
    path.chmod(0o444)


def _selection(root, expected_protocol_sha256, split, stage, attempt_ids):
    if split not in ('train', 'development'):
        raise ValueError('Only train or development exports are eligible; confirmation is never learning data.')
    if stage not in ('demonstrations', 'corrections') or (stage == 'corrections' and split != 'train'):
        raise ValueError('Corrections are training-only; choose demonstrations or corrections.')
    protocol, protocol_sha = _json(root / 'protocol.json')
    if protocol_sha != _hash(expected_protocol_sha256, 'expected protocol'):
        raise ValueError('Protocol differs from caller expectation.')
    registration, registration_sha = _json(root / 'registration.json')
    runtime = study.runtime_hashes()
    if (protocol.get('schema') != study.SCHEMA or registration.get('schema') != study.SCHEMA + '/registration'
            or registration.get('protocol_sha256') != protocol_sha
            or registration.get('source_sha256') != protocol.get('source_sha256')
            or runtime != protocol.get('source_sha256')):
        raise ValueError('Registered source or protocol identity is inconsistent with this exporter.')
    # The registered roster owns group assignments; do not trust an episode's
    # claimed split or a caller-supplied path as an admission rule.
    ownership = {}
    cases = {}
    for part in ('train', 'development', 'confirmation'):
        for case in protocol['cases'][part]:
            if case['split'] != part or (case['group_id'] in ownership and ownership[case['group_id']] != part):
                raise ValueError('Starting-position groups cross registered splits.')
            if case['id'] in cases:
                raise ValueError('Duplicate registered case identity.')
            ownership[case['group_id']] = part; cases[case['id']] = case
    receipt, receipt_sha = _json(root / 'stages' / (stage + '.json'))
    seal, seal_sha = _json(root / 'stages' / (stage + '.integrity.json'))
    if (seal != dict(sha256=receipt_sha, protocol_sha256=protocol_sha)
            or receipt.get('schema') != study.SCHEMA + '/stage' or receipt.get('stage') != stage
            or receipt.get('protocol_sha256') != protocol_sha or receipt.get('status') != 'complete'):
        raise ValueError('A complete sealed stage with the expected protocol is required.')
    roster = protocol['rosters'][stage]
    rows = receipt['rows']
    if receipt['scheduled'] != len(roster) or [r['id'] for r in rows] != [r['id'] for r in roster]:
        raise ValueError('Stage rows must exactly match the ordered registered roster.')
    selected = []
    for row, entry in zip(rows, roster):
        if any(row.get(k) != entry.get(k) for k in ('id', 'case', 'controller', 'negative_control')):
            raise ValueError('Stage case, controller or negative-control binding changed.')
        case = row['case']
        if cases.get(case['id']) != case or row['path'] != f'episodes/{stage}/{row["id"]}':
            raise ValueError('Unregistered case or episode path.')
        eligible = (case['split'] == split and row.get('complete') is True
                    and row.get('initial_anchor_available') is True and row.get('initial_anchor_binding_valid') is True
                    and (stage == 'corrections' or row.get('strict_success') is True))
        if eligible:
            selected.append(row)
    if attempt_ids is not None:
        if (type(attempt_ids) not in (list, tuple) or not attempt_ids or len(set(attempt_ids)) != len(attempt_ids)
                or any(type(x) is not str for x in attempt_ids)):
            raise ValueError('Provide unique ordered eligible attempt IDs.')
        requested = set(attempt_ids)
        selected = [row for row in selected if row['id'] in requested]
        if [row['id'] for row in selected] != list(attempt_ids):
            raise ValueError('Requested attempts are ineligible, in the wrong split or out of protocol order.')
    if not selected:
        raise ValueError('No eligible learning episodes in the selected split.')
    counts = [r['recorded_complete_intervals'] for r in selected]
    if any(type(n) is not int or not 1 <= n <= 1000 for n in counts) or sum(counts) > MAX_SAMPLES:
        raise ValueError('Episode or cache sample budget exceeded.')
    binding = dict(protocol_sha256=protocol_sha, registration_sha256=registration_sha,
                   stage=stage, stage_sha256=receipt_sha, stage_seal_sha256=seal_sha, split=split)
    return selected, binding


def _validate_features(x, counts, blind):
    if not np.isfinite(x).all() or np.any(np.abs(x) > 10.000001):
        raise ValueError('Features must be bounded and finite.')
    names = model.FEATURE_NAMES
    for i, name in enumerate(names):
        if name.endswith('_available') or name.endswith(':history_present'):
            if not np.isin(x[:, i], (0., 1.)).all():
                raise ValueError('Availability and history masks must be binary.')
        if '_age_s/' in name and (np.any(x[:, i] < 0.) or np.any(x[:, i] > 1.000001)):
            raise ValueError('Normalized sensor/history age is outside the causal contract.')
    offset = 0
    for count in counts:
        segment = x[offset:offset + count]; offset += count
        if not np.allclose(segment[:, 8], np.arange(count, dtype=np.float64) * .001, rtol=0., atol=6e-8):
            raise ValueError('Episode rows must retain increasing causal 20 ms order.')
        if np.any(segment[:, :8] != segment[0, :8]) or np.any(segment[:, 3] != 0.):
            raise ValueError('Historical visual anchor and fixed destination must be constant per episode.')
    for lag in model.LAGS:
        prefix = f'lag{round(lag * 1000)}:'
        command_start = names.index(prefix + model.COMMAND_NAMES[0])
        present = x[:, names.index(prefix + 'history_present')]  # command history
        if np.any(x[present == 0., command_start:command_start + len(model.COMMAND_NAMES)] != 0.):
            raise ValueError('Absent command-history slots must remain zero.')
        if blind:
            continue
        start = model.LIVE_FEATURE_START + list(model.LAGS).index(lag) * len(model.SENSOR_NAMES)
        live_present = x[:, start + len(model.SENSOR_NAMES) - 2]
        if not np.array_equal(present, live_present):
            raise ValueError('Command and live history availability must match.')
        if np.any(x[live_present == 0., start:start + len(model.SENSOR_NAMES)] != 0.):
            raise ValueError('Absent live-history slots must remain zero.')
        for field, size, _ in model.SENSOR_SPECS:
            mask_index = names.index(prefix + field + '_available')
            if np.any(x[x[:, mask_index] == 0., mask_index-size:mask_index] != 0.):
                raise ValueError('Unavailable sensor values must stay zero.')
        mask_index = names.index(prefix + 'visual_available')
        if np.any(x[x[:, mask_index] == 0., mask_index-4:mask_index] != 0.):
            raise ValueError('Unavailable visual position and uncertainty must stay zero.')
    if blind and np.any(x[:, model.LIVE_FEATURE_START:] != 0.):
        raise ValueError('Blind exports must mask every live feedback feature.')


def build_cache(study_root, output, *, expected_protocol_sha256, split='train',
                stage='demonstrations', attempt_ids=None, blind=False):
    """Strictly export selected sealed episodes, then seal a fresh local cache.

    The returned pins must be retained outside this cache before subsequent hits.
    Raw data is verified now only. Run with exclusive access to immutable inputs;
    this is content integrity, not an authenticated producer attestation.
    """
    if type(blind) is not bool:
        raise ValueError('blind must be boolean.')
    root, out = _path(study_root), _path(output)
    if out == root or root in out.parents:
        raise ValueError('Cache output must be outside the immutable study.')
    if out.exists() or not out.parent.is_dir():
        raise ValueError('Use a fresh cache directory in an existing parent.')
    if (root / 'worker.lock').exists():
        raise ValueError('Do not cache an actively written study.')
    rows, binding = _selection(root, expected_protocol_sha256, split, stage, attempt_ids)
    _reserve(out.parent, sum(r['recorded_complete_intervals'] for r in rows) * (model.FEATURE_COUNT*4 + 7*8) + 2*1024**2)
    identity = source_identity()
    label = 'issued' if stage == 'demonstrations' else 'teacher_query'
    datasets, episode_bindings = [], []
    for row in rows:
        directory = _path(root / row['path'])
        metadata = {name: sha256(_bytes(directory/name, 128*1024**2)).hexdigest()
                    for name in ('recording.json', 'integrity.json', 'rollout.json')}
        if metadata['integrity.json'] != row['integrity_sha256'] or metadata['rollout.json'] != row['receipt_sha256']:
            raise ValueError('Episode integrity or receipt differs from sealed stage.')
        dataset = export.export_episode(directory, blind=blind, label_source=label)
        source = dataset['provenance']['episodes'][0]
        if (source['recording_sha256'] != metadata['recording.json'] or source['integrity_sha256'] != metadata['integrity.json']
                or source['sample_count'] != row['recorded_complete_intervals']
                or source['case_id'] != row['case']['id'] or source['source_group'] != row['case']['group_id']):
            raise ValueError('Strict export provenance differs from the registered episode.')
        if any(sha256(_bytes(directory/name, 128*1024**2)).hexdigest() != value for name, value in metadata.items()):
            raise ValueError('Source episode changed during strict export.')
        datasets.append(dataset)
        episode_bindings.append(dict(attempt_id=row['id'], path=row['path'], case=row['case'],
                                     label_source=label, metadata_sha256=metadata))
    after_rows, after_binding = _selection(root, expected_protocol_sha256, split, stage, attempt_ids)
    if after_binding != binding or after_rows != rows or source_identity() != identity:
        raise ValueError('Stage or implementation changed during strict export.')
    # Single-episode identity is preserved exactly; multi-episode identity uses
    # the original combiner and its explicit label provenance unchanged.
    dataset = datasets[0] if len(datasets) == 1 else export.combine_datasets(datasets)
    x = np.asarray(dataset['features'], dtype='<f4')
    y = np.asarray(dataset['targets'], dtype='<f8')
    _validate_features(x, [e['sample_count'] for e in dataset['provenance']['episodes']], blind)
    _, _, _, _, dataset_sha = model._dataset(dataset, blind=blind)
    bodies = {}
    for name, value in (('features.npy', x), ('targets.npy', y)):
        stream = io.BytesIO(); np.save(stream, value, allow_pickle=False); bodies[name] = stream.getvalue()
    manifest = dict(schema=SCHEMA, scope=SCOPE, raw_source_reverified_on_hit=False,
                    binding=binding, source_identity=identity, source_sha256=_digest(identity),
                    episodes=episode_bindings, blind=blind, sample_count=len(x),
                    dataset_schema=dataset['schema'], dataset_sha256=dataset_sha, provenance=dataset['provenance'],
                    arrays={name: dict(sha256=sha256(body).hexdigest(), bytes=len(body),
                                      shape=list(x.shape if name == 'features.npy' else y.shape),
                                      dtype='<f4' if name == 'features.npy' else '<f8') for name, body in bodies.items()})
    manifest_body = model.canonical(manifest) + b'\n'
    _reserve(out.parent, sum(map(len, bodies.values())) + len(manifest_body))
    out.mkdir(mode=0o700)  # never overwrite another cache, including an interrupted one
    for name, body in bodies.items():
        _write(out/name, body)
    _write(out/'manifest.json', manifest_body)  # atomic last-file seal
    out.chmod(0o555)
    return dict(manifest_sha256=sha256(manifest_body).hexdigest(), protocol_sha256=expected_protocol_sha256,
                split=split, source_sha256=_digest(identity), dataset_sha256=dataset_sha,
                sample_count=len(x), raw_source_reverified_on_hit=False)


def load_cache(directory, *, expected_manifest_sha256, expected_protocol_sha256,
               expected_split, expected_source_sha256):
    """Validate cached derived bytes using independent caller pins; no raw reads."""
    root = _path(directory)
    if not root.is_dir() or {p.name for p in root.iterdir()} != FILES:
        raise ValueError('Cache must contain exactly its sealed manifest and two arrays.')
    manifest, actual = _json(root/'manifest.json')
    if actual != _hash(expected_manifest_sha256, 'expected manifest'):
        raise ValueError('Cache manifest differs from independently retained caller digest.')
    fields = {'schema', 'scope', 'raw_source_reverified_on_hit', 'binding', 'source_identity', 'source_sha256',
              'episodes', 'blind', 'sample_count', 'dataset_schema', 'dataset_sha256', 'provenance', 'arrays'}
    if set(manifest) != fields or manifest['schema'] != SCHEMA or manifest['scope'] != SCOPE or manifest['raw_source_reverified_on_hit'] is not False:
        raise ValueError('Unknown cache manifest contract.')
    if expected_split not in ('train', 'development') or manifest['binding']['split'] != expected_split:
        raise ValueError('Cache split differs from caller expectation.')
    if manifest['binding']['protocol_sha256'] != _hash(expected_protocol_sha256, 'expected protocol'):
        raise ValueError('Cache protocol differs from caller expectation.')
    identity = source_identity()
    if (_digest(identity) != _hash(expected_source_sha256, 'expected source') or manifest['source_identity'] != identity
            or manifest['source_sha256'] != expected_source_sha256):
        raise ValueError('Cache implementation source differs from caller expectation.')
    n, blind = manifest['sample_count'], manifest['blind']
    if type(n) is not int or not 1 <= n <= MAX_SAMPLES or type(blind) is not bool:
        raise ValueError('Invalid cache shape or feedback mode.')
    if manifest['dataset_schema'] != model.DATASET_SCHEMA or set(manifest['arrays']) != {'features.npy', 'targets.npy'}:
        raise ValueError('Unknown dataset or array roster.')
    arrays = []
    for name, dtype, shape in (('features.npy', '<f4', (n, model.FEATURE_COUNT)), ('targets.npy', '<f8', (n, 7))):
        ref = manifest['arrays'][name]
        if set(ref) != {'sha256', 'bytes', 'shape', 'dtype'} or ref['shape'] != list(shape) or ref['dtype'] != dtype:
            raise ValueError('Array shape or dtype declaration changed.')
        body = _bytes(root/name, n * np.dtype(dtype).itemsize * shape[1] + 4096)
        if len(body) != ref['bytes'] or sha256(body).hexdigest() != ref['sha256']:
            raise ValueError('Cached array bytes changed or were truncated/reordered.')
        stream = io.BytesIO(body)
        value = np.load(stream, allow_pickle=False, max_header_size=4096)
        if (not isinstance(value, np.ndarray) or value.shape != shape or value.dtype.str != dtype
                or not value.flags.c_contiguous or stream.tell() != len(body) or not np.isfinite(value).all()):
            raise ValueError('Invalid finite ordinary NPY array, dtype, layout, shape or trailing bytes.')
        value.flags.writeable = False; arrays.append(value)
    provenance = manifest['provenance']; sources = provenance.get('episodes', [])
    if (provenance.get('blind') is not blind or provenance.get('feature_contract_sha256') != model.FEATURE_CONTRACT_SHA256
            or provenance.get('export_source_sha256') != identity['runtime']['camera_action_export.py']
            or provenance.get('source_identity_sha256') != _digest(sources) or not sources
            or len(sources) != len(manifest['episodes'])):
        raise ValueError('Cached exporter provenance is inconsistent.')
    groups, counts, seen = [], [], set()
    label = 'issued' if manifest['binding']['stage'] == 'demonstrations' else 'teacher_query'
    if manifest['binding']['stage'] not in ('demonstrations', 'corrections') or (label == 'teacher_query' and expected_split != 'train'):
        raise ValueError('Correction supervision cannot enter development data.')
    for source, episode in zip(sources, manifest['episodes']):
        case = episode['case']; count = source['sample_count']
        if (type(count) is not int or not 1 <= count <= 1000 or source['recording_sha256'] in seen
                or case['split'] != expected_split or case['id'] != source['case_id']
                or case['group_id'] != source['source_group'] or episode['label_source'] != label or source['label_source'] != label
                or episode['metadata_sha256']['recording.json'] != source['recording_sha256']
                or episode['metadata_sha256']['integrity.json'] != source['integrity_sha256']):
            raise ValueError('Cached source identity, label authority, split or count is inconsistent.')
        seen.add(source['recording_sha256']); counts.append(count); groups.extend([source['source_group']] * count)
    if len(groups) != n:
        raise ValueError('Episode counts do not partition arrays.')
    _validate_features(arrays[0], counts, blind)
    dataset = dict(schema=manifest['dataset_schema'], features=arrays[0], targets=arrays[1], groups=groups, provenance=provenance)
    if model._dataset(dataset, blind=blind)[-1] != manifest['dataset_sha256']:
        raise ValueError('Reconstructed learning dataset identity changed.')
    return dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('source', help='Print this implementation identity for an independent run record.')
    build = sub.add_parser('build', help='Strictly verify raw episodes and create a fresh derived cache.')
    build.add_argument('study'); build.add_argument('output')
    build.add_argument('--protocol-sha256', required=True)
    build.add_argument('--split', choices=('train', 'development'), required=True)
    build.add_argument('--stage', choices=('demonstrations', 'corrections'), default='demonstrations')
    build.add_argument('--attempt', action='append'); build.add_argument('--blind', action='store_true')
    for name in ('inspect', 'load'):
        check = sub.add_parser(name, help='Verify the entire cache against independent caller pins; no raw-data reads.')
        check.add_argument('cache'); check.add_argument('--manifest-sha256', required=True)
        check.add_argument('--protocol-sha256', required=True)
        check.add_argument('--split', choices=('train', 'development'), required=True)
        check.add_argument('--source-sha256', required=True)
    args = parser.parse_args(argv)
    if args.command == 'source':
        result = dict(source_identity=source_identity(), source_sha256=source_digest())
    elif args.command == 'build':
        result = build_cache(args.study, args.output, expected_protocol_sha256=args.protocol_sha256,
                             split=args.split, stage=args.stage, attempt_ids=args.attempt, blind=args.blind)
    else:
        dataset = load_cache(args.cache, expected_manifest_sha256=args.manifest_sha256,
                             expected_protocol_sha256=args.protocol_sha256, expected_split=args.split,
                             expected_source_sha256=args.source_sha256)
        result = dict(status='verified', dataset_sha256=model._dataset(dataset, blind=dataset['provenance']['blind'])[-1],
                      samples=len(dataset['groups']), scope=SCOPE, raw_source_reverified_on_hit=False)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

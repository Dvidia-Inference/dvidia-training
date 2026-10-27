"""Export recorded PiPER feedback-study evidence and native 3D replay poses.

No policy inference or physics integration occurs here. Native mj_forward
reconstruction must reproduce saved object/TCP positions to 1e-9 metres.
"""
from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import io
import json
from pathlib import Path
import re

from . import reference_sim as sim
from .piper_imitation_viewer import _inside, _json, _poses
from .reference_replay import _decode, _read, _scene, MIT_LICENSE


SCHEMA = 'dvidia.piper-feedback-viewer/v0.2'
STUDY_SCHEMA = 'dvidia.piper-feedback-study/v0.2'
VARIANTS = ('feedback', 'blind', 'feedforward', 'teacher', 'open_jaw')
CONDITIONS = ('nominal', 'jaw_interruption')
NATIVE_DATASET_ROOT = 'https://huggingface.co/datasets/Dvidia/piper-block-placement-feedback-demonstrations-v02/resolve/'
MAX_JSON = 64 * 1024 * 1024
MAX_EPISODE_PACKED = 32 * 1024 * 1024
MAX_EPISODE = 128 * 1024 * 1024
MAX_NATIVE_PACKED = 64 * 1024 * 1024
MAX_BINARY = 64 * 1024 * 1024
MAX_PUBLISHED_FILE = 8 * 1024 * 1024
SCOPE = ('Recorded synthetic PiPER learned-grip simulation. Only the jaw adapter is newly learned; '
         'all six arm commands remain the frozen v0.1 learned primitive. Native model geometry and saved states '
         'reconstructed with mj_forward only; no browser physics, live policy, human-video learning '
         'or physical robot qualification. The blind control is a separately trained learner '
         'without live feedback, not a shuffled-label model.')


def _digest(value):
    return type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _study_documents(root):
    files = {name: _read(_inside(root, name), MAX_JSON) for name in ('report.json', 'protocol.json', 'frozen.json')}
    report, protocol, frozen = (_decode(files[name]) for name in ('report.json', 'protocol.json', 'frozen.json'))
    if (type(report) is not dict or report.get('schema') != STUDY_SCHEMA
            or report.get('physical_robot_ready') is not False or type(report.get('rows')) is not list
            or type(protocol) is not dict or protocol.get('schema') != STUDY_SCHEMA
            or protocol.get('physical_robot_ready') is not False or protocol.get('variants') != list(VARIANTS)
            or type(frozen) is not dict or frozen.get('schema') != STUDY_SCHEMA):
        raise ValueError('Viewer requires a completed simulation-only feedback v0.2 study.')
    protocol_sha, frozen_sha = (sha256(files[name]).hexdigest() for name in ('protocol.json', 'frozen.json'))
    if (report.get('protocol_sha256') != protocol_sha or report.get('frozen_sha256') != frozen_sha
            or frozen.get('protocol_sha256') != protocol_sha):
        raise ValueError('Report, protocol and frozen receipt hashes do not agree.')
    cases = protocol.get('cases', {}).get('confirmation') if type(protocol.get('cases')) is dict else None
    if type(cases) is not list or len(cases) != 24:
        raise ValueError('Feedback v0.2 requires its complete 24-case confirmation schedule.')
    seen = set()
    for case in cases:
        if (type(case) is not dict or type(case.get('id')) is not str
                or re.fullmatch(r'confirmation-[0-9]{2}-goal[01]-(nominal|jaw_interruption)', case['id']) is None
                or case.get('condition') not in CONDITIONS or not case['id'].endswith('-'+case['condition'])
                or case['id'] in seen):
            raise ValueError('Malformed or duplicate confirmation case.')
        seen.add(case['id'])
    if (sum(case['condition'] == 'nominal' for case in cases) != 12
            or sum(case['condition'] == 'jaw_interruption' for case in cases) != 12):
        raise ValueError('Confirmation must retain twelve cases per condition.')
    expected = [(case['id'], case['condition'], variant) for case in cases for variant in VARIANTS]
    actual = []
    for row in report['rows']:
        if (type(row) is not dict or type(row.get('score')) is not dict
                or type(row['score'].get('success')) is not bool or type(row['score'].get('valid')) is not bool):
            raise ValueError('Malformed scored feedback attempt.')
        actual.append((row.get('case_id'), row.get('condition'), row.get('variant')))
    if actual != expected:
        raise ValueError('Report rows must exactly preserve every scheduled case and variant in order.')
    for variant in VARIANTS:
        rows = [row for row in report['rows'] if row['variant'] == variant]
        expected_counts = dict(scheduled=len(rows), successes=sum(row['score']['success'] for row in rows),
                               valid=sum(row['score']['valid'] for row in rows))
        if type(report.get('variants')) is not dict or report['variants'].get(variant) != expected_counts:
            raise ValueError('Report aggregate differs from its scheduled attempt rows.')
    for name_key, sha_key in (('selected', 'selected_sha256'), ('blind', 'blind_sha256')):
        name = frozen.get(name_key)
        if type(name) is not str or re.fullmatch(r'[A-Za-z0-9]{1,32}', name) is None or not _digest(frozen.get(sha_key)):
            raise ValueError('Frozen candidate identity or digest is invalid.')
    runtime = frozen.get('runtime_sha256')
    if (type(runtime) is not dict or not runtime or any(type(name) is not str or not _digest(value) for name, value in runtime.items())
            or frozen['selected'] == frozen['blind'] or not _digest(frozen.get('base_model_sha256'))):
        raise ValueError('Frozen runtime or distinct candidate binding is invalid.')
    return files, report, protocol, frozen


def _unpack_episode(raw):
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            unpacked = stream.read(MAX_EPISODE+1)
    except (OSError, EOFError) as exc:
        raise ValueError('Study episode is not valid gzip.') from exc
    if len(unpacked) > MAX_EPISODE:
        raise ValueError('Study episode exceeds decompressed bounds.')
    return _decode(unpacked)


def _attempt(root, row, case, frozen, protocol_sha, native_revision='main'):
    """Verify original evidence bindings before referencing or copying any file."""
    prefix = f"confirmation/{case['id']}/{row['variant']}"
    receipt_raw = _read(_inside(root, prefix+'/receipt.json'), MAX_JSON)
    receipt = _decode(receipt_raw)
    binding = receipt.get('binding') if type(receipt) is dict else None
    names = {'started.json', 'episode.json.gz', 'native.npz', 'native.json', 'score.json'}
    if (type(binding) is not dict or binding.get('case') != case
            or binding.get('variant') != row['variant'] or binding.get('collect_labels') is not False
            or binding.get('expert_probability') != 1. or binding.get('protocol_sha256') != protocol_sha
            or binding.get('runtime_sha256') != frozen.get('runtime_sha256')
            or receipt.get('case_id') != case['id'] or receipt.get('variant') != row['variant']
            or type(receipt.get('files')) is not dict or set(receipt['files']) != names):
        raise ValueError('Attempt receipt differs from the frozen scheduled request.')
    expected_model = frozen['blind_sha256'] if row['variant'] == 'blind' else frozen['selected_sha256'] if row['variant'] in ('feedback', 'open_jaw') else None
    if binding.get('model_raw_sha256') != expected_model or binding.get('base_model_raw_sha256') != frozen.get('base_model_sha256'):
        raise ValueError('Attempt model binding differs from the frozen variant.')
    copies, parsed = {}, {}
    for name in sorted(names):
        relative = prefix+'/'+name
        limit = MAX_NATIVE_PACKED if name == 'native.npz' else MAX_EPISODE_PACKED if name.endswith('.gz') else MAX_JSON
        raw = _read(_inside(root, relative), limit)
        digest = sha256(raw).hexdigest()
        if not _digest(receipt['files'].get(name)) or digest != receipt['files'][name]:
            raise ValueError('An original attempt file differs from its receipt SHA256.')
        copies[relative] = {'path': _inside(root, relative), 'bytes': len(raw), 'sha256': digest, 'limit': limit}
        if name == 'episode.json.gz':
            parsed['episode'] = _unpack_episode(raw)
        elif name != 'native.npz':
            parsed[name] = _decode(raw)
    episode = parsed['episode']
    if (type(episode) is not dict or type(parsed['native.json']) is not dict
            or parsed['started.json'] != binding or parsed['score.json'] != row['score']
            or parsed['native.json'].get('pulses') != case.get('pulses')
            or episode.get('case_id') != case['id'] or episode.get('condition') != row['condition']
            or episode.get('policy_kind') != row['variant'] or episode.get('seed') != case.get('seed')
            or episode.get('pulses') != case.get('pulses')
            or episode.get('implementation_sha256') != frozen.get('runtime_sha256')
            or episode.get('model_sha256') != binding.get('model_sha256')
            or episode.get('base_model_sha256') != binding.get('base_model_sha256')):
        raise ValueError('Attempt evidence identity or score differs from its exact receipt binding.')
    copies[prefix+'/receipt.json'] = {'path': _inside(root, prefix+'/receipt.json'), 'bytes': len(receipt_raw),
                                     'sha256': sha256(receipt_raw).hexdigest(), 'limit': MAX_JSON}
    native = copies.pop(prefix+'/native.npz')
    evidence = {name: prefix+'/'+name for name in ('episode.json.gz', 'native.json', 'score.json', 'receipt.json')}
    evidence['native.npz'] = NATIVE_DATASET_ROOT+native_revision+'/'+prefix+'/native.npz'
    evidence['native_npz_sha256'] = native['sha256']
    evidence['native_npz_bytes'] = native['bytes']
    return episode, copies, evidence


def selected_replays(report):
    """Outcome-independent: all feedback rows plus first scheduled other variants."""
    selected, controls = [], set()
    for row in report['rows']:
        if row['variant'] == 'feedback' or row['variant'] not in controls:
            selected.append(row)
            controls.add(row['variant'])
    return selected


def export_viewer(run, assets, output, *, native_revision='main'):
    from .piper_task import PiperTaskConfig, _asset_root, build_task_model
    from . import robot_profiles
    root, destination = Path(run), Path(output)
    if root.is_symlink() or not root.is_dir() or destination.exists() or destination.is_symlink():
        raise ValueError('Use an existing non-symlink run and a fresh viewer output directory.')
    if type(native_revision) is not str or re.fullmatch(r'main|[0-9a-f]{40}', native_revision) is None:
        raise ValueError('Native dataset revision must be main or an exact 40-hex commit.')
    files, report, protocol, frozen = _study_documents(root)
    copies = {}
    parent_sha = frozen.get('training_parent_sha256')
    if parent_sha is not None:
        parent_raw = _read(_inside(root, 'training-parent.json'), MAX_JSON)
        if not _digest(parent_sha) or sha256(parent_raw).hexdigest() != parent_sha:
            raise ValueError('Frozen training parent manifest bytes changed.')
        parent = _decode(parent_raw)
        if type(parent) is not dict or parent.get('schema') != 'dvidia.piper-feedback-training-parent/v1':
            raise ValueError('Invalid imported training parent manifest.')
        files['training-parent.json'] = parent_raw
    for name, expected in [('base-model.json', frozen.get('base_model_sha256'))] + [
        (f"candidates/{frozen[key]}/{file}", frozen.get(f'{prefix}_{suffix}_sha256'))
        for key, prefix in (('selected', 'selected'), ('blind', 'blind'))
        for file, suffix in (('model.json', None), ('receipt.json', 'receipt'), ('development.json', 'development'))]:
        if name.endswith('/model.json'):
            expected = frozen['blind_sha256'] if name == f"candidates/{frozen['blind']}/model.json" else frozen['selected_sha256']
        raw = _read(_inside(root, name), MAX_JSON)
        if not _digest(expected) or sha256(raw).hexdigest() != expected:
            raise ValueError('Frozen model or candidate receipt bytes changed.')
        files[name] = raw
    asset_info = robot_profiles.verify_assets('piper', _asset_root(assets))
    _, asset_blobs = robot_profiles._payload('piper', _asset_root(assets))
    source = {'report_sha256': sha256(files['report.json']).hexdigest(),
              'frozen_sha256': sha256(files['frozen.json']).hexdigest(), 'asset_provenance': asset_info,
              'exporter_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
              'pose_exporter_sha256': sha256(Path(__file__).with_name('piper_imitation_viewer.py').read_bytes()).hexdigest(),
              'geometry_exporter_sha256': sha256(Path(__file__).with_name('reference_replay.py').read_bytes()).hexdigest(),
              'model_sha256': frozen['selected_sha256'], 'blind_model_sha256': frozen['blind_sha256'],
              'training_parent_sha256': parent_sha,
              'physical_robot_ready': False}
    selected_ids = {(row['case_id'], row['variant']) for row in selected_replays(report)}
    case_by_id = {case['id']: case for case in protocol['cases']['confirmation']}
    selected, all_rows = [], []
    for row in report['rows']:
        episode, evidence_files, evidence = _attempt(root, row, case_by_id[row['case_id']], frozen,
                                                     report['protocol_sha256'], native_revision)
        copies.update(evidence_files)
        item = dict(id=row['case_id'], condition=row['condition'], variant=row['variant'], score=row['score'],
                    episode=evidence['episode.json.gz'], evidence=evidence,
                    sha256=evidence_files[evidence['episode.json.gz']]['sha256'])
        all_rows.append(item)
        if (row['case_id'], row['variant']) in selected_ids:
            selected.append((item, episode))
    scene, geometry, cases = None, None, []
    residuals = {'tcp_position_m': 0., 'object_position_m': 0.}
    warnings = []
    def binary_meta(name, raw, packed):
        return {'href': name, 'bytes': len(raw), 'sha256': sha256(raw).hexdigest(),
                'compressed_href': name+'.gz', 'compressed_bytes': len(packed), 'compression': 'gzip'}
    with sim.native_warnings(warnings):
        for index, (row, episode) in enumerate(selected):
            if type(episode.get('config')) is not dict:
                raise ValueError('Replay is missing its native task configuration.')
            model = build_task_model(assets, PiperTaskConfig(**episode['config']))
            compiled = sim.model_digest(model)
            if episode.get('compiled_model_sha256') != compiled:
                raise ValueError('Rebuilt native model differs from the recorded model hash.')
            candidate, candidate_geometry = _scene(model, source)
            candidate.update(schema='dvidia.piper-feedback-scene/v0.2', scope=SCOPE)
            if scene is None:
                scene, geometry = candidate, candidate_geometry
            elif candidate != scene or candidate_geometry != geometry:
                raise ValueError('Native body layout or local geometry differs across replay cases.')
            poses, rows = _poses(model, episode, residuals)
            if len(poses) > MAX_PUBLISHED_FILE:
                raise ValueError('One recorded replay exceeds the 8 MiB published chunk bound.')
            for destination_row, state in zip(rows, episode['trajectory']):
                destination_row.update(grasp_contact_counts=state.get('grasp_contact_counts'),
                                       pad_normal_forces_n=state.get('pad_normal_forces_n'),
                                       table_object_contact_count=state.get('table_object_contact_count'),
                                       native_actuator_controls=state.get('native_actuator_controls'))
            pose_name, rows_name = f'poses-{index:02}.bin', f'frames-{index:02}.json'
            pose_packed, rows_raw = gzip.compress(poses, mtime=0), _json(rows)
            if len(rows_raw) > MAX_PUBLISHED_FILE:
                raise ValueError('One recorded frame metadata file exceeds 8 MiB.')
            files.update({pose_name: poses, pose_name+'.gz': pose_packed, rows_name: rows_raw})
            cases.append({**row, 'frame_count': len(rows), 'compiled_model_sha256': compiled,
                          'pose_binary': binary_meta(pose_name, poses, pose_packed),
                          'rows_file': {'href': rows_name, 'bytes': len(rows_raw), 'sha256': sha256(rows_raw).hexdigest()}})
    if warnings:
        raise ValueError('Native forward reconstruction emitted warnings.')
    geometry_packed = gzip.compress(geometry, mtime=0)
    scene['geometry_binary'] = binary_meta('geometry.bin', geometry, geometry_packed)
    metadata = dict(schema=SCHEMA, scope=SCOPE, source=source, cases=cases, all_attempts=all_rows,
                    body_count=len(scene['bodies']), frame_stride_floats=len(scene['bodies'])*7,
                    replay_loading='Each replay has separate hash-bound frame metadata and pose chunks. The browser loads only the selected replay; chunks are at most 8 MiB.',
                    training_parent_manifest='training-parent.json' if parent_sha is not None else None,
                    selection_notice='All 24 feedback confirmation attempts are replayable, plus the first scheduled case of each blind, feedforward, teacher and open-jaw control. Selection never depends on outcome. All 120 attempts retain local 20 ms traces, scores and receipts. Original 1 ms numeric arrays are linked to the public dataset with their source hashes; they are not duplicated in this viewer projection.',
                    native_dataset=dict(repo_id='Dvidia/piper-block-placement-feedback-demonstrations-v02',
                                        revision=native_revision, publication_readback_verified=False,
                                        note='Dataset link targets are configured here; this exporter performs no network access. Consult the final publication receipt for verified upload commits and readback.'),
                    limitations=report.get('limitations', []), reconstruction_max_absolute_residual_m=residuals,
                    binary_pose_precision='little-endian Float32 body positions in metres and quaternions wxyz')
    files.update({'scene.json': _json(scene), 'motion.json': _json(metadata),
                  'geometry.bin': geometry, 'geometry.bin.gz': geometry_packed,
                  'LICENSE': MIT_LICENSE.encode('utf8')})
    source_root = Path(__file__).parent
    for name in ('piper_feedback_viewer.html', 'piper_feedback_viewer.bundle.js', 'piper_feedback_viewer.js', 'reference_replay.third-party.txt'):
        files['index.html' if name.endswith('.html') else name] = _read(source_root/name, 8*1024*1024)
    for name in ('LICENSE', 'README.md'):
        key = 'agilex_piper/'+name
        if key in asset_blobs:
            files['licenses/'+key] = asset_blobs[key]
    files['README.md'] = ('# PiPER learned feedback replay v0.2\n\n'+SCOPE+'\n\n'
        'Serve the complete export over HTTP and open index.html. All rendering code and geometry are local. '
        'Playback does not rerun physics or inference. Frame metadata and pose binaries are per-case, hash-bound '
        'chunks loaded only for the selected replay. Every published file is at most 8 MiB. '
        'This is a viewer projection, not the full study bundle. '
        'All 120 original 20 ms episodes, scores, native metadata and receipts are local; dense 1 ms native NPZ '
        'arrays remain in the linked Hugging Face dataset and complete GitHub evidence archive. Original NPZ hashes '
        'are retained in motion.json and attempt receipts. Dataset links are configured without network verification; '
        'consult the final publication receipt for the verified commit and anonymous readback. '
        'The blind control is trained without live feedback; it is not a shuffled-label learner. '
        'Checksums establish artifact consistency, not authenticity. Three.js and PiPER notices are included.\n').encode('utf8')
    integrity = {'schema': SCHEMA, 'scope': 'Viewer projection consistency only; external native NPZ arrays are not part of this local file inventory.', 'source': source,
                 'files': {name: {'bytes': len(raw), 'sha256': sha256(raw).hexdigest()} for name, raw in files.items()}}
    integrity['files'].update({name: {key: value[key] for key in ('bytes', 'sha256')} for name, value in copies.items()})
    files['viewer-integrity.json'] = _json(integrity)
    if any(len(raw) > MAX_PUBLISHED_FILE for raw in files.values()) or any(row['bytes'] > MAX_PUBLISHED_FILE for row in copies.values()):
        raise ValueError('A viewer projection file exceeds the 8 MiB public hosting bound.')
    destination.mkdir(parents=True, exist_ok=False)
    for name, raw in files.items():
        path = _inside(destination, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
    for name, source_file in copies.items():
        raw = _read(source_file['path'], source_file['limit'])
        if len(raw) != source_file['bytes'] or sha256(raw).hexdigest() != source_file['sha256']:
            raise ValueError('Original confirmation evidence changed during export.')
        path = _inside(destination, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
    return dict(schema=SCHEMA, output=str(destination), replay_cases=len(cases), report_attempts=len(all_rows),
                frames=sum(case['frame_count'] for case in cases), physical_robot_ready=False,
                reconstruction_max_absolute_residual_m=residuals,
                bytes=sum(len(raw) for raw in files.values())+sum(row['bytes'] for row in copies.values()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-revision', default='main', help='Native dataset commit: main or exact 40-hex SHA. Export performs no publication/readback.')
    args = parser.parse_args(argv)
    print(json.dumps(export_viewer(args.run, args.assets, args.output, native_revision=args.native_revision), indent=2))


if __name__ == '__main__':
    main()

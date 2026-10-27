"""Export genuine native PiPER geometry and recorded study states for replay.

No physics ticks or policy inference run here. Native Float64 forward kinematics
must reproduce recorded object/TCP positions before Float32 poses are exported.
"""
from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import io
import json
from pathlib import Path, PurePosixPath
import re

from . import reference_sim as sim
from .reference_replay import _decode, _read, _scene, _vector, MIT_LICENSE


SCHEMA = 'dvidia.piper-imitation-viewer/v1'
SCOPE = ('Recorded synthetic PiPER simulation replay. Native model meshes and saved '
         'states reconstructed with mj_forward only; no live physics, robot control, '
         'video-derived learning, physical qualification or new performance measurement.')
MAX_JSON = 64 * 1024 * 1024
MAX_EPISODE_PACKED = 32 * 1024 * 1024
MAX_EPISODE = 128 * 1024 * 1024
MAX_BINARY = 64 * 1024 * 1024
VARIANTS = ('learned', 'teacher', 'open_jaw', 'hold', 'shuffled_labels')


def _json(value):
    return (json.dumps(value, separators=(',', ':'), ensure_ascii=False, allow_nan=False)+'\n').encode('utf8')


def _inside(root, relative):
    if (type(relative) is not str or re.fullmatch(r'[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*', relative) is None
            or any(part in ('', '.', '..') for part in relative.split('/'))):
        raise ValueError('Study links must be safe relative paths.')
    path = PurePosixPath(relative)
    if path.is_absolute() or len(path.parts) > 8 or len(relative) > 240:
        raise ValueError('Study links must stay inside the run directory.')
    result = root.joinpath(*path.parts)
    if any(part.is_symlink() for part in (result, *result.parents) if part != root and root in part.parents):
        raise ValueError('Study files cannot use symlinks.')
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError('Study file escaped the run directory.')
    return result


def _episode(root, row):
    href = row.get('episode')
    if type(href) is not str or not href.startswith('attempts/') or not href.endswith('.json.gz'):
        raise ValueError('Original attempt traces must use bounded attempts/*.json.gz links.')
    raw = _read(_inside(root, href), MAX_EPISODE_PACKED)
    if sha256(raw).hexdigest() != row.get('sha256'):
        raise ValueError('Study episode SHA256 mismatch.')
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            unpacked = stream.read(MAX_EPISODE+1)
    except (OSError, EOFError) as exc:
        raise ValueError('Study episode is not valid gzip.') from exc
    if len(unpacked) > MAX_EPISODE:
        raise ValueError('Study episode exceeds decompressed bounds.')
    return _decode(unpacked), raw


def _poses(model, episode, residuals):
    mj, np = sim._native_imports()
    states = episode.get('trajectory')
    if type(states) is not list or len(states) > 1501:
        raise ValueError('Replay needs a bounded recorded trajectory.')
    data, poses, rows = mj.MjData(model), [], []
    tcp = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, 'piper_task_tcp')
    block = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'task_block')
    if (model.nq, model.nv, model.nu) != (15, 14, 7) or tcp < 0 or block < 0:
        raise ValueError('Replay requires the native six-axis/single-jaw PiPER task.')
    for index, state in enumerate(states):
        if type(state) is not dict:
            raise ValueError('Recorded state must be a dictionary.')
        data.qpos[:] = _vector(state.get('qpos'), model.nq, 'qpos')
        data.qvel[:] = _vector(state.get('qvel'), model.nv, 'qvel')
        if state.get('native_actuator_controls') is not None:
            data.ctrl[:] = _vector(state['native_actuator_controls'], model.nu, 'native controls')
        mj.mj_forward(model, data)
        for name, native in (('tcp_position_m', data.site_xpos[tcp]), ('object_position_m', data.xpos[block])):
            recorded = _vector(state.get(name), 3, name)
            delta = float(np.max(np.abs(native-recorded)))
            residuals[name] = max(residuals[name], delta)
            if delta > 1e-9:
                raise ValueError(f'Reconstructed {name} differs from the recording by more than 1e-9 m.')
        pose = np.concatenate((data.xpos, data.xquat), axis=1)
        if not np.isfinite(pose).all():
            raise ValueError('Replay reconstructed nonfinite body poses.')
        poses.append(pose)
        stamp = state.get('t_seconds')
        if type(stamp) not in (int, float) or not np.isfinite(stamp) or abs(stamp-index*.02) > 1e-8:
            raise ValueError('Replay frame timestamps must follow recorded 20 ms boundaries.')
        rows.append({'t_seconds': stamp, 'joint_position_rad': _vector(state.get('joint_position_rad'), 6, 'arm positions'),
                     'gripper_joint_position_m': _vector(state.get('gripper_joint_position_m'), 2, 'jaw positions'),
                     'issued_gripper_position_m': state.get('issued_gripper_position_m'),
                     'object_position_m': state['object_position_m'], 'tcp_position_m': state['tcp_position_m']})
    return np.asarray(poses, dtype='<f4').tobytes(order='C'), rows


def export_viewer(run, assets, output):
    """Export a completed study to a fresh self-contained static directory.

    Every learned confirmation case is replayable. The first scheduled case of
    each control is included without selecting on outcome. All report rows and
    their original compressed episode traces remain linked in the export.
    """
    from .piper_task import PiperTaskConfig, _asset_root, build_task_model
    from . import robot_profiles
    root, destination = Path(run), Path(output)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Study run must be an existing non-symlink directory.')
    if destination.exists() or destination.is_symlink():
        raise ValueError('Viewer output must be a fresh directory.')
    if not (root/'report.json').is_file():
        raise ValueError('Study report.json is not ready; export after confirmation completes.')
    files = {name: _read(_inside(root, name), MAX_JSON) for name in ('report.json', 'protocol.json', 'frozen.json')}
    report, protocol, frozen = (_decode(files[name]) for name in ('report.json', 'protocol.json', 'frozen.json'))
    if (type(report) is not dict or report.get('schema') != 'dvidia.piper-imitation-study/v1'
            or report.get('physical_robot_ready') is not False or type(report.get('attempts')) is not list
            or not 1 <= len(report['attempts']) <= 300):
        raise ValueError('Viewer requires a bounded simulation-only PiPER imitation report.')
    if (report.get('protocol_sha256') != sha256(files['protocol.json']).hexdigest()
            or report.get('frozen_sha256') != sha256(files['frozen.json']).hexdigest()
            or frozen.get('protocol_sha256') != sha256(files['protocol.json']).hexdigest()):
        raise ValueError('Report, protocol and frozen receipt hashes do not agree.')
    asset_info = robot_profiles.verify_assets('piper', _asset_root(assets))
    _, asset_blobs = robot_profiles._payload('piper', _asset_root(assets))
    source = {'report_sha256': sha256(files['report.json']).hexdigest(), 'asset_provenance': asset_info,
              'exporter_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
              'geometry_exporter_sha256': sha256(Path(__file__).with_name('reference_replay.py').read_bytes()).hexdigest(),
              'model_sha256': frozen.get('model_sha256'), 'physical_robot_ready': False}
    selected, all_rows, seen, first_controls = [], [], set(), set()
    for row in report['attempts']:
        if type(row) is not dict or row.get('variant') not in VARIANTS or type(row.get('id')) is not str:
            raise ValueError('Malformed study attempt identity.')
        identity = (row['id'], row['variant'])
        if identity in seen or type(row.get('score')) is not dict or type(row['score'].get('success')) is not bool:
            raise ValueError('Duplicate or malformed scored study attempt.')
        seen.add(identity)
        episode, raw = _episode(root, row)
        files[row['episode']] = raw
        all_rows.append({'id': row['id'], 'variant': row['variant'], 'score': row['score'],
                         'episode': row['episode'], 'sha256': row['sha256']})
        if row['variant'] == 'learned' or row['variant'] not in first_controls:
            selected.append((row, episode))
            first_controls.add(row['variant'])
    if not any(row['variant'] == 'learned' for row, _ in selected):
        raise ValueError('Study report has no learned-policy attempts.')
    scene, geometry, motion, cases = None, None, bytearray(), []
    residuals = {'tcp_position_m': 0., 'object_position_m': 0.}
    warnings = []
    with sim.native_warnings(warnings):
        for row, episode in selected:
            if type(episode) is not dict or type(episode.get('config')) is not dict:
                raise ValueError('Selected replay is missing its native task configuration.')
            c = PiperTaskConfig(**episode['config'])
            model = build_task_model(assets, c)
            compiled = sim.model_digest(model)
            if episode.get('compiled_model_sha256') not in (None, compiled):
                raise ValueError('Rebuilt native model differs from recorded model hash.')
            candidate, candidate_geometry = _scene(model, source)
            candidate.update(schema='dvidia.piper-imitation-scene/v1', scope=SCOPE)
            if scene is None:
                scene, geometry = candidate, candidate_geometry
            elif candidate != scene or candidate_geometry != geometry:
                raise ValueError('Native body layout or local geometry differs across replay cases.')
            offset = len(motion)
            poses, rows = _poses(model, episode, residuals)
            motion.extend(poses)
            if len(motion) > MAX_BINARY:
                raise ValueError('Replay motion binary exceeds 64 MiB.')
            cases.append({'id': row['id'], 'variant': row['variant'], 'score': row['score'],
                          'episode': row['episode'], 'frame_count': len(rows), 'pose_byte_offset': offset,
                          'compiled_model_sha256': compiled, 'rows': rows})
    if warnings:
        raise ValueError('Native forward reconstruction emitted warnings: '+str(warnings))
    geometry_packed, motion_packed = gzip.compress(geometry, mtime=0), gzip.compress(bytes(motion), mtime=0)
    def binary_meta(name, raw, packed):
        return {'href': name, 'bytes': len(raw), 'sha256': sha256(raw).hexdigest(),
                'compressed_href': name+'.gz', 'compressed_bytes': len(packed), 'compression': 'gzip'}
    scene['geometry_binary'] = binary_meta('geometry.bin', geometry, geometry_packed)
    motion_meta = {'schema': SCHEMA, 'scope': SCOPE, 'source': source, 'cases': cases, 'all_attempts': all_rows,
                   'body_count': len(scene['bodies']), 'frame_stride_floats': len(scene['bodies'])*7,
                   'motion_binary': binary_meta('motion.bin', motion, motion_packed),
                   'selection_notice': ('Replays include every learned confirmation attempt and the first scheduled '
                                        'case for each control. Selection never depends on success. Every attempt and trace is linked below.'),
                   'limitations': report.get('limitations', []), 'reconstruction_max_absolute_residual_m': residuals,
                   'binary_pose_precision': 'little-endian Float32 body positions in metres and quaternions wxyz'}
    files.update({'scene.json': _json(scene), 'motion.json': _json(motion_meta),
                  'geometry.bin': geometry, 'motion.bin': bytes(motion),
                  'geometry.bin.gz': geometry_packed, 'motion.bin.gz': motion_packed,
                  'LICENSE': MIT_LICENSE.encode('utf8')})
    source_root = Path(__file__).parent
    for name in ('piper_imitation_viewer.html', 'piper_imitation_viewer.bundle.js', 'reference_replay.third-party.txt'):
        files['index.html' if name.endswith('.html') else name] = _read(source_root/name, 8*1024*1024)
    for name in ('LICENSE', 'README.md'):
        key = 'agilex_piper/'+name
        if key in asset_blobs:
            files['licenses/'+key] = asset_blobs[key]
    files['README.md'] = ('# PiPER imitation study replay\n\n'+SCOPE+'\n\n'
        'Serve this directory over local HTTP and open index.html. All rendering code, geometry and states are local. '
        'Original compressed traces are linked for every report row. Integrity checks establish consistency, not authenticity. '
        'Native Float64 reconstruction residuals are recorded in motion.json; rendered body poses use Float32. '
        'PiPER model notices are under licenses/agilex_piper. Three.js/OrbitControls notices are in '
        'reference_replay.third-party.txt.\n').encode('utf8')
    integrity = {'schema': SCHEMA, 'scope': 'Artifact consistency only',
                 'files': {name: {'bytes': len(raw), 'sha256': sha256(raw).hexdigest()} for name, raw in files.items()}}
    files['viewer-integrity.json'] = _json(integrity)
    destination.mkdir(parents=True, exist_ok=False)
    for name, raw in files.items():
        path = _inside(destination, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
    return {'schema': SCHEMA, 'output': str(destination), 'replay_cases': len(cases), 'report_attempts': len(all_rows),
            'frames': sum(case['frame_count'] for case in cases), 'physical_robot_ready': False,
            'reconstruction_max_absolute_residual_m': residuals, 'bytes': sum(map(len, files.values()))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(export_viewer(args.run, args.assets, args.output), indent=2))


if __name__ == '__main__':
    main()

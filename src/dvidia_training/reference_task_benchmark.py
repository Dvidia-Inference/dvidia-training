"""Frozen, local task attempts for the native arm; no learned/physical qualification."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
from html import escape
import json
import os
from pathlib import Path
import platform
import random
import re
import sys
import time

from .reference_replay import _decode, _read, _json_bytes
from . import reference_sim as sim

SCHEMA = 'dvidia.reference-task-benchmark/v1'
CONTROLS = ('placement', 'replay_open_jaw', 'hold')
SCOPE = ('Authored, privileged-state rigid-block task in native reference physics. '
         'No learned policy, shoe physics, human-video action recovery or physical robot qualification.')
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_BUNDLE_BYTES = 256 * 1024 * 1024
SOURCE_FILES = ('reference_task.py', 'reference_task_benchmark.py', 'reference_sim.py', 'reference_assets.lock.json')


def _peak_rss_bytes():
    # Process lifetime high-water mark, including imports and native compilation.
    # Windows has no stdlib resource API; unavailable is not zero memory usage.
    if sys.platform not in ('darwin', 'linux'):
        return None
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == 'darwin' else value*1024)
    except (ImportError, OSError, ValueError):
        return None


def _write(path, value):
    with Path(path).open('xb') as stream:
        stream.write(_json_bytes(value))


def implementation_hashes():
    root = Path(__file__).parent
    return {name: sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def make_protocol(*, seed=411000, nominal=4, stress=2, cohort='development'):
    from .reference_task import ReferenceTaskConfig, PROFILE, task_contract
    for name, value, low, high in (('seed', seed, 0, 999975), ('nominal', nominal, 1, 12), ('stress', stress, 0, 12)):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    if cohort not in ('development', 'confirmation'):
        raise ValueError('Unknown cohort')
    rng = random.Random(seed)
    base = ReferenceTaskConfig()
    scenes = []
    for stratum, count in (('nominal', nominal), ('low_friction', stress)):
        for index in range(count):
            start = list(base.object_position_m)
            target = list(base.target_position_m)
            start[0] += rng.uniform(-.015, .015)
            start[1] += rng.uniform(-.015, .015)
            target[0] += rng.uniform(-.015, .015)
            target[1] += rng.uniform(-.015, .015)
            config = replace(base, object_position_m=tuple(start), target_position_m=tuple(target),
                             friction=.15 if stratum == 'low_friction' else base.friction)
            scenes.append({'id': f'{stratum}-{index:02}', 'stratum': stratum, 'seed': seed+len(scenes),
                           'config': asdict(config)})
    return {'schema': 'dvidia.reference-task-protocol/v1', 'profile': PROFILE, 'contract': task_contract(),
            'cohort': cohort, 'seed': seed, 'scenes': scenes, 'controls': list(CONTROLS),
            'planned_attempts': len(scenes)*len(CONTROLS), 'implementation_sha256': implementation_hashes(),
            'scope': SCOPE, 'physical_robot_ready': False, 'learned_policy': False,
            'split_note': 'Seeded synthetic layouts, not independent physical captures. Confirmation is a user declaration; '
                          'this tool cannot establish whether a seed/layout was previously inspected.',
            'comparison': 'Open-jaw control replays the placement issued arm targets with jaws requested open; '
                          'hold preserves initial arm targets. All attempts use the same scene timeout.',
            'timing_boundary': 'Attempt wall time includes asset checks, compilation, reset, planning, stepping and trace '
                               'capture; separate inner timings identify narrower boundaries. No rendering, sensing, '
                               'training, disk serialization, transport or physical actuation.'}


def _summary(attempt):
    # Traces and the exact command tape remain in the separately sealed attempt.
    return {key: value for key, value in attempt.items() if key not in ('trajectory', 'arm_tape', 'actions')}


def aggregate(attempts, scenes):
    groups = []
    by_scene = {item['id']: item for item in scenes}
    for stratum in dict.fromkeys(x['stratum'] for x in scenes):
        for control in CONTROLS:
            rows = [a for a in attempts if a['control'] == control and by_scene[a['scene_id']]['stratum'] == stratum]
            successes = sum(a.get('valid') is True and a.get('success') is True for a in rows)
            valid = sum(a.get('valid') is True for a in rows)
            # Small synthetic cohorts: counts are descriptive; no population confidence claim.
            groups.append({'stratum': stratum, 'control': control, 'attempted': len(rows),
                           'valid': valid, 'success': successes, 'failed_or_invalid': len(rows)-successes,
                           'success_fraction': successes/len(rows) if rows else None})
    return groups


def _html(report):
    def number(value, scale=1):
        return '—' if not isinstance(value,(int,float)) else f'{value*scale:.2f}'
    rows = []
    for item in report['attempts']:
        status = 'Completed' if item.get('valid') and item.get('success') else 'Failed' if item.get('valid') else 'Invalid'
        metrics = item.get('metrics', {})
        rows.append('<tr><td>'+escape(item['scene_id'])+'</td><td>'+escape(item['control'])+'</td><td>'+status+
                    '</td><td>'+escape(str(item.get('reason') or item.get('error') or '—'))+
                    '</td><td>'+number(metrics.get('target_distance_m'),1000)+
                    '</td><td>'+number(metrics.get('first_success_time_seconds'))+
                    '</td><td>'+number(item.get('attempt_wall_seconds'))+'</td><td><a href="'+
                    escape(item['result_file'], quote=True)+'">Attempt JSON</a></td></tr>')
    sums = ''.join('<li>'+escape(x['stratum'])+' · '+escape(x['control'])+': <strong>'+str(x['success'])+'/'+str(x['attempted'])+
                   '</strong> completed; '+str(x['valid'])+' numerically valid.</li>' for x in report['summary'])
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>DVIDIA · Native grasp task benchmark</title><style>body{font:18px/1.6 Georgia,serif;background:#fafaf7;color:#292d27;'
            'margin:0}main{max-width:1000px;margin:auto;padding:32px 20px}h1,h2{font-weight:400;line-height:1.2}a{color:#396751}'
            'table{border-collapse:collapse;width:100%;font:14px/1.5 system-ui}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}'
            '.scroll{overflow:auto}code{font:13px monospace;overflow-wrap:anywhere}li{margin:6px 0}footer{border-top:1px solid #ddd;padding-top:16px}'
            '</style><main><p>DVIDIA RESEARCH · NATIVE CONTACT TASK</p><h1>Grasp. Carry. Release. Measure.</h1><p>'+escape(SCOPE)+
            '</p><p>A free rigid block must be lifted, placed near a separate target, released and left settled with the tool retreated. '
            'The scorer uses native object state independently of the controller’s phase. The block is never welded to the gripper.</p>'
            '<p><a href="report.json">Full summary</a> · <a href="protocol.json">Frozen protocol</a> · '
            '<a href="integrity.json">Artifact hashes</a></p><h2>Every planned attempt</h2><p>'+str(report['attempted'])+
            ' attempts · '+escape(report['cohort'])+' cohort. Synthetic scenarios; counts do not establish physical reliability.</p><ul>'+sums+
            '</ul><div class="scroll"><table><thead><tr><th>Scene</th><th>Controller</th><th>Outcome</th><th>Reason</th><th>Final error · mm</th>'
            '<th>First dwell · s</th><th>Wall · s</th><th>Evidence</th>'
            '</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div><h2>What this advances</h2><p>This environment provides a task and '
            'negative controls for a future small DVIDIA action policy. Footage review, robot actions, learned control and physical '
            'qualification remain separate stages. Rigid-block contact does not validate a soft shoe, laces or a real gripper.</p>'
            '<h2>Runtime and timing scope</h2><p>'+escape(report['protocol']['timing_boundary'])+'</p><p><code>'+escape(json.dumps(report['hardware']))+
            '</code></p><p>Each attempt retains its timing distribution, complete states, requested commands and outcome measurements. '
            'Warm model inference and camera-to-actuator latency are not measured by this authored-controller benchmark.</p>'
            '<footer><a href="https://research.dvidia.org/">Research</a> · <a href="https://github.com/Dvidia-Inference/dvidia-training">GitHub</a> · '
            '<a href="https://x.com/iammrriver">@iammrriver</a> · <a href="mailto:hello@dvidia.org">hello@dvidia.org</a></footer></main></html>')


def run_benchmark(assets, output, *, seed=411000, nominal=4, stress=2, cohort='development', progress=None):
    from .reference_task import ReferenceTaskConfig, run_episode, PROFILE
    protocol = make_protocol(seed=seed, nominal=nominal, stress=stress, cohort=cohort)
    provenance = sim.verify_assets(assets)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    (destination/'attempts').mkdir()
    _write(destination/'protocol.json', protocol)
    protocol_digest = sha256((destination/'protocol.json').read_bytes()).hexdigest()
    attempts = []
    for scene in protocol['scenes']:
        tape = None
        for control in CONTROLS:
            if implementation_hashes() != protocol['implementation_sha256']:
                raise ValueError('Implementation changed after protocol freeze; keep this incomplete run and use a fresh output')
            attempt_id = scene['id']+'--'+control
            started = time.perf_counter()
            try:
                if control == 'replay_open_jaw' and tape is None:
                    raise ValueError('Placement arm tape unavailable; matched control cannot run')
                result = run_episode(assets, config=ReferenceTaskConfig(**scene['config']), seed=scene['seed'],
                                     control=control, arm_tape=tape if control == 'replay_open_jaw' else None)
                if type(result.get('valid')) is not bool or type(result.get('success')) is not bool:
                    raise ValueError('Task runner omitted outcome framing')
                if control == 'placement':
                    candidate_tape = result.get('arm_tape')
                    expected_commands = round(scene['config']['timeout_s']/scene['config']['control_dt'])
                    tape = candidate_tape if (result['valid'] and type(candidate_tape) is list
                                              and len(candidate_tape) == expected_commands) else None
            except Exception as exc:
                result = {'valid': False, 'success': False, 'error_type': type(exc).__name__,
                          'error': str(exc).replace(str(Path(assets).resolve()), '[assets]')[:500]}
            result.update({'id': attempt_id, 'scene_id': scene['id'], 'control': control,
                           'profile': PROFILE, 'physical_robot_ready': False,
                           'seed': scene['seed'], 'config': scene['config'],
                           'attempt_wall_seconds': time.perf_counter()-started})
            result_file = 'attempts/'+attempt_id+'.json'
            _write(destination/result_file, result)
            if implementation_hashes() != protocol['implementation_sha256']:
                raise ValueError('Implementation changed during an attempt; partial evidence retained without a completed report')
            row = _summary(result)
            row['result_file'] = result_file
            attempts.append(row)
            if progress:
                progress({'id': attempt_id, 'valid': result['valid'], 'success': result['success']})
    report = {'schema': SCHEMA, 'profile': PROFILE, 'scope': SCOPE, 'physical_robot_ready': False, 'learned_policy': False,
              'cohort': cohort, 'protocol': protocol, 'protocol_sha256': protocol_digest,
              'implementation_sha256': protocol['implementation_sha256'], 'asset_provenance': provenance,
              'hardware': {'platform': platform.platform(), 'machine': platform.machine(), 'python': platform.python_version(),
                           'mujoco': '3.15.0', 'gpu_used': False, 'logical_cpu_count': os.cpu_count(),
                           'process_lifetime_peak_rss_bytes': _peak_rss_bytes(),
                           'memory_scope': 'Process lifetime high-water resident memory including imports, native models and traces, '
                                           'before final bundle inspection or replay export; not per-episode memory or minimum RAM. Null means unavailable.'},
              'scenes': protocol['scenes'], 'attempted': len(attempts), 'valid_attempts': sum(x['valid'] for x in attempts),
              'summary': aggregate(attempts, protocol['scenes']), 'attempts': attempts}
    _write(destination/'report.json', report)
    with (destination/'report.html').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(_html(report))
    names = ['report.json', 'report.html', 'protocol.json']+[x['result_file'] for x in attempts]
    files = {name: {'bytes': (destination/name).stat().st_size, 'sha256': sha256((destination/name).read_bytes()).hexdigest()} for name in names}
    _write(destination/'integrity.json', {'schema': 'dvidia.reference-task-integrity/v1', 'files': files})
    inspect_task_bundle(destination)
    return report


def load_task_bundle(directory):
    """Verify bounded files and structural consistency, not authenticity or recomputed physics."""
    from .reference_task import PROFILE
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Task bundle must be a regular directory')
    integrity_bytes = _read(root/'integrity.json', 65536)
    manifest = _decode(integrity_bytes)
    if type(manifest) is not dict or set(manifest) != {'schema', 'files'} or manifest['schema'] != 'dvidia.reference-task-integrity/v1':
        raise ValueError('Invalid task integrity schema')
    declarations = manifest['files']
    if type(declarations) is not dict or not 6 <= len(declarations) <= 75:
        raise ValueError('Invalid task bundle file count')
    files = {}
    total = 0
    for name, expected in declarations.items():
        if name not in ('report.json', 'report.html', 'protocol.json') and (type(name) is not str or re.fullmatch(r'attempts/[a-z0-9_-]{1,100}\.json', name) is None):
            raise ValueError('Task bundle file is outside the allowlist')
        if (root/name).parent.is_symlink():
            raise ValueError('Task artifact parent must not be a symlink')
        data = _read(root/name, MAX_FILE_BYTES)
        total += len(data)
        if total > MAX_BUNDLE_BYTES:
            raise ValueError('Task bundle exceeds byte bound')
        if type(expected) is not dict or set(expected) != {'bytes', 'sha256'} or type(expected['bytes']) is not int or expected['bytes'] != len(data) or expected['sha256'] != sha256(data).hexdigest():
            raise ValueError('Task artifact digest/size mismatch')
        files[name] = data
    if not {'report.json', 'report.html', 'protocol.json'} <= files.keys():
        raise ValueError('Missing task bundle documents')
    report, protocol = _decode(files['report.json']), _decode(files['protocol.json'])
    if type(report) is not dict or report.get('schema') != SCHEMA or report.get('physical_robot_ready') is not False or report.get('learned_policy') is not False:
        raise ValueError('Task bundle scope mismatch')
    if type(protocol) is not dict or protocol.get('schema') != 'dvidia.reference-task-protocol/v1':
        raise ValueError('Invalid frozen task protocol')
    if (report.get('profile') != PROFILE or report['profile'] != protocol.get('profile')
            or report.get('cohort') not in ('development','confirmation') or report['cohort'] != protocol.get('cohort')
            or report.get('implementation_sha256') != protocol.get('implementation_sha256')):
        raise ValueError('Task profile, cohort or implementation identity mismatch')
    hashes = report.get('implementation_sha256')
    if (type(hashes) is not dict or set(hashes) != set(SOURCE_FILES)
            or any(type(value) is not str or re.fullmatch(r'[0-9a-f]{64}',value) is None for value in hashes.values())):
        raise ValueError('Malformed task implementation hashes')
    if report.get('protocol_sha256') != sha256(files['protocol.json']).hexdigest() or report.get('protocol') != protocol or report.get('scenes') != protocol.get('scenes'):
        raise ValueError('Frozen protocol mismatch')
    scenes = report['scenes']
    if type(scenes) is not list or not 1 <= len(scenes) <= 24 or any(
            type(s) is not dict or type(s.get('id')) is not str or re.fullmatch(r'[a-z0-9_-]{1,60}', s['id']) is None
            or s.get('stratum') not in ('nominal', 'low_friction') or type(s.get('config')) is not dict
            or type(s.get('seed')) is not int or not 0 <= s['seed'] <= 999999 for s in scenes):
        raise ValueError('Invalid task scenes')
    ids = [s['id'] for s in scenes]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate task scene')
    expected = {(scene, control) for scene in ids for control in CONTROLS}
    if protocol.get('controls') != list(CONTROLS) or type(protocol.get('planned_attempts')) is not int or protocol['planned_attempts'] != len(expected):
        raise ValueError('Frozen planned task denominator mismatch')
    seen = set()
    attempts = report.get('attempts')
    if type(attempts) is not list or len(attempts) != len(expected) or type(report.get('attempted')) is not int or report['attempted'] != len(expected):
        raise ValueError('Missing planned task attempts')
    names = set()
    for row in attempts:
        if type(row) is not dict:
            raise ValueError('Invalid task attempt')
        if type(row.get('scene_id')) is not str or type(row.get('control')) is not str:
            raise ValueError('Invalid task attempt identity')
        identity = (row.get('scene_id'), row.get('control'))
        if identity not in expected or identity in seen or type(row.get('valid')) is not bool or type(row.get('success')) is not bool or (row['success'] and not row['valid']):
            raise ValueError('Duplicate, missing or inconsistent task attempt')
        seen.add(identity)
        source_scene = next(s for s in scenes if s['id'] == identity[0])
        if (row.get('id') != identity[0]+'--'+identity[1] or row.get('profile') != report['profile']
                or row.get('physical_robot_ready') is not False or row.get('config') != source_scene.get('config')
                or type(row.get('seed')) is not int or row['seed'] != source_scene.get('seed')):
            raise ValueError('Task attempt disagrees with its declared scene or profile')
        name = row.get('result_file')
        if type(name) is not str or name not in files or name in names or not name.startswith('attempts/'):
            raise ValueError('Invalid task attempt file')
        names.add(name)
        raw = _decode(files[name])
        if type(raw) is not dict or _summary(raw) != {k:v for k,v in row.items() if k != 'result_file'}:
            raise ValueError('Attempt summary differs from sealed trace')
    if names != files.keys()-{'report.json','report.html','protocol.json'} or report.get('summary') != aggregate(attempts, scenes):
        raise ValueError('Task denominator or summary mismatch')
    if type(report.get('valid_attempts')) is not int or report['valid_attempts'] != sum(x['valid'] for x in attempts):
        raise ValueError('Task valid count mismatch')
    files['integrity.json'] = integrity_bytes
    return report, files


def inspect_task_bundle(directory):
    report, files = load_task_bundle(directory)
    return {'schema': 'dvidia.reference-task-inspection/v1', 'scope': 'Artifact consistency only; no authentication or physics recomputation',
            'files_verified': len(files)-1, 'attempted': report['attempted'], 'valid_attempts': report['valid_attempts'],
            'summary': report['summary'], 'physical_robot_ready': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description='DVIDIA native reference contact tasks; local CPU, simulation only')
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('benchmark', help='Freeze a protocol and retain every authored/control attempt')
    run.add_argument('assets', type=Path)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--seed', type=int, default=411000)
    run.add_argument('--nominal', type=int, default=4)
    run.add_argument('--stress', type=int, default=2)
    run.add_argument('--cohort', choices=['development','confirmation'], default='development')
    inspect = commands.add_parser('inspect', help='Check sealed artifact and denominator consistency')
    inspect.add_argument('bundle', type=Path)
    replay = commands.add_parser('export-replay', help='Reconstruct recorded poses without stepping physics')
    replay.add_argument('assets', type=Path)
    replay.add_argument('bundle', type=Path)
    replay.add_argument('--output', type=Path, required=True)
    replay.add_argument('--scene', action='append', dest='scene_ids')
    args = parser.parse_args(argv)
    try:
        if args.command == 'benchmark':
            value = run_benchmark(args.assets, args.output, seed=args.seed, nominal=args.nominal, stress=args.stress, cohort=args.cohort,
                                  progress=lambda x: print(json.dumps(x), file=sys.stderr, flush=True))
            value = {'output': str(args.output), 'attempted': value['attempted'], 'summary': value['summary']}
        elif args.command == 'inspect':
            value = inspect_task_bundle(args.bundle)
        else:
            from .reference_task_replay import export_task_replay
            value = export_task_replay(args.assets, args.bundle, args.output, scene_ids=args.scene_ids)
        print(json.dumps(value, indent=2, allow_nan=False))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()

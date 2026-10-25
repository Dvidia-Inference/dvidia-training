"""Persisted video-goal candidates, fresh-process PiPER runs and matched controls.

The source clips are the earlier synthetic destination pilot. Two new starts per
clip test installation/reload under changed initial state, not learned transfer.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import hashlib
from html import escape
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .video_goal_benchmark import score_reference

VARIANTS = ('oracle', 'installed_video', 'shuffled_video', 'open_jaw')
STARTS = ((.294, -.025), (.307, -.038))
MAX_EPISODE_BYTES = 64 * 1024 * 1024


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _cli(args):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parent.parent))
    command = [sys.executable, '-m', 'dvidia_training.video_goal_install', *map(str, args)]
    start = time.perf_counter()
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
    if result.returncode:
        # Native Python messages contain paths/configuration; never print env.
        raise ValueError('Candidate subprocess failed: '+result.stderr[-1400:])
    return json.loads(result.stdout), time.perf_counter()-start


def _read_episode(path):
    with gzip.open(path, 'rb') as stream:
        raw = stream.read(MAX_EPISODE_BYTES+1)
    if len(raw) > MAX_EPISODE_BYTES: raise ValueError('Expanded episode exceeds its byte limit')
    return json.loads(raw)


def _evidence_file(root, name, inventory):
    if (type(name) is not str or not name or name.startswith('/') or '\\' in name
            or any(p in ('', '.', '..') for p in name.split('/')) or name not in inventory):
        raise ValueError('Unsafe or unsealed evidence reference')
    path = Path(root)/name
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Evidence reference escapes its directory')
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError('Evidence references must be local regular files')
    return path


def _save_episode(path, episode):
    with Path(path).open('xb') as stream:
        stream.write(gzip.compress(json.dumps(episode, separators=(',', ':'), allow_nan=False).encode(), mtime=0))


def inspect_evidence(root):
    """Check saved bytes, execution bindings and recomputed scores, not physics."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir(): raise ValueError('Use local evidence')
    inventory = json.loads((root/'integrity.json').read_text())['files']
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p != root/'integrity.json'}
    if actual != set(inventory): raise ValueError('Evidence inventory changed')
    for name, expected in inventory.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name:
            raise ValueError('Unsafe evidence path')
        path = root/path
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('Evidence symlinks are not allowed')
        if path.stat().st_size != expected['bytes'] or _sha(path) != expected['sha256']:
            raise ValueError('Evidence hash mismatch: '+name)
    protocol = json.loads((root/'protocol.json').read_text())
    report = json.loads((root/'report.json').read_text())
    if report['protocol_sha256'] != _sha(root/'protocol.json'):
        raise ValueError('Protocol binding changed')
    if protocol['benchmark_source_sha256'] != _sha(root/'sources'/'video_goal_install_benchmark.py'):
        raise ValueError('Benchmark source binding changed')
    if protocol['scoring_source_sha256'] != _sha(Path(__file__).with_name('video_goal_benchmark.py')):
        raise ValueError('Current scorer differs from frozen scorer')
    cases = {c['id']: c for c in protocol['cases']}
    expected_ids = {f'{c}-start{i}-{v}' for c in cases for i in range(len(protocol['starts_xy_m'])) for v in protocol['variants']}
    if len(report['attempts']) != len(expected_ids) or {r['id'] for r in report['attempts']} != expected_ids:
        raise ValueError('Scheduled attempt inventory changed')
    for row in report['attempts']:
        if not row['attempted']: continue
        episode_path = _evidence_file(root, row['episode_file'], inventory)
        episode = _read_episode(episode_path)
        case = cases[row['case_id']]
        primary = score_reference(episode, case['task_goal_xy_m'])
        secondary = score_reference(episode, case['demonstrated_final_xy_m'])
        if (primary != row['score'] or secondary != row['secondary_score']
                or primary['success'] != row['primary_success']
                or secondary['success'] != row['demonstrated_endpoint_success']):
            raise ValueError('Saved score cannot be recomputed')
        if row['variant'] in ('installed_video', 'shuffled_video'):
            receipt_name = str(Path(row['episode_file']).with_name('receipt.json'))
            receipt = json.loads(_evidence_file(root, receipt_name, inventory).read_text())
            if receipt['episode']['sha256'] != _sha(episode_path):
                raise ValueError('Execution episode binding changed')
            bundle_manifest = _evidence_file(root, 'bundles/'+row['bundle_source_id']+'/manifest.json', inventory)
            if receipt['bundle_manifest_sha256'] != _sha(bundle_manifest):
                raise ValueError('Execution bundle binding changed')
            if receipt['runtime'] != protocol['runtime']:
                raise ValueError('Execution runtime differs from protocol')
    for variant in protocol['variants']:
        rows = [r for r in report['attempts'] if r['variant'] == variant]
        expected = dict(scheduled=len(rows), executed=sum(r['attempted'] for r in rows),
            primary_successes=sum(r['primary_success'] for r in rows),
            demonstrated_endpoint_successes=sum(r['demonstrated_endpoint_success'] for r in rows))
        if report['variants'][variant] != expected: raise ValueError('Reported denominator changed')
    return dict(verified_files=len(inventory), attempts=len(report['attempts']),
                scores_recomputed=sum(r['attempted'] for r in report['attempts']), physics_rerun=False)


def run(source, assets, output):
    from .robot_profiles import verify_assets
    from .video_goal_benchmark import inspect_run
    from .piper_task import PiperTaskConfig, run_episode
    from .video_goal_install import runtime_contract

    source, assets, output = map(Path, (source, assets, output))
    if output.exists() or output.is_symlink(): raise ValueError('Use a fresh output directory')
    inspect_run(source)  # All original sealed files must still match.
    report = json.loads((source/'report.json').read_text())
    if len(report['sources']) != 4: raise ValueError('Use the four-layout source pilot')
    provenance = verify_assets('piper', assets)
    cases = [dict(id=row['id'], clip=row['clip'], task_goal_xy_m=row['task_goal_xy_m'],
                  demonstrated_final_xy_m=row['source_final_xy_m'], seed=row['seed'],
                  clip_sha256=_sha(source/row['clip'])) for row in report['sources']]
    output.mkdir(parents=True)
    protocol = dict(schema='dvidia.video-goal-install-protocol/v1', cases=cases,
        starts_xy_m=[list(p) for p in STARTS], variants=list(VARIANTS),
        scheduled_attempts=len(cases)*len(STARTS)*len(VARIANTS),
        source_report_sha256=_sha(source/'report.json'),
        source_inventory_sha256=_sha(source/'integrity.json'),
        runtime=runtime_contract(), robot_assets=provenance,
        benchmark_source_sha256=_sha(Path(__file__)),
        scoring_source_sha256=_sha(Path(__file__).with_name('video_goal_benchmark.py')),
        primary_reference='Original intended source-task XY; 25mm tolerance and native lift/release/rest/contact gates.',
        secondary_reference='Actually demonstrated terminal object XY; reported separately without replacing the primary reference.',
        selection='All 32 predeclared attempts remain; one invocation each, no result-based reruns or parameter tuning.',
        scope='Synthetic calibrated green rigid block; authored PiPER control using privileged native state. No motor model trained.',
        missing_arm_check='SO101 candidate inspection must reject; no compatible contact task adapter exists.',
        unknown_video_check='Four source terminal-occlusion clips must refuse candidate creation.',
        physical_robot_ready=False, motor_policy_learned=False)
    _write(output/'protocol.json', protocol)
    (output/'sources').mkdir()
    for name in ('video_goal_install.py', 'video_goal_install_benchmark.py', 'video_goal_benchmark.py'):
        shutil.copyfile(Path(__file__).with_name(name), output/'sources'/name)
    shutil.copyfile(source/'calibration.json', output/'calibration.json')
    (output/'clips').mkdir()
    bundles, installs, rejections = {}, [], []
    started = time.perf_counter()
    for case in cases:
        print(json.dumps(dict(stage='create', case=case['id'])), flush=True)
        clip = output/'clips'/case['clip']; shutil.copyfile(source/case['clip'], clip)
        bundle = output/'bundles'/case['id']
        item = dict(case_id=case['id'], created=False, error=None)
        try:
            created, elapsed = _cli(['create', '--video', clip, '--calibration', output/'calibration.json', '--output', bundle])
            loaded, reload_elapsed = _cli(['inspect', '--bundle', bundle, '--profile', 'piper'])
            bundles[case['id']] = bundle
            item.update(created=True, creation_wall_s=elapsed, reload_wall_s=reload_elapsed,
                        manifest=loaded, creation_receipt=created)
        except Exception as exc: item['error'] = str(exc)[:1800]
        installs.append(item)
        rejected = False
        refusal_error = None
        try: _cli(['inspect', '--bundle', bundle, '--profile', 'so101'])
        except ValueError as exc:
            refusal_error = str(exc)
            rejected = item['created'] and 'Only the piper simulation profile is supported' in refusal_error
        rejections.append(dict(case_id=case['id'], intervention='unsupported_so101_profile',
                               correctly_refused=rejected, error=refusal_error))
        fault = output/'clips'/(case['id']+'-terminal_occlusion.mp4')
        shutil.copyfile(source/fault.name, fault)
        rejected = False
        refusal_error = None
        try: _cli(['create', '--video', fault, '--calibration', output/'calibration.json',
                   '--output', output/'refused'/(case['id']+'-occluded')])
        except ValueError as exc:
            refusal_error = str(exc)
            rejected = 'Only a proposed video destination can be installed' in refusal_error
        rejections.append(dict(case_id=case['id'], intervention='terminal_occlusion',
                               correctly_refused=rejected, error=refusal_error))
    rows = []
    for index, case in enumerate(cases):
        for start_index, start in enumerate(STARTS):
            installed_episode = None
            config = replace(PiperTaskConfig(), object_position_m=(*start, .082),
                             target_position_m=(*case['task_goal_xy_m'], .08))
            seed = case['seed']+10000+start_index
            for variant in VARIANTS:
                identifier = f'{case["id"]}-start{start_index}-{variant}'
                print(json.dumps(dict(stage='execute', attempt=identifier)), flush=True)
                row = dict(id=identifier, case_id=case['id'], start_index=start_index,
                    variant=variant, attempted=False, error=None, primary_success=False,
                    demonstrated_endpoint_success=False, score=None, secondary_score=None)
                target = output/'attempts'/identifier
                attempt_start = time.perf_counter()
                try:
                    if variant in ('installed_video', 'shuffled_video'):
                        source_case = case if variant == 'installed_video' else cases[(index+1) % len(cases)]
                        bundle = bundles[source_case['id']]
                        receipt, subprocess_wall = _cli(['run', '--bundle', bundle, '--assets', assets,
                            '--object-x', start[0], '--object-y', start[1], '--output', target, '--seed', seed])
                        episode = _read_episode(target/'episode.json.gz')
                        row.update(execution_receipt=receipt, subprocess_wall_s=subprocess_wall,
                                   bundle_source_id=source_case['id'])
                        if variant == 'installed_video': installed_episode = episode
                    else:
                        target.mkdir(parents=True)
                        if variant == 'open_jaw':
                            if installed_episode is None: raise ValueError('No matched installed episode')
                            tape = installed_episode['arm_tape']
                            if len(tape) != 1000: raise ValueError('No full-duration arm tape for paired open jaw')
                            installed_goal = installed_episode['config']['target_position_m']
                            episode = run_episode(assets, replace(config, target_position_m=tuple(installed_goal)),
                                seed=seed, control='replay_open_jaw', arm_tape=tape)
                        else: episode = run_episode(assets, config, seed=seed, control='placement')
                        _save_episode(target/'episode.json.gz', episode)
                    row['attempted'] = True
                    row['episode_file'] = str((target/'episode.json.gz').relative_to(output))
                    row['score'] = score_reference(episode, case['task_goal_xy_m'])
                    row['primary_success'] = row['score']['success']
                    row['secondary_score'] = score_reference(episode, case['demonstrated_final_xy_m'])
                    row['demonstrated_endpoint_success'] = row['secondary_score']['success']
                except Exception as exc: row['error'] = dict(type=type(exc).__name__, message=str(exc)[:1800])
                row['wall_seconds'] = time.perf_counter()-attempt_start
                rows.append(row)
    result = dict(schema='dvidia.video-goal-install-benchmark/v1', protocol_sha256=_sha(output/'protocol.json'),
        installations=installs, rejections=rejections, attempts=rows,
        variants={v:dict(scheduled=len(cases)*len(STARTS), executed=sum(r['attempted'] for r in rows if r['variant']==v),
            primary_successes=sum(r['primary_success'] for r in rows if r['variant']==v),
            demonstrated_endpoint_successes=sum(r['demonstrated_endpoint_success'] for r in rows if r['variant']==v)) for v in VARIANTS},
        wall_seconds=time.perf_counter()-started, motor_policy_learned=False, physical_robot_ready=False)
    _write(output/'report.json', result)
    _html(output, result)
    _write(output/'integrity.json', dict(files={str(p.relative_to(output)):dict(bytes=p.stat().st_size, sha256=_sha(p))
        for p in sorted(output.rglob('*')) if p.is_file()}))
    return result


def _html(output, report):
    rows = ''.join(f'<tr><td>{escape(v)}</td><td>{r["executed"]}/{r["scheduled"]}</td><td>{r["primary_successes"]}/{r["scheduled"]}</td><td>{r["demonstrated_endpoint_successes"]}/{r["scheduled"]}</td></tr>' for v,r in report['variants'].items())
    details = ''.join(f'<tr><td>{escape(r["id"])}</td><td>{r["primary_success"]}</td><td>{escape(str(r["error"]))}</td></tr>' for r in report['attempts'])
    (output/'report.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DVIDIA · persisted video goal pilot</title><style>body{{max-width:960px;margin:48px auto;padding:0 20px;background:#faf9f5;color:#26312e;font:18px/1.6 Georgia,serif}}h1,h2{{font-family:system-ui}}table{{width:100%;border-collapse:collapse;font-size:15px}}td,th{{text-align:left;border-bottom:1px solid #ddd;padding:10px}}a{{color:#315b4d}}</style><p>DVIDIA research · local candidate</p><h1>A video destination that survives installation.</h1><p>Four synthetic PiPER videos. Two new starts each. Candidates are created, reloaded and executed in separate Python processes. No motor model is trained. The controller uses privileged simulator observations.</p><p>Primary success uses the original intended task, separately from replaying the actual demonstration endpoint. All failed attempts remain.</p><table><thead><tr><th>Condition</th><th>Executed</th><th>Intended task</th><th>Demo endpoint (secondary)</th></tr></thead><tbody>{rows}</tbody></table><p><a href="report.json">Complete results</a> · <a href="protocol.json">Frozen protocol</a> · <a href="integrity.json">Evidence hashes</a></p><h2>Every attempt</h2><table>{details}</table></html>''')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--assets', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    result = run(args.source, args.assets, args.output)
    print(json.dumps(dict(variants=result['variants'], wall_seconds=result['wall_seconds'])))


if __name__ == '__main__': main()

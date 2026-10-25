"""Calibrated video -> destination -> authored PiPER control, with paired controls.

This spatial grounding pilot learns no motor policy. Source clips are native
simulation replays; inference receives encoded pixels and a declared calibration.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gzip
import hashlib
from html import escape
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np

SCHEMA = 'dvidia.video-goal-benchmark/v1'
VARIANTS = ('oracle_goal', 'video_goal', 'shuffled_video', 'miscalibrated_video', 'open_jaw')
FAULTS = ('terminal_occlusion', 'extra_green_candidate', 'terminal_camera_motion')
SOURCE_FILES = ('video_goal_benchmark.py', 'video_goal_bridge.py', 'piper_task.py',
                'reference_task.py', 'reference_sim.py', 'reference_replay.py',
                'robot_profiles.py', 'robot_profiles.lock.json', 'sensor_goal_observers.py',
                'footage_data.py', 'piper_grip_guard.py', 'piper_contact_memory.py',
                'piper_contact_recovery.py', 'piper_reacquisition.py', 'piper_grasp_strategy.py')
LIMITS = [
    'All videos and trials are synthetic PiPER rigid-block simulations, not human footage or shoes.',
    'A known green proxy and a trusted fixed orthographic camera supply XY only. No identity, depth, force or contact is inferred from pixels.',
    'The policy, grasp phases, IK and recovery are authored. This experiment trains no motor model and has no action-label learning curve.',
    'The controller still uses privileged live simulator joint/object/contact observations. Only its destination comes from the video.',
    'A stationary visible centroid cannot certify correct completion or physical rest. A wrong calibration or same-color lookalike can produce a wrong proposal.',
    'Compute latency excludes capture, the observation dwell, network, transport and physical actuation. These trials do not qualify a physical robot.',
]


def _write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_protocol(*, seed=2026102507, count=4):
    from .video_goal_bridge import PlanarVideoCalibration
    if type(seed) is not int or not 0 <= seed <= 9999999999 or type(count) is not int or not 2 <= count <= 4:
        raise ValueError('Use two to four layouts and a bounded integer seed')
    rng = np.random.default_rng(seed)
    targets = ((.34, .08), (.40, -.08), (.44, .07), (.34, -.12))
    cases = []
    for i, target in enumerate(targets[:count]):
        start = [float(x) for x in np.array([.30, -.03])+rng.uniform(-.002, .002, 2)]
        goal = [float(x) for x in np.array(target)+rng.uniform(-.002, .002, 2)]
        cases.append(dict(id=f'layout-{i:02}', seed=int(seed % 900000)+i,
                          object_xy_m=start, task_goal_xy_m=goal,
                          shuffled_source_id=f'layout-{(i+1) % count:02}'))
    return dict(schema=SCHEMA, seed=seed, planned_layouts=count, cases=cases,
        variants=list(VARIANTS), faults=list(FAULTS), planned_source_episodes=count,
        planned_controller_attempts=count*len(VARIANTS), planned_fault_clips=count*len(FAULTS),
        calibration=asdict(PlanarVideoCalibration()),
        video=dict(width=320, height=240, fps=5, maximum_duration_s=30,
                   origin='Native teacher rollout replay at recorded 200ms boundaries; no human demonstrations.'),
        scoring=dict(position_tolerance_m=.025, terminal_dwell_s=.25, proposal_error_tolerance_m=.010,
                     minimum_lift_m=.06, tool_retreat_m=.10,
                     linear_speed_max_m_s=.035, angular_speed_max_rad_s=.3),
        selection='All scheduled layouts, sources, faults and controller attempts remain in the report. No outcome-based retries.',
        physical_robot_ready=False, motor_policy_learned=False, limitations=LIMITS)


def score_reference(attempt, reference_xy, *, tolerance=.025, dwell_s=.25):
    """Score actual trajectory against the same reference, even for wrong-goal arms.

    The controller's own destination/success flag cannot change this reference.
    Native validity, lift and retention diagnostics accompany the terminal test.
    """
    states = attempt.get('trajectory', [])
    metrics = attempt.get('metrics', {})
    streak_start = None
    for state in states:
        position = state['object_position_m']
        qualified = (math.dist(position[:2], reference_xy) <= tolerance
            and state['table_object_contact_count'] > 0
            and not any(state['grasp_contact_counts'].values())
            and state['gripper_joint_position_m'][0] >= .028
            and state['gripper_joint_position_m'][1] <= -.028
            and np.linalg.norm(state['object_linear_velocity_m_s']) <= .035
            and np.linalg.norm(state['object_angular_velocity_rad_s']) <= .3
            and state['tcp_position_m'][2]-position[2] >= .10)
        if not qualified: streak_start = None
        elif streak_start is None: streak_start = state['t_seconds']
    terminal_span = states[-1]['t_seconds']-streak_start if states and streak_start is not None else 0.
    manipulation = (attempt.get('valid') is True and metrics.get('lift_seen') is True
        and metrics.get('carry_complete') is True and metrics.get('retention_lost') is False
        and metrics.get('max_object_lift_m', 0.) >= .06
        and metrics.get('unexpected_contact_native_ticks') == 0)
    return dict(success=bool(manipulation and terminal_span+1e-8 >= dwell_s),
        reference_xy_m=list(reference_xy), terminal_dwell_s=terminal_span,
        terminal_xy_error_m=math.dist(states[-1]['object_position_m'][:2], reference_xy) if states else None,
        valid_native_physics=attempt.get('valid', False), retained_transfer=bool(manipulation),
        controller_self_success=attempt.get('success', False),
        scoring_scope='Native state and contact truth against fixed demonstrated task reference; unavailable to pixel inference.')


def _save_attempt(root, name, attempt):
    path = root/'attempts'/f'{name}.json.gz'; path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(attempt, separators=(',', ':'), allow_nan=False).encode()
    with path.open('xb') as stream: stream.write(gzip.compress(raw, mtime=0))
    return str(path.relative_to(root))


def _encode(path, frames, fps=5):
    height, width, _ = frames[0].shape
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps),
        '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p',
        '-threads', '1', str(path)]
    result = subprocess.run(command, input=b''.join(f.tobytes() for f in frames),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    if result.returncode: raise RuntimeError('Synthetic clip encoding failed')


def render_source(assets, attempt, calibration):
    """Render native 3D geometry from recorded states; no sensing labels encoded."""
    import mujoco as mj
    from .piper_task import build_task_model, PiperTaskConfig
    model = build_task_model(assets, PiperTaskConfig(**attempt['config']))
    model.geom_rgba[model.geom('task_block_geom').id] = [.10, .65, .20, 1.]
    # The task's destination marker must not masquerade as the observed object.
    model.geom_rgba[model.geom('task_target').id] = [0., 0., 0., 0.]
    model.vis.global_.orthographic = 1
    model.vis.global_.fovy = calibration.vertical_span_m
    model.vis.global_.ipd = 0
    model.vis.global_.offwidth = max(calibration.width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(calibration.height, model.vis.global_.offheight)
    camera = mj.MjvCamera(); camera.lookat[:] = [*calibration.center_xy_m, 0.]
    camera.distance = 1.; camera.azimuth = 90.; camera.elevation = -90.
    data = mj.MjData(model); frames, bindings = [], []
    # This is image rendering of an already measured physics trace, not a second
    # physics evaluation. No simulator values are passed to the vision observer.
    with mj.Renderer(model, calibration.height, calibration.width) as renderer:
        for state in attempt['trajectory'][::10]:
            data.qpos[:] = state['qpos']; data.qvel[:] = state['qvel']
            mj.mj_forward(model, data); renderer.update_scene(data, camera=camera)
            frames.append(renderer.render().copy())
            bindings.append(dict(t_s=state['t_seconds'], object_xy_m=state['object_position_m'][:2]))
    if len(frames) < 4: raise ValueError('Native source has insufficient replay history')
    return frames, bindings


def _fault_frames(frames, fault):
    out = [f.copy() for f in frames]
    if fault == 'terminal_occlusion':
        for frame in out[-5:]: frame[:] = 20
    elif fault == 'extra_green_candidate':
        for frame in out: frame[20:40, 275:295] = [20, 180, 30]
    elif fault == 'terminal_camera_motion':
        for i in range(1, 7): out[-7+i] = np.roll(out[-7+i], i*5, axis=1)
    else: raise ValueError('Unknown pixel intervention')
    return out


def _execute(assets, config, seed, *, arm_tape=None):
    from .piper_task import run_episode
    return run_episode(assets, config, seed=seed,
                       control='replay_open_jaw' if arm_tape is not None else 'placement',
                       arm_tape=arm_tape)


def run(assets, output, *, seed=2026102507, count=4):
    from .piper_task import PiperTaskConfig
    from .video_goal_bridge import PlanarVideoCalibration, infer_video_goal
    from .robot_profiles import verify_assets
    output = Path(output)
    if output.exists() or output.is_symlink(): raise ValueError('Choose a fresh output directory')
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'): raise ValueError('Install FFmpeg/ffprobe first')
    assets = Path(assets); provenance = verify_assets('piper', assets)
    protocol = make_protocol(seed=seed, count=count)
    protocol['asset_provenance'] = provenance
    protocol['source_sha256'] = {n: _sha(Path(__file__).with_name(n)) for n in SOURCE_FILES}
    protocol['host'] = dict(platform=platform.platform(), numpy=np.__version__)
    output.mkdir(parents=True)
    _write(output/'protocol.json', protocol)
    for name in SOURCE_FILES:
        path = output/'sources'/name; path.parent.mkdir(exist_ok=True)
        shutil.copyfile(Path(__file__).with_name(name), path)
    calibration = PlanarVideoCalibration(**protocol['calibration'])
    _write(output/'calibration.json', asdict(calibration))
    started = time.perf_counter(); sources = []; attempts = []; faults = []
    def progress(value):
        with (output/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False)+'\n')
        print(json.dumps(value), flush=True)
    for case in protocol['cases']:
        progress(dict(stage='source', case=case['id']))
        config = replace(PiperTaskConfig(), object_position_m=(*case['object_xy_m'], .082),
                         target_position_m=(*case['task_goal_xy_m'], .08))
        episode = _execute(assets, config, case['seed'])
        row = dict(case, source_attempt=_save_attempt(output, case['id']+'-source', episode),
                   source_score=score_reference(episode, case['task_goal_xy_m']), clip=None,
                   proposal=None, error=None)
        try:
            frames, bindings = render_source(assets, episode, calibration)
            clip = output/(case['id']+'.mp4'); _encode(clip, frames)
            _write(output/(case['id']+'-render-truth.json'), dict(frames=bindings, observer_access=False))
            row.update(clip=clip.name, proposal=infer_video_goal(clip, calibration=calibration))
            final_xy = episode['trajectory'][-1]['object_position_m'][:2]
            row['source_final_xy_m'] = final_xy
            row['proposal_error_m'] = math.dist(row['proposal']['goal_xy_m'], final_xy) if row['proposal']['status'] == 'proposed' else None
            row['proposal_within_10mm'] = row['proposal_error_m'] is not None and row['proposal_error_m'] <= .010
            row['within_declared_position_bound'] = row['proposal_error_m'] is not None and row['proposal_error_m'] <= calibration.position_bound_m
            _write(output/(case['id']+'-proposal.json'), row['proposal'])
        except Exception as exc:
            row['error'] = dict(type=type(exc).__name__, message=str(exc)[:500])
        for fault in FAULTS:
            fault_row = dict(case_id=case['id'], fault=fault, clip=None, proposal=None,
                             evaluated=False, correct_abstention=False, error=None)
            try:
                if row['error'] is not None: raise ValueError('Source clip failed; no matched fault clip')
                fault_clip = output/(case['id']+'-'+fault+'.mp4')
                _encode(fault_clip, _fault_frames(frames, fault))
                proposal = infer_video_goal(fault_clip, calibration=calibration)
                fault_row.update(clip=fault_clip.name, proposal=proposal, evaluated=True,
                                 correct_abstention=proposal['status'] == 'unknown')
            except Exception as exc:
                fault_row['error'] = dict(type=type(exc).__name__, message=str(exc)[:500])
            faults.append(fault_row)
        sources.append(row); progress(dict(stage='source_done', case=case['id'], source_success=row['source_score']['success'], error=row['error']))
        _write(output/(case['id']+'-source-result.json'), row)
    indexed = {s['id']: s for s in sources}
    for source in sources:
        video_episode = None
        config = replace(PiperTaskConfig(), object_position_m=(*source['object_xy_m'], .082),
                         target_position_m=(*source['task_goal_xy_m'], .08))
        for variant in VARIANTS:
            progress(dict(stage='controller', case=source['id'], variant=variant))
            row = dict(case_id=source['id'], variant=variant, attempted=False, success=False, error=None,
                       goal_xy_m=None, source_clip_id=source['id'], score=None, episode_file=None)
            try:
                proposal = source['proposal']
                if variant == 'shuffled_video':
                    row['source_clip_id'] = source['shuffled_source_id']
                    proposal = indexed[source['shuffled_source_id']]['proposal']
                elif variant == 'miscalibrated_video' and source['clip']:
                    bad = replace(calibration, center_xy_m=(calibration.center_xy_m[0], calibration.center_xy_m[1]+.06))
                    proposal = infer_video_goal(output/source['clip'], calibration=bad)
                    row['calibration_intervention_m'] = [0., .06]
                if variant == 'oracle_goal': goal = source['task_goal_xy_m']
                elif variant == 'open_jaw':
                    if video_episode is None or not video_episode['valid'] or len(video_episode['arm_tape']) != 1000:
                        raise ValueError('No complete exact-duration video-goal tape for matched open-jaw replay')
                    goal = source['proposal']['goal_xy_m']
                elif proposal is None or proposal['status'] != 'proposed':
                    raise ValueError('No observable video destination; no controller was run')
                else: goal = proposal['goal_xy_m']
                actual = replace(config, target_position_m=(*goal, .08))
                row.update(attempted=True, goal_xy_m=list(goal))
                episode = _execute(assets, actual, source['seed'],
                    arm_tape=video_episode['arm_tape'] if variant == 'open_jaw' else None)
                if variant == 'video_goal': video_episode = episode
                row['episode_file'] = _save_attempt(output, source['id']+'-'+variant, episode)
                row['score'] = score_reference(episode, source['task_goal_xy_m'])
                row['success'] = row['score']['success']
                row['native_error'] = episode.get('error')
                row['wall_seconds'] = episode['timing']['episode_wall_seconds']
            except Exception as exc:
                row['error'] = dict(type=type(exc).__name__, message=str(exc)[:500])
            attempts.append(row)
            _write(output/(source['id']+'-'+variant+'-result.json'), row)
    report = dict(schema=SCHEMA, protocol_sha256=_sha(output/'protocol.json'),
        sources=sources, attempts=attempts, faults=faults, limitations=LIMITS,
        wall_seconds=time.perf_counter()-started, physical_robot_ready=False, motor_policy_learned=False,
        variants={v:dict(scheduled=count, executed=sum(r['attempted'] for r in attempts if r['variant'] == v),
            successes=sum(r['success'] for r in attempts if r['variant'] == v)) for v in VARIANTS},
        fault_summary={f:dict(scheduled=count, evaluated=sum(r['fault'] == f and r['evaluated'] for r in faults),
            abstentions=sum(r['fault'] == f and r['correct_abstention'] for r in faults)) for f in FAULTS})
    _write(output/'report.json', report)
    _html(output, report)
    _write(output/'integrity.json', dict(schema='dvidia.video-goal-integrity/v1', files={
        str(p.relative_to(output)):dict(sha256=_sha(p), bytes=p.stat().st_size)
        for p in sorted(output.rglob('*')) if p.is_file()}))
    return report


def inspect_run(root, *, reprocess_video=False):
    """Verify retained bytes and rederive scores; does not rerun native dynamics."""
    root = Path(root).resolve(strict=True)
    seal = json.loads((root/'integrity.json').read_text())
    if seal.get('schema') != 'dvidia.video-goal-integrity/v1' or not 1 <= len(seal['files']) <= 5000:
        raise ValueError('Invalid bounded benchmark seal')
    for name, entry in seal['files'].items():
        if '\\' in name or ':' in name or any(p in ('', '.', '..') for p in name.split('/')) or name.startswith('/'):
            raise ValueError('Unsafe sealed path')
        path = root/name
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError('Sealed path escaped benchmark')
        if path.stat().st_size != entry['bytes'] or _sha(path) != entry['sha256']:
            raise ValueError('Changed sealed file: '+name)
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name != 'integrity.json'}
    if actual != set(seal['files']): raise ValueError('Benchmark inventory differs from seal')
    report = json.loads((root/'report.json').read_text())
    protocol = json.loads((root/'protocol.json').read_text())
    if report['protocol_sha256'] != _sha(root/'protocol.json'): raise ValueError('Changed protocol binding')
    cases = {c['id']: c for c in protocol['cases']}
    if len(report['sources']) != len(cases) or {s['id'] for s in report['sources']} != set(cases):
        raise ValueError('Missing or duplicated source result')
    expected = {(case, v) for case in cases for v in protocol['variants']}
    got = [(r['case_id'], r['variant']) for r in report['attempts']]
    if len(got) != len(expected) or set(got) != expected: raise ValueError('Missing or duplicated controller attempt')
    expected_faults = {(case, f) for case in cases for f in protocol['faults']}
    got_faults = [(r['case_id'], r['fault']) for r in report['faults']]
    if len(got_faults) != len(expected_faults) or set(got_faults) != expected_faults: raise ValueError('Missing or duplicated fault attempt')
    def episode(name):
        if name not in seal['files']: raise ValueError('Unsealed episode')
        with gzip.open(root/name, 'rb') as stream: raw = stream.read(32*1024*1024+1)
        if len(raw) > 32*1024*1024: raise ValueError('Oversized episode')
        return json.loads(raw)
    for source in report['sources']:
        goal = cases[source['id']]['task_goal_xy_m']
        if score_reference(episode(source['source_attempt']), goal) != source['source_score']:
            raise ValueError('Source score differs from retained native trace')
    for row in report['attempts']:
        if row['episode_file']:
            score = score_reference(episode(row['episode_file']), cases[row['case_id']]['task_goal_xy_m'])
            if score != row['score'] or score['success'] != row['success']:
                raise ValueError('Controller score differs from retained native trace')
        elif row['success']: raise ValueError('Attempt without evidence cannot succeed')
    for variant, summary in report['variants'].items():
        rows = [r for r in report['attempts'] if r['variant'] == variant]
        if summary != dict(scheduled=len(cases), executed=sum(r['attempted'] for r in rows), successes=sum(r['success'] for r in rows)):
            raise ValueError('Variant summary differs from attempted denominators')
    for fault, summary in report['fault_summary'].items():
        rows = [r for r in report['faults'] if r['fault'] == fault]
        if summary != dict(scheduled=len(cases), evaluated=sum(r['evaluated'] for r in rows), abstentions=sum(r['correct_abstention'] for r in rows)):
            raise ValueError('Fault summary differs from scheduled denominators')
    if reprocess_video:
        from .video_goal_bridge import infer_video_goal, PlanarVideoCalibration
        for name, sha in protocol['source_sha256'].items():
            if _sha(Path(__file__).with_name(name)) != sha: raise ValueError('Use the frozen source before reprocessing videos')
        c = PlanarVideoCalibration(**protocol['calibration'])
        for row in [*report['sources'], *report['faults']]:
            if row['clip'] and row['proposal']:
                proposal = infer_video_goal(root/row['clip'], calibration=c)
                for key in ('status', 'reason', 'goal_xy_m', 'video_sha256', 'frame_count'):
                    if proposal[key] != row['proposal'][key]: raise ValueError('Reprocessed pixels differ from retained proposal')
    return dict(status='verified', sealed_files=len(seal['files']), sources=len(cases),
        controller_attempts=len(report['attempts']), fault_attempts=len(report['faults']),
        videos_reprocessed=reprocess_video, native_dynamics_recomputed=False,
        variants=report['variants'])


def _html(output, report):
    rows = ''.join(f'<tr><td>{escape(v.replace("_", " "))}</td><td>{s["successes"]}/{s["scheduled"]}</td><td>{s["executed"]}</td></tr>' for v, s in report['variants'].items())
    sources = ''.join(f'<article><h3>{escape(s["id"])}</h3>' +
        (f'<video controls preload="none" src="{escape(s["clip"], quote=True)}"></video>' if s['clip'] else '<p>Source generation failed.</p>') +
        f'<p>Source native success: {s["source_score"]["success"]}. Proposed XY error versus final source state: {s.get("proposal_error_m")}</p></article>' for s in report['sources'])
    limits = ''.join(f'<li>{escape(x)}</li>' for x in LIMITS)
    content = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DVIDIA · Video goal bridge</title>
<style>body{{max-width:860px;margin:48px auto;padding:0 24px;background:#faf9f5;color:#252c2b;font:18px/1.65 Georgia,serif}}h1,h2,h3,th{{font-family:system-ui,sans-serif}}h1{{line-height:1.15}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #ddd}}video{{width:min(100%,640px);background:#ddd}}article{{margin:32px 0}}a{{color:#315b4d}}</style>
<p>DVIDIA research · synthetic spatial grounding pilot</p><h1>From a demonstrated destination to a PiPER task</h1>
<p>The decoder reads encoded RGB pixels and a declared camera calibration. An existing authored PiPER controller attempts the resulting destination. The scorer uses the same intended reference for every control.</p>
<p><strong>No motor policy was trained.</strong> The input videos show a green rigid block in native simulation.</p>
<table><thead><tr><th>Variant</th><th>Reference successes / scheduled</th><th>Executed</th></tr></thead><tbody>{rows}</tbody></table>
<p>Missing or failed attempts remain in scheduled denominators. Fault clips and full traces are listed in <a href="report.json">report.json</a>; <a href="protocol.json">frozen protocol</a> and <a href="integrity.json">file hashes</a> are retained.</p>
<h2>Source clips</h2>{sources}<h2>Interpretation</h2><ul>{limits}</ul></html>'''
    (output/'report.html').write_text(content)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    benchmark = commands.add_parser('run')
    benchmark.add_argument('--assets', required=True, type=Path)
    benchmark.add_argument('--output', required=True, type=Path)
    benchmark.add_argument('--seed', type=int, default=2026102507)
    benchmark.add_argument('--count', type=int, default=4)
    inspect = commands.add_parser('inspect')
    inspect.add_argument('run', type=Path)
    inspect.add_argument('--reprocess-video', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'inspect': print(json.dumps(inspect_run(args.run, reprocess_video=args.reprocess_video), indent=2))
    else:
        result = run(args.assets, args.output, seed=args.seed, count=args.count)
        print(json.dumps(dict(variants=result['variants'], fault_summary=result['fault_summary'], wall_seconds=result['wall_seconds']), indent=2))


if __name__ == '__main__': main()

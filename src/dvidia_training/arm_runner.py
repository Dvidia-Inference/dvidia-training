"""Execute an installed, authored placement skill in an independently grounded scene."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import time

import mujoco
import numpy as np
from .arm_env import ArmConfig, ArmEnv
from .arm_policy import ArmPickPlacePolicy
from .cli import NetworkGuard, parse_seeds
from .installer import fetch_skill_url, install_source, read_installation, runtime_contract
from .skillspace import (ADAPTER_ID, ARM_PROFILE, PROCEDURE, WORKSPACE_MIN,
                        WORKSPACE_MAX, inspect_skillspace, load_skillspace)

DEFAULT_SCENE = {'object_position': [.42, -.04, .31], 'target_position': [.54, .10, .31], 'seed': 0,
                 'object_size': [.04, .04, .04], 'object_mass': .04, 'object_friction': .8,
                 'jaw_force_limit': 15., 'joint_torque_limit': 40.}
LIMITATIONS = [
    'Authored simulation controller; the source Skillspace envelope contains no learned robot policy.',
    'One original arm profile, privileged simulator observations, rigid box proxy for placement skills.',
    'Native jaw/object/table contacts; arm links and palm are collision-excluded.',
    'No camera perception, physical arm calibration, selfcollision, cup/handle, cable tying or broad skill qualification.',
    'Installation may fetch one public JSON envelope; installed execution and replay can run offline.',
]


def validate_scene(scene: dict | None, *, defaults=None) -> dict:
    if scene is None:
        scene = dict(DEFAULT_SCENE)
    if type(scene) is not dict or set(scene) - set(DEFAULT_SCENE):
        raise ValueError('Unknown scene field. Use placement, box dimensions/material, actuator limits and seed.')
    merged = {**DEFAULT_SCENE, **(defaults or {}), **scene}
    seed = merged['seed']
    if type(seed) is not int or not 0 <= seed <= 999999:
        raise ValueError('Environment seed must be an integer in [0, 999999].')
    for key in ('object_position', 'target_position', 'object_size'):
        point = merged[key]
        if type(point) is not list or len(point) != 3 or any(type(x) not in (int, float) or not math.isfinite(x) for x in point):
            raise ValueError('Scene positions and full box dimensions must contain three finite numbers.')
    for key, low, high in (('object_mass', .02, .08), ('object_friction', .4, 1.5),
                           ('jaw_force_limit', .01, 15.), ('joint_torque_limit', .1, 40.)):
        value = merged[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{key} must lie in [{low}, {high}].')
    scene_config(merged)
    return merged


def scene_config(scene, **kwargs):
    """A validated scene uses full box dimensions, kg, metres, N and N m."""
    return ArmConfig(object_position=scene['object_position'], target_position=scene['target_position'],
                     object_half_size=tuple(x/2 for x in scene['object_size']),
                     object_mass=scene['object_mass'], object_friction=scene['object_friction'],
                     jaw_force_limit=scene['jaw_force_limit'], joint_torque_limit=scene['joint_torque_limit'],
                     scene_jitter=0, **kwargs)


def authored_grounding(document, scene=None) -> dict:
    """Visible, local controller binding; never inferred from demonstration count."""
    scene = validate_scene(scene)
    metadata = inspect_skillspace(document)
    return {'schema_version': 1, 'task_type': metadata['skill_id'],
            'provenance': 'authored_simulation_adapter', 'adapter_id': ADAPTER_ID,
            'arm_profile': ARM_PROFILE, 'frame': 'world', 'units': 'm',
            'language_aliases': [metadata['task']],
            'workspace': {'min': list(WORKSPACE_MIN), 'max': list(WORKSPACE_MAX)},
            'table_height': .29,
            'object': {'kind': 'box', 'position': scene['object_position'], 'size': scene['object_size'],
                       'mass': scene['object_mass'], 'friction': scene['object_friction']},
            'target': {'position': scene['target_position'], 'position_tolerance': .025, 'dwell_seconds': .25},
            'procedure': list(PROCEDURE)}


def config_for(skill, scene=None):
    scene = validate_scene({} if scene is None else scene, defaults={'object_size': list(skill.object_size),
                            'object_mass': skill.object_mass, 'object_friction': skill.object_friction})
    return scene_config(scene, table_height=skill.table_height,
                        position_tolerance=skill.position_tolerance, dwell_seconds=skill.dwell_seconds)


def rollout(config, seed, policy_name='placement', *, action_tape=None, movement_head=None):
    if policy_name not in ('placement', 'open_jaw', 'idle', 'replay_open_jaw'):
        raise ValueError('Unknown placement controller or control check.')
    started = time.perf_counter()
    env = ArmEnv(config)
    obs, reset_info = env.reset(seed=seed)
    setup = time.perf_counter() - started
    if movement_head is not None:
        from .arm_distill import DistilledPlacementPolicy
        policy = DistilledPlacementPolicy(env, movement_head)
    else:
        policy = ArmPickPlacePolicy(env)
    frames = [{**obs, 'time': 0., 'stage': 'observe'}]
    initial_joints = list(obs['joint_position'])
    latencies, rewards, tape = [], [], []
    done = False
    info = reset_info
    while not done:
        began = time.perf_counter()
        if policy_name == 'idle':
            action = {'joint_targets': initial_joints, 'gripper_width': .08}
            stage = 'idle'
        elif policy_name == 'replay_open_jaw':
            if not action_tape:
                raise ValueError('Recorded-arm open-jaw check requires a placement action tape.')
            index = min(len(tape), len(action_tape)-1)
            action = {**action_tape[index]['action'], 'gripper_width': .08}
            stage = action_tape[index]['stage']
        else:
            action = policy(obs)
            stage = policy.diagnostics().get('action_phase', policy.stage) if hasattr(policy, 'diagnostics') else policy.stage
            if policy_name == 'open_jaw':
                action = {**action, 'gripper_width': .08}
        obs, reward, terminated, truncated, info = env.step(action)
        latencies.append(time.perf_counter() - began)
        rewards.append(reward)
        tape.append({'stage': stage, 'action': action})
        if info['valid'] and len(obs['object_position']) == 3:
            frames.append({**obs, 'time': info['simulation_time'], 'stage': stage,
                           'pad_normal_forces': info['pad_normal_forces'],
                           'target_distance': info['target_distance'], 'dwell_elapsed': info['dwell_elapsed']})
        done = terminated or truncated
    elapsed = time.perf_counter()-started
    label = 'distilled_placement' if movement_head is not None and policy_name == 'placement' else policy_name
    return {'seed': seed, 'policy': label, 'success': info['success'], 'reason': info['reason'],
            'final_distance': info['target_distance'], 'simulated_seconds': info['simulation_time'],
            'episode_wall_seconds': elapsed, 'setup_seconds': setup,
            'control_latency_p50_seconds': float(np.median(latencies)),
            'control_latency_p95_seconds': float(np.percentile(latencies, 95)),
            'simulated_seconds_per_wall_second': info['simulation_time']/elapsed,
            'control_steps': len(tape), 'reward': float(sum(rewards)),
            'reset_info': reset_info, 'final_info': info, 'actions': tape, 'trace': frames,
            'controller_failure': getattr(policy, 'failure_reason', None),
            'controller_diagnostics': policy.diagnostics() if hasattr(policy, 'diagnostics') else {}}


def run_skill(source: bytes, grounding: dict | None, scene=None, *, seeds=None, policy='placement', offline=True,
              movement_head=None):
    skill = load_skillspace(source, grounding)
    config = config_for(skill, scene)
    scene = validate_scene({} if scene is None else scene, defaults={'object_size': list(skill.object_size),
                            'object_mass': skill.object_mass, 'object_friction': skill.object_friction})
    seeds = [scene['seed']] if seeds is None else seeds
    if not seeds or len(seeds)>100 or len(set(seeds))!=len(seeds) or any(type(s) is not int or not 0<=s<=999999 for s in seeds):
        raise ValueError('Use 1–100 distinct nonnegative environment seeds.')
    with NetworkGuard(offline) as guard:
        episodes = [rollout(config, seed, policy, movement_head=movement_head) for seed in seeds]
    result = payload_for(skill, config, episodes, guard.record())
    if movement_head is not None:
        result['policy_origin'] = 'learned joint-movement head with authored task phases, grasp and recovery'
        result['movement_head_sha256'] = sha256(json.dumps(movement_head, sort_keys=True,
                       separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        result['limitations'] = [
            'Hybrid controller: joint-movement mapping distilled from authored simulator demonstrations; phases, grasp and recovery remain authored.',
            *LIMITATIONS[1:]]
    return result


def payload_for(skill, config, episodes, guard):
    return {'schema_version': 1, 'task': 'ArmPickPlace-v0', 'scope': 'simulation-only',
            'runtime': runtime_contract(),
            'skill': skill.to_dict(), 'environment_config': asdict(config),
            'scene_binding': 'Object and target positions belong to the environment; the installed controller is reused.',
            'episodes': episodes, 'hardware': {'system': platform.system(), 'machine': platform.machine(),
                    'python': platform.python_version(), 'mujoco': mujoco.__version__, 'numpy': np.__version__},
            'network_guard': guard, 'limitations': LIMITATIONS,
            'timing_scope': 'Episode wall clock includes model compilation, reset/settling, IK/controller, native physics, observations and traces; excludes installation, web UI, process startup and output I/O. No GPU benchmark.',
            'summary': {'successes': sum(e['success'] for e in episodes), 'episodes': len(episodes),
                        'simulated_seconds': sum(e['simulated_seconds'] for e in episodes),
                        'episode_wall_seconds': sum(e['episode_wall_seconds'] for e in episodes)}}


def save_replay(payload, path):
    text = Path(__file__).with_name('studio.html').read_text(encoding='utf8')
    # A visual replay samples saved observations at 100 ms; the full run JSON
    # retains every control frame and commanded action for verification.
    view = {**payload, 'replay_sampling':'100 ms recorded observation samples; full control observations/actions are in run.json'}
    view['episodes'] = []
    for episode in payload['episodes']:
        frames = episode['trace']
        sampled = frames[::5]
        if frames and sampled[-1] is not frames[-1]:
            sampled.append(frames[-1])
        view['episodes'].append({key:value for key,value in episode.items() if key not in ('trace','actions')} | {'trace':sampled})
    embedded = json.dumps(view, allow_nan=False, separators=(',', ':')).replace('<', '\\u003c')
    text = text.replace('<script id="embedded-run" type="application/json">null</script>',
                        '<script id="embedded-run" type="application/json">'+embedded+'</script>')
    Path(path).write_text(text, encoding='utf8')


def write_run(directory, payload, *, source=None, grounding=None):
    directory = Path(directory)
    if source is not None:
        if type(source) is not bytes or payload['skill']['source_sha256'] != sha256(source).hexdigest():
            raise ValueError('Recorded source must match the exact execution input.')
        if load_skillspace(source, grounding).to_dict() != payload['skill']:
            raise ValueError('Recorded grounding must match the exact execution contract.')
    directory.mkdir(parents=True, exist_ok=True)
    if source is not None:
        (directory/'source.json').write_bytes(source)
        (directory/'grounding.json').write_text(json.dumps(grounding, allow_nan=False)+'\n', encoding='utf8')
    (directory/'run.json').write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False)+'\n', encoding='utf8')
    save_replay(payload, directory/'replay.html')
    print(json.dumps({'output': str(directory), **payload['summary']}))


def main():
    parser = argparse.ArgumentParser(description='Install a DVIDIA placement source and execute its authored simulation adapter.')
    sub = parser.add_subparsers(dest='command', required=True)
    install = sub.add_parser('install')
    source = install.add_mutually_exclusive_group(required=True)
    source.add_argument('--url'); source.add_argument('--file', type=Path)
    install.add_argument('--directory', type=Path, default=Path('runs/studio/installed'))
    run = sub.add_parser('run')
    run.add_argument('installation_id')
    run.add_argument('--directory', type=Path, default=Path('runs/studio/installed'))
    run.add_argument('--scene', type=Path)
    run.add_argument('--seeds', type=parse_seeds)
    run.add_argument('--policy', choices=['placement','open_jaw','idle'], default='placement')
    run.add_argument('--output', type=Path, default=Path('runs/arm-placement'))
    args = parser.parse_args()
    if args.command == 'install':
        data, url = fetch_skill_url(args.url) if args.url else (args.file.read_bytes(), None)
        record = install_source(data, authored_grounding(data), args.directory, source_url=url)
        print(json.dumps({key: record[key] for key in ('installation_id','status','scope','skill')}))
    else:
        record = read_installation(args.directory, args.installation_id)
        data = (args.directory/args.installation_id/'source.json').read_bytes()
        grounding = None if record['grounding_mode']=='inline' else record['grounding']
        scene = json.loads(args.scene.read_text()) if args.scene else None
        result = run_skill(data, grounding, scene, seeds=args.seeds, policy=args.policy)
        result['installation_id'] = args.installation_id
        result['source_url'] = record['source_url']
        write_run(args.output, result, source=data, grounding=grounding)


if __name__ == '__main__':
    main()

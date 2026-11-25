"""Replay matched native PiPER grasps and measure a hold versus a 20 mm test lift."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform

import mujoco
import numpy as np

from dvidia_training import piper_task, reference_sim, robot_profiles
from dvidia_training.piper_feedback_env import FeedbackPiperTaskEnv


def digest(document):
    return sha256(json.dumps(document, allow_nan=False, sort_keys=True).encode()).hexdigest()


def advance(env, command):
    observation, info = env.step(command)
    if observation is None or not info['valid'] or env.done:
        raise RuntimeError(f"Step {env.steps} ended with {info['reason']}")
    return observation


def acquire(env, observation, jaw):
    """Use the privileged authored policy to create the prefix only."""
    teacher = piper_task.PiperTaskPolicy(env)
    commands = []
    closing_steps = 0
    while closing_steps < 35:
        command = teacher(observation)
        if teacher.issued_phase == 'close':
            command['gripper_position_m'] = jaw
            closing_steps += 1
        commands.append(command)
        observation = advance(env, command)
    return commands


def branch_identity(env, commands):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(env.model, spec))
    mujoco.mj_getState(env.model, env.data, state, spec)
    history = {name: sha256(values.tobytes()).hexdigest()
               for name, values in env.native_arrays().items()}
    return {
        'model_sha256': reference_sim.model_digest(env.model),
        'integration_state_sha256': sha256(state.tobytes()).hexdigest(),
        'issued_commands_sha256': digest(commands),
        'native_prefix_history_sha256': digest(history),
        'task_status_sha256': digest(env.info()),
        'ramped_joint_targets_rad': env.command_targets.tolist(),
        'ramped_single_finger_target_m': env.command_gripper,
    }


def endpoint(observation):
    return {name: observation[name] for name in (
        'tcp_position_m', 'object_position_m', 'grasp_contact_counts',
        'pad_normal_forces_n', 'table_object_contact_count')}


def observe_action(env, height, jaw):
    initial = env.observation()
    target = np.array(initial['tcp_position_m'])
    first_tick = env.steps * round(env.config.control_dt / env.config.timestep)
    final = initial
    for step in range(1, 11):
        target[2] = initial['tcp_position_m'][2] + height * step / 10
        command = env.pose_action(target.tolist(), initial['tcp_quaternion_wxyz'], jaw)
        final = advance(env, command)
    data = env.native_arrays()
    counts = data['grasp_contact_counts'][first_tick:]
    if len(counts) != 200:
        raise ValueError('The 200 ms probe must include 200 native ticks')
    return {
        'tool_z_target_delta_m': height, 'single_finger_target_m': jaw,
        'final': endpoint(final),
        'tool_displacement_m': (np.array(final['tcp_position_m']) - initial['tcp_position_m']).tolist(),
        'block_displacement_m': (np.array(final['object_position_m']) - initial['object_position_m']).tolist(),
        'native_ticks': len(counts),
        'bilateral_ticks': int(np.all(counts > 0, axis=1).sum()),
        'table_supported_ticks': int((data['table_object_contact_count'][first_tick:] > 0).sum()),
        'critical_contact_ticks': int((data['unexpected_contact_count'][first_tick:] > 0).sum()),
    }


def compare_branches(assets, jaw):
    hold = FeedbackPiperTaskEnv(assets)
    initial, _ = hold.reset(seed=0)
    commands = acquire(hold, initial, jaw)
    lift = FeedbackPiperTaskEnv(assets)
    lift.reset(seed=0)
    for command in commands:
        advance(lift, command)
    first = branch_identity(hold, commands)
    second = branch_identity(lift, commands)
    if first != second:
        changed = [key for key in first if first[key] != second[key]]
        raise ValueError(f'Prefix replay differs at {changed}')

    decision = hold.observation()
    inputs = {name: decision[name] for name in (
        't_seconds', 'joint_position_rad', 'joint_velocity_rad_s', 'gripper_joint_position_m')}
    inputs['previous_issued_command'] = commands[-1]
    return {
        'prefix_control_steps': len(commands), 'matched_prefix_identity': first,
        'predecision_encoder_observation': inputs,
        'offline_native_measurements': {
            'initial': endpoint(decision),
            'hold': observe_action(hold, 0., jaw),
            'lift': observe_action(lift, .020, jaw),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', required=True, type=Path, help='Verified local PiPER asset root')
    parser.add_argument('--output', required=True, type=Path, help='New JSON receipt path')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output path already exists')
    receipt = {
        'schema': 'dvidia.piper-action-branches/v1',
        'scope': 'Deterministic two-condition simulator mechanics probe using a privileged scripted acquisition and simulator pose controller.',
        'acquisition': 'Replay the identical issued commands through 35 close steps. Discard the acquisition teacher at the branch. Compare integration state, servo targets, scorer status and the complete native prefix telemetry.',
        'config': asdict(piper_task.PiperTaskConfig()),
        'provenance': robot_profiles.verify_assets('piper', args.assets),
        'python': platform.python_version(), 'mujoco': mujoco.__version__,
        'script_sha256': sha256(Path(__file__).read_bytes()).hexdigest(), 'cases': [],
    }
    for name, jaw in [('closed', 0.), ('underclosed', .032)]:
        case = {'name': name, 'jaw_target_m': jaw}
        try:
            case.update(compare_branches(args.assets, jaw))
            case['status'] = 'complete'
        except (RuntimeError, ValueError) as error:
            case.update(status='failed', reason=str(error))
        receipt['cases'].append(case)
    with args.output.open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return int(any(case['status'] != 'complete' for case in receipt['cases']))


if __name__ == '__main__':
    raise SystemExit(main())

"""Generate an auditable 30-second native trace for the browser controller parity test."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import time

import mujoco
import numpy as np

from dvidia_training import workstation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.scene / 'workstation.json').read_text())
    model = mujoco.MjModel.from_xml_path(str(args.scene / 'scene.xml'))
    data = mujoco.MjData(model)
    data.qpos[:] = manifest['home_qpos']
    data.ctrl[:] = manifest['home_ctrl']
    mujoco.mj_forward(model, data)
    dt = manifest['timestep']
    stride = round(manifest['control_dt'] / dt)
    total_steps = round(30 / dt)
    commands = [
        {'step': 0, 'type': 'target', 'position': [.38, 0., .25]},
        {'step': 1500, 'type': 'target', 'position': [.32, -.12, .20]},
        {'step': 3003, 'type': 'gripper', 'jaw': .015},
        {'step': 4500, 'type': 'gripper', 'jaw': .035},
        {'step': 5000, 'type': 'target', 'position': [.40, .10, .28]},
        {'step': 7000, 'type': 'target', 'position': [.30, .04, .23]},
        {'step': 9000, 'type': 'target', 'position': manifest['initial_target_xyz']},
        {'step': 10000, 'type': 'gripper', 'jaw': 0.},
        {'step': 11000, 'type': 'push', 'body': manifest['shoe_body_names'][0],
         'impulse': [0., .12, 0.]},
        {'step': 12000, 'type': 'gripper', 'jaw': .035},
        {'step': 13000, 'type': 'target', 'position': [.35, -.03, .24]},
    ]
    schedule = {}
    for command in commands:
        schedule.setdefault(command['step'], []).append(command)
    target = manifest['initial_target_xyz']
    jaw = manifest['home_ctrl'][6]
    shoe = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, manifest['shoe_body_names'][0])
    tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, manifest['tcp_site_name'])
    trace, step_ms, control_ms = [], [], []
    started = time.perf_counter()
    error = None
    completed = 0
    for index in range(total_steps):
        force = None
        for command in schedule.get(index, []):
            if command['type'] == 'target': target = command['position']
            elif command['type'] == 'gripper': jaw = command['jaw']
            else: force = np.asarray(command['impulse']) / dt
        tick = time.perf_counter()
        if index % stride == 0:
            control_started = time.perf_counter()
            desired = workstation.controller_action(model, data, target, jaw)
            control_ms.append((time.perf_counter() - control_started) * 1000)
        data.ctrl[:6] += np.clip(desired[:6] - data.ctrl[:6], -manifest['controller']['joint_rate_rad_s'] * dt,
                                 manifest['controller']['joint_rate_rad_s'] * dt)
        data.ctrl[6] += float(np.clip(desired[6] - data.ctrl[6], -manifest['controller']['gripper_rate_m_s'] * dt,
                                      manifest['controller']['gripper_rate_m_s'] * dt))
        data.xfrc_applied[:] = 0
        if force is not None: data.xfrc_applied[shoe, :3] = force
        mujoco.mj_step(model, data)
        data.xfrc_applied[:] = 0
        mujoco.mj_forward(model, data)
        completed += 1
        step_ms.append((time.perf_counter() - tick) * 1000)
        if not all(np.isfinite(value).all() for value in (data.qpos, data.qvel, data.qacc)):
            error = 'nonfinite_native_state'; break
        if any(warning.number for warning in data.warning):
            error = 'native_solver_warning'; break
        if (index + 1) % stride == 0:
            forces = []
            for contact in range(data.ncon):
                force_vector = np.empty(6)
                mujoco.mj_contactForce(model, data, contact, force_vector)
                forces.append(max(0., float(force_vector[0])))
            trace.append({'completed_steps': index + 1, 'qpos': data.qpos.tolist(), 'qvel': data.qvel.tolist(),
                          'ctrl': data.ctrl.tolist(), 'tcp_position': data.site_xpos[tcp].tolist(),
                          'shoe_position': data.xpos[shoe].tolist(), 'contacts': int(data.ncon),
                          'contact_normal_forces_n': forces, 'active_contacts': sum(force > .001 for force in forces),
                          'tracking_error_m': float(np.linalg.norm(data.site_xpos[tcp] - target))})
    wall_seconds = time.perf_counter() - started
    result = {'schema': 'dvidia.workstation-native-parity/v1', 'engine': mujoco.__version__,
              'platform': platform.platform(), 'native_scene_sha256': sha256((args.scene / 'scene.xml').read_bytes()).hexdigest(),
              'manifest_sha256': sha256((args.scene / 'workstation.json').read_bytes()).hexdigest(),
              'physics_steps': total_steps, 'completed_steps': completed, 'timestep': dt, 'control_stride': stride,
              'commands': commands, 'trace': trace, 'valid': error is None, 'error': error,
              'warnings': [int(warning.number) for warning in data.warning],
              'wall_seconds': wall_seconds, 'real_time_factor': completed * dt / wall_seconds,
              'step_ms_p50': float(np.percentile(step_ms, 50)), 'step_ms_p95': float(np.percentile(step_ms, 95)),
              'control_ms_p95': float(np.percentile(control_ms, 95))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('commands', 'trace')}, indent=2))
    if error: raise SystemExit(1)


if __name__ == '__main__':
    main()

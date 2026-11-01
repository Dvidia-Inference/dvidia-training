"""Installation of data-only camera/action candidates in the exact PiPER sim.

Installation is separate from execution and never awards task qualification.
"""
from hashlib import sha256
import json
from pathlib import Path

from . import arm_interfaces
from . import camera_action_model as model_api
from .reference_replay import _read, _decode

SCHEMA = 'dvidia.camera-action-installation/v1'
RUNNER = 'piper-camera-action-v1'
INTERFACE = 'mujoco-position-v1'
FILES = ('camera_action_skill.py', 'camera_action_capture.py', 'camera_action_model.py',
         'piper_action_capture.py', 'piper_rgbd_tracker.py', 'action_demonstration.py',
         'piper_feedback_env.py', 'piper_feedback_score.py', 'piper_imitation_score.py',
         'piper_imitation.py', 'piper_task.py', 'reference_task.py', 'reference_sim.py',
         'reference_replay.py', 'robot_profiles.py', 'robot_profiles.lock.json',
         'arm_interfaces.py', 'episode_tape.py', 'embodiment_calibration.py',
         'piper_learned_skill.py', 'env.py')
MAX_MODEL_BYTES = 8 * 1024 * 1024
TRAINED_SCOPE = dict(mass_kg=.04, object='known rigid 40 mm block',
                    start_x_m=[.287, .311], start_y_m=[-.042, -.018],
                    known_goals_xy_m=[[.35, .08], [.40, -.08]],
                    camera_pixels=[320, 240], camera_hz=25, control_hz=50,
                    initial_visual_anchor_required=True)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def runtime():
    return {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in FILES}


_LOADED = runtime()


def checked_runtime():
    current = runtime()
    if current != _LOADED:
        raise ValueError('Camera-action runtime changed while loaded. Restart before installation or execution.')
    return current


def check(model, robot, *, interface=INTERFACE):
    if robot != 'piper' or interface != INTERFACE:
        raise ValueError('Camera-action models require the exact PiPER local simulation interface.')
    profile = arm_interfaces.contract(robot)
    raw = _read(model, MAX_MODEL_BYTES)
    value = model_api.validate_model(raw)
    return dict(compatible=True, runner=RUNNER, robot=robot, interface_id=interface,
        model_sha256=sha256(raw).hexdigest(), model_schema=value['schema'],
        feature_contract=value['feature_contract'], blind=value['blind'],
        profile_contract_sha256=profile['contract_sha256'], control_period_s=.02,
        scope='simulation-only', observation_origin=value['observation_origin'],
        physical_robot_ready=False, task_qualified_by_installation=False)


def _record(compatibility, binding):
    record = dict(schema=SCHEMA, status='installed_simulation_candidate',
        compatibility=compatibility, native_binding=binding, runtime_sha256=checked_runtime(),
        model_file='model.json', execution_scope=TRAINED_SCOPE,
        task_gate='Installation does not grant task success. Every run retains its independent native score.',
        physical_robot_ready=False)
    record['installation_sha256'] = sha256(canonical(record)).hexdigest()
    return record


def install(model, robot, assets, output, *, interface=INTERFACE):
    compatibility = check(model, robot, interface=interface)
    raw = _read(model, MAX_MODEL_BYTES)
    if sha256(raw).hexdigest() != compatibility['model_sha256']:
        raise ValueError('Model changed during installation.')
    _, _, binding = arm_interfaces.build(robot, assets, timestep=.001)
    record = _record(compatibility, binding)
    root = Path(output)
    if root.exists() or root.is_symlink():
        raise ValueError('Use a fresh installation directory.')
    root.mkdir(parents=True, exist_ok=False)
    for name, body in [('model.json', raw), ('installation.json', canonical(record)+b'\n')]:
        with (root/name).open('xb') as stream:
            stream.write(body)
    return record


def inspect(directory, assets):
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Use a regular installation directory.')
    stored = _decode(_read(root/'installation.json', 256*1024))
    if type(stored) is not dict or stored.get('schema') != SCHEMA:
        raise ValueError('Unknown camera-action installation.')
    declared = stored.get('compatibility', {})
    compatibility = check(root/'model.json', declared.get('robot'), interface=declared.get('interface_id'))
    _, _, binding = arm_interfaces.build('piper', assets, timestep=.001)
    expected = _record(compatibility, binding)
    if canonical(expected) != canonical(stored):
        raise ValueError('Installed camera model, assets, runtime or limits changed. Install again.')
    return expected


def run(directory, assets, output, *, start_x=.30, start_y=-.03, goal=0):
    from .camera_action_capture import run_rollout
    from .piper_learned_skill import _point
    record = inspect(directory, assets)
    x = _point(start_x, 'start_x', *TRAINED_SCOPE['start_x_m'])
    y = _point(start_y, 'start_y', *TRAINED_SCOPE['start_y_m'])
    if type(goal) is not int or goal not in (0, 1):
        raise ValueError('Select known goal 0 or 1.')
    raw = _read(Path(directory)/'model.json', MAX_MODEL_BYTES)
    if sha256(raw).hexdigest() != record['compatibility']['model_sha256']:
        raise ValueError('Model changed before execution.')
    policy = model_api.ModelPolicy(model_api.validate_model(raw))
    case = dict(id=f'installed-camera-goal{goal}', group_id='installed-camera-example', seed=311499,
                object_position_m=[x, y, .082],
                goal_position_m=[*TRAINED_SCOPE['known_goals_xy_m'][goal], .08])
    report = run_rollout(assets, output, case=case, mode='student', policy=policy)
    result = dict(success=report['strict_success'], reason=report['reason'], status=report['status'],
                  model_sha256=record['compatibility']['model_sha256'],
                  installation_sha256=record['installation_sha256'],
                  physical_robot_ready=False, report=report)
    # The raw rollout stays sealed. Its installation link is adjacent, not an
    # unregistered file added to its evidence inventory.
    link = Path(output).with_name(Path(output).name+'-installation.json')
    with link.open('xb') as stream:
        stream.write(canonical({key: value for key, value in result.items() if key != 'report'})+b'\n')
    return result

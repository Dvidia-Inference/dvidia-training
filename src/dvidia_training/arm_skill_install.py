"""Bind a supported data-only policy to an exact local simulation adapter.

Installation is separate from execution and never authorizes physical motion.
The first policy runner is the existing PiPER block-placement research policy;
other arms can use the shared mechanics benchmark but cannot install this policy.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from . import arm_interfaces
from .piper_imitation import MAX_MODEL_BYTES, validate_model
from .reference_replay import _read, _decode

SCHEMA = 'dvidia.arm-skill-installation/v1'
INTERFACE = 'mujoco-position-v1'
RUNNER = 'piper-block-placement-v1'
RUNTIME_FILES = (
    'arm_skill_install.py', 'arm_interfaces.py', 'piper_learned_skill.py',
    'piper_imitation.py', 'piper_imitation_score.py', 'piper_imitation_benchmark.py',
    'piper_motor_primitive.py', 'piper_task.py', 'reference_task.py',
    'reference_sim.py', 'reference_replay.py', 'robot_profiles.py',
    'robot_profiles.lock.json', 'env.py',
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _runtime():
    return {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in RUNTIME_FILES}


_LOADED_RUNTIME = _runtime()


def _checked_runtime():
    current = _runtime()
    if current != _LOADED_RUNTIME:
        raise ValueError('Runtime changed while loaded. Restart before installing or running.')
    return current


def check_model(model_path, robot, *, interface=INTERFACE):
    """Check policy semantics, not merely joint count. Does not compile or run."""
    profile = arm_interfaces.contract(robot)
    if interface != INTERFACE:
        raise ValueError('Only the local mujoco-position-v1 interface is implemented.')
    if robot != 'piper':
        raise ValueError('This learned policy requires PiPER. Other arms need their own trained '
                         'policy and task benchmark; matching joint counts do not transfer it.')
    raw = _read(model_path, MAX_MODEL_BYTES)
    model = validate_model(raw)
    return {
        'compatible': True, 'runner': RUNNER, 'robot': robot, 'interface_id': interface,
        'model_sha256': sha256(raw).hexdigest(), 'model_schema': model['schema'],
        'feature_contract': model['feature_contract'],
        'profile_contract_sha256': profile['contract_sha256'],
        'control_period_s': .02, 'scope': 'simulation-only',
        'observation_origin': 'privileged_native_simulator_state',
        'physical_robot_ready': False, 'task_qualified_by_installation': False,
    }


def _record(check, binding):
    from .piper_learned_skill import TRAINED_SCOPE
    record = {
        'schema': SCHEMA, 'status': 'installed_simulation_candidate',
        'compatibility': check, 'native_binding': binding,
        'runtime_sha256': _checked_runtime(), 'model_file': 'model.json',
        'trained_scope': TRAINED_SCOPE,
        'task_gate': 'Each run uses the existing bounded block scene and independent scorer. '
                     'Installation alone does not establish task success or physical readiness.',
    }
    record['installation_sha256'] = sha256(_canonical(record)).hexdigest()
    return record


def install_model(model, robot, assets, output, *, interface=INTERFACE):
    """Verify model/assets, then reserve a fresh installation without stepping."""
    check = check_model(model, robot, interface=interface)
    raw = _read(model, MAX_MODEL_BYTES)
    if sha256(raw).hexdigest() != check['model_sha256']:
        raise ValueError('Model changed during compatibility check.')
    _, _, binding = arm_interfaces.build(robot, assets, timestep=.001)
    record = _record(check, binding)
    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise ValueError('Choose a fresh installation directory; existing installations are preserved.')
    destination.mkdir(parents=True, exist_ok=False)
    with (destination/'model.json').open('xb') as stream:
        stream.write(raw)
    with (destination/'installation.json').open('xb') as stream:
        stream.write(_canonical(record)+b'\n')
    return record


def inspect_installation(directory, assets):
    """Reconstruct every installed claim against current model, assets and runtime."""
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Installation must be a regular directory.')
    stored = _decode(_read(root/'installation.json', 256*1024))
    if type(stored) is not dict or stored.get('schema') != SCHEMA:
        raise ValueError('Unknown arm skill installation.')
    compatibility = stored.get('compatibility')
    if type(compatibility) is not dict:
        raise ValueError('Missing policy compatibility contract.')
    check = check_model(root/'model.json', compatibility.get('robot'),
                        interface=compatibility.get('interface_id'))
    _, _, binding = arm_interfaces.build(check['robot'], assets, timestep=.001)
    expected = _record(check, binding)
    if _canonical(stored) != _canonical(expected):
        raise ValueError('Installed model, adapter, assets or runtime changed. Install again.')
    return expected


def run_installation(directory, assets, output, *, start_x=.30, start_y=-.03, goal=0):
    """Revalidate, then dispatch only to the built-in bounded simulation runner."""
    record = inspect_installation(directory, assets)
    from .piper_learned_skill import run_learned_skill
    result = run_learned_skill(Path(directory)/'model.json', assets, output,
                               start_x=start_x, start_y=start_y, goal=goal)
    link = {'schema': 'dvidia.arm-skill-run-binding/v1',
            'installation_sha256': record['installation_sha256'],
            'model_sha256': record['compatibility']['model_sha256'],
            'run_model_matches_installation': result['model_sha256'] == record['compatibility']['model_sha256'],
            'physical_robot_ready': False}
    with (Path(output)/'arm-installation-binding.json').open('xb') as stream:
        stream.write(_canonical(link)+b'\n')
    if not link['run_model_matches_installation']:
        raise ValueError('Model changed between installation inspection and run; discard this run.')
    return result

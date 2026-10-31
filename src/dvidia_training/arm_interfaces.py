"""Exact native MuJoCo command contracts for three pinned robot models.

Listing and describing an interface require only the standard library. Building
uses the existing byte-verified asset loaders and returns a runtime-bound
contract. Commands are native position-servo inputs, not Cartesian targets,
measured jaw gaps, calibrated hardware commands, or portable learned skills.

There is deliberately no hardware transport or command application here. The
caller owns stepping and may assign a validated vector to ``data.ctrl``. Models
returned by build must remain owned by that caller and must not be modified.
Validation checks command topology and servo settings on every call; it does
not rehash the complete mesh-bearing native model on every control boundary.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
import json
import math
from numbers import Real
from pathlib import Path
import weakref

from . import reference_sim as sim
from . import robot_profiles


SCHEMA = 'dvidia.arm-interface/v1'
INTERFACE_ID = 'mujoco-position-v1'
PROFILES = ('piper', 'so101', 'ur5e-2f85')
SCOPE = ('Pinned native MuJoCo position-servo interface; unloaded simulation only. '
         'No physical robot qualification, calibrated TCP, automatic skill retargeting, '
         'task policy, camera perception or hardware command transport.')
_BUILT = weakref.WeakKeyDictionary()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _digest(value):
    return sha256(_canonical(value)).hexdigest()


def _profile(profile):
    if type(profile) is not str or profile not in PROFILES:
        raise ValueError('Profile must be piper, so101 or ur5e-2f85')
    return profile


def _signed(value):
    result = deepcopy(value)
    result['contract_sha256'] = _digest(value)
    return result


@lru_cache(maxsize=3)
def _template(profile):
    """Private immutable-by-convention metadata, frozen at first description."""
    _profile(profile)
    if profile in ('piper', 'so101'):
        item = robot_profiles._profile(profile)
        lock_path = robot_profiles.LOCK_PATH
        lock = robot_profiles._lock()
        files = item['files']
        arm_axes = item['arm_axes']
        joints = item['arm_joint_names'] + item['gripper_joint_names']
        units = item['native_command_units']
        joint_units = item['native_joint_position_units']
        actions = item['actuator_names']
        limits = item['native_command_range']
        dimensions = dict(item['expected_dimensions'], njnt=len(joints), ntendon=0,
                          nkey=1 if profile == 'piper' else 0, nmocap=0)
        initial_arm = [0., 1.57, -1.3485, 0., 0., 0.] if profile == 'piper' else [0.] * arm_axes
        initial_qpos = initial_arm + [0.] * len(item['gripper_joint_names'])
        initial_ctrl = initial_arm + [0.]
        transmissions = [{'type': 'joint', 'target': name} for name in joints[:len(actions)]]
        license_claims = {item['folder']: item['simulation_asset_license']}
        name, tool, tool_scope = item['name'], item['tool_frame_body'], item['tool_frame_scope']
        gripper_command = item['gripper_command']
        force_units = item['native_actuator_force_units']
        hardware_id = 'piper-sdk-planned' if profile == 'piper' else 'lerobot-so101-planned'
        sdk_url = item['sdk_url']
    else:
        lock_path = Path(sim.__file__).with_name('reference_assets.lock.json')
        lock = sim.asset_lock()
        files = sim._files(hands=False)
        arm_axes = 6
        joints = list(sim.ARM_JOINTS) + [f'gripper/{side}_{part}_joint'
                  for side in ('right', 'left') for part in ('driver', 'coupler', 'spring_link', 'follower')]
        actions = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3',
                   'gripper/fingers_actuator']
        units, joint_units = ['rad'] * 6 + ['native_0_255'], ['rad'] * 14
        limits = [[-6.2831, 6.2831], [-6.2831, 6.2831], [-3.1415, 3.1415],
                  [-6.2831, 6.2831], [-6.2831, 6.2831], [-6.2831, 6.2831], [0., 255.]]
        dimensions = dict(nq=14, nv=14, nu=7, neq=3, njnt=14, ntendon=1, nkey=1, nmocap=0)
        initial_arm = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.]
        initial_qpos, initial_ctrl = initial_arm + [0.] * 8, initial_arm + [0.]
        transmissions = [{'type': 'joint', 'target': name} for name in joints[:6]]
        transmissions.append({'type': 'tendon', 'target': 'gripper/split'})
        license_claims = {key: lock['licenses'][key] for key in ('universal_robots_ur5e', 'robotiq_2f85')}
        name, tool = 'UR5e + Robotiq 2F-85', 'wrist_3_link'
        tool_scope = 'terminal arm link origin; not a calibrated fingertip TCP'
        gripper_command = ('native position-servo input [0,255] for the coupled Robotiq tendon; '
                           'not metres, radians, force or a measured jaw gap')
        force_units = ['Nm'] * 6 + ['native_tendon_generalized_force']
        hardware_id, sdk_url = 'ur-rtde-robotiq-planned', None
    source_names = ('arm_interfaces.py', 'robot_profiles.py', 'reference_sim.py', 'env.py')
    sources = {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in source_names}
    return {
        'schema': SCHEMA, 'id': profile, 'name': name, 'interface_id': INTERFACE_ID,
        'arm_axes': arm_axes, 'gripper_axes': 1,
        'action_names': list(actions), 'action_units': list(units), 'action_limits_native': limits,
        'action_force_units': force_units,
        'arm_action_indices': list(range(arm_axes)), 'gripper_action_indices': [arm_axes],
        'actuator_transmissions': transmissions,
        'joint_names': joints, 'joint_position_units': joint_units,
        'joint_qpos_indices': list(range(len(joints))), 'joint_dof_indices': list(range(len(joints))),
        'arm_joint_names': joints[:arm_axes], 'gripper_joint_names': joints[arm_axes:],
        'tool_frame_body': tool, 'tool_frame_scope': tool_scope,
        'gripper_command': gripper_command, 'expected_dimensions': dimensions,
        'initial_state': {'method': 'native_keyframe_0' if dimensions['nkey'] else 'native_zero_joint_state',
                          'qpos_native': initial_qpos, 'ctrl_native': initial_ctrl,
                          'qvel_native_per_second': [0.] * dimensions['nv'],
                          'time_seconds': 0., 'hidden_settling_steps': 0},
        'asset_lock': {'repository': lock['repository'], 'commit': lock['commit'],
                       'lock_file': lock_path.name, 'lock_sha256': sha256(lock_path.read_bytes()).hexdigest(),
                       'files': deepcopy(files), 'simulation_asset_licenses': license_claims},
        'source_sha256': sources, 'runtime_binding': None,
        'scope': SCOPE, 'physical_robot_ready': False, 'task_qualified': False,
        'hardware_interfaces': [{'id': hardware_id, 'implemented': False, 'sdk_url': sdk_url}],
        'model_mutation_policy': ('Caller must not modify the built native model. Command validation checks '
                                  'topology, limits and servo settings, not a complete model hash each call.'),
    }


def contract(profile):
    """Describe exact source/asset locks and command semantics without native imports."""
    return _signed(_template(_profile(profile)))


def catalog():
    """List the three supported simulation interfaces without NumPy or MuJoCo."""
    keys = ('id', 'name', 'arm_axes', 'gripper_axes', 'action_names', 'action_units',
            'interface_id', 'scope', 'physical_robot_ready', 'task_qualified')
    return [{key: deepcopy(_template(profile)[key]) for key in keys} for profile in PROFILES]


def validate_contract(value):
    """Check exact installed metadata and digest; a digest is integrity, not a signature.

    A runtime binding can be inspected after serialization, but command validation
    additionally requires the actual model returned by build in this process.
    """
    if type(value) is not dict:
        raise ValueError('Contract must be an exact dictionary')
    profile = _profile(value.get('id'))
    base = _template(profile)
    if set(value) != set(base) | {'contract_sha256'}:
        raise ValueError('Contract keys differ from the installed interface')
    binding = value.get('runtime_binding')
    if binding is not None:
        expected_keys = {'mujoco_version', 'numpy_version', 'timestep_s', 'compiled_model_sha256'}
        if type(binding) is not dict or set(binding) != expected_keys:
            raise ValueError('Runtime binding keys differ from the native contract')
        if binding['mujoco_version'] != '3.15.0' or type(binding['numpy_version']) is not str:
            raise ValueError('Runtime binding requires MuJoCo 3.15.0 and an explicit NumPy version')
        sim._timestep(binding['timestep_s'])
        checksum = binding['compiled_model_sha256']
        if type(checksum) is not str or len(checksum) != 64 or any(x not in '0123456789abcdef' for x in checksum):
            raise ValueError('Runtime binding requires a lowercase SHA256 model digest')
    expected = dict(base, runtime_binding=binding)
    try:
        actual_unsigned = {key: value[key] for key in base}
        actual_digest = _digest(actual_unsigned)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError('Contract must contain finite canonical JSON data') from exc
    if actual_digest != _digest(expected) or value['contract_sha256'] != actual_digest:
        raise ValueError('Contract metadata/digest differs from the installed interface')
    return deepcopy(value)


def _topology(model, expected):
    """Require the scalar ordered transmission/servo topology, without mesh copying."""
    mj, np = sim._native_imports()
    if not isinstance(model, mj.MjModel):
        raise ValueError('Expected the native model returned by build')
    dimensions = {name: int(getattr(model, name)) for name in expected['expected_dimensions']}
    if dimensions != expected['expected_dimensions']:
        raise ValueError('Native topology dimensions differ from the interface')
    if [mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)] != expected['action_names']:
        raise ValueError('Native actuator order differs from the interface')
    if [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)] != expected['joint_names']:
        raise ValueError('Native joint order differs from the interface')
    if not np.array_equal(model.jnt_qposadr, expected['joint_qpos_indices']) or not np.array_equal(model.jnt_dofadr, expected['joint_dof_indices']):
        raise ValueError('Native scalar joint addresses differ from the interface')
    joint_types = [int(mj.mjtJoint.mjJNT_SLIDE if unit == 'm' else mj.mjtJoint.mjJNT_HINGE)
                   for unit in expected['joint_position_units']]
    if not np.array_equal(model.jnt_type, joint_types):
        raise ValueError('Native joint units differ from the interface')
    if not np.array_equal(model.actuator_ctrlrange, expected['action_limits_native']) or not np.all(model.actuator_ctrllimited):
        raise ValueError('Native command limits differ from the interface')
    for index, transmission in enumerate(expected['actuator_transmissions']):
        is_joint = transmission['type'] == 'joint'
        kind = mj.mjtObj.mjOBJ_JOINT if is_joint else mj.mjtObj.mjOBJ_TENDON
        trn = mj.mjtTrn.mjTRN_JOINT if is_joint else mj.mjtTrn.mjTRN_TENDON
        target = mj.mj_name2id(model, kind, transmission['target'])
        if target < 0 or model.actuator_trntype[index] != trn or model.actuator_trnid[index].tolist() != [target, -1]:
            raise ValueError('Native transmission differs from the interface')
    if mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, expected['tool_frame_body']) <= 0:
        raise ValueError('Native terminal link frame is missing')
    # These scalar position servos are also the only types for which the
    # benchmark may derive command-equivalent feedback from actuator length.
    if (not np.all(model.actuator_gaintype == mj.mjtGain.mjGAIN_FIXED)
            or not np.all(model.actuator_biastype == mj.mjtBias.mjBIAS_AFFINE)
            or not np.all(model.actuator_dyntype == mj.mjtDyn.mjDYN_NONE)
            or not np.all(np.isfinite(model.actuator_gainprm))
            or not np.all(model.actuator_gainprm[:, 0] > 0)):
        raise ValueError('Native position-servo types differ from the interface')


def _control_snapshot(model):
    # Capture the exact mechanics controlling the command mapping and coupling.
    # These arrays are small; visual meshes and inertias remain in the build hash.
    names = ('actuator_gainprm', 'actuator_biasprm', 'actuator_dynprm', 'actuator_gear',
             'actuator_forcerange', 'actuator_forcelimited', 'actuator_actlimited',
             'actuator_actrange', 'jnt_range', 'jnt_limited', 'eq_type', 'eq_obj1id',
             'eq_obj2id', 'eq_data', 'eq_active0')
    return {name: getattr(model, name).copy() for name in names}


def build(profile, assets, *, timestep=.001):
    """Build the exact pinned native model and initialize it without hidden steps.

    Assets are an existing local asset root; this function does not download.
    Supported native steps are 0.5, 1 and 2 ms. All seven/six commands are native
    authored servo inputs, including the UR gripper's 0..255 tendon input.
    """
    profile, dt = _profile(profile), sim._timestep(timestep)
    metadata = contract(profile)
    messages = []
    with sim.native_warnings(messages):
        model = (robot_profiles.build_model(profile, assets, timestep=dt) if profile in ('piper', 'so101')
                 else sim.build_model(assets, timestep=dt))
        mj, np = sim._native_imports()
        _topology(model, metadata)
        data = mj.MjData(model)
        if model.nkey:
            mj.mj_resetDataKeyframe(model, data, 0)
        else:
            data.ctrl[:] = metadata['initial_state']['ctrl_native']
        mj.mj_forward(model, data)
    if messages:
        raise ValueError('Native build/initialization emitted warnings: '+str(messages))
    initial = metadata['initial_state']
    if (data.time != 0. or not np.array_equal(data.qpos, initial['qpos_native'])
            or not np.array_equal(data.qvel, initial['qvel_native_per_second'])
            or not np.array_equal(data.ctrl, initial['ctrl_native'])
            or not all(np.isfinite(array).all() for array in (data.qpos, data.qvel, data.qacc, data.ctrl, data.xpos))
            or np.any(data.qfrc_applied) or np.any(data.xfrc_applied)):
        raise ValueError('Native initialization differs from the explicit interface home state')
    unsigned = {key: metadata[key] for key in _template(profile)}
    unsigned['runtime_binding'] = {
        'mujoco_version': mj.__version__, 'numpy_version': np.__version__,
        'timestep_s': dt, 'compiled_model_sha256': sim.model_digest(model),
    }
    bound = _signed(unsigned)
    _BUILT[model] = {'contract': deepcopy(bound), 'control_snapshot': _control_snapshot(model)}
    validate_command(bound, model, data.ctrl)
    return model, data, bound


def validate_command(contract_value, model, values):
    """Return a fresh ordered native vector; reject invalid data without clipping.

    Requires this process's exact build-returned model and corresponding bound
    contract. This function never writes controls, advances time or sends data.
    One-dimensional NumPy vectors and ordinary lists/tuples of real scalars are
    accepted; booleans, strings, complex numbers and coercible objects are not.
    """
    if type(contract_value) is not dict or contract_value.get('runtime_binding') is None:
        raise ValueError('Commands require a runtime-bound contract returned by build')
    mj, np = sim._native_imports()
    if not isinstance(model, mj.MjModel) or model not in _BUILT:
        raise ValueError('Commands require the exact native model returned by build in this process')
    registered = _BUILT[model]
    # Canonical bytes distinguish booleans from numbers and reject NaN/Infinity.
    try:
        unchanged = _canonical(contract_value) == _canonical(registered['contract'])
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError('Command contract must contain finite canonical JSON data') from exc
    if not unchanged:
        raise ValueError('Command contract differs from the model build binding')
    expected = registered['contract']
    _topology(model, expected)
    if model.opt.timestep != expected['runtime_binding']['timestep_s']:
        raise ValueError('Native timestep differs from the model build binding')
    if any(not np.array_equal(getattr(model, name), value) for name, value in registered['control_snapshot'].items()):
        raise ValueError('Native servo settings or gripper coupling changed after build')
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise ValueError('Command must be a one-dimensional vector')
        scalars = list(values)
    elif type(values) in (list, tuple):
        scalars = list(values)
    else:
        raise ValueError('Command must be a list, tuple or one-dimensional NumPy vector')
    if len(scalars) != len(expected['action_names']):
        raise ValueError('Command length differs from the ordered actuator contract')
    output = []
    for name, scalar, bounds in zip(expected['action_names'], scalars, expected['action_limits_native']):
        if isinstance(scalar, (bool, np.bool_)) or not isinstance(scalar, Real):
            raise ValueError('Command '+name+' must be a finite real scalar, excluding booleans')
        try:
            number = float(scalar)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('Command '+name+' must be a finite real scalar') from exc
        if not math.isfinite(number) or not bounds[0] <= number <= bounds[1]:
            raise ValueError('Command '+name+' is nonfinite or outside its native range')
        output.append(number)
    return output


def named_command(contract_value, model, values):
    """Validate an exact name/value dictionary and return the declared action order."""
    if type(contract_value) is not dict or type(values) is not dict:
        raise ValueError('Named commands require exact contract and value dictionaries')
    names = contract_value.get('action_names')
    if type(names) is not list or any(type(name) is not str for name in names):
        raise ValueError('Invalid action-name contract')
    if set(values) != set(names) or any(type(name) is not str for name in values):
        raise ValueError('Named command must supply exactly the declared actuator names')
    return validate_command(contract_value, model, [values[name] for name in names])

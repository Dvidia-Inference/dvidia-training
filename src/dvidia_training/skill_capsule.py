"""Bounded, data-only movement capsules and scene-specific local receipts.

JSON integrity and declared provenance are checked here.  Importing a capsule
does not execute a controller, fetch data, attest an experiment or qualify a
physical robot.  The caller owns the trusted native execution boundary.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
import re

from .skillspace import ARM_PROFILE, ADAPTER_ID, WORKSPACE_MIN, WORKSPACE_MAX, load_skillspace

MAX_CAPSULE_BYTES = 512 * 1024
MAX_SOURCE_BYTES = 128 * 1024
FORMAT = 'dvidia.skill-capsule'
TOP_FIELDS = {'format', 'schema_version', 'scope', 'status', 'physical_robot_ready',
              'source', 'grounding', 'compatibility', 'runtime', 'movement_head',
              'provenance', 'payload_sha256'}
_SHA = re.compile(r'^[a-f0-9]{64}$')
_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.:-]{0,119}$')
_JOINT_LIMITS = [[-2.6, 2.6], [-2.2, 1.6], [-2.7, 2.7], [-3., 3.], [-2.8, 2.8], [-3.1, 3.1]]


class SkillCapsuleError(ValueError):
    """Invalid, stale or incompatible imported data."""


def _fail(message):
    raise SkillCapsuleError(message)


def canonical_bytes(value) -> bytes:
    """Finite, sorted UTF-8 JSON; no executable or custom object serialization."""
    try:
        def walk(item, depth=0):
            if depth > 48:
                _fail('JSON nesting exceeds 48 levels.')
            if type(item) is dict:
                if any(type(key) is not str for key in item):
                    _fail('JSON object keys must be strings.')
                for key, val in item.items():
                    walk(key, depth+1); walk(val, depth+1)
            elif type(item) is list:
                for val in item:
                    walk(val, depth+1)
            elif type(item) is str:
                if any(0xD800 <= ord(c) <= 0xDFFF for c in item):
                    _fail('Unpaired Unicode surrogates are not allowed.')
            elif type(item) is float:
                if not math.isfinite(item):
                    _fail('JSON numbers must be finite.')
            elif type(item) not in (int, bool, type(None)):
                _fail('Use plain JSON objects, lists and scalar values.')
        walk(value)
        return json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=True, allow_nan=False).encode('utf-8')
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, SkillCapsuleError):
            raise
        raise SkillCapsuleError('Expected finite UTF-8 JSON.') from exc


def canonical_sha256(value) -> str:
    return sha256(canonical_bytes(value)).hexdigest()


def _record(value, fields, name):
    if type(value) is not dict or set(value) != set(fields):
        _fail(name+' has missing or unknown fields.')
    return value


def _digest(value, name):
    if type(value) is not str or not _SHA.fullmatch(value):
        _fail(name+' must be a lowercase SHA-256 digest.')
    return value


def _identity(value, name):
    if type(value) is not str or not _ID.fullmatch(value):
        _fail(name+' must be a bounded artifact ID.')
    return value


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        _fail(name+f' must be an integer in [{low}, {high}].')
    return value


def _number(value, low, high, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        _fail(name+f' must be finite in [{low}, {high}].')
    return value


def _boolean(value, name):
    if type(value) is not bool:
        _fail(name+' must be boolean.')
    return value


def _parse(document):
    if type(document) is dict:
        raw = canonical_bytes(document)
        if len(raw) > MAX_CAPSULE_BYTES:
            _fail('Capsule exceeds the 512 KiB limit.')
        # Deep copy through plain JSON: callers cannot mutate an inspected result
        # by retaining references to the original dictionary.
        return json.loads(raw)
    if type(document) not in (str, bytes):
        _fail('Supply JSON bytes, text or a plain dictionary.')
    try:
        raw = document.encode('utf-8') if type(document) is str else document
        if not raw or len(raw) > MAX_CAPSULE_BYTES:
            _fail('Capsule is empty or exceeds the 512 KiB limit.')
        def pairs(rows):
            value = {}
            for key, item in rows:
                if key in value:
                    _fail('Duplicate JSON field: '+key)
                value[key] = item
            return value
        def constant(token):
            _fail('Nonfinite JSON constant: '+token)
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=constant)
        canonical_bytes(value)  # Includes overflowed exponent rejection.
        return value
    except (ValueError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, SkillCapsuleError):
            raise
        raise SkillCapsuleError('Capsule must be finite UTF-8 JSON.') from exc


def arm_compatibility() -> dict:
    """The one supported embodiment; same joint count alone is insufficient."""
    observations = [
        {'name': 'joint_position', 'shape': [6], 'units': 'rad', 'frame': 'joint'},
        {'name': 'joint_velocity', 'shape': [6], 'units': 'rad/s', 'frame': 'joint'},
        {'name': 'joint_actuator_torque', 'shape': [6], 'units': 'N*m', 'frame': 'joint'},
        {'name': 'end_effector_position', 'shape': [3], 'units': 'm', 'frame': 'world'},
        {'name': 'end_effector_quaternion', 'shape': [4], 'units': 'wxyz', 'frame': 'world'},
        {'name': 'end_effector_velocity', 'shape': [3], 'units': 'm/s', 'frame': 'world'},
        {'name': 'object_position', 'shape': [3], 'units': 'm', 'frame': 'world'},
        {'name': 'object_quaternion', 'shape': [4], 'units': 'wxyz', 'frame': 'world'},
        {'name': 'object_velocity', 'shape': [3], 'units': 'm/s', 'frame': 'world'},
        {'name': 'object_angular_velocity', 'shape': [3], 'units': 'rad/s', 'frame': 'world'},
        {'name': 'target_position', 'shape': [3], 'units': 'm', 'frame': 'world'},
        {'name': 'object_size', 'shape': [3], 'units': 'm', 'frame': 'object'},
        {'name': 'gripper_width', 'shape': [], 'units': 'm', 'frame': 'gripper'},
        {'name': 'grasp_contacts', 'shape': [2], 'order': ['left', 'right'], 'units': 'count', 'frame': 'gripper'},
        {'name': 'pad_normal_forces', 'shape': [2], 'order': ['left', 'right'], 'units': 'N', 'frame': 'gripper'},
        {'name': 'effective_sliding_friction', 'shape': [2], 'order': ['jaw_object', 'table_object'], 'units': 'coefficient', 'frame': 'contact'},
        {'name': 'table_contacts', 'shape': [], 'units': 'count', 'frame': 'contact'},
    ]
    return {'arm_profile': ARM_PROFILE, 'adapter_id': ADAPTER_ID, 'arm_count': 1,
            'joint_order': ['j'+str(i) for i in range(6)], 'joint_limits_rad': [list(x) for x in _JOINT_LIMITS],
            'gripper': {'kind': 'symmetric_parallel_jaw', 'jaw_joints': ['jaw_left', 'jaw_right'],
                        'independent_commands': 1, 'gap_limits_m': [0., .08], 'maximum_jaw_actuator_force_n': 15.},
            'actions': [{'name': 'joint_targets', 'shape': [6], 'units': 'rad', 'semantics': 'absolute_joint_position'},
                        {'name': 'gripper_width', 'shape': [], 'units': 'm', 'semantics': 'total_symmetric_jaw_gap'}],
            'independent_commands': 7, 'physical_robot_joints': 8, 'control_hz': 50.,
            'maximum_joint_actuator_torque_nm': 40.,
            'target_rate_limits': {'joint_rad_s': 3., 'jaw_gap_m_s': .16},
            'tool_orientation': 'fixed_identity_rotation', 'coordinates': 'world_metres',
            'workspace': {'min': list(WORKSPACE_MIN), 'max': list(WORKSPACE_MAX), 'raised_tool_radial_max_m': .56},
            'object_profile': 'upright_axis_aligned_rigid_box_30_to_50_mm',
            'observation_origin': 'privileged_native_simulator_state', 'observations': observations,
            'supervisor_revision': 'dimension-aware-contact-rates-recovery-round2'}


def source_bytes(capsule) -> bytes:
    """Extract the exact UTF-8 task source from an already inspected capsule."""
    if type(capsule) is not dict:
        _fail('Capsule must be a plain JSON object.')
    source = _record(capsule.get('source'), {'text', 'sha256'}, 'source')
    if type(source['text']) is not str:
        _fail('source.text must be UTF-8 text.')
    try:
        raw = source['text'].encode('utf-8')
    except UnicodeError as exc:
        raise SkillCapsuleError('source.text must be UTF-8 text.') from exc
    if not raw or len(raw) > MAX_SOURCE_BYTES:
        _fail('Task source is empty or exceeds 128 KiB.')
    if sha256(raw).hexdigest() != _digest(source['sha256'], 'source.sha256'):
        _fail('Exact task source digest mismatch.')
    return raw


def payload_sha256(capsule) -> str:
    return canonical_sha256({key: value for key, value in capsule.items() if key != 'payload_sha256'})


def provenance_for_model(model, *, evaluation=None) -> dict:
    """Construct explicit hash-bound declarations, not a dataset attestation."""
    head_sha = canonical_sha256(model)
    training = model['training']
    return {'dataset': {'id': 'dataset:'+model['dataset_sha256'][:24],
                         'sha256': model['dataset_sha256'], 'sample_count': training['sample_count'],
                         'train_cases': list(model['train_cases'])},
            'training': {'id': 'training:'+head_sha[:24], 'algorithm': training['algorithm'],
                         'dataset_sha256': model['dataset_sha256'], 'movement_head_sha256': head_sha,
                         'sample_count': training['sample_count'],
                         'regularization': training['regularization'], 'seed': training['seed']},
            'evaluation': evaluation}


def _provenance(capsule):
    model = capsule['movement_head']
    value = _record(capsule['provenance'], {'dataset', 'training', 'evaluation'}, 'provenance')
    expected = provenance_for_model(model)
    for name in ('dataset', 'training'):
        row = _record(value[name], set(expected[name]), 'provenance.'+name)
        _identity(row['id'], 'provenance.'+name+'.id')
        for key, item in expected[name].items():
            if key != 'id' and (type(row[key]) is not type(item) or row[key] != item):
                _fail('Provenance '+name+' binding mismatch: '+key)
    evaluation = value['evaluation']
    if evaluation is None:
        return
    _record(evaluation, {'id', 'protocol', 'protocol_sha256', 'movement_head_sha256',
                        'training_dataset_sha256', 'runtime_sha256', 'trials', 'baseline_trials'},
            'provenance.evaluation')
    _identity(evaluation['id'], 'evaluation.id')
    for key, expected_digest in (
        ('movement_head_sha256', canonical_sha256(model)),
        ('training_dataset_sha256', model['dataset_sha256']),
        ('runtime_sha256', canonical_sha256(capsule['runtime'])),
        ('protocol_sha256', canonical_sha256(evaluation['protocol'])),
    ):
        if _digest(evaluation[key], 'evaluation.'+key) != expected_digest:
            _fail('Evaluation evidence binding mismatch: '+key)
    protocol = _record(evaluation['protocol'], {'id', 'scope', 'scene_selection', 'success_gate',
                       'split_kind', 'controls'}, 'evaluation.protocol')
    _identity(protocol['id'], 'protocol.id')
    if protocol['scope'] != 'simulation-only' or protocol['controls'] != ['replay_open_jaw']:
        _fail('Evaluation protocol must declare simulation-only scope and recorded-arm open-jaw control.')
    for key in ('scene_selection', 'success_gate', 'split_kind'):
        text = protocol[key]
        if type(text) is not str or not 1 <= len(text) <= 500 or any(ord(c) < 32 for c in text):
            _fail('Protocol '+key+' must be bounded plain text.')
    groups = []
    for name in ('trials', 'baseline_trials'):
        rows = evaluation[name]
        if type(rows) is not list or not 1 <= len(rows) <= 1000:
            _fail('Evaluation requires 1–1000 paired trial records.')
        bindings = {}
        for row in rows:
            _record(row, {'id', 'scene', 'scene_sha256', 'success', 'valid', 'reason'}, name+' trial')
            identity = _identity(row['id'], 'trial.id')
            if identity in bindings or identity in model['train_cases']:
                _fail('Evaluation trial IDs must be unique and disjoint from training IDs.')
            scene = normalize_scene(row['scene'])
            digest = _digest(row['scene_sha256'], 'trial.scene_sha256')
            if digest != canonical_sha256(scene):
                _fail('Evaluation scene digest mismatch.')
            _boolean(row['success'], 'trial.success'); _boolean(row['valid'], 'trial.valid')
            if row['success'] and not row['valid']:
                _fail('An invalid evaluation trial cannot declare success.')
            if type(row['reason']) is not str or not 1 <= len(row['reason']) <= 200:
                _fail('Evaluation reason must be bounded text.')
            bindings[identity] = digest
        groups.append(bindings)
    if groups[0] != groups[1]:
        _fail('Evaluation and baseline must use identical paired trial IDs and scenes.')


def inspect_capsule(document, *, expected_runtime=None) -> dict:
    """Validate imported data; returned capsules always retain candidate status."""
    capsule = _record(_parse(document), TOP_FIELDS, 'capsule')
    for key, expected in {'format': FORMAT, 'schema_version': 1, 'scope': 'simulation-only',
                          'status': 'candidate', 'physical_robot_ready': False}.items():
        if type(capsule[key]) is not type(expected) or capsule[key] != expected:
            _fail('Unsupported capsule '+key+'.')
    if _digest(capsule['payload_sha256'], 'payload_sha256') != payload_sha256(capsule):
        _fail('Canonical capsule payload digest mismatch.')
    raw = source_bytes(capsule)
    try:
        source = json.loads(raw)
        if 'simulation_grounding' in source:
            if source['simulation_grounding'] != capsule['grounding']:
                _fail('Inline source grounding does not match capsule grounding.')
            load_skillspace(raw)
        else:
            load_skillspace(raw, capsule['grounding'])
    except (ValueError, TypeError, KeyError) as exc:
        if isinstance(exc, SkillCapsuleError):
            raise
        raise SkillCapsuleError('Task source or authored grounding is invalid: '+str(exc)) from exc
    if canonical_bytes(capsule['compatibility']) != canonical_bytes(arm_compatibility()):
        _fail('Incompatible arm, gripper, observations, actions or authored supervisor.')
    if type(capsule['runtime']) is not dict or not capsule['runtime']:
        _fail('Capsule must bind a nonempty local runtime contract.')
    if expected_runtime is not None and canonical_bytes(capsule['runtime']) != canonical_bytes(expected_runtime):
        _fail('Capsule controller or native dependency is incompatible with the current runtime.')
    from .arm_distill import validate_model
    try:
        validated = validate_model(capsule['movement_head'])
        if validated is not None and validated != capsule['movement_head']:
            _fail('Movement-head validation changed its imported data.')
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise SkillCapsuleError('Invalid movement head: '+str(exc)) from exc
    # Imported coefficients must remain inside a numerical envelope as well
    # as float32's representable range.  Squaring huge finite RBF distances
    # or summing huge finite weights could otherwise overflow at inference.
    def bounded_coefficients(value):
        if type(value) is list:
            return all(bounded_coefficients(item) for item in value)
        return type(value) in (int, float) and math.isfinite(value) and abs(value) <= 1e6
    for name in ('centers', 'context_mean', 'context_scale', 'weights'):
        if not bounded_coefficients(capsule['movement_head'][name]):
            _fail('Imported movement-head '+name+' exceeds the 1e6 numerical envelope.')
    _provenance(capsule)
    return capsule


def build_capsule(source: bytes, grounding: dict | None, movement_head: dict,
                  provenance=None, *, runtime: dict) -> dict:
    """Export exact source bytes, numeric learned head and authored contract."""
    if type(source) is not bytes or not source or len(source) > MAX_SOURCE_BYTES:
        _fail('Export requires 1–128 KiB of exact task-source bytes.')
    try:
        text = source.decode('utf-8')
        parsed = json.loads(text)
    except (UnicodeError, ValueError) as exc:
        raise SkillCapsuleError('Task source must be UTF-8 JSON.') from exc
    if type(parsed) is not dict:
        _fail('Task source must be a JSON object.')
    if 'simulation_grounding' in parsed:
        if grounding is not None and grounding != parsed['simulation_grounding']:
            _fail('Export grounding differs from inline source grounding.')
        grounding = parsed['simulation_grounding']
    capsule = {'format': FORMAT, 'schema_version': 1, 'scope': 'simulation-only',
               'status': 'candidate', 'physical_robot_ready': False,
               'source': {'text': text, 'sha256': sha256(source).hexdigest()}, 'grounding': grounding,
               'compatibility': arm_compatibility(), 'runtime': runtime, 'movement_head': movement_head,
               'provenance': provenance if provenance is not None else provenance_for_model(movement_head)}
    capsule['payload_sha256'] = payload_sha256(capsule)
    return inspect_capsule(capsule, expected_runtime=runtime)


def encode_capsule(capsule) -> bytes:
    capsule = inspect_capsule(capsule)
    raw = canonical_bytes(capsule)+b'\n'
    if len(raw) > MAX_CAPSULE_BYTES:
        _fail('Encoded capsule exceeds the 512 KiB limit.')
    return raw


def capsule_summary(capsule) -> dict:
    capsule = inspect_capsule(capsule)
    return {'format': FORMAT, 'capsule_id': capsule['payload_sha256'][:24],
            'payload_sha256': capsule['payload_sha256'], 'status': 'candidate',
            'scope': 'simulation-only', 'physical_robot_ready': False,
            'arm_profile': capsule['compatibility']['arm_profile'],
            'source_sha256': capsule['source']['sha256'],
            'movement_head_sha256': canonical_sha256(capsule['movement_head']),
            'training_samples': capsule['provenance']['dataset']['sample_count'],
            'training_cases': len(capsule['provenance']['dataset']['train_cases']),
            'has_declared_evaluation': capsule['provenance']['evaluation'] is not None,
            'qualification': 'Local scene execution and native negative control are required after import.',
            'verification_scope': 'JSON shape, limits, compatibility and hash consistency; no competence or authenticity attestation.',
            'policy_origin': 'Learned NumPy movement head with locally authored waypoint, phase, contact and recovery supervisor.'}


SCENE_FIELDS = {'object_position', 'target_position', 'seed', 'object_size', 'object_mass',
                'object_friction', 'jaw_force_limit', 'joint_torque_limit'}


def normalize_scene(scene) -> dict:
    """Canonical physical inputs; no implicit defaults in evidence bindings."""
    _record(scene, SCENE_FIELDS, 'scene')
    out = {'seed': _integer(scene['seed'], 0, 999999, 'scene.seed')}
    for name in ('object_position', 'target_position', 'object_size'):
        value = scene[name]
        if type(value) is not list or len(value) != 3:
            _fail('Scene '+name+' must have exactly three coordinates.')
        out[name] = [float(_number(x, -10., 10., 'scene.'+name)) for x in value]
    for name, low, high in (('object_mass', .02, .08), ('object_friction', .4, 1.5),
                            ('jaw_force_limit', .01, 15.), ('joint_torque_limit', .1, 40.)):
        out[name] = float(_number(scene[name], low, high, 'scene.'+name))
    if any(not .03 <= x <= .05 for x in out['object_size']):
        _fail('Box dimensions must be 30–50 mm.')
    for name in ('object_position', 'target_position'):
        x, y, z = out[name]
        if not .25 <= x <= .60 or not -.20 <= y <= .20 or math.hypot(x, y) > .56:
            _fail('Scene position is outside the authored arm reach.')
        if abs(z-(.29+out['object_size'][2]/2)) > .01000001:
            _fail('Scene object and target must lie on the table support plane.')
    if math.dist(out['object_position'][:2], out['target_position'][:2]) < .075:
        _fail('Scene must move the object by at least 75 mm.')
    return out


def scene_sha256(scene) -> str:
    return canonical_sha256(normalize_scene(scene))


def _native_json(value):
    """Trusted local outputs may contain dataclass tuples; imports may not."""
    if type(value) is tuple:
        return [_native_json(x) for x in value]
    if type(value) is list:
        return [_native_json(x) for x in value]
    if type(value) is dict:
        return {key: _native_json(item) for key, item in value.items()}
    return value


def _expected_config(capsule, scene):
    skill = load_skillspace(source_bytes(capsule),
                           None if 'simulation_grounding' in json.loads(source_bytes(capsule)) else capsule['grounding'])
    return {'timestep': .001, 'control_dt': .02, 'horizon': 18., 'settle_time': .4,
            'object_position': scene['object_position'], 'target_position': scene['target_position'],
            'object_half_size': [x/2 for x in scene['object_size']], 'object_mass': scene['object_mass'],
            'object_friction': scene['object_friction'], 'table_height': skill.table_height,
            'position_tolerance': skill.position_tolerance, 'dwell_seconds': skill.dwell_seconds,
            'scene_jitter': 0., 'joint_torque_limit': scene['joint_torque_limit'],
            'jaw_force_limit': scene['jaw_force_limit'], 'joint_target_rate_limit': 3.,
            'jaw_width_rate_limit': .16, 'gravity': 9.81}


def _number_tree(value, expected, path):
    if type(expected) is list:
        if type(value) is not list or len(value) != len(expected):
            _fail(path+' shape mismatch.')
        for index, (item, wanted) in enumerate(zip(value, expected)):
            _number_tree(item, wanted, path+f'[{index}]')
    elif type(expected) in (int, float):
        if type(value) not in (int, float) or not math.isfinite(value) or value != expected:
            _fail(path+' differs from the exact scene or unchanged simulation configuration.')
    else:
        if type(value) is not type(expected) or value != expected:
            _fail(path+' mismatch.')


def _native_episode(payload, capsule, scene, runtime, policy):
    if type(payload) is not dict:
        _fail('Local native result must be a run payload.')
    for key, expected in {'schema_version': 1, 'task': 'ArmPickPlace-v0', 'scope': 'simulation-only'}.items():
        if type(payload.get(key)) is not type(expected) or payload[key] != expected:
            _fail('Local result task or scope mismatch.')
    if canonical_bytes(payload.get('runtime')) != canonical_bytes(runtime):
        _fail('Local result used another controller or native dependency.')
    skill = load_skillspace(source_bytes(capsule),
                           None if 'simulation_grounding' in json.loads(source_bytes(capsule)) else capsule['grounding'])
    if payload.get('skill') != skill.to_dict():
        _fail('Local result skill or source differs from the installed capsule.')
    config = _record(payload.get('environment_config'), set(_expected_config(capsule, scene)), 'local environment_config')
    for key, wanted in _expected_config(capsule, scene).items():
        _number_tree(config[key], wanted, 'local environment_config.'+key)
    episodes = payload.get('episodes')
    if type(episodes) is not list or len(episodes) != 1:
        _fail('Scene qualification requires exactly one native episode per result.')
    episode = episodes[0]
    if type(episode) is not dict or episode.get('policy') != policy:
        _fail('Unexpected local qualification policy.')
    if type(episode.get('seed')) is not int or episode['seed'] != scene['seed']:
        _fail('Local episode seed does not match the exact scene.')
    _boolean(episode.get('success'), 'local success')
    info = episode.get('final_info')
    if type(info) is not dict or type(info.get('success')) is not bool or info['success'] != episode['success']:
        _fail('Local final success record mismatch.')
    if info.get('valid') is not True or info.get('warnings') != []:
        _fail('Native state and solver must remain valid without warnings.')
    if info.get('task_id') != 'ArmPickPlace-v0' or info.get('arm_profile') != ARM_PROFILE or info.get('adapter_id') != ADAPTER_ID:
        _fail('Native task or embodiment does not match the capsule.')
    if info.get('qualification') != 'simulation-only' or info.get('observations') != 'privileged simulator state':
        _fail('Native observation or qualification scope mismatch.')
    if type(episode.get('reason')) is not str or episode['reason'] != info.get('reason'):
        _fail('Native episode reason does not match final state.')
    _number(episode.get('simulated_seconds'), .02, 18.00000001, 'local simulated_seconds')
    if episode['simulated_seconds'] != info.get('simulation_time'):
        _fail('Local simulation time binding mismatch.')
    actions = episode.get('actions')
    if type(actions) is not list or not 1 <= len(actions) <= 900:
        _fail('Native qualification must retain 1–900 commanded actions.')
    for row in actions:
        _record(row, {'stage', 'action'}, 'local action record')
        if type(row['stage']) is not str or not 1 <= len(row['stage']) <= 80:
            _fail('Action phase is invalid.')
        action = _record(row['action'], {'joint_targets', 'gripper_width'}, 'local action')
        targets = action['joint_targets']
        if type(targets) is not list or len(targets) != 6:
            _fail('Native qualification requires six joint targets.')
        for index, target in enumerate(targets):
            _number(target, *_JOINT_LIMITS[index], 'local joint target')
        _number(action['gripper_width'], 0., .08, 'local jaw gap')
    if type(episode.get('control_steps')) is not int or episode['control_steps'] != len(actions):
        _fail('Native action count mismatch.')
    if not (len(actions)-1)*.02-1e-8 < episode['simulated_seconds'] <= len(actions)*.02+1e-8:
        _fail('Native time does not match its recorded control steps.')
    return episode


def qualification_receipt(capsule, scene, native_result, *, runtime, controls) -> dict:
    """Bind an actual caller-owned native success and negative control to one scene.

    This helper verifies consistency.  The caller must generate both results
    locally; arbitrary imported receipts/results are not execution attestation.
    """
    capsule = inspect_capsule(capsule, expected_runtime=runtime)
    scene = normalize_scene(scene)
    result = _native_json(native_result)
    control_result = _native_json(controls)
    canonical_bytes(result); canonical_bytes(control_result)
    student = _native_episode(result, capsule, scene, runtime, 'distilled_placement')
    control = _native_episode(control_result, capsule, scene, runtime, 'replay_open_jaw')
    if student['success'] is not True or student['reason'] != 'success':
        _fail('Candidate did not acquire the exact local scene skill.')
    info = student['final_info']
    if info.get('grasp_seen') is not True or info.get('lift_seen') is not True:
        _fail('Local placement requires native bilateral loaded grasp and retained lift.')
    _number(info.get('max_object_lift'), .055, 10., 'local retained lift')
    # lift_seen is latched only after 100 ms of native retained lift.  The
    # current lift_dwell_elapsed correctly resets to zero after placement.
    _number(info.get('dwell_elapsed'), capsule['grounding']['target']['dwell_seconds'], 18.00000001, 'local placement dwell')
    _number(info.get('target_distance'), 0., capsule['grounding']['target']['position_tolerance'], 'local target distance')
    _integer(info.get('table_contacts'), 1, 10000, 'local table support')
    if info['dwell_elapsed'] > student['simulated_seconds']+1e-8:
        _fail('Native placement dwell cannot exceed episode duration.')
    diagnostics = student.get('controller_diagnostics')
    if (type(diagnostics) is not dict
            or diagnostics.get('movement_head_sha256') != canonical_sha256(capsule['movement_head'])
            or type(diagnostics.get('teacher_fallback_calls')) is not int
            or diagnostics['teacher_fallback_calls'] != 0
            or type(diagnostics.get('student_inference_calls')) is not int
            or diagnostics['student_inference_calls'] != student['control_steps']):
        _fail('Native result must bind the actual student head with no teacher fallback.')
    traces = student.get('trace')
    if type(traces) is not list or not traces or type(traces[-1]) is not dict:
        _fail('Native qualification must retain the terminal observation.')
    terminal = traces[-1]
    vectors = {}
    for name in ('object_position', 'end_effector_position', 'object_velocity', 'object_angular_velocity'):
        row = terminal.get(name)
        if type(row) is not list or len(row) != 3:
            _fail('Terminal '+name+' must have three finite values.')
        vectors[name] = [float(_number(x, -100., 100., 'terminal.'+name)) for x in row]
    if (terminal.get('grasp_contacts') != {'left': 0, 'right': 0}
            or type(terminal.get('gripper_width')) not in (int, float)
            or not .055 < terminal['gripper_width'] <= .08000001
            or vectors['end_effector_position'][2] < vectors['object_position'][2]+.10-1e-8
            or abs(vectors['object_position'][2]-scene['target_position'][2]) >= .008
            or math.dist(vectors['object_position'], scene['target_position']) > capsule['grounding']['target']['position_tolerance']
            or math.hypot(*vectors['object_velocity']) >= .035
            or math.hypot(*vectors['object_angular_velocity']) >= .3):
        _fail('Terminal native state does not satisfy release, clearance, support and settling conditions.')
    if control['success'] is not False or control['reason'] == 'success':
        _fail('Recorded-arm open-jaw native negative control unexpectedly succeeded.')
    for index, row in enumerate(control['actions']):
        wanted = student['actions'][min(index, len(student['actions'])-1)]
        if row['stage'] != wanted['stage'] or row['action']['joint_targets'] != wanted['action']['joint_targets']:
            _fail('Open-jaw control must replay the identical recorded arm commands.')
        if row['action']['gripper_width'] != .08:
            _fail('Open-jaw negative control must hold the jaws open at 80 mm.')
    if len(control['actions']) < len(student['actions']):
        _fail('Open-jaw control ended before the recorded student tape.')
    receipt = {'format': 'dvidia.local-skill-qualification', 'schema_version': 1,
               'status': 'validated_simulation_scene', 'scope': 'simulation-only',
               'physical_robot_ready': False, 'capsule_payload_sha256': capsule['payload_sha256'],
               'runtime_sha256': canonical_sha256(runtime), 'scene': scene,
               'scene_sha256': canonical_sha256(scene), 'result_sha256': canonical_sha256(result),
               'control_result_sha256': canonical_sha256(control_result),
               'student': {'success': True, 'policy': 'distilled_placement', 'seed': scene['seed'],
                           'simulation_seconds': student['simulated_seconds'], 'target_distance_m': info['target_distance']},
               'control': {'success': False, 'policy': 'replay_open_jaw', 'seed': scene['seed'],
                           'simulation_seconds': control['simulated_seconds'], 'reason': control['reason']},
               'verification_scope': 'Caller-owned native local execution; one exact scene and unchanged success gate. No generalization, authenticity or hardware attestation.'}
    receipt['receipt_sha256'] = canonical_sha256(receipt)
    return receipt


def check_qualification(receipt, capsule, scene, *, runtime) -> dict:
    """Check a locally stored receipt's bindings; this does not execute a test."""
    capsule = inspect_capsule(capsule, expected_runtime=runtime)
    receipt = _parse(receipt)
    fields = {'format', 'schema_version', 'status', 'scope', 'physical_robot_ready', 'capsule_payload_sha256',
              'runtime_sha256', 'scene', 'scene_sha256', 'result_sha256', 'control_result_sha256',
              'student', 'control', 'verification_scope', 'receipt_sha256'}
    _record(receipt, fields, 'qualification receipt')
    for key, expected in {'format': 'dvidia.local-skill-qualification', 'schema_version': 1,
                          'status': 'validated_simulation_scene', 'scope': 'simulation-only', 'physical_robot_ready': False}.items():
        if type(receipt[key]) is not type(expected) or receipt[key] != expected:
            _fail('Unsupported local qualification '+key+'.')
    unsigned = {key: val for key, val in receipt.items() if key != 'receipt_sha256'}
    if _digest(receipt['receipt_sha256'], 'receipt_sha256') != canonical_sha256(unsigned):
        _fail('Stored qualification digest mismatch.')
    normalized = normalize_scene(scene)
    if (receipt['capsule_payload_sha256'] != capsule['payload_sha256']
            or receipt['runtime_sha256'] != canonical_sha256(runtime)
            or receipt['scene'] != normalized or receipt['scene_sha256'] != canonical_sha256(normalized)):
        _fail('Qualification is stale or belongs to another capsule, runtime or scene.')
    for key in ('capsule_payload_sha256', 'runtime_sha256', 'scene_sha256', 'result_sha256', 'control_result_sha256'):
        _digest(receipt[key], key)
    student = _record(receipt['student'], {'success', 'policy', 'seed', 'simulation_seconds', 'target_distance_m'}, 'receipt.student')
    control = _record(receipt['control'], {'success', 'policy', 'seed', 'simulation_seconds', 'reason'}, 'receipt.control')
    for row, policy, success in ((student, 'distilled_placement', True), (control, 'replay_open_jaw', False)):
        if row['success'] is not success or row['policy'] != policy or type(row['seed']) is not int or row['seed'] != normalized['seed']:
            _fail('Qualification student or control summary is invalid.')
        _number(row['simulation_seconds'], .02, 18.00000001, 'receipt simulation_seconds')
    _number(student['target_distance_m'], 0., capsule['grounding']['target']['position_tolerance'], 'receipt distance')
    if type(control['reason']) is not str or not 1 <= len(control['reason']) <= 200 or control['reason'] == 'success':
        _fail('Qualification control reason is invalid.')
    if type(receipt['verification_scope']) is not str or not 1 <= len(receipt['verification_scope']) <= 500:
        _fail('Qualification verification scope is invalid.')
    return receipt

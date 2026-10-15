"""Audit sealed v03 evidence and compile declared models without simulation steps."""
import argparse
from hashlib import sha256
from pathlib import Path
import json
import math
import subprocess
import time
from dvidia_training import graspability, graspability_protocol, graspability_benchmark, reference_sim

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repository', type=Path, default=Path.cwd(),
                    help='Source Git checkout; defaults to the current working directory')
parser.add_argument('--export', required=True, type=Path)
parser.add_argument('--assets', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
REPO = Path(subprocess.check_output(
    ['git', '-C', str(args.repository.resolve()), 'rev-parse', '--show-toplevel'],
    text=True).strip()).resolve()
ROOT = args.export.resolve()
OUTPUT = args.output.resolve()
SOURCE = '73057dc259c6a19344906f4a43a283553e17ade0'
ASSETS = args.assets.resolve()
started = time.perf_counter()
report = json.loads((ROOT / 'benchmark.json').read_bytes())
protocol = json.loads((ROOT / 'protocol.json').read_bytes())
seal = json.loads((ROOT / 'integrity.json').read_bytes())
assert report['source']['commit'] == protocol['source_commit'] == SOURCE
assert report['source']['protocol_sha256'] == sha256((ROOT / 'protocol.json').read_bytes()).hexdigest()
assert protocol['split'] == report['split'] == 'heldout'
assert protocol['fixture_sha256'] == graspability_protocol.FIXTURE_SHA256
assert protocol['attempts'] == graspability_protocol.protocol('heldout')['attempts']
assert len(protocol['attempts']) == len(report['attempts']) == 24
actual_files = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()}
assert actual_files == set(seal['files']) | {'integrity.json'}
for name, entry in seal['files'].items():
    raw = (ROOT / name).read_bytes()
    assert len(raw) == entry['bytes'] and sha256(raw).hexdigest() == entry['sha256'], name
for name, digest in protocol['implementation_sha256'].items():
    rel = 'src/dvidia_training/' + name
    committed = subprocess.check_output(['git', 'show', SOURCE + ':' + rel], cwd=REPO)
    assert sha256(committed).hexdigest() == digest, name
    assert sha256((REPO / rel).read_bytes()).hexdigest() == digest, name
mj, np = reference_sim._native_imports()
# Compiling a model is permitted; executing any native step is forbidden here.
def forbidden_step(*args, **kwargs):
    raise AssertionError('Independent export audit must not rerun native physics')
mj.mj_step = forbidden_step
native_robot_contract = None
rows = []
for planned, row in zip(protocol['attempts'], report['attempts']):
    assert {k: row[k] for k in ('id', 'task', 'policy', 'case')} == planned
    receipt = json.loads((ROOT / 'trials' / (row['id'] + '.json')).read_bytes())
    assert {k: v for k, v in receipt.items() if k != 'native_trace'} == row
    score = graspability_protocol.score_result(receipt)
    assert row['valid'] and score['valid']
    assert score['task_success'] == row['success']
    assert all(row['independent_score'][k] == v for k, v in score.items())
    assert row['independent_score']['controller_claim_agrees']
    assert row['controller_reported_success'] == row['success']
    assert score['reference'] and score['reference']['valid']
    case = row['case']
    model = graspability.build_model(ASSETS, case, tool=case['tool_condition'])
    assert reference_sim.model_digest(model) == row['independent_score']['compiled_model_sha256']
    assert graspability_benchmark.robot_actuator_fingerprint(model) == row['robot_actuator_sha256']
    obj = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, graspability.ws.SHOE)
    assert math.isclose(float(model.body_mass[obj]), case['mass'], abs_tol=1e-12)
    assert model.body_gravcomp[obj] == 0
    assert model.jnt_type[model.body_jntadr[obj]] == mj.mjtJoint.mjJNT_FREE
    assert row['metrics']['object_equality_constraints'] == 0
    assert row['metrics']['object_external_force_used'] is False
    assert row['metrics']['solver_warnings'] == 0
    np.testing.assert_allclose(model.body_pos[obj][:2], [case['object_x'], case['object_y']], atol=1e-12)
    for g in range(int(model.body_geomadr[obj]), int(model.body_geomadr[obj] + model.body_geomnum[obj])):
        np.testing.assert_allclose(model.geom_friction[g], [case['friction'], .005, .0001], atol=1e-12)
    tool = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'reach_rake')
    assert (tool >= 0) == (case['tool_condition'] == 'mounted')
    # Compare named robot links directly as well as the declared actuator hash.
    robot = {}
    for name in ('base_link', *(f'link{i}' for i in range(1, 9))):
        body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        if body < 0:
            continue
        robot[name] = {key: np.asarray(getattr(model, key)[body]).tolist() for key in
                       ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat')}
    if native_robot_contract is None:
        native_robot_contract = robot
    assert robot == native_robot_contract
    trace = np.asarray(receipt['native_trace'])
    columns = receipt['native_trace_columns']
    final = trace[-250:]
    drive = trace[:, columns.index('gripper_actuator_force_n')]
    rows.append({'id': row['id'], 'object_shape': case['object_shape'],
                 'mass_kg': case['mass'], 'tool_condition': case['tool_condition'],
                 'replicate_id': case['replicate_id'], 'friction': case['friction'],
                 'valid': row['valid'], 'success': row['success'],
                 'compiled_model_sha256': row['independent_score']['compiled_model_sha256'],
                 'failed_criteria': score['failed_criteria'],
                 'terminal_relative_lift_m': score['terminal_relative_lift_m'],
                 'minimum_final_collision_surface_z_m': score['minimum_final_collision_surface_z_m'],
                 'final_bilateral_missing_ticks': score['bilateral_contact_missing_ticks'],
                 'final_world_supported_ticks': score['world_supported_ticks'],
                 'final_tool_supported_ticks': score['tool_supported_ticks'],
                 'final_other_robot_supported_ticks': score['other_robot_supported_ticks'],
                 'max_abs_native_gripper_actuator_force_n': float(np.max(np.abs(drive))),
                 'closure_observation': row['metrics']['closure_observation'],
                 'simulation_seconds': row['metrics']['simulation_seconds'],
                 'native_wall_seconds': row['wall_seconds'],
                 'audit_export_seconds': row['export_seconds']})
assert graspability_benchmark.summarize(report['attempts']) == report['groups']
assert graspability_benchmark.paired_comparisons(report['attempts']) == report['paired_comparisons']
assert all(pair['valid_pair'] and pair['same_robot_actuators'] for pair in report['paired_comparisons'])
assert len({row['robot_actuator_sha256'] for row in report['attempts']}) == 1
simulation_seconds = sum(row['simulation_seconds'] for row in rows)
native_wall_seconds = sum(row['native_wall_seconds'] for row in rows)
result = {'schema': 'dvidia.graspability-independent-audit/v3',
          'scope': 'Recomputed all recorded scores and compiled all declared models; no native simulation steps or controller tuning.',
          'source_commit': SOURCE,
          'protocol_sha256': sha256((ROOT / 'protocol.json').read_bytes()).hexdigest(),
          'benchmark_sha256': sha256((ROOT / 'benchmark.json').read_bytes()).hexdigest(),
          'integrity_sha256': sha256((ROOT / 'integrity.json').read_bytes()).hexdigest(),
          'fixture_sha256': graspability_protocol.FIXTURE_SHA256,
          'native_steps_executed_by_audit': 0,
          'compiled_models_verified': len(rows), 'attempts_verified': len(rows),
          'valid_attempts': sum(row['valid'] for row in rows),
          'files_verified': len(seal['files']),
          'implementation_files_bound_to_frozen_commit': len(protocol['implementation_sha256']),
          'robot_link_mass_inertia_unchanged': True,
          'robot_actuator_sha256': report['attempts'][0]['robot_actuator_sha256'],
          'groups': report['groups'], 'paired_comparisons': report['paired_comparisons'],
          'timing': {'simulated_seconds': simulation_seconds,
                     'native_wall_seconds': native_wall_seconds,
                     'simulated_to_native_wall_ratio': simulation_seconds / native_wall_seconds,
                     'audit_export_wall_seconds': sum(row['audit_export_seconds'] for row in rows),
                     'scope': report['timing_scope']},
          'load_context': {'lightweight_control_weight_n': .03 * 9.81,
                           'reference_load_weight_n': .30 * 9.81,
                           'gripper_position_gain_n_per_m': 40.,
                           'scope': 'Assumed model loads and pinned simulator gain. Contact normals can include table pressure; signed actuator force is not directly measured pinch capacity. These runs do not identify physical PiPER payload or isolate a calibrated friction limit.'},
          'rows': rows, 'audit_wall_seconds': time.perf_counter() - started}
OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps({k: result[k] for k in ('source_commit', 'attempts_verified', 'valid_attempts', 'files_verified', 'compiled_models_verified', 'timing', 'groups')}, indent=2))

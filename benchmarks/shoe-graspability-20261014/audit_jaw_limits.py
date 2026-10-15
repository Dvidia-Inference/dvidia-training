"""Read native jaw spans and sampled joint positions; execute no simulation."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from dvidia_training import graspability, reference_sim

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--export', required=True, type=Path)
parser.add_argument('--assets', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
root = args.export
report = json.loads((root / 'benchmark.json').read_bytes())
mj, np = reference_sim._native_imports()
model = graspability.build_model(args.assets, report['attempts'][0]['case'],
                                 tool=report['attempts'][0]['case']['tool_condition'])
ranges = model.jnt_range[:8]
nominal_span = float(ranges[6, 1] - ranges[7, 0])
rows = []
for attempt in report['attempts']:
    receipt = json.loads((root / 'trials' / (attempt['id'] + '.json')).read_bytes())
    columns = receipt['native_trace_columns']
    trace = np.asarray(receipt['native_trace'])
    jaw = trace[:, columns.index('jaw_opening_m')]
    final = jaw[-250:]
    motion = json.loads((root / attempt['replay']['motion']).read_bytes())
    sampled = motion['rows']
    qpos = np.asarray([row['qpos_rad_m'][:8] for row in sampled])
    violations = np.maximum(np.maximum(qpos - ranges[:, 1], ranges[:, 0] - qpos), 0.)
    final_rows = [row for row in sampled if row['t'] > trace[-1, 0] - .5 + 1e-8]
    final_qpos = np.asarray([row['qpos_rad_m'][:8] for row in final_rows])
    final_violation = np.maximum(np.maximum(final_qpos - ranges[:, 1], ranges[:, 0] - final_qpos), 0.)
    ctrl = np.asarray([row['issued_ctrl'] for row in sampled])
    ctrllimits = model.actuator_ctrlrange
    assert np.all(ctrl >= ctrllimits[:, 0] - 1e-10) and np.all(ctrl <= ctrllimits[:, 1] + 1e-10)
    rows.append({'id': attempt['id'], 'object_shape': attempt['case']['object_shape'],
                 'mass_kg': attempt['case']['mass'], 'tool_condition': attempt['case']['tool_condition'],
                 'retention_success_under_frozen_criteria': attempt['success'],
                 'maximum_native_jaw_span_m': float(np.max(jaw)),
                 'minimum_final_native_jaw_span_m': float(np.min(final)),
                 'maximum_final_native_jaw_span_m': float(np.max(final)),
                 'final_native_ticks_above_nominal_span': int(np.sum(final > nominal_span + 1e-9)),
                 'max_joint_limit_violation_on_20ms_frames_rad_m': np.max(violations, axis=0).tolist(),
                 'max_final_joint_limit_violation_on_20ms_frames_rad_m': np.max(final_violation, axis=0).tolist(),
                 'issued_commands_within_declared_control_limits_on_all_exported_frames': True})
result = {'schema': 'dvidia.graspability-jaw-limit-diagnostic/v3',
          'source_commit': report['source']['commit'],
          'benchmark_sha256': sha256((root / 'benchmark.json').read_bytes()).hexdigest(),
          'scope': 'Post-freeze diagnostic on recorded data only. Native jaw-span observations are every2ms; individual joint-position observations are sampled every20ms. No rescoring, controller tuning or simulation steps.',
          'native_steps_executed_by_audit': 0,
          'nominal_slide_coordinate_span_m': nominal_span,
          'pinned_robot_joint_ranges_rad_m': ranges.tolist(),
          'interpretation': 'MuJoCo joint limits are compliant constraints, while actuator command limits bound issued targets. Recorded jaw coordinates can exceed nominal travel under contact. Existing retention criteria did not include a joint-limit tolerance, so scores stay unchanged. Observed overtravel does not establish real hardware aperture or prove that the same lift works with hard travel stops.',
          'rows': rows}
args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
for row in rows:
    if row['object_shape'] == 'current-shoe' and row['retention_success_under_frozen_criteria']:
        print(json.dumps(row, indent=2))

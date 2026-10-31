"""Read-only independent audit of the fixed arm-interface tracking run."""
from hashlib import sha256
import json
import math
from pathlib import Path

root=Path('runs/arm-interfaces-v01-20261031')
report=json.loads((root/'report.json').read_text())
protocol_bytes=(root/'protocol.json').read_bytes()
assert sha256(protocol_bytes).hexdigest()==report['protocol']['sha256']
protocol=json.loads(protocol_bytes)
assert len(protocol['schedule'])==54==len(report['attempts'])
assert all(sha256((root/'source'/f['name']).read_bytes()).hexdigest()==f['sha256'] for f in protocol['source_inventory'])
assert len({a['id'] for a in report['attempts']})==54
rows_total=commands_total=metric_comparisons=0
for scheduled,result in zip(protocol['schedule'],report['attempts']):
    assert all(result[k]==v for k,v in scheduled.items())
    folder=root/result['id']
    assert json.loads((folder/'result.json').read_text())==result
    assert json.loads((folder/'started.json').read_text())['attempt']==scheduled
    raw=(folder/'trace.json').read_bytes()
    assert sha256(raw).hexdigest()==result['trace']['sha256'] and len(raw)==result['trace']['bytes']
    trace=json.loads(raw)
    rows,commands=trace['rows'],trace['commands']
    assert result['valid'] and not result['error'] and not result['warnings'] and not result['warning_counters']
    assert len(rows)==401 and len(commands)==round(.4/result['control_period_s'])
    assert rows[-1]['elapsed_s']==.4 and result['simulated_seconds']==.4
    substeps=round(result['control_period_s']/.001)
    arm_indices=result['contract']['arm_action_indices']
    gripper=result['gripper_coordinate']
    series={key:[] for key in result['metrics'] if key!='state_rows'}
    for index,row in enumerate(rows):
        assert math.isclose(row['elapsed_s'],index*.001,abs_tol=1e-12)
        assert math.isclose(row['native_time_s'],row['elapsed_s'],abs_tol=1e-10)
        if index==0: continue
        held=commands[(index-1)//substeps]['values_native']
        assert held==row['applied_command_native']
        reference=row['continuous_reference_command_native']
        # Compiled interfaces have direct scalar arm joints at these qpos addresses.
        actual=[row['qpos_native'][i] for i in arm_indices]
        series['arm_held_target'].extend(value-held[i] for value,i in zip(actual,arm_indices))
        series['arm_continuous_reference'].extend(value-reference[i] for value,i in zip(actual,arm_indices))
        jaw=-(gripper['bias0']+gripper['bias1']*row['gripper_actuator_length_native'])/gripper['gain']
        series['gripper_held_target'].append(jaw-held[gripper['action_index']])
        series['gripper_continuous_reference'].append(jaw-reference[gripper['action_index']])
        xyz,goal=row['terminal_link_position_m'],row['reference_terminal_link_position_m']
        series['terminal_position_continuous_reference'].append(math.sqrt(sum((a-b)**2 for a,b in zip(xyz,goal))))
        quat,goalquat=row['terminal_link_quaternion_wxyz'],row['reference_terminal_link_quaternion_wxyz']
        dot=min(1.,max(0.,abs(sum(a*b for a,b in zip(quat,goalquat)))))
        series['terminal_orientation_continuous_reference'].append(2*math.acos(dot))
    for key,values in series.items():
        measured=result['metrics'][key]
        assert measured['count']==len(values)
        rms=math.sqrt(sum(v*v for v in values)/len(values))
        assert math.isclose(measured['rms'],rms,abs_tol=1e-10,rel_tol=1e-10),(result['id'],key,measured['rms'],rms)
        assert math.isclose(measured['max_abs'],max(map(abs,values)),abs_tol=1e-10,rel_tol=1e-10)
        metric_comparisons+=2
    assert result['metrics']['arm_continuous_reference']['unit']=='rad'
    assert result['metrics']['gripper_continuous_reference']['unit']==gripper['unit']
    timing=result['timing']
    assert len(timing['native_steps_s'])==400 and len(timing['control_intervals_s'])==len(commands)
    assert timing['offline_control_budget_exceedances']==sum(v>result['control_period_s'] for v in timing['control_intervals_s'])
    rows_total+=len(rows);commands_total+=len(commands)
identity=sha256((root/'report.json').read_bytes()).hexdigest()
output={'schema':'dvidia.arm-interface-audit/v1','report_sha256':identity,'attempts':54,
        'state_rows':rows_total,'issued_command_events':commands_total,'independent_metric_comparisons':metric_comparisons,
        'source_and_trace_digest_mismatches':0,'command_time_alignment_mismatches':0,
        'offline_control_budget_exceedances':sum(a['timing']['offline_control_budget_exceedances'] for a in report['attempts']),
        'scope':'Read-only numerical/digest audit; not a repeat run or physical validation.'}
Path('runs/arm-interfaces-v01-audit-20261031.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))

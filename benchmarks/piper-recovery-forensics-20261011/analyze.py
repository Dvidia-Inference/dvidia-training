"""Read frozen v0.3 evidence; report observed failures without modifying it."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import mujoco
from dvidia_training.piper_task import PiperTaskConfig, build_task_model

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / 'benchmarks/piper-contact-v03-20261011/receipt'
OUTPUT = Path(__file__).resolve().parent

def load(path):
    return json.loads(gzip.decompress(path.read_bytes()))

def stats(values):
    return {'min': min(values), 'max': max(values), 'mean': statistics.mean(values), 'median': statistics.median(values)} if values else None

def snapshot(row):
    d = row['guard_diagnostic']
    return {k: row[k] for k in ['t_seconds', 'grasp_contact_counts', 'pad_normal_forces_n', 'gripper_joint_position_m', 'table_object_contact_count', 'issued_gripper_position_m']} | {
        'gripper_actuator_force_n': row['actuator_force_native'][6],
        'object_rise_mm': None if d['object_height_delta_m'] is None else d['object_height_delta_m']*1000,
        'tcp_rise_mm': None if d['tcp_probe_height_delta_m'] is None else d['tcp_probe_height_delta_m']*1000,
        'tcp_local_drift_mm': None if d['relative_offset_error_m'] is None else d['relative_offset_error_m']*1000,
        'diagnostic': d,
    }

summaries=[]
for file in sorted((BUNDLE/'attempts').glob('*--history_placement.json.gz')):
    a=load(file); rows=a['trajectory']; events=a['grip_guard']['events']; mu=a['config']['friction']; mass=a['config']['mass_kg']
    data={'id':a['id'], 'scene_id': a['scene_id'], 'source_path':file.relative_to(ROOT).as_posix(), 'source_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
          'friction':mu,'mass_kg':mass,'weight_n':mass*9.81,'success':a['success'], 'metrics':a['metrics'], 'native_gripper':{
              'gain_kp':a['native_servo_gains'][6], 'bias':a['native_servo_bias'][6], 'control_limits_m':a['native_control_limits'][6],
              'force_limits_n':a['native_force_limits'][6], 'note':'Target minimum 0 m and kp40 produce about -0.784 N at a stationary 0.0196 m slide; the ±10 N cap is not reached.'},
          'events':[{'type':e['type'],'reason':e['reason'],'retry_index':e['retry_index'],**snapshot(rows[round(e['t_seconds']/.02)])} for e in events], 'probes':[]}
    for event in events:
        if event['type']!='probe_started': continue
        t=event['t_seconds']; end=next(e for e in events if e['t_seconds']>t+1e-9 and e['type'] in ('alarm','acquisition_qualified'))
        window=[r for r in rows if t-1e-9<=r['t_seconds']<=end['t_seconds']+1e-9]
        normals=[sum(r['pad_normal_forces_n'].values()) for r in window]
        jaws=[r['gripper_joint_position_m'][0] for r in window]
        data['probes'].append({'retry_index':event['retry_index'],'start_s':t,'end_s':end['t_seconds'],'duration_s':end['t_seconds']-t,
            'end_reason':end['reason'],'sample_count':len(window),
            'all_supported':all(r['table_object_contact_count']>0 for r in window),
            'bilateral_fraction':statistics.mean(all(r['grasp_contact_counts'].values()) for r in window),
            'all_jaw_requests_zero':all(r['issued_gripper_position_m']==0 for r in window),
            'normal_sum_n':stats(normals), 'left_normal_n':stats([r['pad_normal_forces_n']['left'] for r in window]),
            'right_normal_n':stats([r['pad_normal_forces_n']['right'] for r in window]),
            'jaw_slide_m':stats(jaws),'jaw_slide_velocity_m_s':stats([r['qvel'][6] for r in window]),
            'gripper_actuator_force_n':stats([r['actuator_force_native'][6] for r in window]),
            'friction_capacity_flat_side_approximation_n':stats([mu*n for n in normals]),
            'friction_capacity_over_weight_approximation':stats([mu*n/(mass*9.81) for n in normals]),
            'final_snapshot':snapshot(window[-1])})
    if not a['success']:
        alarms=[e for e in events if e['type']=='alarm']; data['first_alarm_to_final_abort_s']=next(e['t_seconds'] for e in events if e['type']=='abort')-alarms[0]['t_seconds']
        source_probe=load(file.with_name(file.name.replace('history_placement','probe_only_placement')))
        data['history_matches_probe_only_all_qpos_qvel_commands']=all(x['qpos']==y['qpos'] and x['qvel']==y['qvel'] and x['command']==y['command'] for x,y in zip(rows,source_probe['trajectory']))
        baseline=load(file.with_name(file.name.replace('history_placement','baseline_placement')))
        data['baseline_success']=baseline['success'];data['baseline_max_lift_mm']=baseline['metrics']['max_object_lift_m']*1000
        model=build_task_model(ROOT/'runs/robot-profile-upstream/assets',PiperTaskConfig(**a['config'])); native=mujoco.MjData(model)
        pads=set(sum(a['contact_geometry']['pad_geom_ids'].values(),[])); block=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'task_block_geom')
        first=rows[round(alarms[0]['t_seconds']/.02)];native.qpos[:]=first['qpos'];native.qvel[:]=first['qvel'];native.ctrl[:]=first['native_actuator_controls'];mujoco.mj_forward(model,native)
        contacts=[{'geom_ids':list(map(int,c.geom)),'dim':int(c.dim),'friction':c.friction.tolist(),'normal_z':float(c.frame[2])} for c in native.contact if block in c.geom and pads.intersection(c.geom)]
        data['effective_contact_friction_check']={'method':'Pinned model compiled and mj_forward called at recorded first-alarm qpos/qvel/ctrl, no integration or use of reconstructed normal forces. Contact parameters/frames only.', 'contacts':contacts, 'all_mu_match_scene':all(c['friction'][:2]==[mu,mu] for c in contacts),'max_abs_normal_z':max(abs(c['normal_z']) for c in contacts)}
    summaries.append(data)
result={'schema':'dvidia.piper-v04-failure-forensics/v1','source_report_sha256':hashlib.sha256((BUNDLE/'report.json').read_bytes()).hexdigest(),
        'source_bundle_integrity_sha256':hashlib.sha256((BUNDLE/'integrity.json').read_bytes()).hexdigest(),
        'method':'All 12 history placements and six matched baseline/probe-only failures read from sealed v0.3 evidence. Event snapshots use t/20ms row mapping. Probe statistics inclusive endpoints. No thresholds changed or new integrated episodes run.',
        'important_limits':['μ·ΣN is a flat-side Coulomb capacity approximation, not a measured tangential wrench or global proof that all grasps are infeasible.', 'Recorded normal forces omit contacts ≤.01N; small force contributions below that logging threshold are not included.', 'This is a limitation of pinned position-servo settings + side-pinch geometry, not evidence of PiPER hardware weakness.', 'Existing 12-scene cohort is known development evidence for v0.4, not fresh confirmation.'],
        'history_attempts':summaries}
(OUTPUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
failed=[a for a in summaries if not a['success']]
md=['# Why six v0.3 PiPER placements failed','',
'All six failures occurred during acquisition, before the contact-history carry controller had anything to recover. All alarm snapshots were table-supported; 11 of 12 probe windows remained supported throughout, with one unsupported snapshot during the remaining retry. The object moved at most **0.040–0.148 mm** across each complete episode while the first probe raised the TCP **3.428–3.849 mm**. Bilateral contact occupied 94.7–100% of each probe window; occasional single-sample contact gaps did not lead to acquisition. The retry repeated the same grip and failed again.', '',
'| Scene | First alarm (s) | Object / TCP rise at alarm (mm) | Final abort (s) | Retry overhead (s) | Episode max rise (mm) |',
'|---|---:|---:|---:|---:|---:|']
for a in failed:
    first=a['probes'][0]['final_snapshot'];abort=next(e for e in a['events'] if e['type']=='abort')
    md.append(f"| {a['scene_id']} | {first['t_seconds']:.2f} | {first['object_rise_mm']:.3f} / {first['tcp_rise_mm']:.3f} | {abort['t_seconds']:.2f} | {a['first_alarm_to_final_abort_s']:.2f} | {a['metrics']['max_object_lift_m']*1000:.3f} |")
md+=['','The first alarm is `object_not_following` in all six cases. The second alarm is the same in five; μ=.55, 60 g, layout01 reaches `relative_slip` first. These names indicate which condition won the guard’s priority ordering, not independent ground-truth slip labels. In that last case the object is still table-supported and rises only 0.133 mm.', '',
'## Why more waiting or the same retry cannot add grip force','',
'The upstream gripper is a position servo: `F = 40(target − q) − 5 q̇`, before its ±10 N clamp. At a stationary finger slide around 0.0196 m, the legal minimum target is already zero, so the drive supplies about −0.784 N. The coupled pair carries roughly 0.39 N per side. The actuator force cap is **not** being hit; it is the unchanged position target / gain / grasp combination that establishes this squeeze.', '',
'For approximately vertical object sides, a useful static support estimate is `μ × (Nleft + Nright)`. At μ=.15 this is about 0.118 N against weights 0.392 N (40 g) and 0.589 N (60 g). At μ=.55 it is about 0.43 N against 0.589 N for the failed 60 g cases. The .55 / 40 g successes have just enough approximate static support; .8 / 60 g also qualifies but needs carry recovery.', '',
'The compiled native contact check confirms both tangential friction coefficients equal the scene value; the object’s priority1 supplies them. First-alarm contact normals have |z| ≤0.00204, consistent with nearly vertical side contacts. This check evaluates parameters and geometry at recorded poses only. It does not substitute reconstructed normal forces for recorded values.', '',
'This is a diagnostic approximation, not a proof that every alternate grasp is impossible. It omits sub-threshold logged forces and does not reconstruct tangential wrenches. Hardware squeeze control, reorientation, support-assisted manipulation or a different grasp could alter feasibility and would need their own declared interface and benchmark.', '',
'## What v0.4 can honestly improve','',
'1. Recognize a supported failed probe with persistently loaded pads and the jaw already closed, then stop repeating an unchanged grasp. The old retry adds **2.90–2.92 simulated seconds** before reaching the same refusal. The controller should use only observed history, not hidden mass or friction.',
'2. Keep the acquisition and task thresholds unchanged. A quicker refusal is a useful diagnostic result, but remains a failed placement.',
'3. Improve carry recovery separately, where the six successful cases demonstrate that acquisition is feasible. Remove unnecessary time under the risk condition, measure actual velocity, and avoid making “smaller requested joint delta” synonymous with slower motion.',
'4. Preserve false-positive and regression opportunities: same supported condition could be temporary contact settling. Test transient disturbances and ambiguous windows; retain retry for conditions where a new grasp could plausibly change contact.', '',
'All six failed history trajectories match their probe-only counterpart in every recorded qpos, qvel and command. All six unguarded baselines also fail, with maximum lift only 0.108–0.301 mm. This corroborates an acquisition limitation rather than a v0.3 carry-history regression.', '',
'`analysis.json` contains each event, both probe windows, normal-force/jaw statistics, source checksums, and the native contact-parameter check. `analyze.py` reproduces this analysis from the frozen evidence. This known v0.3 cohort is development data for v0.4; fresh confirmation must be separate.','']
(OUTPUT/'README.md').write_text('\n'.join(md))
print('wrote',OUTPUT/'analysis.json');print('\n'.join(md[:14]))

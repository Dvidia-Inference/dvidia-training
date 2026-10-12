#!/usr/bin/env python3
"""Aggregate the three frozen v0.5 confirmations without modifying any evidence.

Refuses partial cohorts and nonmatching preregistration. By default, validates
portable stored consistency only; --assets also repeats native policy/IK audit.
Do not execute while the confirmation simulations are still running.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

SEEDS = (832847, 227037, 862186)
SOURCE_COMMIT = 'e76b2e41256604d5b4314e45ca45959cab9aa571'
MODES = ('recovery', 'reacquire', 'reacquire_grasp')


def read_json(path):
    return json.loads(path.read_bytes())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def descriptor(path, root):
    data=path.read_bytes()
    return {'path':path.relative_to(root).as_posix(),'bytes':len(data),'sha256':sha256(data).hexdigest()}


def distribution(values):
    xs=sorted(values)
    if any(type(v) not in (float,int) or not math.isfinite(v) for v in xs):
        raise ValueError('Nonfinite descriptive sample')
    q=lambda x:xs[max(0,math.ceil(len(xs)*x)-1)] if xs else None
    return {'count':len(xs),'min':xs[0] if xs else None,'mean':sum(xs)/len(xs) if xs else None,
            'p50':q(.5),'p95':q(.95),'p99':q(.99),'max':xs[-1] if xs else None,
            'quantile_method':'nearest_rank'}


def motion(states, start, end):
    rows=[s for s in states if start-1e-9<=s['t_seconds']<=end+1e-9]
    result={'start_s':start,'end_s':end,'duration_s':end-start,'sample_count':len(rows),
            'scope':'20ms sampled motion; does not bound native-substep peak speed or prove instantaneous arrest.'}
    for name,field in (('tcp','tcp_position_m'),('object','object_position_m')):
        segments=[math.dist(a[field],b[field]) for a,b in zip(rows,rows[1:])]
        speeds=[d/(b['t_seconds']-a['t_seconds']) for d,a,b in zip(segments,rows,rows[1:])]
        result[name]={'path_m':sum(segments),'peak_sampled_speed_m_s':max(speeds,default=None),
            'net_vertical_m':rows[-1][field][2]-rows[0][field][2] if rows else None,
            'max_vertical_from_window_start_m':max((s[field][2]-rows[0][field][2] for s in rows),default=None),
            'min_vertical_from_window_start_m':min((s[field][2]-rows[0][field][2] for s in rows),default=None)}
    return result


def failure_stage(raw):
    if not raw['valid']:
        return 'invalid_native_attempt'
    if raw['success']:
        return 'completed'
    g,m=raw['grip_guard'],raw.get('metrics',{})
    if not g.get('acquisition_qualification_count'):
        return 'acquisition_never_qualified'
    if not m.get('lift_seen'):
        return 'qualified_without_full_60mm_lift'
    if m.get('retention_lost'):
        return 'full_lift_but_retention_lost'
    if not m.get('carry_complete'):
        return 'full_lift_without_retained_supported_arrival'
    return 'supported_arrival_without_final_settled_release_and_retreat_dwell'


def attempt_record(raw, seed):
    m,g,states=raw.get('metrics',{}),raw['grip_guard'],raw['trajectory']
    events=g['events']; probes=[]
    for event in events:
        if event['type']!='second_probe_started':
            continue
        start=event['t_seconds']
        finish=next((e for e in events if e['t_seconds']>=start and e['type'] in ('second_probe_qualified','abort')),None)
        end=finish['t_seconds'] if finish else raw['simulated_seconds']
        rows=[s for s in states if start-1e-9<=s['t_seconds']<=end+1e-9]
        first,last=(rows[0],rows[-1]) if rows else (None,None)
        probes.append({'start_s':start,'end_s':end,'duration_s':end-start,
            'outcome':finish['type'] if finish else 'unresolved','reason':finish['reason'] if finish else None,
            'task_completed':raw['valid'] and raw['success'],
            'actual_motion':motion(states,start,end),
            'max_requested_vertical_rise_m':max((s.get('guard_diagnostic',{}).get('second_probe_requested_rise_m',0.) for s in rows if s.get('guard_diagnostic')),default=None),
            'max_relative_drift_m':max((s['guard_diagnostic']['second_probe_relative_drift_m'] for s in rows
                if s.get('guard_diagnostic') and s['guard_diagnostic'].get('second_probe_relative_drift_m') is not None),default=None),
            'start_original_probe_relative_object_height_m':(first.get('guard_diagnostic') or {}).get('object_height_delta_m') if first else None,
            'end_original_probe_relative_object_height_m':(last.get('guard_diagnostic') or {}).get('object_height_delta_m') if last else None,
            'loaded_bilateral_all_samples':bool(rows) and all(all(s['grasp_contact_counts'].values()) and min(s['pad_normal_forces_n'].values())>.01 for s in rows),
            'unsupported_all_samples':bool(rows) and all(not s['table_object_contact_count'] for s in rows)})
    holds=[]
    for event in events:
        if event['type']!='reassessment_started':
            continue
        start=event['t_seconds']
        finish=next((e for e in events if e['t_seconds']>=start and e['type'] in ('carry_resumed','abort')),None)
        end=finish['t_seconds'] if finish else raw['simulated_seconds']
        holds.append({'start_s':start,'end_s':end,'duration_s':end-start,
            'outcome':finish['type'] if finish else 'unresolved','reason':finish['reason'] if finish else None,
            'includes_second_probe':any(start-1e-9<=p['start_s']<=end+1e-9 for p in probes),
            'actual_motion':motion(states,start,end),
            'scope':'Recovery interval may include active second-probe motion; not all of this interval is stationary holding.'})
    forces=[[s['actuator_force_native'][i] for s in states] for i in range(7)]
    actuator_extrema=[{'actuator_index':i,'unit':'N m' if i<6 else 'N',
        'min':min(v,default=None),'max':max(v,default=None),'max_abs':max((abs(x) for x in v),default=None)} for i,v in enumerate(forces)]
    whole_motion=motion(states,0.,raw['simulated_seconds'])
    abort=next((e for e in events if e['type']=='abort'),None)
    result={'id':str(seed)+':'+raw['id'],'cohort_seed':seed,'scene_id':raw['scene_id'],'variant':raw['variant'],
        'mode':raw['policy_mode'],'control':raw['control'],'friction':raw['config']['friction'],
        'mass_kg':raw['config']['mass_kg'],'layout_index':int(raw['scene_id'].rsplit('layout',1)[1]),
        'valid':raw['valid'],'success':raw['valid'] and raw['success'],
        'failure_stage':failure_stage(raw) if raw['control']=='placement' else None,
        'terminal_reason':raw['reason'],'abort_reason':g.get('abort_reason'),
        'abort_time_s':abort['t_seconds'] if abort else None,
        'completion_time_s':m.get('first_success_time_seconds') if raw['valid'] and raw['success'] else None,
        'milestones':{k:m.get(k) for k in ('grasp_seen','lift_seen','carry_complete','retention_lost','dwell_seconds','max_object_lift_m',
            'max_carry_contact_loss_seconds','carry_contact_loss_events','target_distance_m')},
        'qualification_count':g.get('acquisition_qualification_count',0),'retry_count':g.get('retry_count',0),
        'second_probe_count':g.get('second_probe_count',0),'second_probe_resume_count':g.get('second_probe_resume_count',0),
        'resume_count':g.get('resume_count',0),'holds':holds,'second_probes':probes,
        'whole_episode_sampled_motion':whole_motion,
        'max_pad_normal_force_n':m.get('max_pad_normal_force_n'),
        'max_abs_native_joint_velocity_rad_s':[max((abs(s['joint_velocity_rad_s'][i]) for s in states),default=None) for i in range(6)],
        'actuator_force_extrema':actuator_extrema,
        'unexpected_contact_native_ticks':m.get('unexpected_contact_native_ticks'),
        'warnings':raw.get('warnings',[]),'simulated_seconds':raw['simulated_seconds'],
        'episode_wall_seconds':raw['timing'].get('episode_wall_seconds'),
        'attempt_wall_seconds':raw.get('attempt_wall_seconds'),
        'step_compute_seconds':distribution(raw['timing'].get('control_step_samples_seconds',[])),
        'missed_20ms_windows':sum(x>.02 for x in raw['timing'].get('control_step_samples_seconds',[])),
        'policy_event_types':[e['type'] for e in events],
        'final_object_position_m':states[-1]['object_position_m'] if states else None,
        'final_tcp_position_m':states[-1]['tcp_position_m'] if states else None}
    return result


def totals(rows):
    successes=sum(r['success'] for r in rows)
    return {'attempted':len(rows),'valid':sum(r['valid'] for r in rows),'success':successes,
        'failed_or_invalid':len(rows)-successes,'success_fraction':successes/len(rows) if rows else None,
        'failure_stage_counts':dict(sorted(Counter(r['failure_stage'] for r in rows if not r['success']).items())),
        'abort_reason_counts':dict(sorted(Counter(r['abort_reason'] for r in rows if r['abort_reason']).items())),
        'completion_time_seconds':distribution([r['completion_time_s'] for r in rows if r['completion_time_s'] is not None]),
        'second_probes_started':sum(r['second_probe_count'] for r in rows),
        'second_probes_qualified':sum(r['second_probe_resume_count'] for r in rows),
        'probe_attempts_completed':sum(r['second_probe_count']>0 and r['success'] for r in rows),
        'probe_attempts_failed_or_invalid':sum(r['second_probe_count']>0 and not r['success'] for r in rows),
        'resumed_completed':sum(r['resume_count']>0 and r['success'] for r in rows),
        'resumed_failed_or_invalid':sum(r['resume_count']>0 and not r['success'] for r in rows),
        'unexpected_contact_native_ticks':sum(r['unexpected_contact_native_ticks'] or 0 for r in rows),
        'attempts_with_prohibited_contact':sum(bool(r['unexpected_contact_native_ticks']) for r in rows),
        'attempts_with_warnings':sum(bool(r['warnings']) for r in rows),
        'max_pad_force_n':max((r['max_pad_normal_force_n'] for r in rows if r['max_pad_normal_force_n'] is not None),default=None),
        'max_sampled_tcp_speed_m_s':max((r['whole_episode_sampled_motion']['tcp']['peak_sampled_speed_m_s'] for r in rows
            if r['whole_episode_sampled_motion']['tcp']['peak_sampled_speed_m_s'] is not None),default=None)}


def compare(rows,reference,candidate):
    lookup={(r['cohort_seed'],r['scene_id'],r['mode']):r for r in rows if r['control']=='placement'}
    scenes=sorted({(r['cohort_seed'],r['scene_id']) for r in rows if r['control']=='placement'})
    pairs=[]
    for seed,scene in scenes:
        a,b=(lookup[(seed,scene,m)] for m in (reference,candidate))
        sa,sb=a['success'],b['success']
        pairs.append({'cohort_seed':seed,'scene_id':scene,'reference_success':sa,'candidate_success':sb,
            'reference_valid':a['valid'],'candidate_valid':b['valid'],
            'reference_completion_s':a['completion_time_s'],'candidate_completion_s':b['completion_time_s'],
            'completion_delta_s':b['completion_time_s']-a['completion_time_s'] if sa and sb else None,
            'reference_failure_stage':a['failure_stage'],'candidate_failure_stage':b['failure_stage'],
            'reference_abort_reason':a['abort_reason'],'candidate_abort_reason':b['abort_reason'],
            'common_abort_delta_s':b['abort_time_s']-a['abort_time_s'] if not sa and not sb and a['abort_time_s'] is not None and b['abort_time_s'] is not None else None})
    deltas=[p['completion_delta_s'] for p in pairs if p['completion_delta_s'] is not None]
    gained=[p for p in pairs if not p['reference_success'] and p['candidate_success']]
    lost=[p for p in pairs if p['reference_success'] and not p['candidate_success']]
    return {'reference':reference,'candidate':candidate,'attempted_pairs':len(pairs),
        'common_successes':len(deltas),'gained_successes':len(gained),'lost_successes':len(lost),
        'neither_succeeded':sum(not p['reference_success'] and not p['candidate_success'] for p in pairs),
        'common_success_delta_seconds':distribution(deltas),
        'common_success_faster':sum(d<-1e-8 for d in deltas),'common_success_unchanged':sum(abs(d)<=1e-8 for d in deltas),
        'common_success_slower':sum(d>1e-8 for d in deltas),
        'gained_completion_times_seconds':distribution([p['candidate_completion_s'] for p in gained]),
        'lost_reference_completion_times_seconds':distribution([p['reference_completion_s'] for p in lost]),
        'gained_without_losses':bool(gained) and not lost,
        'time_scope':'Candidate minus reference simulated completion time, common successes only. Gains/losses remain separate.',
        'pairs':pairs}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--assets',type=Path,help='Optional independent native policy/IK audit after all simulations finish')
    args=parser.parse_args(); root=args.training_root.resolve()
    destination=args.output or root/'runs/piper-v05-publication/analysis.json'
    reservation_path=root/'benchmarks/piper-reliability-development-20261011/confirmation-reservation.json'
    installed_path=root/'runs/piper-v05-installed-check.json'
    reservation,installed=read_json(reservation_path),read_json(installed_path)
    if tuple(reservation['seeds'])!=SEEDS or installed.get('verified') is not True or installed.get('source_commit')!=SOURCE_COMMIT:
        raise ValueError('Reservation/installed source differs from planned confirmation')
    # Check every completion marker before importing the benchmark or loading
    # any large trace. Partial fresh cohorts are not provisional aggregate data.
    for seed in SEEDS:
        folder=root/f'runs/piper-reliability-confirmation-{seed}'
        for path in (folder/'report.json',folder/'integrity.json',folder.with_name(folder.name+'-execution.json'),folder.with_name(folder.name+'-summary.json')):
            if not path.is_file() or not path.stat().st_size:
                raise SystemExit('Not all three confirmations are complete; no aggregate produced: '+path.relative_to(root).as_posix())
        execution=read_json(folder.with_name(folder.name+'-execution.json'))
        if execution.get('returncode')!=0 or execution.get('source_commit')!=SOURCE_COMMIT or execution.get('seed')!=seed:
            raise ValueError('Confirmation execution failed or changed frozen source')
    sys.path.insert(0,str(root/'src'))
    from dvidia_training import piper_reliability_benchmark as bench
    if bench.implementation_hashes()!=installed['implementation_sha256'] or tuple(bench.MODES)!=MODES:
        raise ValueError('Analysis inspector differs from frozen installed implementation')
    attempts,cohorts,samples_by_variant=[],[],{}
    for seed in SEEDS:
        folder=root/f'runs/piper-reliability-confirmation-{seed}'
        prereg=root/f'benchmarks/piper-reliability-development-20261011/preregistration/protocol-{seed}.json'
        prereg_value,actual_protocol=read_json(prereg),read_json(folder/'protocol.json')
        if (prereg_value!=actual_protocol or actual_protocol['seed']!=seed or actual_protocol['cohort']!='confirmation'
                or actual_protocol['implementation_sha256']!=installed['implementation_sha256']):
            raise ValueError('Emitted protocol differs from decoded preregistration')
        report,files=bench.load_reliability_bundle(folder,assets=args.assets)
        if report['attempted']!=84 or len(files)!=90:
            raise ValueError('Incomplete frozen three-mode cohort')
        seal=read_json(folder/'integrity.json')
        execution_path=folder.with_name(folder.name+'-execution.json'); execution=read_json(execution_path)
        summary_path=folder.with_name(folder.name+'-summary.json'); summary=read_json(summary_path)
        progress_path=folder.with_name(folder.name+'-progress.log')
        progress_lines=[json.loads(line) for line in progress_path.read_text().splitlines() if line.strip()]
        if not progress_lines or progress_lines[0].get('network_denial_self_check')!='EPERM':
            raise ValueError('Missing successful OS network denial self-check')
        if summary.get('attempted')!=84 or summary.get('summary')!=report['summary']:
            raise ValueError('CLI completion summary differs from sealed report')
        cohort_rows=[]
        for row in report['attempts']:
            raw=json.loads(bench.decode_attempt(files[row['result_file']],seal['files'][row['result_file']]))
            item=attempt_record(raw,seed); cohort_rows.append(item);attempts.append(item)
            windows=raw['timing'].get('control_step_samples_seconds',[])
            samples_by_variant.setdefault(row['variant'],[]).extend(windows)
        inventory=[{'path':name,'bytes':len(data),'sha256':sha256(data).hexdigest()} for name,data in sorted(files.items())]
        cohorts.append({'seed':seed,'attempted':report['attempted'],'valid_attempts':report['valid_attempts'],
            'placements':{m:totals([r for r in cohort_rows if r['control']=='placement' and r['mode']==m]) for m in MODES},
            'controls':{'attempted':sum(r['control']!='placement' for r in cohort_rows),
                        'valid':sum(r['valid'] and r['control']!='placement' for r in cohort_rows),
                        'success':sum(r['success'] and r['control']!='placement' for r in cohort_rows)},
            'paired_comparisons':[compare(cohort_rows,a,b) for a,b in ((MODES[0],MODES[1]),(MODES[0],MODES[2]),(MODES[1],MODES[2]))],
            'report_quality_gates':report['quality_gates'],'report_execution_metrics':report['execution_metrics'],
            'native_environment':report['hardware'],'execution':execution,
            'preregistration':descriptor(prereg,root),'emitted_protocol':descriptor(folder/'protocol.json',root),
            'preregistered_json_exactly_equal':True,'canonical_protocol_sha256':sha256(canonical(actual_protocol)).hexdigest(),
            'protocol_byte_identical':prereg.read_bytes()==files['protocol.json'],
            'report_file':descriptor(folder/'report.json',root),'seal_file':descriptor(folder/'integrity.json',root),
            'execution_file':descriptor(execution_path,root),'cli_summary_file':descriptor(summary_path,root),
            'progress_log_file':descriptor(progress_path,root),
            'storage':{'total_files_including_seal':len(files),'sealed_files_excluding_seal':len(seal['files']),
                'stored_bytes_including_seal':sum(len(b) for b in files.values()),
                'stored_bytes_excluding_seal':sum(v['bytes'] for v in seal['files'].values()),
                'decoded_bytes_excluding_seal':sum(v.get('decoded_bytes',v['bytes']) for v in seal['files'].values()),
                'canonical_complete_inventory_sha256':sha256(canonical(inventory)).hexdigest()},
            'artifact_inventory':inventory})
        del files,raw,report
    if len(attempts)!=252 or sum(r['control']!='placement' for r in attempts)!=144:
        raise ValueError('Aggregate denominator differs from preregistration')
    strata=[]
    for mode in MODES:
        for friction in (.8,.55,.15):
            for mass in (.04,.06):
                rows=[r for r in attempts if r['mode']==mode and r['control']=='placement' and r['friction']==friction and r['mass_kg']==mass]
                strata.append({'mode':mode,'friction':friction,'mass_kg':mass,**totals(rows)})
    compute=[]
    for variant,windows in sorted(samples_by_variant.items()):
        rows=[r for r in attempts if r['variant']==variant]
        compute.append({'variant':variant,'attempted':len(rows),'compute_seconds':distribution(windows),
            'missed_20ms_windows':sum(v>.02 for v in windows),'missed_20ms_fraction':sum(v>.02 for v in windows)/len(windows) if windows else None,
            'per_attempt_p95_seconds':distribution([r['step_compute_seconds']['p95'] for r in rows if r['step_compute_seconds']['p95'] is not None]),
            'episode_wall_seconds':distribution([r['episode_wall_seconds'] for r in rows if r['episode_wall_seconds'] is not None])})
    all_windows=[v for windows in samples_by_variant.values() for v in windows]
    host_path=root/'runs/piper-v05-publication/host-load.json'
    result={'schema':'dvidia.piper-reliability-publication-analysis/v0.5','source_commit':SOURCE_COMMIT,
        'analysis_script':descriptor(Path(__file__).resolve(),root),'reservation':descriptor(reservation_path,root),
        'installed_runtime_check':descriptor(installed_path,root),
        'preregistration_check':'Each complete decoded emitted protocol exactly equals its committed preregistration; original file hashes retained, differing whitespace is not byte identity.',
        'inspection':{'portable_stored_checks_passed':True,'independent_native_policy_audit_repeated':args.assets is not None,
            'benchmark_native_policy_audit':'Each frozen benchmark command calls its native policy/IK audit before returning success; all three execution receipts are checked for returncode0.',
            'scope':'Recorded consistency/decision reconstruction, not physical reintegration, authenticity or hardware qualification.'},
        'seeds':list(SEEDS),'independently_sampled_shared_layouts':6,
        'attempted':len(attempts),'valid_attempts':sum(r['valid'] for r in attempts),
        'placements':{m:totals([r for r in attempts if r['mode']==m and r['control']=='placement']) for m in MODES},
        'controls':{'attempted':144,'valid':sum(r['valid'] and r['control']!='placement' for r in attempts),
                    'success':sum(r['success'] and r['control']!='placement' for r in attempts),
                    'per_variant':{v:{'attempted':len(rs),'valid':sum(r['valid'] for r in rs),'success':sum(r['success'] for r in rs)}
                        for v in sorted({r['variant'] for r in attempts if r['control']!='placement'})
                        for rs in [[r for r in attempts if r['variant']==v]]}},
        'paired_comparisons':[compare(attempts,a,b) for a,b in ((MODES[0],MODES[1]),(MODES[0],MODES[2]),(MODES[1],MODES[2]))],
        'strata':strata,'cohorts':cohorts,
        'compute':{'all_control_windows':distribution(all_windows),'missed_20ms_windows':sum(v>.02 for v in all_windows),
            'missed_20ms_fraction':sum(v>.02 for v in all_windows)/len(all_windows) if all_windows else None,
            'per_variant':compute,'whole_installed_command_wall_seconds':sum(c['execution']['command_wall_seconds'] for c in cohorts),
            'total_simulated_seconds':sum(r['simulated_seconds'] for r in attempts),
            'max_child_lifetime_peak_rss_bytes':max(c['execution']['child_lifetime_peak_rss_bytes'] for c in cohorts),
            'max_preinspection_process_peak_rss_bytes':max(c['native_environment']['process_lifetime_peak_rss_bytes'] for c in cohorts),
            'scope':'Measured heavily loaded host. Raw computation includes native integration and snapshots; wall and RSS include declared broader work. No real-time claim, sensor-to-actuator latency estimate, minimum-RAM requirement or isolated-host speed estimate.'},
        'host_load':{'file':descriptor(host_path,root),'snapshot':read_json(host_path)} if host_path.exists() else None,
        'aggregate_storage':{'files_including_seals':sum(c['storage']['total_files_including_seal'] for c in cohorts),
            'stored_bytes_including_seals':sum(c['storage']['stored_bytes_including_seal'] for c in cohorts),
            'decoded_bytes_excluding_seals':sum(c['storage']['decoded_bytes_excluding_seal'] for c in cohorts)},
        'attempts':attempts,
        'limits':['Three cohorts contain six sampled XY layouts shared by mass/friction strata; 36 placements per mode are not 36 independent real-world trials.',
            'All results are descriptive for this narrow synthetic rigid-block distribution. No IID confidence interval or probability of physical reliability is inferred.',
            'Authored controllers use privileged native observations and an existing native lift latch. No learned skills, sensor robustness, deformation calibration or physical transfer demonstrated.',
            'Completion gains/losses and failures remain in denominators. Common-success timing is conditional and cannot stand in for task reliability.',
            'Probe and recovery motion is sampled at20ms. Holding targets is not immediate physical stopping; a recovery interval can include active second lifting.',
            'Every raw overrun and loaded-host result is retained. No outcome-triggered tuning or substitute reruns are used.']}
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':destination.relative_to(root).as_posix(),'attempted':result['attempted'],
        'placement_successes':{m:v['success'] for m,v in result['placements'].items()},'control_successes':result['controls']['success'],
        'raw_20ms_overruns':result['compute']['missed_20ms_windows']},indent=2))


if __name__=='__main__':
    main()

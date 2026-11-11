#!/usr/bin/env python3
"""Summarize every attempt in the registered capture-latency v02 workload."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import statistics

import numpy as np


def read(path):
    return json.loads(path.read_text())


def timing(values):
    return dict(count=len(values), total_ms=float(sum(values)),
        p50_ms=float(np.percentile(values, 50)), p95_ms=float(np.percentile(values, 95)),
        max_ms=float(max(values)), over_20ms=sum(x>20 for x in values))


def run(root, output):
    root, output = Path(root), Path(output)
    plan = read(output/'plan.json')
    confirmation = [read(root/f'confirmation-{index:02}-{variant}'/'profile.json')
                    for index,variant in enumerate(plan['confirmation']['order'],1)]
    diagnostic = {row['id']:read(root/row['id']/'profile.json') for row in plan['diagnostic']}
    assert all(row['complete'] for row in confirmation+list(diagnostic.values()))
    pairs = []
    for index,(a,b) in enumerate(((0,1),(3,2),(4,5)),1):
        baseline,candidate=confirmation[a],confirmation[b]
        pairs.append(dict(pair=index,baseline_control_seconds=baseline['control']['total_seconds'],
            candidate_control_seconds=candidate['control']['total_seconds'],
            control_time_reduction_pct=100*(1-candidate['control']['total_seconds']/baseline['control']['total_seconds']),
            baseline_finish_seconds=baseline['finish_seconds'],candidate_finish_seconds=candidate['finish_seconds'],
            parity=read(root/f'pair-{index}-parity.json')))
    totals={}
    for variant in ('baseline','candidate'):
        rows=[r for r,v in zip(confirmation,plan['confirmation']['order']) if v==variant]
        totals[variant]=dict(mean_control_seconds=statistics.mean(r['control']['total_seconds'] for r in rows),
            mean_finish_seconds=statistics.mean(r['finish_seconds'] for r in rows),
            mean_total_seconds=statistics.mean(r['total_seconds'] for r in rows),
            full_step=timing([x*1000 for r in rows for x in r['interval_seconds']]),
            mean_capture_bytes=statistics.mean(r['capture_bytes'] for r in rows),
            peak_process_rss_bytes=[r['capture_peak_memory']['bytes'] for r in rows],
            control_cpu_seconds=[r['control_cpu_seconds'] for r in rows])
    windowed={}
    for variant in ('baseline','candidate'):
        path=root/f'windowed-{variant}-client.json'; raw=read(path)
        assert raw['ok'] and not raw['errors'] and raw['step_responses']==250
        trace=raw.pop('trace'); raw.pop('initial_state')
        raw['recording']['output_directory']=f'windowed-{variant}/episode-0001'
        ordinary=[r for r in trace if r['request']['type']=='step' and r['recording']['status']!='finished']
        sealing=[r for r in trace if r['request']['type']=='step' and r['recording']['status']=='finished']
        assert len(ordinary)==249 and len(sealing)==1
        raw.update(raw_receipt_sha256=sha256(path.read_bytes()).hexdigest(),
            ordinary_step_rpc=timing([r['rpc_roundtrip_ms'] for r in ordinary]),
            ordinary_step_server=timing([r['step_compute_wall_seconds']*1000 for r in ordinary]),
            final_step_including_seal_rpc_ms=sealing[0]['rpc_roundtrip_ms'],
            final_step_including_seal_server_ms=sealing[0]['step_compute_wall_seconds']*1000)
        windowed[variant]=raw
    for key in ('client_script_sha256','project_sha256','scene_integrity_sha256','expected_model_sha256'):
        assert windowed['baseline'][key]==windowed['candidate'][key]
    a,b=totals['baseline'],totals['candidate']
    result=dict(schema='dvidia.capture-latency-results/v2',scope=plan['scope'],
        all_attempts_retained=True,captures_completed=12,confirmation_pairs=pairs,confirmation_summary=totals,
        mean_control_reduction_pct=100*(1-b['mean_control_seconds']/a['mean_control_seconds']),
        mean_finalization_reduction_pct=100*(1-b['mean_finish_seconds']/a['mean_finish_seconds']),
        mean_capture_storage_increase_pct=100*(b['mean_capture_bytes']/a['mean_capture_bytes']-1),
        windowed=windowed,windowed_parity=read(root/'windowed-parity.json'),
        diagnostic_parity=read(root/'diagnostic-parity.json'),
        independent_native_replay=read(root/'independent-native-replay.json'),
        raw_local_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
        raw_budget_bytes=plan['maximum_raw_budget_bytes'],free_bytes_after=shutil.disk_usage(root).free,
        limitations=['Three pairs on a shared host are not a reliable performance guarantee.',
            'Total control timing is mixed; no sustained50Hz or hardware reaction bound.',
            'PNG compression changes bytes and size; decoded pixels remain exact.',
            'Diagnostic profiles overlap and include profiler overhead; never pool with confirmation.',
            'NumPy build uses reported BLAS backend; effective thread count unmeasured.',
            'Whole-process RSS excludes a separate Godot process and is not a hardware minimum.',
            'No policy fitting, genuine operator demonstrations or task-success qualification.'],
        physical_robot_ready=False,learned_skill_evaluated=False)
    assert result['raw_local_bytes']<result['raw_budget_bytes']
    for path in sorted(root.glob('*/profile.json')):
        target=output/(path.parent.name+'.json')
        with target.open('xb') as f:f.write(path.read_bytes())
    for variant,receipt in windowed.items():
        with (output/f'windowed-{variant}.json').open('x') as f:
            json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
    with (output/'results.json').open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({key:result[key] for key in ('captures_completed','mean_control_reduction_pct',
        'mean_finalization_reduction_pct','mean_capture_storage_increase_pct','raw_local_bytes')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();run(args.runs,args.output)

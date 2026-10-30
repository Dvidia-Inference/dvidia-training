"""Complete only never-started frozen v0.4 cases after an interrupted session.

This is an external, explicitly deviating recovery report. It never supplies
missing evidence, rewrites the frozen protocol, retries a started attempt, or
creates the original study's confirmation aggregate/report. Four incomplete
attempts remain unevaluable and count as failures in the scheduled denominator.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import multiprocessing
from pathlib import Path
import platform
import time

import numpy as np

from dvidia_training import piper_grip_wait_v04_benchmark as bench
from dvidia_training.piper_imitation_benchmark import _atomic_create

try:
    from . import audit_piper_grip_wait_v04 as independent
except ImportError:
    import audit_piper_grip_wait_v04 as independent

SCHEMA = 'dvidia.piper-grip-wait-interruption-recovery/v1'
INCOMPLETE_CASE = 'confirmation-xy04-g1-nominal'
EXPECTED = {'complete': 72, 'interrupted': 4, 'never_started': 436}


def now(): return datetime.now(timezone.utc).isoformat()


def write(path, value):
    _atomic_create(path, (json.dumps(value, indent=2, allow_nan=False)+'\n').encode())


def inventory(root):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink(): raise ValueError('Study evidence cannot contain symlinks.')
        if path.is_file():
            result[path.relative_to(root).as_posix()] = dict(bytes=path.stat().st_size, sha256=bench.sha(path))
        elif not path.is_dir(): raise ValueError('Unexpected evidence file type.')
    return result


def requests(root, protocol):
    return [(case, variant, Path(root)/'confirmation'/case['id']/variant)
            for case in protocol['cases']['confirmation'] for variant in bench.VARIANTS]


def status(folder):
    if folder.is_symlink(): raise ValueError('Attempt cannot be a symlink.')
    if not folder.exists(): return 'never_started'
    if not folder.is_dir(): raise ValueError('Attempt must be a directory.')
    # An existing empty folder is retained as a started attempt too: there is
    # no reliable evidence that a previous runner never touched it.
    return 'complete' if (folder/'receipt.json').is_file() else 'interrupted'


def verify_seal(root):
    root = Path(root)
    protocol = bench.checked(root)
    frozen = bench.read(root/'frozen.json')
    if (frozen['selected'] != 'wait200' or frozen['protocol_sha256'] != bench.sha(root/'protocol.json')
            or frozen['source_sha256'] != protocol['source_sha256']
            or protocol['candidates'][frozen['selected']]['maximum_wait_s'] != .20):
        raise ValueError('Recovery requires the unchanged frozen 200 ms selection.')
    if bench.sha(root/'development-results.json') != frozen['development_sha256']:
        raise ValueError('Frozen development aggregate changed.')
    for name, digest in frozen['development_evidence'].items():
        if bench.sha(root/name) != digest: raise ValueError('Frozen development evidence changed.')
    if len(requests(root, protocol)) != 512: raise ValueError('Expected 512 scheduled requests.')
    return protocol, frozen


def check_original_inventory(root, plan, *, exact=False):
    current = inventory(root)
    for name, expected in plan['original_inventory'].items():
        if current.get(name) != expected: raise ValueError('Original evidence changed: '+name)
    allowed = {row['relative_path']+'/' for row in plan['attempts'] if row['status'] == 'never_started'}
    for name in set(current)-set(plan['original_inventory']):
        if exact or not any(name.startswith(prefix) for prefix in allowed):
            raise ValueError('Unplanned evidence was added: '+name)
    return current


def prepare(study, output):
    root, output = Path(study), Path(output)
    if output.resolve().is_relative_to(root.resolve()): raise ValueError('Recovery output must be outside the frozen study.')
    if output.exists(): raise ValueError('Use a fresh external recovery output directory.')
    protocol, frozen = verify_seal(root)
    rows = [dict(case_id=case['id'], variant=variant, status=status(folder),
                 relative_path=folder.relative_to(root).as_posix())
            for case, variant, folder in requests(root, protocol)]
    counts = {name: sum(row['status'] == name for row in rows) for name in EXPECTED}
    if counts != EXPECTED: raise ValueError('Interrupted inventory differs from the reviewed 72/4/436 state.')
    interrupted = [row for row in rows if row['status'] == 'interrupted']
    if any(row['case_id'] != INCOMPLETE_CASE for row in interrupted):
        raise ValueError('Unexpected interrupted case; explicit review is required.')
    before = inventory(root)
    allowed_folders = {row['relative_path'] for row in rows if row['status'] != 'never_started'}
    for name in before:
        if name.startswith('confirmation/') and '/'.join(name.split('/')[:3]) not in allowed_folders:
            raise ValueError('Confirmation evidence is outside the frozen roster.')
    plan = dict(schema=SCHEMA, prepared_at_utc=now(), study_name=root.name,
        protocol_sha256=bench.sha(root/'protocol.json'), frozen_sha256=bench.sha(root/'frozen.json'),
        source_sha256=protocol['source_sha256'], driver_sha256=bench.sha(__file__),
        independent_auditor_sha256=bench.sha(independent.__file__),
        dependency_auditor_sha256=bench.sha(independent.previous.__file__),
        selected=frozen['selected'], config=protocol['candidates'][frozen['selected']], workers=4,
        scheduled_attempts=512, original_counts=counts, attempts=rows, original_inventory=before,
        recovery_deviation='Session interruption after 72 complete and four started but incomplete confirmation attempts. Complete these 436 never-started requests only; retain all four incomplete attempts untouched and unevaluable. No protocol, policy, labels, configuration or selection changes.',
        missingness='The four interrupted outcomes are unknown, not observed controller failures. Primary scheduled-denominator analysis conservatively imputes failure for all four; sensitivity removes the entire affected XY group, leaving 31 complete groups.',
        timing_scope='The original partial session and resumed session occur on different dates and shared-host loads. Timing is descriptive by session; no uninterrupted-run, isolated throughput or hardware real-time claim.',
        qualified_robot_skill=False, physical_robot_ready=False)
    output.mkdir(parents=True)
    write(output/'recovery-plan.json', plan)
    _atomic_create(output/'README.md', ('# Interrupted v0.4 recovery\n\n'+plan['recovery_deviation']+'\n\n'+plan['missingness']+'\n\n'+plan['timing_scope']+'\n').encode())
    return plan


def checked(study, output):
    root, output = Path(study), Path(output)
    protocol, frozen = verify_seal(root)
    plan = bench.read(output/'recovery-plan.json')
    if (plan.get('schema') != SCHEMA or plan['driver_sha256'] != bench.sha(__file__)
            or plan['independent_auditor_sha256'] != bench.sha(independent.__file__)
            or plan['dependency_auditor_sha256'] != bench.sha(independent.previous.__file__)
            or plan['protocol_sha256'] != bench.sha(root/'protocol.json')
            or plan['frozen_sha256'] != bench.sha(root/'frozen.json')
            or plan['source_sha256'] != protocol['source_sha256']
            or plan['selected'] != frozen['selected'] or plan['config'] != protocol['candidates'][frozen['selected']]
            or plan['workers'] != 4 or plan['original_counts'] != EXPECTED):
        raise ValueError('Recovery plan or frozen identities changed.')
    expected = [(c['id'], v, p.relative_to(root).as_posix()) for c,v,p in requests(root,protocol)]
    actual = [(r['case_id'],r['variant'],r['relative_path']) for r in plan['attempts']]
    if actual != expected or len(actual) != 512 or len(set(actual)) != 512:
        raise ValueError('Recovery plan roster changed.')
    if {k:sum(r['status']==k for r in plan['attempts']) for k in EXPECTED} != EXPECTED:
        raise ValueError('Recovery plan classifications changed.')
    if any(r['case_id'] != INCOMPLETE_CASE for r in plan['attempts'] if r['status']=='interrupted'):
        raise ValueError('Recovery plan interrupted roster changed.')
    for row in plan['attempts']:
        prefix=row['relative_path']+'/'
        names={name[len(prefix):] for name in plan['original_inventory'] if name.startswith(prefix)}
        expected_status='complete' if 'receipt.json' in names else 'interrupted' if names else 'never_started'
        if row['status']!=expected_status:
            raise ValueError('Recovery classifications differ from the original hashed evidence inventory.')
    return protocol, frozen, plan


def never_started_jobs(root, assets, protocol, plan):
    jobs = []
    for (case, variant, folder), row in zip(requests(root,protocol),plan['attempts'],strict=True):
        if row['status'] == 'never_started':
            if folder.exists() or folder.is_symlink():
                raise ValueError('A planned new attempt has already started; it will never be retried.')
            jobs.append((Path(root), assets, case, variant, folder, plan['config']))
    if len(jobs) != 436: raise ValueError('Only 436 never-started requests may run.')
    return jobs


def run(study, output, assets):
    root, output = Path(study), Path(output)
    with bench.old_cli._locked(root):
        protocol, _, plan = checked(root,output)
        check_original_inventory(root,plan,exact=True)
        jobs = never_started_jobs(root,assets,protocol,plan)
        import mujoco
        first=next(row for row in plan['attempts'] if row['status']=='complete')
        previous_platform=bench.load_episode(root/first['relative_path']/'episode.json.gz')['platform']
        current_platform=dict(system=platform.system(),machine=platform.machine(),python=platform.python_version(),
            numpy=np.__version__,mujoco=mujoco.__version__)
        if current_platform!=previous_platform:
            raise ValueError('Native dependency/platform versions differ from the retained original session.')
        intent = dict(schema=SCHEMA, plan_sha256=bench.sha(output/'recovery-plan.json'),
            started_at_utc=now(), workers=4, scheduled_new_attempts=len(jobs),
            driver_sha256=bench.sha(__file__), platform=current_platform,
            platform_matches_original_session=True)
        # Exclusive creation means an interrupted recovery itself is never
        # silently restarted and its newly partial attempts cannot be retried.
        write(output/'execution-intent.json',intent)
        began=time.perf_counter()
        with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
            rows=list(pool.map(bench._job,jobs,chunksize=1))
        checked(root,output); after=check_original_inventory(root,plan)
        result=dict(schema=SCHEMA,plan_sha256=bench.sha(output/'recovery-plan.json'),
            execution_intent_sha256=bench.sha(output/'execution-intent.json'),
            completed_at_utc=now(), wall_seconds=time.perf_counter()-began,
            new_attempts=len(rows), rows=rows, final_inventory=after,
            original_inventory_unchanged=True, all_original_partial_attempts_unchanged=True)
        write(output/'execution-result.json',result)
        return result


def _critical(df,coverage=.975):
    # Numerical inversion avoids adding a SciPy dependency to a local CPU tool.
    # Composite Simpson integration is deterministic, with 4096 subdivisions.
    constant=math.exp(math.lgamma((df+1)/2)-math.lgamma(df/2))/math.sqrt(df*math.pi)
    def cdf(x):
        n=4096; h=x/n
        def density(t):return constant*(1+t*t/df)**(-(df+1)/2)
        return .5+h/3*(density(0)+density(x)+math.fsum((4 if i%2 else 2)*density(i*h) for i in range(1,n)))
    low,high=0.,10.; target=(1+coverage)/2
    for _ in range(60):
        middle=(low+high)/2
        if cdf(middle)<target:low=middle
        else:high=middle
    return (low+high)/2


def contrasts(rows,*,drop_incomplete=False):
    groups={}
    for row in rows:groups.setdefault(tuple(row['start_xy']),[]).append(row)
    if len(groups)!=32:raise ValueError('Expected 32 paired XY groups.')
    if drop_incomplete:groups={key:value for key,value in groups.items() if not any(r['interrupted_unevaluable'] for r in value)}
    expected=31 if drop_incomplete else 32
    if len(groups)!=expected or any(len(group)!=16 for group in groups.values()):raise ValueError('Incomplete paired group roster.')
    critical=bench.old_cli.T975_DF31 if expected==32 else _critical(30)
    values=[]
    for other in ('clock','fixed'):
        differences=[]
        for group in groups.values():
            own={variant:[r for r in group if r['variant']==variant] for variant in bench.VARIANTS}
            if any(len(v)!=4 for v in own.values()):raise ValueError('Expected four cases per controller and XY group.')
            differences.append(sum(bool(r['success']) for r in own['guarded'])/4-sum(bool(r['success']) for r in own[other])/4)
        mean=float(np.mean(differences));half=float(critical*np.std(differences,ddof=1)/math.sqrt(expected))
        values.append(dict(comparator=other,group_differences=differences,mean_difference=mean,interval=[mean-half,mean+half],
            independent_groups=expected,degrees_of_freedom=expected-1,t_critical=critical,interval_coverage=.975,
            simultaneous_coverage=.95,method='Paired whole-XY t interval, Bonferroni for two contrasts.',
            missingness='Drop entire incomplete XY group.' if drop_incomplete else 'Four interrupted outcomes imputed as failures; observed scores unchanged.'))
    return values


def aggregate(rows):
    if len(rows)!=512 or len({(r['case_id'],r['variant']) for r in rows})!=512:raise ValueError('Expected 512 unique scheduled outcomes.')
    result={}
    for variant in bench.VARIANTS:
        own=[r for r in rows if r['variant']==variant]
        measured=[r for r in own if not r['interrupted_unevaluable']]
        if len(own)!=128 or len(measured)!=127:raise ValueError('Each controller must retain one interrupted and 127 scored attempts.')
        result[variant]=dict(scheduled=128,completed_scored=len(measured),
            interrupted_unevaluable=1,success=sum(bool(r['success']) for r in measured),
            **{key:sum(bool(r[key]) for r in measured) for key in ('valid','raw_strict_success','aborted','acquisition_fail','retention_fail','critical_contact')},
            by_condition={condition:dict(scheduled=64,completed_scored=sum(r['condition']==condition for r in measured),
                interrupted_unevaluable=sum(r['condition']==condition and r['interrupted_unevaluable'] for r in own),
                success=sum(r['condition']==condition and bool(r['success']) for r in measured)) for condition in ('nominal','post_grip')})
        result[variant]['timing_by_session']={}
        for session in ('original','resumed'):
            sample=[r for r in measured if r['session']==session]
            timing={target:bench.old_cli._timing([x for r in sample for x in r['timing'][field]])
                    for target,field in (('inference','inference_samples_s'),('full_step','control_step_samples_s'))}
            timing['episode_wall']=bench.old_cli._timing([r['timing']['wall_seconds'] for r in sample])
            result[variant]['timing_by_session'][session]=dict(attempts=len(sample),**timing)
    return result


_AUDIT_CONTEXT=None


def _initialize(context):
    global _AUDIT_CONTEXT
    independent._initialize(context)
    _AUDIT_CONTEXT=context


def _read_only_job(request):
    phase,candidate,case,variant,folder,config=request
    # Both paths require a complete receipt before being called. The frozen
    # execute reuse branch independently checks exact inventory, identity and
    # native scores; the auditor separately reconstructs causal decisions.
    if not (folder/'receipt.json').is_file():raise ValueError('Read-only auditing never starts physics.')
    audited=independent._attempt(request,independent._CONTEXT)
    row=bench.execute(_AUDIT_CONTEXT['root'],None,case,variant,folder,config)
    if any(audited[k]!=row[k] for k in ('valid','raw_strict_success','success','aborted')):
        raise ValueError('Independent audit and retained strict summary differ.')
    return row,audited


def report(study,output):
    root,output=Path(study),Path(output)
    protocol,frozen,plan=checked(root,output)
    before=check_original_inventory(root,plan)
    execution=bench.read(output/'execution-result.json')
    if (execution.get('plan_sha256')!=bench.sha(output/'recovery-plan.json') or execution.get('new_attempts')!=436
            or execution.get('final_inventory')!=before):raise ValueError('Completed recovery receipt or final inventory differs.')
    teacher,intervention=bench.old._options()
    from dataclasses import asdict
    context=dict(root=root,protocol=protocol,protocol_hash=bench.sha(root/'protocol.json'),frozen_hash=bench.sha(root/'frozen.json'),
        base=bench.read(root/'base-model.json'),jaw=bench.read(root/'jaw-model.json'),model=bench.read(root/'selected.json'),
        teacher=asdict(teacher),intervention=intervention)
    all_requests=list(independent._requests(root,protocol,'all',frozen))
    partial={r['relative_path'] for r in plan['attempts'] if r['status']=='interrupted'}
    audit_requests=[request for request in all_requests if request[4].relative_to(root).as_posix() not in partial]
    if len(audit_requests)!=556:raise ValueError('Expected 48 development and 508 confirmation native audits.')
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),initializer=_initialize,initargs=(context,)) as pool:
        pairs=list(pool.map(_read_only_job,audit_requests,chunksize=1))
    measured={};audit_rows=[]
    original={r['relative_path'] for r in plan['attempts'] if r['status']=='complete'}
    for request,(row,audited) in zip(audit_requests,pairs,strict=True):
        audit_rows.append(audited)
        if request[0]=='confirmation':
            row.update(interrupted_unevaluable=False,outcome_status='completed_scored',session='original' if audited['relative_path'] in original else 'resumed')
            measured[(row['case_id'],row['variant'])]=row
    rows=[]
    for case,variant,folder in requests(root,protocol):
        key=(case['id'],variant)
        if folder.relative_to(root).as_posix() in partial:
            rows.append(dict(case_id=case['id'],start_xy=case['start_xy'],goal_xy=case['goal_xy'],condition=case['condition'],variant=variant,
                interrupted_unevaluable=True,outcome_status='interrupted_unevaluable',session='original',
                valid=False,success=False,success_is_failure_imputation=True,raw_strict_success=None,aborted=None,
                receipt_sha256=None,score_reason='session_interrupted_no_complete_evidence'))
        else:rows.append(measured[key])
    if sum(r['audit_mismatches'] for r in audit_rows):raise ValueError('Independent native audit found mismatches; do not publish accuracy claims.')
    after=check_original_inventory(root,plan)
    if before!=after:raise ValueError('Study evidence changed during read-only reporting.')
    value=dict(schema=SCHEMA,study_schema=bench.SCHEMA,protocol_sha256=plan['protocol_sha256'],frozen_sha256=plan['frozen_sha256'],
        recovery_plan_sha256=bench.sha(output/'recovery-plan.json'),execution_result_sha256=bench.sha(output/'execution-result.json'),
        driver_sha256=bench.sha(__file__),independent_auditor_sha256=bench.sha(independent.__file__),
        selected=plan['selected'],config=plan['config'],development=frozen['ranking'],
        scheduled_attempts=512,completed_scored_attempts=508,interrupted_unevaluable_attempts=4,
        valid_scored_attempts=sum(bool(r['valid']) for r in rows),all_512_valid=False,
        original_clean_confirmation_complete=False,protocol_deviation=plan['recovery_deviation'],missingness=plan['missingness'],
        variants=aggregate(rows),contrasts=contrasts(rows),complete_xy_sensitivity=contrasts(rows,drop_incomplete=True),
        audit=dict(audited_attempts=556,development_attempts=48,confirmation_attempts=508,unevaluable_attempts_not_audited=4,audit_mismatches=0,rows=audit_rows),
        rows=rows,timing_scope=plan['timing_scope'],original_inventory_unchanged=True,all_original_partial_attempts_unchanged=True,
        qualified_robot_skill=False,physical_robot_ready=False,limitations=protocol['limitations'])
    write(output/'recovery-report.json',value)
    write(output/'final-evidence-inventory.json',dict(schema=SCHEMA,files=after,original_inventory_unchanged=True))
    return value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','report'))
    parser.add_argument('--study',required=True);parser.add_argument('--output',required=True);parser.add_argument('--assets')
    args=parser.parse_args(argv)
    if args.command=='prepare':result=prepare(args.study,args.output)
    elif args.command=='run':
        if not args.assets:parser.error('run requires --assets')
        result=run(args.study,args.output,args.assets)
    else:result=report(args.study,args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','audit','final_inventory','original_inventory','attempts','source_sha256')},indent=2))


if __name__=='__main__':main()

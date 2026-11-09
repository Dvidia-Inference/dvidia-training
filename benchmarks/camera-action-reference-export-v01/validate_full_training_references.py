"""One bounded metadata-only training reference validation; no fitting/physics.

Build two exact-reference artifacts once, validate each three times, preserve
parent dataset identities. Original frames, query bodies, native arrays and
trained models are prohibited. Original data integrity is inherited from the
independently pinned parent cache; metadata-only checks are not raw revalidation.
"""
from datetime import datetime, timezone
import argparse
from hashlib import sha256
import gc
import json
import os
from pathlib import Path
import platform
try:
    import resource
except ImportError:  # Windows has no resource module.
    resource = None
import shutil
import subprocess
import sys
import time

import numpy as np
from dvidia_training import camera_action_cache as cache
from dvidia_training import camera_action_export as export
from dvidia_training import camera_action_model as model
from dvidia_training import camera_action_reference_export as ref

parser = argparse.ArgumentParser(description=__doc__)
for name in ('source-root', 'study-root', 'parent-cache-root', 'output', 'python-env', 'execution-cwd'):
    parser.add_argument('--'+name, required=True, type=Path)
locations = parser.parse_args()
if any(not path.is_absolute() for path in vars(locations).values()):
    parser.error('Every path must be absolute.')
WORKTREE = locations.source_root.resolve()
STUDY = locations.study_root.resolve()
PARENTS = locations.parent_cache_root.resolve()
OUTPUT = locations.output.resolve()
VENV = locations.python_env.resolve()
RUN_CWD = locations.execution_cwd.resolve()
PROOFS = OUTPUT.parent
if RUN_CWD == WORKTREE or WORKTREE in RUN_CWD.parents:
    parser.error('Use an execution directory outside the source checkout.')
COMMIT = 'a9340b7d00bdd0618421027807faad247f9be99d'
HELPER_SHA = 'ccf6f74f6100721868ccece67e5b27dae066c1b644c8297efe111125d9517753'
SOURCE_SHA = 'e5b26c86013732007c483a23ad7218271980f0f03563c4ffd95c01ad87fd3041'
PARENT_SOURCE_SHA = '0ab01145c66d651eaca96d763b4bec1706327f2cdd4bd8665d05056cbdd6f677'
PROTOCOL_SHA = '576851e321034a0f6d3cb4d0818b3a5ca97975fcc6f2c7fa30b07ffd203a2d76'
PINS = {
    'demonstrations': {'manifest_sha256': '8acd728fcc8ed637b58138c3bd7466432a588fa439acefde4b3c90fa1adfdbaf',
        'dataset_sha256': '07b175f78e4e77cbdde9a461a4bcae4dcb0db7d389ee4f2e702ff28db8bb3dfc',
        'stage_sha256': '5d57eac599f1686d2bdb15bb53ceebe09e899ec71dfe87039aa85de7f853c06b',
        'episodes': 64, 'samples': 64000, 'label_source': 'issued'},
    'corrections': {'manifest_sha256': '0997805831dfaacfe38d87468587f58c4e0b3f748165816e2167c4376676b96b',
        'dataset_sha256': '0561f8793ac6f68318d28c99e15ee8755c85412f3e34c990e1c4f0da4a5c1789',
        'stage_sha256': 'ae0a5653f728f147f929325c60318cb3d164c6d97d846fb96ac07ee9d27ad5cb',
        'episodes': 32, 'samples': 32000, 'label_source': 'teacher_query'},
}
COMBINED_DATASET_SHA = 'b500b3ba348bba44cc79e377266f92910bd7e0165c36cd8b2fe260909b30a2f9'
COMBINED_PROVENANCE_SHA = '52cbeda62e3aa50e8cd6a34688fba7583f9fcec927cb0eb434a6907c99695f4f'
RESERVE = 30*1024**3
MAX_OUTPUT = 32*1024**2
ALLOWED_STUDY_READS = {STUDY/p for p in ('protocol.json', 'registration.json',
    'stages/demonstrations.json', 'stages/demonstrations.integrity.json',
    'stages/corrections.json', 'stages/corrections.integrity.json')}
OBSERVED_STUDY_READS = set()


def peak_rss_bytes():
    if resource is None or (sys.platform != 'darwin' and not sys.platform.startswith('linux')):
        return None  # Unavailable telemetry, never a fabricated zero.
    observed = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return observed if sys.platform == 'darwin' else observed * 1024


def now(): return datetime.now(timezone.utc).isoformat()


def identity(path):
    body=path.read_bytes()
    return {'bytes':len(body), 'sha256':sha256(body).hexdigest()}


def write(path, document):
    with path.open('x') as stream:
        json.dump(document,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def status(state):
    pending=OUTPUT/'status.pending.json'
    pending.write_text(json.dumps(state,sort_keys=True,indent=2)+'\n')
    pending.replace(OUTPUT/'status.json')


def progress(state,stage,**fields):
    row={'at':now(),'stage':stage,**fields}
    with (OUTPUT/'progress.jsonl').open('a') as stream:
        stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    state.update(updated_at=row['at'],last_stage=stage,free_bytes=shutil.disk_usage(OUTPUT).free)
    if state['free_bytes']<RESERVE: raise ValueError('Preserve 30 GiB reserve.')
    status(state);print(json.dumps(row,sort_keys=True),flush=True)


def guard(event,args):
    if event=='import' and (args[0]=='mujoco' or args[0].startswith('mujoco.')):
        raise RuntimeError('No native physics import in metadata validation.')
    if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)): return
    path=Path(os.fsdecode(args[0])).resolve()
    if STUDY==path or STUDY in path.parents:
        mode=args[1];flags=args[2]
        if ((isinstance(mode,str) and any(x in mode for x in 'wax+'))
                or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))):
            raise RuntimeError('Original study writes forbidden.')
        if path not in ALLOWED_STUDY_READS:
            raise RuntimeError('Unregistered/raw/held-out/model read forbidden: '+str(path))
        OBSERVED_STUDY_READS.add(str(path.relative_to(STUDY)))


def args_for(stage):
    return dict(expected_parent_manifest_sha256=PINS[stage]['manifest_sha256'],
        expected_parent_source_sha256=PARENT_SOURCE_SHA, expected_protocol_sha256=PROTOCOL_SHA,
        expected_stage=stage,expected_split='train',expected_source_sha256=SOURCE_SHA)


def preflight():
    if Path(sys.prefix)!=VENV or Path(ref.__file__).resolve()!=WORKTREE/'src/dvidia_training/camera_action_reference_export.py':
        raise ValueError('Use reviewed interpreter and exact reference source checkout.')
    if Path.cwd()!=RUN_CWD or os.environ.get('PYTHONPATH')!=str(WORKTREE/'src'):
        raise ValueError('Use the prepared outside-checkout command and sole pinned source path.')
    if ref.source_digest()!=SOURCE_SHA or identity(Path(ref.__file__))['sha256']!=HELPER_SHA:
        raise ValueError('Exact source binding differs.')
    committed=subprocess.check_output(['git','-C',str(WORKTREE),'show',COMMIT+':src/dvidia_training/camera_action_reference_export.py'])
    if sha256(committed).hexdigest()!=HELPER_SHA or cache.source_digest()!=PARENT_SOURCE_SHA:
        raise ValueError('Committed helper or unchanged parent source differs.')
    if (STUDY/'worker.lock').exists(): raise ValueError('Active original study writer.')
    protocol,protocol_sha=cache._json(STUDY/'protocol.json')
    if protocol_sha!=PROTOCOL_SHA: raise ValueError('Protocol pin differs.')
    rosters={}
    for stage,pin in PINS.items():
        manifest,digest=cache._json(PARENTS/(stage+'-cache')/'manifest.json')
        if digest!=pin['manifest_sha256'] or manifest['dataset_sha256']!=pin['dataset_sha256']:
            raise ValueError('Independent parent pin differs.')
        rosters[stage]=[row['attempt_id'] for row in manifest['episodes']]
        registered=[r['id'] for r in protocol['rosters'][stage] if r['case']['split']=='train']
        if (rosters[stage]!=registered or len(rosters[stage])!=pin['episodes']
                or manifest['sample_count']!=pin['samples'] or manifest['binding']['split']!='train'
                or manifest['binding']['stage']!=stage or manifest['blind'] is not False):
            raise ValueError('The complete registered training roster must be preserved.')
        stage_doc,stage_sha=cache._json(STUDY/'stages'/(stage+'.json'))
        if stage_sha!=pin['stage_sha256']: raise ValueError('Stage byte pin differs.')
        for row,source in zip(manifest['episodes'],manifest['provenance']['episodes']):
            if (row['path']!=f'episodes/{stage}/{row["attempt_id"]}'
                    or row['label_source']!=pin['label_source'] or source['sample_count']!=1000):
                raise ValueError('Episode label/row contract differs.')
            for name in ('recording.json','integrity.json','rollout.json'):
                ALLOWED_STUDY_READS.add(STUDY/row['path']/name)
    if len(ALLOWED_STUDY_READS)!=294: raise ValueError('Expected only six study/stage and288 training metadata files.')
    return rosters


def match(dataset,expected,samples):
    _,_,groups,provenance,digest=model._dataset(dataset,blind=False)
    if digest!=expected or len(groups)!=samples or len(set(groups))!=32:
        raise ValueError('The parent data identity changed.')
    return dict(dataset_sha256=digest,samples=samples,groups=32,provenance_sha256=model.digest(provenance))


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink() or STUDY in OUTPUT.parents:
        raise ValueError('Fresh output required; retain interrupted prefixes.')
    if shutil.disk_usage(PROOFS).free<RESERVE+MAX_OUTPUT:
        raise ValueError('Require 30 GiB reserve plus32MiB bounded outputs.')
    OUTPUT.mkdir(mode=0o700)
    state={'schema':'dvidia.reference-export-validation-process/v1','state':'running','pid':os.getpid(),
        'started_at':now(),'workers':1,'native_runs':0,'fitting_runs':0,'raw_reverification':False}
    write(OUTPUT/'worker.lock',{'pid':os.getpid(),'started_at':state['started_at']})
    status(state)
    started=time.perf_counter()
    try:
        sys.addaudithook(guard)
        rosters=preflight()
        write(OUTPUT/'independent-input-pins.json',dict(commit=COMMIT,helper_sha256=HELPER_SHA,
            source_sha256=SOURCE_SHA,parent_source_sha256=PARENT_SOURCE_SHA,protocol_sha256=PROTOCOL_SHA,
            parents=PINS,ordered_attempts=rosters,script=identity(Path(__file__)),
            combined_dataset_sha256=COMBINED_DATASET_SHA,combined_provenance_sha256=COMBINED_PROVENANCE_SHA))
        progress(state,'preflight',episodes=96,samples=96000,allowed_original_metadata_files=294)
        builds={}
        for stage,pin in PINS.items():
            began=time.perf_counter()
            seal=ref.build_reference_export(STUDY,PARENTS/(stage+'-cache'),OUTPUT/(stage+'-references'),**args_for(stage))
            seconds=time.perf_counter()-began
            if seal['sample_count']!=pin['samples'] or seal['parent_dataset_sha256']!=pin['dataset_sha256']:
                raise ValueError('Reference seal has unexpected parent/sample identity.')
            files={p.name:identity(p) for p in (OUTPUT/(stage+'-references')).iterdir()}
            builds[stage]=dict(seconds=seconds,pins=seal,files=files,bytes=sum(v['bytes'] for v in files.values()))
            write(OUTPUT/(stage+'-pins.json'),seal)
            progress(state,'reference_export_built',reference_stage=stage,seconds=seconds,bytes=builds[stage]['bytes'])
            gc.collect()
        if sum(b['bytes'] for b in builds.values())>MAX_OUTPUT: raise ValueError('Reference outputs exceed declared bound.')
        rounds=[]
        for index in range(3):
            datasets=[]; loads=[]
            for stage,pin in PINS.items():
                began=time.perf_counter()
                loaded=ref.load_reference_export(OUTPUT/(stage+'-references'),PARENTS/(stage+'-cache'),
                    expected_manifest_sha256=builds[stage]['pins']['manifest_sha256'],**args_for(stage))
                seconds=time.perf_counter()-began
                if loaded['previous_references'].shape!=(pin['samples'],7): raise ValueError('Reference row count changed.')
                datasets.append(loaded['parent_dataset'])
                began=time.perf_counter(); parent_check=match(datasets[-1],pin['dataset_sha256'],pin['samples'])
                loads.append(dict(stage=stage,seconds=seconds,reference_rows=pin['samples'],
                    parent_identity=parent_check,parent_identity_check_seconds=time.perf_counter()-began))
                del loaded
            began=time.perf_counter(); combined=export.combine_datasets(datasets);combine_seconds=time.perf_counter()-began
            began=time.perf_counter(); check=match(combined,COMBINED_DATASET_SHA,96000);golden_seconds=time.perf_counter()-began
            if check['provenance_sha256']!=COMBINED_PROVENANCE_SHA: raise ValueError('Combined parent provenance differs.')
            row=dict(index=index+1,loads=loads,parent_combine_seconds=combine_seconds,
                combined_parent_identity=check,combined_parent_golden_check_seconds=golden_seconds)
            rounds.append(row);progress(state,'validated_load',**row)
            del combined,datasets;gc.collect()
        elapsed=time.perf_counter()-started
        result=dict(schema='dvidia.reference-export-full-training-validation/v1',passed=True,qualified=False,
            commit=COMMIT,source_sha256=SOURCE_SHA,script=identity(Path(__file__)),builds=builds,load_rounds=rounds,
            rows=96000,episodes=96,native_runs=0,fitting_runs=0,raw_source_reverified=False,
            original_metadata_files_read=sorted(OBSERVED_STUDY_READS),original_metadata_files_read_count=len(OBSERVED_STUDY_READS),
            wall_seconds=elapsed,peak_process_rss_bytes=peak_rss_bytes(),
            hardware=dict(platform=platform.platform(),machine=platform.machine(),python=platform.python_version(),numpy=np.__version__,shared_host=True),
            timing_scope='Metadata/source/parent validation and label derivation, plus three derived loads. Load timings include parent-cache validation and endpoint recomputation; parent identity checks, combination and combined golden checks are separate. No strict raw export comparator, training, inference or simulation timing.',
            scope='Exact float64 reset/prior normal-reference ancestry and all96k twenty-update recorded normal endpoints verified against sealed metadata. Full original native microstep trajectories are not reread. Mixed feasible labels are counterfactual normal-reference endpoints toward teacher suggestions, not claims of executed teacher control.',
            limitations=['Original raw verification inherited from independently pinned parent caches; source trust/exclusive-access assumptions retained.',
                'No original images, depth, teacher-query bodies, native traces/NPZ, held-out episodes or trained models read.',
                'No model-head/inference integration, fitting, physical execution, manipulation improvement or new qualification.'])
        write(OUTPUT/'result.json',result)
        state.update(state='complete',finished_at=now(),exit_code=0,result=identity(OUTPUT/'result.json'))
        status(state);(OUTPUT/'worker.lock').unlink()
        print(json.dumps({'state':'complete','result':state['result']},sort_keys=True),flush=True)
    except BaseException as error:
        state.update(state='failed',finished_at=now(),exit_code=1,error_type=type(error).__name__,error=str(error))
        status(state)
        if (OUTPUT/'worker.lock').exists(): (OUTPUT/'worker.lock').unlink()
        raise


if __name__=='__main__': main()

import argparse,json,os,resource,subprocess,time
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);args=parser.parse_args()
root=Path('/Users/roguedev/dvidia.org/dvidia-training')
reservation=json.loads((root/'benchmarks/piper-reliability-development-20261011/confirmation-reservation.json').read_text())
if args.seed not in reservation['seeds']:raise SystemExit('Only reserved fresh seeds are permitted')
name=f'piper-reliability-confirmation-{args.seed}';output=root/'runs'/name
if output.exists():raise SystemExit('Fresh confirmation output required; preserve previous attempts')
source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
status=subprocess.check_output(['git','status','--porcelain','--','src','pyproject.toml'],cwd=root,text=True)
if status:raise SystemExit('Freeze and commit all runtime source paths before confirmation')
installed=json.loads((root/'runs/piper-v05-installed-check.json').read_text())
if not installed['verified'] or source!=installed['source_commit']:raise SystemExit('Installed source differs from frozen commit')
env=os.environ.copy();env.pop('PYTHONPATH',None);env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
cmd=['sandbox-exec','-p','(version 1)(allow default)(deny network*)',str(root/'runs/piper-v05-wheel-env/bin/python'),'/private/tmp/dvidia-piper-v05-offline-worker.py','benchmark','--assets',str(root/'runs/robot-profile-upstream/assets'),'--output',str(output),'--seed',str(args.seed),'--cohort','confirmation']
started=time.perf_counter()
with (root/'runs'/f'{name}-summary.json').open('xb') as out,(root/'runs'/f'{name}-progress.log').open('xb') as err:
 result=subprocess.run(cmd,cwd=root,env=env,stdout=out,stderr=err)
elapsed=time.perf_counter()-started
receipt={'schema':'dvidia.local-benchmark-execution/v1','source_commit':source,'cohort':'confirmation','seed':args.seed,'returncode':result.returncode,'execution':'Installed wheel built from a committed source archive; runtime source paths clean; OS process-level network denial with EPERM socket self-check; local pinned assets; PYTHONPATH unset','threads':{'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},'command_wall_seconds':elapsed,'child_lifetime_peak_rss_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,'memory_scope':'macOS RUSAGE_CHILDREN high-water RSS including benchmark and native policy inspection; preceding Git children also belong to child lifetime','wall_scope':'Whole installed command including startup, 84 native attempts, evaluation, compression, serialization, portable inspection and native policy/IK audit. Excludes wheel installation, asset provisioning, browser rendering and publication. Host load uncontrolled.'}
(root/'runs'/f'{name}-execution.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2));raise SystemExit(result.returncode)

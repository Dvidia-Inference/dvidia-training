from pathlib import Path
import gzip,json,hashlib,sys
from dataclasses import replace
from unittest.mock import patch
from dvidia_training import piper_task as task,piper_contact_recovery as recovery
from dvidia_training.piper_reacquisition import ReacquisitionPiperTaskPolicy
root=Path('runs/piper-v05-development/lift')
output=root/sys.argv[1]
output.mkdir(exist_ok=True)
reference=Path(sys.argv[3] if len(sys.argv)>3 else 'runs/piper-recovery-confirmation-941827/attempts')
selected=sorted(reference.glob('*--recovery_placement.json.gz'))
if len(sys.argv)>2:
 selected=[p for p in selected if sys.argv[2] in p.name]
summary=[]
for path in selected:
 if (output/path.name).exists():
  continue
 old=json.loads(gzip.decompress(path.read_bytes()))
 config=task.PiperTaskConfig(**old['config'])
 raw=task.run_episode('runs/robot-profile-upstream/assets',config,seed=old['seed'],policy_mode='reacquire',record_step_timings=True,record_policy_inputs=True)
 raw['development_source']={'policy':'ReacquisitionPiperTaskPolicy','source_sha256':hashlib.sha256(Path('src/dvidia_training/piper_reacquisition.py').read_bytes()).hexdigest(),'reference_path':str(path)}
 (output/path.name).write_bytes(gzip.compress(json.dumps(raw,allow_nan=False).encode()))
 row={'case':path.name,'valid':raw['valid'],'success':raw['success'],'old_success':old['success'],'first_success_s':raw['metrics'].get('first_success_time_seconds'),'abort':raw['grip_guard'].get('abort_reason'),'events':[(e['type'],e['t_seconds'],e['reason']) for e in raw['grip_guard'].get('events',[])],'error':raw.get('error')}
 summary.append(row)
 print(json.dumps(row),flush=True)
 (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

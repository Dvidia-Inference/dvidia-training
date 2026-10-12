from pathlib import Path
import gzip, json, shutil
from hashlib import sha256
from dvidia_training import piper_task as task, piper_recovery_benchmark as previous

root = Path(__file__).resolve().parents[2]
output = root/'runs/piper-v05-development/original-regressions'
output.mkdir(exist_ok=False)
(output/'sources').mkdir()
sources = {}
for name in previous.SOURCE_FILES:
    source = root/'src/dvidia_training'/name
    shutil.copy2(source, output/'sources'/name)
    sources[name] = sha256(source.read_bytes()).hexdigest()
rows=[]
for cohort, source in [
    (813407, root/'benchmarks/piper-recovery-development-20261011/paired-benchmark'),
    (941827, root/'benchmarks/piper-recovery-v04-20261011/receipt'),
]:
    report = json.loads((source/'report.json').read_bytes())
    for entry in report['attempts']:
        if entry['variant'] != 'recovery_placement': continue
        old = json.loads(gzip.decompress((source/entry['result_file']).read_bytes()))
        new = task.run_episode(root/'runs/robot-profile-upstream/assets', task.PiperTaskConfig(**old['config']),
            seed=old['seed'], policy_mode='recovery', record_step_timings=True, record_policy_inputs=True)
        name = str(cohort)+'-'+entry['scene_id']+'.json.gz'
        (output/name).write_bytes(gzip.compress(previous._json_bytes(new),mtime=0))
        same = {'actions':new['actions']==old['actions'],
                'native_states':[(s['qpos'],s['qvel']) for s in new['trajectory']]==[(s['qpos'],s['qvel']) for s in old['trajectory']],
                'metrics':new['metrics']==old['metrics'], 'valid':new['valid']==old['valid'],
                'success':new['success']==old['success']}
        row={'cohort':cohort,'scene_id':entry['scene_id'],'result_file':name,'matches':same,'all_equal':all(same.values())}
        rows.append(row)
        (output/'summary.json').write_bytes(previous._json_bytes({'source_sha256':sources,'attempts':rows,'all_equal':all(r['all_equal'] for r in rows)}))
        print(json.dumps(row),flush=True)
assert len(rows)==24 and all(r['all_equal'] for r in rows)

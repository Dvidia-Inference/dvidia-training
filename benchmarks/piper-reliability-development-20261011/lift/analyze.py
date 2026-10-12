from pathlib import Path
import gzip,hashlib,json
root=Path(__file__).resolve().parent
results=[]
for p in sorted(root.glob('*/*.json.gz')):
 raw=json.loads(gzip.decompress(p.read_bytes()))
 meta=raw.get('development_source',raw.get('development_injection'))
 old=json.loads(gzip.decompress(Path(meta['reference_path']).read_bytes()))
 same=lambda key: all(a[key]==b[key] for a,b in zip(raw['trajectory'],old['trajectory'])) and len(raw['trajectory'])==len(old['trajectory'])
 source=p.parent/'sources'/'piper_reacquisition.py'
 if hashlib.sha256(source.read_bytes()).hexdigest()!=meta['source_sha256']:
  raise ValueError('Candidate source mismatch '+str(p))
 for name,h in {**raw['implementation_sha256'],**raw['policy_implementation_sha256']}.items():
  if hashlib.sha256((p.parent/'sources'/name).read_bytes()).hexdigest()!=h:
   raise ValueError('Runtime source mismatch '+str(p)+' '+name)
 events=raw['grip_guard']['events']
 probes=[e for e in events if e['type']=='second_probe_started']
 qualified=[e for e in events if e['type']=='second_probe_qualified']
 results.append({'revision':p.parent.name,'raw':str(p.relative_to(root)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
 'bytes':p.stat().st_size,'valid':raw['valid'],'success':raw['success'],'reference_success':old['success'],
 'first_success_s':raw['metrics'].get('first_success_time_seconds'),'reference_first_success_s':old['metrics'].get('first_success_time_seconds'),
 'abort_reason':raw['grip_guard']['abort_reason'],'actions_identical':raw['actions']==old['actions'], 'qpos_identical':same('qpos'),'qvel_identical':same('qvel'),
 'native_model_identical':raw['compiled_model_sha256']==old['compiled_model_sha256'],
 'native_gains_identical':raw['native_servo_gains']==old['native_servo_gains'],
 'warnings':raw['warnings'],'unexpected_contact_native_ticks':raw['metrics']['unexpected_contact_native_ticks'],
 'second_probe_started_s':[e['t_seconds'] for e in probes], 'second_probe_qualified_s':[e['t_seconds'] for e in qualified],
 'completed_probe_duration_s':[q['t_seconds']-p['t_seconds'] for p,q in zip(probes,qualified)]})
summary={'scope':'Known-layout development only. The old v0.4 confirmation cohort is now development, not fresh confirmation. All attempted native episodes retained.',
'attempted':len(results),'valid':sum(r['valid'] for r in results),'success':sum(r['success'] for r in results),'raw_bytes':sum(r['bytes'] for r in results),
'by_revision':{v:{'attempts':sum(r['revision']==v for r in results),'success':sum(r['revision']==v and r['success'] for r in results), 'exact_native_trajectory_matches':sum(r['revision']==v and r['qpos_identical'] and r['qvel_identical'] and r['actions_identical'] for r in results)} for v in sorted(set(r['revision'] for r in results))},'results':results}
(root/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='results'},indent=2))

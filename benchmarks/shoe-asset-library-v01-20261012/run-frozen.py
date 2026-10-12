from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import socket
import time
from dvidia_training import asset_library, asset_benchmark, asset_preview

p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--archive',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--assets',required=True,type=Path);p.add_argument('--ur5e-assets',required=True,type=Path);a=p.parse_args()
root=Path(asset_library.__file__).parent
names=['asset_library.py','asset_catalog.json','asset_benchmark.py','asset_preview.py','asset_cli.py','asset_library.js','asset_library.bundle.js','asset_library.html','reference_sim.py','reference_assets.lock.json','robot_profiles.py','robot_profiles.lock.json','reference_replay.py','reference_replay.third-party.txt']
hashes={}
for name in names:
 installed=(root/name).read_bytes(); expected=(a.archive/'src/dvidia_training'/name).read_bytes()
 assert installed==expected,name
 hashes[name]=hashlib.sha256(installed).hexdigest()
s=socket.socket();network_denied=False
try:s.bind(('127.0.0.1',0))
except PermissionError:network_denied=True
finally:s.close()
assert network_denied,'Expected OS network denial'
a.output.mkdir(parents=True,exist_ok=False)
record={'source_commit':a.source,'installed_module_sha256':hashes,'network_denied_socket_bind':network_denied,'versions':{n:importlib.metadata.version(n) for n in ['dvidia-training','mujoco','numpy']},'scope':'Wheel built from frozen Git archive; installed files match archive; OS sandbox denies network during benchmark and previews.'}
(a.output/'installed-runtime.json').write_text(json.dumps(record,indent=2)+'\n')
roots={'ur5e':a.ur5e_assets,'piper':a.assets,'so101':a.assets}
report=asset_benchmark.benchmark(roots,a.output/'benchmark',seed=612047,repeats=3,source_commit=a.source,progress=lambda value:print(json.dumps(value),flush=True))
print(json.dumps({'phase':'benchmark-complete','summary':report['summary'],'timing':report['timing'],'hardware':report['hardware']}),flush=True)
previews={}
for robot,path in roots.items():
 result=asset_preview.export_preview(robot,path,a.output/'previews'/robot)
 previews[robot]={k:result[k] for k in ('bytes','file_count','frame_count')}
 print(json.dumps({'phase':'preview-complete','robot':robot,**previews[robot]}),flush=True)
(a.output/'execution.json').write_text(json.dumps({'source_commit':a.source,'benchmark':asset_benchmark.inspect_benchmark(a.output/'benchmark'),'previews':previews},indent=2)+'\n')

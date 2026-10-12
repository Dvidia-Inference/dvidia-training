"""Add a bounded shoe asset library while retaining every prior Space receipt."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import publish_robot_sim as publisher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
NAMESPACE = 'asset-library/v01'
SCHEMA = 'dvidia.asset-library-release/v0.1'
MAX_LIBRARY_BYTES = 96*1024*1024
MAX_LIBRARY_FILES = 100
ROBOTS = ('ur5e','piper','so101')


def encoded(value):
    return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


def record(raw):
    return {'bytes':len(raw),'sha256':sha256(raw).hexdigest()}


def read_payload(root):
    root = Path(root).absolute()
    publisher._plain(root)
    files = {}
    for path in sorted(root.rglob('*')):
        publisher._plain(path)
        if not path.is_file():
            continue
        name = path.relative_to(root).as_posix()
        publisher._artifact_name(name)
        raw = publisher._bytes(path,publisher.MAX_FILE_BYTES)
        publisher._no_private_artifact_bytes(name,raw)
        files[name] = raw
        if len(files)>MAX_LIBRARY_FILES or sum(map(len,files.values()))>MAX_LIBRARY_BYTES:
            raise ValueError('Asset library exceeds its bounded publication budget')
    return files


def check_library(directory):
    from dvidia_training.asset_benchmark import inspect_benchmark, implementation_hashes
    from dvidia_training.asset_preview import inspect_preview
    from dvidia_training.asset_library import catalog
    root = Path(directory)
    files = read_payload(root)
    seal = json.loads(files.pop('library-integrity.json'))
    if seal.get('schema') != SCHEMA or seal.get('files') != {n:record(b) for n,b in files.items()}:
        raise ValueError('Asset library inventory or bytes changed')
    if json.loads(files['catalog.json']) != catalog():
        raise ValueError('Published catalog differs from the packaged catalog')
    provenance = json.loads(files['provenance.json'])
    if (provenance.get('schema')!=SCHEMA or provenance.get('robots')!=list(ROBOTS)
        or publisher.SHA.fullmatch(provenance.get('source_commit','')) is None
        or provenance.get('physical_robot_ready') is not False or provenance.get('learned_policy') is not False):
        raise ValueError('Invalid asset library provenance')
    inspect_benchmark(root/'benchmark')
    protocol=json.loads(files['benchmark/protocol.json'])
    if protocol.get('source_commit')!=provenance['source_commit'] or protocol.get('implementation_sha256')!=implementation_hashes():
        raise ValueError('Library source or installed runtime differs from the frozen benchmark')
    for robot in ROBOTS:
        if inspect_preview(root/'previews'/robot)['robot']!=robot:
            raise ValueError('Published preview belongs to a different robot')
    return provenance


def landing(source_commit):
    links = ''.join(f'<article><span>Native scene preview</span><h2>{label}</h2><p>{axes} arm axes + gripper. {license_} model assets.</p><a href="previews/{robot}/index.html">Explore shoes with {label} →</a></article>'
                    for robot,label,axes,license_ in [('ur5e','UR5e + Robotiq',6,'BSD'),('piper','PiPER',6,'MIT'),('so101','SO-101',5,'Apache-2.0')])
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DVIDIA · Shoe asset library</title><style>
:root{{font-family:Georgia,serif;color:#262e29;background:#fafaf7}}*{{box-sizing:border-box}}body{{margin:0;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='3'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' opacity='.025' filter='url(%23n)'/%3E%3C/svg%3E")}}main{{max-width:1120px;padding:32px 22px;margin:auto}}header,footer,nav{{display:flex;gap:20px;flex-wrap:wrap;align-items:center;font:13px system-ui}}header{{border-bottom:1px solid #d7ddd4;padding-bottom:20px}}a{{color:#37604b;min-height:44px;display:inline-flex;align-items:center;overflow-wrap:anywhere}}h1{{font-size:clamp(34px,6vw,62px);font-weight:400;line-height:1.12;max-width:18ch}}h2{{font-weight:400}}p{{font-size:17px;line-height:1.7;max-width:78ch}}.tag,article span{{font:12px system-ui;letter-spacing:.06em;color:#566854}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,270px),1fr));gap:16px;margin:30px 0}}article{{border:1px solid #d7ddd4;background:#fff8;padding:22px}}article p{{font-size:15px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf0e9;padding:18px;line-height:1.6;font-size:13px}}.note{{border-left:3px solid #789180;padding-left:18px}}footer{{border-top:1px solid #d7ddd4;margin-top:36px;padding-top:18px}}
</style></head><body><main><header><strong>DVIDIA / ASSET LIBRARY</strong><a href="https://research.dvidia.org/papers/shoe-asset-library-v01/">Research note</a><a href="https://github.com/Dvidia-Inference/dvidia-training/tree/{source_commit}">Pinned source</a></header>
<p class="tag">VERSION 0.1 · LOCAL CPU SIMULATION</p><h1>Load a robot. Add a pair of shoes.</h1><p>Six original rigid shoe proxies across trainers, ankle boots and slippers, with left/right identities, three supported robot models and reusable scene files. Inspect the native 3D geometry, choose a pair, then download the scene and run it locally.</p>
<div class="cards">{links}</div><p class="note">These are synthetic geometry and contact-development assets. Dimensions, mass and friction are declared assumptions. Deformation, shoe grasping and hardware transfer have not been validated. The browser replays recorded settling; native physics runs locally.</p>
<h2>Run your own scene</h2><pre>git clone https://github.com/Dvidia-Inference/dvidia-training.git
cd dvidia-training
git checkout {source_commit}
python3.12 -m venv .venv
.venv/bin/python -m pip install '.[arm]'
.venv/bin/dvidia-assets fetch piper --output runs/assets
.venv/bin/dvidia-assets scene --robot piper --assets runs/assets --pair trainer --output runs/my-shoe-scene
.venv/bin/python -m http.server 8360 --bind 127.0.0.1 --directory runs/my-shoe-scene</pre>
<p>Open localhost:8360 after the command finishes. On Windows use <code>.venv\\Scripts\\</code> executables. Provisioning needs a network connection; scene simulation uses the pinned local models. A discrete GPU is not required. Minimum memory and hard real-time performance remain unmeasured.</p>
<nav><a href="benchmark/report.html">Inspect every benchmark attempt →</a><a href="catalog.json">Download the asset catalog</a><a href="README.md">Installation and evidence guide</a><a href="library-integrity.json">Verify all file hashes</a></nav>
<footer><span>Original shoe assets and tooling: MIT. Robot models retain their own licenses.</span><a href="https://x.com/iammrriver">@iammrriver</a><a href="mailto:hello@dvidia.org">hello@dvidia.org</a></footer></main></body></html>'''.encode()


def bundle(benchmark, previews, output, source_commit):
    from dvidia_training.asset_benchmark import inspect_benchmark, implementation_hashes
    from dvidia_training.asset_preview import inspect_preview
    from dvidia_training.asset_library import catalog
    if publisher.SHA.fullmatch(source_commit) is None or set(previews)!=set(ROBOTS):
        raise ValueError('Three declared robot previews and a pinned source are required')
    inspect_benchmark(benchmark)
    protocol=json.loads((Path(benchmark)/'protocol.json').read_bytes())
    if protocol.get('source_commit')!=source_commit or protocol.get('implementation_sha256')!=implementation_hashes():
        raise ValueError('Requested source or installed runtime differs from the frozen benchmark')
    payload = {'benchmark/'+n:b for n,b in read_payload(benchmark).items()}
    for robot,path in previews.items():
        if inspect_preview(path)['robot']!=robot:
            raise ValueError('Preview mapping names a different robot')
        payload.update({'previews/'+robot+'/'+n:b for n,b in read_payload(path).items()})
    payload.update({'index.html':landing(source_commit),'catalog.json':encoded(catalog()),
                    'README.md':(ROOT/'docs/asset-library.md').read_bytes(),
                    'LICENSE':(ROOT/'LICENSE').read_bytes(),
                    'provenance.json':encoded({'schema':SCHEMA,'source_commit':source_commit,'robots':list(ROBOTS),
                        'physical_robot_ready':False,'learned_policy':False,
                        'scope':'Original synthetic rigid shoe scene assets. Native gravity/support tests and recorded previews; no shoe skill or measured physical calibration.'})})
    payload['library-integrity.json']=encoded({'schema':SCHEMA,'files':{n:record(b) for n,b in sorted(payload.items())}})
    destination=Path(output).absolute()
    publisher._plain(destination)
    destination.mkdir(parents=True,exist_ok=False)
    for name,raw in payload.items():
        path=destination/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(raw)
    check_library(destination)
    return {'files':len(payload),'bytes':sum(map(len,payload.values()))}


def stage(base_plan, library, output, source_commit, expected_remote_commit):
    if (publisher.SHA.fullmatch(source_commit) is None
        or publisher.SHA.fullmatch(expected_remote_commit) is None):
        raise ValueError('Exact source and remote parent revisions are required')
    base_plan=Path(base_plan).absolute()
    base=publisher.checked_plan(base_plan)
    library_provenance=check_library(library)
    if library_provenance['source_commit']!=source_commit:
        raise ValueError('Library source differs from the requested publication source')
    payload={f['path']:publisher._bytes(base_plan.parent/'space'/f['path'],publisher.MAX_FILE_BYTES)
             for f in base['repository']['files']}
    if any(n.startswith(NAMESPACE+'/') for n in payload):
        raise ValueError('This first release must not replace an existing asset library')
    payload.update({NAMESPACE+'/'+n:b for n,b in read_payload(library).items()})
    index=payload['index.html'].decode()
    marker='<section aria-labelledby="replay-title">'
    if index.count(marker)!=1:
        raise ValueError('Expected existing lab entry point before adding library')
    card='<section aria-labelledby="assets-title"><h2 id="assets-title">The shoe asset library is runnable.</h2><p>Choose one of three robot profiles, inspect six original rigid shoe proxies, download a reusable scene and run CPU simulation locally. Synthetic properties are labelled as assumptions; gravity/support benchmarks retain every attempt and matched control.</p><a href="'+NAMESPACE+'/index.html">Open the asset library →</a></section>'
    payload['index.html']=index.replace(marker,card+marker).encode()
    payload['README.md']+=('\n\n## Shoe asset library v0.1\n\n[Explore the asset library]('+NAMESPACE+'/index.html): six original rigid shoe proxies and UR5e/Robotiq, PiPER and SO-101 scenes. [Complete benchmark]('+NAMESPACE+'/benchmark/report.html). Native loading and support tests do not establish shoe grasping, deformation or physical transfer.\n').encode()
    provenance=json.loads(payload['provenance.json'])
    provenance['asset_library']={'path':NAMESPACE,'source_commit':source_commit,
                                'integrity':record(payload[NAMESPACE+'/library-integrity.json'])}
    payload['provenance.json']=encoded(provenance)
    destination=Path(output).absolute()
    publisher._plain(destination)
    (destination/'space').mkdir(parents=True,exist_ok=False)
    for name,raw in payload.items():
        path=destination/'space'/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(raw)
    plan={**base,'source_commit':source_commit,'expected_remote_commit':expected_remote_commit,'remove_files':[],
          'repository':{**base['repository'],'files':[{'path':n,**record(b)} for n,b in sorted(payload.items())]},
          'total_bytes':sum(map(len,payload.values()))}
    path=destination/'publication-plan.json'
    path.write_bytes(encoded(plan))
    publisher.checked_plan(path)
    return {'plan':str(path),'files':len(payload),'bytes':plan['total_bytes']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('bundle')
    p.add_argument('--benchmark',required=True,type=Path)
    for robot in ROBOTS:p.add_argument('--'+robot,required=True,type=Path)
    for command in (p,):
        command.add_argument('--output',required=True,type=Path)
        command.add_argument('--source-commit',required=True)
    p=sub.add_parser('stage')
    p.add_argument('--base-plan',required=True,type=Path)
    p.add_argument('--library',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--expected-remote-commit',required=True)
    args=parser.parse_args()
    if args.command=='bundle':
        result=bundle(args.benchmark,{r:getattr(args,r) for r in ROBOTS},args.output,args.source_commit)
    else:result=stage(args.base_plan,args.library,args.output,args.source_commit,args.expected_remote_commit)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

"""Content-addressed simulation adapter bundle, distinct from a learned/hardware release."""
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from .skillspace import ADAPTER_ID, ARM_PROFILE, load_skillspace
from .installer import runtime_contract, RUNTIME_SOURCE_FILES


def export_arm_pack(directory):
    directory=Path(directory)
    payload=json.loads((directory/'run.json').read_text())
    if payload.get('task')!='ArmPickPlace-v0' or payload.get('scope')!='simulation-only':
        raise ValueError('Arm pack requires a simulation-only placement run.')
    if payload.get('runtime') != runtime_contract():
        raise ValueError('Run must pin the current executable runtime. Repeat the run before exporting changed code.')
    source=(directory/'source.json').read_bytes()
    if sha256(source).hexdigest()!=payload['skill']['source_sha256']:
        raise ValueError('Skill source hash does not match its exact run input.')
    files={}
    for path in Path(__file__).parent.iterdir():
        if path.is_file() and (path.suffix in ('.py','.html','.json','.txt') or path.name=='LICENSE'):
            files['simlab/'+path.name]=path.read_bytes()
    for name in ('run.json','source.json','grounding.json'):
        if (directory/name).exists():
            files['evidence/'+name]=(directory/name).read_bytes()
    files['requirements.lock.txt']=Path(__file__).with_name('requirements.lock.txt').read_bytes()
    files['README.md']=b'''# DVIDIA articulated-arm simulation adapter\n\nThis is an authored simulation controller bound to DVIDIA task metadata. No physical robot qualification or learned video policy is included.\n\nUse Python 3.12+ and the pinned dependencies in requirements.lock.txt. From this extracted directory: `python -m simlab.studio --offline` for the local lab, or `python -m simlab.arm_qualify --source evidence/source.json --output reproduced` to repeat the six-layout protocol. If source has no inline grounding, arm_qualify explicitly binds the documented local adapter. Dependencies must already be installed for offline execution.\n\nHashes check content integrity, not publisher identity or authenticity. See evidence/run.json for exact task, scene, contact exclusions, timings and outcomes.\n'''
    manifest={'schema_version':1,'task':'ArmPickPlace-v0','scope':'simulation-only',
              'status':'authored_adapter_with_recorded_simulation_evidence',
              'arm_profile':payload['skill']['arm_profile'],'adapter_id':payload['skill']['adapter_id'],
              'source_has_robot_policy':False,'physical_robot_ready':False,
              'runtime':payload['runtime'],
              'summary':payload['summary'],'limitations':payload['limitations'],
              'files':{name:sha256(data).hexdigest() for name,data in sorted(files.items())}}
    with ZipFile(directory/'skill-adapter.zip','w',ZIP_DEFLATED) as archive:
        for name,data in sorted(files.items()):archive.writestr(name,data)
        archive.writestr('manifest.json',json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    verify_arm_pack(directory/'skill-adapter.zip')
    return manifest


def verify_arm_pack(path):
    with ZipFile(path) as archive:
        names=archive.namelist()
        if len(names)>128 or len(names)!=len(set(names)) or sum(i.file_size for i in archive.infolist())>128*1024*1024:
            raise ValueError('Duplicate entries or archive resource budget exceeded.')
        manifest=json.loads(archive.read('manifest.json'))
        expected={'schema_version':1,'task':'ArmPickPlace-v0','scope':'simulation-only',
                  'status':'authored_adapter_with_recorded_simulation_evidence',
                  'arm_profile':ARM_PROFILE,'adapter_id':ADAPTER_ID,
                  'source_has_robot_policy':False,'physical_robot_ready':False}
        if any(type(manifest.get(key)) is not type(value) or manifest.get(key)!=value for key,value in expected.items()):
            raise ValueError('Invalid arm adapter task, provenance or scope.')
        if set(names)!=set(manifest['files'])|{'manifest.json'}:
            raise ValueError('Archive entries do not match the manifest.')
        for name,digest in manifest['files'].items():
            if name.startswith('/') or '\\' in name or ':' in name or '..' in Path(name).parts or sha256(archive.read(name)).hexdigest()!=digest:
                raise ValueError('Unsafe archive path or content hash mismatch.')
        payload=json.loads(archive.read('evidence/run.json'))
        # Historical v0 archives remain verifiable. New exports always bind
        # measured execution to the exact code included in their archive.
        if 'runtime' in manifest or 'runtime' in payload:
            runtime = manifest.get('runtime')
            if type(runtime) is not dict or payload.get('runtime') != runtime:
                raise ValueError('Runtime evidence does not match its manifest.')
            for name, digest in runtime.get('files_sha256', {}).items():
                if 'simlab/'+name not in names or sha256(archive.read('simlab/'+name)).hexdigest()!=digest:
                    raise ValueError('Recorded executable differs from the bundled source.')
            if set(runtime.get('files_sha256', {})) != set(RUNTIME_SOURCE_FILES):
                raise ValueError('Runtime dependency closure is incomplete.')
        grounding=json.loads(archive.read('evidence/grounding.json')) if 'evidence/grounding.json' in names else None
        if load_skillspace(archive.read('evidence/source.json'),grounding).to_dict()!=payload['skill']:
            raise ValueError('Skill grounding does not match the recorded execution contract.')
        if (payload.get('task')!='ArmPickPlace-v0' or payload.get('scope')!='simulation-only'
                or payload['skill'].get('arm_profile')!=manifest['arm_profile']
                or payload['skill'].get('adapter_id')!=manifest['adapter_id']
                or payload['skill'].get('source_has_robot_policy') is not False
                or payload['skill'].get('robot_ready') is not False
                or payload.get('limitations')!=manifest['limitations']):
            raise ValueError('Arm evidence task or scope does not match its manifest.')
        if payload['summary']!=manifest['summary'] or payload['skill']['source_sha256']!=sha256(archive.read('evidence/source.json')).hexdigest():
            raise ValueError('Arm adapter evidence does not match its manifest.')
        episodes=payload['episodes']
        if payload['summary']['successes']!=sum(e['success'] for e in episodes) or payload['summary']['episodes']!=len(episodes):
            raise ValueError('Arm outcome count does not match its evidence.')
        if 'by_policy' in payload['summary']:
            groups=payload['summary']['by_policy']
            if set(groups)!=set(e['policy'] for e in episodes):
                raise ValueError('Policy groups do not match the recorded episodes.')
            for policy,group in groups.items():
                selected=[e for e in episodes if e['policy']==policy]
                if group.get('successes')!=sum(e['success'] for e in selected) or group.get('episodes')!=len(selected):
                    raise ValueError('Policy outcome count does not match its evidence.')
    return manifest

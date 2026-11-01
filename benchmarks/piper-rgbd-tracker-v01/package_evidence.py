"""Package reviewed static perception evidence, preserving development failures.

A passed audit is tied to the bytes actually copied and read back from the ZIP.
This verifies local identities, not publisher authentication or hardware safety.
"""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

MAX_FILE = 16*1024*1024
MAX_ARCHIVE = 512*1024*1024
COUNTS = {'development':160, 'confirmation':320}
VARIANTS = ('clear','partial_occlusion','tilt15','tilt30','absent','occluded','ambiguous','invalid_depth','stale','future')
FIXTURE_FILES = {'started.json','rgb.png','rgb.npy','depth.npy','input.json','truth.json'}
ROOT_FILES = {'protocol.json','freeze.json','development-lineage.json','index.html'}
SPLIT_FILES = {'manifest.json','predictions.json','report.json','scheduled.json','renderer-160.json','renderer-320.json'}


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def decode(raw):
    def reject(value):raise ValueError('Nonfinite JSON evidence')
    return json.loads(raw,parse_constant=reject)


def _plain(path):
    path=Path(path)
    if any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('Symlinks are forbidden in evidence paths')
    return path


def _snapshot(path, maximum, *, capture):
    path=_plain(path)
    descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(descriptor,'rb') as stream:
        before=os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size>maximum:
            raise ValueError('Expected bounded plain evidence file')
        chunks=[];digest=sha256();size=0
        while True:
            raw=stream.read(min(1024*1024,maximum-size+1))
            if not raw:break
            size+=len(raw)
            if size>maximum:raise ValueError('Evidence exceeded size bound')
            digest.update(raw)
            if capture:chunks.append(raw)
        after=os.fstat(stream.fileno())
        fields=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
        if any(getattr(before,k)!=getattr(after,k) for k in fields) or size!=before.st_size:
            raise ValueError('Evidence changed while being read')
    return b''.join(chunks) if capture else dict(bytes=size,sha256=digest.hexdigest())


def read(path):return _snapshot(path,MAX_FILE,capture=True)
def identity(raw):return dict(bytes=len(raw),sha256=sha256(raw).hexdigest())
def archive_identity(path):return _snapshot(path,MAX_ARCHIVE,capture=False)


def safe_name(value):
    if not isinstance(value,str) or '\\' in value or value!=PurePosixPath(value).as_posix():
        raise ValueError('Unsafe evidence path')
    parts=PurePosixPath(value).parts
    if not parts or value.startswith('/') or any(p in ('','.','..') or p.startswith('.') for p in parts):
        raise ValueError('Unsafe evidence path')
    return value


def checked_identity(value):
    if (not isinstance(value,dict) or set(value)!={'bytes','sha256'}
            or type(value['bytes']) is not int or not 0<value['bytes']<=MAX_FILE
            or not isinstance(value['sha256'],str) or re.fullmatch('[0-9a-f]{64}',value['sha256']) is None):
        raise ValueError('Malformed evidence identity')
    return value


def bound_read(path, expected):
    raw=read(path)
    if identity(raw)!=checked_identity(expected):raise ValueError('Evidence differs from sealed identity: '+Path(path).name)
    return raw


def _walk(root):
    root=_plain(root)
    if not root.is_dir():raise ValueError('Expected evidence directory')
    found=[]
    for path in sorted(root.rglob('*')):
        _plain(path)
        if path.is_dir():continue
        if not stat.S_ISREG(path.stat().st_mode):raise ValueError('Nonregular evidence entry')
        found.append(path)
    return found


def _allowed_study_path(name, sources):
    parts=PurePosixPath(name).parts
    if len(parts)==1:return name in ROOT_FILES
    if parts[0]=='frozen-source':return len(parts)==2 and parts[1] in sources
    if parts[0] not in COUNTS:return False
    if len(parts)==2:return parts[1] in SPLIT_FILES
    pattern=r'pose\d{2}-(160|320)-('+ '|'.join(VARIANTS)+')'
    if len(parts)==3 and parts[1]=='estimates':
        return parts[2]=='started.json' or re.fullmatch(pattern+r'(-started)?\.json',parts[2]) is not None
    return len(parts)==3 and re.fullmatch(pattern,parts[1]) is not None and parts[2] in FIXTURE_FILES|{'receipt.json'}


def _audit_inventory(proof):
    if (proof.get('schema')!='dvidia.rgbd-independent-audit/v1' or proof.get('passed') is not True
            or proof.get('mismatch_count')!=0 or proof.get('mismatches')!=[]
            or proof.get('source_bytes_verified') is not True
            or proof.get('expected_confirmation_ids')!=COUNTS['confirmation']
            or proof.get('processed_confirmation_ids')!=COUNTS['confirmation']):
        raise ValueError('A complete passed independent audit with verified sources is required')
    inventory=proof.get('evidence_inventory')
    if not isinstance(inventory,dict) or not inventory:raise ValueError('Missing audit evidence inventory')
    if proof.get('evidence_inventory_sha256')!=sha256(canonical(inventory)).hexdigest():
        raise ValueError('Audit inventory digest differs')
    for name,record in inventory.items():safe_name(name);checked_identity(record)
    required={'protocol.json','freeze.json'}|{f'{split}/{name}' for split in COUNTS for name in ('manifest.json','predictions.json','report.json')}
    if not required.issubset(inventory):raise ValueError('Audit omits required source/protocol/prediction/report evidence')
    return inventory


def package(study,audit,output,*,baseline,earlier,failed,card):
    study,audit,output=Path(study),Path(audit),Path(output)
    _plain(output)
    if study.absolute() in output.absolute().parents or output.absolute()==study.absolute():
        raise ValueError('Package output must be outside the study')
    audit_raw=read(audit);proof=decode(audit_raw);audited=_audit_inventory(proof)
    protocol_raw=bound_read(study/'protocol.json',audited['protocol.json'])
    freeze=decode(bound_read(study/'freeze.json',audited['freeze.json']))
    sources=freeze.get('sources')
    if not isinstance(sources,dict) or not sources:raise ValueError('Frozen source inventory is missing')
    if freeze.get('protocol_sha256')!=identity(protocol_raw)['sha256']:raise ValueError('Frozen protocol differs')
    expected=dict(audited);required=set(ROOT_FILES)
    for name,digest in sources.items():
        if Path(name).name!=name or Path(name).suffix not in ('.py','.json'):raise ValueError('Unsafe frozen source')
        key='frozen-source/'+name
        if key not in audited or audited[key]['sha256']!=digest:raise ValueError('Audit does not bind frozen source')
        required.add(key)
    for split,count in COUNTS.items():
        manifest=decode(bound_read(study/split/'manifest.json',audited[f'{split}/manifest.json']))
        predictions=decode(bound_read(study/split/'predictions.json',audited[f'{split}/predictions.json']))
        report=decode(bound_read(study/split/'report.json',audited[f'{split}/report.json']))
        rows=manifest['records'];ids=[row['id'] for row in rows]
        if (len(ids)!=count or len(set(ids))!=count or report['scheduled']!=count
                or [row['id'] for row in report['rows']]!=ids or [row['id'] for row in predictions['predictions']]!=ids
                or report['predictions_sha256']!=audited[f'{split}/predictions.json']['sha256']
                or report['manifest_sha256']!=audited[f'{split}/manifest.json']['sha256']):
            raise ValueError('Expected complete registered report and exact prediction/manifest binding')
        required.update(f'{split}/{name}' for name in SPLIT_FILES)
        required.add(f'{split}/estimates/started.json')
        for row,prediction in zip(rows,predictions['predictions']):
            if set(row['files'])!=FIXTURE_FILES:raise ValueError('Unexpected fixture artifact set')
            for name,record in row['files'].items():
                path=f'{split}/{row["id"]}/{name}';safe_name(path);checked_identity(record)
                if path in expected and expected[path]!=record:raise ValueError('Manifest differs from audited fixture identity')
                expected[path]=record;required.add(path)
            for name,digest in ((f'{split}/{row["id"]}/receipt.json',row['receipt_sha256']),
                                (f'{split}/estimates/{row["id"]}.json',prediction['receipt_sha256'])):
                raw=read(study/name);record=identity(raw)
                if record['sha256']!=digest or (name in expected and expected[name]!=record):
                    raise ValueError('Fixture or estimate receipt differs from audit/manifest')
                expected[name]=record;required.add(name)
            required.add(f'{split}/estimates/{row["id"]}-started.json')
    plan={}
    def register(source,destination,record=None):
        safe_name(destination)
        if Path(destination).suffix not in ('.json','.npy','.png','.py','.md','.txt','.html'):
            raise ValueError('Unexpected evidence format')
        if destination in plan:raise ValueError('Duplicate archive path')
        raw=read(source);observed=identity(raw)
        if record is not None and observed!=checked_identity(record):raise ValueError('Evidence differs from independent audit or sealed manifest')
        plan[destination]=(Path(source),observed)
    observed=set()
    for source in _walk(study):
        name=source.relative_to(study).as_posix();observed.add(name)
        if not _allowed_study_path(name,sources):raise ValueError('Unrecognized study file; refusing to package it')
        register(source,name,expected.get(name))
    if observed!=required or not set(audited).issubset(observed):
        raise ValueError('Study has missing or unexpected registered evidence files')
    # Historical results/source are preserved without duplicating their fixed
    # rendered frames. Their copied-byte identities remain explicit in inventory.
    for label,root in (('initial-development',Path(baseline)),('revised-development',Path(earlier))):
        _plain(root)
        for name in ('predictions.json','report.json'):
            register(root/'development'/name,f'history/{label}/{name}')
        snapshots=sorted(root.glob('development-source-*/*'))
        if not snapshots:raise ValueError('Historical source snapshot is missing')
        for source in snapshots:
            if source.name not in {'piper_rgbd_benchmark.py','piper_rgbd_tracker.py','test_piper_rgbd_tracker.py','test_piper_rgbd_benchmark.py'}:
                raise ValueError('Unexpected historical source snapshot')
            register(source,f'history/{label}/source/{source.name}')
    register(Path(failed)/'development/manifest.json','history/sandbox-graphics-failure/manifest.json')
    register(Path(failed)/'protocol.json','history/sandbox-graphics-failure/protocol.json')
    register(audit,'independent-audit.json',identity(audit_raw))
    repo=Path(__file__).resolve().parents[2]
    register(repo/'LICENSE','LICENSE.txt')
    for folder in ('third_party/mujoco','third_party/robot-models'):
        for source in sorted((repo/folder).glob('*.txt')):register(source,source.relative_to(repo).as_posix())
    card_raw=read(card)
    output.mkdir(parents=True,exist_ok=False);tree=output/'evidence';tree.mkdir()
    roster={}
    for destination,(source,record) in sorted(plan.items()):
        raw=bound_read(source,record)  # Check the exact bytes actually copied.
        target=tree/destination;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(raw)
        roster[destination]=record
    inventory=dict(schema='dvidia.rgbd-evidence-inventory/v1',audit_sha256=identity(audit_raw)['sha256'],
                   files=[dict(path=name,**info) for name,info in sorted(roster.items())])
    inventory_raw=canonical(inventory)+b'\n'
    with (tree/'inventory.json').open('xb') as stream:stream.write(inventory_raw)
    zip_roster=dict(roster,**{'inventory.json':identity(inventory_raw)})
    archive=output/'piper-rgbd-tracker-v01.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zipped:
        for name,record in sorted(zip_roster.items()):
            raw=bound_read(tree/name,record)
            info=zipfile.ZipInfo(name,date_time=(2026,11,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
            zipped.writestr(info,raw)
    with zipfile.ZipFile(archive) as zipped:
        if sorted(zipped.namelist())!=sorted(zip_roster):raise ValueError('ZIP has missing, duplicate or unexpected members')
        for info in zipped.infolist():
            if info.file_size>MAX_FILE or identity(zipped.read(info))!=zip_roster[info.filename]:
                raise ValueError('Archived bytes differ from sealed evidence')
    archive_record=archive_identity(archive)
    payload=output/'huggingface';payload.mkdir()
    with archive.open('rb') as source,(payload/archive.name).open('xb') as target:
        while raw:=source.read(1024*1024):target.write(raw)
    if archive_identity(archive)!=archive_record or archive_identity(payload/archive.name)!=archive_record:
        raise ValueError('Release archive changed during payload copy')
    for source_name,name in (('confirmation/report.json','report.json'),('protocol.json','protocol.json'),
                             ('independent-audit.json','inspection.json'),('LICENSE.txt','LICENSE')):
        raw=bound_read(tree/source_name,roster[source_name])
        with (payload/name).open('xb') as stream:stream.write(raw)
        if read(payload/name)!=raw:raise ValueError('Payload copy differs from archived evidence')
    with (payload/'README.md').open('xb') as stream:stream.write(card_raw)
    if read(payload/'README.md')!=card_raw:raise ValueError('Payload card changed')
    result=dict(archive=archive.name,**archive_record,inventory_files=len(roster),
                independent_audit_sha256=identity(audit_raw)['sha256'],zip_readback_verified=True)
    with (output/'package.json').open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','audit','output','baseline','earlier','failed','card'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();print(json.dumps(package(a.study,a.audit,a.output,baseline=a.baseline,earlier=a.earlier,failed=a.failed,card=a.card),indent=2))

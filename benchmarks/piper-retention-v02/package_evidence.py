"""Package the reviewed retention run and independent audit, offline.

The entire selected run and audit trees are archived in deterministic order.
Packaging never imports frozen runtime code, runs physics, reads credentials or
publishes. It refuses incomplete or failed audit evidence.
"""
from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
import stat
import sys
import zipfile

SCHEMA = 'dvidia.piper-retention-study/v1'
VARIANTS = {'measured_joint', 'issued_joint', 'cartesian'}
ARCHIVE_NAME = 'piper-retention-v02.zip'
HF_NAMES = {'README.md', 'LICENSE', ARCHIVE_NAME, 'report.json', 'protocol.json', 'inspection.json'}
MAX_FILE = 64 * 1024 * 1024
MAX_ARCHIVE = 192 * 1024 * 1024
MAX_TOTAL = 480 * 1024 * 1024
MAX_FILES = 2048
MAX_JSON = 16 * 1024 * 1024
PRIVATE = (b'/'+b'Users/', b'/private/'+b'tmp/', b'/opt/'+b'homebrew/', b'/var/'+b'folders/')
SECRETS = tuple(re.compile(value) for value in
                (rb'hf_[A-Za-z0-9]{20,}', rb'gh[pousr]_[A-Za-z0-9]{20,}',
                 rb'apikey_[A-Za-z0-9_]{20,}', rb'AKIA[A-Z0-9]{16}'))


def encoded(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()+b'\n'


def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate JSON fields.')
        result[key] = value
    return result


def nonfinite(_value):
    raise ValueError('Nonfinite JSON number.')


def load_json(path, limit=MAX_JSON):
    raw = read(path, limit)
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
    return value, raw


def read(path, limit=MAX_FILE):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError('Missing, linked or oversized evidence file.')
    with path.open('rb') as stream:
        raw = stream.read(limit+1)
    if len(raw) > limit:
        raise ValueError('Evidence grew beyond its byte limit.')
    return raw


def safe_path(name):
    if (type(name) is not str or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/-]{0,399}', name)
            or any(part in ('', '.', '..') for part in name.split('/'))):
        raise ValueError('Unsafe archive path.')
    return name


def scan_stream(stream, limit=MAX_FILE):
    size, tail = 0, b''
    while chunk := stream.read(65536):
        size += len(chunk)
        if size > limit:
            raise ValueError('Decoded evidence exceeds its byte limit.')
        block = tail+chunk
        if any(value in block for value in PRIVATE) or any(pattern.search(block) for pattern in SECRETS):
            raise ValueError('Private path or credential-shaped bytes in evidence.')
        tail = block[-256:]
    return size


def parse_json_bytes(raw):
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
    scan_stream(io.BytesIO(json.dumps(value, ensure_ascii=False, allow_nan=False).encode()))
    return value


def scan_file(path):
    raw = read(path)
    scan_stream(io.BytesIO(raw))
    if path.suffix == '.json':
        parse_json_bytes(raw)
    elif path.suffix == '.jsonl':
        for line in raw.splitlines():
            if line.strip():
                parse_json_bytes(line)
    elif path.suffix == '.gz':
        total = 0
        with gzip.open(path, 'rb') as stream:
            chunks = []
            while chunk := stream.read(65536):
                total += len(chunk)
                if total > MAX_FILE:
                    raise ValueError('Gzip content exceeds its byte limit.')
                scan_stream(io.BytesIO(chunk))
                chunks.append(chunk)
        decoded = b''.join(chunks)
        if decoded.lstrip().startswith((b'{', b'[')):
            parse_json_bytes(decoded)
    elif path.suffix in ('.npz', '.zip'):
        with zipfile.ZipFile(path) as bundle:
            entries = bundle.infolist()
            names = [item.filename for item in entries]
            if (not entries or len(entries) > MAX_FILES or len(names) != len(set(names))
                    or sum(item.file_size for item in entries) > MAX_TOTAL):
                raise ValueError('Nested archive exceeds its member/byte bound.')
            for item in entries:
                safe_path(item.filename)
                if (item.file_size > MAX_FILE or item.flag_bits & 1
                        or stat.S_IFMT(item.external_attr >> 16) not in (0, stat.S_IFREG)
                        or item.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
                    raise ValueError('Unsafe nested archive member.')
                with bundle.open(item) as stream:
                    scan_stream(stream)


def identity(path, limit=MAX_FILE):
    raw = read(path, limit)
    return {'bytes': len(raw), 'sha256': sha256(raw).hexdigest()}


def tree(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Evidence root must be a regular directory.')
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError('Evidence tree contains a link or special file.')
        if path.is_dir():
            continue
        name = safe_path(path.relative_to(root).as_posix())
        if name in result:
            raise ValueError('Duplicate evidence path.')
        scan_file(path)
        result[name] = identity(path)
        if len(result) > MAX_FILES or sum(item['bytes'] for item in result.values()) > MAX_TOTAL:
            raise ValueError('Evidence tree exceeds its count/size bound.')
    if not result:
        raise ValueError('Evidence tree is empty.')
    return result


def validate_inputs(study, audit):
    study, audit = Path(study), Path(audit)
    if 'piper-retention-v02' not in study.name:
        raise ValueError('Study directory must be the new piper-retention-v02 run.')
    study_files, audit_files = tree(study), tree(audit)
    for name in ('report.json', 'protocol.json', 'inspection.json'):
        if name not in study_files:
            raise ValueError('Run is missing '+name+'.')
    if 'audit.json' not in audit_files:
        raise ValueError('Independent audit is missing audit.json.')
    report, report_raw = load_json(study/'report.json')
    protocol, protocol_raw = load_json(study/'protocol.json')
    inspection, inspection_raw = load_json(study/'inspection.json')
    audit_doc, audit_raw = load_json(audit/'audit.json')
    for name, raw, identities in (('report.json', report_raw, study_files),
                                   ('protocol.json', protocol_raw, study_files),
                                   ('inspection.json', inspection_raw, study_files),
                                   ('audit.json', audit_raw, audit_files)):
        if {'bytes': len(raw), 'sha256': sha256(raw).hexdigest()} != identities[name]:
            raise ValueError('Parsed metadata differs from its pinned evidence identity.')
    rows = report.get('rows', report.get('attempts')) if type(report) is dict else None
    if (type(report) is not dict or report.get('schema') != SCHEMA or type(rows) is not list or len(rows) != 36
            or report.get('planned_attempts') != 36 or report.get('attempted') != 36
            or type(protocol) is not dict or type(inspection) is not dict or type(audit_doc) is not dict):
        raise ValueError('Run report, protocol or audit schema is incomplete.')
    inspection_rows = inspection.get('rows')
    if (inspection.get('schema') != 'dvidia.piper-retention-inspection/v1'
            or inspection.get('attempts') != 36 or type(inspection_rows) is not list
            or len(inspection_rows) != 36
            or [(row.get('id'), row.get('evaluation_recomputed')) for row in inspection_rows
                if type(row) is dict] != [(row.get('id'), True) for row in rows]):
        raise ValueError('Inspection does not cover all 36 attempts with recomputed evaluations.')
    if (audit_doc.get('schema') != 'dvidia.retention-independent-audit/v1'
            or type(audit_doc.get('passed')) is not bool or audit_doc['passed'] is not True
            or audit_doc.get('source_and_raw_inputs_unchanged') is not True
            or inspection.get('passed') is False or inspection.get('status') in ('FAIL', 'failed', 'refused')):
        raise ValueError('The independent audit or run inspection did not pass.')
    roster = {}
    scene_sets = {}
    pairs_seen = set()
    for row in rows:
        variant = row.get('variant') if type(row) is dict else None
        scene = row.get('scene_id') if type(row) is dict else None
        if variant not in VARIANTS or type(scene) is not str:
            raise ValueError('Attempt records must identify a scene and one of the three variants.')
        pair = (variant, scene)
        if pair in pairs_seen:
            raise ValueError('Duplicate scene/variant attempt in report.')
        pairs_seen.add(pair)
        roster.setdefault(variant, set()).add(scene)
    if (set(roster) != VARIANTS or any(len(scenes) != 12 for scenes in roster.values())
            or len({frozenset(scenes) for scenes in roster.values()}) != 1):
        raise ValueError('Expected twelve scenes for each of the three variants.')
    for field in ('protocol_sha256',):
        expected = sha256(protocol_raw).hexdigest()
        for doc in (audit_doc, report, inspection):
            if field in doc and doc[field] != expected:
                raise ValueError('Protocol digest differs across run and audit records.')
        if audit_doc.get(field) != expected:
            raise ValueError('Independent audit is not bound to this frozen protocol.')
    source_hashes = protocol.get('source_sha256')
    if (type(source_hashes) is not dict or report.get('source_sha256') != source_hashes
            or audit_doc.get('source_sha256') != source_hashes
            or audit_doc.get('frozen_raw_files') != study_files):
        raise ValueError('Independent audit source or frozen-input inventory differs from this run.')
    audit_rows = audit_doc.get('rows')
    if (type(audit_rows) is not list or len(audit_rows) != 36
            or [(row.get('id'), row.get('valid')) for row in audit_rows]
            != [(row.get('id'), row.get('valid')) for row in rows]):
        raise ValueError('Independent audit does not cover the exact 36 ordered run attempts.')
    return {'study': study, 'audit': audit, 'study_files': study_files,
            'audit_files': audit_files, 'report': report, 'report_raw': report_raw,
            'protocol_raw': protocol_raw, 'inspection_raw': inspection_raw,
            'audit_raw': audit_raw, 'report_doc': report, 'protocol': protocol,
            'inspection': inspection, 'audit_doc': audit_doc}


VERIFY_SOURCE = r'''"""Verify/reconstruct a DVIDIA PiPER retention archive; no pickle or imports."""
import argparse, hashlib, json, re, stat, zipfile
from pathlib import Path
MAX_FILE=64*1024*1024
MAX_TOTAL=480*1024*1024
def unique(pairs):
    out={}
    for k,v in pairs:
        if k in out: raise ValueError('duplicate JSON field')
        out[k]=v
    return out
def safe(name):
    if type(name) is not str or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/-]{0,399}',name) or any(p in ('','.','..') for p in name.split('/')): raise ValueError('unsafe archive path')
    return name
def verify(path,output=None):
    with zipfile.ZipFile(path) as z:
        es=z.infolist(); names=[e.filename for e in es]
        if not 1<=len(es)<=2050 or len(names)!=len(set(names)) or sum(e.file_size for e in es)>MAX_TOTAL: raise ValueError('archive bounds/roster invalid')
        for e in es:
            safe(e.filename)
            if e.file_size>MAX_FILE or e.flag_bits&1 or stat.S_IFMT(e.external_attr>>16) not in (0,stat.S_IFREG) or e.compress_type not in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED): raise ValueError('unsafe archive member')
        if 'inventory.json' not in names or z.getinfo('inventory.json').file_size>4*1024*1024: raise ValueError('missing/large inventory')
        inv=json.loads(z.read('inventory.json'),object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite')))
        if inv.get('schema')!='dvidia.piper-retention-inventory/v1' or type(inv.get('files')) is not list: raise ValueError('unknown inventory schema')
        records={}
        for r in inv['files']:
            if type(r) is not dict or set(r)!={'path','bytes','sha256'}: raise ValueError('malformed inventory record')
            n=safe(r['path'])
            if n in records or n=='inventory.json' or type(r['bytes']) is not int or not 0<=r['bytes']<=MAX_FILE or type(r['sha256']) is not str or not re.fullmatch('[0-9a-f]{64}',r['sha256']): raise ValueError('invalid inventory identity')
            records[n]=r
        if set(names)!=set(records)|{'inventory.json'}: raise ValueError('inventory roster differs')
        for name,r in records.items():
            h=hashlib.sha256(); size=0
            with z.open(name) as f:
                while c:=f.read(65536):
                    size+=len(c)
                    if size>r['bytes']: raise ValueError('member exceeds expected bytes')
                    h.update(c)
            if size!=r['bytes'] or h.hexdigest()!=r['sha256']: raise ValueError('member digest differs')
        if output is not None:
            root=Path(output)
            if root.exists() or root.is_symlink(): raise ValueError('output must be new')
            root.mkdir(parents=True); base=root.resolve()
            for n in names:
                dest=root/n; dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.parent.resolve().is_relative_to(base): raise ValueError('output escape')
                with z.open(n) as src,dest.open('xb') as dst:
                    while c:=src.read(65536): dst.write(c)
    return {'verified':True,'files':len(records),'logical_bytes':sum(r['bytes'] for r in records.values()),'reconstructed':output is not None}
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('archive'); p.add_argument('--output'); a=p.parse_args(); print(json.dumps(verify(a.archive,a.output),sort_keys=True))
if __name__=='__main__': main()
'''


def archive(path, files, generated, *, expected_identities):
    if set(expected_identities) != set(files):
        raise ValueError('Archive input roster differs from pinned identities.')
    expected = {name: dict(value) for name, value in expected_identities.items()}
    inventory = []
    for name, source in sorted(files.items()):
        safe_path(name)
        if identity(source) != expected[name]:
            raise ValueError('Archive input differs from its pinned evidence identity.')
        inventory.append({'path': name, **expected[name]})
    for name, raw in sorted(generated.items()):
        safe_path(name)
        scan_stream(io.BytesIO(raw))
        inventory.append({'path': name, 'bytes': len(raw), 'sha256': sha256(raw).hexdigest()})
    inventory.sort(key=lambda item: item['path'])
    if len(inventory) > MAX_FILES or sum(item['bytes'] for item in inventory) > MAX_TOTAL-1024*1024:
        raise ValueError('Expanded archive exceeds its limits.')
    content = {**files, **generated, 'inventory.json': encoded({'schema':'dvidia.piper-retention-inventory/v1',
                'files':inventory,'scope':'All run/audit artifacts and included tools; inventory excludes itself.'})}
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=True) as z:
        for name, source in sorted(content.items()):
            info=zipfile.ZipInfo(name, date_time=(1980,1,1,0,0,0)); info.create_system=3
            info.external_attr=(stat.S_IFREG|0o644)<<16
            info.compress_type=zipfile.ZIP_STORED if isinstance(source,Path) and source.suffix in ('.gz','.npz') else zipfile.ZIP_DEFLATED
            with z.open(info,'w',force_zip64=True) as dest:
                if isinstance(source,Path):
                    if source.is_symlink() or not source.is_file():
                        raise ValueError('Archive input is no longer a plain file.')
                    copied, copied_hash = 0, sha256()
                    with source.open('rb') as stream:
                        while chunk:=stream.read(65536):
                            copied += len(chunk)
                            if copied > expected[name]['bytes']:
                                raise ValueError('Copied archive input exceeds its pinned size.')
                            copied_hash.update(chunk)
                            dest.write(chunk)
                    if {'bytes': copied, 'sha256': copied_hash.hexdigest()} != expected[name]:
                        raise ValueError('Copied archive input differs from its pinned evidence identity.')
                else:
                    dest.write(source)
    return inventory


def archive_digest(path):
    h=sha256(); size=0
    with path.open('rb') as stream:
        while chunk:=stream.read(65536):
            size+=len(chunk)
            if size>MAX_ARCHIVE: raise ValueError('Compressed archive exceeds its limit.')
            h.update(chunk)
    return {'bytes':size,'sha256':h.hexdigest()}


def prepare(study, audit, output, repo_root):
    output=Path(output)
    if output.exists() or output.is_symlink(): raise ValueError('Package output must be new.')
    frozen=validate_inputs(study,audit)
    initial=(frozen['study_files'],frozen['audit_files'])
    files={**{'study/'+name:frozen['study']/name for name in frozen['study_files']},
           **{'audit/'+name:frozen['audit']/name for name in frozen['audit_files']}}
    expected_identities = {**{'study/'+name: value for name, value in frozen['study_files'].items()},
                           **{'audit/'+name: value for name, value in frozen['audit_files'].items()}}
    tool_dir=Path(__file__).parent
    files.update({
        'tools/package_evidence.py':Path(__file__),
        'tools/render_report.py':tool_dir/'render_report.py',
        'tools/test_package_evidence.py':tool_dir/'test_package_evidence.py',
        'tools/independent_audit.py':tool_dir/'independent_audit.py',
        'tools/verify_archive.py':None,
        'licenses/agilex_piper-LICENSE.txt':Path(repo_root)/'third_party/robot-models/agilex_piper-LICENSE.txt',
        'licenses/mujoco-LICENSE.txt':Path(repo_root)/'third_party/mujoco/LICENSE.txt'})
    generated_tools={name:raw for name,raw in files.items() if raw is None}
    files={name:path for name,path in files.items() if path is not None}
    readme=b'''# DVIDIA | PiPER retention v0.2\n\nA bounded native CPU study compares measured-joint, issued-joint and Cartesian hold strategies on twelve fresh synthetic scenes (36 attempts). The primary outcome is a continuous five-second hold after admission, checked against false confirmation. Every scheduled attempt and failure is retained. See report.json and the protocol for the frozen results and definitions.\n\nThis is not a learned policy, full transport or placement task, physical-sensor validation, hardware qualification or real-object reliability estimate. The simulator uses authored privileged state.\n\nThe ZIP preserves the complete new run tree under study/ and its independent audit under audit/. It includes deterministic inventory hashes and tools/verify_archive.py. To verify and optionally reconstruct without importing project code, extract the archive and run:\n\n    python tools/verify_archive.py piper-retention-v02.zip --output evidence\n\nResearch article: https://research.dvidia.org/papers/piper-retention-v02/\nSource release: https://github.com/Dvidia-Inference/dvidia-training/releases/tag/piper-retention-v0.2-preview\nEvidence dataset: https://huggingface.co/datasets/Dvidia/piper-retention-v02\n\nDVIDIA source and authored synthetic records are MIT. Upstream simulator and robot asset notices are included. No physical robot footage, credentials or hardware data are bundled.\n'''
    generated={**generated_tools,'README.md':readme,'LICENSE':read(Path(repo_root)/'LICENSE'),
               'tools/verify_archive.py':VERIFY_SOURCE.encode()}
    # Top-level project README/renderer/tool paths are included in the archive; scanner validates them too.
    for name, source in files.items():
        scan_file(source)
        if name not in expected_identities:
            expected_identities[name] = identity(source)
    output.mkdir(parents=True); stage=output/'huggingface'; stage.mkdir()
    archive_path=stage/ARCHIVE_NAME
    inventory=archive(archive_path,files,generated,expected_identities=expected_identities)
    verifier={'__name__':'retention_archive_verifier'}
    exec(compile(VERIFY_SOURCE,'verify_archive.py','exec'),verifier)
    verified=verifier['verify'](archive_path)
    if (tree(frozen['study'])!=initial[0] or tree(frozen['audit'])!=initial[1]):
        raise ValueError('Study or audit changed during packaging.')
    archive_id=archive_digest(archive_path)
    table = ['\n## Frozen results\n', '| Hold | Admissions | False initial | Native 5 s grip | Grip + TCP stationarity | Final controller + native |',
             '|---|---:|---:|---:|---:|---:|']
    for variant in ('measured_joint', 'issued_joint', 'cartesian'):
        result = frozen['report']['aggregates'][variant]
        table.append('| '+variant+' | '+str(result['admissions'])+'/12 | '+str(result['false_initial_admissions'])+' | '+str(result['native_5s_pause_success'])+'/12 | '+str(result['grip_retention_and_stationarity'])+'/12 | '+str(result['final_both_qualified'])+'/12 |')
    table.append('\nFresh finite development cells, not a population accuracy estimate. The separate development-smoke directory preserves the earlier selected old-case Cartesian refusal; it is excluded from all 36-attempt counts.\n')
    card=b'''---\nlicense: mit\npretty_name: DVIDIA PiPER retention v0.2\ntask_categories:\n- robotics\ntags:\n- robotics\n- simulation\n- piper\n- mujoco\n- research\n---\n\n'''+readme+'\n'.join(table).encode()+b'\nArchive SHA-256: `'+archive_id['sha256'].encode()+b'`\n'
    payload={'README.md':card,'LICENSE':generated['LICENSE'],
             'report.json':frozen['report_raw'],'protocol.json':frozen['protocol_raw'],
             'inspection.json':frozen['inspection_raw'],ARCHIVE_NAME:archive_path.read_bytes()}
    for name, raw in payload.items():
        scan_stream(io.BytesIO(raw),MAX_ARCHIVE if name==ARCHIVE_NAME else MAX_FILE)
        if name == ARCHIVE_NAME:
            if (stage/name).read_bytes() != raw:
                raise ValueError('Staged archive bytes changed while preparing dataset payload.')
        else:
            with (stage/name).open('xb') as stream: stream.write(raw)
    if {p.name for p in stage.iterdir()}!=HF_NAMES: raise ValueError('Dataset payload must contain exactly six files.')
    inventory_path=output/'inventory.json'; inventory_path.write_bytes(encoded({'schema':'dvidia.piper-retention-package/v1','files':[{'path':p.name,**identity(p,MAX_ARCHIVE if p.name==ARCHIVE_NAME else MAX_FILE)} for p in sorted(stage.iterdir())]}))
    receipt={'schema':'dvidia.piper-retention-package-receipt/v1','remote_operations':False,
             'target_dataset':'Dvidia/piper-retention-v02','source_tag':'piper-retention-v0.2-preview',
             'attempts':36,'variants':sorted(VARIANTS),'study_files':len(frozen['study_files']),
             'audit_files':len(frozen['audit_files']),'archive':{'filename':ARCHIVE_NAME,**archive_id},
             'verified':verified,'frozen_inputs_unchanged':True,
             'pinned_archive_inputs_verified':{'files':len(expected_identities),'at_index':True,'at_stream_copy':True},
             'inventory_sha256':sha256(inventory_path.read_bytes()).hexdigest()}
    (output/'package-receipt.json').write_bytes(encoded(receipt))
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True); parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:
        result=prepare(args.study,args.audit,args.output,args.repo)
        print(json.dumps({'archive':result['archive'],'verified':result['verified']},indent=2))
        return 0
    except (ValueError,OSError,KeyError,TypeError,zipfile.BadZipFile) as exc:
        print('Evidence package refused: '+str(exc),file=sys.stderr); return 2


if __name__=='__main__': raise SystemExit(main())

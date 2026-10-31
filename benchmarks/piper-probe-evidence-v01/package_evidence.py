"""Deterministically package reviewed PiPER records, offline and without physics.

Every frozen study file is bound to the independent audit before indexing and
again after copying. Archive readback verifies every inventory digest. No code
from frozen src is imported, no credentials are read, and nothing is published.
"""
from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import stat
import zipfile

AUDIT_SHA256 = 'b333f788d388f5e43a035b6bcd85bb6517e197d24a18c92c12efbd6ae3ccbe8f'
SOURCE_COMMIT = 'a6a1d15db905214da52eb3942ebead3f85254762'
ARCHIVE_NAME = 'piper-probe-evidence-v01.zip'
HF_NAMES = {'README.md', 'LICENSE', ARCHIVE_NAME, 'report.json', 'protocol.json', 'inspection.json'}
MAX_FILE = 32*1024*1024
MAX_TOTAL = 256*1024*1024
PRIVATE = (b'/'+b'Users/', b'/private/'+b'tmp/', b'/opt/'+b'homebrew/', b'/var/'+b'folders/')
SECRETS = tuple(re.compile(value) for value in (rb'hf_[A-Za-z0-9]{20,}',
    rb'gh[pousr]_[A-Za-z0-9]{20,}', rb'apikey_[A-Za-z0-9_]{20,}', rb'AKIA[A-Z0-9]{16}'))

VERIFY_SOURCE = r'''"""Verify/reconstruct fixed DVIDIA evidence; no code imports or pickle reads."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import stat
import zipfile

MAX_FILE = 32*1024*1024
MAX_TOTAL = 256*1024*1024

def safe_path(name):
    if (type(name) is not str or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/-]{0,399}', name)
            or any(part in ('', '.', '..') for part in name.split('/'))):
        raise ValueError('Unsafe artifact path')
    return name

def unique(pairs):
    result = {}
    for key,value in pairs:
        if key in result: raise ValueError('Duplicate JSON field')
        result[key] = value
    return result

def verify_archive(path, output=None):
    with zipfile.ZipFile(path) as bundle:
        entries = bundle.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)) or not 1 <= len(names) <= 512:
            raise ValueError('Duplicate or excessive archive members')
        if sum(entry.file_size for entry in entries) > MAX_TOTAL:
            raise ValueError('Archive exceeds expanded byte budget')
        for entry in entries:
            safe_path(entry.filename)
            if (entry.file_size > MAX_FILE or entry.flag_bits & 1
                    or stat.S_IFMT(entry.external_attr >> 16) != stat.S_IFREG
                    or entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
                raise ValueError('Unsafe archive member')
        if 'inventory.json' not in names or bundle.getinfo('inventory.json').file_size > 1024*1024:
            raise ValueError('Missing or oversized inventory')
        inventory = json.loads(bundle.read('inventory.json'), object_pairs_hook=unique)
        if inventory.get('schema') != 'dvidia.probe-evidence-inventory/v1' or type(inventory.get('files')) is not list:
            raise ValueError('Unknown inventory')
        records = {}
        for row in inventory['files']:
            if type(row) is not dict or set(row) != {'path', 'bytes', 'sha256'}:
                raise ValueError('Invalid inventory row')
            name = safe_path(row['path'])
            if (name in records or name == 'inventory.json' or type(row['bytes']) is not int
                    or not 0 <= row['bytes'] <= MAX_FILE
                    or type(row['sha256']) is not str or not re.fullmatch(r'[0-9a-f]{64}',row['sha256'])):
                raise ValueError('Invalid or duplicate identity')
            records[name] = row
        if set(names) != set(records) | {'inventory.json'}:
            raise ValueError('Inventory roster differs')
        for name in records:
            parts = name.split('/')
            if any('/'.join(parts[:index]) in records for index in range(1,len(parts))):
                raise ValueError('File and directory paths collide')
        for name,row in records.items():
            if bundle.getinfo(name).file_size != row['bytes']:
                raise ValueError('Archive size differs from inventory')
            digest,size = sha256(),0
            with bundle.open(name) as stream:
                while chunk := stream.read(65536):
                    size += len(chunk)
                    if size > row['bytes']: raise ValueError('Member exceeds expected size')
                    digest.update(chunk)
            if size != row['bytes'] or digest.hexdigest() != row['sha256']:
                raise ValueError('Archive digest differs from inventory')
        if output is not None:
            root = Path(output)
            if root.exists() or root.is_symlink(): raise ValueError('Output must be new')
            root.mkdir(parents=True)
            resolved = root.resolve()
            for name in sorted(names):
                target = root/name
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.parent.resolve().is_relative_to(resolved): raise ValueError('Output escape')
                with bundle.open(name) as source,target.open('xb') as sink:
                    while chunk := source.read(65536): sink.write(chunk)
        return {'verified':True,'files':len(records),'logical_bytes':sum(r['bytes'] for r in records.values()),
                'reconstructed':output is not None}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    print(json.dumps(verify_archive(args.archive,args.output),sort_keys=True))

if __name__=='__main__': main()
'''


def encoded(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()+b'\n'


def identity(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Expected regular evidence file')
    digest, size = sha256(), 0
    with path.open('rb') as stream:
        while chunk := stream.read(65536):
            size += len(chunk)
            if size > MAX_FILE:
                raise ValueError('Evidence file exceeds byte bound')
            digest.update(chunk)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def safe_path(name):
    if (type(name) is not str or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/-]{0,399}', name)
            or any(part in ('', '.', '..') for part in name.split('/'))):
        raise ValueError('Unsafe evidence path')
    return name


def tree(root):
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Evidence root must be a regular directory')
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Evidence links are forbidden')
        if path.is_dir():
            continue
        name = safe_path(path.relative_to(root).as_posix())
        result[name] = identity(path)
    return result


def scan_stream(stream):
    size, tail = 0, b''
    while chunk := stream.read(65536):
        size += len(chunk)
        if size > MAX_FILE:
            raise ValueError('Decoded scan exceeds its byte bound')
        data = tail+chunk
        if any(prefix in data for prefix in PRIVATE) or any(pattern.search(data) for pattern in SECRETS):
            raise ValueError('Private path or credential-shaped bytes in public evidence')
        tail = data[-256:]
    return size


def scan_file(path):
    with path.open('rb') as stream:
        scan_stream(stream)
    if path.suffix == '.gz':
        with gzip.open(path, 'rb') as stream:
            raw = stream.read(MAX_FILE+1)
        if len(raw) > MAX_FILE:
            raise ValueError('Gzip expands beyond byte bound')
        scan_stream(io.BytesIO(raw))
        decoded = json.dumps(json.loads(raw), ensure_ascii=False, allow_nan=False).encode()
        scan_stream(io.BytesIO(decoded))
    elif path.suffix == '.json':
        decoded = json.dumps(json.loads(path.read_bytes()), ensure_ascii=False, allow_nan=False).encode()
        scan_stream(io.BytesIO(decoded))
    elif path.suffix == '.npz':
        import numpy as np
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if (len(entries) != 22 or len({row.filename for row in entries}) != 22
                    or sum(row.file_size for row in entries) > MAX_FILE
                    or any(not re.fullmatch(r'[a-z0-9_]+\.npy', row.filename) or row.file_size > 2*1024*1024
                           for row in entries)):
                raise ValueError('Unsupported numeric archive')
            for row in entries:
                with archive.open(row) as stream:
                    scan_stream(stream)
        with np.load(path, allow_pickle=False) as arrays:
            for name in arrays.files:
                values = arrays[name]
                if (values.dtype != np.dtype('float64') or values.ndim not in (1, 2)
                        or not np.isfinite(values).all() or not 1 <= values.shape[0] <= 30000):
                    raise ValueError('Expected bounded finite numeric-only sidecar')


def frozen_inputs(repo, study, audit):
    audit_raw = (audit/'audit.json').read_bytes()
    if sha256(audit_raw).hexdigest() != AUDIT_SHA256:
        raise ValueError('Independent audit identity changed')
    proof = json.loads(audit_raw)
    if not proof['passed'] or not proof['raw_files_unchanged'] or not proof['source_files_unchanged']:
        raise ValueError('Independent audit did not pass')
    if tree(study) != proof['frozen_raw_files']:
        raise ValueError('Study roster or bytes differ from the pinned independent audit')
    protocol = json.loads((study/'protocol.json').read_bytes())
    if sha256((study/'protocol.json').read_bytes()).hexdigest() != proof['protocol_sha256']:
        raise ValueError('Frozen protocol differs')
    for name, digest in protocol['source_sha256'].items():
        if identity(repo/'src/dvidia_training'/name)['sha256'] != digest:
            raise ValueError('Frozen runtime source changed')
    audit_files = tree(audit)
    if set(audit_files) != {'audit.py', 'audit.json', 'guarded-audit.json', 'guarded-invocation.json'}:
        raise ValueError('Unexpected independent-audit roster')
    if audit_files['audit.py'] != proof['script'] or audit_files['guarded-audit.json'] != audit_files['audit.json']:
        raise ValueError('Original or guarded audit bytes differ')
    guard = json.loads((audit/'guarded-invocation.json').read_bytes())
    if (guard['optimization_level'] != 0 or guard['assertions_enabled'] is not True
            or guard['original_audit_sha256'] != proof['script']['sha256']
            or guard['guard_sha256'] != identity(repo/'benchmarks/piper-probe-evidence-v01/audit_guarded.py')['sha256']
            or guard['readback_output'] != audit_files['guarded-audit.json']):
        raise ValueError('Guarded normal-mode invocation differs')
    expected = {'study/'+name: record for name, record in proof['frozen_raw_files'].items()}
    expected.update({'audit/'+name: record for name, record in audit_files.items()})
    expected['audit/audit.json'] = {'bytes': len(audit_raw), 'sha256': AUDIT_SHA256}
    return proof, expected


def write_archive(path, files, generated, expected):
    records = []
    for name, source in sorted(files.items()):
        safe_path(name)
        record = identity(source)
        if name in expected and record != expected[name]:
            raise ValueError('Frozen evidence changed before indexing: '+name)
        scan_file(source)
        records.append({'path': name, **record})
    for name, raw in sorted(generated.items()):
        safe_path(name)
        if name in files:
            raise ValueError('Generated path collision')
        scan_stream(io.BytesIO(raw))
        records.append({'path': name, 'bytes': len(raw), 'sha256': sha256(raw).hexdigest()})
    records.sort(key=lambda row: row['path'])
    if len(records) > 511 or sum(row['bytes'] for row in records) > MAX_TOTAL-1024*1024:
        raise ValueError('Archive exceeds total evidence bound')
    inventory = {'schema': 'dvidia.probe-evidence-inventory/v1', 'files': records,
                 'source_commit': SOURCE_COMMIT,
                 'scope': 'Inventory hashes every file except inventory.json itself. The external package receipt hashes the complete ZIP.'}
    content = {**files, **generated, 'inventory.json': encoded(inventory)}
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, source in sorted(content.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_STORED if isinstance(source, Path) and source.suffix in ('.gz', '.npz') else zipfile.ZIP_DEFLATED
            digest, size = sha256(), 0
            with archive.open(info, 'w', force_zip64=True) as sink:
                with source.open('rb') if isinstance(source, Path) else io.BytesIO(source) as stream:
                    while chunk := stream.read(65536):
                        size += len(chunk)
                        if size > MAX_FILE:
                            raise ValueError('File grew during packaging')
                        digest.update(chunk)
                        sink.write(chunk)
            if name != 'inventory.json':
                record = next(row for row in records if row['path'] == name)
                if record['bytes'] != size or record['sha256'] != digest.hexdigest():
                    raise ValueError('File changed while copying: '+name)
    return inventory


def archive_identity(path):
    digest, size = sha256(), 0
    with path.open('rb') as stream:
        while chunk := stream.read(65536):
            size += len(chunk)
            if size > MAX_TOTAL:
                raise ValueError('ZIP exceeds its publication byte bound')
            digest.update(chunk)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def prepare(repo, study, audit, output):
    if output.exists() or output.is_symlink():
        raise ValueError('Package output must be new')
    proof, expected = frozen_inputs(repo, study, audit)
    files = {'study/'+name: study/name for name in proof['frozen_raw_files']}
    files.update({'audit/'+name: audit/name for name in ('audit.py', 'audit.json', 'guarded-audit.json', 'guarded-invocation.json')})
    files.update({'benchmarks/piper-probe-evidence-v01/README.md': repo/'benchmarks/piper-probe-evidence-v01/README.md',
                  'benchmarks/piper-probe-evidence-v01/render_report.py': repo/'benchmarks/piper-probe-evidence-v01/render_report.py',
                  'tools/package_evidence.py': Path(__file__),
                  'tools/test_package_evidence.py': repo/'benchmarks/piper-probe-evidence-v01/test_package_evidence.py',
                  'tools/audit_guarded.py': repo/'benchmarks/piper-probe-evidence-v01/audit_guarded.py',
                  'licenses/agilex_piper-LICENSE.txt': repo/'third_party/robot-models/agilex_piper-LICENSE.txt',
                  'licenses/mujoco-LICENSE.txt': repo/'third_party/mujoco/LICENSE.txt',
                  'licenses/mujoco-THIRD_PARTY_NOTICES.txt': repo/'third_party/mujoco/THIRD_PARTY_NOTICES.txt'})
    readme = b'''# DVIDIA | PiPER probe evidence v0.1

Simulation-only authored short-lift/hold research on 12 fixed synthetic scenes,
once per controller (24 attempts). Independent reconstruction checked 480,000
native 1 ms rows. Evidence gate: 6 admissions, 0/6 false initial qualifications;
four later hold losses, final 2/12 qualified. Timer: 12 admissions, 8/12 false
initial qualifications; final 2/12 qualified. Closed-loop trajectories can differ.
These counts do not establish general accuracy, speed or hardware safety.

The historical learned 5/12 placement result is unchanged. No training, model
weights, full transport/placement or physical sensor qualification is included.
The 10 N per-finger ceiling is an arbitrary synthetic assumption, not calibration.

Paths under study/ preserve every frozen study artifact. Paths under audit/
preserve the independent original audit and its receipt, plus a guarded normal-
mode reanalysis. The original audit used assertions and was executed with normal
Python. Always use tools/audit_guarded.py for reuse: it refuses -O/-OO and pins
the original script. The guarded reanalysis was byte-identical to the original.

After checking the release ZIP checksum, copy tools/verify_archive.py from the
ZIP or reviewed repository and verify/reconstruct it with standard Python:

    python verify_archive.py piper-probe-evidence-v01.zip --output evidence

With the project dependencies and frozen checkout available, re-read the saved
records (no physics rerun), writing new outputs outside the archived study:

    python evidence/tools/audit_guarded.py --audit-script evidence/audit/audit.py --guard-receipt new-guard.json --study evidence/study --repo /path/to/frozen-checkout --output new-audit.json

Frozen source: https://github.com/Dvidia-Inference/dvidia-training/tree/a6a1d15db905214da52eb3942ebead3f85254762
Research article (publication target): https://research.dvidia.org/papers/piper-probe-evidence-v01/
Source release (publication target): https://github.com/Dvidia-Inference/dvidia-training/releases/tag/piper-probe-evidence-v0.1-preview
Evidence dataset (publication target): https://huggingface.co/datasets/Dvidia/piper-probe-evidence-v01

Original DVIDIA code and authored synthetic records are MIT. Upstream notices
remain included. No robot meshes, simulator binaries or credentials are bundled.
DVIDIA research: https://research.dvidia.org | hello@dvidia.org
'''
    generated = {'README.md': readme, 'LICENSE': (repo/'LICENSE').read_bytes(),
                 'tools/verify_archive.py': VERIFY_SOURCE.encode(),
                 'SOURCE.json': encoded({'frozen_commit': SOURCE_COMMIT,
                                         'protocol_sha256': proof['protocol_sha256'],
                                         'independent_audit_sha256': AUDIT_SHA256,
                                         'normal_mode_reanalysis': 'audit/guarded-invocation.json'})}
    output.mkdir(parents=True)
    stage = output/'huggingface'; stage.mkdir()
    archive = stage/ARCHIVE_NAME
    inventory = write_archive(archive, files, generated, expected)
    verify_namespace = {'__name__': 'dvidia_evidence_verifier'}
    exec(compile(VERIFY_SOURCE, 'verify_archive.py', 'exec'), verify_namespace)
    verified = verify_namespace['verify_archive'](archive)
    frozen_inputs(repo, study, audit)
    archive_id = archive_identity(archive)
    card = b'''---
license: mit
pretty_name: DVIDIA PiPER probe evidence v0.1
task_categories:
- robotics
tags:
- robotics
- simulation
- piper
- mujoco
- cpu
- research
---

'''+readme+b'\nArchive SHA-256: `'+archive_id['sha256'].encode()+b'`\n'
    payload = {'README.md': card, 'LICENSE': generated['LICENSE'],
               **{name: (study/name).read_bytes() for name in ('report.json', 'protocol.json', 'inspection.json')}}
    for name, raw in payload.items():
        scan_stream(io.BytesIO(raw))
        if name in ('report.json', 'protocol.json', 'inspection.json') and {'bytes': len(raw), 'sha256': sha256(raw).hexdigest()} != expected['study/'+name]:
            raise ValueError('Frozen top-level payload changed')
        with (stage/name).open('xb') as stream:
            stream.write(raw)
    if set(path.name for path in stage.iterdir()) != HF_NAMES:
        raise ValueError('Hugging Face payload must contain the exact six files')
    receipt = {'schema': 'dvidia.probe-evidence-package/v1', 'remote_operations': False,
               'source_commit': SOURCE_COMMIT, 'target_dataset': 'Dvidia/piper-probe-evidence-v01',
               'archive': {'filename': ARCHIVE_NAME, **archive_id}, 'verified': verified,
               'frozen_study_files': len(proof['frozen_raw_files']), 'frozen_inputs_unchanged': True,
               'scan': {'passed': True, 'decoded_gzip_json': True, 'numeric_only_npz': True,
                        'source_and_text_paths_and_credential_shapes': True},
               'guarded_reanalysis': json.loads((audit/'guarded-invocation.json').read_bytes()),
               'payload': [{'path': path.name, **(archive_id if path.name == ARCHIVE_NAME else identity(path))}
                           for path in sorted(stage.iterdir())]}
    (output/'inventory.json').write_bytes(encoded(inventory))
    (output/'package-receipt.json').write_bytes(encoded(receipt))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.repo, args.study, args.audit, args.output)
    print(json.dumps({'archive': result['archive'], 'verified': result['verified']}))


if __name__ == '__main__':
    main()

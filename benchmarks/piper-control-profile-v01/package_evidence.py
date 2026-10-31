"""Deterministic, offline packaging of reviewed synthetic control-profile evidence.

Logical files retain their original bytes, represented once per SHA256 blob.
This does not rerun physics, change frozen evidence, publish, or create weights.
"""
from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

SCHEMA = 'dvidia.evidence-blobs/v1'
AUDIT_SHA = '1a4f4a1093961b59c03a3a244d123afed35f5a5f77c98d1467a393a20c2d6fd4'
MAX_BLOB = 128*1024*1024
MAX_EXPANDED = 512*1024*1024
MAX_ARCHIVE = 2*1024*1024*1024
# Split test prefixes so this packaged audit source is not itself a path leak.
PRIVATE = (b'/'+b'Users/', b'/private/'+b'tmp/', b'/opt/'+b'homebrew/', b'file:///'+b'Users/')
SECRETS = tuple(re.compile(value) for value in (
    rb'hf_[A-Za-z0-9]{20,}', rb'gh[pousr]_[A-Za-z0-9]{20,}',
    rb'apikey_[A-Za-z0-9_]{20,}', rb'AKIA[A-Z0-9]{16}'))

VERIFY_SOURCE = r'''"""Verify or reconstruct a DVIDIA content-addressed evidence archive, offline.

No archive code is imported, no pickle is read, and extractall is never used.
"""
from hashlib import sha256
import argparse
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

MAX_BLOB = 128*1024*1024
MAX_TOTAL = 2*1024*1024*1024
MAX_FILES = 4096
HEADER_FILES = {'README.md', 'LICENSE', 'verify_extract.py', 'manifest.json'}

def safe_path(value):
    if (type(value) is not str or len(value) > 400 or not value
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/-]*', value)
            or any(part in ('', '.', '..') for part in value.split('/'))):
        raise ValueError('Unsafe logical path')
    return value

def unique(rows):
    value = {}
    for key, item in rows:
        if key in value: raise ValueError('Duplicate JSON field')
        value[key] = item
    return value

def decode(raw):
    def bad(token): raise ValueError('Nonfinite JSON: '+token)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=bad)

def verify(archive, output=None):
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)) or len(names) > MAX_FILES:
            raise ValueError('Duplicate or excessive archive members')
        if sum(entry.file_size for entry in entries) > MAX_TOTAL:
            raise ValueError('Archive expands beyond the total budget')
        for entry in entries:
            if (entry.filename not in HEADER_FILES
                    and not re.fullmatch(r'blobs/[0-9a-f]{64}', entry.filename)):
                raise ValueError('Unexpected archive member')
            if (entry.file_size > MAX_BLOB or entry.flag_bits & 1
                    or stat.S_IFMT(entry.external_attr >> 16) == stat.S_IFLNK
                    or entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
                raise ValueError('Unsafe archive member or expansion budget')
        if not HEADER_FILES.issubset(names) or bundle.getinfo('manifest.json').file_size > 8*1024*1024:
            raise ValueError('Missing or oversized archive manifest')
        manifest = decode(bundle.read('manifest.json'))
        if (manifest.get('schema') != 'dvidia.evidence-blobs/v1'
                or type(manifest.get('files')) is not list
                or not 1 <= len(manifest['files']) <= MAX_FILES):
            raise ValueError('Unknown evidence manifest')
        paths, blobs, records = set(), {}, {}
        total_logical = 0
        for row in manifest['files']:
            if type(row) is not dict or set(row) != {'path', 'bytes', 'sha256', 'blob'}:
                raise ValueError('Invalid logical file record')
            path = safe_path(row['path'])
            digest = row['sha256']
            if (path in paths or type(digest) is not str
                    or not re.fullmatch(r'[0-9a-f]{64}', digest)
                    or type(row['bytes']) is not int or not 0 <= row['bytes'] <= MAX_BLOB
                    or row['blob'] != 'blobs/'+digest):
                raise ValueError('Duplicate path or invalid blob identity')
            paths.add(path)
            records[path] = row
            total_logical += row['bytes']
            if row['blob'] in blobs and blobs[row['blob']] != row['bytes']:
                raise ValueError('Conflicting blob lengths')
            blobs[row['blob']] = row['bytes']
        if total_logical > MAX_TOTAL:
            raise ValueError('Logical reconstruction exceeds the total budget')
        if any(str(parent) in paths for path in paths for parent in PurePosixPath(path).parents
               if str(parent) != '.'):
            raise ValueError('File and directory paths collide')
        if set(names) != HEADER_FILES | set(blobs):
            raise ValueError('Missing or unreferenced blob')
        for name, expected_size in blobs.items():
            if bundle.getinfo(name).file_size != expected_size:
                raise ValueError('Blob size mismatch')
            digest, size = sha256(), 0
            with bundle.open(name) as stream:
                while chunk := stream.read(65536):
                    size += len(chunk)
                    if size > expected_size: raise ValueError('Blob exceeds declared size')
                    digest.update(chunk)
            if size != expected_size or digest.hexdigest() != name[6:]:
                raise ValueError('Blob digest mismatch')
        for physical, logical in (('README.md', 'README.md'), ('LICENSE', 'LICENSE'),
                                  ('verify_extract.py', 'tools/verify_extract.py')):
            row = records.get(logical)
            content = bundle.read(physical)
            if row is None or len(content) != row['bytes'] or sha256(content).hexdigest() != row['sha256']:
                raise ValueError('Header differs from its logical inventory')
        if output is not None:
            destination = Path(output)
            if destination.exists() or destination.is_symlink():
                raise ValueError('Reconstruction destination must be new')
            destination.mkdir(parents=True, exist_ok=False)
            root = destination.resolve()
            for path in sorted(paths):
                target = destination.joinpath(*PurePosixPath(path).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.parent.resolve().is_relative_to(root):
                    raise ValueError('Reconstruction escaped its destination')
                with bundle.open(records[path]['blob']) as source, target.open('xb') as sink:
                    while chunk := source.read(65536): sink.write(chunk)
        return {'verified': True, 'logical_files': len(paths), 'unique_blobs': len(blobs),
                'logical_bytes': total_logical, 'unique_blob_bytes': sum(blobs.values()),
                'reconstructed': output is not None}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.archive, args.output), sort_keys=True))

if __name__ == '__main__': main()
'''


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def pretty(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()+b'\n'


def file_identity(path, maximum=MAX_BLOB):
    digest, size = sha256(), 0
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024*1024):
            digest.update(chunk)
            size += len(chunk)
            if size > maximum: raise ValueError('File exceeds the packaging budget')
    return {'bytes': size, 'sha256': digest.hexdigest()}


def scan_stream(stream, maximum=MAX_EXPANDED):
    tail, size = b'', 0
    while chunk := stream.read(65536):
        size += len(chunk)
        if size > maximum: raise ValueError('Audit scan exceeded its expansion budget')
        data = tail+chunk
        if any(pattern in data for pattern in PRIVATE) or any(pattern.search(data) for pattern in SECRETS):
            raise ValueError('Private path or credential-shaped content in public payload')
        tail = data[-256:]
    return size


def scan_file(path):
    with Path(path).open('rb') as stream: scan_stream(stream, MAX_BLOB)
    if path.suffix == '.gz':
        with gzip.open(path, 'rb') as stream: scan_stream(stream)
    elif path.suffix == '.npz':
        with zipfile.ZipFile(path) as archive:
            if sum(row.file_size for row in archive.infolist()) > MAX_EXPANDED:
                raise ValueError('Numeric archive exceeds scan budget')
            for row in archive.infolist():
                with archive.open(row) as stream: scan_stream(stream)


def safe_path(path):
    if (type(path) is not str or not path or len(path) > 400
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/-]*', path)
            or any(part in ('', '.', '..') for part in path.split('/'))):
        raise ValueError('Unsafe logical evidence path')
    return path


def check_frozen_evidence(root, audit):
    """Require the audit's exact scientific roster and bytes, not fresh claims."""
    expected = audit.get('frozen_evidence')
    if type(expected) is not dict or not expected:
        raise ValueError('Missing audited frozen evidence inventory')
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Frozen evidence root must be a regular directory')
    for name, record in expected.items():
        safe_path(name)
        if (type(record) is not dict or set(record) != {'bytes', 'sha256'}
                or type(record['bytes']) is not int or not 0 <= record['bytes'] <= MAX_BLOB
                or type(record['sha256']) is not str
                or not re.fullmatch(r'[0-9a-f]{64}', record['sha256'])):
            raise ValueError('Malformed audited frozen file identity')
    actual = set()
    for source in root.rglob('*'):
        if source.is_symlink(): raise ValueError('Frozen evidence contains a symlink')
        if source.is_dir(): continue
        if not source.is_file(): raise ValueError('Frozen evidence contains a nonregular file')
        actual.add(source.relative_to(root).as_posix())
    if actual != set(expected) | {'readback-audit.json'}:
        raise ValueError('Frozen evidence roster differs from the pinned audit')
    for name, record in expected.items():
        if file_identity(root/name) != record:
            raise ValueError('Frozen evidence differs from the pinned audit: '+name)
    return {name: dict(record) for name, record in expected.items()}


def zip_entry(name, content=None, source=None):
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def build_archive(output, files, generated, excluded, expected_identities=None):
    """Build a deterministic archive; originals are rehashed while copying."""
    logical, blobs, scanned = [], {}, set()
    for name, source in sorted(files.items()):
        safe_path(name)
        if source.is_symlink() or not source.is_file(): raise ValueError('Only regular evidence files are accepted')
        identity = file_identity(source)
        if name in (expected_identities or {}) and identity != expected_identities[name]:
            raise ValueError('Frozen evidence changed before archive indexing: '+name)
        if identity['sha256'] not in scanned:
            scan_file(source)
            scanned.add(identity['sha256'])
        logical.append({'path': name, **identity, 'blob': 'blobs/'+identity['sha256']})
        blobs.setdefault(identity['sha256'], source)
    for name, content in sorted(generated.items()):
        safe_path(name)
        if name in files: raise ValueError('Generated and original logical paths collide')
        import io
        scan_stream(io.BytesIO(content), MAX_BLOB)
        digest = sha256(content).hexdigest()
        logical.append({'path': name, 'bytes': len(content), 'sha256': digest, 'blob': 'blobs/'+digest})
        blobs.setdefault(digest, content)
    logical.sort(key=lambda row: row['path'])
    manifest = {'schema': SCHEMA, 'files': logical, 'excluded_originals': excluded,
                'layout': 'Original logical bytes reconstructed from SHA256 blobs; identical payloads share one blob.',
                'metadata_scope': 'manifest.json describes every logical file; it does not hash itself. Direct README/LICENSE/verifier copies match their logical inventory.',
                'license': 'Existing project LICENSE retained; upstream notices included. No robot asset meshes, simulator binaries or learned model weights bundled.'}
    manifest_bytes = pretty(manifest)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=True) as archive:
        for name, content in [('README.md', generated['README.md']), ('LICENSE', generated['LICENSE']),
                              ('verify_extract.py', generated['tools/verify_extract.py']), ('manifest.json', manifest_bytes)]:
            archive.writestr(zip_entry(name), content, compresslevel=9)
        for digest, source in sorted(blobs.items()):
            info = zip_entry('blobs/'+digest)
            # Gzip/NPZ data are already compressed. Fixed STORE avoids needless
            # second compression while preserving exact original bytes.
            if isinstance(source, Path) and source.suffix in ('.gz', '.npz'):
                info.compress_type = zipfile.ZIP_STORED
            with archive.open(info, 'w', force_zip64=True) as sink:
                checksum = sha256()
                if isinstance(source, bytes):
                    sink.write(source); checksum.update(source)
                else:
                    with source.open('rb') as stream:
                        while chunk := stream.read(1024*1024): sink.write(chunk); checksum.update(chunk)
                if checksum.hexdigest() != digest: raise ValueError('Evidence changed while packaging')
    return manifest


def prepare(repo, comparison, profile, output):
    if output.exists() or output.is_symlink(): raise ValueError('Release output must be new')
    audit_raw = (comparison/'readback-audit.json').read_bytes()
    if sha256(audit_raw).hexdigest() != AUDIT_SHA: raise ValueError('Supplemental audit identity changed')
    audit = json.loads(audit_raw)
    original_raw = (comparison/'comparison.json').read_bytes()
    original = json.loads(original_raw)
    if (not audit['passed'] or not audit['evidence_unchanged']
            or not audit['eligible_under_predeclared_rule_after_complete_readback']
            or original['all_passed'] is not False
            or file_identity(comparison/'comparison.json') != audit['original_report']):
        raise ValueError('Original fail-closed report or supplemental audit differs')
    frozen = check_frozen_evidence(comparison, audit)
    expected_identities = {'comparison/'+name: record for name, record in frozen.items()}
    expected_identities['comparison/readback-audit.json'] = {'bytes': len(audit_raw), 'sha256': AUDIT_SHA}
    files, excluded = {}, []
    for prefix, root in (('comparison', comparison), ('profiling', profile)):
        for source in sorted(root.rglob('*')):
            if source.is_dir(): continue
            name = prefix+'/'+source.relative_to(root).as_posix()
            if source.is_symlink(): raise ValueError('Evidence may not contain symlinks')
            if source.name == 'raw.pstats':
                excluded.append({'path': name, **file_identity(source),
                                 'reason': 'Original raw profiler format contains local source paths. Portable stripped-directory stats and complete JSON tables are included.'})
                continue
            if source.suffix == '.pyc' or '__pycache__' in source.parts:
                excluded.append({'path': name, **file_identity(source), 'reason': 'Interpreter cache is not research evidence'})
                continue
            try:
                if source.suffix == '.txt' or 'traceback' in source.name:
                    scan_file(source)
            except ValueError:
                if source.suffix == '.txt' or 'traceback' in source.name:
                    excluded.append({'path': name, **file_identity(source), 'reason': 'Local log contains private paths or credential-shaped content'})
                    continue
                raise
            files[name] = source
    extras = {
        'tools/profile.py': repo/'benchmarks/piper-control-profile-v01/profile.py',
        'tools/compare.py': repo/'benchmarks/piper-control-profile-v01/compare.py',
        'tools/package_evidence.py': Path(__file__),
        'tools/README.md': repo/'benchmarks/piper-control-profile-v01/README.md',
        'tools/readback_audit.py': repo/'benchmarks/piper-control-profile-v01/readback_audit.py',
        'tests/test_profile.py': repo/'benchmarks/piper-control-profile-v01/test_profile.py',
        'tests/test_package_evidence.py': repo/'benchmarks/piper-control-profile-v01/test_package_evidence.py',
        'tests/test_piper_control_compare.py': repo/'tests/test_piper_control_compare.py',
        'tests/test_piper_contact_scratch.py': repo/'tests/test_piper_contact_scratch.py',
        'licenses/agilex_piper-LICENSE.txt': repo/'third_party/robot-models/agilex_piper-LICENSE.txt',
        'licenses/mujoco-LICENSE.txt': repo/'third_party/mujoco/LICENSE.txt',
        'licenses/mujoco-THIRD_PARTY_NOTICES.txt': repo/'third_party/mujoco/THIRD_PARTY_NOTICES.txt',
    }
    files.update(extras)
    pooled = original['adoption']['pooled']
    summary = {'schema': 'dvidia.piper-control-profile-public-summary/v1', 'source_branch': 'feat/piper-control-profile-v01',
               'scope': original['scope'], 'timing_pairs': 2, 'planned_attempts': original['planned_attempts'],
               'timing': pooled, 'paired_timing': original['paired_timing'],
               'original_report': {'all_passed': False, **file_identity(comparison/'comparison.json'),
                                   'reason': 'Original dense reader rejected three case pairs exceeding its 64 MiB expansion bound.'},
               'supplemental_readback': {'passed': True, **file_identity(comparison/'readback-audit.json'),
                                         'expanded_bound_bytes': audit['expanded_bound_bytes'],
                                         'total_dense_snapshots_compared': audit['total_dense_snapshots_compared'],
                                         'evidence_unchanged': True,
                                         'eligible_under_predeclared_rule_after_complete_readback': True},
               'historical_learned_heldout_result': {'successes': 5, 'attempts': 12, 'unchanged': True},
               'production_promotion': False, 'new_wheel': False, 'physical_robot_ready': False,
               'limitations': ['Two matched shared-host pairs on one fixed synthetic command tape.',
                               'No learned inference in timed environment-step measurements.',
                               'Dense diagnostics are separate from timing runs.',
                               'No new held-out accuracy, physical calibration, recovery-robustness or real-time claim.',
                               'Scheduled jaw pulse also succeeds; it is perturbation-equivalence coverage.']}
    readme = b'''# DVIDIA | PiPER control profiling v0.1\n\nThis archive preserves one bounded allocation-only optimization experiment, its\noriginal fail-closed report, and a separate complete readback of the same evidence.\nIt contains synthetic simulator traces and development tools, not model weights.\n\nTwo matched AB/BA pairs observed 15.7% lower pooled mean environment-step wall\ncompute and 12.6% lower process CPU. These are shared-host observations on one\npreserved command tape, with no policy inference in the timed steps. They do not\nestablish general speedup, real-time operation or physical robot performance.\nThe historical learned held-out result remains 5/12. No new wheel or production\npromotion is part of this evidence release.\n\nAll 14 scheduled attempts are retained: four timing attempts plus five dense\nconditions for both source variants. The original comparison.json remains false\nbecause three dense JSON pairs exceeded its 64 MiB reader bound. The supplemental\nreadback-audit.json uses a 512 MiB bound and confirms exact numeric state, contacts,\ncadence and scorer equivalence across 200,010 saved snapshots, without rerunning\nphysics or timing. The fixed pulse also succeeds; this is not recovery validation.\n\nArchive layout is content-addressed: manifest.json maps every logical path to\nits exact byte count and SHA256 blob. Identical payloads share one blob. Original\nraw.pstats files are excluded because they contain local paths; portable stripped\nstats and complete JSON profiling tables remain. Any excluded logs are listed.\n\nCopy the top-level verify_extract.py from this archive after checking the release\narchive checksum. It uses only Python's standard library:\n\n```sh\npython verify_extract.py piper-control-profile-v01-evidence.zip\npython verify_extract.py piper-control-profile-v01-evidence.zip --output evidence\n```\n\nThe helper rejects traversal, symlinks, duplicate members, unknown blobs, oversized\ncontent and digest mismatches. It never imports archive code or reads pickle.\nReconstruction restores the original logical paths and bytes. Raw native NPZ\narrays and canonical gzip contact/scorer traces remain unchanged.\n\nSource branch: feat/piper-control-profile-v01. Frozen source snapshots, profiling\nand comparison harnesses, tests, the independent readback verifier and upstream\nnotices are included. The project's existing MIT LICENSE is retained. Robot asset\nmeshes, simulator binaries, credentials and learned model weights are omitted.\n\nDVIDIA research: https://research.dvidia.org | hello@dvidia.org\n'''
    generated = {'README.md': readme, 'LICENSE': (repo/'LICENSE').read_bytes(),
                 'tools/verify_extract.py': VERIFY_SOURCE.encode(), 'summary.json': pretty(summary)}
    output.mkdir(parents=True, exist_ok=False)
    stage = output/'huggingface'
    stage.mkdir()
    archive = stage/'piper-control-profile-v01-evidence.zip'
    manifest = build_archive(archive, files, generated, excluded, expected_identities)
    namespace = {'__name__': 'dvidia_evidence_verifier'}
    exec(compile(VERIFY_SOURCE, 'verify_extract.py', 'exec'), namespace)
    verified = namespace['verify'](archive)
    check_frozen_evidence(comparison, audit)
    if file_identity(comparison/'readback-audit.json') != expected_identities['comparison/readback-audit.json']:
        raise ValueError('Pinned readback receipt changed while packaging')
    summary['archive'] = {'filename': archive.name, **file_identity(archive, MAX_ARCHIVE),
                          'logical_files': verified['logical_files'], 'unique_blobs': verified['unique_blobs'],
                          'logical_bytes': verified['logical_bytes'], 'unique_blob_bytes': verified['unique_blob_bytes']}
    card = ('''---\nlicense: mit\npretty_name: DVIDIA PiPER control profiling v0.1\ntask_categories:\n- robotics\ntags:\n- robotics\n- simulation\n- mujoco\n- piper\n- cpu\n- reproducibility\n---\n\n''').encode()+readme
    payload = {'README.md': card, 'LICENSE': generated['LICENSE'], 'summary.json': pretty(summary),
               'comparison.json': original_raw, 'readback-audit.json': audit_raw}
    for name, content in payload.items(): (stage/name).write_bytes(content)
    if set(path.name for path in stage.iterdir()) != set(payload) | {archive.name}:
        raise ValueError('Hugging Face staging must contain exactly six files')
    for path in stage.iterdir():
        if path.suffix != '.zip': scan_file(path)
    receipt = {'schema': 'dvidia.control-profile-evidence-package/v1',
               'target_dataset': 'Dvidia/piper-control-profile-v01', 'remote_operations': False,
               'archive': summary['archive'], 'verified': verified,
               'frozen_comparison_verified': {'files': len(frozen), 'roster_exact': True,
                                               'audit_sha256': AUDIT_SHA,
                                               'checked_before_indexing_during_copy_and_after_archive': True},
               'scan': {'passed': True, 'scope': 'Raw blobs, decoded gzip JSON, numeric ZIP members and staged text; private local-path and credential-prefix checks.',
                        'excluded_originals': excluded},
               'staged_files': [{'path': path.name, **file_identity(path, MAX_ARCHIVE)} for path in sorted(stage.iterdir())]}
    (output/'manifest.json').write_bytes(pretty(manifest))
    (output/'package-receipt.json').write_bytes(pretty(receipt))
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path('.'))
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = prepare(args.repo_root, args.comparison, args.profile, args.output)
    print(json.dumps(receipt['archive'], sort_keys=True))


if __name__ == '__main__': main()

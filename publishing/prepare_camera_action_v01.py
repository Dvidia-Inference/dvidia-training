"""Build deterministic raw-evidence shards from a stopped, sealed local study.

This is an offline packaging command. It performs no simulation, training,
credential lookup or upload, and never changes study artifacts. A diagnostic
audit remains explicitly diagnostic-only when a larger study is packaged.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile
import zipfile

from huggingface.camera_action_hardware import MAX_BYTES as MAX_HARDWARE_BYTES, validate_hardware

SCHEMA = 'dvidia.camera-action-evidence/v1'
MAX_FILE = 256 * 1024 ** 2
# The complete reviewed repository includes historical benchmark evidence.
# Keep this aligned with the uploader's per-file cap; expanded/member limits
# remain independent below (2 GiB total, 256 MiB per member).
MAX_SOURCE = 1024 ** 3
MAX_EPISODE = 1024 ** 3
MAX_TOTAL = 120 * 1024 ** 3
DISK_RESERVE = 30 * 1024 ** 3
ZIP_LEVEL = 1
TEXT_SUFFIXES = {'.py', '.js', '.ts', '.json', '.html', '.md', '.txt', '.toml', '.yaml', '.yml'}
SECRET = re.compile(rb'(?:hf_|gh[pousr]_|apikey_)[A-Za-z0-9_]{24,}')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def plain(path):
    path = Path(path).absolute()
    if any(p.is_symlink() and str(p) not in ('/tmp', '/var') for p in (path, *path.parents)):
        raise ValueError('Evidence paths cannot traverse symbolic links.')
    return path


def relative(name):
    if (not isinstance(name, str) or not name or '\\' in name or ':' in name
            or str(PurePosixPath(name)) != name or PurePosixPath(name).is_absolute()
            or any(p in ('', '.', '..') or p.startswith('.') for p in PurePosixPath(name).parts)
            or any(ord(c) < 32 for c in name)):
        raise ValueError('Expected a plain relative evidence path.')
    return PurePosixPath(name)


def scan(raw):
    if SECRET.search(raw):
        raise ValueError('Credential-shaped bytes found in proposed public evidence.')


def identity(path, *, maximum=MAX_FILE, secret_scan=False):
    path = plain(path)
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= maximum:
            raise ValueError('Missing, nonregular or oversized evidence file.')
        digest, size, tail = sha256(), 0, b''
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            size += len(chunk)
            if size > maximum:
                raise ValueError('Evidence exceeded its byte limit while reading.')
            digest.update(chunk)
            if secret_scan:
                scan(tail + chunk)
                tail = chunk[-256:]
        after = os.fstat(stream.fileno())
        if any(getattr(before, key) != getattr(after, key) for key in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')):
            raise ValueError('Evidence changed while reading.')
        if size != before.st_size:
            raise ValueError('Evidence byte count changed.')
    return dict(bytes=size, sha256=digest.hexdigest())


def read(path):
    path = plain(path)
    if path.stat().st_size > MAX_FILE:
        raise ValueError('JSON evidence exceeds the bounded format.')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON evidence field.')
            result[key] = value
        return result
    def invalid(_value):
        raise ValueError('Nonfinite JSON evidence.')
    return json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=invalid)


def write(path, value):
    path = plain(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(canonical(value) + b'\n')


def files(root):
    root = plain(root)
    paths = []
    for path in sorted(root.rglob('*')):
        plain(path)
        if not (path.is_dir() or path.is_file()):
            raise ValueError('Special files are not public evidence.')
        if path.is_file():
            relative(path.relative_to(root).as_posix())
            paths.append(path)
    return paths


def checked_identity(value):
    if (not isinstance(value, dict) or set(value) != {'bytes', 'sha256'}
            or type(value['bytes']) is not int or not 0 <= value['bytes'] <= MAX_FILE
            or not isinstance(value['sha256'], str) or not re.fullmatch('[a-f0-9]{64}', value['sha256'])):
        raise ValueError('Malformed sealed evidence identity.')
    return value


def episode_inventory(root):
    root = plain(root)
    manifest = read(root / 'integrity.json')
    if manifest.get('schema') != 'dvidia.camera-action-rollout/v2/integrity':
        raise ValueError('Only sealed v2 native rollout evidence can be archived.')
    expected = manifest.get('files')
    if not isinstance(expected, dict) or not 2 <= len(expected) <= 9000:
        raise ValueError('Episode file inventory is outside the bounded format.')
    for name, record in expected.items():
        relative(name)
        checked_identity(record)
    observed = {p.relative_to(root).as_posix() for p in files(root)}
    if observed != set(expected) | {'integrity.json'}:
        raise ValueError('Episode contains unsealed, missing or extra files.')
    result = dict(expected, **{'integrity.json': identity(root / 'integrity.json')})
    if sum(row['bytes'] for row in result.values()) > MAX_EPISODE:
        raise ValueError('Episode exceeds its raw byte budget.')
    return result


def archive(root, inventory, destination, *, prefix='', compresslevel=ZIP_LEVEL):
    """Stream byte-bound files into a deterministic ZIP, then read every member."""
    root, destination = plain(root), plain(destination)
    if compresslevel not in (1, 6):
        raise ValueError('Use a reviewed ZIP compression level: 1 or 6.')
    if destination.exists():
        raise ValueError('Archive exists; existing publication artifacts are immutable.')
    if prefix:
        relative(prefix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free - sum(r['bytes'] for r in inventory.values()) < DISK_RESERVE:
        raise ValueError('Archive preparation would consume the protected disk reserve.')
    expected = {}
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=compresslevel, allowZip64=True) as output:
        for name, record in sorted(inventory.items()):
            relative(name)
            checked_identity(record)
            path = plain(root / name)
            member = (prefix + '/' if prefix else '') + name
            relative(member)
            info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info._compresslevel = compresslevel
            h, size, tail = sha256(), 0, b''
            with path.open('rb') as source, output.open(info, 'w', force_zip64=True) as target:
                before = os.fstat(source.fileno())
                if not stat.S_ISREG(before.st_mode) or before.st_size != record['bytes']:
                    raise ValueError('Source file identity changed before archiving.')
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    size += len(chunk)
                    if size > record['bytes']:
                        raise ValueError('Source file grew while archiving.')
                    if path.suffix in TEXT_SUFFIXES:
                        scan(tail + chunk)
                        tail = chunk[-256:]
                    h.update(chunk)
                    target.write(chunk)
                after = os.fstat(source.fileno())
                if (dict(bytes=size, sha256=h.hexdigest()) != record or any(getattr(before, key) != getattr(after, key)
                        for key in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))):
                    raise ValueError('Sealed source bytes changed while archiving.')
            expected[member] = record
    with zipfile.ZipFile(destination) as verification:
        if len(verification.infolist()) != len(expected) or set(verification.namelist()) != set(expected):
            raise ValueError('Archive member roster differs from sealed evidence.')
        for member, record in expected.items():
            h, size = sha256(), 0
            with verification.open(member) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    size += len(chunk)
                    h.update(chunk)
            if dict(bytes=size, sha256=h.hexdigest()) != record:
                raise ValueError('Archive decompression differs from sealed evidence.')
    return dict(**identity(destination, maximum=MAX_EPISODE), files=len(expected),
                unpacked_bytes=sum(r['bytes'] for r in expected.values()),
                compression_level=compresslevel,
                inventory_sha256=sha256(canonical(expected)).hexdigest())


def _runtime_members(path, sources, *, wheel):
    """Check the actual wheel/source archive contains each registered runtime byte."""
    identities = {}
    total = 0
    def source_name(name):
        if (not isinstance(name, str) or not name or '\\' in name or ':' in name
                or PurePosixPath(name).is_absolute() or str(PurePosixPath(name)) != name
                or any(part in ('', '.', '..', '.git', '.env', '.venv', 'node_modules')
                    or part.startswith('.env.') for part in PurePosixPath(name).parts)
                or any(ord(c) < 32 for c in name)):
            raise ValueError('Unsafe or private source artifact path.')
    if wheel:
        with zipfile.ZipFile(path) as archive_file:
            entries = archive_file.infolist()
            if len(entries) > 50000 or len({e.filename for e in entries}) != len(entries):
                raise ValueError('Invalid source wheel member roster.')
            for entry in entries:
                if entry.is_dir():
                    source_name(entry.filename.rstrip('/'))
                    continue
                source_name(entry.filename)
                if stat.S_ISLNK(entry.external_attr >> 16) or entry.file_size > MAX_FILE:
                    raise ValueError('Source wheel contains an unsupported member.')
                total += entry.file_size
                if total > 2 * MAX_EPISODE:
                    raise ValueError('Expanded source wheel exceeds its bounded size.')
                name = PurePosixPath(entry.filename).name
                raw = archive_file.read(entry) if PurePosixPath(entry.filename).suffix in TEXT_SUFFIXES or name in sources else None
                if raw is not None:
                    scan(raw)
                if entry.filename.startswith('dvidia_training/') and name in sources:
                    if name in identities:
                        raise ValueError('Duplicate registered runtime member.')
                    identities[name] = sha256(raw).hexdigest()
    else:
        with tarfile.open(path, mode='r:*') as archive_file:
            seen = set()
            for entry in archive_file:
                source_name(entry.name.rstrip('/'))
                if entry.name in seen or len(seen) >= 50000:
                    raise ValueError('Invalid source archive member roster.')
                seen.add(entry.name)
                if entry.isdir():
                    continue
                if not entry.isfile() or entry.size > MAX_FILE:
                    raise ValueError('Source archive contains links or special members.')
                total += entry.size
                if total > 2 * MAX_EPISODE:
                    raise ValueError('Expanded source archive exceeds its bounded size.')
                name = PurePosixPath(entry.name).name
                raw = None
                if PurePosixPath(entry.name).suffix in TEXT_SUFFIXES or name in sources:
                    with archive_file.extractfile(entry) as stream:
                        raw = stream.read(MAX_FILE + 1)
                    scan(raw)
                if '/src/dvidia_training/' in '/' + entry.name and name in sources:
                    if name in identities:
                        raise ValueError('Duplicate registered runtime member.')
                    identities[name] = sha256(raw).hexdigest()
    if identities != sources:
        raise ValueError('Wheel/source archive does not contain the exact registered runtime.')
    return identities


def _copy(source, destination, *, maximum=MAX_FILE):
    expected = identity(source, maximum=maximum, secret_scan=Path(source).suffix in TEXT_SUFFIXES)
    destination = plain(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with plain(source).open('rb') as read_stream, destination.open('xb') as write_stream:
        shutil.copyfileobj(read_stream, write_stream, 1024 * 1024)
    if identity(destination, maximum=maximum) != expected:
        raise ValueError('Copied publication bytes differ from reviewed source.')
    return expected


def verify_study_audit(study, proof_path, receipts, state):
    """Bind an optional independent audit to every currently sealed stage."""
    proof = read(proof_path)
    if (proof.get('schema') != 'dvidia.camera-action-study-audit/v1' or proof.get('passed') is not True
            or proof.get('errors') != [] or any(proof.get(k) is not False for k in
                ('physics_executed', 'estimator_executed', 'model_executed', 'source_authenticated', 'physical_robot_ready'))
            or proof.get('protocol_sha256') != identity(study / 'protocol.json')['sha256']):
        raise ValueError('Full-study audit does not pass or is bound to a different protocol.')
    inventory = proof['evidence_inventory']
    if sha256(canonical(inventory)).hexdigest() != proof['evidence_inventory_sha256']:
        raise ValueError('Full-study audit inventory digest differs.')
    for name, expected in inventory.items():
        if identity(study / relative(name)) != expected:
            raise ValueError('Full-study audit evidence changed after auditing.')
    expected_rows = {(name, row['id']): row for name, receipt in receipts.items() for row in receipt.get('rows', [])}
    observed = {(row['stage'], row['id']): row for row in proof['results']}
    expected_stages = {name for name, receipt in receipts.items() if 'rows' in receipt}
    if (len(observed) != len(proof['results']) or set(observed) != set(expected_rows)
            or proof['attempts_verified'] != len(expected_rows) or set(proof['stages_verified']) != expected_stages):
        raise ValueError('Full-study audit omits or invents a sealed stage or attempt.')
    if set(proof.get('training_stages_verified', [])) != {name for name in receipts if name.startswith('train-')}:
        raise ValueError('Full-study audit omits or invents a sealed fitting stage.')
    required = {'protocol.json', 'registration.json', 'fixed-model.json'}
    for name in ('report.json', 'freeze.json'):
        if (study / name).exists(): required.add(name)
    for stage, receipt in receipts.items():
        required.update('stages/' + stage + suffix for suffix in ('.json', '.integrity.json', '.started.json'))
        for model in receipt.get('models', {}).values(): required.add(model['path'])
    for key, row in expected_rows.items():
        result = observed[key]
        for name in ('case', 'controller', 'negative_control', 'status', 'strict_success', 'complete'):
            if result.get(name) != row.get(name):
                raise ValueError('Full-study audit outcome or registered scene differs.')
        if row.get('path'):
            for name in ('integrity.json', 'rollout.json', 'recording.json'):
                path = str(relative(row['path'])) + '/' + name
                if (study / path).exists(): required.add(path)
    if not required <= set(inventory):
        raise ValueError('Full-study audit lacks essential source evidence identities.')
    qualification = proof.get('qualification')
    if qualification is not None and qualification.get('qualified') is not state['qualified']:
        raise ValueError('Full-study audit and publication qualification differ.')
    if state['qualified'] and qualification is None:
        raise ValueError('Qualified publication requires independently reconstructed qualification.')
    helpers = Path(__file__).resolve().parents[1] / 'benchmarks/camera-action-v01'
    if proof.get('scorer_source_sha256') != read(study / 'protocol.json')['source_sha256']['piper_feedback_score.py']:
        raise ValueError('Independent audit used a different native scorer.')
    for path, field in ((helpers / 'study_audit.py', 'audit_source_sha256'),
                        (helpers / 'independent_audit.py', 'diagnostic_helper_source_sha256')):
        if proof.get(field) != identity(path)['sha256']:
            raise ValueError('Independent audit implementation differs from the reviewed proof.')
    return proof


def verify_audit_source_archive(source_archive, proof):
    """The published source snapshot must actually include this audit code."""
    expected = {'benchmarks/camera-action-v01/study_audit.py': proof['audit_source_sha256'],
        'benchmarks/camera-action-v01/independent_audit.py': proof['diagnostic_helper_source_sha256']}
    with tarfile.open(source_archive, 'r:*') as archive:
        members = archive.getmembers()
        for name, expected_hash in expected.items():
            matches = [m for m in members if m.name == name or m.name.endswith('/' + name)]
            if len(matches) != 1 or not matches[0].isfile() or matches[0].size > MAX_FILE:
                raise ValueError('Reviewed source archive must contain each independent audit implementation exactly once.')
            with archive.extractfile(matches[0]) as stream:
                digest = sha256()
                for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)
            if digest.hexdigest() != expected_hash:
                raise ValueError('Reviewed source archive differs from the independent audit implementation.')


def prepare(study, output, *, audit, card, wheel, source_archive, source_commit, phase_coverage=None, study_audit=None, hardware=None, episode_cache=None):
    from dvidia_training import camera_action_study as workflow
    study, output = plain(study), plain(output)
    if output.exists() or output == study or output.is_relative_to(study):
        raise ValueError('Use a fresh publication directory outside the study.')
    if (study / 'worker.lock').exists():
        raise ValueError('A running or interrupted study worker must be resolved before packaging.')
    if episode_cache is not None:
        episode_cache = plain(episode_cache).resolve()
        if any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
                for a, b in ((episode_cache, study.resolve()), (episode_cache, output.resolve()))):
            raise ValueError('Episode cache must be separate from study and publication output trees.')
        if (episode_cache / 'worker.lock').exists():
            raise ValueError('Wait for the episode cache worker to finish before final packaging.')
    if not re.fullmatch('[a-f0-9]{40}', source_commit):
        raise ValueError('An exact source commit is required.')
    state = workflow.inspect(study)
    if state['status'] not in ('blocked', 'complete'):
        raise ValueError('Publish only a stopped blocked or completed study, not a partial active pipeline.')
    if state['qualified'] and study_audit is None:
        raise ValueError('A qualified-skill publication requires a full-study independent audit.')
    if hardware is not None:
        identity(hardware, maximum=MAX_HARDWARE_BYTES, secret_scan=True)
        validate_hardware(read(hardware))
    protocol = read(study / 'protocol.json')
    proof = read(audit)
    if (proof.get('schema') != 'dvidia.camera-action-diagnostic-audit/v1' or proof.get('passed') is not True
            or proof.get('errors') != [] or proof.get('attempts_verified') != 4
            or proof.get('physics_executed') is not False or proof.get('estimator_executed') is not False):
        raise ValueError('The explicit passed four-episode diagnostic audit is required.')
    diagnostic = read(study / 'stages/diagnostic.json')
    if {r['case_id'] for r in proof['results']} != {r['case']['id'] for r in diagnostic['rows']}:
        raise ValueError('Diagnostic audit cases differ from this study.')
    for result in proof['results']:
        row = next(r for r in diagnostic['rows'] if r['case']['id'] == result['case_id'])
        if row['strict_success'] is not result['strict_success']:
            raise ValueError('Diagnostic audit and study outcomes differ.')
    if phase_coverage is not None:
        phase_proof = read(phase_coverage)
        if (phase_proof.get('schema') != 'dvidia.camera-action-phase-coverage/v1'
                or phase_proof.get('protocol_sha256') != identity(study / 'protocol.json')['sha256']
                or phase_proof.get('diagnostic_audit_sha256') != identity(audit)['sha256']):
            raise ValueError('Phase coverage proof is not bound to this protocol and diagnostic audit.')
    _runtime_members(wheel, protocol['source_sha256'], wheel=True)
    _runtime_members(source_archive, protocol['source_sha256'], wheel=False)
    with tarfile.open(source_archive, 'r:*') as source_snapshot:
        if source_snapshot.pax_headers.get('comment') != source_commit:
            raise ValueError('Source archive must carry the exact git-archive commit in its PAX comment.')
    metadata = {}
    allowed_root = {'protocol.json', 'registration.json', 'fixed-model.json', 'freeze.json', 'report.json'}
    episode_rows = []
    models = []
    for path in files(study):
        relative_path = path.relative_to(study).as_posix()
        parts = PurePosixPath(relative_path).parts
        if parts[0] == 'episodes':
            continue
        allowed = (len(parts) == 1 and relative_path in allowed_root
            or len(parts) == 2 and parts[0] == 'stages' and path.suffix == '.json'
            or len(parts) == 4 and parts[0] == 'models' and parts[1] in ('initial', 'corrected')
                and parts[2] in ('full', 'blind') and parts[3] in ('model.json', 'receipt.json', 'started.json'))
        if not allowed:
            raise ValueError('Unexpected study artifact; review it before publication.')
        metadata[relative_path] = identity(path, secret_scan=True)
        if parts[0] == 'models' and parts[-1] == 'model.json':
            training = workflow._require(study, 'train-' + parts[1])
            if training['models'][parts[2]]['model_sha256'] != metadata[relative_path]['sha256']:
                raise ValueError('Model does not match completed training evidence.')
            models.append(relative_path)
    all_expected = {}; receipts = {}
    for stage in workflow.STAGES:
        receipt = workflow._stage(study, stage)
        if receipt is None:
            continue
        receipts[stage] = receipt
        for row in receipt.get('rows', []):
            if 'path' not in row:
                continue
            path = relative(row['path'])
            if len(path.parts) != 3 or path.parts[0] != 'episodes' or path.parts[1] != stage or path.parts[2] != row['id']:
                raise ValueError('Episode path differs from the sealed stage identity.')
            directory = study / path
            if row.get('receipt_sha256') != identity(directory / 'rollout.json')['sha256'] or row.get('integrity_sha256') != identity(directory / 'integrity.json')['sha256']:
                raise ValueError('Episode bytes differ from the sealed stage receipt.')
            inventory = episode_inventory(directory)
            all_expected[str(path)] = inventory
            episode_rows.append(dict(stage=stage, id=row['id'], group_id=row['case']['group_id'],
                split=row['case']['split'], controller=row['controller'], negative_control=row['negative_control'],
                status=row['status'], strict_success=row['strict_success'], source_directory=str(path),
                integrity_sha256=row['integrity_sha256'], receipt_sha256=row['receipt_sha256']))
    observed_episode_dirs = {p.parent.relative_to(study).as_posix() for p in (study / 'episodes').glob('*/*/started.json')}
    if observed_episode_dirs != set(all_expected):
        raise ValueError('Unsealed, unregistered or missing episode attempts cannot be silently omitted.')
    if study_audit is not None:
        full_proof = verify_study_audit(study, study_audit, receipts, state)
        verify_audit_source_archive(source_archive, full_proof)
    raw_bytes = sum(v['bytes'] for inventory in all_expected.values() for v in inventory.values())
    if raw_bytes > MAX_TOTAL:
        raise ValueError('Study exceeds the registered raw evidence budget.')
    output.mkdir(parents=True)
    inventory = {}
    repo = Path(__file__).resolve().parents[1]
    for source, destination in ((card, 'README.md'), (repo / 'LICENSE', 'LICENSE'),
        (audit, 'independent-audit.json'), (study / 'protocol.json', 'protocol.json'),
        (study / 'registration.json', 'registration.json')):
        inventory[destination] = _copy(source, output / destination)
    if phase_coverage is not None:
        inventory['diagnostic-phase-coverage.json'] = _copy(phase_coverage, output / 'diagnostic-phase-coverage.json')
    if study_audit is not None:
        inventory['full-study-audit.json'] = _copy(study_audit, output / 'full-study-audit.json')
    if hardware is not None:
        inventory['hardware.json'] = _copy(hardware, output / 'hardware.json', maximum=MAX_HARDWARE_BYTES)
    write(output / 'report.json', state['report'])
    inventory['report.json'] = identity(output / 'report.json')
    for source, label in ((wheel, 'wheel'), (source_archive, 'source')):
        source = plain(source)
        if source.name.startswith('.') or len(PurePosixPath(source.name).parts) != 1:
            raise ValueError('Unexpected source artifact filename.')
        name = 'source/' + source.name
        inventory[name] = _copy(source, output / name, maximum=MAX_SOURCE)
    metadata_archive = archive(study, metadata, output / 'study-metadata.zip')
    inventory['study-metadata.zip'] = {k: metadata_archive[k] for k in ('bytes', 'sha256')}
    for row in episode_rows:
        name = row['source_directory'] + '.zip'
        sealed = None
        if episode_cache is not None:
            from episode_archive_cache import reuse
            sealed = reuse(study / row['source_directory'], all_expected[row['source_directory']],
                           output / name, prefix=row['source_directory'], cache=episode_cache)
        if sealed is None:
            sealed = archive(study / row['source_directory'], all_expected[row['source_directory']],
                             output / name, prefix=row['source_directory'])
        row['archive'] = name
        row['archive_receipt'] = sealed
        inventory[name] = {k: sealed[k] for k in ('bytes', 'sha256')}
        print(json.dumps(dict(stage='archive_verified', episode=row['id'], bytes=sealed['bytes'])), flush=True)
    for name in models:
        inventory[name] = _copy(study / name, output / name)
    value = dict(schema=SCHEMA, repo_id='Dvidia/piper-camera-action-v01', repo_type='dataset',
        source_commit=source_commit, registered_runtime_sha256=protocol['source_sha256'],
        status=state['report']['status'], pipeline_status=state['status'],
        qualification=state['qualification'], qualified=state['qualified'],
        physical_robot_ready=False, original_footage_included=False,
        recorded_evidence_origin='Synthetic MuJoCo RGB-D, encoders, controls and native scorer traces.',
        independent_audit_scope=('All sealed stages are independently byte-, authority-, clock- and numerically audited in full-study-audit.json. Missing stages remain unrun; no physics, estimator or policy was re-executed.' if study_audit is not None else
            'Four diagnostic episodes only. Other shards are checked against sealed manifests; this is not a full-study independent audit.'),
        raw_evidence_bytes=raw_bytes, episode_count=len(episode_rows), episodes=episode_rows,
        trained_model_files=models, files=inventory,
        hardware_context_file='hardware.json' if hardware is not None else None,
        reconstruction='Extract study-metadata.zip and every episodes/**/*.zip into one new directory. Compare SHA256SUMS first. Each recording retains its original integrity.json.',
        archive_format=f'ZIP64, sorted members, 1980 timestamp, regular 0644 files; all members read back byte-for-byte. Fresh archives use deterministic DEFLATE level {ZIP_LEVEL}. Reused cache archives have verified DEFLATE streams but their compression level is not inferable and is recorded as null.')
    write(output / 'manifest.json', value)
    inventory_with_manifest = dict(inventory, **{'manifest.json': identity(output / 'manifest.json')})
    sums = ''.join(f'{record["sha256"]}  {name}\n' for name, record in sorted(inventory_with_manifest.items()))
    with (output / 'SHA256SUMS').open('x') as stream:
        stream.write(sums)
    return dict(status='complete', output=str(output), episode_count=len(episode_rows), raw_evidence_bytes=raw_bytes,
        public_bytes=sum(v['bytes'] for v in inventory_with_manifest.values()), trained_models=len(models),
        qualification=state['qualification'], manifest_sha256=identity(output / 'manifest.json')['sha256'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--card', required=True)
    parser.add_argument('--wheel', required=True)
    parser.add_argument('--source-archive', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--phase-coverage', help='Optional diagnostic-only phase coverage proof bound to the same protocol and audit.')
    parser.add_argument('--study-audit', help='Optional independent audit of all sealed study stages, byte-bound to this stopped study.')
    parser.add_argument('--hardware', help='Optional bounded public hardware/software and timing-scope JSON; no hostname or serial fields.')
    parser.add_argument('--episode-cache', help='Optional separately stored ZIP cache; source bytes and every cached member are independently verified.')
    args = parser.parse_args(argv)
    try:
        print(json.dumps(prepare(args.study, args.output, audit=args.audit, card=args.card,
            wheel=args.wheel, source_archive=args.source_archive, source_commit=args.source_commit,
            phase_coverage=args.phase_coverage, study_audit=args.study_audit, hardware=args.hardware,
            episode_cache=args.episode_cache), indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile, tarfile.TarError) as exc:
        print(json.dumps(dict(status='refused', reason=str(exc))))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

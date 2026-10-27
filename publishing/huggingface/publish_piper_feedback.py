"""Stage, check, then explicitly publish the simulation-only PiPER feedback study.

``stage`` and ``check`` are offline and never read credentials. ``publish`` uses
the existing named-Keychain client. Every destination is fixed; the shared Space
is append-only inside a new namespace, against an explicitly reviewed baseline.
Large native evidence belongs to the dataset, not the static Space.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import gzip
from hashlib import sha256
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
import time
import zipfile

from publish import PublicationError, checked_remote, client
from publish_piper_imitation_batches import _tree, _matches, _repo_not_found

SCHEMA = 'dvidia.piper-feedback-publication/v0.2'
STUDY_SCHEMA = 'dvidia.piper-feedback-study/v0.2'
INTEGRITY_SCHEMA = 'dvidia.piper-feedback-integrity/v0.2'
AUDIT_SCHEMA = 'dvidia.piper-feedback-audit/v0.2'
VIEWER_SCHEMA = 'dvidia.piper-feedback-viewer/v0.2'
MODEL_REPO = 'Dvidia/piper-block-placement-feedback-v02'
DATASET_REPO = 'Dvidia/piper-block-placement-feedback-demonstrations-v02'
SPACE_REPO = 'Dvidia/robot-simulation-lab'
SPACE_NAMESPACE = 'learned-placement/v02'
VIEWER_URL = 'https://dvidia-robot-simulation-lab.static.hf.space/' + SPACE_NAMESPACE + '/viewer/index.html'
REPOSITORIES = {'model': MODEL_REPO, 'dataset': DATASET_REPO, 'space': SPACE_REPO}
VARIANTS = ('feedback', 'blind', 'feedforward', 'teacher', 'open_jaw')
MAX_FILE = BATCH_MAX_BYTES = 8 * 1024 * 1024
MAX_TOTAL = 3 * 1024 * 1024 * 1024
MAX_MODEL = 64 * 1024 * 1024
MAX_SPACE = 256 * 1024 * 1024
MAX_DECODED_FILE = 128 * 1024 * 1024
MAX_DECODED_TOTAL = 12 * 1024 * 1024 * 1024
MAX_FILES = 10000
MAX_PLAN = 8 * 1024 * 1024
BATCH_MAX_FILES = 25
MAX_RETRIES = 3
SHA = re.compile(r'[a-f0-9]{64}\Z')
COMMIT = re.compile(r'[a-f0-9]{40}\Z')
TOKEN = re.compile(rb'(?:hf_[A-Za-z0-9]{30,}|apikey_[a-f0-9]{20,}_[a-f0-9]{20,}|sk-[A-Za-z0-9_-]{30,}|Bearer\s+[A-Za-z0-9_-]{20,})')
PRIVATE = re.compile(rb'(?:/Users/[^/\s]+/|/private/tmp/|file://)')
PRIVATE_KEYS = {'authorization', 'api_key', 'apikey', 'credential', 'credentials', 'token', 'raw_response'}
SUFFIXES = {'.json', '.gz', '.npz', '.py', '.md', '.txt', '.html', '.js', '.css', '.bin', '.png', '.svg', '.webp'}
ROOT_FILES = {'base-model.json', 'protocol.json', 'report.json', 'frozen.json', 'integrity.json', 'audit.json', 'training-parent.json', 'LICENSE', 'README.md'}
RUN_DIRS = {'sources', 'demonstrations', 'dagger', 'candidates', 'confirmation', 'docs', 'contracts', 'licenses', 'prior-studies'}
LFS_SUFFIXES = ('.gz', '.bin', '.npz')
JOURNAL_NAME = 'piper-feedback-batch-journal.json'


def _plain(path):
    path = Path(path).absolute()
    # macOS exposes these system aliases. Any artifact-owned symlink is rejected.
    aliases = {Path('/var'), Path('/tmp')}
    if any(p.is_symlink() and p not in aliases for p in (path, *path.parents)):
        raise PublicationError('Publication paths may not traverse symlinks.')
    return path.resolve()


def _relative(name):
    if not isinstance(name, str) or not name or '\\' in name:
        raise PublicationError('Invalid public artifact path.')
    p = PurePosixPath(name)
    if (p.is_absolute() or p.as_posix() != name or any(v.startswith('.') for v in p.parts)
            or '__pycache__' in p.parts or (p.suffix not in SUFFIXES and p.name != 'LICENSE')):
        raise PublicationError('Unsafe or unexpected public artifact path.')
    return p


def _read(path, limit=MAX_FILE):
    path = _plain(path)
    if not path.is_file() or path.stat().st_size > limit:
        raise PublicationError('Missing or oversized publication artifact.')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise PublicationError('Publication artifact exceeded its size bound.')
    return raw


def _json(raw):
    try:
        return json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, UnicodeError) as exc:
        raise PublicationError('Expected finite UTF-8 JSON.') from exc


def _encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def _receipt(raw):
    return dict(bytes=len(raw), sha256=sha256(raw).hexdigest())


def _scan(raw, *, json_value=False):
    if TOKEN.search(raw) or PRIVATE.search(raw):
        raise PublicationError('Possible credential or private path in a public artifact.')
    if json_value:
        values = [_json(raw)]
        while values:
            value = values.pop()
            if isinstance(value, dict):
                if any(str(k).lower() in PRIVATE_KEYS for k in value):
                    raise PublicationError('Credential fields cannot be published.')
                values.extend(value.values())
            elif isinstance(value, list):
                values.extend(value)


def _inspect(name, raw):
    _scan(raw, json_value=name.endswith('.json'))
    decoded = 0
    if name.endswith('.json.gz'):
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
                body = stream.read(MAX_DECODED_FILE + 1)
        except (OSError, EOFError) as exc:
            raise PublicationError('Invalid compressed episode evidence.') from exc
        if len(body) > MAX_DECODED_FILE:
            raise PublicationError('Compressed evidence exceeds its decoded bound.')
        _scan(body, json_value=True)
        decoded = len(body)
    elif name.endswith('.npz'):
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                names = set()
                for member in archive.infolist():
                    p = PurePosixPath(member.filename)
                    if (member.filename in names or len(p.parts) != 1 or p.suffix != '.npy'
                            or p.name.startswith('.') or member.file_size > MAX_DECODED_FILE
                            or member.flag_bits & 1):
                        raise PublicationError('Native arrays have unsafe members or size.')
                    names.add(member.filename)
                    decoded += member.file_size
                    if decoded > MAX_DECODED_FILE:
                        raise PublicationError('Array archive exceeds its decoded bound.')
                    body = archive.read(member)
                    # Inspect the NPY header without importing NumPy or unpickling.
                    if len(body) < 10 or body[:6] != b'\x93NUMPY' or body[6] not in (1, 2, 3):
                        raise PublicationError('Native archive contains a non-NPY member.')
                    width = 2 if body[6] == 1 else 4
                    length = int.from_bytes(body[8:8+width], 'little')
                    if length > 10000 or 8+width+length > len(body):
                        raise PublicationError('Native array header is malformed.')
                    header = ast.literal_eval(body[8+width:8+width+length].decode('latin1').strip())
                    if (not isinstance(header, dict) or not isinstance(header.get('descr'), str)
                            or 'O' in header['descr']):
                        raise PublicationError('Object or structured arrays cannot be published.')
                    _scan(body)
        except (OSError, zipfile.BadZipFile, RuntimeError, SyntaxError, UnicodeError) as exc:
            raise PublicationError('Native array archive is invalid.') from exc
    return decoded


def _inventory(root):
    root = _plain(root)
    if not root.is_dir():
        raise PublicationError('Expected a plain artifact directory.')
    files, total, decoded = {}, 0, 0
    for path in sorted(root.rglob('*')):
        _plain(path)
        if path.is_dir():
            continue
        if not path.is_file():
            raise PublicationError('Only plain publication files are supported.')
        name = path.relative_to(root).as_posix()
        _relative(name)
        raw = _read(path)
        decoded += _inspect(name, raw)
        total += len(raw)
        files[name] = _receipt(raw)
        if len(files) > MAX_FILES or total > MAX_TOTAL or decoded > MAX_DECODED_TOTAL:
            raise PublicationError('Artifact inventory exceeds its publication bounds.')
    return files


def _attempt_inventory(root, files, protocol, runtime, protocol_sha):
    """Bind each original attempt against its own study, never a new protocol."""
    directories = {str(PurePosixPath(n).parent) for n in files if n.endswith('/started.json')}
    for folder in sorted(directories):
        receipt = _json(_read(root/folder/'receipt.json'))
        bound = receipt.get('binding') if isinstance(receipt, dict) else None
        names = {'started.json', 'episode.json.gz', 'native.npz', 'native.json', 'score.json'}
        if isinstance(bound, dict) and bound.get('collect_labels') is True:
            names.add('samples.npz')
        actual = {PurePosixPath(n).name for n in files if str(PurePosixPath(n).parent) == folder}
        case = bound.get('case') if isinstance(bound, dict) else None
        scheduled = protocol.get('cases', {})
        if (not isinstance(bound, dict) or bound.get('protocol_sha256') != protocol_sha
                or bound.get('runtime_sha256') != runtime
                or not isinstance(scheduled, dict) or not any(case in rows for rows in scheduled.values() if isinstance(rows, list))
                or receipt.get('files') != {n: files.get(folder+'/'+n, {}).get('sha256') for n in sorted(names)}
                or actual != names | {'receipt.json'} or _json(_read(root/folder/'started.json')) != bound):
            raise PublicationError('Incomplete or unbound native attempt evidence.')
    return directories


def _training_parent(root, files, protocol, audit):
    parent = _json(_read(root/'training-parent.json'))
    parent_sha = files['training-parent.json']['sha256']
    if (not isinstance(parent, dict) or parent.get('schema') != 'dvidia.piper-feedback-training-parent/v1'
            or protocol.get('training_parent_manifest_sha256') != parent_sha
            or audit.get('training_parent_sha256') != parent_sha):
        raise PublicationError('Training-parent manifest is not bound by the active protocol and audit.')
    prefix = 'prior-studies/study01/'
    history = {n.removeprefix(prefix): value for n, value in files.items() if n.startswith(prefix)}
    if (not history or parent.get('files') != history
            or any(n.startswith('prior-studies/') and not n.startswith(prefix) for n in files)
            or any(n.startswith(('demonstrations/', 'dagger/')) for n in files)):
        raise PublicationError('Imported history differs from its exact manifest or duplicates active collection.')
    oldroot = root/'prior-studies/study01'
    oldprotocol = _json(_read(oldroot/'protocol.json'))
    runtime = parent.get('runtime_sha256')
    if (not isinstance(oldprotocol, dict) or oldprotocol.get('schema') != STUDY_SCHEMA
            or history.get('protocol.json', {}).get('sha256') != parent.get('protocol_sha256')
            or not isinstance(runtime, dict) or not runtime
            or any(history.get('sources/'+n, {}).get('sha256') != digest for n, digest in runtime.items())
            or 'frozen.json' in history or 'report.json' in history
            or any(n.startswith('confirmation/') for n in history)):
        raise PublicationError('Original unfrozen study protocol, source snapshots or confirmation boundary changed.')
    cases = oldprotocol.get('cases', {})
    train, development = cases.get('train'), cases.get('development')
    if (not isinstance(train, list) or len(train) != 64 or not isinstance(development, list) or len(development) != 16
            or protocol.get('cases', {}).get('train') != train):
        raise PublicationError('Original training and development schedules are incomplete.')
    teacher_samples = {f"demonstrations/{c['id']}/samples.npz": history.get(f"demonstrations/{c['id']}/samples.npz", {}).get('sha256') for c in train}
    if any(not isinstance(d, str) or not SHA.fullmatch(d) for d in teacher_samples.values()) or parent.get('teacher_sample_files') != teacher_samples:
        raise PublicationError('Training-parent manifest must bind all 64 original teacher sample archives.')
    attempts = _attempt_inventory(oldroot, history, oldprotocol, runtime, parent['protocol_sha256'])
    teachers = {f"demonstrations/{c['id']}" for c in train}
    dagger = {f"dagger/{c['id']}" for c in train if c.get('condition') == 'jaw_interruption'}
    candidates = {PurePosixPath(n).parts[1] for n in history if n.startswith('candidates/')}
    expected = teachers | dagger | {f"candidates/{name}/development/{c['id']}" for name in candidates for c in development}
    if len(candidates) != 5 or len(dagger) != 32 or attempts != expected:
        raise PublicationError('Original history must retain 64 teachers, 32 DAgger attempts and all five complete candidate evaluations.')
    for name in candidates:
        evaluation = _json(_read(oldroot/f'candidates/{name}/development.json'))
        receipt = _json(_read(oldroot/f'candidates/{name}/receipt.json'))
        model_sha = history.get(f'candidates/{name}/model.json', {}).get('sha256')
        if (not model_sha or receipt.get('model_sha256') != model_sha
                or receipt.get('protocol_sha256') != parent['protocol_sha256'] or receipt.get('source_sha256') != runtime
                or evaluation.get('model_sha256') != model_sha or evaluation.get('protocol_sha256') != parent['protocol_sha256']
                or evaluation.get('runtime_sha256') != runtime or evaluation.get('scheduled') != 16
                or not isinstance(evaluation.get('rows'), list)
                or [r.get('case_id') for r in evaluation['rows']] != [c['id'] for c in development]
                or any(r.get('score') != _json(_read(oldroot/f"candidates/{name}/development/{r['case_id']}/score.json")) for r in evaluation['rows'])
                or evaluation.get('successes') != 0
                or evaluation.get('successes') != sum(r['score']['success'] for r in evaluation['rows'])):
            raise PublicationError('Original negative candidate aggregate differs from its retained receipts and scores.')
    return parent_sha


def _binding_documents(root, files):
    required = {'protocol.json', 'report.json', 'frozen.json', 'integrity.json', 'audit.json', 'base-model.json'}
    if not required <= set(files):
        raise PublicationError('Only a fully frozen, completed and audited study can be staged.')
    protocol, report, frozen, audit = [_json(_read(root/name)) for name in ('protocol.json', 'report.json', 'frozen.json', 'audit.json')]
    if any(not isinstance(v, dict) for v in (protocol, report, frozen, audit)):
        raise PublicationError('Study bindings must be JSON objects.')
    if (protocol.get('schema') != STUDY_SCHEMA or report.get('schema') != STUDY_SCHEMA
            or frozen.get('schema') != STUDY_SCHEMA or report.get('physical_robot_ready') is not False
            or protocol.get('physical_robot_ready') is not False or protocol.get('variants') != list(VARIANTS)
            or not isinstance(report.get('limitations'), list) or not report['limitations']
            or any(not isinstance(s, str) or not s.strip() for s in report['limitations'])):
        raise PublicationError('Study must declare the exact simulation-only protocol and limitations.')
    bindings = {key: files[name]['sha256'] for key, name in
                (('protocol_sha256', 'protocol.json'), ('report_sha256', 'report.json'), ('frozen_sha256', 'frozen.json'))}
    if (report.get('protocol_sha256') != bindings['protocol_sha256']
            or report.get('frozen_sha256') != bindings['frozen_sha256']
            or frozen.get('protocol_sha256') != bindings['protocol_sha256']
            or frozen.get('base_model_sha256') != files['base-model.json']['sha256']
            or protocol.get('base_model_raw_sha256') != files['base-model.json']['sha256']
            or audit.get('schema') != AUDIT_SCHEMA or audit.get('status') != 'passed'
            or any(audit.get(k) != v for k, v in bindings.items())):
        raise PublicationError('Completed report, independent audit and frozen protocol bindings disagree.')
    runtime = frozen.get('runtime_sha256')
    if not isinstance(runtime, dict) or not runtime:
        raise PublicationError('Frozen study must bind its exact runtime sources.')
    for name, digest in runtime.items():
        if (not isinstance(name, str) or len(PurePosixPath(name).parts) != 1
                or not isinstance(digest, str) or not SHA.fullmatch(digest)
                or files.get('sources/'+name, {}).get('sha256') != digest):
            raise PublicationError('Source snapshots differ from the frozen runtime.')
    for key, prefix in (('selected', 'selected'), ('blind', 'blind')):
        name = frozen.get(key)
        if not isinstance(name, str) or re.fullmatch(r'[A-Za-z0-9]{1,32}', name) is None:
            raise PublicationError('Invalid frozen candidate identity.')
        for filename, field in (('model.json', prefix+'_sha256'), ('receipt.json', prefix+'_receipt_sha256'),
                                ('development.json', prefix+'_development_sha256')):
            if files.get(f'candidates/{name}/{filename}', {}).get('sha256') != frozen.get(field):
                raise PublicationError('Frozen model or candidate receipt bytes disagree.')
    if frozen['selected'] == frozen['blind']:
        raise PublicationError('Feedback and blind controls must be distinct frozen candidates.')
    cases = protocol.get('cases')
    if (not isinstance(cases, dict) or set(cases) != {'train', 'development', 'confirmation'}
            or any(not isinstance(cases[k], list) or len(cases[k]) != n for k, n in
                   (('train', 64), ('development', 16), ('confirmation', 24)))):
        raise PublicationError('Study split roster is incomplete.')
    for split in cases:
        ids = [c.get('id') if isinstance(c, dict) else None for c in cases[split]]
        if any(not isinstance(i, str) or '/' in i or not i.startswith(split+'-') for i in ids) or len(set(ids)) != len(ids):
            raise PublicationError('Malformed or duplicate scheduled case.')
    expected = [(c['id'], c['condition'], v) for c in cases['confirmation'] for v in VARIANTS]
    rows = report.get('rows')
    if not isinstance(rows, list) or [(r.get('case_id'), r.get('condition'), r.get('variant')) for r in rows if isinstance(r, dict)] != expected:
        raise PublicationError('Report must preserve all 120 ordered confirmation attempts.')
    for row in rows:
        score = row.get('score')
        if not isinstance(score, dict) or type(score.get('success')) is not bool or type(score.get('valid')) is not bool:
            raise PublicationError('Malformed per-attempt result.')
        folder = f"confirmation/{row['case_id']}/{row['variant']}"
        saved = _json(_read(root/folder/'score.json'))
        if saved != score:
            raise PublicationError('Report result differs from its retained attempt score.')
    counts = {v: dict(scheduled=24, successes=sum(r['score']['success'] for r in rows if r['variant'] == v),
                      valid=sum(r['score']['valid'] for r in rows if r['variant'] == v)) for v in VARIANTS}
    if report.get('variants') != counts:
        raise PublicationError('Report variant counts differ from scheduled outcomes.')
    # Every discovered attempt must be complete and exact, including failed
    # candidates and DAgger collection. Never select a subset by success.
    lineage = 'training_parent_manifest_sha256' in protocol
    if lineage:
        parent_sha = _training_parent(root, files, protocol, audit)
        for key in ('selected', 'blind'):
            receipt = _json(_read(root/f"candidates/{frozen[key]}/receipt.json"))
            if receipt.get('training_parent_sha256') != parent_sha:
                raise PublicationError('Active candidate training receipt differs from its bound original training parent.')
    elif 'training-parent.json' in files or any(n.startswith('prior-studies/') for n in files):
        raise PublicationError('Imported training history requires an explicit active protocol lineage binding.')
    active_files = {n: value for n, value in files.items() if not n.startswith('prior-studies/')}
    attempt_dirs = _attempt_inventory(root, active_files, protocol, runtime, bindings['protocol_sha256'])
    required_dirs = set() if lineage else {f"demonstrations/{c['id']}" for c in cases['train']}
    required_dirs |= {f"confirmation/{c['id']}/{v}" for c in cases['confirmation'] for v in VARIANTS}
    if not required_dirs <= attempt_dirs:
        raise PublicationError('Teacher or confirmation evidence roster is incomplete.')
    return protocol, report, frozen, bindings


def checked_run(root):
    root = _plain(root)
    if not all((root/name).is_file() for name in ('protocol.json', 'report.json', 'frozen.json', 'integrity.json', 'audit.json')):
        raise PublicationError('Only a fully frozen, completed and audited study can be staged.')
    files = _inventory(root)
    for name in files:
        first = PurePosixPath(name).parts[0]
        if first not in ROOT_FILES and first not in RUN_DIRS:
            raise PublicationError('Study export contains an unexpected directory.')
    protocol, report, frozen, bindings = _binding_documents(root, files)
    seal = _json(_read(root/'integrity.json'))
    if (not isinstance(seal, dict) or seal.get('schema') != INTEGRITY_SCHEMA
            or seal.get('files') != {n: r for n, r in files.items() if n != 'integrity.json'}):
        raise PublicationError('Study export differs from its complete integrity seal.')
    return files, report, frozen, bindings


def checked_viewer(root, bindings):
    root = _plain(root)
    files = _inventory(root)
    seal = _json(_read(root/'viewer-integrity.json'))
    if (not isinstance(seal, dict) or seal.get('schema') != VIEWER_SCHEMA
            or seal.get('files') != {n: r for n, r in files.items() if n != 'viewer-integrity.json'}
            or not {'index.html', 'motion.json', 'scene.json', 'report.json', 'protocol.json', 'frozen.json'} <= set(files)
            or any(n.endswith('.npz') for n in files)
            or any(files[n]['sha256'] != bindings[k] for n, k in
                   (('report.json', 'report_sha256'), ('protocol.json', 'protocol_sha256'), ('frozen.json', 'frozen_sha256')))):
        raise PublicationError('Viewer projection must bind this study and omit dense native NPZ arrays.')
    return files


def _card(kind, report):
    table = '\n'.join(f"| {v} | {report['variants'][v]['successes']} / {report['variants'][v]['scheduled']} | {report['variants'][v]['valid']} |" for v in VARIANTS)
    front = ('---\nlicense: mit\n' + ('task_categories:\n- robotics\n' if kind == 'dataset' else 'tags:\n- robotics\n- imitation-learning\n') + '---\n\n') if kind != 'space' else ''
    return (front + '# DVIDIA · PiPER learned feedback v0.2\n\n'
        'A simulation research pilot: a learned controller paired with the frozen v0.1 motion primitive, '
        'trained on synthetic authored-expert labels and privileged simulator sensing. The matched blind control '
        'is separately trained without live feedback. Native servos, actuator limits and gravity compensation '
        'remain authored. This is not human-video learning, general slip recovery, hardware qualification, '
        'or proof of one-shot robot skill transfer.\n\n'
        '| Variant | Strict successes / scheduled | Valid |\n| --- | ---: | ---: |\n' + table + '\n\n'
        'All scheduled outcomes are retained, including failures. The dataset contains original 1 ms native '
        'NPZ arrays, teacher demonstrations, every development candidate, the DAgger collection (if used), '
        'confirmation traces, sources and receipts. The static replay is a smaller recorded projection; '
        'it does not run physics or inference in the browser.\n\n'
        'When `training-parent.json` is present, original teachers and failed experiments live unchanged '
        'under `prior-studies/study01/` with their original protocol and source hashes. Preserving the '
        'old DAgger collection does not imply that the selected new model trains on those mixed-policy '
        'samples; consult its bound training receipt for the actual recipe.\n\n'
        '[Model](https://huggingface.co/'+MODEL_REPO+') · '
        '[Complete evidence dataset](https://huggingface.co/datasets/'+DATASET_REPO+') · '
        '[Recorded replay]('+VIEWER_URL+')\n\n'
        '## Limits\n\n' + '\n'.join('- '+s for s in report['limitations']) + '\n\n'
        'Original DVIDIA code and artifacts: MIT License. Upstream robot assets and rendering '
        'components retain their licenses; notices are included where applicable. Hashes establish '
        'artifact consistency, not independent authenticity.\n').encode()


def _atomic(path, value):
    path = _plain(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.feedback-publication-', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(_encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    try:
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError:
        # Some filesystems cannot fsync directories; the file itself was fsynced
        # before the atomic replacement, and the previous journal stays valid.
        pass


def stage(run, viewer, source_commit, space_baseline, output):
    run, viewer, output = map(_plain, (run, viewer, output))
    if not COMMIT.fullmatch(source_commit or '') or not COMMIT.fullmatch(space_baseline or ''):
        raise PublicationError('Exact source and reviewed Space baseline commits are required.')
    if output.exists() or any(output == p or output in p.parents or p in output.parents for p in (run, viewer)):
        raise PublicationError('Staging output must be fresh and separate from source artifacts.')
    files, report, frozen, bindings = checked_run(run)
    viewer_files = checked_viewer(viewer, bindings)
    license_raw = _read(Path(__file__).resolve().parents[2]/'LICENSE')
    if not license_raw.startswith(b'MIT License'):
        raise PublicationError('Expected the DVIDIA MIT artifact license.')
    selected_prefixes = tuple(f"candidates/{frozen[key]}/" for key in ('selected', 'blind'))
    model_names = {n for n in files if (n in ROOT_FILES or n.startswith(('sources/', 'docs/', 'contracts/', 'licenses/'))
                  or n.startswith(selected_prefixes) and len(PurePosixPath(n).parts) == 3)}
    maps = {'model': {n: (run/n) for n in model_names}, 'dataset': {n: run/n for n in files},
            'space': {SPACE_NAMESPACE+'/viewer/'+n: viewer/n for n in viewer_files}}
    generated = {kind: {'README.md': _card(kind, report), 'LICENSE': license_raw} for kind in REPOSITORIES}
    generated['space'] = {SPACE_NAMESPACE+'/README.md': _card('space', report), SPACE_NAMESPACE+'/LICENSE': license_raw,
        SPACE_NAMESPACE+'/index.html': b'<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=viewer/index.html"><title>DVIDIA PiPER feedback replay</title><p><a href="viewer/index.html">Open the recorded replay</a></p></html>'}
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for kind, repo in REPOSITORIES.items():
        base = output/kind
        base.mkdir()
        for name, source in maps[kind].items():
            target = base/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        for name, raw in generated[kind].items():
            if name in maps[kind] and name not in {'README.md', 'LICENSE'}:
                raise PublicationError('Generated publication path collides with sealed evidence.')
            target = base/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        staged = _inventory(base)
        row = dict(repo_type=kind, repo_id=repo, directory=kind, files=[dict(path=n, **r) for n, r in staged.items()])
        if kind == 'space':
            row.update(append_only_namespace=SPACE_NAMESPACE, baseline_commit=space_baseline)
        rows.append(row)
    # Generated README/LICENSE may replace a run's root versions only in the
    # model. Dataset is a complete immutable copy, with cards at publication/.
    dataset_base = output/'dataset'
    for name in ('README.md', 'LICENSE'):
        if name in files and (dataset_base/name).read_bytes() != (run/name).read_bytes():
            (dataset_base/name).write_bytes((run/name).read_bytes())
            target = dataset_base/'docs'/'publication'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(generated['dataset'][name])
    # Preserve all source bytes including optional README/LICENSE. Root model
    # cards are derived artifacts; the complete originals remain in the dataset.
    rows[1]['files'] = [dict(path=n, **r) for n, r in _inventory(dataset_base).items()]
    plan = dict(schema=SCHEMA, private=False, source_commit=source_commit,
                source_integrity_sha256=files['integrity.json']['sha256'],
                source_files=files, viewer_files=viewer_files, bindings=bindings,
                selected=frozen['selected'], blind=frozen['blind'], repositories=rows)
    _atomic(output/'publication-plan.json', plan)
    return checked_plan(output/'publication-plan.json')


def checked_plan(path):
    path = _plain(path)
    plan = _json(_read(path, MAX_PLAN))
    if (not isinstance(plan, dict) or plan.get('schema') != SCHEMA or plan.get('private') is not False
            or not COMMIT.fullmatch(plan.get('source_commit', ''))
            or not SHA.fullmatch(plan.get('source_integrity_sha256', ''))):
        raise PublicationError('Expected a reviewed feedback v0.2 publication plan.')
    rows = plan.get('repositories')
    if (not isinstance(rows, list) or len(rows) != 3
            or [r.get('repo_type') for r in rows if isinstance(r, dict)] != list(REPOSITORIES)):
        raise PublicationError('Plan must name exactly the three fixed ordered destinations.')
    total = 0
    for row in rows:
        kind = row['repo_type']
        if row.get('repo_id') != REPOSITORIES[kind] or row.get('directory') != kind:
            raise PublicationError('Publication destination differs from fixed study repositories.')
        actual = _inventory(path.parent/kind)
        if row.get('files') != [dict(path=n, **r) for n, r in actual.items()]:
            raise PublicationError('Staged artifacts differ from the reviewed file inventory.')
        total += sum(r['bytes'] for r in actual.values())
        if kind in {'model', 'space'} and sum(r['bytes'] for r in actual.values()) > {'model': MAX_MODEL, 'space': MAX_SPACE}[kind]:
            raise PublicationError('Model or recorded viewer projection exceeds its bounded size.')
        if kind == 'space':
            if (row.get('append_only_namespace') != SPACE_NAMESPACE
                    or not COMMIT.fullmatch(row.get('baseline_commit', ''))
                    or any(not n.startswith(SPACE_NAMESPACE+'/') or n.endswith('.npz') for n in actual)
                    or SPACE_NAMESPACE+'/index.html' not in actual):
                raise PublicationError('Space plan exceeds the fixed small replay namespace.')
    dataset_root = path.parent/'dataset'
    # Allow only two explicitly derived dataset publication card files beyond
    # the original scientific seal, never omitted or modified source evidence.
    dataset_inventory = _inventory(dataset_root)
    source_files = plan.get('source_files')
    if not isinstance(source_files, dict) or any(dataset_inventory.get(n) != r for n, r in source_files.items()):
        raise PublicationError('Dataset omits or modifies sealed source evidence.')
    extras = set(dataset_inventory)-set(source_files)
    if extras - {'README.md', 'LICENSE', 'docs/publication/README.md', 'docs/publication/LICENSE'}:
        raise PublicationError('Dataset contains unreviewed additions beyond sealed evidence.')
    # Revalidate the original run with its own seal despite derived Hub cards.
    source_view = {n: dataset_inventory[n] for n in source_files}
    _, _, frozen, bindings = _binding_documents(dataset_root, source_view)
    seal = _json(_read(dataset_root/'integrity.json'))
    if (seal.get('schema') != INTEGRITY_SCHEMA or seal.get('files') != {n: r for n, r in source_view.items() if n != 'integrity.json'}
            or source_view['integrity.json']['sha256'] != plan['source_integrity_sha256']
            or plan.get('bindings') != bindings or plan.get('selected') != frozen['selected'] or plan.get('blind') != frozen['blind']):
        raise PublicationError('Publication plan source seal or scientific bindings changed.')
    viewer = path.parent/'space'/SPACE_NAMESPACE/'viewer'
    if checked_viewer(viewer, bindings) != plan.get('viewer_files'):
        raise PublicationError('Viewer projection differs from its reviewed source inventory.')
    model_inventory = _inventory(path.parent/'model')
    expected_model = {n for n in source_files if (n in ROOT_FILES or n.startswith(('sources/', 'docs/', 'contracts/', 'licenses/'))
        or any(n.startswith(f"candidates/{frozen[k]}/") and len(PurePosixPath(n).parts) == 3 for k in ('selected', 'blind')))}
    expected_model |= {'README.md', 'LICENSE'}
    if set(model_inventory) != expected_model or any(model_inventory[n] != source_files[n] for n in expected_model-{'README.md', 'LICENSE'}):
        raise PublicationError('Model repository exceeds or differs from the selected bounded artifact set.')
    if total > MAX_TOTAL:
        raise PublicationError('Combined publication exceeds its total size bound.')
    return plan


def _downloaded_bytes(value, limit):
    # The SDK's snapshot cache intentionally uses symlinks into its blob store.
    # These public, untrusted bytes are read with a size bound and compared with
    # the reviewed SHA; unlike artifact paths they are never copied blindly.
    if isinstance(value, bytes):
        raw = value
    else:
        path = Path(value)
        if not path.is_file() or path.stat().st_size > limit:
            raise PublicationError('Anonymous downloaded artifact exceeds its byte bound.')
        with path.open('rb') as stream:
            raw = stream.read(limit+1)
    if len(raw) > limit:
        raise PublicationError('Anonymous downloaded artifact exceeds its byte bound.')
    return raw


def _download(row, revision, item, downloader=None):
    if downloader is None:
        from huggingface_hub import hf_hub_download
        def downloader(repo_id, name, kind, revision):
            return hf_hub_download(repo_id, name, repo_type=kind, revision=revision, token=False, endpoint='https://huggingface.co')
    for attempt in range(MAX_RETRIES):
        try:
            result = downloader(row['repo_id'], item['path'], row['repo_type'], revision)
            raw = _downloaded_bytes(result, item['bytes'])
            if _receipt(raw) != {k: item[k] for k in ('bytes', 'sha256')}:
                raise PublicationError('Anonymous public bytes differ from reviewed publication bytes.')
            return raw
        except PublicationError:
            raise
        except Exception as exc:
            if attempt + 1 == MAX_RETRIES:
                raise PublicationError('Anonymous public read failed after bounded retries.') from exc
            time.sleep(.2 * 2**attempt)


def verify_public(row, revision, downloader=None, paths=None):
    selected = [r for r in row['files'] if paths is None or r['path'] in paths]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda item: _download(row, revision, item, downloader), selected))


def _info(api, row):
    try:
        value = api.repo_info(row['repo_id'], repo_type=row['repo_type'])
    except Exception as exc:
        if _repo_not_found(exc):
            return None
        raise
    checked_remote(value, row['repo_type'])
    return value


def _lfs_rules(row, tree, revision, downloader):
    entry = tree.get('.gitattributes')
    if not entry:
        raise PublicationError('Pinned Space baseline has no LFS attributes.')
    if downloader is None:
        from huggingface_hub import hf_hub_download
        def downloader(repo_id, name, kind, revision):
            return hf_hub_download(repo_id, name, repo_type=kind, revision=revision, token=False, endpoint='https://huggingface.co')
    value = downloader(row['repo_id'], '.gitattributes', 'space', revision)
    raw = _downloaded_bytes(value, 1024*1024)
    if not _matches(raw, entry):
        raise PublicationError('Pinned Space attributes differ from baseline tree.')
    rules = set()
    for line in raw.decode('utf8').splitlines():
        fields = line.split()
        for suffix in LFS_SUFFIXES:
            if fields and fields[0] == '*'+suffix and {'filter=lfs', 'diff=lfs', 'merge=lfs', '-text'} <= set(fields[1:]):
                rules.add(suffix)
    if not set(LFS_SUFFIXES) <= rules:
        raise PublicationError('Pinned Space lacks canonical binary LFS rules required by this replay.')
    return sorted(rules)


def _batches(row, missing):
    items = {f['path']: f for f in row['files']}
    result, current, size = [], [], 0
    for name in sorted(missing):
        count = items[name]['bytes']
        if count > BATCH_MAX_BYTES:
            raise PublicationError('One artifact exceeds the bounded batch size.')
        if current and (len(current) == BATCH_MAX_FILES or size+count > BATCH_MAX_BYTES):
            result.append(current)
            current, size = [], 0
        current.append(name)
        size += count
    return result + ([current] if current else [])


def _initial(plan, api, downloader):
    states = {}
    for row in plan['repositories']:
        kind, info = row['repo_type'], _info(api, row)
        if kind != 'space':
            if info is not None:
                raise PublicationError('New model and dataset IDs must not already exist without this journal.')
            states[kind] = dict(baseline_sha=None, baseline_tree={}, created_sha=None, batches=[], pending=None)
        else:
            if info is None or info.sha != row['baseline_commit']:
                raise PublicationError('Space head differs from the explicitly reviewed baseline commit.')
            tree = _tree(api, row['repo_id'], kind, info.sha)
            if any(n == SPACE_NAMESPACE or n.startswith(SPACE_NAMESPACE+'/') for n in tree):
                raise PublicationError('The new version namespace already exists in the shared Space.')
            states[kind] = dict(baseline_sha=info.sha, baseline_tree=tree, created_sha=None, batches=[], pending=None,
                                lfs_suffixes=_lfs_rules(row, tree, info.sha, downloader))
    return states


def _load_journal(path, plan, plan_sha):
    value = _json(_read(path, MAX_PLAN))
    if (not isinstance(value, dict) or value.get('schema') != SCHEMA+'/journal'
            or value.get('plan_sha256') != plan_sha or set(value.get('repositories', {})) != set(REPOSITORIES)):
        raise PublicationError('Recovery journal belongs to another publication plan.')
    for row in plan['repositories']:
        state = value['repositories'][row['repo_type']]
        if not isinstance(state, dict) or not isinstance(state.get('baseline_tree'), dict) or not isinstance(state.get('batches'), list):
            raise PublicationError('Malformed recovery journal state.')
        if row['repo_type'] == 'space' and state.get('baseline_sha') != row['baseline_commit']:
            raise PublicationError('Journal Space baseline differs from the reviewed plan.')
        previous, seen = state.get('created_sha') or state.get('baseline_sha'), set()
        allowed = {i['path'] for i in row['files']}
        for batch in state['batches']:
            if (not isinstance(batch, dict) or batch.get('parent_commit') != previous
                    or not COMMIT.fullmatch(batch.get('commit', '')) or type(batch.get('reconciled')) is not bool
                    or not isinstance(batch.get('paths'), list) or not batch['paths']
                    or len(batch['paths']) > BATCH_MAX_FILES or set(batch['paths']) & seen
                    or not set(batch['paths']) <= allowed or len(set(batch['paths'])) != len(batch['paths'])):
                raise PublicationError('Recovery journal batch chain is invalid.')
            if len(_batches(row, batch['paths'])) != 1:
                raise PublicationError('Journal batch exceeds its byte bound.')
            seen.update(batch['paths'])
            previous = batch['commit']
        pending = state.get('pending')
        if pending is not None and (not isinstance(pending, dict) or pending.get('parent_commit') != previous
                or not isinstance(pending.get('paths'), list) or not pending['paths']
                or set(pending['paths']) & seen or not set(pending['paths']) <= allowed
                or len(set(pending['paths'])) != len(pending['paths']) or len(_batches(row, pending['paths'])) != 1):
            raise PublicationError('Recovery journal pending batch is invalid.')
    return value


def _live(row, state, api, downloader, *, journal, journal_path):
    info = _info(api, row)
    kind = row['repo_type']
    if info is None:
        if state.get('baseline_sha') or state.get('created_sha'):
            raise PublicationError('Previously observed repository disappeared.')
        return set(), None
    tree = _tree(api, row['repo_id'], kind, info.sha)
    if not state.get('baseline_sha') and not state.get('created_sha'):
        # Creation itself can return an unknown outcome; only the conventional
        # empty Hub repository is accepted. No user data is adopted or replaced.
        if set(tree) != {'.gitattributes'}:
            raise PublicationError('Unjournaled repository creation contains unexpected files.')
        state.update(created_sha=info.sha, baseline_tree=tree)
        _atomic(journal_path, journal)
    baseline = state['baseline_tree']
    if any(tree.get(n) != entry for n, entry in baseline.items()):
        raise PublicationError('An existing repository file changed or disappeared.')
    items = {i['path']: i for i in row['files']}
    if set(tree)-set(baseline)-set(items):
        raise PublicationError('Repository changed outside the exact reviewed append inventory.')
    present = set(tree)&set(items)
    for name in present:
        entry, item = tree[name], items[name]
        if entry['size'] != item['bytes'] or entry.get('lfs_sha256') and entry['lfs_sha256'] != item['sha256']:
            raise PublicationError('Remote file metadata differs from reviewed evidence.')
        if kind == 'space' and bool(entry.get('lfs_sha256')) != any(name.endswith(s) for s in state['lfs_suffixes']):
            raise PublicationError('Space binary representation differs from pinned LFS rules.')
    verify_public(row, info.sha, downloader, present)
    known = {n for b in state['batches'] for n in b['paths']}
    if not known <= present:
        raise PublicationError('A previously published evidence file disappeared.')
    unknown = present-known
    anchor = state['batches'][-1]['commit'] if state['batches'] else state.get('created_sha') or state['baseline_sha']
    if unknown:
        pending = state.get('pending')
        if pending is None or set(pending['paths']) != unknown or pending['parent_commit'] != anchor or info.sha == anchor:
            raise PublicationError('Unknown remote additions do not exactly match the durable pending batch.')
        state['batches'].append(dict(parent_commit=anchor, commit=info.sha, paths=pending['paths'], reconciled=True))
        state['pending'] = None
        _atomic(journal_path, journal)
    elif info.sha != anchor:
        raise PublicationError('Repository head moved outside its journaled commit chain.')
    elif state.get('pending') is not None:
        # Exact prior head and no additions means the pending commit did not land.
        state['pending'] = None
        _atomic(journal_path, journal)
    return present, info.sha


def publish(path, *, api=None, downloader=None, operation_cls=None, progress=False):
    path = _plain(path)
    plan_sha = sha256(_read(path, MAX_PLAN)).hexdigest()
    plan = checked_plan(path)
    if plan_sha != sha256(_read(path, MAX_PLAN)).hexdigest():
        raise PublicationError('Publication plan changed during offline review.')
    if (path.parent/'publication-receipt.json').exists():
        raise PublicationError('A final publication receipt already exists; refusing to overwrite it.')
    if api is None:
        api, token = client(path.parent/'publisher-cache')
        if 'Dvidia' not in {r.get('name') for r in api.whoami().get('orgs', [])}:
            raise PublicationError('Authenticated account is not a member of Dvidia.')
    else:
        token = None
    journal_path = path.parent/JOURNAL_NAME
    if journal_path.exists():
        journal = _load_journal(journal_path, plan, plan_sha)
    else:
        journal = dict(schema=SCHEMA+'/journal', plan_sha256=plan_sha, repositories=_initial(plan, api, downloader))
        _atomic(journal_path, journal)  # Durable before any remote mutation.
    states = journal['repositories']
    # Reconcile all three destinations before issuing any new remote writes.
    for row in plan['repositories']:
        _live(row, states[row['repo_type']], api, downloader, journal=journal, journal_path=journal_path)
    if operation_cls is None:
        from huggingface_hub import CommitOperationAdd
        operation_cls = CommitOperationAdd
    for row in plan['repositories']:
        kind, state = row['repo_type'], states[row['repo_type']]
        if not state.get('baseline_sha') and not state.get('created_sha'):
            try:
                api.create_repo(row['repo_id'], repo_type=kind, private=False, exist_ok=False)
            except Exception as exc:
                raise PublicationError('Repository creation has an uncertain outcome; resume the same plan to reconcile.') from exc
        present, head = _live(row, state, api, downloader, journal=journal, journal_path=journal_path)
        for batch in _batches(row, {i['path'] for i in row['files']}-present):
            operations = []
            items = {i['path']: i for i in row['files']}
            for name in batch:
                raw = _read(path.parent/kind/name)
                if _receipt(raw) != {k: items[name][k] for k in ('bytes', 'sha256')}:
                    raise PublicationError('Reviewed bytes changed before freezing an upload batch.')
                if token and token.encode() in raw:
                    raise PublicationError('Credential detected in an upload artifact.')
                op = operation_cls(path_in_repo=name, path_or_fileobj=raw)
                if kind == 'space':
                    op._upload_mode = 'lfs' if any(name.endswith(s) for s in state['lfs_suffixes']) else 'regular'
                    op._should_ignore = False
                operations.append(op)
            state['pending'] = dict(parent_commit=head, paths=batch)
            _atomic(journal_path, journal)  # Exact intended batch survives ambiguous replies.
            try:
                committed = api.create_commit(row['repo_id'], repo_type=kind, parent_commit=head,
                    operations=operations, commit_message='Publish learned PiPER feedback v0.2 evidence (bounded batch)')
            except Exception as exc:
                raise PublicationError('Batch outcome is uncertain; resume the identical plan to reconcile exact public bytes.') from exc
            revision = getattr(committed, 'oid', None)
            if not isinstance(revision, str) or not COMMIT.fullmatch(revision):
                raise PublicationError('Commit returned no valid revision; resume the same plan.')
            verify_public(row, revision, downloader, set(batch))
            if kind == 'space':
                tree = _tree(api, row['repo_id'], kind, revision)
                for op in operations:
                    entry = tree.get(op.path_in_repo)
                    if not entry or not _matches(op.path_or_fileobj, entry) or bool(entry.get('lfs_sha256')) != (op._upload_mode == 'lfs'):
                        raise PublicationError('Committed Space representation differs from pinned Git/LFS policy.')
            state['batches'].append(dict(parent_commit=head, commit=revision, paths=batch, reconciled=False))
            state['pending'] = None
            _atomic(journal_path, journal)
            head = revision
            if progress:
                print(json.dumps(dict(phase='batch-verified', repo_type=kind, files=len(batch),
                    bytes=sum(items[n]['bytes'] for n in batch), commit=revision)), flush=True)
    results = []
    for row in plan['repositories']:
        present, head = _live(row, states[row['repo_type']], api, downloader, journal=journal, journal_path=journal_path)
        if present != {i['path'] for i in row['files']}:
            raise PublicationError('Final repository is missing reviewed evidence.')
        kind = row['repo_type']
        results.append(dict(repo_id=row['repo_id'], repo_type=kind, commit=head, verified_files=len(present),
            url='https://huggingface.co/'+{'model': '', 'dataset': 'datasets/', 'space': 'spaces/'}[kind]+row['repo_id']))
    receipt = dict(schema=SCHEMA+'/receipt', plan_sha256=plan_sha, source_commit=plan['source_commit'],
        source_integrity_sha256=plan['source_integrity_sha256'], bindings=plan['bindings'], repositories=results,
        space_baseline_commit=states['space']['baseline_sha'], space_baseline_files_preserved=len(states['space']['baseline_tree']),
        anonymous_readback=True, viewer_url=VIEWER_URL, physical_robot_ready=False)
    _atomic(path.parent/'publication-receipt.json', receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('stage')
    for name in ('run', 'viewer', 'output'):
        prep.add_argument('--'+name, required=True, type=Path)
    prep.add_argument('--source-commit', required=True)
    prep.add_argument('--space-baseline', required=True)
    for command in ('check', 'publish'):
        commands.add_parser(command).add_argument('plan', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'stage':
            result = stage(args.run, args.viewer, args.source_commit, args.space_baseline, args.output)
        elif args.command == 'check':
            result = checked_plan(args.plan)
        else:
            print(json.dumps(publish(args.plan, progress=True), indent=2, sort_keys=True))
            return 0
        print(json.dumps(dict(repositories=[dict(repo_id=r['repo_id'], files=len(r['files']),
            bytes=sum(i['bytes'] for i in r['files'])) for r in result['repositories']],
            network=False, credentials_read=False, remote_mutations=False), indent=2, sort_keys=True))
    except KeyboardInterrupt:
        print('Interrupted; resume the identical plan to reconcile remote state.', file=sys.stderr)
        return 130
    except Exception as exc:
        if isinstance(exc, PublicationError):
            print(str(exc), file=sys.stderr)
        else:
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            print(f'Publication stopped: {type(exc).__name__}; HTTP status {status}. Raw error suppressed.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

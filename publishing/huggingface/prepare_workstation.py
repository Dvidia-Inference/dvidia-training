"""Stage the live workstation and separate public evidence beside prior artifacts."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile
import zlib

import publish_robot_sim as publisher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
NAMESPACE = 'workstation/v01'
EVIDENCE_NAMESPACE = 'workstation-evidence/v01'
EVIDENCE_SCHEMA = 'dvidia.live-workstation-evidence/v1'
MAX_EVIDENCE_FILES = 128
MAX_EVIDENCE_BYTES = 128*1024*1024
EVIDENCE_SEAL = 'evidence-integrity.json'
PRIVATE_PATTERN = re.compile(
    rb'(?:/(?:Users|home)/|/private/tmp/|/var/folders/|[A-Za-z]:\\Users\\|'
    rb'hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|'
    rb'sk-[A-Za-z0-9_-]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)')


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()


def record(raw):
    return {'bytes': len(raw), 'sha256': sha256(raw).hexdigest()}


def _private(raw):
    publisher._no_private_bytes(raw)
    if PRIVATE_PATTERN.search(raw):
        raise ValueError('Possible credential or private path in workstation evidence')


def _png(raw):
    """Bound screenshot structure; reject text/EXIF metadata and trailing data."""
    if raw[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('Evidence PNG signature mismatch')
    offset, chunks, data_seen = 8, [], False
    allowed = {b'IHDR', b'PLTE', b'IDAT', b'IEND', b'tRNS', b'sRGB', b'gAMA', b'cHRM', b'pHYs'}
    while offset < len(raw):
        if offset+12 > len(raw):
            raise ValueError('Evidence PNG is truncated')
        length = struct.unpack_from('>I', raw, offset)[0]
        kind, end = raw[offset+4:offset+8], offset+12+length
        if kind not in allowed or end > len(raw):
            raise ValueError('Evidence PNG has unsupported metadata or chunk size')
        content = raw[offset+8:offset+8+length]
        if zlib.crc32(kind+content)&0xffffffff != struct.unpack_from('>I', raw, end-4)[0]:
            raise ValueError('Evidence PNG checksum mismatch')
        if kind == b'IHDR':
            if chunks or length != 13:
                raise ValueError('Evidence PNG header mismatch')
            width, height = struct.unpack_from('>II', content)
            if not 1 <= width <= 16384 or not 1 <= height <= 16384 or width*height > 32*1024*1024:
                raise ValueError('Evidence PNG dimensions exceed bounds')
        if kind == b'IDAT':
            data_seen = True
        chunks.append(kind)
        offset = end
        if len(chunks) > 4096:
            raise ValueError('Evidence PNG chunk count exceeds bounds')
        if kind == b'IEND':
            if length or offset != len(raw) or not data_seen:
                raise ValueError('Evidence PNG terminator mismatch')
            break
    if not chunks or chunks[0] != b'IHDR' or chunks[-1] != b'IEND':
        raise ValueError('Evidence PNG lacks its image framing')


def read_evidence(directory):
    """Freeze bounded data/screenshots and check decoded JSON strings for privacy."""
    from dvidia_training.reference_replay import _decode
    root = Path(directory).absolute()
    publisher._plain(root)
    if not root.is_dir():
        raise ValueError('Evidence requires a plain directory')
    files, total, entries = {}, 0, 0
    for path in root.rglob('*'):
        entries += 1
        if entries > 2*MAX_EVIDENCE_FILES:
            raise ValueError('Evidence directory entry count exceeds bounds')
        publisher._plain(path)
        name = path.relative_to(root).as_posix()
        if (len(name) > 240 or len(path.relative_to(root).parts) > 8
                or re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*(?:/[A-Za-z0-9_-][A-Za-z0-9_.-]*)*', name) is None):
            raise ValueError('Evidence path must be a bounded safe relative name')
        if path.is_dir():
            continue
        publisher._artifact_name(EVIDENCE_NAMESPACE+'/'+name)
        if not path.is_file() or path.suffix not in {'.json', '.md', '.png'}:
            raise ValueError('Evidence accepts only plain JSON, Markdown and PNG files')
        if name == EVIDENCE_SEAL:
            raise ValueError('Evidence seal name is reserved for the generated inventory')
        raw = publisher._bytes(path, publisher.MAX_FILE_BYTES)
        _private(raw)
        if path.suffix == '.json':
            value = _decode(raw)
            stack = [(value, 0)]
            while stack:
                item, depth = stack.pop()
                if depth > 64:
                    raise ValueError('Evidence JSON nesting exceeds bounds')
                if type(item) is dict:
                    for key, child in item.items():
                        _private(key.encode('utf-8'))
                        stack.append((child, depth+1))
                elif type(item) is list:
                    stack.extend((child, depth+1) for child in item)
                elif type(item) is str:
                    _private(item.encode('utf-8'))
        elif path.suffix == '.md':
            try:
                raw.decode('utf-8')
            except UnicodeError as exc:
                raise ValueError('Evidence Markdown must be UTF-8') from exc
            if b'\0' in raw:
                raise ValueError('Evidence Markdown contains binary content')
        else:
            _png(raw)
        files[name] = raw
        total += len(raw)
        if len(files) > MAX_EVIDENCE_FILES or total > MAX_EVIDENCE_BYTES:
            raise ValueError('Evidence exceeds its bounded publication budget')
    if 'README.md' not in files or not files['README.md'].strip():
        raise ValueError('Evidence requires a nonempty README.md')
    return files


def stage(base_plan, workstation, output, source_commit, expected_remote_commit, *, evidence):
    """Freeze a complete additive release against an explicit remote parent."""
    if any(type(value) is not str or re.fullmatch(r'[a-f0-9]{40}', value) is None for value in
           (source_commit, expected_remote_commit)):
        raise ValueError('Exact source and remote parent commits are required')
    base_plan = Path(base_plan).absolute()
    base = publisher.checked_plan(base_plan)
    root = Path(workstation).absolute()
    publisher._plain(root)
    evidence_root = Path(evidence).absolute()
    if root == evidence_root or root in evidence_root.parents or evidence_root in root.parents:
        raise ValueError('Workstation and evidence inputs require separate directory trees')
    evidence_files = read_evidence(evidence_root)
    from dvidia_training.workstation import inspect_export
    inspect_export(root)
    payload = {f['path']: publisher._bytes(base_plan.parent/'space'/f['path'], publisher.MAX_FILE_BYTES)
               for f in base['repository']['files']}
    if any(name.startswith(namespace+'/') for name in payload for namespace in (NAMESPACE, EVIDENCE_NAMESPACE)):
        raise ValueError('First workstation release must not replace an existing workstation or evidence namespace')
    added = {}
    for path in sorted(root.rglob('*')):
        publisher._plain(path)
        if not path.is_file():
            continue
        name = NAMESPACE+'/'+path.relative_to(root).as_posix()
        publisher._artifact_name(name)
        raw = publisher._bytes(path, publisher.MAX_FILE_BYTES)
        publisher._no_private_artifact_bytes(name, raw)
        added[name] = raw
    if not 8 <= len(added) <= 128 or sum(map(len, added.values())) > 96*1024*1024:
        raise ValueError('Workstation exceeds its release budget')
    seal = {'schema': EVIDENCE_SCHEMA, 'source_commit': source_commit,
        'scope': 'Complete public evidence inventory; hashes establish consistency, not authentication or physical qualification',
        'workstation_path': NAMESPACE,
        'workstation_integrity': record(added[NAMESPACE+'/workstation-integrity.json']),
        'files': {name: record(raw) for name, raw in sorted(evidence_files.items())}}
    evidence_files[EVIDENCE_SEAL] = encoded(seal)
    added_evidence = {EVIDENCE_NAMESPACE+'/'+name: raw for name, raw in evidence_files.items()}
    payload.update(added)
    payload.update(added_evidence)
    index = payload['index.html'].decode()
    marker = '<section aria-labelledby="assets-title">'
    if index.count(marker) != 1:
        raise ValueError('Existing asset-library entry point has changed')
    card = ('<section aria-labelledby="workstation-title"><h2 id="workstation-title">A live PiPER shoe workstation.</h2>'
            '<p>Drag the gripper target, open and close the jaws, and push a free rigid shoe. '
            'MuJoCo physics runs locally in your browser. Explore contacts and download a session receipt.</p>'
            '<p>Authored control and synthetic shoe properties; no learned policy or hardware qualification.</p>'
            '<a href="workstation/v01/index.html">Enter the live workstation →</a></section>')
    payload['index.html'] = index.replace(marker, card+marker).encode()
    payload['README.md'] += ('\n\n## Live PiPER shoe workstation v0.1\n\n'
        '[Open the live workstation](workstation/v01/index.html). MuJoCo 3.15.0 WASM runs in a browser worker. '
        'Drag targets, bounded joint commands, native gripper actuation and free-object contact are live. '
        'The shoe is a rigid synthetic proxy with uncalibrated properties. No learned skill or physical robot qualification. '
        '[Complete evidence and failures](workstation-evidence/v01/README.md), '
        '[evidence file hashes](workstation-evidence/v01/evidence-integrity.json), and '
        '[research report](https://research.dvidia.org/papers/live-piper-shoe-workstation-v01/). '
        'DVIDIA code and shoes: MIT; PiPER assets: MIT; Three.js: MIT; MuJoCo: Apache-2.0 with retained dependency notices.\n').encode()
    provenance = json.loads(payload['provenance.json'])
    provenance['live_workstation'] = {'path': NAMESPACE, 'source_commit': source_commit,
        'mujoco': '3.15.0', 'physical_robot_ready': False, 'learned_policy': False,
        'files': {n[len(NAMESPACE)+1:]: record(b) for n, b in added.items()},
        'evidence': {'path': EVIDENCE_NAMESPACE, 'source_commit': source_commit,
            'integrity': record(evidence_files[EVIDENCE_SEAL]),
            'files': {name: record(raw) for name, raw in sorted(evidence_files.items())}}}
    payload['provenance.json'] = encoded(provenance)
    destination = Path(output).absolute()
    publisher._plain(destination)
    if destination.exists():
        raise ValueError('Workstation publication output must be fresh')
    if any(parent == directory for parent in destination.parents for directory in (root, evidence_root, base_plan.parent)):
        raise ValueError('Publication output must be outside input trees')
    if len(payload) > publisher.MAX_FILES or sum(map(len, payload.values())) > publisher.MAX_TOTAL_BYTES:
        raise ValueError('Complete publication exceeds its file or byte budget')
    plan = {**base, 'source_commit': source_commit, 'expected_remote_commit': expected_remote_commit,
        'remove_files': [], 'repository': {**base['repository'],
        'files': [{'path': n, **record(b)} for n, b in sorted(payload.items())]},
        'total_bytes': sum(map(len, payload.values()))}
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.workstation-publication-', dir=destination.parent))
    try:
        for name, raw in payload.items():
            path = staging/'space'/name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(raw)
        (staging/'publication-plan.json').write_bytes(encoded(plan))
        # Reinspect copied simulation bytes. Evidence is outside this strict tree.
        inspect_export(staging/'space'/NAMESPACE)
        publisher.checked_plan(staging/'publication-plan.json')
        if destination.exists() or destination.is_symlink():
            raise ValueError('Publication output must remain fresh')
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {'plan': str(destination/'publication-plan.json'), 'files': len(payload),
            'added_files': len(added), 'evidence_files': len(added_evidence), 'total_bytes': plan['total_bytes']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-plan', required=True)
    parser.add_argument('--workstation', required=True)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--expected-remote-commit', required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.base_plan, args.workstation, args.output,
                           args.source_commit, args.expected_remote_commit, evidence=args.evidence)))


if __name__ == '__main__':
    main()

"""Append a sealed prerequisites release without replacing historical artifacts.

snapshot is read-only; stage/check are offline. Publishing needs an explicit
source/results/QA gate and retains a complete Git/LFS inventory of the parent.
Credentials use the existing in-memory macOS Keychain adapter only.
"""
from __future__ import annotations

import argparse
from hashlib import sha1, sha256
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import sys
import zlib

from publish import PublicationError, checked_remote, client, verified_downloads

REPO_ID = 'Dvidia/robot-simulation-lab'
NAMESPACE = 'prerequisites/v01'
RELEASES = {
    NAMESPACE: {'schema': 'dvidia.skill-prerequisites/v1', 'provenance_key': 'prerequisites',
        'commit_message': 'Publish two native skill-prerequisite simulations and complete benchmark evidence'},
    'prerequisites/v02': {'schema': 'dvidia.continuous-shoe-grip/v2', 'provenance_key': 'continuous_shoe_grip',
        'commit_message': 'Publish continuous shoe grip comparison and complete benchmark evidence'},
    'prerequisites/v03': {'schema': 'dvidia.graspability/v3', 'provenance_key': 'shoe_graspability',
        'commit_message': 'Publish shoe graspability diagnostics and complete benchmark evidence'},
}
ROOT_FILES = {'index.html', 'README.md', 'provenance.json'}
SHA = re.compile(r'[a-f0-9]{40}')
MAX_FILE = 32 * 1024 * 1024
MAX_TOTAL = 192 * 1024 * 1024
MAX_DECODED = 128 * 1024 * 1024
MAX_FILES = 256


def release_config(namespace):
    if type(namespace) is not str or namespace not in RELEASES:
        raise PublicationError('Unsupported explicit release namespace.')
    return RELEASES[namespace]


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def record(raw):
    return {'bytes': len(raw), 'sha256': sha256(raw).hexdigest()}


def plain(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise PublicationError('Publication paths may not contain symlinks.')
    return path


def read(path, limit=MAX_FILE):
    path = plain(path)
    if not path.is_file() or path.stat().st_size > limit:
        raise PublicationError('Expected a bounded plain publication file.')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise PublicationError('Publication file exceeds its size bound.')
    return raw


def path_name(name):
    if (type(name) is not str or not name or '\\' in name
            or re.search(r'[\x00-\x20\x7f]', name)):
        raise PublicationError('Invalid artifact path.')
    path = PurePosixPath(name)
    if (path.is_absolute() or path.as_posix() != name
            or any(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', part) is None for part in path.parts)):
        raise PublicationError('Artifact paths must be plain relative names.')
    if path.suffix not in {'.html', '.js', '.json', '.txt', '.md', '.png', '.jpg', '.svg', '.bin', '.gz', '.xml'} and path.name != 'LICENSE':
        raise PublicationError('Unsupported prerequisites artifact type.')
    if path.suffix == '.gz' and not name.endswith(('.json.gz', '.bin.gz')):
        raise PublicationError('Only bounded JSON or binary gzip artifacts are allowed.')
    return path


def inspect_bytes(name, raw):
    path_name(name)
    decoded = raw
    if name.endswith('.gz'):
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            decoded = decoder.decompress(raw, MAX_DECODED + 1)
        except zlib.error as exc:
            raise PublicationError('Invalid compressed artifact.') from exc
        if len(decoded) > MAX_DECODED or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise PublicationError('Compressed artifact exceeds its bound or contains trailing data.')
    candidates = [raw, decoded]
    if name.endswith(('.json', '.json.gz')):
        try:
            value = json.loads(decoded, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            candidates.append(json.dumps(value, ensure_ascii=False, allow_nan=False).encode())
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise PublicationError('Invalid finite JSON artifact.') from exc
    if any(re.search(rb'hf_[A-Za-z0-9]{30,}', data) or b'/Users/roguedev/' in data for data in candidates):
        raise PublicationError('Possible credential or private path in artifact.')


def inventory(root):
    root = plain(root)
    if not root.is_dir():
        raise PublicationError('Expected a plain artifact directory.')
    payload, total = {}, 0
    for path in sorted(root.rglob('*')):
        plain(path)
        if path.is_dir():
            continue
        name = path.relative_to(root).as_posix()
        raw = read(path)
        inspect_bytes(name, raw)
        payload[name] = raw
        total += len(raw)
        if len(payload) > MAX_FILES or total > MAX_TOTAL:
            raise PublicationError('Prerequisites release exceeds its publication budget.')
    return payload


def graspability_scorer():
    """Load the repository's independent stdlib scorer, never uploaded code."""
    path = Path(__file__).resolve().parents[2] / 'src/dvidia_training/graspability_protocol.py'
    spec = importlib.util.spec_from_file_location('_dvidia_publication_graspability_score', path)
    if spec is None or spec.loader is None:
        raise PublicationError('Independent graspability scorer is unavailable.')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.score_result


def check_export(root, source_commit=None, namespace=NAMESPACE):
    config = release_config(namespace)
    payload = inventory(root)
    if not {'index.html', 'benchmark.json', 'protocol.json', 'integrity.json'} <= payload.keys():
        raise PublicationError('Missing benchmark, protocol, viewer or integrity seal.')
    seal = json.loads(payload['integrity.json'])
    if seal.get('files') != {name: record(raw) for name, raw in payload.items() if name != 'integrity.json'}:
        raise PublicationError('Release inventory or bytes differ from the integrity seal.')
    protocol, benchmark = (json.loads(payload[name]) for name in ('protocol.json', 'benchmark.json'))
    source = benchmark.get('source', {})
    if (protocol.get('schema') != config['schema']
            or benchmark.get('schema') != protocol['schema']
            or SHA.fullmatch(protocol.get('source_commit', '') or '') is None
            or source.get('commit') != protocol['source_commit']
            or (source_commit is not None and source['commit'] != source_commit)
            or source.get('protocol_sha256') != record(payload['protocol.json'])['sha256']
            or benchmark.get('split') != protocol.get('split')
            or benchmark.get('learned_policy') is not False or benchmark.get('hardware_validated') is not False):
        raise PublicationError('Source, protocol and simulation-only scope must agree.')
    planned, measured = protocol.get('attempts'), benchmark.get('attempts')
    if (type(planned) is not list or type(measured) is not list or not 1 <= len(planned) <= 100
            or len(measured) != len(planned) or len({row['id'] for row in planned}) != len(planned)):
        raise PublicationError('Every planned attempt must occur exactly once.')
    score_graspability = graspability_scorer() if namespace == 'prerequisites/v03' else None
    for expected, actual in zip(planned, measured):
        if (any(actual.get(name) != expected.get(name) for name in ('id', 'task', 'policy', 'case'))
                or type(actual.get('valid')) is not bool or type(actual.get('success')) is not bool
                or (actual['success'] and not actual['valid'])):
            raise PublicationError('Measured attempts differ from the frozen protocol.')
        name = 'trials/' + actual['id'] + '.json'
        receipt = json.loads(payload[name]) if name in payload else None
        if type(receipt) is not dict:
            raise PublicationError('Complete attempt receipts must match the benchmark.')
        trace = receipt.pop('native_trace', None)
        if namespace in ('prerequisites/v02', 'prerequisites/v03') and actual['valid']:
            columns = actual.get('native_trace_columns')
            score = actual.get('independent_score')
            score_schema = ('dvidia.graspability-score/v3' if namespace == 'prerequisites/v03'
                            else 'dvidia.continuous-shoe-grip-score/v2')
            if (type(score) is not dict or score.get('schema') != score_schema
                    or score.get('valid') is not True or type(score.get('task_success')) is not bool
                    or score['task_success'] != actual['success']
                    or re.fullmatch(r'[a-f0-9]{64}', score.get('compiled_model_sha256', '') or '') is None
                    or type(trace) is not list or not trace
                    or type(columns) is not list or not columns
                    or any(type(row) is not list or len(row) != len(columns) for row in trace)):
                raise PublicationError('Valid grip attempts require an independent score and complete native trace columns.')
            if namespace == 'prerequisites/v03':
                required_columns = {'t_s', 'tool_object_force_n', 'object_robot_force_n',
                    'left_finger_force_n', 'right_finger_force_n', 'object_x_m',
                    'object_y_m', 'object_z_m', 'world_object_force_n'}
                reference = score.get('reference')
                if (not required_columns <= set(columns) or len(set(columns)) != len(columns)
                        or re.fullmatch(r'[a-f0-9]{64}', actual.get('robot_actuator_sha256', '') or '') is None
                        or type(score.get('strict_retention_pass')) is not bool
                        or type(score.get('final_window_complete')) is not bool
                        or type(score.get('final_window_ticks')) is not int
                        or score.get('relative_lift_min_m') != .035
                        or score.get('required_final_window_s') != .5
                        or actual.get('policy') != 'shared-v03'
                        or actual.get('task') != 'in-reach-graspability'
                        or actual['case'].get('object_shape') not in ('easy-block', 'simplified-shoe', 'current-shoe')
                        or actual['case'].get('tool_condition') not in ('mounted', 'absent')):
                    raise PublicationError('Graspability attempts require the declared direct-lift evidence and shared policy.')
                if actual['success'] and (score['strict_retention_pass'] is not True
                        or score['final_window_complete'] is not True or score['final_window_ticks'] != 250
                        or type(reference) is not dict or reference.get('valid') is not True
                        or reference.get('window_ticks') != 50
                        or type(reference.get('t_s')) not in (int, float) or abs(reference['t_s'] - .4) > 1e-8
                        or any(reference.get(k) is not True for k in (
                            'world_supported', 'robot_clear', 'position_stable', 'at_declared_initial_time'))):
                    raise PublicationError('A successful graspability attempt needs an initial settled reference and full retention window.')
        if trace is not None:
            if (type(trace) is not list or len(trace) > 100000
                    or any(type(row) is not list or not 1 <= len(row) <= 64
                           or any(type(value) not in (int, float) for value in row) for row in trace)
                    or any(b[0] <= a[0] for a, b in zip(trace, trace[1:]))):
                raise PublicationError('Native trace must contain bounded numeric observations in time order.')
            if namespace == 'prerequisites/v03' and actual['valid']:
                times = [row[columns.index('t_s')] for row in trace]
                if (min(abs(times[0]), abs(times[0] - .002)) > 1e-8
                        or any(abs(b - a - .002) > 1e-8 for a, b in zip(times, times[1:]))):
                    raise PublicationError('Graspability evidence must retain every native 2 ms observation.')
                try:
                    independently_derived = score_graspability({**actual, 'native_trace': trace})
                except (ValueError, TypeError, KeyError, IndexError) as exc:
                    raise PublicationError('Graspability trace cannot independently establish its declared evidence.') from exc
                if any(score.get(name) != value for name, value in independently_derived.items()):
                    raise PublicationError('Graspability score differs from independently recomputed native evidence.')
        if receipt != actual:
            raise PublicationError('Complete attempt receipts must match the benchmark.')
    if namespace == 'prerequisites/v03':
        fingerprints = {row['robot_actuator_sha256'] for row in measured if row['valid']}
        if len(fingerprints) > 1:
            raise PublicationError('Graspability contrasts must retain one shared robot actuator fingerprint.')
    return payload


def git_blob(raw):
    return sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


def tree(api, revision):
    result = {}
    for item in api.list_repo_tree(REPO_ID, repo_type='space', revision=revision, recursive=True):
        if not hasattr(item, 'blob_id'):
            continue
        lfs = getattr(item, 'lfs', None)
        result[item.path] = {'blob': item.blob_id, 'size': item.size,
            'lfs_sha256': (lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)) if lfs else None}
    return result


def public_bytes(name, revision, limit=MAX_FILE):
    from huggingface_hub import hf_hub_download
    # Hub cache snapshots are symlinks owned by the SDK, never upload inputs.
    path = Path(hf_hub_download(REPO_ID, name, repo_type='space', revision=revision, token=False))
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise PublicationError('Public artifact exceeds its download bound.')
    return raw


def matches_tree(raw, entry):
    digest = entry.get('lfs_sha256')
    return len(raw) == entry['size'] and (sha256(raw).hexdigest() == digest if digest else git_blob(raw) == entry['blob'])


def snapshot(output, parent_commit, namespace=NAMESPACE):
    release_config(namespace)
    if SHA.fullmatch(parent_commit) is None:
        raise PublicationError('Snapshot requires the exact reviewed parent commit.')
    destination = plain(output)
    destination.mkdir(parents=True, exist_ok=False)
    # Public inspection needs no publisher credential.
    from huggingface_hub import HfApi
    api = HfApi(token=False)
    info = api.repo_info(REPO_ID, repo_type='space')
    checked_remote(info, 'space')
    if info.sha != parent_commit:
        raise PublicationError('Space moved; review its new parent before staging.')
    previous = tree(api, parent_commit)
    if any(name == namespace or name.startswith(namespace + '/') for name in previous):
        raise PublicationError('This release namespace already exists.')
    if not (ROOT_FILES | {'.gitattributes'}) <= previous.keys():
        raise PublicationError('Existing Space root files are incomplete.')
    for name in ROOT_FILES | {'.gitattributes'}:
        raw = public_bytes(name, parent_commit)
        if not matches_tree(raw, previous[name]):
            raise PublicationError('Public root file differs from its pinned Git/LFS entry.')
        (destination / name).write_bytes(raw)
    (destination / 'snapshot.json').write_bytes(encoded({'schema': 'dvidia.prerequisites-parent/v1',
        'repo_id': REPO_ID, 'namespace': namespace, 'parent_commit': parent_commit, 'tree': previous}))
    return {'parent_commit': parent_commit, 'retained_tree_entries': len(previous)}


def updated_roots(parent, source_commit, seal, namespace=NAMESPACE):
    config = release_config(namespace)
    index = parent['index.html'].decode()
    marker = '<section aria-labelledby="replay-title">'
    if index.count(marker) != 1:
        raise PublicationError('Expected one recorded-replay section in the existing landing page.')
    card = ('<section aria-labelledby="prerequisites-title"><h2 id="prerequisites-title">'
        'Before a skill can succeed.</h2><p>Explore a hinged shoe box and an out-of-reach shoe. '
        'Recorded native MuJoCo attempts retain failed access, withdrawal and tool-assisted reach. '
        'Authored policies; synthetic properties; no physical robot qualification.</p>'
        '<a href="prerequisites/v01/index.html">Inspect both simulations and every result →</a></section>')
    index = index.replace(marker, card + marker)
    readme = parent['README.md'] + ('\n\n## Skill prerequisites v0.1\n\n'
        '[Explore both simulations](prerequisites/v01/index.html). Recorded replays of native '
        'MuJoCo runs; run the open-source Python environments locally for fresh physics. '
        'The rake is mounted, not autonomously acquired. Lid tests measure access and recovery, '
        'not shoe extraction. Failures and full benchmark evidence remain visible.\n\n'
        f'[Pinned source](https://github.com/Dvidia-Inference/dvidia-training/tree/{source_commit}) · '
        '[Research](https://research.dvidia.org/papers/skill-prerequisites-v01/)\n').encode()
    if namespace == 'prerequisites/v02':
        index = parent['index.html'].decode()
        marker = '<section aria-labelledby="prerequisites-title">'
        if index.count(marker) != 1:
            raise PublicationError('Expected one prior prerequisites section in the existing landing page.')
        card = ('<section aria-labelledby="continuous-grip-title"><h2 id="continuous-grip-title">'
            'From reaching a shoe to keeping hold.</h2><p>Compare a fixed grasp sequence with continuous '
            'acquisition checks, a measured lift checkpoint and bounded grip-loss recovery. Recorded native MuJoCo '
            'attempts preserve the full comparison, failures and independent scoring. '
            'Authored policies; synthetic contact; no physical robot qualification.</p>'
            '<a href="prerequisites/v02/index.html">Inspect the grip comparison and every result →</a></section>')
        index = index.replace(marker, card + marker)
        readme = parent['README.md'] + ('\n\n## Continuous shoe grip v0.2\n\n'
            '[Explore the grip comparison](prerequisites/v02/index.html). Recorded replays of native '
            'MuJoCo runs compare the previous mounted-rake controller with continuous acquisition and '
            'grip-loss checks on matched scenes. Run the open-source Python benchmark locally for fresh '
            'physics. All planned attempts and independent outcome scores remain visible. '
            'Policies use privileged simulator observations; contact is synthetic and hardware transfer remains open.\n\n'
            f'[Pinned source](https://github.com/Dvidia-Inference/dvidia-training/tree/{source_commit}) · '
            '[Research](https://research.dvidia.org/papers/continuous-shoe-grip-v02/)\n').encode()
    if namespace == 'prerequisites/v03':
        index = parent['index.html'].decode()
        marker = '<section aria-labelledby="continuous-grip-title">'
        if index.count(marker) != 1:
            raise PublicationError('Expected one prior continuous-grip section in the existing landing page.')
        card = ('<section aria-labelledby="graspability-title"><h2 id="graspability-title">'
            'Give the gripper a fair test.</h2><p>Compare a small block, a simplified shoe and the current '
            'shoe across light and heavy loads, with the rake mounted or absent. One shared authored '
            'policy and independent retention checks expose how the setup affects the lift. '
            'Recorded native physics; rigid synthetic objects; no physical robot qualification.</p>'
            '<a href="prerequisites/v03/index.html">Explore the graspability comparison →</a></section>')
        index = index.replace(marker, card + marker)
        readme = parent['README.md'] + ('\n\n## Shoe graspability v0.3\n\n'
            '[Explore the graspability comparison](prerequisites/v03/index.html). Recorded native '
            'MuJoCo trials isolate shape, load and mounted-tool condition with one shared policy. '
            'The absent-tool condition starts without the rake; it does not demonstrate autonomous '
            'tool removal or a complete retrieval task. Every planned attempt and independent '
            'retention score remains visible. Run the Python benchmark locally for fresh physics. '
            'Objects are rigid, contact is synthetic and observations are privileged simulator state.\n\n'
            f'[Pinned source](https://github.com/Dvidia-Inference/dvidia-training/tree/{source_commit}) · '
            '[Research](https://research.dvidia.org/papers/shoe-graspability-v03/)\n').encode()
    provenance = json.loads(parent['provenance.json'])
    key = config['provenance_key']
    if key in provenance:
        raise PublicationError('Release provenance already exists.')
    provenance[key] = {'path': namespace, 'source_commit': source_commit,
        'integrity': record(seal), 'browser_mode': 'recorded-native-replay',
        'learned_policy': False, 'physical_robot_ready': False}
    return {'index.html': index.encode(), 'README.md': readme, 'provenance.json': encoded(provenance)}


def stage(export, parent, output, source_commit, namespace=NAMESPACE):
    release_config(namespace)
    if SHA.fullmatch(source_commit) is None:
        raise PublicationError('An exact source commit is required.')
    parent, destination, export = plain(parent), plain(output), plain(export)
    if any(destination == source or source in destination.parents or destination in source.parents
           for source in (parent, export)):
        raise PublicationError('Staging output must be outside both input trees.')
    baseline = json.loads(read(parent / 'snapshot.json', 2 * 1024 * 1024))
    if (baseline.get('repo_id') != REPO_ID or SHA.fullmatch(baseline.get('parent_commit', '')) is None
            or baseline.get('namespace', NAMESPACE) != namespace):
        raise PublicationError('Unexpected parent snapshot.')
    previous = baseline['tree']
    if any(name == namespace or name.startswith(namespace + '/') for name in previous):
        raise PublicationError('Prerequisites namespace must be new.')
    parent_files = {name: read(parent / name) for name in ROOT_FILES}
    attributes = read(parent / '.gitattributes', 65536)
    if any(not matches_tree(raw, previous[name]) for name, raw in {**parent_files, '.gitattributes': attributes}.items()):
        raise PublicationError('Pinned parent root bytes changed.')
    release = check_export(export, source_commit, namespace)
    payload = {namespace + '/' + name: raw for name, raw in release.items()}
    payload.update(updated_roots(parent_files, source_commit, release['integrity.json'], namespace))
    if set(payload) & set(previous) != ROOT_FILES:
        raise PublicationError('Only the three root pointers may replace existing files.')
    for name, raw in payload.items():
        inspect_bytes(name, raw)
    destination.mkdir(parents=True, exist_ok=False)
    for name, raw in payload.items():
        path = destination / 'payload' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    (destination / 'parent-gitattributes.txt').write_bytes(attributes)
    plan = {'schema': 'dvidia.prerequisites-publication/v1', 'repo_id': REPO_ID,
        'namespace': namespace, 'source_commit': source_commit, 'parent_commit': baseline['parent_commit'],
        'parent_tree': previous, 'parent_roots': {name: record(raw) for name, raw in parent_files.items()},
        'parent_attributes': record(attributes), 'files': {name: record(raw) for name, raw in sorted(payload.items())}}
    (destination / 'plan.json').write_bytes(encoded(plan))
    check(destination / 'plan.json')
    return {'plan': str(destination / 'plan.json'), 'files': len(payload), 'bytes': sum(map(len, payload.values()))}


def check(path):
    path = plain(path)
    plan = json.loads(read(path, 2 * 1024 * 1024))
    namespace = plan.get('namespace')
    release_config(namespace)
    if (plan.get('schema') != 'dvidia.prerequisites-publication/v1' or plan.get('repo_id') != REPO_ID
            or any(SHA.fullmatch(plan.get(name, '')) is None for name in ('source_commit', 'parent_commit'))):
        raise PublicationError('Unexpected prerequisites publication plan.')
    payload = inventory(path.parent / 'payload')
    if plan.get('files') != {name: record(raw) for name, raw in payload.items()}:
        raise PublicationError('Payload changed after staging.')
    if (set(payload) & set(plan['parent_tree']) != ROOT_FILES
            or any(name not in ROOT_FILES and not name.startswith(namespace + '/') for name in payload)
            or any(name == namespace or name.startswith(namespace + '/') for name in plan['parent_tree'])):
        raise PublicationError('Publication may only append the fresh namespace and change its three root pointers.')
    attributes = read(path.parent / 'parent-gitattributes.txt', 65536)
    if record(attributes) != plan['parent_attributes'] or not matches_tree(attributes, plan['parent_tree']['.gitattributes']):
        raise PublicationError('Parent metadata changed.')
    check_export(path.parent / 'payload' / namespace, plan['source_commit'], namespace)
    return plan, payload


def check_gate(gate, plan):
    namespace = plan['namespace']
    release_config(namespace)
    if (gate.get('source_commit') != plan['source_commit']
            or any(gate.get(name) is not True for name in ('tests_passed', 'browser_qa_passed', 'benchmark_reviewed'))
            or gate.get('benchmark_sha256') != plan['files'][namespace + '/benchmark.json']['sha256']
            or gate.get('integrity_sha256') != plan['files'][namespace + '/integrity.json']['sha256']):
        raise PublicationError('Publication requires source-bound benchmark, tests and browser acceptance.')


def verify_tree(before, after, payload, old_attributes, new_attributes):
    if set(after) != set(before) | set(payload):
        raise PublicationError('Publication added or removed unexpected paths.')
    unchanged = set(before) - set(payload) - {'.gitattributes'}
    if any(before[name] != after[name] for name in unchanged):
        raise PublicationError('Historical Git/LFS entry changed.')
    if not matches_tree(new_attributes, after['.gitattributes']) or not new_attributes.startswith(old_attributes):
        raise PublicationError('Existing Hub metadata changed.')
    additions = new_attributes[len(old_attributes):].splitlines()
    allowed = {(name + ' filter=lfs diff=lfs merge=lfs -text').encode() for name in payload}
    if len(additions) != len(set(additions)) or any(line not in allowed for line in additions):
        raise PublicationError('Unexpected Hub metadata addition.')
    return len(unchanged)


def publish(path, gate_path):
    path = plain(path)
    plan, payload = check(path)
    gate = json.loads(read(gate_path, 1024 * 1024))
    check_gate(gate, plan)
    api, token = client(path.parent / 'publisher-cache')
    if any(token.encode() in raw for raw in payload.values()):
        raise PublicationError('Credential occurs in upload bytes.')
    if 'Dvidia' not in {org.get('name') for org in api.whoami().get('orgs', [])}:
        raise PublicationError('Publisher is not a member of the intended organization.')
    pending_path = plain(path.parent / 'pending.json')
    plan_hash = record(read(path, 2 * 1024 * 1024))['sha256']
    if pending_path.exists():
        pending = json.loads(read(pending_path))
        if pending.get('plan_sha256') != plan_hash or SHA.fullmatch(pending.get('commit', '')) is None:
            raise PublicationError('Retry receipt belongs to another plan.')
        revision = pending['commit']
    else:
        info = api.repo_info(REPO_ID, repo_type='space')
        checked_remote(info, 'space')
        if info.sha != plan['parent_commit'] or tree(api, info.sha) != plan['parent_tree']:
            raise PublicationError('Space changed after the reviewed parent snapshot.')
        from huggingface_hub import CommitOperationAdd
        result = api.create_commit(REPO_ID, repo_type='space', parent_commit=plan['parent_commit'],
            operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=raw) for name, raw in payload.items()],
            commit_message=release_config(plan['namespace'])['commit_message'])
        revision = result.oid
        pending_path.write_bytes(encoded({'commit': revision, 'plan_sha256': plan_hash}))
    after = tree(api, revision)
    attributes = public_bytes('.gitattributes', revision, 65536)
    unchanged = verify_tree(plan['parent_tree'], after, payload,
        read(path.parent / 'parent-gitattributes.txt', 65536), attributes)
    row = {'repo_id': REPO_ID, 'repo_type': 'space', 'files': [
        {'path': name, **expected} for name, expected in plan['files'].items()]}
    verified_downloads(row, revision, False)
    receipt = {'repo_id': REPO_ID, 'namespace': plan['namespace'], 'commit': revision, 'parent_commit': plan['parent_commit'],
        'source_commit': plan['source_commit'], 'plan_sha256': plan_hash,
        'verified_public_files': len(payload), 'unchanged_historical_tree_entries': unchanged,
        'files': plan['files'], 'gate': gate}
    plain(path.parent / 'after-tree.json').write_bytes(encoded(after))
    plain(path.parent / 'publication-receipt.json').write_bytes(encoded(receipt))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('snapshot')
    command.add_argument('--output', required=True, type=Path)
    command.add_argument('--parent-commit', required=True)
    command.add_argument('--namespace', choices=RELEASES, default=NAMESPACE)
    command = commands.add_parser('stage')
    for name in ('export', 'parent', 'output'):
        command.add_argument('--' + name, required=True, type=Path)
    command.add_argument('--source-commit', required=True)
    command.add_argument('--namespace', choices=RELEASES, default=NAMESPACE)
    for name in ('check', 'publish'):
        command = commands.add_parser(name)
        command.add_argument('plan', type=Path)
        if name == 'publish':
            command.add_argument('--gate', required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'snapshot':
            result = snapshot(args.output, args.parent_commit, args.namespace)
        elif args.command == 'stage':
            result = stage(args.export, args.parent, args.output, args.source_commit, args.namespace)
        elif args.command == 'check':
            plan, payload = check(args.plan)
            result = {'files': len(payload), 'source_commit': plan['source_commit']}
        else:
            result = publish(args.plan, args.gate)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        # SDK exceptions may include credential-bearing request details.
        print(str(exc) if isinstance(exc, PublicationError) else 'Prerequisites publication stopped; raw error suppressed.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Correct one colliding YAML catalog tag after the initial v0.3 publication.

Only README.md and integrity.json may change; three metadata provenance files
are appended. The original publication's complete Git/LFS identities bind all
scientific bytes without downloading the multi-gigabyte dataset again. Stage
uses anonymous reads; publish alone accesses the named memory-only client.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys

import publish_grip_clock_v03 as pub
from publish import PublicationError


SCHEMA = 'dvidia.grip-clock-catalog-overlay/v0.3'
PLAN, JOURNAL, RECEIPT = 'catalog-plan.json', 'catalog-journal.json', 'catalog-receipt.json'
KINDS = ('model', 'dataset')
ORIGINAL_CARD = 'metadata/original-README.md'
ORIGINAL_SEAL = 'metadata/original-integrity.json'
OVERLAY = 'metadata/catalog-overlay.json'
APPENDED = (ORIGINAL_CARD, ORIGINAL_SEAL, OVERLAY)
ALLOWED = ('README.md', 'integrity.json', *APPENDED)
IMPLEMENTATION = ('fix_grip_clock_catalog_v03.py', 'publish_grip_clock_v03.py', 'publish.py', 'prepare.py')
MAX_BATCH = 16*1024**2


def _implementation():
    return {name: pub.receipt(pub.read(Path(__file__).with_name(name))) for name in IMPLEMENTATION}


def _initial(raw):
    value = pub.decoded(raw)
    if (type(value) is not dict or value.get('schema') != pub.SCHEMA+'/receipt'
            or value.get('anonymous_full_readback') is not True or value.get('physical_robot_ready') is not False
            or type(value.get('plan')) is not dict or set(value['plan']) != {'bytes', 'sha256'}
            or type(value['plan'].get('bytes')) is not int or not 0 < value['plan']['bytes'] <= pub.MAX_PLAN
            or not isinstance(value['plan'].get('sha256'), str) or pub.SHA.fullmatch(value['plan']['sha256']) is None
            or type(value.get('repositories')) is not list
            or [row.get('repo_type') for row in value['repositories']] != list(KINDS)):
        raise PublicationError('Catalog correction requires the completed exact initial model/dataset publication receipt.')
    trees, heads = {}, {}
    for row in value['repositories']:
        kind = row['repo_type']
        if (row.get('repo_id') != pub.REPOS[kind] or not isinstance(row.get('commit'), str)
                or pub.OID.fullmatch(row['commit']) is None or type(row.get('commits')) is not list
                or not row['commits']):
            raise PublicationError('Initial receipt destination or final revision changed.')
        tree, parent = {}, None
        for index, commit in enumerate(row['commits']):
            if (type(commit) is not dict or commit.get('index') != index
                    or commit.get('anonymous_readback') is not True
                    or not isinstance(commit.get('parent'), str) or pub.OID.fullmatch(commit['parent']) is None
                    or not isinstance(commit.get('commit'), str) or pub.OID.fullmatch(commit['commit']) is None
                    or index and commit['parent'] != parent
                    or type(commit.get('paths')) is not list or not commit['paths']
                    or len(set(commit['paths'])) != len(commit['paths'])
                    or type(commit.get('entries')) is not dict or set(commit['entries']) != set(commit['paths'])):
                raise PublicationError('Initial receipt does not retain an exact anonymously verified commit chain.')
            if commit.get('message') != pub._message(value['plan']['sha256'], kind, index):
                raise PublicationError('Initial commit message differs from its original hash-bound publisher intent.')
            pub._tree_valid(commit['entries'])
            if index == 0 and commit['paths'] != ['.gitattributes']:
                raise PublicationError('Initial new-repository creation must begin with its exact generated attributes.')
            if set(tree)&set(commit['paths']):
                raise PublicationError('Initial publication must contain a unique append-only payload prefix.')
            tree.update(commit['entries']); parent = commit['commit']
        if (parent != row['commit'] or type(row.get('verified_files')) is not int or row['verified_files'] != len(tree)
                or type(row.get('verified_bytes')) is not int
                or row['verified_bytes'] != sum(entry['size'] for entry in tree.values())
                or not {'README.md', 'integrity.json', '.gitattributes'} <= set(tree)):
            raise PublicationError('Initial receipt final head, file count or byte count differs from its committed inventory.')
        heads[kind], trees[kind] = parent, tree
    return value, heads, trees


def replace_tag(raw):
    """Change exactly one list value in the initial YAML tags field only."""
    if not raw.startswith(b'---\n'):
        raise PublicationError('Expected the existing LF-delimited YAML Hub card.')
    boundary = raw.find(b'\n---\n', 4)
    if boundary < 0:
        raise PublicationError('Hub card has no bounded closing YAML delimiter.')
    header, body = raw[:boundary+1], raw[boundary+1:]
    if header.count(b'tags:\n') != 1 or header.count(b'- piper\n') != 1 or b'- agilex-piper\n' in header:
        raise PublicationError('The YAML tags field must contain exactly one old colliding tag and no replacement tag.')
    lines, inside, hits = header.splitlines(keepends=True), False, 0
    for line in lines:
        if line == b'tags:\n':
            inside = True
        elif inside:
            if not line.startswith(b'- '):
                inside = False
            elif line == b'- piper\n':
                hits += 1
    if hits != 1:
        raise PublicationError('The colliding literal is outside the YAML tags list.')
    changed = header.replace(b'- piper\n', b'- agilex-piper\n', 1)+body
    if changed[changed.find(b'\n---\n', 4)+1:] != body:
        raise PublicationError('Catalog correction cannot modify card body content.')
    return changed


def _manifest(raw, kind, tree, card):
    value = pub.decoded(raw)
    if type(value) is not dict or type(value.get('files')) is not dict:
        raise PublicationError('Initial root integrity must contain its original declared file map.')
    files = value['files']
    if set(files) != set(tree)-{'integrity.json', '.gitattributes'}:
        raise PublicationError('Initial root inventory must cover every original file except its own seal and generated attributes.')
    for name, item in files.items():
        pub.relative(name)
        if (type(item) is not dict or set(item) != {'bytes', 'sha256'} or type(item['bytes']) is not int
                or not 0 <= item['bytes'] <= pub.MAX_FILE or not isinstance(item['sha256'], str)
                or pub.SHA.fullmatch(item['sha256']) is None or tree[name]['size'] != item['bytes']
                or tree[name]['lfs_sha256'] is not None and tree[name]['lfs_sha256'] != item['sha256']):
            raise PublicationError('Original inventory differs from its receipt-bound public Git/LFS tree.')
    if files.get('README.md') != pub.receipt(card):
        raise PublicationError('Original card bytes differ from their original declared SHA256 receipt.')
    if any(name in tree for name in APPENDED):
        raise PublicationError('Catalog provenance paths must be entirely new.')
    if kind not in KINDS:
        raise PublicationError('Only the fixed model and dataset cards can be corrected.')
    return value


def derive(kind, parent, tree, card, integrity, attributes, initial_receipt):
    if kind not in KINDS or not isinstance(parent, str) or pub.OID.fullmatch(parent) is None:
        raise PublicationError('Expected a fixed model/dataset and its exact initial final head.')
    pub._tree_valid(tree)
    for name, raw in [('README.md', card), ('integrity.json', integrity), ('.gitattributes', attributes)]:
        if not pub._matches(raw, tree.get(name, {})):
            raise PublicationError('Anonymous original metadata bytes differ from their receipt-bound public objects.')
        pub.scan(raw, name)
    if attributes != pub.ATTRIBUTES:
        raise PublicationError('Original generated attributes must retain their exact reviewed native LFS rules.')
    old = _manifest(integrity, kind, tree, card)
    new_card = replace_tag(card)
    overlay = dict(schema=SCHEMA, repo_id=pub.REPOS[kind], repo_type=kind, parent_commit=parent,
        initial_publication_receipt=initial_receipt, old_readme=pub.receipt(card), new_readme=pub.receipt(new_card),
        original_root_inventory=pub.receipt(integrity), original_card_path=ORIGINAL_CARD,
        original_inventory_path=ORIGINAL_SEAL, change='YAML tags list: piper -> agilex-piper',
        scope='tag_collision_only', readme_body_unchanged=True, scientific_file_bytes_changed=False,
        physical_robot_ready=False,
        verification='Original receipt-bound full Git/LFS tree preservation; only metadata files read back, no new multi-gigabyte science download.')
    payload = {'README.md': new_card, ORIGINAL_CARD: card, ORIGINAL_SEAL: integrity, OVERLAY: pub.encoded(overlay)}
    new = deepcopy(old)
    new['files']['README.md'] = pub.receipt(new_card)
    new['files']['.gitattributes'] = pub.receipt(attributes)
    for name in APPENDED:
        new['files'][name] = pub.receipt(payload[name])
    new['catalog_metadata_overlay'] = dict(schema=SCHEMA, parent_commit=parent,
        original_root_inventory=pub.receipt(integrity), overlay_record=OVERLAY, physical_robot_ready=False)
    payload['integrity.json'] = pub.encoded(new)
    if set(new['files']) != (set(tree)|set(APPENDED))-{'integrity.json'}:
        raise PublicationError('Corrected root inventory does not cover the exact resulting repository.')
    if sum(len(raw) for raw in payload.values()) > MAX_BATCH:
        raise PublicationError('Catalog metadata commit exceeds its bounded size.')
    for name, raw in payload.items():
        pub.scan(raw, name)
        if len(raw) > pub.MAX_FILE:
            raise PublicationError('Catalog metadata file exceeds its individual size bound.')
    return payload


def stage(initial_receipt, output, *, api=None, downloader=None):
    output = pub.plain(output)
    if output.exists():
        raise PublicationError('Catalog staging output must be fresh.')
    initial_raw = pub.read(initial_receipt, pub.MAX_PLAN)
    pub.scan(initial_raw, 'initial-publication-receipt.json')
    initial, heads, trees = _initial(initial_raw)
    initial_proof = pub.receipt(initial_raw)
    api, downloader = api or pub.public_client(), downloader or pub.download
    captured, payloads = {}, {}
    # All anonymous capture and derivation succeeds before any output is made.
    for kind in KINDS:
        head, tree = pub._remote(api, kind, heads[kind])
        if (head, tree) != (heads[kind], trees[kind]) or pub._remote(api, kind) != (head, tree):
            raise PublicationError('Initial final head or full public inventory moved before catalog correction.')
        files = {name: pub._downloaded(downloader(pub.REPOS[kind], name, kind, head))
                 for name in ('README.md', 'integrity.json', '.gitattributes')}
        payloads[kind] = derive(kind, head, tree, files['README.md'], files['integrity.json'],
                                files['.gitattributes'], initial_proof)
        if pub._remote(api, kind) != (head, tree):
            raise PublicationError('Repository moved during anonymous catalog staging.')
        captured[kind] = files
    for kind in KINDS:
        if pub._remote(api, kind) != (heads[kind], trees[kind]):
            raise PublicationError('A repository moved before the complete catalog staging boundary.')
    output.mkdir(parents=True, exist_ok=False)
    (output/'initial-publication-receipt.json').write_bytes(initial_raw)
    rows = []
    for kind in KINDS:
        directory = output/kind
        (directory/'before').mkdir(parents=True)
        (directory/'payload').mkdir()
        for name, raw in captured[kind].items():
            (directory/'before'/name).write_bytes(raw)
        for name, raw in payloads[kind].items():
            path = directory/'payload'/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        baseline = dict(repo_id=pub.REPOS[kind], repo_type=kind, commit=heads[kind], tree=trees[kind])
        pub.atomic(directory/'baseline.json', baseline, exclusive=True)
        rows.append(dict(repo_id=pub.REPOS[kind], repo_type=kind, parent_commit=heads[kind],
            baseline=pub.receipt(pub.read(directory/'baseline.json', pub.MAX_PLAN)),
            before={name: pub.receipt(raw) for name, raw in captured[kind].items()},
            payload={name: pub.receipt(raw) for name, raw in payloads[kind].items()},
            original_files=len(trees[kind]), resulting_files=len(trees[kind])+3))
    (output/'implementation').mkdir()
    implementation = _implementation()
    for name in implementation:
        (output/'implementation'/name).write_bytes(pub.read(Path(__file__).with_name(name)))
    plan = dict(schema=SCHEMA, initial_publication_receipt=initial_proof, repositories=rows,
                implementation=implementation, scope='tag_collision_only', physical_robot_ready=False)
    pub.atomic(output/PLAN, plan, exclusive=True)
    checked_plan(output/PLAN)
    return plan


def checked_plan(path):
    path = pub.plain(path)
    plan = pub.decoded(pub.read(path, pub.MAX_PLAN))
    if (type(plan) is not dict or plan.get('schema') != SCHEMA or plan.get('scope') != 'tag_collision_only'
            or plan.get('physical_robot_ready') is not False or plan.get('implementation') != _implementation()
            or type(plan.get('repositories')) is not list
            or [row.get('repo_type') for row in plan['repositories']] != list(KINDS)):
        raise PublicationError('Catalog plan scope, destinations or implementation changed.')
    initial_raw = pub.read(path.parent/'initial-publication-receipt.json', pub.MAX_PLAN)
    pub.scan(initial_raw, 'initial-publication-receipt.json')
    _, heads, trees = _initial(initial_raw)
    if plan.get('initial_publication_receipt') != pub.receipt(initial_raw):
        raise PublicationError('Catalog plan initial publication receipt changed.')
    if {item.name for item in (path.parent/'implementation').iterdir()} != set(IMPLEMENTATION):
        raise PublicationError('Catalog publisher snapshot inventory changed.')
    for name, expected in plan['implementation'].items():
        if pub.receipt(pub.read(path.parent/'implementation'/name)) != expected:
            raise PublicationError('Catalog publisher snapshot bytes changed.')
    payloads = {}
    for row in plan['repositories']:
        kind = row['repo_type']; directory = path.parent/kind
        baseline_raw = pub.read(directory/'baseline.json', pub.MAX_PLAN)
        baseline = pub.decoded(baseline_raw)
        if (row.get('repo_id') != pub.REPOS[kind] or row.get('parent_commit') != heads[kind]
                or row.get('baseline') != pub.receipt(baseline_raw)
                or baseline != dict(repo_id=pub.REPOS[kind], repo_type=kind, commit=heads[kind], tree=trees[kind])
                or row.get('original_files') != len(trees[kind]) or row.get('resulting_files') != len(trees[kind])+3):
            raise PublicationError('Catalog parent inventory differs from its exact initial public receipt.')
        expected_before = {'README.md', 'integrity.json', '.gitattributes'}
        if {item.name for item in (directory/'before').iterdir()} != expected_before:
            raise PublicationError('Original metadata inventory changed.')
        before = {name: pub.read(directory/'before'/name) for name in expected_before}
        if row.get('before') != {name: pub.receipt(raw) for name, raw in before.items()}:
            raise PublicationError('Original metadata bytes changed.')
        actual_names = set()
        for item in (directory/'payload').rglob('*'):
            pub.plain(item)
            if item.is_file():
                actual_names.add(item.relative_to(directory/'payload').as_posix())
        if actual_names != set(ALLOWED):
            raise PublicationError('Only the exact five catalog metadata paths may be published.')
        payload = {name: pub.read(directory/'payload'/name) for name in ALLOWED}
        if row.get('payload') != {name: pub.receipt(raw) for name, raw in payload.items()}:
            raise PublicationError('Catalog payload bytes changed.')
        derived = derive(kind, heads[kind], trees[kind], before['README.md'], before['integrity.json'],
                         before['.gitattributes'], pub.receipt(initial_raw))
        if payload != derived:
            raise PublicationError('Catalog payload differs from its exact YAML-only derivation and root inventory.')
        payloads[kind] = payload
    return plan, trees, payloads


def _message(plan_sha, kind):
    return 'DVIDIA v0.3 catalog tag correction '+plan_sha+' '+kind


def _wanted(tree, payload):
    return dict(tree, **{name: pub._entry(raw) for name, raw in payload.items()})


def _finish(api, downloader, row, tree, payload, revision):
    kind = row['repo_type']
    if not isinstance(revision, str) or pub.OID.fullmatch(revision) is None or revision == row['parent_commit']:
        raise PublicationError('Catalog commit reply lacks a new exact revision; retain its intent.')
    head, actual = pub._remote(api, kind)
    if head != revision or actual != _wanted(tree, payload):
        raise PublicationError('Catalog commit changed unexpected original scientific or metadata identities.')
    for name, raw in payload.items():
        if pub._downloaded(downloader(pub.REPOS[kind], name, kind, head)) != raw:
            raise PublicationError('Anonymous changed-file readback differs from the exact catalog payload.')
    if pub._remote(api, kind) != (head, actual):
        raise PublicationError('Repository moved during catalog metadata readback.')
    return head


def publish(path, *, api=None, downloader=None, operation_cls=None):
    path = pub.plain(path)
    with pub._lock(path.parent):
        plan, trees, payloads = checked_plan(path)
        plan_raw = pub.read(path, pub.MAX_PLAN); plan_sha = sha256(plan_raw).hexdigest()
        receipt_path, journal_path = path.parent/RECEIPT, path.parent/JOURNAL
        if receipt_path.exists():
            completed = pub.decoded(pub.read(receipt_path, pub.MAX_PLAN))
            if (type(completed) is not dict or completed.get('schema') != SCHEMA+'/receipt'
                    or completed.get('plan') != pub.receipt(plan_raw) or completed.get('anonymous_changed_file_readback') is not True
                    or completed.get('physical_robot_ready') is not False
                    or completed.get('initial_publication_receipt') != plan['initial_publication_receipt']):
                raise PublicationError('Completed catalog receipt differs from this exact reviewed plan.')
            return completed
        state = pub.decoded(pub.read(journal_path, pub.MAX_PLAN)) if journal_path.exists() else dict(
            schema=SCHEMA+'/journal', plan_sha256=plan_sha,
            repositories={row['repo_type']: dict(pending=False, commit=None, reconciled=False,
                parent_commit=row['parent_commit'], message=_message(plan_sha, row['repo_type']),
                paths=list(ALLOWED)) for row in plan['repositories']})
        if (state.get('schema') != SCHEMA+'/journal' or state.get('plan_sha256') != plan_sha
                or type(state.get('repositories')) is not dict or set(state['repositories']) != set(KINDS)):
            raise PublicationError('Catalog journal differs from its exact plan.')
        for kind, value in state['repositories'].items():
            row = next(row for row in plan['repositories'] if row['repo_type'] == kind)
            if (type(value) is not dict or set(value) != {'pending', 'commit', 'reconciled', 'parent_commit', 'message', 'paths'}
                    or type(value['pending']) is not bool or type(value['reconciled']) is not bool
                    or value['parent_commit'] != row['parent_commit'] or value['message'] != _message(plan_sha, kind)
                    or value['paths'] != list(ALLOWED)
                    or value['commit'] is not None and (not isinstance(value['commit'], str) or pub.OID.fullmatch(value['commit']) is None)
                    or value['commit'] is not None and value['pending']):
                raise PublicationError('Malformed catalog recovery state.')
        token = None
        if api is None:
            api, token = pub.client(path.parent/'publisher-cache')
            identity = api.whoami()
            if 'dvidia' not in {str(item.get('name', '')).lower() for item in identity.get('orgs', [])}:
                raise PublicationError('Publishing identity is outside the fixed Dvidia organization.')
        downloader = downloader or pub.download
        if token and any(token.encode() in raw for payload in payloads.values() for raw in payload.values()):
            raise PublicationError('Publishing credential appeared in a staged metadata file.')
        # Validate every destination before a fresh remote mutation. Reconcile
        # only the exact unknown child, using history membership, never dates.
        for row in plan['repositories']:
            kind = row['repo_type']; value = state['repositories'][kind]
            head, tree = pub._remote(api, kind)
            if value['commit'] is not None:
                if head != value['commit'] or tree != _wanted(trees[kind], payloads[kind]):
                    raise PublicationError('Completed catalog repository moved outside this journal.')
            elif value['pending'] and head != row['parent_commit']:
                before, after = pub._history(api, kind, row['parent_commit']), pub._history(api, kind, head)
                if set(after) != set(before)|{head} or after.get(head) != _message(plan_sha, kind):
                    raise PublicationError('Unknown catalog reply is not the exact intended immediate child.')
                value.update(commit=_finish(api, downloader, row, trees[kind], payloads[kind], head),
                             pending=False, reconciled=True)
                pub.atomic(journal_path, state)
            elif (head, tree) != (row['parent_commit'], trees[kind]) or pub._remote(api, kind) != (head, tree):
                raise PublicationError('Catalog parent moved outside its complete initial receipt-bound tree.')
        if not journal_path.exists():
            pub.atomic(journal_path, state, exclusive=True)
        for row in plan['repositories']:
            kind = row['repo_type']; value = state['repositories'][kind]
            if value['commit'] is not None:
                continue
            if _implementation() != plan['implementation'] or pub.read(path, pub.MAX_PLAN) != plan_raw:
                raise PublicationError('Catalog plan or publisher changed during publication.')
            for name, raw in payloads[kind].items():
                if pub.read(path.parent/kind/'payload'/name) != raw:
                    raise PublicationError('Frozen catalog payload changed before its commit.')
            if pub._remote(api, kind) != (row['parent_commit'], trees[kind]):
                raise PublicationError('Catalog parent moved before its guarded metadata commit.')
            value['pending'] = True; pub.atomic(journal_path, state)
            if operation_cls is None:
                from huggingface_hub import CommitOperationAdd
                operation_cls = CommitOperationAdd
            operations = []
            for name in ALLOWED:
                operation = operation_cls(path_in_repo=name, path_or_fileobj=payloads[kind][name])
                operation._upload_mode = 'regular'; operation._should_ignore = False
                operations.append(operation)
            try:
                result = api.create_commit(pub.REPOS[kind], repo_type=kind, parent_commit=row['parent_commit'],
                    operations=operations, commit_message=_message(plan_sha, kind))
            except Exception:
                raise PublicationError('Catalog commit reply is uncertain; preserve its exact pending intent.') from None
            value.update(commit=_finish(api, downloader, row, trees[kind], payloads[kind], getattr(result, 'oid', None)),
                         pending=False, reconciled=False)
            pub.atomic(journal_path, state)
        rows = []
        for row in plan['repositories']:
            kind = row['repo_type']; value = state['repositories'][kind]
            if pub._remote(api, kind) != (value['commit'], _wanted(trees[kind], payloads[kind])):
                raise PublicationError('Final catalog inventory differs from its reviewed metadata overlay.')
            unchanged = {name: entry for name, entry in trees[kind].items() if name not in ('README.md', 'integrity.json')}
            rows.append(dict(repo_id=pub.REPOS[kind], repo_type=kind, parent_commit=row['parent_commit'],
                commit=value['commit'], reconciled=value['reconciled'], changed_paths=list(ALLOWED),
                changed_files={name: pub.receipt(raw) for name, raw in payloads[kind].items()},
                original_files=row['original_files'], resulting_files=row['resulting_files'],
                unchanged_git_lfs_files=len(unchanged), unchanged_git_lfs_sha256=sha256(pub.encoded(unchanged)).hexdigest()))
        completed = dict(schema=SCHEMA+'/receipt', plan=pub.receipt(plan_raw),
            initial_publication_receipt=plan['initial_publication_receipt'], repositories=rows,
            scope='tag_collision_only', scientific_file_bytes_changed=False,
            anonymous_changed_file_readback=True, full_unchanged_git_lfs_identities_preserved=True,
            physical_robot_ready=False)
        pub.atomic(receipt_path, completed, exclusive=True)
        return completed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('stage')
    prepare.add_argument('--initial-receipt', required=True)
    prepare.add_argument('--output', required=True)
    for name in ('check', 'publish'):
        commands.add_parser(name).add_argument('plan')
    args = parser.parse_args(argv)
    try:
        if args.command == 'stage':
            result = stage(args.initial_receipt, args.output)
        elif args.command == 'check':
            result = checked_plan(args.plan)[0]
        else:
            result = publish(args.plan)
        print(json.dumps(dict(schema=result['schema'], command=args.command, complete=True)), flush=True)
        return 0
    except Exception as error:
        print('Catalog correction stopped: '+(str(error) if isinstance(error, PublicationError) else type(error).__name__), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

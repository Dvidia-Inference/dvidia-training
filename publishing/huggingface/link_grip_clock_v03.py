"""Expose verified v0.3 results through two guarded shared-Space root documents.

Stage uses anonymous pinned reads. Publish alone reads the named publishing
credential through the existing memory-only client. One parent-bound commit may
replace index.html and README.md; every other Git/LFS identity is preserved.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import publish_grip_clock_v03 as pub
from publish import PublicationError


SCHEMA = 'dvidia.grip-clock-space-links/v0.3'
PLAN, JOURNAL, RECEIPT = 'link-plan.json', 'link-journal.json', 'link-receipt.json'
ALLOWED = ('README.md', 'index.html')
VIEWER = pub.NAMESPACE+'viewer/index.html'
OLD_SCOPE = (b'<p class="scope">This is recorded simulation from an authored controller with privileged simulator state. '
             b'The browser does not run physics or connect to hardware. Learned skills, shoe manipulation and physical transfer remain future stages.</p>')
NEW_SCOPE = (b'<p class="scope">Each entry states whether it runs live browser physics or replays recorded simulation. '
             b'Studies include authored controllers and synthetically trained models using privileged simulator state. '
             b'Physical robot transfer is not qualified.</p>')
MARKER = b'<!-- dvidia-grip-clock-v03 -->'
VARIANTS = ('clock', 'blind_clock', 'fixed', 'teacher')
LABELS = {'clock': 'learned grip clock', 'blind_clock': 'blind clock',
          'fixed': 'fixed reference timing', 'teacher': 'authored grip-wait teacher'}
IMPLEMENTATION = ('link_grip_clock_v03.py', 'publish_grip_clock_v03.py', 'publish.py', 'prepare.py')


def _implementation():
    return {name: pub.receipt(pub.read(Path(__file__).with_name(name))) for name in IMPLEMENTATION}


def counts(report):
    if (type(report) is not dict or report.get('schema') != 'dvidia.piper-grip-clock-study/v0.3'
            or report.get('physical_robot_ready') is not False or report.get('qualified_robot_skill') is not False
            or type(report.get('variants')) is not dict or set(report['variants']) != set(VARIANTS)
            or type(report.get('rows')) is not list or len(report['rows']) != 512):
        raise PublicationError('Root links require the complete unqualified four-variant confirmation report.')
    result, rosters = {}, {}
    for variant in VARIANTS:
        summary = report['variants'][variant]
        if (type(summary) is not dict or summary.get('scheduled') != 128
                or type(summary.get('strict_successes')) is not int or not 0 <= summary['strict_successes'] <= 128
                or type(summary.get('valid')) is not int or not 0 <= summary['valid'] <= 128):
            raise PublicationError('Every variant must retain its exact 128-attempt denominator.')
        rows = [row for row in report['rows'] if type(row) is dict and row.get('variant') == variant]
        if len(rows) != 128 or any(type(row.get('case_id')) is not str or type(row.get('score')) is not dict for row in rows):
            raise PublicationError('Confirmation rows do not retain every scheduled variant/case.')
        ids = {row['case_id'] for row in rows}
        if len(ids) != 128 or {condition: sum(row.get('condition') == condition for row in rows)
                              for condition in ('nominal', 'post_grip')} != {'nominal': 64, 'post_grip': 64}:
            raise PublicationError('Confirmation must retain paired case identities and both conditions.')
        if any(type(row['score'].get(key)) is not bool for row in rows for key in ('valid', 'success')):
            raise PublicationError('Confirmation scores require explicit validity and strict-success booleans.')
        valid = sum(row['score']['valid'] for row in rows)
        successful = sum(row['score']['valid'] and row['score']['success'] for row in rows)
        if valid != summary['valid'] or successful != summary['strict_successes']:
            raise PublicationError('Displayed counts differ from the complete retained score rows.')
        result[variant] = dict(strict_successes=successful, scheduled=128, valid=valid)
        rosters[variant] = ids
    if any(ids != rosters['clock'] for ids in rosters.values()):
        raise PublicationError('Variant counts must refer to exactly the same paired case roster.')
    return result


def derive(before, report, dataset_revision):
    if not isinstance(dataset_revision, str) or pub.OID.fullmatch(dataset_revision) is None:
        raise PublicationError('Root links require an exact pinned dataset revision.')
    if set(before) != set(ALLOWED) or any(type(raw) is not bytes for raw in before.values()):
        raise PublicationError('Expected exactly the two pinned parent root documents.')
    values = counts(report)
    if (before['index.html'].count(OLD_SCOPE) != 1 or MARKER in before['index.html']
            or MARKER in before['README.md'] or b'id="piper-grip-clock-v03"' in before['index.html']):
        raise PublicationError('The exact old global scope must occur once and the new entry must be absent.')
    summary = '; '.join(f"{LABELS[variant]} {values[variant]['strict_successes']}/128" for variant in VARIANTS)
    dataset_url = f"https://huggingface.co/datasets/{pub.REPOS['dataset']}/tree/{dataset_revision}"
    section = (f'\n{MARKER.decode()}\n<section aria-labelledby="piper-grip-clock-v03">'
        '<h2 id="piper-grip-clock-v03">Learning when to wait for a grip.</h2>'
        '<p>A learned hold/advance clock retimes a frozen PiPER manipulation reference. Simulation only.</p>'
        f'<p>{summary}.</p>'
        '<p>All 128 scheduled attempts per variant remain in the denominator, including invalid attempts and '
        'unactivated interventions. Synthetic demonstrations and privileged simulator sensing; this is recorded '
        'native-physics evidence, with no physical robot qualification.</p>'
        f'<div class="links"><a href="{VIEWER}">Inspect the grip-clock comparison and recorded attempts →</a>'
        f'<a href="{dataset_url}">Pinned complete evidence ↗</a></div></section>\n').encode()
    html = before['index.html'].replace(OLD_SCOPE, NEW_SCOPE+section, 1)
    paragraph = (f'\n\n{MARKER.decode()}\n## Learned PiPER grip clock v0.3\n\n'
        f'{summary}. All 128 scheduled attempts per variant remain in the denominator. '
        'Simulation-only research using synthetic demonstrations and privileged native simulator state; '
        'the new comparison is a recorded replay. Physical robot transfer is not qualified.\n\n'
        f'[Open the recorded grip-clock comparison]({VIEWER}) · '
        f'[Pinned complete dataset]({dataset_url}).\n').encode()
    payload = {'index.html': html, 'README.md': before['README.md']+paragraph}
    for name, raw in payload.items():
        pub.scan(raw, name)
        if len(raw) > pub.MAX_FILE:
            raise PublicationError('Derived root document exceeds its bounded size.')
    return payload, values


def stage(parent, report, dataset_revision, output, *, api=None, downloader=None):
    if not isinstance(parent, str) or pub.OID.fullmatch(parent) is None:
        raise PublicationError('Root-link staging requires the exact final Space parent revision.')
    output = pub.plain(output)
    if output.exists():
        raise PublicationError('Root-link staging output must be fresh.')
    report_raw = pub.read(report, pub.MAX_PLAN)
    pub.scan(report_raw, 'report.json')
    report_value = pub.decoded(report_raw)
    counts(report_value)
    api, downloader = api or pub.public_client(), downloader or pub.download
    head, tree = pub._remote(api, 'space', parent)
    if pub._remote(api, 'space') != (head, tree) or VIEWER not in tree:
        raise PublicationError('The verified v0.3 viewer must exist at the current exact parent.')
    viewer = pub._downloaded(downloader(pub.REPOS['space'], VIEWER, 'space', head))
    if not pub._matches(viewer, tree[VIEWER]):
        raise PublicationError('Pinned v0.3 viewer entry differs from its public Git/LFS object.')
    before = {}
    for name in ALLOWED:
        if name not in tree:
            raise PublicationError('Expected both existing root documents.')
        raw = pub._downloaded(downloader(pub.REPOS['space'], name, 'space', head))
        if not pub._matches(raw, tree[name]):
            raise PublicationError('Anonymous parent document differs from its pinned public tree.')
        before[name] = raw
    payload, values = derive(before, report_value, dataset_revision)
    if pub._remote(api, 'space') != (head, tree):
        raise PublicationError('Space moved during anonymous root-document staging.')
    output.mkdir(parents=True, exist_ok=False)
    for directory, files in [('before', before), ('payload', payload)]:
        (output/directory).mkdir()
        for name, raw in files.items():
            (output/directory/name).write_bytes(raw)
    (output/'report.json').write_bytes(report_raw)
    baseline = dict(repo_id=pub.REPOS['space'], repo_type='space', commit=head, tree=tree)
    pub.atomic(output/'baseline.json', baseline, exclusive=True)
    (output/'implementation').mkdir()
    implementation = _implementation()
    for name in implementation:
        (output/'implementation'/name).write_bytes(pub.read(Path(__file__).with_name(name)))
    plan = dict(schema=SCHEMA, repo_id=pub.REPOS['space'], repo_type='space', parent=head,
        dataset_revision=dataset_revision, report=pub.receipt(report_raw),
        baseline=pub.receipt(pub.read(output/'baseline.json', pub.MAX_PLAN)),
        before={name: pub.receipt(raw) for name, raw in before.items()},
        payload={name: pub.receipt(raw) for name, raw in payload.items()}, counts=values,
        viewer_entry=tree[VIEWER], implementation=implementation, physical_robot_ready=False)
    pub.atomic(output/PLAN, plan, exclusive=True)
    checked_plan(output/PLAN)
    return plan


def checked_plan(path):
    path = pub.plain(path)
    plan = pub.decoded(pub.read(path, pub.MAX_PLAN))
    if (type(plan) is not dict or plan.get('schema') != SCHEMA or plan.get('repo_id') != pub.REPOS['space']
            or plan.get('repo_type') != 'space' or plan.get('physical_robot_ready') is not False
            or not isinstance(plan.get('parent'), str) or pub.OID.fullmatch(plan['parent']) is None
            or plan.get('implementation') != _implementation()):
        raise PublicationError('Root-link plan destination, scope or implementation changed.')
    baseline_raw, report_raw = pub.read(path.parent/'baseline.json', pub.MAX_PLAN), pub.read(path.parent/'report.json', pub.MAX_PLAN)
    if plan.get('baseline') != pub.receipt(baseline_raw) or plan.get('report') != pub.receipt(report_raw):
        raise PublicationError('Root-link baseline or report bytes changed.')
    baseline = pub.decoded(baseline_raw)
    if (baseline.get('repo_id') != pub.REPOS['space'] or baseline.get('repo_type') != 'space'
            or baseline.get('commit') != plan['parent']):
        raise PublicationError('Root-link baseline differs from its exact parent.')
    tree = baseline.get('tree'); pub._tree_valid(tree)
    if VIEWER not in tree or plan.get('viewer_entry') != tree[VIEWER]:
        raise PublicationError('Root-link viewer identity changed.')
    if {item.name for item in (path.parent/'implementation').iterdir()} != set(IMPLEMENTATION):
        raise PublicationError('Root-link implementation snapshot inventory changed.')
    for name, expected in plan['implementation'].items():
        if pub.receipt(pub.read(path.parent/'implementation'/name)) != expected:
            raise PublicationError('Root-link implementation snapshot changed.')
    before, payload = {}, {}
    for directory, collection in [('before', before), ('payload', payload)]:
        if {item.name for item in (path.parent/directory).iterdir()} != set(ALLOWED):
            raise PublicationError('Only the exact two root documents may be staged.')
        for name in ALLOWED:
            raw = pub.read(path.parent/directory/name)
            if pub.receipt(raw) != plan.get(directory, {}).get(name):
                raise PublicationError('Root-link parent or payload bytes changed.')
            collection[name] = raw
    if any(not pub._matches(before[name], tree[name]) for name in ALLOWED):
        raise PublicationError('Root parent bytes differ from their pinned Git/LFS identities.')
    expected, values = derive(before, pub.decoded(report_raw), plan.get('dataset_revision'))
    if payload != expected or plan.get('counts') != values:
        raise PublicationError('Root-link content does not derive exactly from the bound report and dataset pin.')
    return plan, tree, payload


def _message(plan_sha):
    return 'DVIDIA grip clock v0.3 discoverability '+plan_sha


def _finish(api, downloader, plan, tree, payload, revision):
    if not isinstance(revision, str) or pub.OID.fullmatch(revision) is None or revision == plan['parent']:
        raise PublicationError('Root-link reply lacks a new exact commit; retain the intent.')
    head, actual = pub._remote(api, 'space')
    wanted = dict(tree, **{name: pub._entry(raw) for name, raw in payload.items()})
    if head != revision or actual != wanted:
        raise PublicationError('Root-link commit changed unexpected Git/LFS identities.')
    for name, raw in payload.items():
        if pub._downloaded(downloader(pub.REPOS['space'], name, 'space', head)) != raw:
            raise PublicationError('Anonymous root-document readback differs from the reviewed bytes.')
    if pub._remote(api, 'space') != (head, actual):
        raise PublicationError('Space moved during anonymous root-link readback.')
    return head


def publish(path, *, api=None, downloader=None, operation_cls=None):
    path = pub.plain(path)
    with pub._lock(path.parent):
        plan, tree, payload = checked_plan(path)
        plan_raw = pub.read(path, pub.MAX_PLAN); plan_sha = sha256(plan_raw).hexdigest()
        receipt_path, journal_path = path.parent/RECEIPT, path.parent/JOURNAL
        if receipt_path.exists():
            completed = pub.decoded(pub.read(receipt_path, pub.MAX_PLAN))
            if (type(completed) is not dict or completed.get('schema') != SCHEMA+'/receipt'
                    or completed.get('plan') != pub.receipt(plan_raw) or completed.get('anonymous_readback') is not True
                    or completed.get('physical_robot_ready') is not False):
                raise PublicationError('Completed root-link receipt differs from this exact plan.')
            return completed
        message = _message(plan_sha)
        state = dict(schema=SCHEMA+'/intent', plan_sha256=plan_sha, parent=plan['parent'], message=message)
        pending = journal_path.exists()
        if pending and pub.decoded(pub.read(journal_path, pub.MAX_PLAN)) != state:
            raise PublicationError('Root-link intent differs from its exact reviewed request.')
        token = None
        if api is None:
            api, token = pub.client(path.parent/'publisher-cache')
            identity = api.whoami()
            if 'dvidia' not in {str(item.get('name', '')).lower() for item in identity.get('orgs', [])}:
                raise PublicationError('Publishing identity is outside the fixed Dvidia organization.')
        downloader = downloader or pub.download
        if token and any(token.encode() in raw for raw in payload.values()):
            raise PublicationError('A publishing credential appeared in a staged root document.')
        head, current = pub._remote(api, 'space')
        reconciled = False
        if pending and head != plan['parent']:
            before, after = pub._history(api, 'space', plan['parent']), pub._history(api, 'space', head)
            if set(after) != set(before)|{head} or after.get(head) != message:
                raise PublicationError('Unknown root-link reply is not the exact intended immediate child.')
            revision = _finish(api, downloader, plan, tree, payload, head)
            reconciled = True
        else:
            if (head, current) != (plan['parent'], tree) or pub._remote(api, 'space') != (head, current):
                raise PublicationError('Root-link parent moved outside the reviewed full-tree baseline.')
            if _implementation() != plan['implementation'] or pub.read(path, pub.MAX_PLAN) != plan_raw:
                raise PublicationError('Root-link plan or implementation changed before publication.')
            # A late original request and this same-parent resumed body can
            # produce at most one child under the server's atomic parent guard.
            if not pending:
                pub.atomic(journal_path, state, exclusive=True)
            if operation_cls is None:
                from huggingface_hub import CommitOperationAdd
                operation_cls = CommitOperationAdd
            operations = []
            for name in ALLOWED:
                operation = operation_cls(path_in_repo=name, path_or_fileobj=payload[name])
                operation._upload_mode = 'regular'; operation._should_ignore = False
                operations.append(operation)
            try:
                result = api.create_commit(pub.REPOS['space'], repo_type='space', parent_commit=plan['parent'],
                                           operations=operations, commit_message=message)
            except Exception:
                raise PublicationError('Root-link commit reply is uncertain; preserve its exact intent for reconciliation.') from None
            revision = _finish(api, downloader, plan, tree, payload, getattr(result, 'oid', None))
        unchanged = {name: entry for name, entry in tree.items() if name not in ALLOWED}
        completed = dict(schema=SCHEMA+'/receipt', plan=pub.receipt(plan_raw), repo_id=pub.REPOS['space'],
            parent=plan['parent'], commit=revision, dataset_revision=plan['dataset_revision'],
            report=plan['report'], counts=plan['counts'], changed_paths=list(ALLOWED),
            changed_files={name: pub.receipt(raw) for name, raw in payload.items()},
            unchanged_git_lfs_files=len(unchanged), unchanged_git_lfs_sha256=sha256(pub.encoded(unchanged)).hexdigest(),
            viewer_entry=plan['viewer_entry'], viewer_url='https://dvidia-robot-simulation-lab.static.hf.space/'+VIEWER,
            anonymous_readback=True, physical_robot_ready=False, reconciled=reconciled)
        pub.atomic(receipt_path, completed, exclusive=True)
        return completed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('stage')
    for name in ('parent', 'report', 'dataset-revision', 'output'):
        prepare.add_argument('--'+name, required=True)
    for name in ('check', 'publish'):
        commands.add_parser(name).add_argument('plan')
    args = parser.parse_args(argv)
    try:
        if args.command == 'stage':
            result = stage(args.parent, args.report, args.dataset_revision, args.output)
        elif args.command == 'check':
            result = checked_plan(args.plan)[0]
        else:
            result = publish(args.plan)
        print(json.dumps(dict(schema=result['schema'], command=args.command, complete=True)), flush=True)
        return 0
    except Exception as error:
        print('Root-link publication stopped: '+(str(error) if isinstance(error, PublicationError) else type(error).__name__), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

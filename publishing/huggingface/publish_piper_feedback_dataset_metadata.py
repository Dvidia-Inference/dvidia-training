"""Bounded dataset-card overlay after the completed scientific publication.

Stage/check are offline, read-only over the original staged dataset, and never
access credentials. Only publish uses the existing named-Keychain client. It
creates one parent-bound text commit and preserves the original scientific
README through an explicit path map without changing its original seal.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys

import publish_piper_feedback as study
from publish import PublicationError, checked_remote, client
from publish_piper_imitation_batches import _matches, _tree


SCHEMA = 'dvidia.piper-feedback-dataset-metadata/v1'
BASELINE_SCHEMA = 'dvidia.piper-feedback-dataset-baseline/v1'
ROOT_README = 'README.md'
SOURCE_README = 'docs/publication/source-README.md'
OVERLAY = 'docs/publication/metadata-overlay.json'
CARD = 'docs/publication/README.md'
ALLOWED = {ROOT_README, SOURCE_README, OVERLAY}
MAX_BYTES, MAX_FILES = 8*1024*1024, 25
COMMIT = re.compile(r'[0-9a-f]{40}\Z')
SHA = re.compile(r'[0-9a-f]{64}\Z')
PRESERVATION = ('\n## Scientific README preservation\n\n'
    'This Hub card is a publication metadata overlay. The exact original root '
    '[scientific README](docs/publication/source-README.md) is preserved byte for byte. '
    'The [original scientific integrity manifest](integrity.json) remains unchanged; '
    'apply the explicit [README path mapping](docs/publication/metadata-overlay.json) '
    'when verifying the original scientific file inventory.\n')


def _json_file(path):
    return study._json(study._read(path, study.MAX_PLAN))


def _commit(value):
    if not isinstance(value, str) or COMMIT.fullmatch(value) is None:
        raise PublicationError('Dataset parent must be an explicit 40-hex commit.')


def _baseline(raw, parent):
    value = study._json(raw)
    if (not isinstance(value, dict) or set(value) != {'schema', 'repo_id', 'repo_type', 'commit', 'tree'}
            or value['schema'] != BASELINE_SCHEMA or value['repo_id'] != study.DATASET_REPO
            or value['repo_type'] != 'dataset' or value['commit'] != parent
            or not isinstance(value['tree'], dict) or not 1 <= len(value['tree']) <= study.MAX_FILES):
        raise PublicationError('Baseline must bind the exact public dataset commit and complete tree.')
    for name, row in value['tree'].items():
        if name != '.gitattributes':
            study._relative(name)
        if (not isinstance(row, dict) or set(row) != {'blob', 'size', 'lfs_sha256'}
                or not isinstance(row['blob'], str) or COMMIT.fullmatch(row['blob']) is None
                or type(row['size']) is not int or not 0 <= row['size'] <= study.MAX_FILE
                or row['lfs_sha256'] is not None and (not isinstance(row['lfs_sha256'], str)
                                                    or SHA.fullmatch(row['lfs_sha256']) is None)):
            raise PublicationError('Malformed public dataset-tree entry.')
    return value['tree']


def _completed_receipt(raw, parent):
    receipt = study._json(raw)
    if (not isinstance(receipt, dict) or receipt.get('schema') != study.SCHEMA+'/receipt'
            or receipt.get('anonymous_readback') is not True or receipt.get('physical_robot_ready') is not False
            or not isinstance(receipt.get('source_commit'), str) or COMMIT.fullmatch(receipt['source_commit']) is None
            or not isinstance(receipt.get('plan_sha256'), str) or SHA.fullmatch(receipt['plan_sha256']) is None
            or not isinstance(receipt.get('source_integrity_sha256'), str)
            or SHA.fullmatch(receipt['source_integrity_sha256']) is None
            or not isinstance(receipt.get('bindings'), dict)
            or set(receipt['bindings']) != {'protocol_sha256', 'report_sha256', 'frozen_sha256'}
            or any(not isinstance(value, str) or SHA.fullmatch(value) is None for value in receipt['bindings'].values())):
        raise PublicationError('A completed, anonymously verified initial publication receipt is required.')
    rows = receipt.get('repositories')
    if (not isinstance(rows, list) or len(rows) != 3
            or {row.get('repo_type') for row in rows if isinstance(row, dict)} != set(study.REPOSITORIES)):
        raise PublicationError('Initial publication must be complete for all three fixed repositories.')
    for row in rows:
        if (row.get('repo_id') != study.REPOSITORIES[row['repo_type']]
                or not isinstance(row.get('commit'), str) or COMMIT.fullmatch(row['commit']) is None
                or type(row.get('verified_files')) is not int or row['verified_files'] < 1):
            raise PublicationError('Initial publication receipt has an invalid repository binding.')
    dataset = next(row for row in rows if row['repo_type'] == 'dataset')
    if dataset['commit'] != parent:
        raise PublicationError('Reviewed dataset parent differs from its completed initial publication.')
    return receipt, dataset['verified_files']


def _counts(report):
    if (not isinstance(report, dict) or report.get('schema') != study.STUDY_SCHEMA
            or report.get('physical_robot_ready') is not False or not isinstance(report.get('rows'), list)
            or len(report['rows']) != 120 or not isinstance(report.get('limitations'), list)
            or not report['limitations']):
        raise PublicationError('Expected the complete original simulation report.')
    identities = set()
    result = {v: dict(scheduled=0, successes=0, valid=0) for v in study.VARIANTS}
    for row in report['rows']:
        if not isinstance(row, dict) or row.get('variant') not in result or not isinstance(row.get('case_id'), str):
            raise PublicationError('Malformed original report identity.')
        identity = row['case_id'], row['variant']
        score = row.get('score')
        if identity in identities or not isinstance(score, dict) or type(score.get('success')) is not bool or type(score.get('valid')) is not bool:
            raise PublicationError('Original report contains duplicate or malformed outcomes.')
        identities.add(identity)
        total = result[row['variant']]
        total['scheduled'] += 1; total['successes'] += score['success']; total['valid'] += score['valid']
    if any(value['scheduled'] != 24 for value in result.values()) or result != report.get('variants'):
        raise PublicationError('Original report counts differ from its 120 retained outcomes.')
    return result


def _source(dataset, receipt_raw, parent, tree):
    """Verify all staged science against its seal and initial public object IDs."""
    dataset = study._plain(dataset)
    receipt, verified_count = _completed_receipt(receipt_raw, parent)
    raw_seal = study._read(dataset/'integrity.json')
    if sha256(raw_seal).hexdigest() != receipt['source_integrity_sha256']:
        raise PublicationError('Original scientific integrity seal differs from the completed publication.')
    seal = study._json(raw_seal)
    if (not isinstance(seal, dict) or seal.get('schema') != study.INTEGRITY_SCHEMA
            or not isinstance(seal.get('files'), dict) or not seal['files'] or 'integrity.json' in seal['files']):
        raise PublicationError('Invalid original scientific file seal.')
    scientific = dict(seal['files'], **{'integrity.json': study._receipt(raw_seal)})
    inventory = {}
    for path in dataset.rglob('*'):
        study._plain(path)
        if path.is_file():
            name = path.relative_to(dataset).as_posix(); study._relative(name)
            inventory[name] = path
    if (len(inventory) != verified_count or set(tree)-{'.gitattributes'} != set(inventory)
            or set(inventory)-set(scientific) != {CARD}
            or not {'README.md', 'report.json', 'protocol.json', 'frozen.json', 'LICENSE'} <= set(scientific)
            or ALLOWED-{ROOT_README} & set(tree)):
        raise PublicationError('Initial dataset inventory is incomplete or an overlay path already exists.')
    for name, path in inventory.items():
        entry = tree[name]
        expected = scientific.get(name)
        if expected is not None:
            if (not isinstance(expected, dict) or set(expected) != {'bytes', 'sha256'}
                    or type(expected['bytes']) is not int or not 0 <= expected['bytes'] <= study.MAX_FILE
                    or not isinstance(expected['sha256'], str) or SHA.fullmatch(expected['sha256']) is None):
                raise PublicationError('Malformed original scientific file receipt.')
            if path.stat().st_size != expected['bytes'] or entry['size'] != expected['bytes']:
                raise PublicationError('A sealed original file size differs from the public tree.')
        # LFS identity equals the original scientific SHA256. Verify local bytes
        # too, so check never silently trusts a changed staged dataset.
        raw = study._read(path)
        if expected is not None and study._receipt(raw) != expected or not _matches(raw, entry):
            raise PublicationError('An original staged/public scientific object changed.')
    original, card, report_raw = (study._read(dataset/name) for name in (ROOT_README, CARD, 'report.json'))
    report = study._json(report_raw)
    counts = _counts(report)
    if (any(receipt['bindings'][key+'_sha256'] != scientific[key+'.json']['sha256']
            for key in ('report', 'protocol', 'frozen'))
            or receipt['bindings'].get('report_sha256') != sha256(report_raw).hexdigest()
            or scientific['report.json'] != study._receipt(report_raw)):
        raise PublicationError('Scientific report differs from the completed publication binding.')
    if (not card.startswith(b'---\nlicense: mit\ntask_categories:\n- robotics\n---\n\n')
            or card != study._card('dataset', report) or original.startswith(b'---\n')):
        raise PublicationError('Derived dataset card must retain its exact MIT YAML and scientific report counts.')
    if not study._read(dataset/'LICENSE').startswith(b'MIT License'):
        raise PublicationError('Original DVIDIA artifact license is not MIT.')
    return original, card, raw_seal, counts, len(scientific)


def _payload(original, card, seal_raw, parent, scientific_count):
    new_readme = card.rstrip()+b'\n'+PRESERVATION.encode()
    overlay = dict(schema=SCHEMA, initial_dataset_commit=parent,
        original_scientific_integrity=dict(path='integrity.json', **study._receipt(seal_raw)),
        original_scientific_paths=scientific_count,
        source_readme=dict(original_path=ROOT_README, preserved_path=SOURCE_README, **study._receipt(original)),
        derived_readme=dict(path=ROOT_README, **study._receipt(new_readme)),
        restoration_path_mapping={ROOT_README: SOURCE_README},
        scope='Publication metadata only. The original scientific seal is unchanged. Restore its root README through this explicit path mapping; all other original scientific paths and bytes remain unchanged.')
    return {ROOT_README: new_readme, SOURCE_README: original, OVERLAY: study._encoded(overlay)}


def stage(dataset, publication_receipt, dataset_commit, baseline_tree, output):
    _commit(dataset_commit)
    dataset, publication_receipt, baseline_tree, output = map(study._plain, (dataset, publication_receipt, baseline_tree, output))
    if output.exists() or output == dataset or output in dataset.parents or dataset in output.parents:
        raise PublicationError('Metadata staging must be fresh and separate from the immutable staged dataset.')
    receipt_raw, baseline_raw = study._read(publication_receipt), study._read(baseline_tree)
    tree = _baseline(baseline_raw, dataset_commit)
    original, card, seal_raw, counts, scientific_count = _source(dataset, receipt_raw, dataset_commit, tree)
    payload = _payload(original, card, seal_raw, dataset_commit, scientific_count)
    if set(payload) != ALLOWED or len(payload) > MAX_FILES or sum(map(len, payload.values())) > MAX_BYTES:
        raise PublicationError('Dataset metadata exceeds the bounded three-file text commit.')
    for name, raw in payload.items():
        study._scan(raw, json_value=name.endswith('.json'))
    output.mkdir(parents=True, exist_ok=False)
    (output/'initial-publication-receipt.json').write_bytes(receipt_raw)
    (output/'baseline-tree.json').write_bytes(baseline_raw)
    for name, raw in payload.items():
        path = output/'payload'/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    plan = dict(schema=SCHEMA, repo_id=study.DATASET_REPO, repo_type='dataset', parent_commit=dataset_commit,
        staged_dataset_directory=os.path.relpath(dataset, output),
        initial_publication_receipt=study._receipt(receipt_raw), baseline_tree=study._receipt(baseline_raw),
        original_scientific_integrity_sha256=sha256(seal_raw).hexdigest(), scientific_paths=scientific_count,
        original_dataset_files=len(tree)-int('.gitattributes' in tree), original_report_counts=counts,
        files=[dict(path=name, before=study._receipt(original) if name == ROOT_README else None,
                    after=study._receipt(raw)) for name, raw in sorted(payload.items())],
        physical_robot_ready=False,
        scope='Only root README plus two appended preservation/overlay files. No original scientific seal, model, source, trace, native metadata or LFS object is modified.')
    (output/'metadata-plan.json').write_bytes(study._encoded(plan))
    checked_plan(output/'metadata-plan.json')
    return plan


def checked_plan(path):
    path = study._plain(path)
    plan = _json_file(path)
    if (not isinstance(plan, dict) or plan.get('schema') != SCHEMA or plan.get('repo_id') != study.DATASET_REPO
            or plan.get('repo_type') != 'dataset' or plan.get('physical_robot_ready') is not False):
        raise PublicationError('Only the fixed DVIDIA dataset metadata overlay is supported.')
    _commit(plan.get('parent_commit'))
    reference = plan.get('staged_dataset_directory')
    if (not isinstance(reference, str) or Path(reference).is_absolute() or '\\' in reference
            or not 1 <= len(reference) <= 1000):
        raise PublicationError('Metadata check needs its explicitly staged local dataset reference.')
    dataset = study._plain(path.parent/reference)
    receipt_raw = study._read(path.parent/'initial-publication-receipt.json')
    baseline_raw = study._read(path.parent/'baseline-tree.json')
    if (study._receipt(receipt_raw) != plan.get('initial_publication_receipt')
            or study._receipt(baseline_raw) != plan.get('baseline_tree')):
        raise PublicationError('Reviewed initial receipt or full public baseline tree changed.')
    tree = _baseline(baseline_raw, plan['parent_commit'])
    original, card, seal, counts, scientific_count = _source(dataset, receipt_raw, plan['parent_commit'], tree)
    expected = _payload(original, card, seal, plan['parent_commit'], scientific_count)
    if (plan.get('original_scientific_integrity_sha256') != sha256(seal).hexdigest()
            or plan.get('scientific_paths') != scientific_count or plan.get('original_report_counts') != counts
            or plan.get('original_dataset_files') != len(tree)-int('.gitattributes' in tree)):
        raise PublicationError('Original scientific identity or report counts changed in the overlay plan.')
    rows = plan.get('files')
    if not isinstance(rows, list) or len(rows) != 3 or {row.get('path') for row in rows if isinstance(row, dict)} != ALLOWED:
        raise PublicationError('Exactly the three reviewed metadata paths may change.')
    actual = {}
    for item in (path.parent/'payload').rglob('*'):
        study._plain(item)
        if item.is_file(): actual[item.relative_to(path.parent/'payload').as_posix()] = study._read(item)
    if actual != expected or sum(map(len, actual.values())) > MAX_BYTES or len(actual) > MAX_FILES:
        raise PublicationError('Metadata payload differs from the exact reversible card derivation.')
    for row in rows:
        before = study._receipt(original) if row['path'] == ROOT_README else None
        if row.get('before') != before or row.get('after') != study._receipt(actual[row['path']]):
            raise PublicationError('Metadata plan payload receipts differ from their reviewed bytes.')
        study._scan(actual[row['path']], json_value=row['path'].endswith('.json'))
    return plan, tree, actual


def _public_download(repo, name, kind, revision):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, name, repo_type=kind, revision=revision, token=False, endpoint='https://huggingface.co')


def publish(path, *, api=None, downloader=None, operation_cls=None):
    path = study._plain(path)
    plan, baseline, payload = checked_plan(path)
    plan_sha = sha256(study._read(path)).hexdigest()
    pending = path.parent/'metadata-pending.json'
    if pending.exists():
        raise PublicationError('An earlier metadata commit has an uncertain outcome; inspect it before any retry.')
    if (path.parent/'metadata-publication-receipt.json').exists():
        raise PublicationError('Metadata publication is already completed; inspect its receipt instead of retrying.')
    token = None
    if api is None:
        api, token = client(path.parent/'publisher-cache')
    if operation_cls is None:
        from huggingface_hub import CommitOperationAdd
        operation_cls = CommitOperationAdd
    downloader = downloader or _public_download
    if token and any(token.encode() in raw for raw in payload.values()):
        raise PublicationError('Credential appeared in a reviewed metadata payload.')
    info = api.repo_info(study.DATASET_REPO, repo_type='dataset')
    checked_remote(info, 'dataset')
    if info.sha != plan['parent_commit'] or _tree(api, study.DATASET_REPO, 'dataset', info.sha) != baseline:
        raise PublicationError('Dataset head/full tree differs from the reviewed initial publication.')
    dataset = study._plain(path.parent/plan['staged_dataset_directory'])
    before_names = (ROOT_README, CARD, 'integrity.json', 'report.json')
    for name in before_names:
        raw = study._downloaded_bytes(downloader(study.DATASET_REPO, name, 'dataset', info.sha), study.MAX_FILE)
        if raw != study._read(dataset/name) or not _matches(raw, baseline[name]):
            raise PublicationError('Anonymous initial metadata bytes differ from the sealed staged publication.')
    if sha256(study._read(path)).hexdigest() != plan_sha:
        raise PublicationError('Reviewed metadata plan changed before its commit.')
    study._atomic(pending, dict(schema=SCHEMA+'/pending', plan_sha256=plan_sha,
        parent_commit=info.sha, files=[dict(path=name, **study._receipt(raw)) for name, raw in sorted(payload.items())]))
    operations = []
    for name, raw in sorted(payload.items()):
        operation = operation_cls(path_in_repo=name, path_or_fileobj=raw)
        operation._upload_mode, operation._should_ignore = 'regular', False
        operations.append(operation)
    try:
        result = api.create_commit(study.DATASET_REPO, repo_type='dataset', parent_commit=info.sha,
            operations=operations, commit_message='Expose MIT dataset card and preserve original scientific README')
    except Exception:
        raise PublicationError('Metadata commit outcome is uncertain; retained the exact pending request. Inspect before retrying.') from None
    revision = getattr(result, 'oid', None); _commit(revision)
    after = _tree(api, study.DATASET_REPO, 'dataset', revision)
    if (set(after) != set(baseline)|{SOURCE_README, OVERLAY}
            or any(after.get(name) != entry for name, entry in baseline.items() if name != ROOT_README)
            or any(not _matches(raw, after.get(name, {})) or after[name].get('lfs_sha256') is not None for name, raw in payload.items())):
        raise PublicationError('Dataset changed outside the three allowed metadata paths or an original LFS identity changed.')
    for name in (*sorted(payload), 'integrity.json', 'report.json'):
        raw = study._downloaded_bytes(downloader(study.DATASET_REPO, name, 'dataset', revision), study.MAX_FILE)
        if raw != (payload[name] if name in payload else study._read(dataset/name)):
            raise PublicationError('Anonymous metadata readback differs from the exact reviewed bytes.')
    receipt = dict(schema=SCHEMA+'/receipt', plan_sha256=plan_sha,
        repo_id=study.DATASET_REPO, parent_commit=plan['parent_commit'], commit=revision,
        original_scientific_integrity_sha256=plan['original_scientific_integrity_sha256'],
        preservation_path_mapping={ROOT_README: SOURCE_README}, changed_paths=sorted(payload),
        original_dataset_files=plan['original_dataset_files'], unchanged_original_tree_entries=len(baseline)-1,
        unchanged_original_lfs_objects=sum(entry['lfs_sha256'] is not None for entry in baseline.values()),
        original_scientific_paths_restored=plan['scientific_paths'],
        anonymous_before_paths=list(before_names), anonymous_after_paths=[*sorted(payload), 'integrity.json', 'report.json'],
        anonymous_before_and_after_readback=True, scientific_seal_unchanged=True, physical_robot_ready=False)
    study._atomic(path.parent/'metadata-publication-receipt.json', receipt)
    pending.unlink()
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('stage')
    for name in ('dataset', 'publication-receipt', 'dataset-commit', 'baseline-tree', 'output'):
        prep.add_argument('--'+name, required=True)
    for command in ('check', 'publish'):
        sub.add_parser(command).add_argument('plan')
    args = parser.parse_args(argv)
    try:
        if args.command == 'stage':
            value = stage(args.dataset, args.publication_receipt, args.dataset_commit, args.baseline_tree, args.output)
            value = dict(schema=value['schema'], changed_paths=[r['path'] for r in value['files']], network=False, credentials=False)
        elif args.command == 'check':
            value, _, _ = checked_plan(args.plan)
            value = dict(schema=value['schema'], changed_paths=[r['path'] for r in value['files']], network=False, credentials=False)
        else:
            value = publish(args.plan)
        print(json.dumps(value, indent=2))
        return 0
    except Exception as exc:
        print('PiPER dataset metadata stopped: '+(str(exc) if isinstance(exc, PublicationError) else type(exc).__name__), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

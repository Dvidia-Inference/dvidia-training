"""Anonymous, revision-pinned evidence capture; this module cannot upload.

``capture`` consumes the completed initial publication receipt, then records
the dataset/Space trees and Space root files required by the two overlay
stagers. ``verify-model`` checks the public model against its immutable local
publication plan, including its MIT card, contract and every planned file.
Neither command accesses a Keychain credential or modifies a Hub repository.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import logging
import os
from pathlib import Path
import sys

import publish_piper_feedback as study
import publish_piper_feedback_dataset_metadata as metadata
import publish_piper_feedback_hosting as hosting
from publish import PublicationError, checked_remote
from publish_piper_imitation_batches import _matches


SCHEMA = 'dvidia.piper-feedback-public-capture/v1'
MODEL_SCHEMA = 'dvidia.piper-feedback-public-model-check/v1'
MAX_TREE_FILES = 20000


def public_client():
    """Explicit anonymous auth on both API and downloads, even if logged in."""
    os.environ.update(HF_DEBUG='0', HF_HUB_DISABLE_TELEMETRY='1',
        HF_HUB_DISABLE_PROGRESS_BARS='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
        HF_HUB_DISABLE_UPDATE_CHECK='1')
    logging.disable(logging.CRITICAL)
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()

    def download(repo_id, path, repo_type, revision):
        return hf_hub_download(repo_id, path, repo_type=repo_type, revision=revision,
                               token=False, endpoint='https://huggingface.co')
    return HfApi(token=False, endpoint='https://huggingface.co'), download


def _fresh(path):
    path = study._plain(path)
    if path.exists():
        raise PublicationError('Capture output must be a fresh directory.')
    return path


def _info(api, kind, revision):
    if revision is not None and (not isinstance(revision, str) or study.COMMIT.fullmatch(revision) is None):
        raise PublicationError('Public reads require an exact 40-hex revision.')
    info = api.repo_info(study.REPOSITORIES[kind], repo_type=kind, revision=revision)
    checked_remote(info, kind)
    if study.COMMIT.fullmatch(info.sha) is None or revision is not None and info.sha != revision:
        raise PublicationError('Public repository resolved to a different revision.')
    return info


def _tree(api, kind, revision):
    tree = {}
    for entry in api.list_repo_tree(study.REPOSITORIES[kind], repo_type=kind,
                                    revision=revision, recursive=True):
        if not hasattr(entry, 'blob_id'):
            continue
        name = entry.path
        if name in tree or len(tree) >= MAX_TREE_FILES:
            raise PublicationError('Duplicate or oversized public tree.')
        lfs = getattr(entry, 'lfs', None)
        tree[name] = dict(blob=entry.blob_id, size=entry.size,
            lfs_sha256=(lfs.get('sha256') if isinstance(lfs, dict)
                        else getattr(lfs, 'sha256', None)) if lfs else None)
    # The hosting validator accepts ordinary existing Space asset types, while
    # still rejecting unsafe paths and malformed Git/LFS object identities.
    hosting._tree_document(dict(schema=hosting.BASELINE_SCHEMA, repo_id=study.SPACE_REPO,
        repo_type='space', commit=revision, tree=tree), revision)
    return tree


def _read_public(download, kind, revision, name, tree):
    if name not in tree or tree[name]['size'] > study.MAX_FILE:
        raise PublicationError('Missing or oversized public readback file.')
    raw = study._downloaded_bytes(download(study.REPOSITORIES[kind], name, kind, revision), study.MAX_FILE)
    if not _matches(raw, tree[name]):
        raise PublicationError('Anonymous readback differs from its pinned public tree.')
    return raw


def _save(output, payload, receipt_name, receipt):
    # All remote reads and checks finish before a completed local snapshot is
    # created. Exclusive writes prevent accidental replacement or reuse.
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in payload.items():
        target = output/name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(raw)
    receipt['files'] = {name: study._receipt(raw) for name, raw in sorted(payload.items())}
    raw = study._encoded(receipt)
    study._scan(raw, json_value=True)
    with (output/receipt_name).open('xb') as stream:
        stream.write(raw)
    return receipt


def capture(publication_receipt, output, *, api=None, download=None):
    output = _fresh(output)
    receipt_raw = study._read(publication_receipt)
    preliminary = study._json(receipt_raw)
    rows = preliminary.get('repositories') if isinstance(preliminary, dict) else None
    parents = [r.get('commit') for r in rows if isinstance(r, dict) and r.get('repo_type') == 'dataset'] if isinstance(rows, list) else []
    if len(parents) != 1:
        raise PublicationError('Capture requires the completed initial publication receipt.')
    initial, _ = metadata._completed_receipt(receipt_raw, parents[0])
    if api is None and download is None:
        api, download = public_client()
    if api is None or download is None:
        raise PublicationError('Provide both anonymous API and downloader boundaries.')
    payload, summaries = {}, {}
    for kind, schema in [('dataset', metadata.BASELINE_SCHEMA), ('space', hosting.BASELINE_SCHEMA)]:
        row = next(r for r in initial['repositories'] if r['repo_type'] == kind)
        revision = row['commit']
        _info(api, kind, revision)
        # A stale snapshot is not a safe parent for either subsequent overlay.
        if _info(api, kind, None).sha != revision:
            raise PublicationError('Public head moved after the initial publication; review a fresh baseline.')
        tree = _tree(api, kind, revision)
        if kind == 'dataset' and len(tree)-int('.gitattributes' in tree) != row['verified_files']:
            raise PublicationError('Dataset tree does not contain the completed initial publication inventory.')
        document = dict(schema=schema, repo_id=row['repo_id'], repo_type=kind, commit=revision, tree=tree)
        if kind == 'dataset':
            metadata._baseline(study._encoded(document), revision)
        payload[kind+'/tree.json'] = study._encoded(document)
        names = ['README.md', 'index.html'] if kind == 'space' else ['README.md']
        for name in names:
            payload[kind+'/'+name] = _read_public(download, kind, revision, name, tree)
        if _info(api, kind, None).sha != revision:
            raise PublicationError('Public head changed during baseline capture.')
        summaries[kind] = dict(repo_id=row['repo_id'], commit=revision, tree_files=len(tree),
            lfs_objects=sum(value['lfs_sha256'] is not None for value in tree.values()))
    return _save(output, payload, 'capture-receipt.json', dict(schema=SCHEMA,
        initial_publication_receipt=study._receipt(receipt_raw), repositories=summaries,
        anonymous_readback=True, credentials_accessed=False, remote_mutations=False,
        scope='Pinned public baseline trees and root text only; no overlay or upload performed.'))


def verify_model(publication_plan, output, *, api=None, download=None):
    output = _fresh(output)
    plan_raw = study._read(publication_plan, study.MAX_PLAN)
    plan = study._json(plan_raw)
    rows = plan.get('repositories') if isinstance(plan, dict) else None
    model_rows = [r for r in rows if isinstance(r, dict) and r.get('repo_type') == 'model'] if isinstance(rows, list) else []
    if (not isinstance(plan, dict) or plan.get('schema') != study.SCHEMA or plan.get('private') is not False or len(model_rows) != 1
            or model_rows[0].get('repo_id') != study.MODEL_REPO
            or not isinstance(model_rows[0].get('files'), list)):
        raise PublicationError('Expected the immutable initial publication plan and fixed public model.')
    planned = {}
    for row in model_rows[0]['files']:
        if (not isinstance(row, dict) or set(row) != {'path', 'bytes', 'sha256'}
                or type(row['bytes']) is not int or not 0 <= row['bytes'] <= study.MAX_FILE
                or not isinstance(row['sha256'], str) or study.SHA.fullmatch(row['sha256']) is None):
            raise PublicationError('Malformed planned model-file receipt.')
        study._relative(row['path'])
        if row['path'] in planned:
            raise PublicationError('Duplicate planned model file.')
        planned[row['path']] = dict(bytes=row['bytes'], sha256=row['sha256'])
    required = {'README.md', 'LICENSE', 'contracts/skill.json', 'report.json', 'frozen.json', 'audit.json', 'base-model.json'}
    if not required <= set(planned) or len(planned) > 1000 or sum(r['bytes'] for r in planned.values()) > study.MAX_MODEL:
        raise PublicationError('Incomplete or oversized planned model inventory.')
    if api is None and download is None:
        api, download = public_client()
    if api is None or download is None:
        raise PublicationError('Provide both anonymous API and downloader boundaries.')
    revision = _info(api, 'model', None).sha
    _info(api, 'model', revision)
    tree = _tree(api, 'model', revision)
    if set(tree)-{'.gitattributes'} != set(planned):
        raise PublicationError('Public model inventory differs from the immutable publication plan.')
    files = {}
    for name, expected in planned.items():
        raw = _read_public(download, 'model', revision, name, tree)
        if study._receipt(raw) != expected:
            raise PublicationError('Public model bytes differ from the immutable publication plan.')
        files[name] = raw
    report = study._json(files['report.json'])
    counts = metadata._counts(report)
    contract = study._json(files['contracts/skill.json'])
    if (files['README.md'] != study._card('model', report) or not files['LICENSE'].startswith(b'MIT License')
            or contract.get('schema') != 'dvidia.piper-learned-grip-skill/v0.2'
            or contract.get('publisher') != 'DVIDIA' or contract.get('physical_robot_ready') is not False
            or contract.get('teacher_at_student_inference') is not False or contract.get('confirmation') != counts):
        raise PublicationError('Public MIT card, research contract or confirmation counts are inconsistent.')
    models = contract.get('models')
    if not isinstance(models, dict) or set(models) != {'base', 'feedback', 'blind'}:
        raise PublicationError('Expected three explicitly hash-bound contract models.')
    for item in list(models.values())+[contract.get('wheel')]:
        if not isinstance(item, dict) or item.get('path') not in files or item.get('sha256') != planned[item['path']]['sha256']:
            raise PublicationError('Public contract artifact binding differs from the downloaded model.')
    for name in ('frozen', 'audit'):
        if contract.get(name+'_sha256') != planned[name+'.json']['sha256']:
            raise PublicationError('Public contract evidence binding is inconsistent.')
    for name in ('protocol', 'report', 'frozen'):
        if plan.get('bindings', {}).get(name+'_sha256') != planned.get(name+'.json', {}).get('sha256'):
            raise PublicationError('Public scientific evidence differs from the initial plan bindings.')
    return _save(output, {}, 'model-check.json', dict(schema=MODEL_SCHEMA, repo_id=study.MODEL_REPO,
        repo_type='model', commit=revision, publication_plan=study._receipt(plan_raw),
        verified_files=len(files), verified_bytes=sum(map(len, files.values())), verified_file_hashes=planned,
        tree=study._receipt(study._encoded(tree)), license='mit', card_matches_initial_plan=True,
        contract_models=models, confirmation=counts, anonymous_readback=True,
        credentials_accessed=False, remote_mutations=False, physical_robot_ready=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    baseline = commands.add_parser('capture')
    baseline.add_argument('--publication-receipt', required=True, type=Path)
    baseline.add_argument('--output', required=True, type=Path)
    model = commands.add_parser('verify-model')
    model.add_argument('--publication-plan', required=True, type=Path)
    model.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'capture':
            result = capture(args.publication_receipt, args.output)
        else:
            result = verify_model(args.publication_plan, args.output)
        print(study._encoded({key: value for key, value in result.items() if key not in {'files', 'verified_file_hashes'}}).decode(), end='')
        return 0
    except Exception as exc:
        # SDK errors may embed URLs/headers. Never echo them or credential data.
        print('Public capture failed ('+type(exc).__name__+'); no remote mutation was attempted.', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

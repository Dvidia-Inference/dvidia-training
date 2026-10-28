"""Build offline, hash-bound model and dataset payloads for PiPER grip-clock v0.3.

This copies an already frozen and independently audited study. It performs no
training, simulation, network access or publication.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

from dvidia_training import piper_grip_clock_cli as lifecycle
from dvidia_training.reference_replay import MIT_LICENSE


SCHEMA = 'dvidia.piper-grip-clock-payload/v0.3'
AUDIT_SCHEMA = 'dvidia.piper-grip-clock-independent-audit/v1'
STUDY_SCHEMA = 'dvidia.piper-grip-clock-study/v0.3'
AUDIT_MODULE = Path(__file__).with_name('audit_piper_grip_clock_v03.py')
CANDIDATES = ('full4', 'full6', 'blind4', 'blind6')
VARIANTS = ('clock', 'blind_clock', 'fixed', 'teacher')
MAX_FILE = 8 * 1024 * 1024
MAX_MODEL = 64 * 1024 * 1024
MAX_DATASET = 8 * 1024**3
MAX_FILES = 20_000


def _pairs(rows):
    result = {}
    for key, value in rows:
        if key in result:
            raise ValueError('JSON contains a duplicate key.')
        result[key] = value
    return result


def _read_json(path):
    path = _safe_file(path)
    if path.stat().st_size > MAX_FILE:
        raise ValueError('JSON artifact exceeds the 8 MiB publisher file bound: ' + path.name)
    raw = path.read_bytes()
    return json.loads(raw, object_pairs_hook=_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON.')))


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def _unsafe_symlink(item):
    return item.is_symlink() and item not in {Path('/var'), Path('/tmp')}


def _safe_file(path):
    path = Path(path).absolute()
    if any(_unsafe_symlink(item) for item in (path, *path.parents)):
        raise ValueError('Artifact inputs cannot traverse symlinks.')
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError('Missing artifact input: ' + str(path))
    return resolved


def _safe_dir(path):
    path = Path(path).absolute()
    if any(_unsafe_symlink(item) for item in (path, *path.parents)):
        raise ValueError('Artifact directories cannot traverse symlinks.')
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError('Expected an ordinary source directory.')
    return resolved


def _hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _copy_file(source, destination):
    source = _safe_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if _hash(source) != _hash(destination):
        raise ValueError('Copied artifact bytes differ: ' + source.name)


def _plain_tree(root, *, exclude_lock=False):
    root = _safe_dir(root)
    entries = []
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Study evidence cannot contain symlinks: ' + str(path))
        relative = path.relative_to(root).as_posix()
        if exclude_lock and relative == '.lifecycle.lock':
            continue
        if path.is_file():
            if not relative or PurePosixPath(relative).is_absolute() or '..' in PurePosixPath(relative).parts:
                raise ValueError('Unsafe study-relative path.')
            entries.append((path, relative))
    if any(path.stat().st_size > MAX_FILE for path, _ in entries):
        raise ValueError('Study contains a file above the publisher 8 MiB limit.')
    return root, entries


def _verify_inputs(study, audit_path):
    study = _safe_dir(study)
    audit_path = _safe_file(audit_path)
    if audit_path.is_relative_to(study):
        raise ValueError('The independent audit must be saved outside the study.')
    protocol, frozen = lifecycle._frozen(study)
    protocol_path, frozen_path, report_path = (study / name for name in ('protocol.json', 'frozen.json', 'report.json'))
    report = _read_json(report_path)
    audit = _read_json(audit_path)
    expected_cases = lifecycle.make_cases()
    if (protocol.get('schema') != STUDY_SCHEMA
            or protocol.get('cases') != expected_cases
            or protocol.get('variants') != list(VARIANTS)
            or len(protocol.get('cases', {}).get('confirmation', [])) != 128
            or protocol.get('source_sha256') != frozen.get('lifecycle_source_sha256')
            or protocol.get('physical_robot_ready') is not False
            or frozen.get('physical_robot_ready') is not False):
        raise ValueError('Protocol roster, variants or frozen source inventory differs from v0.3.')
    protocol_hash, frozen_hash = _hash(protocol_path), _hash(frozen_path)
    expected_rows = {(case['id'], variant) for case in protocol['cases']['confirmation'] for variant in VARIANTS}
    report_rows = report.get('rows')
    report_roster = {(row.get('case_id'), row.get('variant')) for row in report_rows or []}
    if (report.get('schema') != STUDY_SCHEMA or report.get('protocol_sha256') != protocol_hash
            or report.get('frozen_sha256') != frozen_hash or len(report_rows or []) != 512
            or report_roster != expected_rows or report.get('physical_robot_ready') is not False
            or report.get('qualified_robot_skill') is not False):
        raise ValueError('Saved report is not the complete, bound 512-attempt confirmation report.')
    for variant in VARIANTS:
        summary = report.get('variants', {}).get(variant, {})
        if (summary.get('scheduled') != 128 or type(summary.get('valid')) is not int
                or not 0 <= summary['valid'] <= 128 or type(summary.get('strict_successes')) is not int
                or not 0 <= summary['strict_successes'] <= 128):
            raise ValueError('Confirmation report has incomplete attempts for ' + variant + '.')
    if (audit.get('schema') != AUDIT_SCHEMA or audit.get('split') != 'all'
            or audit.get('protocol_sha256') != protocol_hash or audit.get('frozen_sha256') != frozen_hash
            or audit.get('source_sha256') != protocol.get('source_sha256')
            or audit.get('audited_attempts') != 768 or audit.get('audit_mismatches') != 0
            or len(audit.get('rows', [])) != 768
            or _hash(AUDIT_MODULE) != audit.get('audit_implementation_sha256')):
        raise ValueError('Independent audit must cover all 768 attempts with zero mismatches and the current auditor.')
    expected_audit_counts = {
        ('train', 'teacher'): 64,
        ('development', 'fixed'): 32, ('development', 'teacher'): 32,
        ('development', 'full4'): 32, ('development', 'full6'): 32,
        ('development', 'blind4'): 32, ('development', 'blind6'): 32,
        **{('confirmation', variant): 128 for variant in VARIANTS},
    }
    expected_audit_roster = {
        ('train', case['id'], 'teacher') for case in protocol['cases']['train']
    }
    expected_audit_roster |= {
        ('development', case['id'], candidate)
        for case in protocol['cases']['development']
        for candidate in ('fixed', 'teacher', *CANDIDATES)
    }
    expected_audit_roster |= {
        ('confirmation', case['id'], variant)
        for case in protocol['cases']['confirmation'] for variant in VARIANTS
    }
    observed, observed_roster = {}, []
    for row in audit['rows']:
        key = (row.get('partition'), row.get('candidate'))
        observed[key] = observed.get(key, 0) + 1
        observed_roster.append((row.get('partition'), row.get('case_id'), row.get('candidate')))
        if row.get('audit_mismatches') != 0:
            raise ValueError('An audited attempt has a mismatch.')
    if (observed != expected_audit_counts or len(set(observed_roster)) != 768
            or set(observed_roster) != expected_audit_roster):
        raise ValueError('Independent audit partition/candidate roster differs from the frozen 768-attempt plan.')
    audit_confirmation = {(row['case_id'], row['variant']): row for row in audit['rows']
                          if row['partition'] == 'confirmation'}
    report_by_variant = {variant: {'valid': 0, 'strict_successes': 0} for variant in VARIANTS}
    for row in report_rows:
        score = row.get('score')
        reconstructed = audit_confirmation.get((row['case_id'], row['variant']))
        if (type(score) is not dict or reconstructed is None
                or score.get('valid') is not reconstructed.get('valid')
                or score.get('success') is not reconstructed.get('strict_success')):
            raise ValueError('Saved confirmation scores differ from the independent numeric audit.')
        summary = report_by_variant[row['variant']]
        summary['valid'] += score['valid'] is True
        summary['strict_successes'] += score['valid'] is True and score['success'] is True
    for variant, observed_summary in report_by_variant.items():
        saved = report['variants'][variant]
        if (observed_summary['valid'] != saved['valid']
                or observed_summary['strict_successes'] != saved['strict_successes']):
            raise ValueError('Saved confirmation summaries differ from the audited per-attempt scores.')
    model_names = set(lifecycle.CANDIDATES)
    if model_names != set(CANDIDATES):
        raise ValueError('Lifecycle candidate recipe roster changed.')
    for name in CANDIDATES:
        candidate = study / 'candidates' / name
        receipt = _read_json(candidate / 'receipt.json')
        model = _read_json(candidate / 'model.json')
        teacher_manifest_hash = _hash(study / 'teacher-manifest.json')
        if (receipt.get('name') != name or receipt.get('model_sha256') != _hash(candidate / 'model.json')
                or receipt.get('source_sha256') != protocol['source_sha256']
                or receipt.get('protocol_sha256') != protocol_hash
                or receipt.get('teacher_manifest_sha256') != teacher_manifest_hash
                or model.get('training') != receipt.get('training')):
            raise ValueError('Candidate fit or receipt is not bound: ' + name)
    if (_hash(study / 'selected.json') != frozen.get('selected_sha256')
            or _hash(study / 'blind.json') != frozen.get('blind_sha256')):
        raise ValueError('Selected full/blind model bytes differ from the freeze.')
    return study, audit_path, protocol, frozen, report, audit


def _frontmatter(*, dataset=False):
    if dataset:
        return ('---\nlicense: mit\ntags:\n- robotics\n- simulation\n- synthetic-data\n- piper\n'
                '---\n\n')
    return ('---\nlicense: mit\nlibrary_name: dvidia-training\npipeline_tag: robotics\n'
            'tags:\n- robotics\n- simulation-only\n- piper\n---\n\n')


def _result_rows(report):
    result = []
    for variant in VARIANTS:
        row = report['variants'][variant]
        result.append(f"| {variant} | {row['strict_successes']}/{row['scheduled']} | {row['valid']}/{row['scheduled']} |")
    return '\n'.join(result)


def _model_card(frozen, report):
    selected, blind = frozen['selected'], frozen['blind']
    return _frontmatter() + f"""# PiPER grip clock v0.3

A compact learned progress clock pauses a frozen PiPER movement reference while acquiring a grasp on one 40 g rigid block. The clock uses privileged simulator state and the original arm and jaw reference. Waiting delays both arm progress and jaw release. It does not learn grasp geometry.

The matched blind clock uses the same teacher rows, fit recipe, and tree depth as the selected full clock, with live grip and history inputs masked. The selected pair is `{selected}` and `{blind}` at matched depth. All four candidate fits, receipts, protocol, freeze, report, independent audit, runtime source snapshots, and integrity hashes are included.

## Frozen confirmation results

Strict successes and valid attempts are reported for every scheduled controller. The denominator includes all 128 cases per controller.

| Controller | Strict successes | Valid attempts |
|---|---:|---:|
{_result_rows(report)}

The independent numeric audit covered {report['variants']['clock']['scheduled'] * 4 + 256} attempts (64 training, 192 development, 512 confirmation) with zero mismatches. The confirmation roster contains 32 fresh XY groups, each paired across two known destinations and nominal/post-grip conditions.

## Scope

The acquisition teacher checks bilateral pad contact and force plus tool-relative drift over six sampled observations spanning 100 ms. Its latch releases after acquisition. This is acquisition-only supervision, not continuous learned slip recovery or physical grasp certification. The experiment uses one synthetic rigid 40 g block, two known destinations, privileged simulator sensing, and a narrow PiPER starting envelope. It does not establish arbitrary-object, camera-based, other-arm, or hardware transfer.

Shared-host timing describes four spawn-isolated simulator workers on an Apple M5 host. It excludes cameras, transport, rendering, and physical actuation; it is not an isolated throughput result or a hard-real-time guarantee. `physical_robot_ready` and `qualified_robot_skill` remain false.

## Files and reproduction

This model artifact is published as `Dvidia/piper-grip-clock-v03`. The complete synthetic demonstration and attempt lineage is in the [DVIDIA dataset repository](https://huggingface.co/datasets/Dvidia/piper-grip-clock-demonstrations-v03). The reproduction guide and host record are in `docs/`. The static viewer entry point is [learned-placement/v03/viewer/index.html](https://huggingface.co/spaces/Dvidia/robot-simulation-lab/tree/main/learned-placement/v03/viewer/index.html); the paper record is [research.dvidia.org/papers/learned-piper-grip-clock-v03/](https://research.dvidia.org/papers/learned-piper-grip-clock-v03/).

Code: [Dvidia-Inference/dvidia-training](https://github.com/Dvidia-Inference/dvidia-training), release tag `piper-grip-clock-v0.3`. Contact: hello@dvidia.org · X: [@Xiammrriver](https://x.com/Xiammrriver).

DVIDIA-authored code, models, and synthetic numeric evidence are MIT licensed. Upstream PiPER geometry and renderer notices remain with their assets and replay; no robot assets or wheel are bundled here.
"""


def _dataset_card(report):
    return _frontmatter(dataset=True) + f"""# PiPER grip clock v0.3 synthetic study dataset

Published as `Dvidia/piper-grip-clock-demonstrations-v03`, this dataset preserves the finite lineage for the simulation-only PiPER grip-clock v0.3 experiment: 64 synthetic acquisition-teacher demonstrations, four finite candidate fits, 192 development attempts, and 512 fresh confirmation attempts. All 768 scheduled attempts were independently reconstructed with zero numeric audit mismatches.

The confirmation comparison uses 32 fresh XY groups. Each group contains two known destinations and nominal/post-grip conditions, paired across the full clock, matched blind clock, fixed reference, and acquisition teacher. The saved report retains every scheduled attempt in each denominator.

| Confirmation controller | Strict successes | Valid attempts |
|---|---:|---:|
{_result_rows(report)}

## Data scope

Episodes are authored simulator outputs for one synthetic 40 g rigid block, two known destinations, and a narrow PiPER start envelope. Training labels come from a causal sampled acquisition teacher using privileged simulator state. The teacher checks bilateral contact/force and tool-relative drift across 100 ms, then releases a latch; this dataset does not demonstrate continuous learned recovery. It contains no human videos, camera observations, physical robot trials, arbitrary-object transfer, or hardware qualification.

The `study/` directory is a byte-preserving copy of the frozen study, excluding only its local lifecycle lock. `audit.json` contains the independent all-partition numeric audit. `docs/` includes the reproduction guide, measured shared-host record, and development lineage. `integrity.json` binds every other file in this payload by size and SHA-256.

DVIDIA-authored code and synthetic numeric evidence are MIT licensed. No robot asset bundle is included. Upstream asset notices remain applicable when assets are fetched separately. Questions: hello@dvidia.org · X: [@Xiammrriver](https://x.com/Xiammrriver).
"""


def _write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)


def _copy_study(study, destination):
    study, entries = _plain_tree(study, exclude_lock=True)
    for source, relative in entries:
        _copy_file(source, destination / relative)


def _payload_integrity(root, payload_name):
    files = {}
    total = 0
    paths = sorted(path for path in root.rglob('*') if path.is_file())
    for path in paths:
        if path.is_symlink():
            raise ValueError('Payload cannot contain symlinks.')
        relative = path.relative_to(root).as_posix()
        if relative == 'integrity.json':
            continue
        size = path.stat().st_size
        if size > MAX_FILE:
            raise ValueError(f'Payload file exceeds the 8 MiB publisher limit: {relative}')
        files[relative] = {'bytes': size, 'sha256': _hash(path)}
        total += size
    limit = MAX_MODEL if payload_name == 'model' else MAX_DATASET
    manifest_raw = _json_bytes({'schema': SCHEMA, 'payload': payload_name, 'files': files})
    if (len(files) + 1 > MAX_FILES or total + len(manifest_raw) > limit
            or len(manifest_raw) > MAX_FILE):
        raise ValueError(f'{payload_name} payload exceeds its file-count or total-size bound.')
    _write(root/'integrity.json', manifest_raw)
    return len(files) + 1, total + len(manifest_raw)


def _prepare_locked(study, audit, output, *, guide, host, lineage=None):
    study, audit_path, protocol, frozen, report, audit_result = _verify_inputs(study, audit)
    guide, host = _safe_file(guide), _safe_file(host)
    lineage = _safe_file(lineage or guide.parent / 'piper-grip-clock-v03' / 'development-lineage.md')
    output = Path(output).absolute()
    if any(_unsafe_symlink(item) for item in (output, *output.parents)):
        raise ValueError('Output path cannot traverse symlinks.')
    if output.exists() or output.is_symlink():
        raise ValueError('Output must be a fresh payload directory.')
    parent = output.parent.resolve(strict=True)
    if study == output or output in study.parents or study in output.parents or audit_path == output or audit_path in output.parents:
        raise ValueError('Output must be separate from all inputs.')

    staging = Path(tempfile.mkdtemp(prefix='.piper-clock-payload-', dir=parent))
    try:
        model_root, dataset_root = staging/'model', staging/'dataset'
        model_root.mkdir(); dataset_root.mkdir()
        for name in ('base-model.json', 'jaw-model.json', 'selected.json', 'blind.json',
                     'protocol.json', 'prepared.json', 'frozen.json', 'report.json', 'teacher-manifest.json'):
            _copy_file(study/name, model_root/name)
        for name in CANDIDATES:
            source = study/'candidates'/name
            target = model_root/'candidates'/name
            for file_name in ('model.json', 'receipt.json', 'development.json'):
                _copy_file(source/file_name, target/file_name)
        for name in ('fixed.json', 'teacher.json'):
            _copy_file(study/'development'/name, model_root/'development'/name)
        for name, expected in sorted(protocol['source_sha256'].items()):
            if Path(name).name != name or _hash(study/'sources'/name) != expected:
                raise ValueError('Frozen runtime source snapshot hash differs: ' + name)
            _copy_file(study/'sources'/name, model_root/'runtime'/name)
        _copy_file(audit_path, model_root/'audit.json')
        _copy_file(guide, model_root/'docs'/'guide.md')
        _copy_file(host, model_root/'docs'/'host.json')
        _write(model_root/'README.md', _model_card(frozen, report).encode())
        _write(model_root/'LICENSE', MIT_LICENSE.encode())

        _copy_study(study, dataset_root/'study')
        _copy_file(audit_path, dataset_root/'audit.json')
        _copy_file(guide, dataset_root/'docs'/'guide.md')
        _copy_file(host, dataset_root/'docs'/'host.json')
        _copy_file(lineage, dataset_root/'docs'/'development-lineage.md')
        _write(dataset_root/'README.md', _dataset_card(report).encode())
        _write(dataset_root/'LICENSE', MIT_LICENSE.encode())

        model_count, model_bytes = _payload_integrity(model_root, 'model')
        data_count, data_bytes = _payload_integrity(dataset_root, 'dataset')
        os.rename(staging, output)
        return dict(schema=SCHEMA, output=str(output), model_files=model_count, model_bytes=model_bytes,
                    dataset_files=data_count, dataset_bytes=data_bytes, protocol_sha256=_hash(output/'model'/'protocol.json'),
                    frozen_sha256=_hash(output/'model'/'frozen.json'), audit_sha256=_hash(output/'model'/'audit.json'),
                    audited_attempts=audit_result['audited_attempts'], audit_mismatches=audit_result['audit_mismatches'],
                    selected=frozen['selected'], blind=frozen['blind'], physical_robot_ready=False)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def prepare(study, audit, output, *, guide, host, lineage=None):
    """Validate and copy complete frozen evidence to a fresh offline payload root."""
    study = _safe_dir(study)
    with lifecycle._locked(study):
        return _prepare_locked(study, audit, output, guide=guide, host=host, lineage=lineage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True, help='Saved independent audit with split=all.')
    parser.add_argument('--output', type=Path, required=True, help='Fresh payload root containing model/ and dataset/.')
    parser.add_argument('--guide', type=Path, default=Path('docs/piper-grip-clock-v03.md'))
    parser.add_argument('--host', type=Path, default=Path('docs/piper-grip-clock-v03/host.json'))
    parser.add_argument('--lineage', type=Path, default=Path('docs/piper-grip-clock-v03/development-lineage.md'))
    args = parser.parse_args(argv)
    result = prepare(args.study, args.audit, args.output, guide=args.guide, host=args.host, lineage=args.lineage)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

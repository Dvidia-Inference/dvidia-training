"""Prepare and publish the first learned PiPER placement evidence release.

Preparation is local and hash-bound. Publishing is an explicit separate command
that appends to three fixed Hugging Face destinations and never edits Space root
pointers. The source run must be complete and carry its own integrity manifest.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import gzip
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import zipfile

from publish import PublicationError, checked_remote, client, verified_downloads
from publish_prerequisites import matches_tree, tree

MODEL_REPO = 'Dvidia/piper-block-placement-v01'
DATASET_REPO = 'Dvidia/piper-block-placement-demonstrations-v01'
SPACE_REPO = 'Dvidia/robot-simulation-lab'
SPACE_NAMESPACE = 'learned-placement/v01'
SPACE_VIEWER_URL = 'https://dvidia-robot-simulation-lab.static.hf.space/' + SPACE_NAMESPACE + '/viewer/'
SPACE_LANDING_PAGE = (b'<!doctype html><html lang="en"><meta charset="utf-8">'
                      b'<meta http-equiv="refresh" content="0; url=viewer/">'
                      b'<title>PiPER learned placement replay</title>'
                      b'<p>Opening the recorded replay. <a href="viewer/">Open the viewer</a>.</p></html>')
KIND = 'dvidia.piper-imitation-publication'
SCHEMA_VERSION = 1
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 768 * 1024 * 1024
MAX_DECODED_TOTAL_BYTES = 768 * 1024 * 1024
MAX_FILES = 5000
MAX_DECODED_BYTES = 64 * 1024 * 1024
SHA = re.compile(r'[a-f0-9]{64}\Z')
TOKEN = re.compile(rb'hf_[A-Za-z0-9]{30,}')
PRIVATE_TEXT = re.compile(rb'(?:/Users/[^/\s]+/|/private/tmp/|file://|Bearer\s+[A-Za-z0-9_-]{20,})')
PRIVATE_KEYS = {'authorization', 'api_key', 'apikey', 'credential', 'credentials',
                'token', 'headers', 'raw_response', 'response'}
ALLOWED_SUFFIXES = {'.bin', '.css', '.gz', '.html', '.js', '.json', '.md', '.npz', '.png',
                    '.py', '.svg', '.txt', '.wasm', '.webp'}
REPO_ROWS = (
    ('model', MODEL_REPO, 'model'),
    ('dataset', DATASET_REPO, 'dataset'),
    ('space', SPACE_REPO, 'space'),
)
RUN_ROOT_FILES = {'protocol.json', 'report.json', 'integrity.json', 'frozen.json',
                  'confirmation-opened.json', 'audit.json', 'driver-recovery.json'}
RUN_DIRECTORIES = {'training', 'model', 'demonstrations', 'attempts', 'attempt-starts', 'attempt-scores',
                   'development_models', 'sources', 'viewer', 'driver-recovery'}


def _plain(path: Path) -> Path:
    path = Path(path).absolute()
    # Canonicalize platform aliases such as macOS /var -> /private/var, then
    # inspect each actual artifact as it is traversed for an in-tree symlink.
    if path.is_symlink():
        raise PublicationError('Publication paths may not contain symlinks.')
    return path.resolve()


def _read(path: Path, limit: int = MAX_FILE_BYTES) -> bytes:
    path = _plain(path)
    if not path.is_file() or path.stat().st_size > limit:
        raise PublicationError('Missing or oversized plain publication file.')
    raw = path.read_bytes()
    if len(raw) > limit:
        raise PublicationError('Publication file grew beyond its byte bound.')
    return raw


def _json(raw: bytes):
    try:
        return json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(
            PublicationError('Nonfinite JSON values cannot be published.')))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicationError('Expected valid UTF-8 JSON in the sealed run.') from exc


def _no_private(value):
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if any(str(key).lower() in PRIVATE_KEYS for key in item):
                raise PublicationError('Credential and raw-response fields cannot be published.')
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)


def _scan_text(raw: bytes):
    if TOKEN.search(raw) or PRIVATE_TEXT.search(raw):
        raise PublicationError('Possible credential or private execution path in a public artifact.')


def _inspect_compressed(name: str, raw: bytes):
    if name.endswith('.json.gz'):
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
                decoded = stream.read(MAX_DECODED_BYTES + 1)
        except (OSError, EOFError) as exc:
            raise PublicationError('Compressed JSON evidence is invalid.') from exc
        if len(decoded) > MAX_DECODED_BYTES:
            raise PublicationError('Decoded rollout exceeds its publication bound.')
        _no_private(_json(decoded))
        _scan_text(decoded)
        return len(decoded)
    elif name.endswith('.npz'):
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                decoded_total = 0
                for member in archive.infolist():
                    if member.file_size > MAX_DECODED_BYTES:
                        raise PublicationError('Training array exceeds its decoded size bound.')
                    decoded_total += member.file_size
                    if decoded_total > MAX_TOTAL_BYTES:
                        raise PublicationError('Training archive exceeds its decoded size bound.')
                    content = archive.read(member)
                    _scan_text(content)
        except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
            raise PublicationError('Training sample archive is invalid.') from exc
        return decoded_total
    return 0


def _relative(name: str) -> PurePosixPath:
    if not isinstance(name, str):
        raise PublicationError('Invalid artifact path in the integrity manifest.')
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or '\\' in name or
            any(part in {'', '.', '..'} or part.startswith('.') for part in path.parts)):
        raise PublicationError('Unsafe or hidden path in the integrity manifest.')
    if path.suffix not in ALLOWED_SUFFIXES and path.name != 'LICENSE':
        raise PublicationError('Unexpected artifact type in the integrity manifest.')
    return path


def inventory(run: Path) -> dict[str, bytes]:
    """Read and verify the complete run inventory before preparing cards."""
    root = _plain(run)
    if not root.is_dir():
        raise PublicationError('Expected a plain completed PiPER imitation run directory.')
    payload: dict[str, bytes] = {}
    total = 0
    decoded_total = 0
    for path in sorted(root.rglob('*')):
        _plain(path)
        if path.is_dir():
            continue
        name = path.relative_to(root).as_posix()
        rel = _relative(name)
        if rel.parts[0] not in RUN_ROOT_FILES and rel.parts[0] not in RUN_DIRECTORIES:
            raise PublicationError('Unexpected run directory outside the study export allowlist.')
        raw = _read(path)
        if rel.suffix == '.json':
            _no_private(_json(raw))
        decoded_total += _inspect_compressed(name, raw)
        if decoded_total > MAX_DECODED_TOTAL_BYTES:
            raise PublicationError('Decoded study evidence exceeds its total size bound.')
        _scan_text(raw)
        payload[name] = raw
        total += len(raw)
        if len(payload) > MAX_FILES or total > MAX_TOTAL_BYTES:
            raise PublicationError('PiPER imitation run exceeds its publication bounds.')
    required = {'protocol.json', 'report.json', 'integrity.json', 'frozen.json',
                'confirmation-opened.json', 'training/dataset.json', 'training/samples.npz', 'model/model.json',
                'model/shuffled-model.json',
                'viewer/index.html', 'viewer/report.json', 'driver-recovery.json',
                'sources/piper_imitation_driver.py'}
    if not required <= payload.keys():
        raise PublicationError('Run is missing its protocol, report, model metadata or viewer.')
    if not any(name.startswith('demonstrations/') and name.endswith('.json.gz') for name in payload):
        raise PublicationError('Run must retain its full synthetic teacher demonstrations.')
    if not any(name.startswith('attempts/') and name.endswith('.json.gz') for name in payload):
        raise PublicationError('Run must retain full compressed rollout attempts.')
    if not any(name.startswith('viewer/') and name not in {'viewer/index.html', 'viewer/report.json'}
               for name in payload):
        raise PublicationError('Run must include selected replay assets in the viewer directory.')
    if not any(name.startswith('driver-recovery/') and name.endswith('.json') for name in payload):
        raise PublicationError('Run must retain per-attempt confirmation recovery records.')

    report = _json(payload['report.json'])
    if not isinstance(report, dict):
        raise PublicationError('Completed study report must be a JSON object.')
    limitations = report.get('limitations', report.get('limits'))
    if (report.get('status') != 'completed'
            or report.get('schema') != 'dvidia.piper-imitation-study/v1'
            or not isinstance(report.get('variants'), dict) or not report['variants']
            or not isinstance(limitations, (list, dict)) or not limitations):
        raise PublicationError('Only a complete study report with variant counts and declared limitations may publish.')
    if isinstance(limitations, list) and any(not isinstance(item, str) or not item.strip() for item in limitations):
        raise PublicationError('Declared limitations must be readable text.')
    if isinstance(limitations, dict) and any(not isinstance(key, str) or not isinstance(value, str)
                                               for key, value in limitations.items()):
        raise PublicationError('Declared limitations must be readable text.')
    for name, row in report['variants'].items():
        if (not isinstance(name, str) or not isinstance(row, dict)
                or type(row.get('scheduled')) is not int or row['scheduled'] < 0
                or type(row.get('successes')) is not int or not 0 <= row['successes'] <= row['scheduled']):
            raise PublicationError('Report variant counts must have valid scheduled and success counts.')
    if (not isinstance(report.get('attempts'), list)
            or sum(row['scheduled'] for row in report['variants'].values()) != len(report['attempts'])):
        raise PublicationError('Report attempt rows do not match its variant denominators.')
    protocol = _json(payload['protocol.json'])
    model = _json(payload['model/model.json'])
    frozen = _json(payload['frozen.json'])
    training = _json(payload['training/dataset.json'])
    opened = _json(payload['confirmation-opened.json'])
    if (not isinstance(protocol, dict) or protocol.get('schema') != 'dvidia.piper-imitation-study/v1'
            or not isinstance(model, dict) or not isinstance(frozen, dict)
            or not isinstance(training, dict) or not isinstance(opened, dict)):
        raise PublicationError('Protocol, frozen checkpoint and training receipt must be JSON objects.')
    if (_record_value(payload['protocol.json']) != frozen.get('protocol_sha256')
            or _record_value(payload['training/dataset.json']) != frozen.get('dataset_sha256')
            or _record_value(payload['model/model.json']) != frozen.get('model_sha256')
            or _record_value(payload['model/shuffled-model.json']) != frozen.get('shuffled_model_sha256')
            or training.get('protocol_sha256') != frozen.get('protocol_sha256')
            or training.get('samples_sha256') != _record_value(payload['training/samples.npz'])
            or opened.get('frozen_sha256') != _record_value(payload['frozen.json'])
            or report.get('training') != training):
        raise PublicationError('Frozen checkpoint, training data and protocol hashes do not agree.')
    if (report.get('protocol_sha256') != frozen.get('protocol_sha256')
            or report.get('frozen_sha256') != _record_value(payload['frozen.json'])):
        raise PublicationError('Completed report does not match the frozen model and protocol.')
    recovery = _json(payload['driver-recovery.json'])
    adapter_sha = _record_value(payload['sources/piper_imitation_driver.py'])
    recovery_names = sorted(name for name in payload
                            if name.startswith('driver-recovery/') and name.endswith('.json'))
    recovery_records = recovery.get('records') if isinstance(recovery, dict) else None
    if (not isinstance(recovery, dict)
            or recovery.get('schema') != 'dvidia.piper-imitation-driver-recovery/v1'
            or recovery.get('frozen_sha256') != _record_value(payload['frozen.json'])
            or recovery.get('adapter_sha256') != adapter_sha
            or type(recovery.get('serialization_guard_resumes')) is not int
            or recovery.get('serialization_guard_resumes') != len(recovery_names)
            or type(recovery.get('completed_tapes_reexecuted')) is not int
            or recovery.get('completed_tapes_reexecuted') != 0
            or recovery.get('frozen_runtime_changed') is not False
            or not isinstance(recovery.get('issue'), str) or not recovery['issue'].strip()
            or recovery_records != recovery_names):
        raise PublicationError('Confirmation recovery summary does not match its frozen provenance.')
    for record_name in recovery_names:
        record = _json(payload[record_name])
        case_id = record.get('id') if isinstance(record, dict) else None
        variant = record.get('variant') if isinstance(record, dict) else None
        tape_path = f'attempts/{case_id}-{variant}.json.gz'
        tape = payload.get(tape_path)
        if (not isinstance(case_id, str) or not isinstance(variant, str)
                or record_name != f'driver-recovery/{case_id}-{variant}.json'
                or record.get('frozen_sha256') != _record_value(payload['frozen.json'])
                or record.get('adapter_sha256') != adapter_sha
                or record.get('reason') != 'configuration_tuple_list_comparison'
                or record.get('physical_attempt_repeated') is not False
                or tape is None or record.get('tape_sha256') != _record_value(tape)):
            raise PublicationError('Per-attempt recovery record is not bound to its preserved frozen tape.')
    cases = protocol.get('cases', {}).get('confirmation', []) if isinstance(protocol.get('cases'), dict) else []
    variants = protocol.get('variants')
    case_ids = [case.get('id') for case in cases if isinstance(case, dict)]
    if (not cases or len(case_ids) != len(cases)
            or any(not isinstance(case_id, str) for case_id in case_ids)
            or len(set(case_ids)) != len(case_ids)
            or not isinstance(variants, list) or any(not isinstance(item, str) for item in variants)
            or len(set(variants)) != len(variants)
            or set(variants) != set(report['variants'])
            or any(row['scheduled'] != len(case_ids) for row in report['variants'].values())):
        raise PublicationError('Report does not cover every frozen confirmation scene and policy variant.')
    expected_attempts = {(case_id, variant) for case_id in case_ids for variant in variants}
    observed_attempts = set()
    observed_counts = {name: {'scheduled': 0, 'successes': 0} for name in report['variants']}
    for attempt in report['attempts']:
        variant = attempt.get('variant') if isinstance(attempt, dict) else None
        episode = attempt.get('episode') if isinstance(attempt, dict) else None
        digest = attempt.get('sha256') if isinstance(attempt, dict) else None
        score = attempt.get('score') if isinstance(attempt, dict) else None
        case_id = attempt.get('id') if isinstance(attempt, dict) else None
        if (not isinstance(variant, str) or variant not in observed_counts
                or not isinstance(case_id, str) or not isinstance(episode, str) or not isinstance(score, dict)
                or type(score.get('success')) is not bool or not isinstance(digest, str)
                or not SHA.fullmatch(digest)):
            raise PublicationError('Report contains an invalid per-attempt result row.')
        episode_path = _relative(episode)
        episode_raw = payload.get(episode_path.as_posix())
        if (episode_raw is None or episode_path.as_posix() != f'attempts/{case_id}-{variant}.json.gz'
                or sha256(episode_raw).hexdigest() != digest):
            raise PublicationError('Report attempt row differs from its retained rollout trace.')
        identity = (case_id, variant)
        if identity not in expected_attempts or identity in observed_attempts:
            raise PublicationError('Report attempt identities differ from the frozen confirmation schedule.')
        observed_attempts.add(identity)
        observed_counts[variant]['scheduled'] += 1
        observed_counts[variant]['successes'] += int(score['success'])
    if observed_counts != report['variants'] or observed_attempts != expected_attempts:
        raise PublicationError('Variant counts differ from the retained per-attempt results.')
    training_rows = training.get('rows')
    train_cases = protocol.get('cases', {}).get('train', [])
    train_ids = [case.get('id') for case in train_cases if isinstance(case, dict)] if isinstance(train_cases, list) else []
    if (not isinstance(training_rows, list) or not isinstance(train_cases, list)
            or len(training_rows) != len(train_cases) or len(train_ids) != len(train_cases)
            or any(not isinstance(case_id, str) for case_id in train_ids)
            or len(set(train_ids)) != len(train_ids)):
        raise PublicationError('Training receipt does not include every scheduled teacher demonstration.')
    train_cases_by_id = {case['id']: case for case in train_cases}
    expected_train_ids = set(train_ids)
    observed_train_ids = set()
    for row in training_rows:
        case_id = row.get('id') if isinstance(row, dict) else None
        if not isinstance(row, dict) or not isinstance(case_id, str):
            raise PublicationError('Training receipt contains an invalid teacher demonstration row.')
        demo_name, demo_digest = row.get('episode'), row.get('sha256')
        demo_path = _relative(demo_name) if isinstance(demo_name, str) else None
        demo_raw = payload.get(demo_path.as_posix()) if demo_path else None
        scheduled_case = train_cases_by_id.get(case_id)
        if (demo_raw is None or not demo_path.as_posix().startswith('demonstrations/')
                or demo_path.as_posix() != f'demonstrations/{case_id}.json.gz'
                or sha256(demo_raw).hexdigest() != demo_digest
                or case_id not in expected_train_ids or case_id in observed_train_ids
                or not isinstance(scheduled_case, dict)
                or row.get('start_xy') != scheduled_case.get('start_xy')
                or row.get('goal_xy') != scheduled_case.get('goal_xy')
                or row.get('seed') != scheduled_case.get('seed')):
            raise PublicationError('Training receipt differs from its retained teacher demonstration.')
        observed_train_ids.add(case_id)
    if observed_train_ids != expected_train_ids:
        raise PublicationError('Training receipt omits a frozen training scene.')

    seal = _json(payload['integrity.json'])
    sealed_files = seal.get('files') if isinstance(seal, dict) else None
    actual_files = {name: receipt_bytes(raw) for name, raw in payload.items() if name != 'integrity.json'}
    if (not isinstance(seal, dict) or seal.get('schema') != 'dvidia.piper-imitation-integrity/v1'
            or not isinstance(sealed_files, dict) or sealed_files != actual_files):
        raise PublicationError('Run files differ from the complete integrity seal.')
    return payload


def receipt_bytes(raw: bytes) -> dict[str, object]:
    return {'bytes': len(raw), 'sha256': sha256(raw).hexdigest()}


def _record_value(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _card(kind: str, run: dict[str, bytes]) -> bytes:
    report = _json(run['report.json'])
    selected_model = _json(run['model/model.json'])
    model_schema = selected_model.get('schema') if isinstance(selected_model, dict) else None
    model_families = {
        'dvidia.piper-motor-primitive/v1': 'time-conditioned RBF feedforward motion primitive',
        'dvidia.piper-motor-primitive/v2': 'high-resolution time-conditioned RBF feedforward motion primitive (16 cubic spatial terms × 401 temporal centers, σ=0.025 s)',
        'dvidia.piper-imitation-mlp/v1': 'multilayer perceptron policy',
        'dvidia.piper-imitation-mlp/v2': 'multilayer perceptron policy',
    }
    family = model_families.get(model_schema)
    family_note = (f'The selected policy is a {family} (schema `{model_schema}` in `model/model.json`).'
                   if family else 'The selected policy is a learned movement policy; its schema is recorded in `model/model.json`.')
    primitive_limit = (' The feedforward primitive does not learn reactive contact correction or recovery; '
                       'the native servos remain authored.'
                       if model_schema in {'dvidia.piper-motor-primitive/v1',
                                           'dvidia.piper-motor-primitive/v2'} else '')
    variants = report['variants']
    limits = report.get('limitations', report.get('limits'))
    result_table = '\n'.join(f"| {name} | {row['successes']} / {row['scheduled']} |"
                             for name, row in variants.items())
    displayed_limits = limits
    if isinstance(limits, list) and model_schema == 'dvidia.piper-motor-primitive/v2':
        displayed_limits = [
            ('Earlier MLP candidates used privileged simulator joint, TCP, object and contact measurements; '
             'no vision pipeline was trained. This broad report statement describes those candidates, not '
             'the selected v2 primitive input contract.')
            if item == ('Learned inference uses privileged simulator joint, TCP, object and contact measurements; '
                        'no vision pipeline is trained.') else item
            for item in limits
        ]
    limitation_text = ('\n'.join(f'- {item}' for item in displayed_limits) if isinstance(displayed_limits, list)
                       else '\n'.join(f'- {name}: {value}' for name, value in displayed_limits.items()))
    selected_input_note = (
        'Selected v2 primitive inputs: reset-time object XY, one of the two known goal coordinates, and elapsed '
        'time. Live joint positions are used only by generic native joint-target caps. The selected policy does '
        'not consume TCP or contact estimates and does not learn contact feedback or recovery. The retained '
        'demonstrations and evaluation traces contain richer privileged simulator data; those records do not '
        'change this selected-policy input contract.\n\n'
        if model_schema == 'dvidia.piper-motor-primitive/v2' else ''
    )
    recovery_note = (
        'Confirmation encountered a serialization guard: the live configuration used tuples, while '
        'the persisted JSON representation used lists. A bounded adapter checked the exact error and '
        'the frozen bindings, then resumed scoring from each already-saved native episode tape; it '
        'did not rerun physics. The adapter source, root summary, and per-attempt recovery records '
        'are retained in `sources/piper_imitation_driver.py`, `driver-recovery.json`, and '
        '`driver-recovery/`. The published report keeps the same frozen confirmation roster and '
        'result counts, and the recovery summary records zero completed tapes reexecuted.\n\n'
    )
    if kind == 'model':
        body = (
            'A DVIDIA research artifact for learned PiPER block placement. ' + family_note +
            ' It is trained by imitation from demonstrations generated by an authored synthetic '
            'simulator teacher, using privileged simulator data. It does not establish learning '
            'from human video, camera-only control, physical robot performance, or hardware readiness.\n\n'
            'The selected policy metadata and weights are in `model/`; the shuffled-label model '
            'is retained as a negative control. The protocol and aggregate report are included. '
            'The dataset preserves all scheduled demonstrations, earlier failed MLP '
            'candidate checkpoints and their development traces, and all confirmation '
            'attempts. Browse the '
            '[complete dataset](https://huggingface.co/datasets/' + DATASET_REPO +
            ') or open the [recorded replay viewer](' + SPACE_VIEWER_URL + ').\n\n' + selected_input_note + recovery_note +
            '## Run the policy in native simulation\n\n'
            'Use Python 3.12 or later. This runs locally in pinned native simulation and does not '
            'connect to robot hardware.\n\n'
            '```sh\n'
            'python -m pip install --force-reinstall "dvidia-training[arm] @ '
            'https://github.com/Dvidia-Inference/dvidia-training/releases/download/'
            'piper-imitation-v0.1/dvidia_training-0.1.0-py3-none-any.whl"\n'
            'dvidia-robot-sim fetch piper --output robot-assets\n'
            'wget https://huggingface.co/' + MODEL_REPO +
            '/resolve/main/model/model.json -O model.json\n'
            'dvidia-piper-learned --model model.json --assets robot-assets '
            '--start-x 0.30 --start-y -0.03 --goal 0 --output learned-run\n'
            '```\n\n' + primitive_limit + '\n\n'
            'Original DVIDIA artifact files are released under the MIT License. The PiPER model '
            'and other upstream components retain their own licenses.\n')
        front = '---\nlicense: mit\ntags:\n- robotics\n- imitation-learning\n- simulation\n---\n\n# PiPER block placement v0.1\n\n'
    elif kind == 'dataset':
        body = (
            'Synthetic simulator demonstrations for the first learned PiPER block-placement '
            'benchmark. The demonstrations come from a synthetic teacher and contain privileged '
            'simulator state/action observations; they are not human demonstrations or footage. The dataset '
            'includes all scheduled teacher demonstrations, held-out attempts, source snapshots, '
            'earlier failed MLP candidate checkpoints and all their development traces, the frozen protocol, and '
            'declared limitations for audit. ' + selected_input_note + recovery_note + 'Open the [recorded replay '
            'evidence](https://huggingface.co/spaces/' + SPACE_REPO + '/tree/main/' + SPACE_NAMESPACE +
            ') for the browser viewer.\n\n'
            'These artifacts do not '
            'establish camera-only control, physical robot performance, or hardware readiness.\n\n'
            'Original DVIDIA artifact files are released under the MIT License. Upstream robot '
            'assets retain their own licenses.\n')
        front = '---\nlicense: mit\ntask_categories:\n- robotics\nlanguage: en\n---\n\n# PiPER block-placement demonstrations v0.1\n\n'
    else:
        body = (
            'Static, recorded evidence for the learned PiPER block-placement benchmark. '
            + family_note + ' It is trained from demonstrations generated by an authored synthetic '
            'simulator teacher and uses privileged simulator data.' + primitive_limit +
            ' This evidence does not establish '
            'human-video learning, camera-only control, physical performance, or hardware readiness.\n\n'
            + selected_input_note + recovery_note +
            'Open `index.html` to inspect the selected replay and `report.json` for the recorded '
            'variant counts and declared limits. Full attempt traces and the source model are '
            'retained under this versioned path.\n')
        front = '---\ntitle: PiPER learned placement v0.1\nsdk: static\n---\n\n# PiPER learned placement v0.1\n\n'
    body += (f'\nThe report declares {len(variants)} variants and {len(limits)} limitation entries. '
             'Variant results are recorded below; read the report for per-attempt outcomes.\n\n'
             '| Variant | Successes / scheduled |\n| --- | ---: |\n' + result_table +
             '\n\n## Declared limitations\n\n' + limitation_text + '\n')
    return (front + body).encode('utf-8')


def _destination_payloads(run: dict[str, bytes], license_bytes: bytes) -> dict[str, dict[str, bytes]]:
    dataset_prefixes = ('demonstrations/', 'attempts/', 'attempt-starts/', 'attempt-scores/',
                        'development_models/', 'sources/', 'driver-recovery/')
    training_names = [name for name in run if name.startswith('training/')]
    attempt_names = [name for name in run if name.startswith(dataset_prefixes)]
    evidence_names = [name for name in run if name in {'protocol.json', 'report.json', 'integrity.json',
                                                        'frozen.json', 'confirmation-opened.json', 'audit.json',
                                                        'driver-recovery.json'}]
    model_names = [name for name in run if name.startswith('model/') or name.startswith('sources/')]
    model_names += [name for name in run if name == 'training/dataset.json']
    dataset = {'README.md': _card('dataset', run), 'LICENSE': license_bytes,
               **{name: run[name] for name in evidence_names}}
    dataset.update({name: run[name] for name in training_names + attempt_names})
    dataset['model/model.json'] = run['model/model.json']
    model = {'README.md': _card('model', run), 'LICENSE': license_bytes,
             **{name: run[name] for name in evidence_names}}
    model.update({name: run[name] for name in model_names})
    space = {f'{SPACE_NAMESPACE}/index.html': SPACE_LANDING_PAGE,
             f'{SPACE_NAMESPACE}/LICENSE': license_bytes,
             f'{SPACE_NAMESPACE}/README.md': _card('space', run)}
    for name, raw in run.items():
        # Preserve the viewer's directory-relative asset layout.
        target = f'{SPACE_NAMESPACE}/{name}'
        space[target] = raw
    return {'model': model, 'dataset': dataset, 'space': space}


def stage(run_path: Path, source_commit: str, output: Path) -> dict:
    run_path, destination = _plain(run_path), _plain(output)
    if not re.fullmatch(r'(?:[a-f0-9]{40}|[a-f0-9]{64})', source_commit):
        raise PublicationError('An exact source commit is required.')
    if destination.exists() or any(destination == item or item in destination.parents or destination in item.parents
                                   for item in (run_path,)):
        raise PublicationError('Stage output must be fresh and separate from the source run.')
    run = inventory(run_path)
    repo_root = Path(__file__).resolve().parents[2]
    license_bytes = _read(repo_root / 'LICENSE', 1024 * 1024)
    if not license_bytes.startswith(b'MIT License'):
        raise PublicationError('The DVIDIA artifact license must be MIT.')
    payloads = _destination_payloads(run, license_bytes)
    if sum(map(len, payloads.values())) > MAX_TOTAL_BYTES:
        raise PublicationError('Staged publication exceeds its total size bound.')
    destination.mkdir(parents=True, exist_ok=False)
    repo_info = {'model': MODEL_REPO, 'dataset': DATASET_REPO, 'space': SPACE_REPO}
    directories = {'model': 'model', 'dataset': 'dataset', 'space': 'space'}
    rows = []
    for kind in ('model', 'dataset', 'space'):
        directory = directories[kind]
        base = destination / directory
        base.mkdir()
        files = []
        for name, raw in sorted(payloads[kind].items()):
            _relative(name)
            target = base / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            files.append({'path': name, **receipt_bytes(raw)})
        row = {'repo_id': repo_info[kind], 'repo_type': kind, 'directory': directory, 'files': files}
        if kind == 'space':
            row['append_only_namespace'] = SPACE_NAMESPACE
        rows.append(row)
    plan = {'kind': KIND, 'schema_version': SCHEMA_VERSION, 'private': False,
            'source_commit': source_commit,
            'source_integrity_sha256': _record_value(_read(run_path / 'integrity.json')),
            'repositories': rows}
    plan_path = destination / 'publication-plan.json'
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return checked_plan(plan_path)


def checked_plan(path: Path) -> dict:
    path = _plain(path)
    if not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
        raise PublicationError('Expected a bounded local publication plan.')
    plan = _json(path.read_bytes())
    if (not isinstance(plan, dict) or plan.get('kind') != KIND or plan.get('schema_version') != SCHEMA_VERSION
            or plan.get('private') is not False or not re.fullmatch(r'(?:[a-f0-9]{40}|[a-f0-9]{64})', plan.get('source_commit', ''))
            or not SHA.fullmatch(plan.get('source_integrity_sha256', ''))):
        raise PublicationError('Expected a reviewed PiPER imitation publication plan.')
    rows = plan.get('repositories')
    if not isinstance(rows, list) or len(rows) != 3:
        raise PublicationError('Plan must name exactly the three fixed destinations.')
    expected = {kind: (repo, directory) for kind, repo, directory in REPO_ROWS}
    total = 0
    decoded_total = 0
    decoded_artifacts: set[str] = set()
    for row in rows:
        kind = row.get('repo_type')
        if kind not in expected or row.get('repo_id') != expected[kind][0] or row.get('directory') != expected[kind][1]:
            raise PublicationError('Plan destination differs from the fixed study repositories.')
        if kind == 'space' and row.get('append_only_namespace') != SPACE_NAMESPACE:
            raise PublicationError('Space uploads are restricted to the fixed version namespace.')
        base = path.parent / row['directory']
        _plain(base)
        if not base.is_dir():
            raise PublicationError('Missing staged publication directory.')
        files = row.get('files')
        if not isinstance(files, list) or not files or len(files) > MAX_FILES:
            raise PublicationError('Invalid staged publication inventory.')
        expected_paths = set()
        for item in files:
            rel = _relative(item.get('path'))
            name = rel.as_posix()
            if name in expected_paths:
                raise PublicationError('Duplicate staged publication path.')
            expected_paths.add(name)
            if kind == 'space' and not name.startswith(SPACE_NAMESPACE + '/'):
                raise PublicationError('Space publication may only append inside the fixed version namespace.')
            if kind == 'model':
                allowed = (name in {'README.md', 'LICENSE', 'protocol.json', 'report.json', 'integrity.json',
                                    'frozen.json', 'confirmation-opened.json', 'audit.json', 'driver-recovery.json',
                                    'training/dataset.json'}
                           or name.startswith(('model/', 'sources/')))
                if not allowed:
                    raise PublicationError('Model repository path is outside the study artifact allowlist.')
            if kind == 'dataset':
                allowed = (name in {'README.md', 'LICENSE', 'protocol.json', 'report.json', 'integrity.json',
                                    'frozen.json', 'confirmation-opened.json', 'audit.json', 'driver-recovery.json',
                                    'model/model.json'}
                           or name.startswith(('training/', 'demonstrations/', 'attempts/', 'attempt-starts/', 'attempt-scores/',
                                               'development_models/', 'sources/', 'driver-recovery/')))
                if not allowed:
                    raise PublicationError('Dataset repository path is outside the study artifact allowlist.')
            source = base.joinpath(*rel.parts)
            raw = _read(source)
            record = receipt_bytes(raw)
            if item.get('bytes') != record['bytes'] or item.get('sha256') != record['sha256']:
                raise PublicationError('Staged artifact differs from the reviewed plan.')
            _scan_text(raw)
            if rel.suffix == '.json':
                _no_private(_json(raw))
            decoded = _inspect_compressed(name, raw)
            digest = sha256(raw).hexdigest()
            # The same immutable traces are deliberately copied to both the
            # dataset and Space. Count their decoded expansion once toward the
            # aggregate source-size bound, while validating every copy.
            if digest not in decoded_artifacts:
                decoded_artifacts.add(digest)
                decoded_total += decoded
            if decoded_total > MAX_DECODED_TOTAL_BYTES:
                raise PublicationError('Decoded staged evidence exceeds its total size bound.')
            total += len(raw)
        actual_paths = set()
        for current in base.rglob('*'):
            _plain(current)
            if current.is_file():
                actual_paths.add(current.relative_to(base).as_posix())
        if actual_paths != expected_paths:
            raise PublicationError('Staged inventory changed after plan creation.')
        if total > MAX_TOTAL_BYTES:
            raise PublicationError('Publication plan exceeds its total size bound.')
    if {row['repo_type'] for row in rows} != set(expected):
        raise PublicationError('Plan has duplicate or missing repository types.')
    space_root = path.parent / 'space' / SPACE_NAMESPACE
    source_seal = _read(space_root / 'integrity.json')
    if _record_value(source_seal) != plan['source_integrity_sha256']:
        raise PublicationError('Plan source integrity hash differs from the staged Space evidence.')
    seal = _json(source_seal)
    if (not isinstance(seal, dict) or seal.get('schema') != 'dvidia.piper-imitation-integrity/v1'
            or not isinstance(seal.get('files'), dict)):
        raise PublicationError('Staged Space source seal is invalid.')
    rows_by_type = {row['repo_type']: row for row in rows}
    staged = {}
    for kind in ('dataset', 'space'):
        staged[kind] = {item['path']: (item['bytes'], item['sha256'])
                         for item in rows_by_type[kind]['files']}
    for source_name, record in seal['files'].items():
        expected = (record.get('bytes'), record.get('sha256')) if isinstance(record, dict) else None
        if (kind := source_name.split('/', 1)[0]) not in RUN_ROOT_FILES | RUN_DIRECTORIES:
            raise PublicationError('Integrity seal names a path outside the study export allowlist.')
        space_name = SPACE_NAMESPACE + '/' + source_name
        source_plan_item = staged['space'].get(space_name)
        if not isinstance(expected, tuple) or source_plan_item != expected:
            raise PublicationError('Space evidence is missing a sealed run artifact.')
        if source_name.startswith(('training/', 'demonstrations/', 'attempts/', 'attempt-starts/', 'attempt-scores/',
                                   'development_models/', 'sources/', 'driver-recovery/')):
            if staged['dataset'].get(source_name) != expected:
                raise PublicationError('Dataset omits a required source, demonstration or attempt artifact.')
    source_names = set(seal['files']) | {'integrity.json'}
    expected_space = {SPACE_NAMESPACE + '/' + name for name in source_names}
    expected_space |= {SPACE_NAMESPACE + '/index.html', SPACE_NAMESPACE + '/README.md',
                       SPACE_NAMESPACE + '/LICENSE'}
    if set(staged['space']) != expected_space:
        raise PublicationError('Space plan must contain the full sealed run and only the new version namespace.')
    landing_receipt = receipt_bytes(SPACE_LANDING_PAGE)
    if staged['space'].get(SPACE_NAMESPACE + '/index.html') != (
            landing_receipt['bytes'], landing_receipt['sha256']):
        raise PublicationError('Space namespace landing page must redirect to the relative viewer directory.')
    dataset_sources = {name for name in source_names if name in RUN_ROOT_FILES
                       or name.startswith(('training/', 'demonstrations/', 'attempts/', 'attempt-starts/', 'attempt-scores/',
                                           'development_models/', 'sources/', 'driver-recovery/'))}
    expected_dataset = dataset_sources | {'README.md', 'LICENSE', 'model/model.json'}
    if set(staged['dataset']) != expected_dataset:
        raise PublicationError('Dataset plan must retain every demonstration and scheduled attempt.')
    model_sources = {name for name in source_names if name in RUN_ROOT_FILES
                     or name.startswith(('model/', 'sources/')) or name == 'training/dataset.json'}
    expected_model = model_sources | {'README.md', 'LICENSE'}
    if set({item['path'] for item in rows_by_type['model']['files']}) != expected_model:
        raise PublicationError('Model plan must contain only the selected study model and its provenance.')
    return plan


def dry_run(path: Path) -> dict:
    plan = checked_plan(path)
    rows = []
    for row in plan['repositories']:
        rows.append({'repo_type': row['repo_type'], 'repo_id': row['repo_id'],
                     **({'append_only_namespace': row['append_only_namespace']} if row['repo_type'] == 'space' else {}),
                     'files': row['files'], 'bytes': sum(item['bytes'] for item in row['files'])})
    summary = {'source_commit': plan['source_commit'], 'repositories': rows,
               'network': False, 'credentials_read': False, 'remote_mutations': False}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def publish(path: Path):
    path = _plain(path)
    plan = checked_plan(path)
    receipt_path = path.parent / 'publication-receipt.json'
    _plain(receipt_path)
    if receipt_path.exists():
        raise PublicationError('Publication receipt already exists; do not overwrite it.')
    # Read credentials only after the reviewed plan passes its offline checks.
    # Client construction is local; no repository request occurs until below.
    api, token = client(path.parent / 'publisher-cache')
    # Freeze every reviewed byte before the first remote call.
    payload: dict[tuple[str, str], bytes] = {}
    for row in plan['repositories']:
        base = path.parent / row['directory']
        for item in row['files']:
            raw = _read(base / _relative(item['path']))
            if receipt_bytes(raw) != {'bytes': item['bytes'], 'sha256': item['sha256']}:
                raise PublicationError('Artifact changed before freezing the upload.')
            if token.encode() in raw or TOKEN.search(raw) or PRIVATE_TEXT.search(raw):
                raise PublicationError('Possible credential found in frozen upload bytes.')
            payload[row['repo_type'], item['path']] = raw

    identity = api.whoami()
    if 'Dvidia' not in {org.get('name') for org in identity.get('orgs', [])}:
        raise PublicationError('Authenticated account is not a member of Dvidia.')
    from huggingface_hub import CommitOperationAdd
    from huggingface_hub.errors import RepositoryNotFoundError

    # Preflight all three targets before the first create/commit operation.
    destinations = []
    for row in plan['repositories']:
        kind = row['repo_type']
        try:
            info = api.repo_info(row['repo_id'], repo_type=kind)
        except RepositoryNotFoundError:
            info = None
        if info is not None:
            checked_remote(info, kind)
            existing = {item.rfilename for item in info.siblings}
            if kind == 'space':
                if any(name == SPACE_NAMESPACE or name.startswith(SPACE_NAMESPACE + '/') for name in existing):
                    raise PublicationError('The learned-placement/v01 Space namespace already exists.')
                selected = {item['path'] for item in row['files']}
                # The existing Space is expected to contain historical releases. Only the new namespace is added.
                if selected & existing:
                    raise PublicationError('A planned Space path already exists.')
                parent_tree = tree(api, info.sha)
            else:
                permitted = {item['path'] for item in row['files']} | {'.gitattributes'}
                if existing - permitted:
                    raise PublicationError('Model or dataset repository contains unrelated files.')
                for item in row['files']:
                    if item['path'] in existing:
                        remote_row = {'repo_id': row['repo_id'], 'repo_type': kind,
                                      'files': [item]}
                        verified_downloads(remote_row, info.sha, False)
                parent_tree = None
            destinations.append((row, info, parent_tree))
        else:
            if kind == 'space':
                raise PublicationError('The planned append-only Space does not exist.')
            destinations.append((row, None, None))

    results = []
    for row, info, parent_tree in destinations:
        kind = row['repo_type']
        if info is None:
            api.create_repo(row['repo_id'], repo_type=kind, private=False,
                            space_sdk='static' if kind == 'space' else None, exist_ok=False)
            info = api.repo_info(row['repo_id'], repo_type=kind)
            checked_remote(info, kind)
            existing_after_create = {item.rfilename for item in info.siblings}
            planned_after_create = {item['path'] for item in row['files']}
            if existing_after_create - planned_after_create - {'.gitattributes'}:
                raise PublicationError('New model or dataset repository has unexpected starter files.')
            for item in row['files']:
                if item['path'] in existing_after_create:
                    verified_downloads({**row, 'files': [item]}, info.sha, False)
        if kind == 'space' and any(item.rfilename == SPACE_NAMESPACE or
                                   item.rfilename.startswith(SPACE_NAMESPACE + '/') for item in info.siblings):
            raise PublicationError('The learned-placement/v01 Space namespace appeared during preflight.')
        existing = {item.rfilename for item in info.siblings}
        operations = []
        for item in row['files']:
            if item['path'] in existing:
                continue
            operation = CommitOperationAdd(path_in_repo=item['path'],
                                           path_or_fileobj=payload[kind, item['path']])
            if kind == 'space':
                # Existing Space attributes may classify compressed evidence as LFS;
                # keep every new viewer and replay asset directly on the static origin.
                operation._upload_mode = 'regular'
                operation._should_ignore = False
            operations.append(operation)
        if operations:
            commit = api.create_commit(row['repo_id'], repo_type=kind, operations=operations,
                                       parent_commit=info.sha,
                                       commit_message='Publish PiPER learned placement v0.1 evidence')
            revision = commit.oid
        else:
            revision = info.sha
        if kind == 'space':
            after_tree = tree(api, revision)
            expected_tree = set(parent_tree) | {item['path'] for item in row['files']}
            if set(after_tree) != expected_tree:
                raise PublicationError('Space append changed paths outside the new namespace.')
            if any(after_tree[name] != parent_tree[name] for name in parent_tree):
                raise PublicationError('Space history or its Git attributes changed during publication.')
            for item in row['files']:
                entry = after_tree[item['path']]
                if entry.get('lfs_sha256') is not None or not matches_tree(payload[kind, item['path']], entry):
                    raise PublicationError('New Space files must match regular Git bytes.')
        verified_downloads(row, revision, False)
        result = {'repo_id': row['repo_id'], 'repo_type': kind, 'commit': revision,
                  'verified_files': len(row['files']),
                  'url': 'https://huggingface.co/' +
                         ({'model': '', 'dataset': 'datasets/', 'space': 'spaces/'}[kind]) + row['repo_id']}
        results.append(result)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.publication-receipt-',
                                     delete=False, encoding='utf-8') as staged:
        json.dump({'schema': KIND + '/receipt-v1', 'repositories': results}, staged, indent=2, sort_keys=True)
        staged.write('\n')
    os.replace(staged.name, receipt_path)
    print(json.dumps({'repositories': results}, indent=2, sort_keys=True))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare_parser = commands.add_parser('prepare')
    prepare_parser.add_argument('--run', type=Path, required=True)
    prepare_parser.add_argument('--source-commit', required=True)
    prepare_parser.add_argument('--output', type=Path, required=True)
    for name in ('check', 'dry-run', 'publish'):
        sub = commands.add_parser(name)
        sub.add_argument('plan', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            result = stage(args.run, args.source_commit, args.output)
            print(json.dumps({'plan': str(args.output / 'publication-plan.json'),
                              'repositories': [row['repo_id'] for row in result['repositories']]}, indent=2))
        elif args.command == 'check':
            plan = checked_plan(args.plan)
            print(json.dumps({'checked_repositories': [row['repo_id'] for row in plan['repositories']]}, indent=2))
        elif args.command == 'dry-run':
            dry_run(args.plan)
        else:
            publish(args.plan)
    except KeyboardInterrupt:
        print('Publishing interrupted; completed public commits remain available.', file=sys.stderr)
        return 130
    except Exception as exc:
        if isinstance(exc, PublicationError):
            print(str(exc), file=sys.stderr)
        else:
            code = getattr(getattr(exc, 'response', None), 'status_code', None)
            print(f'Publishing stopped: {type(exc).__name__}; HTTP status {code}. Raw error suppressed.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Pure, shared model-card receipt contract; no files, credentials or network."""
from __future__ import annotations

import re

SCHEMA = 'dvidia.camera-action-cards/v1'
RECEIPT_FILE = 'model-cards-receipt.json'
MAX_CARD_BYTES = 256 * 1024
MAX_RECEIPT_BYTES = 2 * 1024 * 1024
MODEL = re.compile(r'models/(initial|corrected)/(full|blind)/model\.json\Z')
SOURCE_REQUIRED = {'summary.json', 'generator.py', 'summarize_results.py'}
SOURCE_OPTIONAL = {'hardware.json', 'full-study-audit.json', 'diagnostic-phase-coverage.json',
                   'independent-audit.json', 'diagnostic_phase_coverage.py'}
FIELDS = {'schema', 'source_commit', 'protocol_sha256', 'report_sha256', 'status',
          'qualification', 'qualified', 'full_study_audited', 'physical_robot_ready',
          'summary_reconstructed', 'physics_executed', 'model_executed',
          'source_inputs', 'study_bindings', 'files', 'model_cards'}


def file_identity(value, maximum=1024 ** 3):
    if (type(value) is not dict or set(value) != {'bytes', 'sha256'}
            or type(value['bytes']) is not int or not 0 <= value['bytes'] <= maximum
            or type(value['sha256']) is not str or not re.fullmatch('[a-f0-9]{64}', value['sha256'])):
        raise ValueError('Malformed model-card file identity.')
    return value


def card_name(model):
    match = MODEL.fullmatch(model)
    if match is None:
        raise ValueError('Unsupported model-card artifact path.')
    return f'model-cards/{match[1]}/{match[2]}/README.md'


def validate_receipt(receipt, *, source_commit, protocol_sha256, report_sha256, report, models):
    """Bind the exact card/model roster and qualification to supplied sealed facts."""
    if (type(receipt) is not dict or set(receipt) != FIELDS or receipt['schema'] != SCHEMA
            or not re.fullmatch('[a-f0-9]{40}', source_commit or '')
            or receipt['source_commit'] != source_commit
            or receipt['protocol_sha256'] != protocol_sha256 or receipt['report_sha256'] != report_sha256
            or any(receipt[key] != report[key] for key in ('status', 'qualification', 'qualified'))
            or type(receipt['qualified']) is not bool or type(receipt['full_study_audited']) is not bool
            or any(receipt[key] is not value for key, value in (
                ('physical_robot_ready', False), ('summary_reconstructed', True),
                ('physics_executed', False), ('model_executed', False)))):
        raise ValueError('Model-card receipt identity or qualification differs from sealed evidence.')
    if len(models) > 4:
        raise ValueError('Model-card roster exceeds the registered four candidates.')
    expected = []
    for model, identity in sorted(models.items()):
        file_identity(identity)
        match = MODEL.fullmatch(model)
        card = card_name(model)
        expected.append(dict(card=card, model=model, model_sha256=identity['sha256'],
            selected=match[1] == report.get('selected_stage')))
    stages = {MODEL.fullmatch(name)[1] for name in models}
    if (any({name for name in models if name.startswith('models/'+stage+'/')} !=
            {f'models/{stage}/{variant}/model.json' for variant in ('full', 'blind')} for stage in stages)
            or report.get('selected_stage') is not None and report['selected_stage'] not in stages):
        raise ValueError('Model-card roster lacks its matched or selected model pair.')
    records = receipt['model_cards']
    if (type(records) is not list or any(type(row) is not dict or set(row) !=
            {'card', 'model', 'model_sha256', 'selected'} or type(row['selected']) is not bool for row in records)
            or sorted(records, key=lambda row: row['model']) != expected):
        raise ValueError('Model cards differ from the exact sealed model roster or selection.')
    if type(receipt['files']) is not dict or set(receipt['files']) != {'README.md'} | {r['card'] for r in expected}:
        raise ValueError('Generated model-card file roster differs.')
    for value in receipt['files'].values():
        file_identity(value, MAX_CARD_BYTES)
    sources = receipt['source_inputs']
    if (type(sources) is not dict or not SOURCE_REQUIRED <= set(sources)
            or set(sources) - SOURCE_REQUIRED - SOURCE_OPTIONAL
            or receipt['full_study_audited'] is not ('full-study-audit.json' in sources)):
        raise ValueError('Model-card source input roster differs.')
    phase = {'diagnostic-phase-coverage.json', 'diagnostic_phase_coverage.py'}
    if set(sources) & phase and not phase | {'independent-audit.json'} <= set(sources):
        raise ValueError('Model-card phase provenance is incomplete.')
    for value in sources.values():
        file_identity(value)
    bindings = receipt['study_bindings']
    if type(bindings) is not dict or not 3 <= len(bindings) <= 80:
        raise ValueError('Model-card study binding roster is invalid.')
    for name, value in bindings.items():
        if not (name in {'protocol.json', 'registration.json', 'report.json', 'freeze.json'}
                or re.fullmatch(r'stages/(diagnostic|demonstrations|train-initial|development-initial|corrections|train-corrected|development-corrected|confirmation|negative)(\.integrity|\.started)?\.json', name)
                or re.fullmatch(r'models/(initial|corrected)/(full|blind)/(model|receipt|started)\.json', name)):
            raise ValueError('Unexpected model-card study binding path.')
        file_identity(value)
    if (bindings.get('protocol.json', {}).get('sha256') != protocol_sha256
            or bindings.get('report.json', {}).get('sha256') != report_sha256
            or 'registration.json' not in bindings
            or any(bindings.get(name) != identity for name, identity in models.items())):
        raise ValueError('Model-card study bindings differ from public model or report identities.')
    return [row['card'] for row in expected]

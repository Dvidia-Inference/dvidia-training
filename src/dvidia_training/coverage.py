"""Bounded, data-only capture planning; declarations never establish robot ability.

This companion contract does not change the footage manifest, run models, inspect
media, award rewards, or execute a benchmark. Source identities and review/rights
statements are declarations whose truth must be established outside this tool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


FORMAT = 'dvidia.skillspace-coverage-plan'
SCHEMA_VERSION = 1
MAX_PLAN_BYTES = 1024 * 1024
MAX_CASES = 64
MAX_EVIDENCE = 1200
OUTCOMES = ('success', 'failure', 'recovery', 'unknown')
REVIEW_STATES = ('accepted', 'proposed', 'rejected', 'needs_changes', 'stale')
PLAN_FIELDS = {'format', 'schema_version', 'plan_id', 'plan_revision', 'skill',
               'cases', 'benchmark', 'evidence'}
CASE_FIELDS = {'id', 'label', 'brief', 'outcome', 'required_groups', 'priority'}
EVIDENCE_FIELDS = {'id', 'plan_revision', 'case_id', 'media_sha256',
                   'source_recording_id', 'session_id', 'object_instance_id',
                   'outcome', 'complete', 'review_state', 'review_origin',
                   'rights_status', 'split'}
LINEAGE_FIELDS = ('source_recording_id', 'session_id', 'object_instance_id')


class CoverageError(ValueError):
    """A plan cannot be interpreted without relaxing its declared contract."""


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=False, allow_nan=False).encode('utf-8')
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise CoverageError('Plan must be finite UTF-8 JSON.') from exc


def _read(value):
    if type(value) is dict:
        raw = _canonical(value)
        document = value
    elif type(value) in (str, bytes):
        try:
            raw = value.encode('utf-8') if type(value) is str else value
        except UnicodeError as exc:
            raise CoverageError('Plan must be valid UTF-8 JSON.') from exc
        if len(raw) > MAX_PLAN_BYTES:
            raise CoverageError('Plan exceeds the 1 MiB JSON limit.')

        def pairs(items):
            result = {}
            for key, item in items:
                if key in result:
                    raise CoverageError('Duplicate JSON keys are not allowed.')
                result[key] = item
            return result

        def constant(_):
            raise CoverageError('Nonfinite JSON values are not allowed.')

        try:
            document = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs,
                                  parse_constant=constant)
        except (ValueError, UnicodeError, RecursionError) as exc:
            if isinstance(exc, CoverageError):
                raise
            raise CoverageError('Plan must be valid finite UTF-8 JSON.') from exc
        _canonical(document)
    else:
        raise CoverageError('Expected a plain object, JSON text, or JSON bytes.')
    if len(raw) > MAX_PLAN_BYTES:
        raise CoverageError('Plan exceeds the 1 MiB JSON limit.')
    return document


def _record(value, fields, path):
    if type(value) is not dict or set(value) != fields:
        raise CoverageError(path + ': expected exactly the documented fields.')
    return value


def _text(value, path, maximum=1600, *, blank=False):
    if type(value) is not str or len(value) > maximum or (not blank and not value.strip()):
        raise CoverageError(path + ': expected bounded text' + (' or an empty string.' if blank else '.'))
    if any((ord(c) < 32 and c not in '\t\n\r') or 127 <= ord(c) <= 159
           or 0xD800 <= ord(c) <= 0xDFFF for c in value):
        raise CoverageError(path + ': unsupported control characters or Unicode.')
    return value.strip()


def _identity(value, path):
    if type(value) is not str or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}', value):
        raise CoverageError(path + ': expected a stable identifier of at most 96 characters.')
    return value


def _integer(value, path, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise CoverageError(path + f': expected an integer in [{minimum}, {maximum}].')
    return value


def _choice(value, choices, path):
    if type(value) is not str or value not in choices:
        raise CoverageError(path + ': unsupported value.')
    return value


def validate_plan(plan):
    """Return a detached normalized contract, rejecting invented qualification."""
    row = _record(_read(plan), PLAN_FIELDS, 'plan')
    if row['format'] != FORMAT or type(row['schema_version']) is not int or row['schema_version'] != 1:
        raise CoverageError('Unsupported coverage plan format or schema version.')
    identity = _identity(row['plan_id'], 'plan_id')
    revision = _integer(row['plan_revision'], 'plan_revision', 1, 1_000_000)
    skill = _record(row['skill'], {'name', 'goal', 'operating_envelope'}, 'skill')
    skill = {'name': _text(skill['name'], 'skill.name', 120),
             'goal': _text(skill['goal'], 'skill.goal'),
             'operating_envelope': _text(skill['operating_envelope'], 'skill.operating_envelope')}
    if type(row['cases']) is not list or not 1 <= len(row['cases']) <= MAX_CASES:
        raise CoverageError('cases: expected 1–64 planned cases.')
    cases, case_ids = [], set()
    for index, item in enumerate(row['cases']):
        path = f'cases[{index}]'
        item = _record(item, CASE_FIELDS, path)
        case_id = _identity(item['id'], path + '.id')
        if case_id in case_ids:
            raise CoverageError('Case identities must be unique.')
        case_ids.add(case_id)
        cases.append({'id': case_id, 'label': _text(item['label'], path + '.label', 180),
                      'brief': _text(item['brief'], path + '.brief'),
                      'outcome': _choice(item['outcome'], OUTCOMES, path + '.outcome'),
                      'required_groups': _integer(item['required_groups'], path + '.required_groups', 1, 1000),
                      'priority': _integer(item['priority'], path + '.priority', 1, 10)})
    benchmark = _record(row['benchmark'], {'status', 'environment', 'reason',
                                          'success_predicate', 'failure_cases'}, 'benchmark')
    benchmark = {'status': _choice(benchmark['status'], ('blocked', 'specified'), 'benchmark.status'),
                 'environment': _text(benchmark['environment'], 'benchmark.environment', blank=True),
                 'reason': _text(benchmark['reason'], 'benchmark.reason', blank=True),
                 'success_predicate': _text(benchmark['success_predicate'], 'benchmark.success_predicate', blank=True),
                 'failure_cases': benchmark['failure_cases']}
    if type(benchmark['failure_cases']) is not list or len(benchmark['failure_cases']) > 64:
        raise CoverageError('benchmark.failure_cases: expected a list of at most 64 cases.')
    benchmark['failure_cases'] = [_text(item, 'benchmark.failure_cases', 600)
                                  for item in benchmark['failure_cases']]
    if benchmark['status'] == 'blocked' and not benchmark['reason']:
        raise CoverageError('A blocked benchmark needs an explicit reason.')
    if benchmark['status'] == 'specified' and (not benchmark['environment']
            or not benchmark['success_predicate'] or not benchmark['failure_cases']):
        raise CoverageError('A specified benchmark needs its environment, success predicate and failure cases.')
    if type(row['evidence']) is not list or len(row['evidence']) > MAX_EVIDENCE:
        raise CoverageError('evidence: expected a list of at most 1,200 declarations.')
    evidence, evidence_ids = [], set()
    for index, item in enumerate(row['evidence']):
        path = f'evidence[{index}]'
        item = _record(item, EVIDENCE_FIELDS, path)
        evidence_id = _identity(item['id'], path + '.id')
        if evidence_id in evidence_ids:
            raise CoverageError('Evidence identities must be unique.')
        evidence_ids.add(evidence_id)
        digest = item['media_sha256']
        if type(digest) is not str or not re.fullmatch(r'[a-f0-9]{64}', digest):
            raise CoverageError(path + '.media_sha256: expected a lowercase SHA-256 digest.')
        if type(item['complete']) is not bool:
            raise CoverageError(path + '.complete: expected an explicit boolean.')
        entry = {'id': evidence_id,
                 'plan_revision': _integer(item['plan_revision'], path + '.plan_revision', 1, 1_000_000),
                 'case_id': _identity(item['case_id'], path + '.case_id'),
                 'media_sha256': digest, 'complete': item['complete'],
                 'outcome': _choice(item['outcome'], OUTCOMES, path + '.outcome'),
                 'review_state': _choice(item['review_state'], REVIEW_STATES, path + '.review_state'),
                 'review_origin': _choice(item['review_origin'], ('human', 'ai', 'unknown'), path + '.review_origin'),
                 'rights_status': _choice(item['rights_status'], ('approved', 'unknown', 'denied'), path + '.rights_status'),
                 'split': _choice(item['split'], ('train', 'dev', 'confirmation', 'unassigned'), path + '.split')}
        for key in LINEAGE_FIELDS:
            entry[key] = None if item[key] is None else _identity(item[key], path + '.' + key)
        evidence.append(entry)
    return {'format': FORMAT, 'schema_version': SCHEMA_VERSION, 'plan_id': identity,
            'plan_revision': revision, 'skill': skill, 'cases': cases,
            'benchmark': benchmark, 'evidence': evidence}


def _groups(evidence):
    """Connect byte aliases and all declared lineage, including indirect links."""
    parent = list(range(len(evidence)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen = {}
    for index, item in enumerate(evidence):
        for field in ('media_sha256',) + LINEAGE_FIELDS:
            value = item[field]
            if value is None:
                continue
            key = (field, value)
            if key in seen:
                left, right = find(index), find(seen[key])
                if left != right:
                    parent[left] = right
            else:
                seen[key] = index
    return [find(index) for index in range(len(evidence))]


def summarize_plan(plan):
    """Inspect declared planning coverage, without running media or benchmarks."""
    plan = validate_plan(plan)
    cases = {item['id']: item for item in plan['cases']}
    evidence = plan['evidence']
    groups = _groups(evidence)
    group_splits, reviewed_outcomes = {}, {}
    for group, item in zip(groups, evidence):
        group_splits.setdefault(group, set()).add(item['split'])
        if (item['plan_revision'] == plan['plan_revision'] and item['complete']
                and item['review_state'] == 'accepted' and item['review_origin'] == 'human'
                and item['outcome'] != 'unknown'):
            reviewed_outcomes.setdefault(item['media_sha256'], set()).add(item['outcome'])
    by_case = {case_id: set() for case_id in cases}
    accepted_groups, accepted_ids, exclusions = set(), [], []
    outcome_stats = {outcome: {'submitted': 0, 'accepted': 0} for outcome in OUTCOMES}
    for group, item in zip(groups, evidence):
        outcome_stats[item['outcome']]['submitted'] += 1
        reasons = []
        case = cases.get(item['case_id'])
        if item['plan_revision'] != plan['plan_revision']:
            reasons.append('stale plan revision')
        if case is None:
            reasons.append('case absent from this plan revision')
        if not item['complete']:
            reasons.append('partial attempt')
        if item['review_state'] != 'accepted':
            reasons.append('review is ' + item['review_state'])
        if item['review_origin'] != 'human':
            reasons.append('human review not declared')
        if item['rights_status'] != 'approved':
            reasons.append('intended-use rights are ' + item['rights_status'])
        if any(item[key] is None for key in LINEAGE_FIELDS):
            reasons.append('unknown recording, session or object lineage')
        if item['outcome'] == 'unknown':
            reasons.append('outcome unknown')
        elif case is not None and item['outcome'] != case['outcome']:
            reasons.append('outcome does not match the planned case')
        if len(reviewed_outcomes.get(item['media_sha256'], ())) > 1:
            reasons.append('identical media has conflicting accepted human outcome declarations')
        splits = group_splits[group]
        if 'confirmation' in splits:
            reasons.append('connected group reserved for confirmation')
        elif {'train', 'dev'} <= splits:
            reasons.append('connected group crosses train and development splits')
        elif item['split'] == 'unassigned':
            reasons.append('split unassigned')
        if reasons:
            exclusions.append(item['id'] + ': ' + '; '.join(reasons) + '.')
            continue
        by_case[item['case_id']].add(group)
        accepted_groups.add(group)
        accepted_ids.append(item['id'])
        outcome_stats[item['outcome']]['accepted'] += 1
    cells = []
    for case in plan['cases']:
        count = len(by_case[case['id']])
        gap = max(0, case['required_groups'] - count)
        cells.append({**case, 'accepted_groups': count, 'gap': gap,
                      'status': 'met' if gap == 0 else 'gap'})
    gaps = sorted((cell for cell in cells if cell['gap']), key=lambda c: (-c['priority'], c['id']))
    next_capture = None
    if gaps:
        case = gaps[0]
        why = (f"Planning target needs {case['gap']} more distinct declared source group(s). "
               'Recording alone does not meet the target; completeness, review and intended-use rights are required.')
        if case['outcome'] == 'unknown':
            why = 'Define this case\'s intended observed outcome before requesting qualifying evidence.'
        next_capture = {'case_id': case['id'], 'label': case['label'], 'brief': case['brief'], 'why': why}
    benchmark = plan['benchmark']
    warnings = [
        'Planning targets are not scientifically sufficient dataset sizes or proof that a skillspace is exhausted.',
        'Counts use declared metadata only. This tool does not inspect videos or independently verify source novelty, reviews, outcomes or rights.',
        'Related sources stay connected through hashes, recordings, sessions and physical object instances; confirmation groups do not satisfy capture targets.',
        'A specified benchmark is a plan, not executed evidence. Coverage does not establish training readiness, acquired skill or hardware qualification.',
    ]
    if any(case['outcome'] == 'unknown' for case in plan['cases']):
        warnings.append('Cases with an unknown intended outcome cannot satisfy a capture target.')
    if plan['plan_id'].startswith('demo-'):
        warnings.append('This example contains fabricated demonstration metadata, not actual footage or a published dataset.')
    return {'format': 'dvidia.skillspace-coverage-summary', 'schema_version': 1,
            'plan_id': plan['plan_id'], 'plan_revision': plan['plan_revision'],
            'plan_sha256': hashlib.sha256(_canonical(plan)).hexdigest(),
            'skill': plan['skill'],
            'coverage': {'cells_met': sum(cell['status'] == 'met' for cell in cells),
                         'cells_total': len(cells), 'accepted_groups': len(accepted_groups),
                         'submitted_attempts': len(evidence), 'accepted_attempts': len(accepted_ids),
                         'excluded_attempts': len(evidence) - len(accepted_ids),
                         'all_cells_met': not gaps},
            'cells': cells, 'gaps': gaps, 'next_capture': next_capture,
            'benchmark': {**benchmark, 'executed': False, 'qualified': False},
            'outcomes': outcome_stats, 'warnings': warnings, 'exclusions': sorted(exclusions),
            'recognition': {'policy': 'Proposed recognition for reviewed, useful contributions; no automatic awards, points or cash.',
                            'accepted_contributions': len(accepted_groups),
                            'eligible_evidence_records': len(accepted_ids),
                            'distinct_declared_groups': len(accepted_groups),
                            'payments_active': False, 'awards_issued': False,
                            'note': 'Recognition candidates are counted once per connected source group, not per alias or cut. These counts do not measure causal training benefit. Campaign terms and any attribution require a separate explicit agreement.'},
            'robot_skill_acquired': False, 'physical_robot_ready': False}


def example_plan():
    """Fabricated planner example; no media, independent evidence or run implied."""
    cases = [
        {'id': 'nominal', 'label': 'Nominal placement', 'brief': 'Show one complete rigid-object placement, from initial state through release and stable final position.', 'outcome': 'success', 'required_groups': 2, 'priority': 6},
        {'id': 'new-instance', 'label': 'Different physical object', 'brief': 'Use another physical instance within the supported size and material envelope. Preserve its stable object identity.', 'outcome': 'success', 'required_groups': 2, 'priority': 8},
        {'id': 'empty-grasp', 'label': 'Empty grasp', 'brief': 'Record an observed empty closure with the jaws and object visible, preserving the failed outcome. Do not label commanded closure as holding.', 'outcome': 'failure', 'required_groups': 1, 'priority': 9},
        {'id': 'recovery', 'label': 'Recovery after missed grasp', 'brief': 'Capture the missed grasp, corrective attempt and final observed result in one complete episode.', 'outcome': 'recovery', 'required_groups': 2, 'priority': 10},
        {'id': 'occlusion', 'label': 'Brief visibility loss', 'brief': 'Capture a complete placement with a documented short visibility interruption and observable final state. Retain uncertainty where contact is not visible.', 'outcome': 'success', 'required_groups': 2, 'priority': 7},
        {'id': 'rotated-start', 'label': 'Rotated starting object', 'brief': 'Vary the initial object orientation within the supported envelope, keeping the whole attempt and final outcome visible.', 'outcome': 'success', 'required_groups': 2, 'priority': 5},
    ]

    def attempt(identity, case_id, digest, *, split='train', review='accepted', origin='human', rights='approved'):
        return {'id': 'demo-' + identity, 'plan_revision': 1, 'case_id': case_id,
                'media_sha256': digest * 64, 'source_recording_id': 'demo-recording-' + identity,
                'session_id': 'demo-session-' + identity, 'object_instance_id': 'demo-object-' + identity,
                'outcome': next(case['outcome'] for case in cases if case['id'] == case_id),
                'complete': True, 'review_state': review, 'review_origin': origin,
                'rights_status': rights, 'split': split}

    return validate_plan({'format': FORMAT, 'schema_version': 1, 'plan_id': 'demo-rigid-placement',
                          'plan_revision': 1,
                          'skill': {'name': 'Rigid object placement · planning example',
                                    'goal': 'Move one supported rigid box onto a marked target and verify stable release.',
                                    'operating_envelope': 'Authored six-DOF arm with parallel jaws and bounded rigid-box table scene. This is a capture-planning example; shoes, deformables and physical hardware are outside the current adapter.'},
                          'cases': cases,
                          'benchmark': {'status': 'specified',
                                        'environment': 'Proposed contact-pick-place-box-v0 simulation suite; no new suite is executed by this planner.',
                                        'reason': 'Descriptor only. Fresh frozen scenarios and artifact-bound results are still required.',
                                        'success_predicate': 'Observe grasp, lift, transfer, release and target stability using the independent native task gate; count every attempted trial.',
                                        'failure_cases': ['Empty grasp', 'Slip or drop', 'Unreachable start', 'Insufficient jaw force', 'Observation loss', 'Matched open-jaw negative control']},
                          'evidence': [attempt('nominal-a', 'nominal', 'a'), attempt('nominal-b', 'nominal', 'b'),
                                       attempt('instance-a', 'new-instance', 'c', split='dev'),
                                       attempt('failure-a', 'empty-grasp', 'd'),
                                       attempt('recovery-proposal', 'recovery', 'e', review='proposed', origin='ai'),
                                       attempt('occlusion-pending', 'occlusion', 'f', review='proposed', rights='unknown')]})


def _output(value, path):
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if path is None:
        print(text, end='')
    else:
        with Path(path).open('x', encoding='utf-8') as stream:
            stream.write(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Inspect declared Skillspace capture coverage; no training, benchmark execution or rewards.')
    commands = parser.add_subparsers(dest='command', required=True)
    example = commands.add_parser('example', help='Emit fabricated demonstration metadata.')
    example.add_argument('--output', type=Path, help='Create a new JSON file; existing files are not overwritten.')
    inspect = commands.add_parser('inspect', help='Inspect one bounded coverage-plan JSON file.')
    inspect.add_argument('plan', type=Path)
    inspect.add_argument('--output', type=Path, help='Create a new summary JSON file.')
    args = parser.parse_args(argv)
    try:
        if args.command == 'example':
            value = example_plan()
        else:
            with args.plan.open('rb') as stream:
                value = summarize_plan(stream.read(MAX_PLAN_BYTES + 1))
        _output(value, args.output)
    except (CoverageError, OSError) as exc:
        print('Coverage planning failed: ' + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

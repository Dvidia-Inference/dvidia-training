#!/usr/bin/env python3
"""Reconstruct named critical-contact pairs from saved PiPER grip evidence.

No model loading, training, simulator integration or network access occurs.
Pair counts are native ticks with >.01 N normal force, not independent events.
The classification repeats the frozen task's explicit critical-pair rules;
the independent audit binds the underlying numeric simulator evidence.
"""
from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Complete jaw-only study folder with passed audit.json.')
    parser.add_argument('--output', type=Path, required=True, help='Fresh output JSON file; never overwritten.')
    args = parser.parse_args(argv)
    root = args.run
    report, audit, frozen = (read(root/name) for name in ('report.json', 'audit.json', 'frozen.json'))
    if (audit.get('status') != 'passed' or audit.get('report_sha256') != digest(root/'report.json')
            or audit.get('frozen_sha256') != digest(root/'frozen.json')
            or report.get('frozen_sha256') != digest(root/'frozen.json')):
        raise ValueError('A passed audit must bind the exact complete report and frozen study.')
    planned = report['variants']['feedback']['scheduled']
    rows = [row for row in report['rows'] if row['variant'] == 'feedback']
    if len(rows) != planned or len({row['case_id'] for row in rows}) != planned:
        raise ValueError('Feedback report roster is incomplete or duplicated.')
    cases, pair_summary = [], {}
    finger_cases = {'left': set(), 'right': set()}
    critical_cases = set()
    for row in rows:
        case_id = row['case_id']
        if type(case_id) is not str or not case_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in case_id):
            raise ValueError('Unsafe case ID.')
        directory = root/'confirmation'/case_id/'feedback'
        with gzip.open(directory/'episode.json.gz', 'rt') as stream:
            episode = json.load(stream)
        metadata = read(directory/'native.json')
        if episode.get('case_id') != case_id or episode.get('policy_kind') != 'feedback':
            raise ValueError('Saved episode identity differs from the report roster.')
        contact = metadata['contact_names']
        geoms = {geom['geom_id']: geom for geom in contact['geoms']}
        pads = set(sum(contact['pad_geom_ids'].values(), []))
        robot = {i for i, geom in geoms.items() if 1 <= geom['body_id'] <= 9}
        moving = {i for i in robot if geoms[i]['body_id'] > 1}
        arm = {i for i in robot if 2 <= geoms[i]['body_id'] <= 7}
        block, table, floor = (contact[name] for name in ('block_geom_id', 'table_geom_id', 'floor_geom_id'))
        selected = []
        for key, ticks in episode['metrics']['contact_pair_native_ticks'].items():
            pair = set(map(int, key.split(':')))
            if len(pair) != 2 or not pair <= set(geoms) or type(ticks) is not int or not 0 <= ticks <= 20000:
                raise ValueError('Invalid native contact-pair counter.')
            if block in pair:
                critical = bool(pair & (robot-pads)) or floor in pair
            elif pair & {table, floor} and pair & moving:
                critical = True
            else:
                critical = bool(pair & arm and pair <= robot)
            if not critical or ticks == 0:
                continue
            side = next((side for side, pad_ids in contact['pad_geom_ids'].items()
                         if table in pair and pair & set(pad_ids)), None)
            kind = 'finger_table' if side is not None else 'other_frozen_task_critical_pair'
            if side is not None:
                finger_cases[side].add(case_id)
            named = [dict(geoms[i]) for i in sorted(pair)]
            selected.append(dict(geom_ids=sorted(pair), geoms=named, kind=kind, finger_side=side,
                                 pair_native_ticks=ticks))
            label = ':'.join(map(str, sorted(pair)))
            total = pair_summary.setdefault(label, dict(geom_ids=sorted(pair), geoms=named,
                kind=kind, finger_side=side, case_ids=[], pair_native_ticks=0))
            if total['geoms'] != named:
                raise ValueError('Native geometry naming changed between cases.')
            total['case_ids'].append(case_id)
            total['pair_native_ticks'] += ticks
        saved_critical = row['score']['native_recomputed']['unexpected_contact_native_ticks']
        if bool(selected) != bool(saved_critical):
            raise ValueError('Critical-pair classification disagrees with the reconstructed native task.')
        if selected:
            critical_cases.add(case_id)
        files = ('episode.json.gz', 'native.json', 'native.npz', 'score.json')
        cases.append(dict(case_id=case_id, condition=row['condition'], success=row['score']['success'],
            unexpected_contact_native_ticks=saved_critical, critical_pairs=selected,
            inputs={str((directory/name).relative_to(root)): digest(directory/name) for name in files}))
    union = finger_cases['left'] | finger_cases['right']
    intersection = finger_cases['left'] & finger_cases['right']
    result = dict(schema='dvidia.piper-grip-clearance-analysis/v1',
        script_sha256=digest(Path(__file__)),
        inputs={name: digest(root/name) for name in ('report.json', 'audit.json', 'frozen.json', 'protocol.json')},
        frozen_task_source_sha256=frozen['runtime_sha256']['piper_task.py'],
        physics_run=False, model_retrained=False, network=False, variant='feedback',
        scheduled=planned, critical_contact_cases=len(critical_cases),
        left_finger_table_cases=len(finger_cases['left']), right_finger_table_cases=len(finger_cases['right']),
        finger_table_union_cases=len(union), both_fingers_table_cases=len(intersection),
        all_critical_cases_explained_by_finger_table=critical_cases == union,
        pair_summary=pair_summary, cases=cases,
        scope='Named native contact-pair counters under the frozen task rules, not a physics rerun or closest-distance/mesh-accuracy validation.',
        counting='Pair counts are native ticks with normal force >.01 N. Different pairs can overlap at one native tick; sums are not independent collision events.',
        interpretation='Finger-table contact motivates testing reference-path clearance separately from acquisition timing. This diagnostic alone does not establish a corrected path or a successful recovery.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('scheduled', 'critical_contact_cases',
        'left_finger_table_cases', 'right_finger_table_cases', 'finger_table_union_cases',
        'both_fingers_table_cases', 'all_critical_cases_explained_by_finger_table')}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

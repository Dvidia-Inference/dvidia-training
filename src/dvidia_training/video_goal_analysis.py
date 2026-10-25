"""Post-run error-budget diagnosis; does not replace the frozen primary score."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from .video_goal_benchmark import inspect_run, score_reference, _sha


def analyze(root):
    root = Path(root)
    verified = inspect_run(root)
    report = json.loads((root/'report.json').read_text())
    rows = []
    for source in report['sources']:
        trial = next(a for a in report['attempts'] if a['case_id'] == source['id'] and a['variant'] == 'video_goal')
        row = dict(id=source['id'], primary_task_success=trial['success'], error_budget=None,
                   secondary_destination_replay_success=False, inference_ms=None)
        if source['proposal'] is not None:
            row['inference_ms'] = source['proposal']['timing_ms']['total']
        if trial['episode_file'] and source['proposal']['status'] == 'proposed':
            episode = json.loads(gzip.decompress((root/trial['episode_file']).read_bytes()))
            task = np.array(source['task_goal_xy_m'])
            demo = np.array(source['source_final_xy_m'])
            vision = np.array(source['proposal']['goal_xy_m'])
            final = np.array(episode['trajectory'][-1]['object_position_m'][:2])
            vectors = dict(demonstration=demo-task, perception=vision-demo, execution=final-vision, total=final-task)
            row['error_budget'] = {key: dict(norm_mm=float(np.linalg.norm(value)*1000),
                xy_mm=(value*1000).tolist()) for key, value in vectors.items()}
            row['secondary_destination_replay_success'] = score_reference(episode, demo.tolist())['success']
        rows.append(row)
    return dict(schema='dvidia.video-goal-error-analysis/v1', report_sha256=_sha(root/'report.json'),
        integrity_sha256=_sha(root/'integrity.json'), validation=verified, layouts=rows,
        primary_task_successes=sum(r['primary_task_success'] for r in rows),
        secondary_destination_replay_successes=sum(r['secondary_destination_replay_success'] for r in rows),
        scheduled_layouts=len(rows),
        interpretation='Primary scores retain the intended task center frozen before generation. '
            'The exploratory secondary score instead asks whether execution matches the observed source endpoint. '
            'The oracle receives the exact intended center; the video only sees an imperfect demonstrated outcome. '
            'Offset vectors add; their norms generally do not. Components diagnose accumulated offsets, not their physical cause.',
        motor_policy_learned=False, physical_robot_ready=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path); parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    result = analyze(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(primary_task_successes=result['primary_task_successes'],
        secondary_destination_replay_successes=result['secondary_destination_replay_successes'],
        scheduled_layouts=result['scheduled_layouts'])))


if __name__ == '__main__': main()

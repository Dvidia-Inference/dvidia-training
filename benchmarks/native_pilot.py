"""Reproduce synthetic native recordings, training and one exact-scene test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from dvidia_training.footage_example import create_example
from dvidia_training.footage_pipeline import _fresh, inspect_run, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from dvidia_training.arm_entrypoints import require_arm
    require_arm()
    from dvidia_training.capsule_installer import install_capsule, capsule_worker
    output = _fresh(args.output)
    create_example(output / 'source', clips=10, native_arm=True)
    start = time.perf_counter()
    trained = run(output / 'source', output / 'trained', task_source=output / 'source' / 'task.skill.json')
    elapsed = time.perf_counter() - start
    inspect_run(output / 'trained')
    dataset = json.loads((output / 'trained' / 'dataset' / 'dataset.json').read_bytes())
    test = next(row for row in dataset['episodes'] if row['split'] == 'test')
    index = int(test['id'].split('-')[-1])
    scene = {'object_position': [.34 + .013 * (index % 6), -.11 + .03 * (index % 7), .31],
             'target_position': [.44 + .012 * (index % 6), .12 - .021 * (index % 7), .31],
             'object_size': [.04, .04, .04], 'object_mass': .04, 'object_friction': .8,
             'jaw_force_limit': 15., 'joint_torque_limit': 40., 'seed': 8000 + index}
    (output / 'heldout-scene.json').write_text(json.dumps(scene, indent=2) + '\n')
    capsule = (output / 'trained' / 'candidate.skill-capsule.json').read_bytes()
    installation = install_capsule(capsule, output / 'installs')
    result = capsule_worker(output / 'installs', installation['installation_id'], scene,
                            output / 'qualification', qualify=True)
    receipt = result['local_qualification']
    document = {'kind': 'dvidia.synthetic-native-training-pilot', 'schema_version': 1,
                'scope': 'simulation-only', 'physical_robot_ready': False,
                'pipeline_wall_seconds': elapsed, 'pipeline_report_seconds': trained['elapsed_seconds'],
                'counts': trained['counts'], 'movement_test': trained['movement']['test'],
                'movement_training': trained['movement']['training'],
                'qualification': receipt,
                'limitations': ['One exact held-out synthetic recording/scene; no general skill or hardware qualification.',
                                'Movement labels come from native state and authored targets, not inferred human actions.',
                                'Gripper, phases and recovery remain authored; camera perception is not used for control.']}
    (output / 'pilot.json').write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
    print(json.dumps(document, indent=2, allow_nan=False))
    if not receipt['student']['success'] or receipt['control']['success']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

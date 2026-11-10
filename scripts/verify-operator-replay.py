#!/usr/bin/env python3
"""Independently execute recorded issued commands; compare native evidence.

No teacher, camera rendering, fitting or physical commands. A failed task may
replay exactly: parity is not placement success or skill qualification.
"""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from dvidia_training import operator_capture as capture
from dvidia_training import camera_action_capture as camera
from dvidia_training import piper_action_capture as files
from dvidia_training.piper_feedback_env import FeedbackPiperTaskEnv
from dvidia_training.piper_feedback_score import score_feedback_episode
from dvidia_training.piper_task import PiperTaskConfig


def verify(assets,directory):
    root=Path(directory)
    inspection=capture.inspect_capture(root)
    recording=capture.load_capture(root)
    report=capture._read(root,'capture.json')
    original=capture._read(root,'native_episode.json')
    if report['uncommitted_native_intervals']:
        raise ValueError('Only committed-prefix native evidence can be replayed by this verifier.')
    config=PiperTaskConfig(**original['config'])
    env=FeedbackPiperTaskEnv(assets,config)
    began=time.perf_counter()
    observation,info=env.reset(report['case']['seed'])
    states=[deepcopy(observation)];actions=[]
    for step in recording['steps']:
        action=deepcopy(step['command']['issued'])
        action_row=dict(t_seconds=step['before']['timestamp_seconds'],**action,phase=report['source_kind'])
        states[-1].update(issued_joint_targets_rad=action['joint_targets_rad'],
            issued_gripper_position_m=action['gripper_position_m'],command=action_row)
        observation,info=env.step(action)
        actions.append(action_row);states.append(deepcopy(observation))
    replay=camera._native_episode(SimpleNamespace(info=info,env=env,task_config=config),states,actions)
    arrays=env.native_arrays();metadata=env.native_metadata()
    comparison={};values=0;maximum=0.;exact=True
    with np.load(root/'native.npz',allow_pickle=False) as saved:
        if set(saved.files)!=set(arrays):raise ValueError('Native evidence field inventory differs.')
        for key,actual in arrays.items():
            expected=saved[key]
            if actual.shape!=expected.shape:raise ValueError('Native evidence shape differs: '+key)
            error=float(np.max(np.abs(actual-expected))) if actual.size else 0.
            matched=bool(np.array_equal(actual,expected))
            comparison[key]=dict(shape=list(actual.shape),max_abs_error=error,exact=matched)
            values+=actual.size;maximum=max(maximum,error);exact=exact and matched
    score=score_feedback_episode(replay,arrays,metadata,report['case']['goal_position_m'][:2])
    matches_score=score==report['score']
    return dict(schema='dvidia.operator-native-replay/v1',exact_native_arrays=exact,
        native_values_compared=int(values),maximum_abs_error=maximum,fields=comparison,
        independent_score_matches=matches_score,score=score,
        original_strict_success=inspection['strict_success'],control_intervals=len(recording['steps']),
        simulated_seconds=env.elapsed,wall_seconds=time.perf_counter()-began,
        recording_sha256=sha256((root/'recording.json').read_bytes()).hexdigest(),
        capture_integrity_sha256=sha256((root/'integrity.json').read_bytes()).hexdigest(),
        source_kind=report['source_kind'],physical_robot_ready=False,teacher_queries=0,
        scope='CPU native command replay including reset; excludes camera, Godot, fitting and hardware. '
              'Native parity is separate from successful task completion.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',required=True,type=Path)
    parser.add_argument('--capture',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    result=verify(args.assets,args.capture)
    files._json(args.output,result)
    print(json.dumps({key:result[key] for key in ('exact_native_arrays','native_values_compared',
        'maximum_abs_error','independent_score_matches','control_intervals','simulated_seconds')},indent=2))
    return 0 if result['exact_native_arrays'] and result['independent_score_matches'] else 1


if __name__=='__main__':raise SystemExit(main())

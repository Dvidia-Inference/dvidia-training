"""Reproduce the frozen, bounded first-build qualification protocol."""
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
import sys
import time
import numpy as np
from .cli import NetworkGuard, payload_for, rollout, train, write_run
from .env import CableEndReachEnv, Config
from .pack import encode
from .policies import FeedbackPolicy


def numerical_probe():
    """A fixed-tape regression with independent settling, not physical accuracy."""
    coarse = CableEndReachEnv()
    fine = CableEndReachEnv(replace(Config(), timestep=.0005))
    first, _ = coarse.reset(seed=2)
    second, _ = fine.reset(seed=2)
    initial_delta = float(np.linalg.norm(np.asarray(first['endpoint_position'])-second['endpoint_position']))
    target_delta = float(np.linalg.norm(np.asarray(first['target'])-second['target']))
    initial = [np.asarray(first['endpoint_position']),np.asarray(second['endpoint_position'])]
    frames = []
    for i in range(12):
        force = FeedbackPolicy()(first)
        a = coarse.step(force)
        b = fine.step(force)
        frames.append({'control_step':i+1, 'action':force,
                       'coarse_position':a[0]['endpoint_position'], 'fine_position':b[0]['endpoint_position'],
                       'endpoint_delta_m':float(np.linalg.norm(np.asarray(a[0]['endpoint_position'])-b[0]['endpoint_position'])),
                       'displacement_delta_m':float(np.linalg.norm((np.asarray(a[0]['endpoint_position'])-initial[0])-(np.asarray(b[0]['endpoint_position'])-initial[1]))),
                       'coarse_valid':a[4]['valid'], 'fine_valid':b[4]['valid']})
        first, second = a[0], b[0]
    environment = CableEndReachEnv()
    observation, info = environment.reset(seed=2)
    length = info['parameters']['rope_length']
    errors = []
    for _ in range(30):
        observation, _, done, truncated, info = environment.step(FeedbackPolicy()(observation))
        points = np.asarray(observation['rope_points'])
        errors.append(abs(float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())-length))
        if done or truncated:
            break
    return {'scope':'Numerical regression only. Independent settling changes initial tip and goal; this is not matched-start convergence or physical accuracy.',
            'seed':2, 'physics_timesteps_seconds':[.001,.0005], 'initial_endpoint_delta_m':initial_delta,
            'target_delta_m':target_delta, 'frames':frames,
            'max_endpoint_delta_m':max(f['endpoint_delta_m'] for f in frames),
            'authored_rope_length_m':length, 'max_centerline_length_error_m':max(errors),
            'regression_limits':{'endpoint_delta_m':.002, 'centerline_length_error_m':1e-7},
            'passed':all(f['coarse_valid'] and f['fine_valid'] for f in frames) and max(f['endpoint_delta_m'] for f in frames)<.002 and max(errors)<1e-7}


def main():
    output = Path(sys.argv[1]) if len(sys.argv)>1 else Path('runs/release-v0')
    config = Config()
    base = FeedbackPolicy(force_limit=config.force_limit).parameters()
    args = Namespace(seeds=[0,1,2], eval_seeds=list(range(200,210)), search_seed=31415, iterations=2, population=4)
    started = time.perf_counter()
    with NetworkGuard(True) as guard:
        payload, selected = train(config,args,base,guard)
        comparisons = {}
        for name in ['feedback','zero','release']:
            episodes = [rollout(config,seed,base,name,trace=False) for seed in args.eval_seeds]
            comparisons[name] = payload_for(config,episodes,base,guard)
        numeric = numerical_probe()
        report = {'schema_version':1, 'task':'CableEndReach-v0', 'scope':'simulation-only',
                  'protocol':{'training_seeds':args.seeds, 'evaluation_seeds':args.eval_seeds,
                              'controller_frozen_before_evaluation':True,
                              'training':'9 candidates (baseline plus 2 iterations of 4), 3 training seeds each',
                              'baseline':'PD kp10, kd0.25, upward bias0.1N; norm cap2N',
                              'negative_controls':'zero force; baseline feedback for 10 control steps followed by zero force',
                              'success':'actual endpoint within25mm for200ms continuous dwell; horizon6s',
                              'prior_development_evaluation_seeds':[100,101,102,103,104]},
                  'trained_evaluation':payload['summary'], 'selected_parameters':selected,
                  'comparisons':comparisons, 'numerical_probe':numeric, 'network_guard':guard.record(),
                  'qualification_execution_wall_seconds_before_output':time.perf_counter()-started,
                  'physical_evidence':None, 'gpu_measurements':None,
                  'limitations':'Small authored profile; ideal endpoint force; privileged state; no grasp slip, axial elasticity, sensor model or calibration. No claim of universal coverage or physical transfer.'}
        if guard.attempts or not numeric['passed']:
            raise RuntimeError('Qualification execution failed its declared offline/numerical checks.')
        payload['qualification_protocol'] = report
        payload['execution_wall_seconds_before_output'] = time.perf_counter()-started
    write_run(output,payload,selected)
    (output/'qualification.json').write_bytes(encode(report))
    print('Frozen qualification recorded with all outcomes, including negative controls.')


if __name__ == '__main__':
    main()

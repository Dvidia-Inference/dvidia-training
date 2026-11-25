"""Run the six preregistered development-only actuator-fault trials.

The frozen policy and scorer are unchanged. The alternate session injects the
existing MuJoCo jaw pulse and measures the completed episode's native sidecar.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dvidia_training import piper_first_attempt_eval as study
from dvidia_training import piper_first_attempt_policy as policy
from dvidia_training.piper_feedback_env import FeedbackPiperTaskEnv, JawPulse


ON, OFF = 10.50, 12.50


def native_metrics(arrays):
    t = arrays['t_seconds']
    both = np.all(arrays['grasp_contact_counts'] > 0, axis=1)
    lifted = arrays['object_position_m'][:, 2] - arrays['object_position_m'][0, 2]
    in_air = arrays['table_object_contact_count'] == 0
    before = (t >= ON-.1-1e-10) & (t < ON-1e-10)
    loss = (~both) & (t >= ON)
    consecutive = np.convolve(loss.astype(int), np.ones(40, dtype=int), mode='valid')
    starts = np.flatnonzero(consecutive == 40)
    metrics = dict(held_100ms_before_fault=bool(np.all((both & (lifted >= .06) & in_air)[before])),
        measured_prefault_ticks=int(before.sum()), loss_start_native_s=None,
        loss_confirmed_native_s=None, tool_path_2s_after_confirmed_loss_m=None,
        fault_native_ticks=int(np.sum(arrays['pulse_active'])))
    if starts.size:
        first = int(starts[0])
        confirmed = first + 39
        last = confirmed + 2000
        if last >= len(t):
            raise ValueError('Fixed two-second post-loss analysis window exceeds horizon.')
        xyz = arrays['tcp_position_m'][confirmed:last+1]
        metrics.update(loss_start_native_s=float(t[first]),
            loss_confirmed_native_s=float(t[confirmed]),
            tool_path_2s_after_confirmed_loss_m=float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()))
    return metrics


def run(args, case, controller, plan):
    metrics = {}

    class ScheduledJawFault(study._MaterialSession):
        def _environment(self, task):
            pulse = JawPulse(start_s=ON, end_s=OFF, minimum_opening_m=.035)
            return FeedbackPiperTaskEnv(self.assets, self.task_config, pulses=(pulse,))

        def native_evidence(self):
            arrays, metadata = super().native_evidence()
            metrics.update(native_metrics(arrays))
            return arrays, metadata

    # Swap only the environment factory for this one episode. The controller
    # has no access to its native sidecar or any part of the fault schedule.
    with patch.object(study, '_MaterialSession', ScheduledJawFault):
        result = study.run_attempt(args.assets, case, controller,
            calibration=args.calibration, trace=True)
    decision = next((p['t_seconds'] for p in result['phases'] if p['phase']=='lost_grip'), None)
    confirmed = metrics.get('loss_confirmed_native_s')
    metrics.update(grip_loss_decision_s=decision,
        grip_loss_delay_s=None if decision is None or confirmed is None else decision-confirmed)
    result['supplemental_metrics'] = metrics
    result['supplemental_plan_sha256'] = sha256(args.plan.read_bytes()).hexdigest()
    result['supplemental_driver_sha256'] = sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def main():
    cli = argparse.ArgumentParser()
    for flag in ('plan','assets','calibration','output'):
        cli.add_argument('--'+flag, type=Path, required=True)
    args=cli.parse_args()
    plan=json.loads(args.plan.read_text())
    if sha256(args.calibration.read_bytes()).hexdigest()!=plan['calibrated_scene_sha256']:
        raise ValueError('Calibration changed.')
    if sha256(Path(policy.__file__).read_bytes()).hexdigest()!=plan['policy_sha256']:
        raise ValueError('Policy changed.')
    args.output.mkdir(exist_ok=False)
    for case in plan['cases']:
        for c in plan['controllers']:
            out=run(args,case,c,plan)
            path=args.output/(case['id']+'-'+c+'.json')
            with path.open('x') as f:
                json.dump(out,f,indent=2,sort_keys=True,allow_nan=False)
                f.write('\n')
            print(case['id'],c,out['status'],out['reason'],out['supplemental_metrics'],flush=True)


if __name__=='__main__':
    main()

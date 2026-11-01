"""Three bounded, authored native-PiPER holds after the unchanged evidence probe.

All inputs are privileged simulator observations. Position regulation and a
synthetic force ceiling do not implement physical stopping or force control.
The fixed parameters below are engineering assumptions, not tuned hardware
calibration. No full transport, squeeze escalation or requalification occurs.
"""
from __future__ import annotations

from copy import deepcopy
import math

from . import piper_probe as probe

SCHEMA = 'dvidia.piper-retention/v2'
VARIANTS = ('measured_joint', 'issued_joint', 'cartesian')
FORCE_CEILING_N = 10.
POSITION_ERROR_CEILING_M = .004
ORIENTATION_ERROR_CEILING_RAD = .05
INITIAL_BIAS_CEILING_RAD = .08
CORRECTION_CEILING_RAD = .03
CORRECTION_RATE_RAD_S = .10
JOINT_MARGIN_RAD = .02


def retention_contract():
    return {
        'schema': SCHEMA, 'variants': list(VARIANTS), 'physical_robot_ready': False,
        'observation_origin': probe.ORIGIN, 'control_dt_s': probe.DT,
        'acquisition': 'Unchanged ProbeStopPolicy evidence acquisition, including its bounded supported retry. All variants share the same prefix through the first admission observation.',
        'measured_joint': 'Original hold: latch measured six joint coordinates and the last issued jaw target at admission.',
        'issued_joint': 'Latch the preceding issued six joint targets and the same last issued jaw target at admission.',
        'cartesian': 'Latch measured TCP position/orientation and preceding issued targets. Fixed joint bias equals issued target minus pose_action at the admission pose. First command repeats the exact issued target; later pose_action updates receive that fixed bias.',
        'limits': {'per_finger_force_ceiling_n': FORCE_CEILING_N,
                   'tcp_position_error_ceiling_m': POSITION_ERROR_CEILING_M,
                   'tcp_orientation_error_ceiling_rad': ORIENTATION_ERROR_CEILING_RAD,
                   'initial_joint_bias_ceiling_rad': INITIAL_BIAS_CEILING_RAD,
                   'joint_correction_ceiling_rad': CORRECTION_CEILING_RAD,
                   'joint_correction_rate_rad_s': CORRECTION_RATE_RAD_S,
                   'native_joint_target_margin_rad': JOINT_MARGIN_RAD},
        'rate_scope': 'The extra correction slews at at most 0.002 rad per 20 ms per joint. Original native 1 ms target slew, servos, gains and force caps remain unchanged.',
        'refusal': 'Original eligibility loss, unexpected contact or observed pad force >10 N terminally requests measured joint hold and the frozen jaw target. Cartesian error, initial-bias, total-correction or native-margin exhaustion also refuses terminally; it never chases an already failed grip.',
        'invalid_measurements': 'Malformed, nonfinite or noncausal observations terminally refuse and raise without issuing a command. No measured-position stop is claimed from invalid measurements.',
        'final_observation': 'Repeat force, original eligibility and Cartesian pose-error checks. Do not solve or issue a hypothetical next control at the terminal observation.',
        'sampling': 'Control checks occur at 20 ms boundaries. Independent saved 1 ms native evidence must judge a continuous 5 s hold; missed between-boundary failures remain findings.',
        'qualifications': 'Engineering limits and the 10 N ceiling are authored synthetic assumptions, not damage limits or calibrated physical telemetry. Position targets are not physical stops or force limiters. No load/friction inference, model training or complete placement is implemented.',
    }


class RetentionPolicy(probe.ProbeStopPolicy):
    """Preserve acquisition while changing only the admitted hold request."""

    def __init__(self, environment, variant='measured_joint'):
        if type(variant) is not str or variant not in VARIANTS:
            raise ValueError('Unknown bounded retention variant')
        super().__init__(environment, variant='evidence')
        self.retention_variant = variant
        self.hold_reference, self.hold_records, self.input_refusals = None, [], []
        self.input_refused = False
        self.correction = [0.]*6
        self.calculation = {}
        self.final_inspection = False
        ranges = environment.model.actuator_ctrlrange
        if len(ranges) != 7:
            raise ValueError('Retention requires the native seven-actuator PiPER model')
        self.joint_limits = []
        for row in ranges[:6]:
            if len(row) != 2:
                raise ValueError('Malformed native arm control limits')
            low, high = (probe._number(float(value), 'native joint limit') for value in row)
            if high-low <= 2*JOINT_MARGIN_RAD:
                raise ValueError('Native joint interval cannot contain the unchanged margin')
            self.joint_limits.append([low, high])

    def _clock(self, observation):
        try:
            super()._clock(observation)
        except ValueError:
            self.input_refused = True
            self.admitted = self.qualified = False
            self.aborted, self.state = True, 'abort'
            self.status, self.abort_reason = 'invalid_measurement_refused', 'invalid_measurement_or_cadence'
            refusal = {'last_valid_observation_s': self.last_observed_s,
                       'requested_control_count': len(self.requested_controls),
                       'reason': self.abort_reason, 'command_issued': False,
                       'scope': 'No trusted current position exists; no new hold target was issued.'}
            self.input_refusals.append(refusal)
            raise
        if (self.status not in ('invalidated_hold', 'timed_out_hold')
                and max(observation['pad_normal_forces_n'].values()) > FORCE_CEILING_N):
            self.diagnostic = self._diagnose(observation)
            self._terminal_hold(observation, 'declared_force_ceiling_exceeded')

    def _event(self, kind, observation, reason):
        super()._event(kind, observation, reason)
        self.events[-1]['retention_variant'] = self.retention_variant

    def _pose_errors(self, observation):
        if self.hold_reference is None:
            return None, None
        reference = self.hold_reference
        position = math.dist(observation['tcp_position_m'], reference['position_m'])
        product = abs(sum(a*b for a, b in zip(observation['tcp_quaternion_wxyz'], reference['quaternion_wxyz'])))
        orientation = 2*math.acos(min(1., product))
        return position, orientation

    def _range_failure(self, targets, *, margin):
        return any(not low+margin-1e-12 <= value <= high-margin+1e-12
                   for value, (low, high) in zip(targets, self.joint_limits))

    def _refuse_bound(self, observation, reason):
        self.calculation['refused_bounds'] = reason
        return self._terminal_hold(observation, reason)

    def _admit(self, observation, kind):
        if self.last_requested is None:
            return self._refuse_bound(observation, 'missing_prior_issued_target')
        issued = probe._vector(self.last_requested['joint_targets_rad'], 'prior issued targets', 6)
        jaw = self._jaw_to_freeze(observation)
        self.hold_reference = {
            't_seconds': observation['t_seconds'], 'position_m': list(observation['tcp_position_m']),
            'quaternion_wxyz': list(observation['tcp_quaternion_wxyz']),
            'issued_joint_targets_rad': issued, 'measured_joint_position_rad': list(observation['joint_position_rad']),
            'frozen_jaw_target_m': jaw, 'raw_pose_joint_targets_rad': None, 'fixed_bias_rad': None,
        }
        if self.retention_variant == 'cartesian':
            try:
                raw = self.env.pose_action(self.hold_reference['position_m'], self.hold_reference['quaternion_wxyz'], jaw)
                raw = probe._vector(raw['joint_targets_rad'], 'admission pose action', 6)
            except (ValueError, TypeError, KeyError, ArithmeticError):
                return self._refuse_bound(observation, 'pose_solver_refused')
            bias = [target-current for target, current in zip(issued, raw)]
            self.hold_reference.update(raw_pose_joint_targets_rad=raw, fixed_bias_rad=bias)
            self.calculation = {'raw_pose_joint_targets_rad': raw, 'desired_correction_rad': [0.]*6,
                                'correction_rate_limited_axes': [], 'refused_bounds': None}
            if max(map(abs, bias)) > INITIAL_BIAS_CEILING_RAD+1e-12:
                return self._refuse_bound(observation, 'initial_bias_exhausted')
            if self._range_failure(issued, margin=JOINT_MARGIN_RAD):
                return self._refuse_bound(observation, 'native_joint_margin_exhausted')
        action = super()._admit(observation, kind)
        if self.retention_variant != 'measured_joint':
            # Explicitly reuse the prior vector: raw+bias can differ by rounding.
            self.stop_targets = list(issued)
            action = self._hold()
        self.events[-1]['retention_hold_reference'] = deepcopy(self.hold_reference)
        self.events[-1]['diagnostic']['decision'] = 'bounded_'+self.retention_variant+'_hold'
        return action

    def _hold_check(self, observation):
        action = super()._hold_check(observation)
        if not self.admitted or self.retention_variant != 'cartesian':
            return action
        position_error, orientation_error = self._pose_errors(observation)
        self.calculation = {'raw_pose_joint_targets_rad': None, 'desired_correction_rad': None,
                            'correction_rate_limited_axes': [], 'refused_bounds': None}
        if position_error > POSITION_ERROR_CEILING_M+1e-12:
            return self._refuse_bound(observation, 'tcp_position_error_exhausted')
        if orientation_error > ORIENTATION_ERROR_CEILING_RAD+1e-12:
            return self._refuse_bound(observation, 'tcp_orientation_error_exhausted')
        if self.final_inspection:
            return action
        reference = self.hold_reference
        try:
            raw = self.env.pose_action(reference['position_m'], reference['quaternion_wxyz'], self.frozen_jaw_target)
            raw = probe._vector(raw['joint_targets_rad'], 'retention pose action', 6)
        except (ValueError, TypeError, KeyError, ArithmeticError):
            return self._refuse_bound(observation, 'pose_solver_refused')
        # raw+bias-issued == raw-raw_at_admission; keep the fixed bias exactly.
        desired = [value-initial for value, initial in zip(raw, reference['raw_pose_joint_targets_rad'])]
        self.calculation.update(raw_pose_joint_targets_rad=raw, desired_correction_rad=desired)
        if max(map(abs, desired)) > CORRECTION_CEILING_RAD+1e-12:
            return self._refuse_bound(observation, 'joint_correction_exhausted')
        step = CORRECTION_RATE_RAD_S*probe.DT
        limited = [index for index, (wanted, prior) in enumerate(zip(desired, self.correction))
                   if abs(wanted-prior) > step+1e-12]
        correction = [prior+max(-step, min(step, wanted-prior)) for wanted, prior in zip(desired, self.correction)]
        targets = [baseline+value for baseline, value in zip(reference['issued_joint_targets_rad'], correction)]
        self.calculation['correction_rate_limited_axes'] = limited
        if self._range_failure(targets, margin=JOINT_MARGIN_RAD):
            return self._refuse_bound(observation, 'native_joint_margin_exhausted')
        self.correction, self.stop_targets = correction, targets
        self.diagnostic['decision'] = 'bounded_cartesian_pose_regulation'
        return self._hold()

    def _record_hold(self, observation, action, *, final=False):
        if self.hold_reference is None and self.status not in ('invalidated_hold', 'timed_out_hold'):
            return
        reference = self.hold_reference or {}
        position_error, orientation_error = self._pose_errors(observation)
        self.hold_records.append({
            't_seconds': observation['t_seconds'], 'final_observation': final, 'command_issued': not final,
            'status': self.status, 'admitted': self.admitted, 'variant': self.retention_variant,
            'reference_position_m': reference.get('position_m'), 'reference_quaternion_wxyz': reference.get('quaternion_wxyz'),
            'baseline_issued_joint_targets_rad': reference.get('issued_joint_targets_rad'),
            'joint_targets_rad': list(action['joint_targets_rad']) if action is not None else None,
            'frozen_jaw_target_m': self.frozen_jaw_target,
            'raw_pose_joint_targets_rad': deepcopy(self.calculation.get('raw_pose_joint_targets_rad')),
            'fixed_bias_rad': reference.get('fixed_bias_rad'),
            'desired_correction_rad': deepcopy(self.calculation.get('desired_correction_rad')),
            'correction_rad': list(self.correction) if self.admitted and self.retention_variant == 'cartesian' else None,
            'correction_rate_limited': bool(self.calculation.get('correction_rate_limited_axes')),
            'correction_rate_limited_axes': list(self.calculation.get('correction_rate_limited_axes', [])),
            'tcp_position_error_m': position_error, 'tcp_orientation_error_rad': orientation_error,
            'refused_bounds': self.calculation.get('refused_bounds'), 'terminal_reason': self.abort_reason,
        })

    def __call__(self, observation):
        if self.input_refused:
            raise ValueError('Retention input refusal is terminal; no further command may be issued')
        # A stopped boundary must not inherit a preceding boundary's IK solve
        # or rate-limiting flags and present them as current computations.
        self.calculation = {}
        action = super().__call__(observation)
        self._record_hold(observation, action)
        return action

    def observe_final(self, observation):
        if self.input_refused:
            raise ValueError('Retention input refusal is terminal; final state is unknown')
        self.calculation = {}
        self.final_inspection = True
        try:
            super().observe_final(observation)
        finally:
            self.final_inspection = False
        self._record_hold(observation, self.last_requested, final=True)
        return self.summary()

    def summary(self):
        result = super().summary()
        result.update(schema=SCHEMA, variant=self.retention_variant, acquisition_variant='evidence',
                      acquisition_contract=result['contract'], contract=retention_contract())
        result['retention'] = deepcopy({'variant': self.retention_variant, 'hold_reference': self.hold_reference,
                                        'hold_records': self.hold_records, 'input_refusals': self.input_refusals,
                                        'native_joint_control_limits_rad': self.joint_limits})
        return result

"""One fixed incremental pose hold, preserving the frozen v0.2 controls.

The existing pose solver returns measured joints plus an error step. This
candidate accumulates that step around the last target issued before admission.
All observations and limits have the same synthetic simulator scope as v0.2.
"""
from __future__ import annotations

from copy import deepcopy

from . import piper_probe as probe
from . import piper_retention as retention


SCHEMA = 'dvidia.piper-incremental-hold/v3'
VARIANT = 'incremental'
UPDATE_GAIN = 1.0
UPDATE_LAW = ('increment = pose_action(frozen_admission_pose) - current_measured_joints; '
              'proposal = previous_correction + 1.0 * increment; '
              'refuse unlimited proposal outside +/-0.03 rad; '
              'slew accepted correction by at most 0.002 rad per joint per 20 ms; '
              'target = admission_issued_target + accepted_correction')


def incremental_hold_contract():
    """The single preregistered candidate; no runtime gain selection."""
    base = retention.retention_contract()
    return {
        'schema': SCHEMA, 'variants': [VARIANT], 'candidate': VARIANT,
        'physical_robot_ready': False, 'observation_origin': probe.ORIGIN,
        'control_dt_s': probe.DT, 'update_gain_per_boundary': UPDATE_GAIN,
        'update_law': UPDATE_LAW, 'limits': deepcopy(base['limits']),
        'acquisition': base['acquisition'],
        'admission': 'Freeze the measured TCP pose and preceding issued target. Check the unchanged 0.08 rad initial bias and 0.02 rad joint margin. Repeat the preceding issued target exactly; initialize cumulative correction to zero.',
        'measured_joint_eligibility': 'At admission, subsequent updates and final observations, refuse measured joints outside the native 0.02 rad target margin before solving. This prevents absolute-target clipping from manufacturing a nonzero increment at zero pose error.',
        'anti_windup': 'Refuse the unlimited cumulative proposal before slew limiting. Commit correction only after all checks pass. Refusal is terminal and leaves accumulated correction unchanged.',
        'rate_scope': base['rate_scope'], 'invalid_measurements': base['invalid_measurements'],
        'refusal': 'Original grip, force, pose, initial-bias, joint-margin and timeout refusals remain terminal. A valid terminal observation latches measured joints and the frozen jaw; its fallback is not active incremental regulation.',
        'final_observation': 'Check force, original eligibility and pose error without solving, integrating or issuing another command.',
        'sampling': base['sampling'], 'qualifications': base['qualifications'],
    }


class IncrementalHoldPolicy(retention.RetentionPolicy):
    """Add one candidate without changing any original retention variant."""

    def __init__(self, environment):
        # Reuse validated native limits and unchanged evidence acquisition. The
        # hold-specific methods below never dispatch v0.2 Cartesian regulation.
        super().__init__(environment, variant='issued_joint')
        self.retention_variant = VARIANT

    def _event(self, kind, observation, reason):
        super()._event(kind, observation, reason)
        self.events[-1].update(
            variant=VARIANT, acquisition_variant='evidence', candidate=VARIANT,
            update_law=UPDATE_LAW, update_gain_per_boundary=UPDATE_GAIN,
            accumulated_correction_rad=list(self.correction),
            proposed_cumulative_correction_rad=deepcopy(self.calculation.get('desired_correction_rad')),
            pose_error_increment_rad=deepcopy(self.calculation.get('pose_error_increment_rad')),
            refused_bounds=self.calculation.get('refused_bounds'),
            active_regulation=self.admitted, final_observation=self.final_inspection)

    def _admit(self, observation, kind):
        if self.last_requested is None:
            return self._refuse_bound(observation, 'missing_prior_issued_target')
        issued = probe._vector(self.last_requested['joint_targets_rad'], 'prior issued targets', 6)
        jaw = self._jaw_to_freeze(observation)
        self.hold_reference = {
            't_seconds': observation['t_seconds'],
            'position_m': list(observation['tcp_position_m']),
            'quaternion_wxyz': list(observation['tcp_quaternion_wxyz']),
            'issued_joint_targets_rad': issued,
            'measured_joint_position_rad': list(observation['joint_position_rad']),
            'frozen_jaw_target_m': jaw, 'raw_pose_joint_targets_rad': None,
            'fixed_bias_rad': None,
        }
        if self._range_failure(observation['joint_position_rad'], margin=retention.JOINT_MARGIN_RAD):
            return self._refuse_bound(observation, 'measured_joint_margin_exhausted')
        try:
            raw = self.env.pose_action(self.hold_reference['position_m'],
                                       self.hold_reference['quaternion_wxyz'], jaw)
            raw = probe._vector(raw['joint_targets_rad'], 'admission pose action', 6)
        except (ValueError, TypeError, KeyError, ArithmeticError):
            return self._refuse_bound(observation, 'pose_solver_refused')
        bias = [target-current for target, current in zip(issued, raw)]
        self.hold_reference.update(raw_pose_joint_targets_rad=raw, fixed_bias_rad=bias)
        self.calculation = {
            'raw_pose_joint_targets_rad': raw, 'desired_correction_rad': [0.]*6,
            'correction_before_update_rad': list(self.correction),
            'correction_rate_limited_axes': [], 'refused_bounds': None,
            'correction_update_accepted': False,
        }
        if max(map(abs, bias)) > retention.INITIAL_BIAS_CEILING_RAD+1e-12:
            return self._refuse_bound(observation, 'initial_bias_exhausted')
        if self._range_failure(issued, margin=retention.JOINT_MARGIN_RAD):
            return self._refuse_bound(observation, 'native_joint_margin_exhausted')
        # Call the original acquisition admission directly, then preserve the
        # prior issued vector exactly (no roundoff through raw + bias).
        probe.ProbeStopPolicy._admit(self, observation, kind)
        self.stop_targets = list(issued)
        self.diagnostic['decision'] = 'bounded_incremental_hold'
        self.events[-1]['retention_hold_reference'] = deepcopy(self.hold_reference)
        self.events[-1]['diagnostic']['decision'] = self.diagnostic['decision']
        return self._hold()

    def _hold_check(self, observation):
        action = probe.ProbeStopPolicy._hold_check(self, observation)
        if not self.admitted:
            return action
        position_error, orientation_error = self._pose_errors(observation)
        self.calculation = {
            'raw_pose_joint_targets_rad': None, 'desired_correction_rad': None,
            'pose_error_increment_rad': None,
            'correction_before_update_rad': list(self.correction),
            'correction_rate_limited_axes': [], 'refused_bounds': None,
            'correction_update_accepted': False,
        }
        if position_error > retention.POSITION_ERROR_CEILING_M+1e-12:
            return self._refuse_bound(observation, 'tcp_position_error_exhausted')
        if orientation_error > retention.ORIENTATION_ERROR_CEILING_RAD+1e-12:
            return self._refuse_bound(observation, 'tcp_orientation_error_exhausted')
        if self._range_failure(observation['joint_position_rad'], margin=retention.JOINT_MARGIN_RAD):
            return self._refuse_bound(observation, 'measured_joint_margin_exhausted')
        if self.final_inspection:
            self.diagnostic['decision'] = 'check_incremental_hold_without_update'
            return action
        reference = self.hold_reference
        try:
            raw = self.env.pose_action(reference['position_m'], reference['quaternion_wxyz'],
                                       self.frozen_jaw_target)
            raw = probe._vector(raw['joint_targets_rad'], 'incremental pose action', 6)
        except (ValueError, TypeError, KeyError, ArithmeticError):
            return self._refuse_bound(observation, 'pose_solver_refused')
        increment = [target-current for target, current in zip(raw, observation['joint_position_rad'])]
        proposed = [prior+UPDATE_GAIN*step for prior, step in zip(self.correction, increment)]
        self.calculation.update(raw_pose_joint_targets_rad=raw,
                                pose_error_increment_rad=increment,
                                desired_correction_rad=proposed)
        # Check the unlimited accumulation first: slew must not conceal an
        # exhausted total bound. Failed proposals never modify the integrator.
        if max(map(abs, proposed)) > retention.CORRECTION_CEILING_RAD+1e-12:
            return self._refuse_bound(observation, 'joint_correction_exhausted')
        step = retention.CORRECTION_RATE_RAD_S*probe.DT
        limited = [index for index, (wanted, prior) in enumerate(zip(proposed, self.correction))
                   if abs(wanted-prior) > step+1e-12]
        correction = [prior+max(-step, min(step, wanted-prior))
                      for wanted, prior in zip(proposed, self.correction)]
        targets = [baseline+value for baseline, value in zip(reference['issued_joint_targets_rad'], correction)]
        self.calculation['correction_rate_limited_axes'] = limited
        if self._range_failure(targets, margin=retention.JOINT_MARGIN_RAD):
            return self._refuse_bound(observation, 'native_joint_margin_exhausted')
        self.correction, self.stop_targets = correction, targets
        self.calculation['correction_update_accepted'] = True
        self.diagnostic['decision'] = 'bounded_incremental_pose_regulation'
        return self._hold()

    def _record_hold(self, observation, action, *, final=False):
        count = len(self.hold_records)
        super()._record_hold(observation, action, final=final)
        if len(self.hold_records) == count:
            return
        record = self.hold_records[-1]
        record.update(
            candidate=VARIANT, update_law=UPDATE_LAW, update_gain_per_boundary=UPDATE_GAIN,
            measured_joint_position_rad=list(observation['joint_position_rad']),
            pose_error_increment_rad=deepcopy(self.calculation.get('pose_error_increment_rad')),
            correction_before_update_rad=list(self.calculation.get('correction_before_update_rad', self.correction)),
            proposed_cumulative_correction_rad=deepcopy(self.calculation.get('desired_correction_rad')),
            accumulated_correction_rad=list(self.correction),
            applied_cumulative_correction_rad=list(self.correction) if self.admitted and not final else None,
            correction_rad=list(self.correction) if self.admitted else None,
            correction_update_accepted=self.calculation.get('correction_update_accepted', False),
            active_regulation=self.admitted)
        if final:
            # Preserve the last issued target separately from this observation,
            # which deliberately has no new requested command.
            record['preceding_issued_joint_targets_rad'] = record['joint_targets_rad']
            record['joint_targets_rad'] = None

    def summary(self):
        result = super().summary()
        result.update(schema=SCHEMA, candidate=VARIANT, contract=incremental_hold_contract())
        result['retention'].update(
            candidate=VARIANT, update_law=UPDATE_LAW, update_gain_per_boundary=UPDATE_GAIN,
            accumulated_correction_rad=list(self.correction), active_regulation=self.admitted,
            final_controller_status=self.status)
        return result

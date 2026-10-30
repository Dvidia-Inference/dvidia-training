# Action demonstrations: preserve the motor evidence

The `dvidia.action-demonstration/v1` format records aligned observations and
explicit actuator targets for the PiPER simulation profile. Validation and
inspection use Python's standard library. They do not run a simulator, move a
robot, train a model, recover actions from video, or qualify a skill.

The first example,
[`authored-example.json`](../examples/action-demonstration-v01/authored-example.json),
is a synthetic metadata fixture. It is **not a corrective demonstration or a
camera dataset**, and contains no real pixels. Strict target export refuses
authored fixtures by default. Passing a format check does not turn this
example into evidence of successful manipulation.

## Preserve the actual PiPER coordinates

PiPER has six radian arm targets and one native finger-slide target in metres:
zero means closed and 0.035 means open. There are **seven native control
channels**. The two measured finger coordinates are coupled signed positions;
the command is not a measured full gripper gap.

The older [episode tape](episode-tape.md) describes a different authored arm
and two jaw controls. Its limits, eight native controls and full-gap convention
cannot be relabeled as PiPER data. Existing PiPER observations also lack camera
frames and jaw velocity. This recorder does not automatically convert them,
infer missing actions from images, or fill absent measurements with zeros.

## Record one causal control boundary

Create `Recorder(header, initial, initial_applied_targets)`, then call
`append(step)` for each recorded interval. Each step retains:

1. The **before** observation available to the decision.
2. The **requested** action, separately from the **issued** command after any
   declared correction, limiter or override.
3. The interval's **applied** final native seven-channel target.
4. The **after** measurement, which belongs to the response rather than the
   current decision's inputs.

Applied targets are commands, not measured joint positions or contact forces.
The final native target does not describe every intermediate 1 ms command or
certify contact peaks within a control interval. `initial_applied_targets`
means the producer's starting ramped position reference, not measured joint
positions or the compensated native control. Declared ranges may tighten the
packaged PiPER ranges; rehashing cannot widen them. Endpoint ramp changes must
respect declared rate bounds, which still cannot certify intermediate ticks.

Finish with an explicit `complete` or `interrupted` status, task result,
reason, and applicable scorer/evidence hashes. Success remains separate from
recording completeness. Missing, stale and partial evidence stays explicit;
incomplete records are not silently repaired into training pairs.
The recorder is in memory. Save an interrupted prefix explicitly if the
producer stops; this version is not a crash-safe capture service.

## Bind time, geometry and provenance

Headers bind the supported profile, scene, channel order and limits, together
with calibration and source identities. Keep related source recordings,
sessions, corrections and retries together when assigning dataset splits;
adjacent frames are not independent demonstrations.

Only the declared simulation clock is supported. Device and hardware clocks
are rejected. Signal availability must precede its decision; camera exposure
must also end before that decision. A frame captured earlier but received
afterward cannot become a causal input. Missing readings have null values and
times. Acquisition and frame identities remain ordered across missing gaps;
observations are never forward filled. Native timestamps are preserved with
1 ns comparison tolerance, including roundoff at the 20 s cap. Actual longer
recordings and intervals beyond the 20 ms tolerance are refused.

Camera references bind staged frame artifacts by hash. Optional local artifact
readback checks those bytes, not depth accuracy, tracker quality or physical
calibration. Profile, scene and calibration hashes establish internal
consistency; they cannot authenticate a real recording or validate its physics.

## Inspect before exporting targets

From a checkout with the package installed:

```sh
python -m dvidia_training.action_demonstration validate examples/action-demonstration-v01/authored-example.json
python -m dvidia_training.action_demonstration inspect examples/action-demonstration-v01/authored-example.json
python -m dvidia_training.action_demonstration targets --help
python -m dvidia_training.action_demonstration replay-check --help
```

`validate_demonstration(document, require_targets=False)` checks the data
contract. `load_demonstration` reads it; `write_demonstration` creates an
exclusive new file. `inspect_demonstration` summarizes it without changing it.

`learning_pairs` applies stricter completeness and fresh required-signal/camera
checks, then requires local frame-byte readback; CLI `targets` uses
`--artifacts`. Its standalone JSON retains profile/binding/calibration context,
source kind and outcome as **non-feature metadata**, plus ordered residual
units: six radians, then one finger-slide metre. Inputs are named explicitly;
current issued targets, applied controls and future state remain excluded.
Export preserves complete failure and unverified episodes as motor labels.
It does **not** select good corrections: the proposed experiment must separately
review successful operator segments and their scoring evidence before fitting.
Source declarations and hashes cannot establish that these are good actions.

`replay_check(reference, replayed)` compares **two recorded demonstrations**.
It does not perform a physics replay, execute targets or establish task
competence. Its strict checks refuse missing, stale or partial evidence. This
alignment tool prepares a future action-learning experiment; no new policy is
trained by recording, exporting or comparing these files.

## Installed Terminal command

DVIDIA Training 0.1.2 adds `dvidia-action-demo`, with the same `validate`, `inspect`, `targets` and `replay-check` commands as the module CLI. Validation/inspection are standard-library data operations; no GPU, simulator, network or robot connection is used. The base package installation still has its documented NumPy dependency. Real synchronized capture and reviewed corrective demonstrations remain separate work.

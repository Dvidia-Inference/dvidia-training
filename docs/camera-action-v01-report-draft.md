# Camera-to-action skill v0.1 — research report draft

**Status: diagnostic evidence only; student results are pending.** This document
does not claim a learned placement success rate, improvement over either
comparator, a qualified downloadable skill or physical robot readiness.

## Question and bounded experiment

Can a small local model learn a complete PiPER block-placement task from
synchronized RGB-D, measured robot state and explicit synthetic action labels?
The intended learned sequence is approach, grasp, lift, carry, release and
retreat. The study uses one simulated PiPER, its existing gripper, a known rigid
40 mm block and two fixed destinations. It does not recover motor actions from
ordinary human video.

All starts, controller assignments, training settings and acceptance gates were
registered before collection. The runtime is frozen at
`c9a7c55531533f60e24651a99d509f721e195051`; the protocol SHA-256 is
`576851e321034a0f6d3cb4d0818b3a5ca97975fcc6f2c7fa30b07ffd203a2d76`.

There are 72 fresh starting-position groups: 32 training, eight development and
32 confirmation. Both destinations remain together in their starting-position
group. Starts span x=0.287–0.311 m and y=−0.042–−0.018 m, with the configured
initial block height 0.082 m. This small interpolation domain is a deliberate
first proof, not general spatial manipulation. A 3 mm inset from the prior
fixed primitive's envelope buffers camera error; estimates are not clipped or
replaced with simulator truth.

| Registered stage | Budget | Role |
| --- | ---: | --- |
| Initial teacher diagnostic | 4 trajectories | Verify actual motion, initial camera acquisition and complete teacher placement |
| Training teacher demonstrations | 64 episodes | Two goals for each of 32 training starts |
| Development teacher demonstrations | 16 episodes | Prediction diagnostics; excluded from fitting and normalization |
| Mixed-control corrections | 32 episodes | One training-only round; teacher chosen with probability 0.5 |
| Model development | 64 rollouts | Initial/corrected × full/blind × 16 development cases |
| Frozen confirmation | 256 attempts | 64 cases each for learner, matched blind model, fixed primitive and teacher |
| Negative controls | 16 attempts | Blank initial RGB, invalid old initial timestamps, malformed extrinsics and forced-open gripper |

The maximum registered native workload is 452 attempts, with no replacement of
failed starts or hidden retry. A 20-second complete episode contains 501
320×240 RGB-D pairs at 25 Hz, 1,000 motor intervals at 50 Hz, and 20,000 native
physics ticks at 1 ms. Camera capture/availability timestamps are simulated
clocks; wall-clock processing is recorded separately. They do not establish
physical camera-to-actuator latency.

## Inputs, supervision and acceptance

The CPU NumPy model has two 64-unit hidden layers and predicts six absolute
joint targets plus one jaw target. Its 269 features contain an explicitly
historical initial visual anchor, destination, elapsed time, encoders, jaw
measurements, encoder-derived link6 body pose, previous applied servo references and
160 ms of causal history. Current missing object estimates have zero numerical
fills, an unavailable mask and their source frame's original age. The historical
anchor is never presented as a new live observation. The frozen recording names
the link6 body-origin pose `tcp_position_estimate_m` and
`tcp_quaternion_estimate_wxyz`; these legacy field names do not identify a
calibrated gripper contact point. The native task scorer instead uses a distal
site offset by 0.12003 m along link6’s local z axis. The independent auditor
reconstructs this explicit offset when comparing the two coordinate records.

Teacher phase, current teacher command, native object/contact truth and future
observations are excluded. Teacher commands are synthetic labels from a
privileged authored controller. In the mixed-control round, the one teacher
query per step is stored separately from the command actually executed; it is
not described as an operator correction or a proven recovery. The blind model
uses identical source episodes and initial context but has all live and
historical sensor feedback masked out. It retains past command history.

Each fit uses seed 17, 50 epochs, batch size 512 and Adam learning rate 0.003.
Normalization uses training data only, including explicit missing fills and
masks. Candidate selection uses development task success, critical contacts,
then prediction error normalized by the same physical action scales across
candidates. The matched blind model is taken from the selected stage.

Before training, every training start must have at least one successful usable
teacher demonstration, and each goal must have at least 28 successes. Final
qualification requires all of the following:

- At least 58/64 strict learned placements on the scheduled unseen cases.
- Positive 97.5% paired bootstrap intervals against both fixed and blind
  comparators, resampling whole starting-position groups with both goals paired.
- Positive comparisons also on at least 24 common admitted groups, preventing
  initial comparator refusals from being the only basis for improvement.
- No increase in the frequency of critical-contact episodes versus the fixed
  comparator; all negative controls behave as specified.
- Complete attempt accounting, independent native scoring, no privileged
  learner inputs and no teacher intervention during evaluation.

The teacher comparator has privileged simulation observations and is an upper
reference, not evidence of human-video learning. Installation alone cannot
award qualification. Failed candidates remain publishable research findings.

The fault definitions are deliberately narrow: `initial_missing` zeros the
initial RGB image while the physical object remains in the scene;
`initial_stale` sets capture and availability times to −0.2 s at episode time
zero; `altered_calibration` scales one rotation-matrix element by 1.1; and
`open_gripper` overrides issued jaw targets to 0.035 m. The first three exercise
input-contract refusal. They do not establish detection of a physically absent
object, valid-but-old frames or small valid rigid calibration errors.

## Completed diagnostic evidence

An independent reader verified **24,040 files**, exact camera/control/native
cadences, source and frame bindings, issued/query separation, camera-position
errors and independently recomputed task scores. All four teacher diagnostics
completed strict placement successfully. They represent **two development
starting positions, each with two goals**, not four independent starting groups.

| Diagnostic | Initial camera error | Available-pose error p95 | Available decisions | Longest missing stretch |
| --- | ---: | ---: | ---: | ---: |
| Development 00, goal 0 | 0.196 mm | 0.314 mm | 620/1,000 | 7.24 s |
| Development 00, goal 1 | 0.196 mm | 0.504 mm | 618/1,000 | 7.28 s |
| Development 01, goal 0 | 0.249 mm | 0.816 mm | 602/1,000 | 7.96 s |
| Development 01, goal 1 | 0.249 mm | 0.816 mm | 604/1,000 | 7.92 s |

Native object displacement reached 138.6–160.0 mm, confirming actual
manipulation rather than a static-camera or replay-only check. Errors above
are conditional on the tracker returning an estimate. Adjacent 50 Hz decisions
often reuse the same 25 Hz frame and are not independent tracking trials.

The major diagnostic finding is **where visibility disappears**. An offline
analysis of sealed teacher-phase and perception sidecars found zero available
object estimates during close, lift, transfer, place and release: **0/749 motor
decisions** across the four trajectories. Approach and final verification were
fully visible. Teacher phase was used only to analyze this evidence, never as
a model input.

Therefore sub-millimetre available-pose errors must not be described as tracking
throughout manipulation. During the contact-heavy sequence the proposed
learner must rely on measured robot state and its historical initial visual
context. This is a measured limitation of the frozen upright-block tracker.
The current experiment will test whether those remaining signals are enough;
it must not silently add camera views or substitute native object positions.

Evidence files retained for publication:

- `runs/camera-action-v01-proofs-20261101/diagnostic-audit.json`
- `runs/camera-action-v01-proofs-20261101/diagnostic-phase-coverage.json`
- The registered protocol, stage receipts and complete sealed diagnostic recordings.

## Pending results and interpretation

**Pending:** training-data quality, model fitting, initial and corrected
development rollouts, frozen confirmation, negative controls, measured model
size/fit cost/inference cost, and fresh-install execution. Fill these sections
only from sealed receipts after independent readback; do not infer learned
success from successful teacher demonstrations.

This study can establish a reproducible camera-and-action learning loop for a
bounded synthetic task. It cannot establish arbitrary video-to-skill transfer,
arbitrary-arm compatibility, real-time hardware operation, real-world depth or
contact calibration, shoe manipulation or physical qualification. A useful next
experiment after the fixed results is to measure a second camera or another
explicitly uncertain observation method during grasp and carry, where the
present tracker provides no object measurements.

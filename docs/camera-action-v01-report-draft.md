# Camera-to-action skill v0.1 — research report draft

**Status: teacher data admitted; initial development learners each completed 0/16 strict placements.**
The registered correction round is running. Frozen confirmation is pending.
No qualified downloadable skill or physical robot readiness is claimed.

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
five causal history samples at 0, 40, 80, 120 and 160 ms. Standardized
features are clipped to ±10 at inference. Current missing object estimates have zero numerical
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

![Native MuJoCo RGB frame at 10.00 simulated seconds, development start 00 and destination 0. The authored teacher controls the arm; the frozen tracker reports no object position in this frame. This is diagnostic evidence, not a learned-policy result.](https://research.dvidia.org/downloads/topics/physical-grounding/runs/2026-11-01-camera-to-action-v01/viewer/frames/development-00-goal0/0250.png)

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

[Inspect the real diagnostic camera frames and action traces](https://research.dvidia.org/downloads/topics/physical-grounding/runs/2026-11-01-camera-to-action-v01/viewer/index.html), [independent audit](https://research.dvidia.org/downloads/topics/physical-grounding/runs/2026-11-01-camera-to-action-v01/viewer/diagnostic-audit.json), [phase coverage analysis](https://research.dvidia.org/downloads/topics/physical-grounding/runs/2026-11-01-camera-to-action-v01/viewer/diagnostic-phase-coverage.json) and [registered protocol](https://research.dvidia.org/downloads/topics/physical-grounding/runs/2026-11-01-camera-to-action-v01/viewer/protocol.json). The compact viewer contains diagnostic evidence only. It preserves every motor decision and copies native RGB frames byte-for-byte at 2.5 Hz from the original 25 Hz capture. Its display sampling is distinct from the camera cadence used by the controller. Full raw recordings remain separate from this compact diagnostic subset.

## Completed demonstration and data-quality evidence

All **80/80 registered teacher demonstrations** completed strict placement:
64/64 training episodes and 16/16 development episodes. The development
recordings remain excluded from fitting and normalization. All 32 training
starting-position groups have usable successful demonstrations with an
available, bound initial camera estimate. Each training destination has
**32 successful usable episodes**, exceeding the registered minimum of 28;
the sealed teacher-data gate passed with no missing training group.

| Teacher evidence | Strict successful / scheduled | Data use |
| --- | ---: | --- |
| Initial diagnostics | 4/4 | Initial camera and task checks; not additional fitting episodes |
| Training demonstrations | 64/64 | Issued-action supervision for 32 training starts × two goals |
| Development demonstrations | 16/16 | Held-out prediction and missing-observation diagnostics |

An independent readback passed with **504,840 files verified and 84 native task
scores recomputed** across the four diagnostics plus 80 demonstrations. This
count includes the earlier four diagnostics; it is not 84 additional episodes.
The audit reports no errors and executes no physics, estimator or learned
model. Its protocol, sealed demonstration receipt and stage-integrity identities
match the original registration and saved bytes.

The retained `demonstrations-audit.json` proof has SHA-256
`3405d789c0b79066d142d476f1b45468463d41516350d7c80766881a6cecf753`;
its evidence-inventory SHA-256 is
`319074d85fcd84758d3cc464e6040f2032832179d354c4adcb86f300b743f811`.
This is an audit of the diagnostic and demonstration stages only: no fitting
stage or student rollout is covered by this proof. Successful teacher actions
establish usable synthetic supervision, not successful imitation or transfer.

## Measured host context

The retained `hardware-native.json` describes a shared local **Apple M5** host
(`Mac17,4`, arm64) with **10 physical / 10 logical CPU cores and 24 GiB of
installed memory**. The environment is Darwin 25.5.0, Python 3.12.9, NumPy 2.5.3
and MuJoCo 3.15.0, with one native worker. Native physics and the policy use the
CPU; rendering uses the existing macOS CGL graphics path. No CUDA requirement
is introduced. The hardware-context file SHA-256 is
`fdc8007b27219c7c91dc600ae54a95f4a8abc64565db652002faa0f5a1e9f6d0`.

Installed memory is host capacity, not the model's measured memory requirement.
This shared-host setup does not establish minimum hardware, dedicated-host
throughput, other-platform compatibility or physical sensor-to-actuator
latency. Inference distributions and full-loop wall-clock measurements remain
pending the completed study and independent readback. Initial fit costs below
are separate from the much longer rendering, recording and verified export.

## Initial learning result: prediction is not execution

The sealed initial fitting stage contains 64,000 training samples and 16,000
held-out teacher-observation samples. The full learner fitted in 6.56 seconds
and occupies 521,715 bytes; the matched blind learner fitted in 5.99 seconds
and occupies 517,296 bytes. Shared byte-verified data export took a further
685.44 seconds. These are measured offline wall-clock durations on the shared
host, not video-to-skill completion times. Process peak RSS reached 643,891,200
bytes after the full fit and 789,200,896 bytes after the blind fit; the second
value is a process high-water mark, not the blind model's isolated memory use.

Both initial candidates completed all 16 registered development attempts and
made **zero strict placements**. The full learner's held-out action-scale MSE
on teacher observations was 0.0001133, yet its closed-loop success was 0/16.
The blind learner also scored 0/16. One-step prediction on a demonstrated
state does not establish execution after the model's own actions change it.

These are intermediate development results, not frozen confirmation. The
single preregistered mixed-control round continues unchanged. No replacement
starts, new hyperparameters or extra correction round have been introduced.
Final independent full-study readback must bind these sealed receipts before
publication as completed benchmark evidence.

**Pending:** corrected fitting and development, frozen confirmation, negative
controls, final inference and full-loop timings, and fresh-install execution.
The candidate selection and acceptance thresholds remain fixed.

This study can establish a reproducible camera-and-action learning loop for a
bounded synthetic task. It cannot establish arbitrary video-to-skill transfer,
arbitrary-arm compatibility, real-time hardware operation, real-world depth or
contact calibration, shoe manipulation or physical qualification. A useful next
experiment after the fixed results is to measure a second camera or another
explicitly uncertain observation method during grasp and carry, where the
present tracker provides no object measurements.

## Source and reuse

[The frozen learning source](https://github.com/Dvidia-Inference/dvidia-training/commit/c9a7c55531533f60e24651a99d509f721e195051) and the [DVIDIA Training repository](https://github.com/Dvidia-Inference/dvidia-training) contain the capture, mask-aware exporter, CPU model and study runner. Model/data releases and the completed study report are pending; no download here is labeled a qualified skill.

Original DVIDIA code and generated records use MIT; PiPER and simulator notices retain their upstream terms. The viewer includes actual native renderer output and no private human footage, credentials, model weights or redistributed robot meshes. Research writing follows this site's CC BY 4.0 license.

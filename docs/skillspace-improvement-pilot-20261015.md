# Skillspace improvement pilot — 15 October 2026

**The targeted addition did not improve task completion over the original
baseline.** Both completed 12/12 nominal scenes and 3/4 actuator-stress scenes.
The generic addition completed 9/12 and 2/4 respectively. This demonstrates a
working, measured comparison and a negative result for the proposed improvement;
it does not establish that targeted collection generally improves learning, or
that adding generic data generally hurts it.

The completed local run is `runs/improvement-experiment-20261015`. Its immutable
receipts are the evidence for the numbers below. This document interprets those
receipts without rerunning or changing the frozen experiment.

A [compact receipt archive](../benchmarks/improvement-20261015/README.md) is
included for the local viewer. It preserves original JSON bytes and hashes, but
does not include raw traces, media or learned weights for independent replay.

## What was tested

The task was the **simplified native rigid-box placement movement head**. Training
used existing simulated recordings with aligned numerical actions: six joint
targets in radians and symmetric gripper width in metres. The runtime still
uses privileged simulator state and an authored phase/contact/gripper/recovery
supervisor. It is not a policy learned from ordinary human video, a test of the
newer PiPER/UR5e controller, or evidence of physical-robot readiness.

The hypothesis came from the already-inspected historical `sample-count-19047`
nominal `scene-04`: the four-group candidate hit its time limit while its paired
authored teacher succeeded. A nearer unused training layout might help. The
selection distance was Euclidean distance over initial object XYZ and target
XYZ, not a measure of contact quality or a causal diagnosis.

Membership and selection provenance were frozen before fitting, and all three
models were frozen before evaluation. All used training seed 19047, the same
learning recipe, and the same development sources for regularization selection.
The new evaluation seed was 381527. Twelve nominal and four actuator-stress
scenes were paired across all candidates, excluding all ten declared source
layouts and all twelve known historical evaluation layouts. Exact exclusions
do not prove that related situations were never seen elsewhere.

## Cohorts and actual budgets

Each group here contains one source episode. Development remained `example-00`
and `example-03`; the existing source test remained `example-06`.

| Cohort | Training episodes | Groups | Aligned action samples | Source video duration | Training pipeline time |
| --- | --- | ---: | ---: | ---: | ---: |
| `baseline4` | 01, 02, 04, 09 | 4 | 2,281 | 45.8 s | 0.923706 s |
| `generic5` | Baseline + 08 | 5 | 2,857 | 57.4 s | 0.972467 s |
| `targeted5` | Baseline + 05 | 5 | 2,866 | 57.5 s | 0.957306 s |

The targeted reserve was `example-05`, the nearest unused TRAIN group to the
historical failure, at joint layout distance 0.2024664971 m. The other reserves
were `example-08` (0.2936961567 m) and `example-07` (0.3238826399 m). The generic
reserve was `example-08`, chosen by seeded SHA256 ordering after excluding the
targeted group. No fresh evaluation result determined either addition.

The additions had equal **one-group** budgets, not identical action-row or time
budgets. Generic added 576 action samples/11.6 seconds; targeted added 585/11.7.
The nine-row, 0.1-second difference remains a confound to report, not hide.
Training times include intake, frame decoding and learning, but exclude final
receipt hashes, run JSON, ZIP creation and integrity verification. They are
single timings, not a latency benchmark.

## Measured evaluation

| Candidate | Nominal completion | Actuator-stress completion | All scenes | Nominal mean final target error |
| --- | ---: | ---: | ---: | ---: |
| Authored teacher | 12/12 | 3/4 | 15/16 | — |
| `baseline4` | 12/12 | 3/4 | 15/16 | 0.386578 mm |
| `generic5` | 9/12 | 2/4 | 11/16 | 42.872307 mm |
| `targeted5` | 12/12 | 3/4 | 15/16 | 0.324910 mm |

Final-error means include failed scenes. Targeted's smaller nominal error is a
diagnostic difference, not an additional completed task. It matched baseline
scene by scene, including the shared stress failure. It completed three more
nominal and one more stress scene than this one generic composition.

There were zero execution errors and **112 native episodes**: 16 shared teacher
runs plus 3 cohorts × 16 scenes × 2 candidate/control runs. These are only
**16 distinct evaluation scenes**, not 112 independent trials. Every open-jaw
control failed (0/48). The control replays the candidate's recorded arm targets
with the jaws forced open; it is evidence for that intervention, not a broad
causal attribution of success to every learned component.

Generic's additional failures were nominal scenes 04, 09 and 11 (time limit),
and stress scene 14 (`object_dropped`). All candidates and the teacher failed
stress scene 15 at the time limit. Scene IDs are local to each protocol: new
nominal scene 04 is not the historical scene 04 used for the hypothesis.

The historical source-test movement MSE was 1.1087842308e-6 for baseline,
1.4674716000e-6 for generic, and 7.8255410891e-7 for targeted. These are correlated
action samples from one previously inspected episode (621 samples), not a fresh
independent test of generalization. The historical failure layout was excluded
from the new suite, so this experiment does not establish that the original
failure itself was repaired.

## Measured runtime and cost

The complete experiment clock was **138.562881 seconds** (2 minutes 18.563
seconds). The nested benchmark clock was **137.987675 seconds**. The experiment
clock covers planning, source checks, fitting, evaluation and artifact output
before its summary; process startup and the final summary write are excluded.
The nested clock additionally documents its own scope in `benchmark/report.json`.

This was local CPU execution with MuJoCo 3.15.0 and NumPy 2.5.3. The runtime
receipt does not identify the machine model, energy use or a hardware rental
rate. External inference API spend was **$0.00**. Local compute, electricity,
hardware and operator costs are **unpriced**, not zero. These measurements are
for this small simulation experiment, not the cost of analyzing an uploaded
human video, running a large world model, or training a production robot policy.

## The remaining failure points to a control diagnostic

Follow-up: the [48-episode grip diagnostic](grip-recovery-diagnostic-20261015.md)
has now tested this hypothesis with pre-action instrumentation. The account below
preserves what could be inferred from this earlier experiment's saved traces;
its benchmark outcomes are unchanged.

Stress `scene-15` used seed 381542, a 0.06 kg box with friction 1.2, a 0.3 N jaw
force limit and a 5 N·m arm-joint torque limit, with an 18-second horizon. Nominal
limits are 15 N and 40 N·m. The authored teacher also failed, so the evidence
does not isolate a learned-data shortage.

The baseline trace shows the following observed progression:

| Simulation time | Observed state |
| ---: | --- |
| 9.04 s | First lift begins with bilateral jaw contact. |
| 11.02 s | Transfer begins; box centre is approximately 0.421 m high. |
| 13.22 s | First recovery action has opened the jaws; the resulting observation has zero left contacts and one right contact. |
| 13.54 s | Box has resettled on the table and recovery moves the hand upward. |
| 16.08 s | A second lift begins after the single allowed recovery. |
| 18.00 s | Horizon reached while still lifting; final target error is 0.249375 m. |

The controller records `lost_bilateral_contact_or_relative_pose` as its recovery
reason, which combines two different conditions. **The baseline trace does not
show that missing jaw contact initiated recovery.** At 13.20 s, the last saved
observation before the recovery command, both jaws still had four native
contacts. All 306 saved observations during loaded motion had bilateral contact.
The recovery procedure then commanded the jaws open while the object was above
the table. The contact loss visible at 13.22 s follows that command and must not
be presented as evidence of the trigger.

Reconstructing the supervisor's held object/TCP offset from the observation
consumed by its final initial `close` call gives approximately **12.34 mm of
offset drift at 13.20 s**. The preceding five observations exceeded the
controller's 12 mm threshold, consistent with its five-sample retention gate.
The analogous last pre-recovery observations for teacher and targeted also had
four contacts on each jaw, with approximately 12.24 mm and 12.31 mm offset drift.
These are calculations from saved 50 Hz observations and the frozen controller
logic, not recorded pre-action trigger telemetry. Native contact interruptions
between those samples remain possible. The next diagnostic must record each
trigger's input contacts, held-offset error and controller transition directly.

Requested normal grip force was 0.8 N, capped at 0.3 N, with both jaw actuators
reaching that cap. Native 1 ms measurements show the baseline's largest arm
torque peak was 3.068862 N·m, below its 5 N·m limit. Teacher and targeted also
performed one recovery and ran out of time; their final errors were 0.149773 m
and 0.174343 m respectively. Baseline's `lift_seen` remained false despite its
transient maximum lift: the native success predicate requires continuous
bilateral-contact lift dwell, which a 50 Hz trace cannot establish by itself.

The next diagnostic should freeze this scene and the existing models, then
change one factor per arm: jaw force alone; arm torque alone as a control;
loaded-motion speed/acceleration alone; and a separately labelled longer-horizon
diagnostic. Measure the pre-action reason for recovery separately from the
contact loss after opening, object/hand relative pose, native force peaks,
recovery duration, and lift/placement dwell. Raising the jaw cap also changes
the existing supervisor's force target and loaded-motion response; report that
intervention as increased actuator capability with controller adaptation, not
an isolated controller improvement. Keep the original 18-second
result as a failure. A longer time budget must not retroactively count as the
same benchmark pass.

This tests the hypothesis that retention and recovery timing dominate this
failure. It does not prove an irreducible hardware limit, and requesting more
similar videos before that diagnostic would be premature. Any later capture
brief should target the measured gap, with object mass/friction/size, approach
and transfer motion, contact visibility, state before/after, and success/failure
outcome explicitly recorded. Human demonstrations can supply observations and
coverage; aligned robot actions remain a separate prerequisite for this learner.

## Reproduction and provenance

From the repository with the existing source dataset and optional arm
dependencies, reproduce into a **new** output directory:

```sh
PYTHONPATH=src .venv/bin/python -m dvidia_training.improvement_experiment run \
  --dataset runs/huggingface-publication/dataset/native/prepared/dataset.json \
  --prior-protocol benchmarks/sample-count-19047/protocol.json \
  --prior-report benchmarks/sample-count-19047/report.json \
  --source-evidence runs/huggingface-publication/dataset/native/source/evidence \
  --task-source runs/huggingface-publication/dataset/native/source/task.skill.json \
  --seed 381527 --nominal 12 --stress 4 \
  --output runs/improvement-experiment-20261015-reproduction
```

The original command used `--output runs/improvement-experiment-20261015`.
Matching seed and inputs reproduces a regression experiment, not a fresh
confirmation. Timing-bearing output hashes will differ across reproductions.
Protocol and model identity require the same recorded source and dependencies.

SHA256 values below refer to the completed original run's file bytes:

| Artifact | SHA256 |
| --- | --- |
| Prepared source dataset | `e0edc2a275d465ad58ebe1649e1df69c6ec37c30028f64f1c21c483c0f95cad9` |
| Historical protocol | `1e6d475aed2b505ec1ed4ef3990c723b7f9171863647bbfeb01610ba9c8c8eec` |
| Historical report | `e8035534485bfd7473c215c368213c687584573013b07a4e75bc428c668e6d08` |
| `experiment-plan.json` | `d525adaf1310f691ed036c4f3c22b1a847d9eea3875bd814675efb9178ecc800` |
| `benchmark/protocol.json` | `ec842fcbf4a99c0144a48463cef6d203149531066d4089f2f4fb0146ef98c9b2` |
| `benchmark/models-frozen.json` | `44ff1729b7c0770a6eb21e873e133c9248a08a2549b76013f43318e9dad94b38` |
| `benchmark/report.json` | `1bfb368fa7393c35c8b808f5f7b96ac2c314c9d4a6ae7617f9da7368b0dc90ce` |
| `report.json` | `2f1d9f0101ed0e8d5b87ca8e1dbda6d71ee9671095a314868458402de57d8b28` |
| Baseline capsule | `32c62628ed40064de57e3bcb3e32c041b16b268a48dd59307c32f778c3b5ec4d` |
| Generic capsule | `2877c0c99a878bb4d748215032026b7a79e1683d4c3adac423407fd38d591b82` |
| Targeted capsule | `7608e64e9b6fc770d121f57a831d835d07c68d2a86236bbe9ac3e3c3a4dede4a` |

The plan stores full group IDs, source-file hashes, reserve ranking and selection
provenance. The protocol stores all scene declarations and runtime hashes. Raw
failure evidence is in `benchmark/qualification/baseline4/scene-15/run.json`,
with teacher evidence in `benchmark/teacher/scene-15/run.json` and matching
candidate/control receipts under each cohort directory.

This is one seed, one selected reserve per condition, one narrow simulated task
and a very small scene set. No reward, contributor value, premium-plan promise,
physical deployment or automatic promotion follows. All scenes inspected here
now belong to development/regression evidence; any future improvement claim
needs a newly frozen confirmation suite and repeated cohort compositions.

# Grip and recovery diagnostic — 15 October 2026

Follow-up: [controlled lowering and release](controlled-recovery-experiment-20261015.md)
have now been tested at the original limits and deadline. This earlier experiment
and its results remain unchanged.

**The remaining failure combines weak retention, a recovery that opens the
jaws, and a tight task deadline. More arm torque did nothing.** A tested higher
jaw capability removed the retention problem; the unchanged student then
completed in **18.307 seconds**, outside the original 18-second budget. No
software-only intervention tested here repaired the original benchmark failure.

The original model, hardware conditions, success predicate and 18-second failure
remain unchanged. There was no retraining or automatic promotion. The result
does not establish a minimum required force, an irreducible hardware limit, a
generally optimal controller, or readiness for physical robots.

## Evidence and scope

Two frozen local experiments reused baseline4 from the
[Skillspace improvement pilot](skillspace-improvement-pilot-20261015.md) and its
already inspected actuator-stress `scene-15` (seed 381542). This is the simplified
native rigid-box placement profile, not the newer PiPER/UR5e controller. Its
learned movement head still depends on privileged simulator state and authored
phases, grip and recovery logic.

The scene has a 0.06 kg box, friction 1.2, a 0.3 N **jaw actuator force cap** and a
5 N·m joint torque cap. Actuator force and summed pad contact force are different
measurements; contact forces can briefly exceed the actuator cap.

The first experiment froze seven conditions before execution. Each condition
ran the existing student and authored teacher, each followed by its own arm
trajectory replay with the jaws forced open: **28 episodes**. The second
experiment froze a 2 × 2 capability/deadline comparison and a separate 4 mm
tracking-cap exploration: **20 episodes**. All 48 attempts completed without an
execution error. All 24 open-jaw controls failed. These are **one distinct scene
and deterministic condition repeats**, not 48 independent generalization trials.

The new diagnostic harness exactly reproduced the original student and teacher
action tapes, outcomes, simulation times and final target errors before any
intervention. Their original run bytes, the capsule, runtime sources, diagnostic
source and complete plans were hashed and snapshotted. Production runtime
files, numerical model weights and qualification gates were not edited.

## What actually triggers recovery

The new instrumentation records the controller's **input observation before
each action**, separately from the next observation after that action. This
clarifies the earlier report's coarse observation at 13.22 seconds:

| Student observation | Measured evidence |
| --- | --- |
| 13.12 s, transfer | Relative object-to-hand offset first exceeds the 12 mm retention threshold: 12.086 mm. Both pads still report four contacts. |
| 13.20 s, transfer | Fifth consecutive failing check: 12.341 mm offset. Both pads still report four contacts; recovery is selected. |
| 13.22 s, recovery | The recovery action has opened the jaws; the next observation reports zero left and one right contact. |

The sampled loss of bilateral contact **after recovery begins is not evidence
that bilateral contact loss caused this transition**. The retained pre-action
evidence identifies the relative-offset branch of the existing guard. The
student's offset accumulates primarily downward during loaded movement.

The teacher also triggers recovery with both pads in contact and a 12.235 mm
relative-offset error at 12.32 seconds. It had an earlier intermittent missing
left-contact sample; that alone did not produce the eventual transition.

Sampling is 50 Hz; native physics and the lift predicate run at 1 kHz. All
control samples can show contact while shorter contact interruptions reset
native lift dwell. The original student never satisfies the required 100 ms
sustained retained lift (`lift_seen=false`); its maximum **sampled** running lift
dwell is 73 ms. It is therefore not valid to disable the slip guard or count a
transient high object position as success. The teacher's sustained lift passes,
but its later placement still fails.

## Frozen one-factor results

All rows preserve the original 18-second horizon except the explicit last row.
“Fail” means the complete unchanged native success predicate was not met.

| Condition | Student | Authored teacher | Observed mechanism |
| --- | --- | --- | --- |
| Original, 0.3 N jaws / 5 N·m joints | Fail | Fail | Relative-offset guard triggers recovery; both run out of time. |
| Jaw cap 0.6 N | Fail | Pass, 16.482 s | Neither needs recovery. Student reaches 2.216 mm target error but remains in retreat at 18 s. |
| Jaw cap 0.8 N | Fail | Pass, 16.502 s | Neither needs recovery. Student reaches 1.574 mm target error but remains in retreat at 18 s. |
| Joint torque cap 40 N·m | Fail | Fail | Same actions and outcomes as original; no observed torque-cap bottleneck. |
| Loaded goal cap 2.5 mm | Fail | Fail | Retained lift improves, but both hit the unchanged four-second lift-phase timeout. |
| Loaded goal cap 1.25 mm | Fail | Fail | Lift movement is too slow for the unchanged phase timeout; sustained lift remains unestablished. |
| Horizon 30 s, original capabilities | Fail | Fail | Both lose retention again after their single recovery; more time alone is insufficient. |

The original student's native torque peak is 3.069 N·m, below its 5 N·m cap,
while both jaw actuators reach their 0.3 N cap. Raising jaw capability changes
both the native cap **and the unchanged supervisor's adaptive response**: normal
force target, feasibility check and its existing force-margin tracking cap.
It is a one-declared-factor intervention, not an isolated estimate of physical
jaw force alone.

The loaded-goal cap changes requested Cartesian tracking distance before the
unchanged IK or learned mapping. It does not directly specify acceleration or
guarantee a physical end-effector speed. Phase timeouts were deliberately kept
unchanged so a slower command could not silently acquire a larger budget.

## Capability × deadline follow-up

The 2 × 2 experiment directly tests whether the higher-capability student's
remaining failure is the deadline. All other native predicates remain unchanged.

| Jaw cap | Horizon | Student | Authored teacher |
| --- | ---: | --- | --- |
| 0.3 N | 18 s | Fail | Fail |
| 0.6 N | 18 s | Fail | Pass, 16.482 s |
| 0.3 N | 24 s | Fail, second retention loss | Fail, second retention loss |
| 0.6 N | 24 s | **Pass, 18.307 s** | Pass, 16.482 s |

At 0.6 N the student maintains sustained lift, places the box, releases it,
clears the hand and satisfies dwell. Its 18.307-second completion is 0.307
seconds beyond the original deadline. The result belongs to the explicitly
changed capability and time budget; the original 18-second result stays failed.
The tested 0.6 N cap is not a measured minimum requirement.

The separately frozen **4 mm loaded-goal cap** used the original 0.3 N cap and
18-second horizon. Both policies still failed. The student now satisfied native
lift dwell and transferred farther, but triggered the same offset guard at
17.80 seconds (12.333 mm, both pads in contact); its post-action recovery frame
began at 17.82 seconds. This delays the failure rather than repairing it. We
stopped tuning after this predeclared exploratory condition.

## What this changes for the Skillspace loop

This evidence changes the next action from “upload more generic examples” to
**investigate and declare the execution envelope**. The supported task must
state robot capability, object/material range, timing and completion predicates.
It should retain the original failure, show the tested changed-condition success
separately, and never turn that result into a production capability claim.

Future software hypotheses are smoother loaded motion that retains useful
throughput, and controlled lowering before opening when contact persists. These
remain hypotheses. They must preserve force/success constraints and pass a
newly frozen paired regression suite before promotion. The experiment did not
find a software-only fix, so no improved policy was sent to a fresh confirmation
suite in this round.

If new demonstrations are collected, request contrasting retention and recovery
cases with measured object mass/friction/size, force limits, continuous object
and hand state, commands, slip, support, recovery and final outcome. The existing
learner needs aligned robot actions; ordinary human video alone cannot provide
those labels or prove physical capability.

## Measured cost, artifacts and reproduction

| Experiment | Native episodes | Complete wall clock | External inference spend |
| --- | ---: | ---: | ---: |
| One-factor diagnostic | 28 | 30.688776 s | $0.00 |
| Factorial + 4 mm exploration | 20 | 22.716543 s | $0.00 |
| Combined | 48 | **53.405319 s** | **$0.00** |

These clocks include validation, snapshots, native rollouts, telemetry and replay
writes, but exclude process startup and the final report write. Local CPU,
electricity, hardware and operator costs remain **unpriced**, not zero. The
Python socket guard records zero attempted network calls; it does not audit
native dependency networking. This is not the cost of analyzing human video or
training a production visual policy.

Exact compact receipts are in
[benchmarks/grip-recovery-20261015](../benchmarks/grip-recovery-20261015/README.md).
Full traces, source snapshots, weights and replay HTML stay in the ignored local
run directories. Reproduce into new directories; matching the same inspected
scene is a regression, not fresh confirmation:

```sh
PYTHONPATH=src .venv/bin/python -m dvidia_training.grip_diagnostic \
  --benchmark runs/improvement-experiment-20261015/benchmark \
  --output runs/grip-diagnostic-reproduction

PYTHONPATH=src .venv/bin/python -m dvidia_training.grip_followup \
  --diagnostic runs/grip-diagnostic-reproduction \
  --output runs/grip-followup-reproduction
```

Timing-bearing JSON hashes will differ on a repeat. Original completed hashes:

| Artifact | SHA256 |
| --- | --- |
| Phase-one protocol | `2a69227760e5aacb70984d1923614eed7a28cb2e079281a9c2694481e863bead` |
| Phase-one report | `0ee7101614723aa0f2fe79d288f6c8662138f0352320aca0f829fc60c950d6bb` |
| Phase-two protocol | `cdebe680658a8ef3f229f460f53234413f682b9dd28bca1f1ac9eb8918e8ad08` |
| Phase-two report | `cf4488f3f2e20802d8ba6f9f5f512508d5058e3cc36292929fb35c242e10d9a9` |

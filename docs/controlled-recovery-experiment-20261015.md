# Supported release during recovery — 15 October 2026

**The experimental controller now lowers the box to the table, confirms support,
opens the jaws while holding the hand still, and verifies jaw clearance before
withdrawing.** In the known failing scene this removes the original unsupported
opening commands and greatly reduces sampled descent speed. It does **not**
repair completion within the original 18-second deadline: the unchanged student
and both candidates still pass 15 of the 16 original benchmark cases.

No production controller, model weights, force caps, success predicate or
qualification status changed. This is a simulation recovery improvement with a
preserved task failure, not a claim of physical robot readiness.

## Why this experiment

The [preceding grip diagnostic](grip-recovery-diagnostic-20261015.md) established
that relative object-to-hand drift triggers the original recovery while both
jaws still have contact. The recovery immediately requests open jaws. A later
frame showing contact loss therefore cannot establish that contact loss caused
the recovery. Raising arm torque did nothing; a tested larger jaw capability
and longer task budget could complete the case, but did not fix the original
capability and deadline.

The software hypothesis tested here was narrower: when recovery starts, lower
the object to supported rest before opening. Keep the existing trigger, limits,
one-retry allowance, student movement head and authored teacher mapping.

## Frozen comparison

Each of two independently frozen phases ran the complete original 16-case
student suite, baseline versus candidate, plus baseline versus candidate for the
authored teacher on stress `scene-15`. Every episode was followed by its own
recorded arm trajectory replay with jaws forced open. That is **68 native
episodes per phase**, or **136 episodes total**, on the same 16 previously
inspected scenes.

All attempts are retained. Both phases completed without errors. Each reproduced
all 17 original baseline action tapes and outcomes exactly. Every candidate
matched its original actions up to the first recovery intervention. All 68
open-jaw controls across the two phases failed. There was no retraining and no
new held-out data in these regressions.

The benchmark, its original run bytes, capsule, runtime, experimental sources,
settings and attempt list were hashed before execution. The second phase binds
the first phase's protocol and report hashes. Exact compact copies are in the
[receipt archive](../benchmarks/controlled-recovery-20261015/README.md); complete
synthetic traces and frozen source snapshots are retained locally under `runs/`.

## Experimental recovery contract

The first revision preserves the grasp command and lowers with a **3 mm maximum
requested Cartesian goal increment**. The command still goes through the same
student movement head or teacher IK. It is not a guaranteed physical speed or
acceleration. Opening requires five consecutive 50 Hz observations with:

- Native filtered table contact.
- Object center within 8 mm of its configured resting height.
- Linear speed below 0.035 m/s and angular speed below 0.3 rad/s.
- The existing workspace and orientation checks at the supported opening.

The second revision adds a clearance phase: keep the same measured joint pose
while the jaws open, and require actual gap **greater than 65 mm**, zero native
grasp contacts on both sides, and the same supported-settled condition for five
consecutive observations. The confirmation tick still holds position; withdrawal
can begin only afterward. Lowering and clearance share a **four-second budget
from the original recovery trigger**, within the unchanged absolute 18-second
episode. Failure to confirm support or clearance ends in a bounded hold, not a
forced unsupported opening or a deadline reset.

These are experiment-only subclasses. Normal phase budgets, forces, native
success tests and runtime defaults are unchanged. The support signal alone is
not a measurement of full weight support; the height, speed and persistence
checks strengthen this operational predicate without claiming force certainty.

## What the first revision revealed

The first revision successfully delayed the first open command until sampled
support and settling. It then entered the existing upward recovery phase too
soon, before the physical jaws had cleared the object. The teacher's first
withdrawal briefly lost table contact; its object center rose by about 0.27 mm
at that sample. The student also had a small upward disturbance while retaining
some table contact.

This made the distinction important: **a supported first open command is not
proof of a supported physical release**. The second revision was frozen before
rerunning the same complete paired suite; it holds the hand through actual jaw
clearance rather than changing a reporting counter.

## Measured target-case results

The failing scene retains its original 0.3 N jaw actuator cap, 5 N·m joint cap,
0.06 kg box, friction 1.2 and 18-second horizon. Contact force and actuator force
are distinct measurements.

| Observation | Original student | Revised student | Original teacher | Revised teacher |
| --- | ---: | ---: | ---: | ---: |
| First recovery trigger | 13.20 s | 13.20 s | 12.32 s | 12.32 s |
| First requested open | 13.20 s, unsupported | 15.60 s, supported | 12.32 s, unsupported | 14.60 s, supported |
| Measured clearance + settled support confirmed | Not instrumented by baseline policy | **15.98 s** | Not instrumented by baseline policy | **14.98 s** |
| Unsupported recovery-open command samples | 9 | **0** | 9 | **0** |
| Sampled peak descent speed before first table contact | 1.278 m/s | **0.060 m/s** | 1.299 m/s | **0.061 m/s** |
| Final target distance at deadline | 249.375 mm | 222.454 mm | 149.773 mm | 164.336 mm |
| Complete native task within 18 s | Fail | **Fail** | Fail | **Fail** |

No completion regressions does not mean every metric improved: the revised
teacher ends farther from the target than the original. This remains a failed
placement; the demonstrated improvement is the observed release handoff.

The revised student confirms release 2.78 seconds after recovery begins; the
teacher does so after 2.66 seconds. Both show approximately 79 mm measured jaw
gap, zero grasp contacts, four table contacts and five consecutive qualifying
samples at confirmation. The hand begins the upward phase after clearance.
Independent trace review found 20 identical joint-target commands from opening
through clearance for each controller, and no zero-table-contact sample from
opening through the first eight withdrawal frames. The first revision's brief
teacher table-contact dropout is absent in that observed window.

During lowering, the student has no missing bilateral-contact sample out of 120;
the teacher has one out of 114. This is **50 Hz sampled evidence**, not proof of
continuous contact at the 1 kHz physics rate or that the object remained grasped
throughout lowering. The reported descent speed is not impact force. Unsupported
opening counts requested open commands while table contacts are zero, not
directly observed physical release events.

The revised student's native `lift_seen` eventually becomes true during recovery.
That does not repair its original lift/transfer: the pre-trigger trajectory is
identical, and task completion still fails. Holding or supporting the object
until the deadline is not counted as a successful placement.

## Regression outcome and next decision

| Original student cases | Baseline | First revision | Release-confirming revision |
| --- | ---: | ---: | ---: |
| Nominal | 12/12 | 12/12 | 12/12 |
| Actuator stress | 3/4 | 3/4 | 3/4 |
| Total | **15/16** | **15/16** | **15/16** |
| Paired gains / regressions | Reference | **0 / 0** | **0 / 0** |

The predeclared fresh-confirmation gate requires a strict task-completion gain,
zero regressions and execution errors, and failed matched controls. It was not
met, so no fresh confirmation run or promotion followed. There is no evidence
here of better generalization or improved benchmark completion. The useful
result is a more controlled supported release in the observed failure case.

The remaining work is to prevent loaded drift while retaining enough throughput
for the original deadline, or explicitly declare and validate a different robot
capability/task envelope. Do not relax the 12 mm retention guard, native lift
dwell or deadline to manufacture a pass. Another controller change needs a
freshly frozen experiment and all paired regressions. Broader support surfaces,
objects, contacts, occlusion, dropped-object conditions and recovery triggers
also need dedicated validation before using this strategy outside this profile.

For the Skillspace loop, keep recovery quality and task success as separate
fields. Request demonstrations with known support surfaces, slip and recovery
outcomes, aligned state and commands, and final task outcomes. More ordinary
videos alone neither supply robot action labels nor remove a force/timing limit.

## Cost and reproduction

| Phase | Native episodes | Measured complete wall clock | External inference spend |
| --- | ---: | ---: | ---: |
| Supported opening | 68 | 66.698673 s | $0.00 |
| Supported release + clearance | 68 | 70.969774 s | $0.00 |
| Combined | **136** | **137.668447 s** | **$0.00** |

Wall clocks include validation, snapshots, native rollouts, telemetry and replay
writes, excluding process startup and final report write. Local CPU, electricity,
hardware and operator costs are unpriced. The Python socket guard observed zero
network attempts; it does not audit networking inside native dependencies. This
is not the price of human-video analysis or training a production vision policy.

Fifteen focused tests cover support persistence, reset and timeout conditions,
student mapping without IK fallback, unchanged actions before recovery, held
pose through measured clearance, the shared deadline, and report gate behavior.

Reproduce into new directories, preserving the completed evidence:

```sh
PYTHONPATH=src .venv/bin/python -m dvidia_training.controlled_recovery_experiment \
  --benchmark runs/improvement-experiment-20261015/benchmark \
  --output runs/controlled-recovery-reproduction

PYTHONPATH=src .venv/bin/python -m dvidia_training.controlled_recovery_release_experiment \
  --benchmark runs/improvement-experiment-20261015/benchmark \
  --prior-experiment runs/controlled-recovery-reproduction \
  --output runs/controlled-recovery-release-reproduction
```

Repeated timing-bearing hashes will differ. The archive preserves the original
hashes and exact report bytes. These simplified rigid-box, privileged-state
episodes are not the newer PiPER/UR5e controller and are not physical-robot tests.

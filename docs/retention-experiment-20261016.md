# Retention before recovery — 16 October 2026

**Two targeted controller experiments did not repair the remaining 18-second
failure.** Smoothing motion delayed the student's slip recovery by 0.42 seconds.
A bounded compliant transfer path delayed it by 1.10 seconds and allowed a native
sustained lift, but still exhausted its movement allowance and reached the
original slip guard. The student remains at **15/16 cases**, with no regressions
and no task-completion gains.

The tested smoothing did not remove the failure; downward object-to-hand drift
also persists during steady loaded motion under the original low jaw-force
capability. It does not establish an irreducible hardware
limit, a physically calibrated friction model, or that no better controller
exists. We stopped after the two frozen, bounded experiments; nothing was
promoted.

## What the trace showed first

The original student's first lift begins at 9.02 seconds. By the transfer command
at 11.00 seconds, object-to-hand offset error has already accumulated to about
5.8 mm. The object rises roughly 2–3 mm/s slower than the hand in the sampled
steady lift; pad forces are near the original 0.3 N actuator cap. Changing into
transfer adds a disturbance, but drift existed before that direction change.

The student first exceeds 12 mm at 13.12 seconds, then the fifth consecutive
failing check triggers recovery at 13.20 seconds. Both jaws still have contacts
at that input observation. As established by the [grip diagnostic](grip-recovery-diagnostic-20261015.md),
the ensuing open command causes the later sampled contact loss; it is incorrect
to infer the trigger from that later frame alone.

The original student never satisfies the native 100 ms retained-lift dwell. Its
50 Hz observations can show bilateral contact while interruptions at the 1 kHz
physics rate reset the native predicate. Control samples cannot establish
continuous native contact.

## Frozen scope and invariants

Each experiment compares baseline and candidate across the complete original
16-case student suite, plus the authored teacher on known failing scene 15.
Every controller run has its own recorded-arm replay with the jaws forced open:
**34 controller episodes and 34 controls per phase**, or **136 total episodes**
across the same 16 previously inspected cases.

The low-headroom gate is the existing static-force-margin threshold of 1.25.
Only scene 15 activates it (margin approximately 1.223); the other 15 scenes
have margins at least 1.529. Those 15 candidate trajectories and outcomes are
exactly unchanged in both experiments. Their regression pass protects existing
behavior, but does not validate the new intervention across 15 varied active
conditions.

The benchmark, original trajectories, model capsule, runtime, experimental
sources, settings and attempt list were hashed and snapshotted before running.
Phase two references the completed first-phase protocol and report hashes. All
attempts are included, with no execution errors. Each phase exactly reproduced
17 original baseline trajectories and outcomes and verified 17 candidate action
prefixes before intervention. All 68 open-jaw controls across both phases failed.

Both experiments preserve original forces, masses, materials, physics, model
weights, success scoring, phase budgets and the absolute **18-second deadline**.
The 12 mm/five-consecutive-check retention guard and its original held-offset
reference are unchanged. The learned student uses its same movement head; the
teacher uses its original IK. No student IK fallback was added.

The [supported-release recovery experiment](controlled-recovery-experiment-20261015.md)
remains a separate preserved result. These new experiments retain original
recovery only to isolate the effect before recovery. They do not replace or
discard that experimental supported-release strategy. A future successful combined
controller would require its own frozen paired retest and confirmation.

## Phase one: smoother tracking increments

The candidate applies a convex low-pass filter with alpha **0.2 at 50 Hz** to
the requested Cartesian increments during low-headroom lift and transfer. It
carries the increment across their direction change and resets outside those
phases. The original 5 mm maximum remains bounded; place, release and recovery
are untouched. This is a command filter, not a guaranteed physical acceleration
or velocity limit.

| Target scene observation | Original student | Smoothed student | Original teacher | Smoothed teacher |
| --- | ---: | ---: | ---: | ---: |
| First lift command | 9.02 s | 9.02 s | 7.60 s | 7.60 s |
| First transfer command | 11.00 s | 11.10 s | 9.46 s | 9.54 s |
| First sampled offset >12 mm | 13.12 s | 13.54 s | 12.24 s | 12.30 s |
| Recovery selected | 13.20 s | 13.62 s | 12.32 s | 12.38 s |
| Native retained lift observed | No | No | Yes | Yes |
| Complete task by 18 s | Fail | **Fail** | Fail | **Fail** |

The student still accumulates offset during steady transfer, from 7.46 mm at
the 12.00-second post-control frame's input to 10.25 mm at the corresponding
13.00-second frame's input. Smoothing does not remove that sustained drift. We
did not search different smoothing constants after seeing the negative result.

## Phase two: bounded compliant transit path

The separate follow-up changes the authored transfer path under the same low
headroom. It commands the hand toward its original grasp height relative to the
observed object, allowing some downward motion instead of insisting on a fixed
transit height. It never edits the observed object position or resets the
retention reference.

The frozen settings are a **0.5 vertical-error gain**, at most **2 mm requested
vertical correction**, at most **5 mm total requested increment**, and a
**25 mm requested downward allowance** from transfer entry. Compliance requires
bilateral contact and at least **75 mm observed object clearance** above its
initial resting position; otherwise the original mapping is used. If actual
servo motion crosses the requested floor, the correction points upward and
remains bounded, rather than forcing the simulated object or hand to a position.

Because the transit height changes, the candidate also changes that authored
phase's completion check: original 12 mm planar arrival tolerance, bilateral
contact, at least 75 mm clearance, and original relative-offset error no greater
than 12 mm. The last condition prevents a new early transition from bypassing a
pending slip check. The native task's retained lift, release, hand clearance,
target position, settling and dwell predicates remain unchanged.

This is a **path and controller-phase change**, not merely another smoothing
constant and not a relaxed task score.

| Target scene observation | Original student | Compliant student | Original teacher | Compliant teacher |
| --- | ---: | ---: | ---: | ---: |
| First transfer command | 11.00 s | 11.00 s | 9.46 s | 9.46 s |
| First native retained-lift observation | Never | **11.12 s** | 9.62 s | 9.64 s |
| Recovery selected | 13.20 s | **14.30 s** | 12.32 s | **12.32 s** |
| Minimum sampled transfer object clearance | 108.16 mm | 79.21 mm | 104.25 mm | 80.03 mm |
| Maximum sampled transfer hand descent from phase entry | 0 mm | 25.146 mm | 2.464 mm | 25.337 mm |
| Complete task by 18 s | Fail | **Fail** | Fail | **Fail** |

The student approaches within 0.5 mm of the requested descent floor at 12.00 s;
the teacher does so at 10.28 s. Both still have about 6.5 mm offset error then,
which subsequently grows to the unchanged guard. The student now meets native
retained lift before recovery, a useful intermediate result, but still cannot
retain and complete the transfer within the task.

Actual sampled hand descent slightly exceeds the requested 25 mm allowance due
to physical servo response. The command bounds are not certified physical
trajectory bounds. Recorded candidate requests stay within 5 mm total and have
vertical peaks of 1.868 mm student / 1.910 mm teacher. Native jaw actuator peaks
remain 0.3 N; joint peaks remain below the original 5 N·m cap. The object stays
above 75 mm clearance in the sampled transfer observations, which does not prove
continuous clearance between samples.

Final target distance changes from 249.37 to 161.86 mm for the student, but from
149.77 to 154.60 mm for the teacher. These endpoint changes are not successful
placements or evidence of uniform improvement.

## Outcome and next decision

| Student suite | Baseline | Smoothing | Compliant transfer |
| --- | ---: | ---: | ---: |
| Nominal cases | 12/12 | 12/12 | 12/12 |
| Actuator-stress cases | 3/4 | 3/4 | 3/4 |
| Total | **15/16** | **15/16** | **15/16** |
| Paired task gains / regressions | Reference | **0 / 0** | **0 / 0** |

The predeclared confirmation gate requires a strict student task-completion
gain, zero regressions/errors and failed matched controls. Neither experiment
meets it. No new held-out suite was run and no controller or model was promoted.
The evidence supports neither an original-budget solution nor broad retention
generalization.

Further work should first quantify the 1 kHz contact interruptions and phase
time requirements, then freeze a new, mechanically distinct hypothesis. The
current evidence does not justify endless tuning, increasing force silently,
relaxing the slip/lift predicates or relabeling longer survival as success.
Training data should distinguish material/capability boundaries, native contact
behavior, movement commands, retained lift, recovery and final placement. More
unlabelled human footage alone cannot identify these robot action or force
requirements.

## Timeline evidence, cost and reproduction

Each result stores actual timestamped `measurement.retention.events`, including
phase commands, sampled 2/6/12 mm offset crossings and the first observed native
retained lift. `time` includes its explicit pre- or post-action basis;
`frame_time` points to the corresponding stored replay frame. Phase event IDs
can repeat after a retry. Do not collapse those into one event or mistake a
post-action picture for the pre-action trigger observation.

`measurement.retention.curve` contains a **10 Hz subset of actual 50 Hz
pre-action observations**, including loaded phases after any recovery retry.
It is not an interpolated curve or a pre-recovery-only series. The separate
maximum-offset field stops before the first recovery action. Full native-run
traces and replay pages are stored with every episode.

| Experiment | Native episodes | Measured complete wall clock | External inference spend |
| --- | ---: | ---: | ---: |
| Increment smoothing | 68 | 98.311345 s | $0.00 |
| Compliant transfer | 68 | 67.524272 s | $0.00 |
| Combined | **136** | **165.835617 s** | **$0.00** |

These clocks include validation, snapshots, rollouts, telemetry and replay
writes, excluding process startup and final report write. Local CPU, electricity,
hardware and operator costs remain unpriced. The Python socket guard recorded
zero network attempts; it does not audit native dependency networking. These
costs are not human-video analysis or production visual-policy training costs.

Fourteen focused tests cover mapping/cap preservation, phase resets, no-op
reproduction, support and clearance fallback, the unchanged retention guard,
pending-slip arrival rejection and the task-gain gate. Exact source and evidence
hashes and compact receipts are in the [archive](../benchmarks/retention-20261016/README.md).

Reproduce into fresh directories to preserve completed evidence:

```sh
PYTHONPATH=src .venv/bin/python -m dvidia_training.retention_experiment \
  --benchmark runs/improvement-experiment-20261015/benchmark \
  --output runs/retention-smoothing-reproduction

PYTHONPATH=src .venv/bin/python -m dvidia_training.retention_compliance_experiment \
  --benchmark runs/improvement-experiment-20261015/benchmark \
  --prior-experiment runs/retention-smoothing-reproduction \
  --output runs/retention-compliance-reproduction
```

This remains a privileged-state, simplified rigid-box simulation profile,
separate from the newer PiPER/UR5e work. It is not a physical robot test.

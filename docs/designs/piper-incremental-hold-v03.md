# Bounded incremental pose correction after a test lift

Status: reviewed before source freeze, 31 October 2026 local date.
This specification is reviewed before native trials. No success is presumed.

## Problem and decision

The frozen v0.2 Cartesian controller produced a nearly constant actuator target
while TCP displacement crossed its 4 mm limit. `pose_action` returns an absolute
joint target: measured joint position plus a damped inverse-kinematic error step.
Subtracting its admission value nearly cancels measured drift against that step.
The saved-record explanation is testable; it is not proof of a working remedy.

Preserve the three v0.2 controls unchanged and add exactly one authored candidate.
The candidate's first admitted command equals the preceding issued target. At
each subsequent 20 ms boundary, obtain the current pose-error increment as
`pose_action(admission_pose) - current_measured_joint_position`. Add this increment
to the accumulated correction around the preceding issued admission target.
The update gain is exactly 1.0 per control boundary, chosen before any native run.
Limit each accepted correction change to 0.002 rad per joint. Refuse if the
unlimited cumulative proposal exceeds ±0.03 rad; do not clip or integrate beyond
that total bound. Update internal correction only after all checks pass. This
terminal refusal supplies the anti-windup rule. Do not tune gains on the old or
new benchmark. A failure remains a failure.

## Invariants

- Original evidence-gated acquisition, supported retry and terminal refusal.
- Same fixed admission pose; initial bias bound 0.08 rad; target margin 0.02 rad.
- Candidate-only measured-joint eligibility: at admission, every subsequent
  update and final observation, refuse measured joints outside that same 0.02 rad
  native target margin before calling the pose solver. Its absolute-target
  clipping would otherwise create a nonzero increment at zero pose error.
  Record `measured_joint_margin_exhausted` as a terminal bound refusal, without
  integrating correction. The three original variants remain unchanged.
- Same jaw target after admission, native gains, force caps and 1 ms target slew.
- Same 4 mm / 0.05 rad terminal pose error, original relative-grip checks, and
  synthetic 10 N per-finger checking ceiling. These are not physical safety limits.
- Invalid or noncausal measurements refuse without issuing a command. Final
  observations check evidence but do not solve or issue a hypothetical next action.
- No hardware control, new camera model, learning, transport or placement.
- Keep candidate identity, update law, increments, proposed and applied cumulative
  corrections, bound refusals and final controller status explicit in records.

## Fixed comparison

Eight fresh finite development scenes: masses 0.0475 / 0.0525 kg; friction
0.78 / 0.52; starts (0.304, -0.024, 0.082) and (0.308, -0.035, 0.082) metres.
Seeds 311200–311207. Four variants: measured_joint, issued_joint, cartesian,
incremental. Thirty-two attempts, each 20 simulated seconds, once each.
Four balanced orders repeated twice: [0,1,3,2], [1,2,0,3], [2,3,1,0], [3,0,2,1].
Each variant occupies every run position twice. No trial resume, outcome-based
retry, candidate selection, or post-result parameter changes.

Primary native task endpoint: correctly qualified 100 ms admission prefix,
continuous native grip retention AND TCP stationarity for the next five seconds.
Use every 1 ms native sample and exactly the v0.2 definitions: bilateral contacts,
forces strictly >0.01 N and <=10 N, unsupported, at least 6 mm probe rise, at most
3 mm TCP-local drift, TCP displacement <=4 mm and orientation <=0.05 rad. Same
distance epsilon 1e-12. Require full valid warning-free recording, no unexpected
contact anywhere, and no post-admission force-ceiling violation. Missing admission
or incomplete follow-up fails. All attempts remain in the denominator.

Report separately: grip-only diagnostic; controller retained at 5 s; conjunction
of native task success and controller qualification; false initial admissions;
false maintained claims against grip and against pose; final qualification;
failure timing/reasons; active correction magnitude/rate; wall and CPU timings.
A terminal fallback may retain an object: its outcome cannot be attributed to
continued active Cartesian regulation. No general success or speedup claim follows
from these deterministic cells. Previously published learned placement is unchanged.

## Verification and publication gate

1. Fresh-context reviewer receives this document and referenced source only.
2. Pure tests establish correct sign, accumulation, zero-error invariance,
   bumpless admission, rate/total bounds, no update on refusal, frozen jaw, invalid
   measurements, terminal status and final non-command observation behavior.
3. Freeze source commit, source map, protocol, runtime and asset hashes before
   the 32 native trials. Root alone runs the native matrix once.
4. Reconstruct requested-to-applied controls; verify exact common acquisition
   prefixes across four variants up through earliest treatment observation,
   excluding the first treatment command. Preserve unknown/incomplete records.
5. Independently recompute native grip/pose and correction bounds from saved
   numeric arrays and command records, with no runtime/scorer import or physics.
6. Publish all failures and exact source/evidence only after tests, review and
   actual browser inspection. Adoption into the stable local product is a
   separate decision; this comparison cannot establish hardware readiness.

## Workflow sources and boundary

The approach takes inspiration from EGM's context-independent design review and
Harness2's propose/probe/compose and retained lessons. We implement an original,
bounded robotics experiment using our existing sandbox and native evaluator.
We do not install their plugins or adopt unrestricted execution. Model weights
are not being trained by either workflow. This study does not run Harness2 or
claim its benchmark results for robotics. The final evaluation stays separate
from proposal development; inference about a next change is explicitly post-hoc.

- https://github.com/google/elephant-goldfish-model/tree/403d8881d62022490637d740657b33f01e321ea8
- https://github.com/google-research/harness2/tree/1cfe254e8d25b15f37048c3730134686abe60335
- Existing source: src/dvidia_training/piper_retention.py and piper_task.py.
- Existing method: benchmarks/piper-retention-v02/README.md.
- Frozen baseline: bc106d6f66367f8896787a2303aa0a124c69ea56.

## Pre-execution review disposition

Fresh review found that absolute-target clipping could manufacture a nonzero
update at zero pose error outside the joint margin. The candidate now refuses
that measurement before admission, updates and final inspection. A pure test
reproduces the counterexample. Reciprocal review found setup/export exceptions
and nonfinite JSON could truncate the scheduled matrix; the runner now retains
invalid attempts and explicit invalid-number markers. A mocked all-invalid
32-attempt lifecycle checks retention and refusal to resume. Both reviewers
cleared their blockers before any native trial. The three old controllers are
byte-for-byte unchanged.

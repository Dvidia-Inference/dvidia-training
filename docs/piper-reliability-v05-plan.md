# PiPER reliability v0.5: evaluation plan

Status: selected candidates and evaluation requirements, frozen before fresh
confirmation. Development has passed; fresh results are not yet known. The
preregistered protocols identify all three selected modes, constants, reserved
seeds and source hashes. This is not physical hardware qualification.

The current implementation selects `recovery`, `reacquire` and
`reacquire_grasp`: frozen v0.4, the bounded second lift, and the second lift with
a fixed alternate grasp planner. Each has its own open-jaw tape control, with
one shared hold, for **84 attempts per twelve-scene cohort**. The selected grasp
uses a fixed −45° world yaw and a 0.10 rad grasp-orientation IK error bound;
selection precedes the episode and does not read hidden object properties.
`piper_reliability_benchmark` provides portable inspection and a separate
asset-backed complete policy/IK audit. A completed benchmark command runs both.

The objective is **more completed placements under the same physical model and
task criteria**. Earlier refusal of a failed grasp is useful, but does not reduce
the placement failure rate. The frozen v0.4 `recovery` mode is the primary
reference. Published seeds 813407 and 941827 are now development data.

## What the evidence supports

The v0.4 fresh cohort completed 4/12 placements in each mode, with no gained or
lost completions. Its two resumed successful placements finished 2.440 and
2.640 seconds sooner. Four unsuccessful grasps were refused earlier.

There are two separate failure mechanisms to address:

1. **Stable but low after a hold.** The two nominal-friction 60 g cases regain
   bilateral contact and remain suspended, but fall below the unchanged 6 mm
   resume height before completing its 100 ms dwell. The original motion never
   resumes. A bounded second load-bearing lift or gentler initial acceleration
   is a testable candidate; simply lowering the resume height is not this test.
2. **Supported object does not follow the pinch.** The original six difficult
   cases cannot be lifted by the current side pinch with the pinned servo. Its
   nominal actuator cap is not reached. A repeat of the same supported grasp
   changes nothing. An alternative, geometrically distinct grasp can be tested
   using the existing model. It must earn contact, load-bearing and complete-task
   evidence independently. No outcome is promised for the lowest-friction cases.

The force diagnosis concerns these synthetic blocks, this servo and this grasp.
It neither identifies the capacity of the physical PiPER nor proves all grasps
impossible. Higher gains, force limits, friction, collision geometry or compliant
pads would change the experiment and must be a separately labelled comparison.

## Preserve the task

Keep the pinned model, free-object dynamics, initial-state construction, gravity,
native gains, force limits, command units, 1 ms integration, 20 ms control cadence
and full 20-second episode unchanged. Keep `PiperTaskConfig` task thresholds and
the independent native `_outcome_tick` scorer unchanged:

- A bilateral, unsupported rise of at least 60 mm sustained for 100 ms.
- Retained carry to a supported destination within the existing 25 mm tolerance,
  with the existing contact-loss latch and no prohibited contacts.
- Actual open jaws, no pad contact, supported and settled object, 100 mm tool
  retreat clearance and 250 ms destination dwell.

Every completed, refused, failed or invalid attempt remains in the planned
denominator. An invalid placement cannot donate a made-up arm tape to its control.
No drop can be relabelled as a successful placement after landing in the target.
No hold, resume, short probe or controller label is task success.

## Candidate boundaries

A second lift must be an explicit new probe state, not permission to carry below
the original resume gate. Before it starts, require current and past observations
of loaded bilateral contacts, absent table support and bounded relative motion.
Require minimum observed span; a single quiet frame is insufficient. Keep jaws
closed and record the actual issued targets, observed displacement and native
object response. Bound its distance, duration and retry count before running it.

The probe must re-earn the original height and settling dwell before carry
resumes. Persistent contact loss, excessive relative drift, unexpected contact,
support return or exhausted probe budget must take the declared closed-hold/abort
path. Any separate probe time budget consumes the original 20-second task budget;
it does not silently extend the task or reset the number of reassessments.

An alternative acquisition grasp must begin from a confirmed supported object.
Record the changed approach geometry and observed support throughout any open
and recenter phases. An unsupported object cannot be released to try a new grasp.
The original acquisition test is rerun after repositioning. A new grasp cannot
be selected using hidden scene mass, friction, a future observation or the scorer.
The existing downstream teacher's use of its original lift flag must be disclosed;
it does not make the new grip decision sensor-based.

## Paired comparisons

The lean default has three placement modes:

| Mode | Purpose |
| --- | --- |
| Frozen v0.4 `recovery` | Matched reference |
| Candidate with only the selected carry change | Isolate the stable-low/initial-motion change |
| Full candidate, adding the selected acquisition change | Measure the additional changed-grasp effect |

Each placement receives its own complete open-jaw replay of its arm commands.
Each scene also receives the common open-jaw initial-arm hold control. This gives
seven variants and 84 attempts for the existing twelve-scene cohort. If only one
change is selected, omit the redundant candidate and preregister five variants
(60 attempts) instead of running an identical ablation. If two different carry
ideas are both included, isolate them during development or explicitly add the
necessary ablation; do not attribute their combined result to either one alone.

Use identical native models, starting states and full duration within every scene.
Keep the three friction values, two masses and two shared XY layouts per cohort.
The source hash may change when new modes are added to dispatch; the old
controller's native commands and states must still reproduce the prior behavior.

Completion rate and gained/lost matched completions are primary. Report all
friction/mass/layout strata, including those that remain impossible for the
selected grasp. Compare completion time only on common successes. Report new
success times separately. Report all unresolved holds, retries and aborts;
faster aborts have their own measure and never enter completion improvement.

## Development, freeze and confirmation

1. Retain every candidate and failed native development rollout on published
   scenes, with source/config identity and the reason each change was kept or
   rejected. A failed approach is evidence, not an artifact to delete.
2. Confirm old-mode native commands, qpos, qvel and task outcomes on the two
   published cohorts. Preserve the old success cases. Freeze selected behavior,
   all thresholds, task/model/scorer, metrics and inspector before confirmation.
3. Build and verify an installed wheel from the frozen source. Seal the exact
   source/config/model identities and the predetermined fresh scene protocol.
   The release owner reserves fresh seeds independently of development.
4. Prefer three reserved fresh twelve-scene cohorts, run without tuning between
   cohorts, if the run/publication budget supports it. Keep separate bounded
   bundles and report each cohort and the combined matched counts. One cohort
   remains a useful pilot, but gives substantially less evidence.
5. Preserve all outcomes. A change made after viewing any confirmation result
   starts a new version/development cycle; the viewed cohort is no longer fresh.

The twelve scenes within one cohort share only two independently sampled layouts.
Mass/friction strata and repeated deterministic simulations are not twelve
independent real-world trials. Do not attach an IID confidence interval or a
real-world reliability probability to these counts. Multiple fresh layouts test
this narrow distribution, not object-shape, sensor or hardware generalization.

## Release gates and measurement

Separate correct execution from effectiveness. A technically valid negative
result can be published, but should not be described as a reliability improvement.

| Gate or metric | Requirement |
| --- | --- |
| Full placement reliability | At least one gained completion and no lost reference completions on the preregistered fresh comparison before calling the candidate an improvement |
| Known successful cases | No lost prior successful placements on the two development cohorts |
| Negative controls | Zero successes for every own-tape open-jaw and hold control |
| Native integrity | No invalid physics, solver warnings or prohibited native contact ticks in the claimed clean release; preserve and report any violation |
| Actuation and model | Same model digest within pairs; legal units/ranges; no force/gain/scorer change or object attachment/reset during an episode |
| New probe/regrasp decisions | Recorded causal prerequisites, bounded budgets, original qualification gates and no unsupported opening |
| Grasp and retention | Qualification, full lift, retention-loss latch, supported arrival and final success shown separately |
| Actual motion | Sampled TCP/object speed and travel, hold/probe overshoot, object fall and contact-loss duration; command scaling is not actual speed |
| Contact/force | Native prohibited-contact ticks and maximum pad force; retain actuator-force traces without claiming calibrated damage limits |
| Compute | Per-mode raw decision durations, p50/p95/p99/max and count/fraction exceeding 20 ms; wall time and simulated-seconds/wall-second throughput |
| Runtime budget | Investigate any new 20 ms computation overrun against matched baseline on the same host; p95 below 20 ms is a soft compute gate, not a physical real-time guarantee |
| Memory | Measured peak RSS and its process-lifetime/inspection scope; do not present it as minimum system RAM |
| Provenance | Source/model hashes, dependency and host versions, full commands/states, bounded sealed inventory, retained failures and exact publication verification |

Do not average away a failed stratum or compare a faster subset with all baseline
successes. Peak force and velocity comparisons are diagnostic unless a physical
limit is independently justified. A clean nominal simulation is not a safety
certificate. Current observations remain privileged native state; recorded
execution latency excludes camera, estimator, communication and physical
sensor-to-actuator delay.

## Lean implementation and meaningful checks

Reuse `piper_task.run_episode`, the pinned asset verification, the existing native
episode checks, bounded gzip decoder, raw timing validation, completion timestamp
corroboration and motion-window calculations. Reuse the paired-outcome semantics.
Add only the new protocol and explicit causal decision audit needed for v0.5.

The v0.4 runner/inspector hardcodes variant names, 12 scenes, 84 attempts, 89 sealed
entries, seeded scene construction and supported event types. Do not monkeypatch
its module globals or weaken its old checks to accept a new receipt. If extracting
common mechanics, keep old public entry points/schema validation unchanged and
regression-test their published bundles. Any compatibility view must follow
independent verification of the actual new movements; it cannot erase a probe,
fabricate a hold or substitute invented native states.

Meaningful controller tests exercise insufficient history, an unloaded pad,
table support, low-but-moving versus low-and-settled observations, persistent
loss, a single-frame glitch, probe timeout/travel/retry limits, supported-only
opening, reacquisition and the exact qualification boundary. Native tests must
actually enter the selected intervention, not merely finish a nominal episode.

Inspector mutation tests should alter a probe prerequisite or target, remove its
transition, forge a qualification/outcome, use another mode's arm tape, omit a
failed attempt or timing sample, change the model/config and reseal the bundle.
Each forged receipt must still be rejected. At least one completed native probe
and one refused/failed probe must pass unchanged. A hash-only check or a test that
copies the controller's own success label is insufficient.

## Disturbance follow-up

Start with tests justified by the observed mechanism. Small initial-layout changes
are already part of the locked cohort. Known-scene developer checks should also
exercise a short bilateral-contact dropout and supported-versus-suspended stable
history to validate the new branch's causal decisions and bounded refusal. These
observation-sequence tests establish policy behavior only, not physical robustness.

After nominal gains, add a separately preregistered native disturbance cohort:
a bounded horizontal load pulse during lifted carry, with the same pulse at the
same simulation time in each matched variant and its controls. Define magnitude,
duration, axis, timing and force application before viewing results; retain every
applied native force. Use a dedicated simulator input path, never qpos/qvel edits,
and check the force is zero outside the prescribed window. No such disturbance
path is established by the current v0.4 receipts. Do not mix these trials into the
nominal denominator or claim disturbance robustness without running them.

Sensor delay, noisy depth, deformable shoes, friction calibration, actuator
backlash, electrical limits and physical emergency stopping need additional
models and hardware evidence. They are subsequent experiments, not conclusions
of this rigid-block reliability release.

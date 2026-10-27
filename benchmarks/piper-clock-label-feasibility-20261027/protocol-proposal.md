# PiPER grip v0.3 protocol proposal (draft)

**Status: proposal for review; not frozen.** No v0.3 physics or training is included here. The v0.2 confirmation is already opened and is regression evidence only. Any qualification claim requires a newly generated, hash-bound protocol, selected model, and fresh confirmation starts.

## Question and scope

Can a small learned progress clock improve the frozen PiPER movement path under a verified post-grasp jaw disturbance? Keep the pinned six-axis PiPER, coupled jaw, 40 g rigid block, native servo/rate limits, two trained destinations, and v0.2 start envelope. The first study is clock-only: preserve the v0.2 six-joint path references and geometry. Do not alter clearance, add a camera or new arm/object, use human video, train unrestricted joint residuals, or claim physical transfer. Treat clearance as a follow-up with its own path version and benchmark.

Compare the same-path feedforward reference, a learned clock, and a matched blind clock learner. Also report the authored teacher under the same task/disturbance schedule as an anchor, not as the learned-policy comparator. The blind learner uses the same architecture, labels, episode budget and fitting recipe, with live grip/history inputs masked. Keep the frozen v0.2 jaw model, target choices, rate limits and jaw-feature contract byte-for-byte identical across all three learned/reference variants; only the learned clock may change arm-path timing.

## Inputs, clocks, and bounded action

Keep the policy input contract causal and finite: reset XY, known goal, physical episode time, frozen nominal jaw reference, and the v0.2 current/40 ms/120 ms grip snapshots (object-relative-to-tool pose, goal-relative-to-object vector, object velocity, measured jaw positions, pad forces, finger contacts, and support state). Freeze ordered feature names, units, normalization, sample age/missing handling, and model/API hashes. Exclude future frames, task-phase labels, intervention IDs, success labels, teacher state, IK, absolute tool pose, joint state, and actuator-control history from the learned clock.

The arm clock has one virtual progress state `p` in `[0, 1]`, initialized to zero and constrained to be nondecreasing. Each 20 ms step may hold progress or advance it by a bounded increment; the maximum increment/rate must be declared and validated before training. This permits a hold during grasp acquisition. A separate finite episode timeout bounds indefinite holds; terminal settling uses the existing fixed scorer window and does not extend the learned path. A zero-progress episode that reaches timeout is a failed attempt, not a silently skipped label.

Keep **physical episode time** and **virtual path progress** as distinct variables. The frozen jaw model continues to receive the same physical-time and nominal-jaw-reference values as v0.2; virtual progress must not replace, rescale, or be added to its time/jaw features. Record both clocks and their alignment at every step. Confirm in development that clock-induced timing changes do not create an unintended jaw-feature distribution shift; do not retrain or tune the jaw model to compensate. If the frozen jaw model cannot be held identical or its input distribution becomes invalid under the new timing, stop and revise the design before confirmation.

## Open implementation requirement: define and validate the clock target

The existing authored teacher may not expose a scalar progress variable along the frozen path. Do not assume its native joint target or command directly supplies a progress-increment label. Before collecting training labels, implement and validate an offline monotonic path-alignment procedure. A candidate is projection of recorded causal TCP positions onto the frozen reference path to obtain scalar `p`; the method must specify path parameterization, projection window, tie-breaking, backtracking/jump handling, and treatment of contact-induced deviations. This projected value is an offline label only, never a student input.

Development must show that the projection is identifiable and stable on teacher trajectories, including acquisition holds and failures: quantify projection ambiguity, false forward jumps, apparent backtracking, and label noise, and compare labels against any explicit reference progress if one can be logged. Define in advance the acceptance limits and a rule for ineligible/ambiguous intervals. If alignment cannot be validated, the clock-learning study does not proceed; do not substitute future state or task-phase labels as model inputs.

Once the target is validated, align each label to one 20 ms boundary: use only the causal observation available immediately before command issue, and label the bounded nonnegative progress increment for that interval. Store timestamp, physical time, virtual progress, state age, requested/applied jaw target, measured jaw response, contact/support signals, and the next-boundary state for audit only. No future frame, future phase, or eventual outcome may enter the policy features. Failed acquisitions remain finite sequences with explicit termination reasons; never synthesize labels after the last observed decision.

## Data, disturbance, and denominators

Collect a fixed declared budget of synthetic authored-teacher episodes only after the target and logger are frozen. Fit the clock and matched blind models from the same training rows/weights. Teacher episodes used for labels are not evaluation trials. No DAgger or additional label collection after fresh confirmation is opened; adaptive collection belongs to a separately versioned development lineage.

Use nominal and post-grasp-interruption strata in every partition. Schedule the same disturbance per paired case at a fixed delay after verified bilateral loaded grip: both pad contacts and positive normal force sustained at least 100 ms while the block is lifted clear of the table by a declared threshold. Apply a bounded jaw-target opening pulse only after eligibility and preserve native rate limits. Freeze delay, pulse magnitude/duration, lift threshold, and jaw interval using development evidence. Record requested and applied jaw targets, and count actual applied-target changes separately from scheduled and eligible events.

For every policy/stratum report scheduled attempts, valid attempts, acquisition failures, eligible post-grasp attempts, disturbance activations, activations with measured applied-target change, loaded-contact loss, loaded-contact restoration, and strict successes. Primary success denominator is **all scheduled attempts**, including failures to acquire and cases where the disturbance is ineligible or does not activate. Conditional recovery among actually activated cases is secondary and must retain its own denominator. Every nonactivation has a reason; scheduled-but-unactivated cases are not called disturbed trials.

## Partitions and lineage

Group by whole initial XY position; keep both goals, nominal/interruption conditions, and paired controller variants together within each group. Establish disjoint training, development, and fresh confirmation groups before data collection. Development is tuning evidence permanently. Seal confirmation starts, disturbance timing, random seeds, and policy/protocol/runtime hashes until candidate and thresholds are frozen. The v0.2 opened cases and all v0.3 development cases are regression/tuning evidence only.

Determine confirmation size prospectively after the success endpoint, expected paired effect, and acceptable uncertainty are specified. Do not imply that 48 cases per variant or another convenient count has adequate precision without that calculation. Do not stop early for favorable results. Retain and independently rescore every scheduled attempt and failed prefix.

## Outcomes and reporting

**Primary outcome:** strict task success over all scheduled attempts using the frozen native scorer: bilateral lift ≥6 cm held ≥100 ms, retained carry with no ≥40 ms loaded-contact-loss gap before supported destination contact, release, and settled endpoint within 25 mm for 250 ms. Report paired differences against same-path feedforward and matched blind clock controls with confidence intervals and per-start results.

Report terminal-only settling separately from strict success; include acquisition failure, eligible/activated disturbance counts, loaded-contact loss/restoration, nominal false intervention, finger-table contacts, collisions, invalid traces, and aborts. Keep all rates denominator-explicit. Clock is the only changed learned factor in this study; no clearance effect can be claimed.

Measure inference and full control-step latency on named hardware with dense logging enabled. Report p50/p95/p99/max, count and fraction above 20 ms, dropped/late steps, and wall time. State that camera, transport, and physical actuator latency are excluded. Simulator timing does not establish a hard real-time guarantee.

## Provisional stop/go criteria

All numerical gates remain **provisional** and require review before protocol freeze; no precision or power claim is made by this draft.

Potential go to a follow-up simulation study only if the predeclared minimum fraction of interrupted scheduled cases reaches a verified post-grasp applied disturbance; the learned clock improves strict all-attempt success versus both feedforward and blind controls by the predeclared meaningful margin and uncertainty criterion; nominal success and false-intervention limits are met; collision/table-contact rates do not worsen; invalid/aborted traces remain below a predeclared cap and stay in denominators; and full-step deadline tails meet a predeclared host-specific limit. Set these values after development and a prospective precision review, before confirmation is unsealed.

Stop or redesign if clock labels fail the alignment checks, the intervention rarely activates after established grip, the learned clock does not beat both controls, nominal performance regresses beyond the frozen limit, collisions/contact worsen, or trace/latency limits fail. A failed gate is diagnostic evidence; do not broaden the model or reuse confirmation cases to tune it.

## Separate clearance follow-up

A Cartesian clearance offset changes path geometry and generally requires a separately derived joint-space path or IK solution. It cannot be described as preserving all six frozen arm targets. Any follow-up that changes clearance must assign a new path/version hash, declare how targets are generated and bounded for joints/workspace/collisions, and evaluate it as a separate fixed-path condition. Only after the clock target and timing/jaw coupling are resolved should a predeclared timing-by-path-version comparison be considered; keep the exact frozen v0.2 geometry as its control and do not tune either path from fresh confirmation.

# Camera-to-action v0.2: proposed control-learning ablations

## Status addendum — 9 November 2026

The original 1 November proposal is preserved below as a dated record. Its
statements that v0.1 results were pending and no v0.2 work had run describe that
original evidence cutoff, not the present state.

The completed v0.1 study reports **5/64 strict learned placements**, versus
9/64 matched blind, 31/64 fixed primitive and 64/64 authored teacher. The model
remains an unqualified research candidate. The resumed independent audit passed
with zero errors across all 452 scheduled attempts, 452 numeric outcome
reconstructions and 2,384,532 verified files; it executed no physics, estimator
or model. The audit receipt is `full-study-audit-resumed-20261109.json`, SHA-256
`2774959f8e1a5be7283cea129da0ffea000d8375eb445140c8c7a60ab15efed8`.
This byte/clock/authority/outcome audit is not source authentication or hardware
qualification. The interrupted earlier audit output is preserved separately.

Two additive pieces of groundwork now exist:

- The [numerical limiter preflight](camera-action-v02-preflight.md) passes the
  Cabs/CΔ reference-equivalence prerequisite on 4,242 bounded numerical cases
  under a predeclared 1e-12 native-unit tolerance. Closed-form float64 projection
  is not universally bitwise identical. No model was trained or evaluated.
- The [verified export cache](camera-action-export-cache-v01.md) reduces repeated
  preparation cost on one measured training episode while preserving numeric
  features, targets and dataset identity. Hits verify derived cached bytes, not
  raw recordings; this is not a control-performance gain.

The five-candidate, **164-attempt learning screen below remains unrun**. Its
proposal has not been converted into a registration or qualification result.
Matching model-head initialization, training-only normalization and fresh group
registration remain required. Neither new tool changes the frozen v0.1 sources,
models, scenes, confirmation results or gates.

## Original proposal — 1 November 2026

**Proposal only, 2026-11-01. Nothing below is implemented, run, or registered.**
The v0.1 correction and confirmation results are pending at this proposal's
evidence cutoff. Finish, audit and report that study unchanged. This document
must not change its models, starts, budgets, selection or qualification gates.

The first scheduled initial full-student development case diverged from its
matching teacher by 10 mm at 0.409 s, before its first missing object estimate
at 1.56 s. It commanded a single-finger target below 20 mm at 1.20 s while
133 mm horizontally from the block; that command did not mean its fingers had
already closed. It rate-limited 977/1,000 intervals versus the teacher's 75,
never acquired bilateral contact, and accumulated 1,427 critical-contact ticks.
These are observations from one sealed local diagnostic, not a full-study
audit, failure-frequency estimate, or causal proof. Covariate shift, target
scaling and command representation remain competing explanations.
The student renderer also warned that `ARB_clip_control` was unavailable for
`mjDEPTH_ZEROFAR` and that depth accuracy would be limited. Repeated warning records are not
accuracy measurements, and this is not evidence that the warning caused the
failure. Keep it visible as a depth-validation limitation; do not change the
frozen renderer or treat exact camera transforms as validated depth.

## What prior work supports

DAgger collects expert labels on states visited by the learner and aggregates
them with demonstrations. It explicitly permits mixtures of expert and learner
control; repeatedly visiting only expert states does not test the same
distribution. Its theoretical guarantees require assumptions and an iterative
procedure that our small neural experiment does not establish. v0.1's single
50% mixture round is already a useful first test; it is not yet a result.
[Ross, Gordon and Bagnell, AISTATS 2011](https://proceedings.mlr.press/v15/ross11a.html).

A position target is a servo reference, not a measured pose or an instantaneous
movement. MuJoCo separates control/reference inputs from generated actuator
force. Its motor documentation also distinguishes reference slew limits from
setpoint bounds and torque limits. Our current experiment uses its existing
position servos and software limiter, **not** that documented DC-motor model.
[MuJoCo actuation](https://mujoco.readthedocs.io/en/stable/computation/index.html#actuation-model),
[DC Motor Model, §2.5.3–2.5.5](https://mujoco.readthedocs.io/en/3.7.0/_static/dcmotor.pdf).

Action components can use different normalization and must retain identical
ordering, units and decoding in training and deployment. That motivates a
controlled normalization comparison, not a claim that one scale is universally
better. Residual-control research combines a conventional controller with a
learned correction; its RL results do not establish that our supervised
increment predictor will work. A previous-reference increment is also distinct
from a residual around a task-solving controller.
[robomimic action configuration](https://robomimic.github.io/docs/tutorials/action_configs.html),
[Johannink et al., Residual Reinforcement Learning for Robot Control](https://arxiv.org/abs/1812.03201).

## Three contrasts, registered before any new rollout

Use five explicitly named full-feedback candidates. **B0** is the common
control: original 64 training demonstrations plus the sealed 32 v0.1 mixed
episodes, absolute issued/query targets, and v0.1 output whitening. Fit it anew
under the v0.2 manifest; it is not automatically the selected v0.1 checkpoint.
Freeze feature normalization and the baseline output statistics from this
training set for the coverage comparison. Other changes below are separate;
do not combine winning choices in this experiment.

| Contrast | Single question and intervention | Evidence against the proposed explanation |
| --- | --- | --- |
| **A versus B0: learner-state coverage** | Replace the 32 mixed episodes with 32 pure-student episodes on the exact same registered training cases, using the same sealed **initial** full checkpoint that generated v0.1's mixtures. Query the stateful teacher once per decision for labels; execute only the learner. Fit on original 64 + new 32, with identical sample and optimizer budgets. | More learner exposure without improved held-out completion weakens the claim that the 50% intervention rate was the principal bottleneck. If the confidence interval excludes the proposed practical improvement, reject that effect size for this bounded recipe. |
| **B versus B0: output scaling** | Same data and absolute targets; keep the training mean but remove empirical output standard-deviation division. Use the existing fixed scales `[3 rad × 6, 0.035 m]`. Keep feature processing unchanged. | Lower normalized training loss alone does not count. No closed-loop improvement under matched native-unit scoring rejects a practical benefit from this change at the stated budget. |
| **CΔ versus Cabs: reference-relative control** | Same B0 data; both learn the *same* feasible next-reference labels. Cabs predicts the absolute endpoint; CΔ predicts its increment from the preceding applied reference. Both use the same physical-scale endpoint loss and identical final feasibility projection. | If only the reference bounds improve metrics, both candidates should benefit similarly; CΔ must beat Cabs to support the representation hypothesis. Reduced command clipping alone is not evidence of better manipulation. |

For the third contrast, let `r` be the preceding applied servo reference,
`a*` the teacher target, and `P(r, a*, 20 ms)` the exact existing limiter advanced
analytically through its 20 native ticks. Set `r* = P(r, a*, 20 ms)`.
Cabs predicts `r*`; CΔ predicts `(r* − r) / (rate_limit × 20 ms)` and decodes
relative to **r**, never to a future reference or measured joint position.
Use the same seeded hidden layers and analytically scale/offset the final heads
so their decoded outputs match on every registered reset context before fitting;
verify this in a numerical preflight, without rollout or hyperparameter search.
Report any differences on later training states explicitly. This matches initial
command bias, not the function classes or their subsequent feedback dynamics.
Before collection, require a pure numerical equivalence check that issuing
the projected endpoint reproduces all 20 teacher-reference microsteps; if it
does not, block this contrast and report the reason. Preserve original joint,
jaw, force and rate limits. Endpoint feasibility does not bound actual joint
velocity or guarantee safe contact. A residual around the fixed task primitive
is deferred: it would test a hybrid controller, not complete learned control.

All candidates retain causal 160 ms history, honest missing masks, historical
initial anchor and the original input whitelist. Teacher phase, current query
target, native object/contact truth and future observations remain excluded.
Native truth may score attempts offline. New query labels stay separate from
executed actions. Failures and interrupted prefixes are retained; there are no
replacement scenes or fallback teacher actions. Incomplete or unequal usable
training-label counts block the affected comparison rather than silently
changing its budget. The stateful teacher's suggestions off its usual
trajectory are supervision, not proof of optimal recovery.

## Bounded screen and falsifiable decision

Proposed maximum: **32 new training rollouts**, five CPU NumPy 2×64 fits
(seed 17, 50 epochs, batch 512, Adam 0.003), and **80 learned evaluation
attempts**: eight fresh start groups × two goals × five candidates. Add the
teacher and fixed controller on the same 16 cases (**32 attempts**), and four
existing negative-control types once per candidate (**20 attempts**).
Total: **164 new attempts**, each at most 20 simulated seconds. Reuse sealed
training data; never train on any v0.1 development or confirmation recording.
Generate the eight new groups with a separately registered seed, reject exact
overlap with all old starts before running, and freeze the resulting coordinates,
case order, source hashes and resource limits. This small screen is intentionally
underpowered for a general success claim; it awards no qualified-skill label.

Predeclare the three paired contrasts above. Primary outcome is strict complete
placement on all 16 scheduled cases, with aborted/unattempted cases counted as
failures and separately reported. A promising contrast requires at least
**two additional successes out of 16**, a positive lower bound of a **98⅓%
two-sided paired bootstrap interval** grouped by the eight starting positions
(50,000 resamples, seed 171701), no increase in either critical-contact episode
count or total critical-contact ticks, and every negative control behaving as
specified. The interval level allocates 5% across the three planned comparisons;
with eight groups it remains an approximate screen, not strong population
evidence. An upper bound below +12.5 percentage points rules out that proposed
practical improvement at this resolution; otherwise an unmet gate is
**inconclusive**, not proof of no effect. A lower prediction loss or a stationary
arm with fewer contacts cannot pass by itself.

For contextual comparisons against the fixed controller, preserve the existing
common-admission check at the operating-envelope boundaries. Report both all
16 scheduled attempts and the subset where both goals of a starting group were
admitted by both controllers; require at least six of eight complete groups
before making a fixed-comparator claim. Refusals remain failures in the scheduled
denominator. This subset never replaces the three primary learned contrasts.

Report bilateral-grasp acquisition, first-closure geometry, lift/retention,
strict success, critical contacts, issued-versus-applied target error, command
variation, perception missingness, standardized-feature clipping, training/export
time, process peak memory, inference and full-loop latency. Compare losses only
after decoding to the same native targets and fixed scales; report joint-radian
and jaw-metre errors separately. Teacher path separation is descriptive because
a different successful path is allowed. No confirmation-selected hyperparameters,
new fitting seeds, or posthoc combinations are added. Any promising candidate
needs a separately registered larger confirmation and matched blind comparison.

## Keep the next cycle local and measure its real cost

The sealed initial fits took **6.56 s full / 5.99 s blind**, while their shared
verified export took **685.44 s**. Each complete episode has **6,010 files**.
This shows that fitting the tiny MLP was not the expensive part of that measured
workflow; it does not by itself separate disk I/O from hashing, decoding and
validation. Profile those components before proposing a GPU remedy. Default
execution remains one local worker with the existing renderer requirements;
there is no cloud model, new hardware, or physical-arm command. Simulated time
is not a wall-clock duration estimate. Preflight storage for the fixed roster
and preserve the existing 30 GiB free-space reserve; stop with an explicit
incomplete result if that budget cannot be met.

A later **separate operational benchmark** can compare the existing per-file
layout with compact immutable chunks/columnar sidecars on the exact same sealed
records. Measure bytes, file opens, hash/decode/validation time and peak memory;
require lossless reconstruction, identical per-record identities and outcomes,
and rejection of modified, reordered, truncated or mismatched records. Retain
raw media hashes, frame/calibration/model bindings and atomic sealing. No format
change or audit relaxation belongs in the current frozen study.

Evidence: [v0.1 research report](camera-action-v01-report-draft.md) and
[reproducible first-case diagnosis](../benchmarks/camera-action-v01/diagnose_first_student.py).
The local diagnosis proof is retained separately until the full-study publication
binds the same episode bytes; this proposal does not claim that audit has passed.
All numerical thresholds and budgets above are proposed choices, not measured
v0.2 results or established optimum settings.

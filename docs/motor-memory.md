# Motor memory v0.2

This experiment asks whether a compact fitted actuator model can help an authored
motion primitive move faster while still settling accurately and retaining its
load. It adds local calibration, three controller comparisons, and a report that
keeps task completion, false completion, contact, sensing and computation cost
separate. The reaction framework v0.1 and its frozen evidence remain unchanged.

The learned component is **simulated actuator dynamics**: a small fit from native
calibration transitions, with a development residual envelope. The action profile
and expected outcome contract are authored metadata.
It does not learn a task from human video or establish a whole-arm robotic skill.
The environment is one horizontal travel axis and parallel jaws with synthetic
sensed state. Physical sensor, material and motor calibration are still needed.

## Train, benchmark and inspect locally

Install the current checkout with the optional native engine:

```sh
.venv/bin/python -m pip install ".[arm]"
.venv/bin/dvidia-motor-memory train --output runs/motor-memory-v0.2
.venv/bin/dvidia-motor-memory inspect runs/motor-memory-v0.2
.venv/bin/dvidia-motor-memory benchmark \
  --memory runs/motor-memory-v0.2 --output runs/motor-benchmark-v0.2 --repeats 3
.venv/bin/dvidia-motor-memory inspect runs/motor-benchmark-v0.2
```

The equivalent module entry point is `python -m dvidia_training.motor_benchmark`.
Use a new output directory for each training or benchmark run. Training and
benchmark execution run locally on a CPU using the arm extra; these fixtures
require no GPU, account, model API, camera footage or network connection. These
are dependency requirements, not measured minimum specifications for future
robots or larger models. The older frozen package and Hugging Face pilot predate
this tool.

Training saves the calibration protocol, complete calibration rows, fitted
memory, measured braking coupons, source identities and an evidence manifest.
Benchmarking freezes the candidate, source and protocol identities before trials,
saves every attempted record, and writes `report.json` plus a self-contained
offline `report.html`. Inspection checks hashes and identities and recomputes
scoring/aggregates from the saved trace. A matching hash checks consistency; it
does not authenticate physical capture or establish accuracy against hardware.
The HTML keeps every attempt's scalar outcomes, errors and completion history,
with trace count/first/last rows. Its relative links open the complete traces in
`report.json` and `trials.jsonl`; the presentation excerpt does not change that
machine-readable evidence.

## What is learned and what is supplied

The calibration fit estimates one-axis stiffness, damping, bias and response
limits from privileged native position/velocity/applied-command transitions.
Those transitions are explicit calibration evidence, distinct from deployable
sensed inputs. The predictive controller uses the resulting model with delayed
observations and acknowledged commands to reconstruct state at the current
control decision. The
uncertainty envelope is an empirical development assumption; it is not a
calibrated confidence interval or guaranteed safety bound.

Task goals, expected contacts, completion tolerance, speed tolerance, dwell and
load requirements are authored. They are not inferred from the calibration fit.
The fixture's empirical braking estimate also comes from bounded native coupons;
it does not guarantee deceleration under every load, impact or future robot.
The independent reaction monitor can request a hold when sensing/contact/slip
conditions require it. This version does not invent regripping, recovery or
automatic resumption after that hold.

## Compare three controllers fairly

| Controller | Purpose |
| --- | --- |
| Slow feedback | Conservative observation-based motion. The protocol's slow-mode factor reduces its speed below the named speed tier. |
| Fast fixed | Observation-based motion at the full tier using the fixed primitive, without the fitted predictive adjustment. |
| Adaptive predictive | Uses the compact fitted actuator model and observation/command history to adjust motion at the same full tier. |

The named speed tiers are **0.12, 0.20 and 0.28 m/s**. Each controller is evaluated
at each tier with matching case, seed and exogenous schedule. Reported speed is
the tier's configured limit; actual speed and the slow-mode factor are separate
controller behavior. The comparison exposes a speed/precision tradeoff rather
than assuming the learned controller wins.

Development cases cover distances, public goal updates, changed loads, delay and
missing input. Confirmation cases use authored combinations excluded from
calibration/development and include changed goals, heavier/slipperier loads,
longer delays, blackout and slip. The candidate and protocol are frozen before
execution. Once confirmation results are inspected, those cases become regression
evidence for further changes; another improvement needs fresh confirmation.
Repeated authored cases remain correlated and are not independent physical trials.

All controllers continue through the same fixed simulation horizon. Reaching a
goal, issuing a completion claim or requesting a reaction hold does not end
physics or shorten the measured workload. Every attempted trial—including native
errors, unsuccessful tasks and sensor-fault cases—stays in the report.

## Completion is verified, then audited

The public observed-state verifier requires position within **0.003 m**, speed
within **0.005 m/s**, and a **0.05-second dwell**. Public goal changes and
missing/stale observations reset verification. Earlier completion claims remain
in the receipt, including whether truth supported them at the time.

After a claim, simulation continues. Truth-audited success requires the final
goal's sustained settled dwell, retained load and the declared contact diagnostic
criteria through the evaluation horizon. Truth is available to scoring, separately
from deployment inputs. An appropriate reaction hold is not task success: fault
cases can show an appropriate response and an incomplete task simultaneously.

The report separately counts any false claim, revoked claim, later loss of rest,
expected-but-censored completion, native errors and unknown outcomes. An earlier
false claim stays visible even when a later goal change clears the final
`completed_s` timestamp. Contact thresholds are authored engineering diagnostics,
not physical safety thresholds; force/penetration accuracy still needs numerical
and hardware validation.

## Read the speed–precision report

Development and confirmation appear in separate sections. Each controller × speed
row displays all attempts, invalids, conditional task success, task success over
**all attempts**, false/revoked claims and censoring. Counts for expected-completion
and completion-not-expected cases stay separate in the detailed metrics.

The SVG plots show median observed completion time against median **native truth
error at that observed claim**, over the **same successful trials with no false
claim**. Fixed-horizon final error is retained separately; a long settling horizon
can make that final value nearly zero for several otherwise different controllers.
Every dot prints its measured count alongside all-attempt counts. Failed,
censored, invalid and false-claim trials cannot become zero-time points. Missing
dots are not evidence of fast completion. The complete row and receipts must be
read alongside the successful-trial plot; the plot alone cannot rank a controller.
This is an observed speed–precision frontier, not a universal optimality claim.

| Measurement | Interpretation |
| --- | --- |
| Error and speed at completion | Native truth at the last-goal-segment observed claim, plus peak error after that claim; earlier claim history stays separate. |
| Final error and speed | Native truth at the fixed horizon, with units and valid measured/missing counts. |
| RMS tracking error and overshoot | The saved trace against the public goal over the trial; goal changes belong to the protocol. |
| Completion time | All observed claims are separate from verified successful completion; null times remain censored. |
| Acceleration and jerk | Finite differences of the saved 10 ms assessment trace; sampled peaks can miss faster transients. |
| Force proxy, contact impulse and penetration | Native 1 ms interval summaries retained in the 10 ms trace; authored external forces/contact proxies are not calibrated wrist readings. |
| Load outcomes | Native retention/drop criteria for holding fixtures; null for unavailable or inapplicable outcomes. |
| State-estimation audit error | The `predictor_rmse_m` field compares reconstructed state against native truth at the same control decision time, separately from task success and only when estimates exist. |
| Reaction and unusable sensing | Reaction time/reason, missing/stale observation exposure and continued task outcomes. |
| Command delay | Modeled simulated application latency, distinct from host controller execution time. |
| Controller latency and throughput | Host p50/p95/worst call cost and all-attempt physics throughput, named host and measurement boundaries. |

Rates include 95% Wilson intervals where their denominators are nonzero. The
intervals are descriptive under a binomial assumption; repeats and related
authored scenarios do not prove broad generalization. Nearest-rank p95 from a
small sample is not a tail guarantee. Missing timing prevents a misleading
throughput ratio; invalid attempts still consumed their recorded runtime cost.
There is no single combined accuracy/speed/safety score or readiness badge.

Matched controller deltas require the same split, case, seed and speed tier. A
missing/duplicate variant or invalid pair cannot silently supply a comparison.
Completion-time deltas additionally require two truth-successful attempts with
no false claim. Other valid-attempt deltas retain success/false/revoked flags.
Dynamic metrics, state-estimation error and host costs should be assessed alongside
completion and retention rather than optimized as isolated headline numbers.

Goals are exact commanded coordinates in this coupon. Robot state is delayed or
noisy, while camera-based goal localization is outside this experiment. Comparing
three complete controller configurations cannot identify the causal contribution
of an individual fitted-model, uncertainty or governor component; that needs
matched ablations with those components varied separately.

## Limits and next integrations

The native model, sensor delay/noise, reaction assumptions and operating envelope
are authored. Impact-force convergence was unresolved in v0.1; finite engine
outputs alone do not establish physical contact accuracy. This experiment needs
material/actuator validation, wider fresh combinations and eventual measured
hardware telemetry before transfer claims.

Next adapters can provide six-axis geometry, robot-specific commanded/applied
state, gripper calibration, task-relative targets and recorded action profiles.
Each new skill still needs its own outcomes, nominal/fault cases and fixed
benchmark. Shoe organization and shoe tying require their respective task assets
and physics. The compact motor fit can support that pipeline without claiming
that fitting one actuator teaches those tasks.

See [reaction framework](reaction-framework.md), [episode tape](episode-tape.md),
[execution map](roadmap.md) and [hardware planning](hardware.md).

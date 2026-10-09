# Reaction framework v0.1

This is a local, CPU-first framework for measuring anticipation, contact reaction,
braking and load retention under declared simulated conditions. It combines a
reusable reaction monitor, a native MuJoCo linear-actuator and parallel-jaw coupon
fixture, and an evidence report. It does not modify the existing frozen arm
placement benchmark or promote an installable skill to physical qualification.

The fixture supplies **synthetic sensed state**. It does not read RGB images,
detect objects from video, learn a policy, model an actual camera's accuracy, or
protect every link of a six-axis arm. Whole-arm geometry, calibrated sensing and
physical stopping measurements need separate adapters and evaluations.

## Run locally

Install the current source checkout with the optional native engine first:

```sh
.venv/bin/python -m pip install ".[arm]"
.venv/bin/dvidia-reaction benchmark --output runs/reaction-v0.1 --repeats 3 --convergence
.venv/bin/dvidia-reaction inspect runs/reaction-v0.1
```

The module entry point is equivalent:

```sh
.venv/bin/python -m dvidia_training.reaction_benchmark benchmark \
  --output runs/reaction-v0.1 --repeats 3 --convergence
```

Use a new output directory for each run. The report contains the declared
protocol, engine/source versions, machine and runtime boundaries, every attempted
episode, and summary measurements. Its self-contained HTML can be read offline;
`report.html` is the readable result and `report.json` is the machine-readable
evidence. `protocol.json`, its hash, source hashes, append-only `trials.jsonl`
and the evidence manifest preserve the run's identities and attempted trials.
Report generation and metric
aggregation use Python's standard library. Native fixture execution additionally
requires the arm extra. No GPU, cloud account, model API or footage download is
needed for these fixtures.

The package release version and this framework's protocol version are separate.
Use the current checkout; the earlier frozen package and Hugging Face pilot do
not contain this framework. These requirements are dependency requirements, not
a verified minimum RAM/CPU specification for arbitrary future environments.

## What the monitor measures

The monitor consumes timestamped sensed state and an explicit validity/age
contract. Its decisions can reflect predicted clearance, unexpected contact,
grip slip and stale or missing observations. The fixture's known physical state
is retained separately for outcome auditing. Sensor delay and dropout operate on
the sensed input, instead of freezing the simulated world or inventing fresh
measurements.

The scheduled disturbance and positive/negative labels are declared before a
trial. A positive collision-course case stays positive when braking avoids
contact. Labels therefore do not depend on the outcome of the intervention.
Cases are separated into motion, contact, grip and sensor-fault strata. A sensor
fault can itself be a declared positive condition: detection accuracy always
refers to the protocol's authored trigger, rather than a universal definition of
danger.

Each case is run with the monitor enabled and a matched `disabled_control`
baseline. The baseline still evaluates the monitor in shadow mode to preserve
its detection decision, but does not request or apply monitor braking. Its
braking/stopping fields remain null when no braking intervention occurs. Detection
accuracy measures the decision; differences in contact, penetration and retention
measure the intervention's effect. Exogenous disturbance schedules and sensor
seeds match up to intervention; trajectories and later sensed geometry can then
diverge. Repeats of one fixture are not new independent physical demonstrations.

## A stop request continues physics

The fixture keeps stepping the native engine after a brake request. Command
application delay, finite actuator response, contact and the held object's motion
remain observable. Ending an episode is not counted as an instantaneous stop.
Rest requires the measured carriage joint/tool translation speed below the protocol's threshold for a
sustained **0.05-second dwell**. The thresholds, horizon and actuator settings
belong in the saved protocol. `stopped_s` records completion of that dwell; a
later loss of rest is separately counted rather than hidden by the first stop.
The hold reference assumes an ideal local encoder captured at brake application;
it is a separate assumption from the monitor's delayed/noisy motion tracks.

Record the available event chain separately:

1. Authored hazard/fault onset.
2. Sensor sample and delivery.
3. Monitor decision and brake request.
4. Actual command application.
5. Sustained rest, or an unstopped/censored result at the horizon.

The first interval can be negative when an anticipatory decision precedes the
declared onset; interpret it against that fixture's definition. The ordered
sample → delivery → decision → request → application → rest chain must remain
chronologically coherent. Missing events remain null. An unstopped episode does
not contribute a zero to a stopping-time distribution. Sensor sample/delivery
timestamps can be negative in the rollout-relative clock when they came from
the sensor priming period before rollout.

## Measurement contract

| Question | Evidence |
| --- | --- |
| Did it recognize the declared condition? | Episode-level TP, FP, TN and FN; positive/negative denominators; accuracy, precision, recall and false-positive rate with 95% Wilson intervals. |
| Did difficult trials disappear from the denominator? | Every attempt, invalid positive/negative/unknown counts, unknown labels, errors, and valid-correct decisions divided by all attempted labeled cases alongside conditional accuracy. |
| How quickly did it react? | Onset-to-detection, sample/delivery/decision/request/application intervals and application-to-sustained-rest; median, nearest-rank p95, worst and measured/missing counts in simulated seconds. |
| How far did it move while braking? | Path length from brake application to rest, net displacement, request-to-rest path including command-delay travel, maximum forward excursion from the request, and censored/unstopped counts. |
| What happened at contact? | Peak unexpected force, integrated unexpected-contact impulse, minimum clearance and maximum penetration when available, with units and sample counts. Grip/contact definitions belong to each fixture. |
| Did it retain its load? | Drop and retention outcomes for holding fixtures; null for unmeasured or inapplicable load outcomes. |
| Did sensing become unusable? | Observation availability, missing/blackout, stale and unusable durations when available; stale/missing input fixtures and their outcomes. |
| How expensive was the computation? | All-attempt wall time, native steps per wall second and simulated seconds per wall second, named host/runtime, repeats and timing boundaries. Missing timing prevents a misleading aggregate ratio. |

Each per-field distribution exposes its measured and missing counts. Invalid
episodes do not enter valid accuracy or physical-measurement numerators, but stay
in attempted counts and recorded runtime cost. Unknown labels cannot be silently
treated as negatives. The valid-correct/all-labeled-attempts fraction exposes the
effect of excluded invalid labeled trials; it is not a safety or readiness score.

The exposure-normalized false-positive statistic counts **negative episodes
with an alarm per simulated hour**, at most one decision per episode. It is not a
continuous-stream false-alarm frequency. A dedicated long-duration stream test
would need alarm episode boundaries, reset/refractory rules and exposure time.

Wilson intervals are descriptive uncertainty estimates under a binomial sampling
assumption. A small authored fixture suite, repeated seeds and related scenario
variants do not establish broad operating coverage. Nearest-rank p95 from a small
sample does not establish a reliable tail bound. No composite safety score,
hardware-readiness badge or universal reaction-time target is computed.

## Accuracy and speed boundaries

`valid` means the native state remained finite and emitted no solver warning.
It does not mean contact forces are physically credible. The frozen developmental
protocol separately flags penetration above 3 mm or external-force proxy above
1,000 N. Those are authored diagnostic thresholds, not calibrated material
properties or permissible hardware forces. Failed diagnostics remain visible
even when the condition was correctly detected and the carriage eventually stopped.

`--convergence` additionally runs eight attempts on the crossing and late-entry
cases at 1 ms and 0.5 ms native physics ticks. Sensor acquisition stays at 10 ms,
reaction polling at 1 ms, and command delay, horizons and actor trajectories stay
fixed. `convergence.json` retains every attempt and paired changes in event timing,
stopping path, force, impulse, penetration and clearance. The report shows separate
timing and contact agreement against the saved developmental tolerances.
Agreement at two timesteps is a numerical diagnostic; it does not establish
hardware accuracy. Impact-force disagreement is a reason to refine the fixture,
contact model or integration cadence before making precise force claims.

The external-force proxy combines authored carriage pushes and native obstacle
contact force magnitudes. It is not a simulated calibrated wrist transducer.
Slip is measured as payload velocity relative to the carriage in the x/z plane,
tangential to the y-normal pads. Grip availability comes from native bilateral
pad contact; the task's obligation to retain a load is supplied separately.
Stopping horizontal motion alone does not provide a regrip or set-down recovery.

CPU monitor-call p50/p95/maximum latency and total update time are recorded
separately from simulated reaction timing. Instrumentation, sensor production,
physics, scoring and trace retention consume wall time too. These measured
call latencies are observations, not worst-case execution-time guarantees.
Evidence inspection checks complete planned attempts, byte identities and
recomputed measurements; self-consistent hashes do not authenticate provenance.

Measured distances, force and timing are native-engine outputs under authored
assets and parameters. They are not comparisons against calibrated hardware
truth. Numerical validation needs analytic/contact coupon fixtures, timestep
convergence, and eventually measured material, sensor and actuator responses.
The same control protocol should be retained when comparing native tick rates or
implementation changes. Reducing accuracy checks to increase throughput changes
the experiment.

Simulation time describes modeled sensor/command delay and physical evolution.
Host wall time describes how quickly this implementation runs on the recorded
machine. Neither is an end-to-end physical-robot reaction-time measurement.
Episode throughput has an explicit boundary; process startup/imports, model
construction, report writing and peak-memory measurements must be identified
separately when measured. A single-host run is not a verified hardware minimum.

## How it becomes part of an installable skill

The next integration supplies each supported robot's collision geometry, sensor
and action contract, stopping behavior and gripper limits. Each skill then adds
its own expected contacts, clearance envelope, holding requirements, recoveries
and fault cases. The reaction report can be bound to the skill's versioned
benchmark receipt and source hashes. Passing a fixture suite cannot stand in for
task success or physical qualification.

Keep developmental and fresh confirmation scenarios distinct. Once results have
been inspected, they are regression evidence for subsequent changes. Freeze a
candidate and protocol before opening fresh confirmation cases. Real robot
telemetry and supervised hardware testing are separate later stages.

Related contracts: [execution map](roadmap.md), [episode tape](episode-tape.md),
[skill benchmark protocol](benchmarks.md), and [hardware planning](hardware.md).

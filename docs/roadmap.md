# Execution map: footage to installable skills

Working map, updated 31 October 2026. This is the team's sequence of builds and evidence
gates. A completed software component does not establish robot competence.

The new [task environment studio preview](https://github.com/Dvidia-Inference/dvidia-training/releases/tag/task-environments-v0.2-preview), **0.1.3.dev2**, adds a local browser editor and `dvidia-environment` Terminal workflow for CPU physics with a pinned PiPER, an open bin and multiple original rigid objects. The frozen passive checks accept 6 completion fixtures, reject 6 negative fixtures and refuse 4 invalid layouts. PiPER holds home; these results establish environment and verifier infrastructure, with no learned manipulation skill. Stable default-branch SDK 0.1.2 remains separate.
[Local UI/Terminal guide](https://github.com/Dvidia-Inference/dvidia-training/blob/task-environments-v0.2-preview/docs/task-environments-v02.md) · [Studio and recorded benchmark](https://dvidia-robot-simulation-lab.static.hf.space/task-environments/v02/index.html) · [Complete evidence](https://huggingface.co/datasets/Dvidia/task-environments-v02) · [Research](https://research.dvidia.org/papers/task-environments-v02/).

## Current priority: collect usable corrective action evidence

The [action-demonstration recorder](action-demonstrations.md) now preserves causal
PiPER observations and motor targets with explicit native coordinates, missingness
and outcomes; 27 focused tests pass. Stable SDK 0.1.2 provides that data-only
contract and Terminal tool; its example remains synthetic metadata.

The separate [native capture preview](https://github.com/Dvidia-Inference/dvidia-training/releases/tag/piper-action-capture-v0.1-preview), 0.1.3.dev1, now records genuine rendered RGB-D, measured encoders and native commands. Two identical 20-second runs preserve 1,000 intervals each; an interrupted prefix preserves two intervals with an unknown outcome. All 37 focused tests and a fresh installed native smoke pass. These are authored-controller repeats of one fixed start, without operator corrections. Four object-tracker channels remain missing, so strict training export refuses; depth accuracy remains unverified.

A separate [OpenUSD workstation](https://research.dvidia.org/papers/openusd-workstation-v01/)
passes all 45 frozen passive physics checks, with a local installed-package smoke.
It validates an original rigid-object scene bridge, not a manipulation policy.
Recovered [bounded grip waiting](https://research.dvidia.org/papers/bounded-grip-waiting-v04/)
completes 42/128 scheduled attempts versus 32/128 learned clock and 44/128 fixed.
Both paired comparisons are inconclusive; retention and critical contacts remain.

Next: validate a frozen camera estimator, collect and review corrective segments,
verify their replay evidence, then fit a small action adapter. Keep repeated
trajectories together when assigning train/test groups. A future camera-cadence
sweep must preserve acquisition age rather than invent fresh frames.
The [proposed next experiment](piper-next-learning-experiment.md) fixes held-out
groups, matched post-lift disturbances, blind/fixed controls and strict physical
outcomes before execution. It remains unrun. Human-video motor recovery, other
embodiments and physical-arm qualification remain separate gates.

## The product loop

```mermaid
flowchart LR
  A[Define skill goal and operating envelope] --> B[Specify its simulation benchmark]
  B --> C[Request recordings that fill coverage gaps]
  C --> D[Review footage, rights, outcomes and source groups]
  D --> E[Freeze development and confirmation groups]
  E --> F[Train compatible candidates locally]
  F --> G[Evaluate every attempted simulation trial]
  G --> H{Meets the declared task criteria?}
  H -->|No| I[Diagnose development failures and request new evidence]
  I --> C
  H -->|Yes| J[Publish versioned candidate, compatibility and results]
  J --> K[Install through a compatible adapter and calibrate]
  K --> L[Separate physical robot qualification]
```

Confirmation evidence is opened after a candidate is frozen. Once inspected,
that suite becomes regression evidence for later changes. Further qualification
needs fresh confirmation scenarios; it cannot silently reuse a known test set.

## Earlier foundation map, 9 October

| Order | Build and owner role | Current state | Evidence needed to advance |
| --- | --- | --- | --- |
| 1 | Collection planning — product + data | Local coverage planner, versioned briefs and declared source-group tallies implemented. | First skill's observable goal, operating envelope, actual recordings and reviewed capture lineage. |
| 2 | Contribution review — data + product | Proposed official-site connection; the local planner imports declarations only. | Immutable review receipts tied to original bytes, plan revision, intended-use permission and complete outcomes; source-group and near-duplicate audits. |
| 3 | Episode evidence — simulation | Recorder preserves state, requests, applied commands, sensor age and authored phases at control boundaries. | Task-specific episode corpus, retained failure cases and synchronized inspector. Tapes are not consumed by a learner yet. |
| 4 | Task and state models — learning | Existing small visual predictor and bounded movement head; outcome/phase supervisor proposed. | Choose deployable sensor inputs, compare with simple baselines and measure held-out boundary, outcome and false-completion errors. Keep authored labels and privileged state outside deployment inputs. |
| 5 | Task simulation — simulation | Existing rigid-box placement adapter and frozen benchmark; other tasks need environments. | Validated assets/contact behavior, robot/gripper/sensor contract, reset distributions and independently observable task outcomes. |
| 6 | Qualification and sample efficiency — evaluation | Local benchmark runner and frozen sample-count evidence exist. | Learning curves over independent source groups, multiple collection orders, frozen candidates, nominal/stress/control strata, all attempted trials and measured cost. |
| 7 | Installable skill — runtime | Simulation candidate capsule and compatibility checks exist. | Bundle model, task, observation/action contract, adapter, data/protocol hashes, calibration requirements and artifact-bound qualification results. |
| 8 | Physical transfer — robotics | Unqualified. | Hardware timing/calibration checks, measured limits and a separate supervised physical evaluation. Simulation results alone cannot advance this stage. |

The coverage planner and recorder are available from the current source checkout.
They are not included in the frozen v0.1.0 release or the older Hugging Face pilot.
No new task policy is trained by adding these components.

## First incoming skills

Start with one bounded task and one supported robot/gripper rather than declaring
a universal skill. For shoe organization, distinguish pairing, alignment and
placing on a shelf: each needs a goal, observable result and task environment.
Record initial conditions, the complete attempt and its outcome. Five- or
six-second clips are useful when they contain that whole transition. Preserve
longer originals and shared recording/session/object IDs across every cut.

Collection starts with selected high-value conditions: different physical
instances, starting poses, views, clutter, failed grasps and recoveries. Initial
quotas are workload estimates. Use development errors to refine the plan;
version any changed brief and re-review earlier evidence against it. Freeze
confirmation groups before fitting or adjusting the policy.

Human footage can support visual representations and task labels. Joint commands,
forces and calibrated geometry require robot telemetry or a validated action
bridge. Synthetic variations preserve their parent lineage and are reported
separately from independently captured evidence. Shoe tying needs a deformable
task environment; it cannot inherit the rigid-box adapter's qualification.

## What “enough data” means

Report independent groups, conditions and outcomes, not just video count or
seconds. Run nested training subsets while keeping development evidence fixed.
Predeclare the target success, false-completion and critical-error limits with
uncertainty estimates and a cost budget. Freeze the selected candidate before
opening fresh confirmation trials. A plateau below the target calls for a
diagnosis of observations, actions, model or simulator, not a “done” badge.

Collection coverage, usable data, trained candidate, simulation qualification and
physical qualification remain separate states. A collection quota cannot promote
any of the later states.

## Every skill's benchmark contract

At creation, record a success predicate, meaningful failure cases and whether an
environment exists. A missing environment blocks qualification while collection
and environment work continue. The planner's description is the start of the
contract; an executable benchmark additionally needs:

- Task/environment, asset, physics, controller and sensor versions; observation
  and action units, coordinate frames, degrees of freedom and gripper limits.
- Reset distributions, seeds, episode timeout and nominal, stress, recoverable,
  impossible-task and sensor-fault strata where applicable.
- Authored/simple baselines and matched negative controls; a policy hash and
  frozen dataset/protocol hashes before evaluation.
- All attempted episodes, including failures, errors and incompatibilities;
  success, false completion, recovery and critical-error rates with denominators.
- Wall time, step throughput, end-to-end control latency and memory with named
  hardware and measurement boundaries. CPU-only support is measured per task.
- A supported operating envelope and explicit simulation-only or physical scope.

The next engineering milestone is a tape inspector and a small state-based task
outcome model on the existing placement environment. The first real Skillspace
will then test the collection and review loop. See [collection design](skillspace-coverage.md),
[episode evidence](episode-tape.md) and [existing benchmark protocol](benchmarks.md).

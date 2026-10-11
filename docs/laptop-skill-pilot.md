# A laptop-first Skillspace pilot

DVIDIA's next useful step is a small learned component that predicts the measured
effect of a robot command. A large language model does not need to run at every
control boundary. The current transition pilot learns from native simulator
episode tapes and runs locally on a CPU. It does not learn shoe organization
from footage or control a physical robot.

The proposed shoe pilot is separate: collect complete visual examples together
with synchronized robot state and action evidence, then evaluate a learned policy
within a declared robot, gripper and task envelope. A large number of short videos
alone cannot supply the missing robot action measurements.

## What the existing tools contribute

| Component | Implemented contribution | Remaining gap |
| --- | --- | --- |
| Skillspace intake and review | Local footage manifests, source hashes, lineage grouping, observation overlays and editable annotations | Independently reviewed shoe data and synchronized robot actions |
| Coverage planner | Capture missions, declared review/rights gates and unresolved conditions | Collected evidence; the planner does not inspect or approve videos |
| Visual learner | A CPU PCA/ridge next-frame predictor from frozen source groups | Object geometry, grasping and action-conditioned task behavior |
| Motion alpha | Seeded image tracking, image speed and short future-path extrapolation | Calibrated depth, contact dynamics and measured live-stream latency |
| Episode tapes | Before/after state, requested/issued commands and native measurements at control boundaries | Equivalent calibrated hardware sensing and action alignment |
| Native arm and capsules | Bounded rigid-box placement, simulation candidates and compatibility checks | A validated shoe task, learned shoe policy and physical qualification |
| Reaction and motor memory | Separate native anticipation, braking, retention and fitted actuator experiments | Whole-arm hardware measurements and integration into a shoe policy |

These pieces provide evidence and reusable interfaces. They are not already a
single end-to-end footage-to-shoe-policy system. See [episode tapes](episode-tape.md),
[motion observations](motion-observations.md) and the [execution map](roadmap.md).

## Run the compact transition pilot

Use the `feat/laptop-skill-pilot-v0.1` candidate source branch and
[installation instructions](installation.md). It is not included in the frozen
Observation Lab v0.1 release or hosted demo. Corpus generation requires the
optional `arm` extra; fitting uses NumPy and saved-model inference uses Python's
standard library. Python 3.12+ is required by the package.

```sh
.venv/bin/python -m pip install ".[arm]"
.venv/bin/dvidia-transition corpus \
  --output runs/transition-corpus --seed-start 311000
.venv/bin/dvidia-transition run runs/transition-corpus/corpus.json \
  --output runs/transition-pilot
.venv/bin/dvidia-transition inspect runs/transition-pilot
```

Use fresh output directories. Open `runs/transition-pilot/report.html` locally;
`report.json` retains the machine-readable results. On Windows PowerShell,
replace `.venv/bin/COMMAND` with `.\.venv\Scripts\COMMAND.exe` and put continued
commands on one line. The corpus command writes a directory containing
`corpus.json` and its episode tapes.

The default corpus has 12 seeded native scene groups, each recorded with nominal,
open-jaw and release-after-lift commands. All three interventions for one scene
remain in the same split. The 8/2/2 training/development/test grouping is an
algorithm demonstration, not 12 independently captured physical examples.
Related recording/session identities must also remain connected; changing an ID
does not create new evidence.

The learned ridge model has 20 inputs and seven outputs at the native 20 ms
control interval:

| Contract | Values |
| --- | --- |
| Current observation | Six joint positions, six joint velocities and measured jaw gap |
| Issued action | Six joint targets and the requested jaw gap |
| Prediction | Change in six joint positions and measured jaw gap at the next boundary |

All present recordings use privileged native simulator measurements. The input
contract resembles encoder/command signals a robot could expose; no physical
encoder or gripper has been validated here. Missing, stale or delayed measurements
must retain their status instead of becoming zero-valued readings. This
fixed-boundary model requires contemporaneous required observations; it does not
propagate aged sensor readings to the present.

Authored phase, intervention name, scene identity, terminal outcome, future object
state and the final applied servo controls are excluded from the prediction
inputs. The final applied controls are recorded after the transition and are not
known when the action is requested. Object/contact truth remains separate from
the deployment feature vector.

Training groups fit the normalization and model. Development groups choose the
regularization; the selected model and evaluation protocol are frozen before
test scoring. The report compares the action-conditioned candidate with
training-mean delta, persistence, constant-velocity and state-only predictions.
These comparisons ask whether commands add predictive value beyond current
state. Low next-step error does not establish grasp retention, terminal task
success or a shoe policy.

Report model inference separately from loading and feature extraction, with
sample counts, median, p95 and worst times. Camera capture, image processing,
communication, physical command application and stopping behavior are outside
this timing boundary. A fast numerical prediction is one component of a future
closed control loop, not evidence that the full robot reacts at that speed.
Cold saved-model loading is not measured; corpus-load timings cover episode
validation rather than deployment startup.

The [frozen benchmark](../benchmarks/actuator-pilot-20261011/README.md) retains
the trained models and exact run receipt. On the measured laptop, the 6,438-byte
action model took 0.207 ms at warm p95. Commands improved aggregate joint and
gripper RMSE, but aggregate gripper mean and worst errors slightly regressed
against persistence. The first matched opening transition was substantially
overpredicted. Those results motivate a history-aware gripper model and fresh
confirmation groups; they do not establish a usable shoe controller.

## Hardware and offline operation

| Operation | Current dependency requirement |
| --- | --- |
| Generate native episode tapes | CPU, Python, NumPy and MuJoCo through the `arm` extra |
| Fit or evaluate the transition model | CPU, Python and NumPy; no GPU or model API |
| Read the saved HTML report | A local browser; no hosted service |
| Run footage observation/motion tools | Their separate OpenCV/FFmpeg and model requirements |
| Operate a physical shoe robot | A compatible robot/gripper, calibrated sensing and a measured adapter; not implemented by this pilot |

After dependencies are installed, the transition experiment needs no network,
account, language model, renderer, video conversion or downloaded model weights.
This does not make the observation pipeline's detector or camera processing
part of its inference measurements. Minimum RAM, supported laptop throughput
and end-to-end control deadlines must come from measured runs on specified
machines; these dependency requirements are not certified minimum hardware.

A future imitation policy may need more expensive training. ACT learns action
chunks from robot demonstrations and is a candidate for that stage, rather than
the transition predictor implemented here. A smaller student policy may then be
evaluated for local execution. GPU training and CPU execution are separate
questions. LeRobot's deployment documentation explicitly makes CPU use
conditional on checkpoint support and measured control rate. Neither source
establishes the speed or shoe competence of a DVIDIA policy.
[ACT research](https://tonyzhaozh.github.io/aloha/),
[LeRobot policy inference](https://huggingface.co/docs/lerobot/main/inference).

## Prepare the first shoe Skillspace

The [shoe capture plan](../examples/shoe-organizing-coverage.json) starts with
one shoe moved from a marked pickup area into one marked slot. It contains no
fabricated evidence. Its benchmark is blocked because a validated shoe task
environment and compatible hardware adapter have not been supplied.

The proposed final criteria are toe direction within 15 degrees of the slot,
center within 20 mm of its reference and observed stability for one second.
These are study proposals, not validated pass thresholds. Define the shoe
reference point, toe direction, slot coordinates, uncertainty and observation
procedure using calibrated measurements before freezing a benchmark.

Start with one fixed table and calibrated camera, verified reachable regions and
an observed grasp region compatible with the arm's payload and gripper. Use
tucked or absent loose laces, one shoe and no clutter. Pair matching, arbitrary
footwear and lace manipulation are later tasks with different requirements.

An initial workload suggestion is 120 complete attempts across 12 disjoint
physical shoe-pair/source groups, with eight groups for training, two for
development and two reserved for confirmation. Ten attempts per group is a
planning starting point, not a fixed minimum or known sufficient sample count.
Choose separate collection partitions before recording. Preserve true recording,
session and pair lineage; shared sessions can connect otherwise different pairs
into one group. Confirmation evidence never fills the training collection quota.

The per-condition group targets in the JSON are revisable workload estimates.
A group can contribute different complete attempts to multiple conditions;
summing cell quotas does not give a count of independent examples. Include
changed starts, slot offsets, different pairs, genuine missed grasps and observed
supervised recoveries. Retain operator assistance and all unsuccessful attempts.
Do not induce an unsafe drop to populate a failure cell.

Five- or six-second clips are useful only when they preserve the evidence needed
for their labels. A complete attempt must show acquisition, transfer, release
and the final outcome, even when that takes longer. Retain originals and stable
source IDs before cutting or augmenting recordings.

Inspect the proposed plan with the existing coverage validator:

```sh
.venv/bin/dvidia-skillspace-coverage inspect \
  examples/shoe-organizing-coverage.json
```

`inspect` validates the plan and computes declared gaps. It does not collect
footage, perform human review or run a shoe benchmark. See
[coverage rules](skillspace-coverage.md) for complete attempts, review, rights
and connected-source grouping.

## From collection to an installable skill

The intended path is:

1. A Skillspace defines the task, operating envelope and capture missions.
2. Complete footage receives review, rights checks and source grouping.
3. Robot teleoperation pairs observations with synchronized measured state and
   issued actions. Human hand motion alone is not a robot action label.
4. A trained policy consumes only explicitly available deployment inputs;
   development evidence selects its revision and diagnoses weak conditions.
5. A validated task simulator evaluates fresh held-out scenarios and matched
   controls, retaining every attempted episode and false completion.
6. A supported robot/gripper adapter receives an artifact with its feature,
   calibration and compatibility contracts. Physical trials establish the
   measured deployment envelope before qualification.

The transition pilot adds a small learned action-to-motion component to this
path. The existing capsule format supports one simulation arm/rigid-box adapter;
installing that capsule cannot install shoe competence on an arbitrary arm.
The next shoe milestones are real synchronized capture, observable task labels,
a learned closed-loop policy and a validated shoe benchmark. Data volume and
model speed become useful evidence when those measurements improve held-out
task completion without concealing failures.

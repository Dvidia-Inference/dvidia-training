![DVIDIA Training — toward installable robot skills. Open source, offline, CPU.](assets/branding/github-header.png)

# DVIDIA Training

[Install locally](docs/installation.md) · [Hardware guide](docs/hardware.md) · [Execution map](docs/roadmap.md) · [Robot simulations](docs/robot-simulation-library.md) · [Upcoming arms](docs/hardware-roadmap.md) · [Official site](https://dvidia.org/) · [Research](https://research.dvidia.org/papers/skillspace-footage-training/) · [Frozen benchmark evidence](benchmarks/sample-count-19047/README.md)

[Recorded demo](https://huggingface.co/spaces/Dvidia/dvidia-training) · [Synthetic datasets](https://huggingface.co/datasets/Dvidia/dvidia-training-examples) · [Pilot models](https://huggingface.co/Dvidia/dvidia-training-pilot)

**Research alpha.** The recorded demo covers the original one-scene simulation pilot; the frozen benchmark is a separate experiment. Training runs locally. No human-video action bridge or hardware skill is established.

The new [native grasp task benchmark](docs/native-task-benchmark.md) connects the
pinned six-axis arm and coupled gripper to free-object grasp, carry and release
attempts. It retains matched open-jaw/hold controls, frozen protocols and local
3D replay. This candidate source uses an authored controller; it is a test
environment for the next learned skill, separate from the older hosted pilot.

The [robot simulation library](docs/robot-simulation-library.md) adds pinned
PiPER (six arm axes) and SO-101 (five arm axes) profiles with separate grippers,
explicit CPU asset provisioning, native audits and unloaded motion benchmarks.
PiPER also has a [native contact pilot](docs/piper-contact-pilot.md) with an
authored pickup-and-place controller and matched controls. SO-101 remains at
unloaded mechanics; learned policies and physical qualification remain open. The
[Robot Simulation Lab](https://huggingface.co/spaces/Dvidia/robot-simulation-lab)
publishes recorded evidence and local run instructions.

The [shoe asset library v0.1](docs/asset-library.md) adds six original rigid shoe
proxies, versioned scene descriptors and a `dvidia-assets` loader for UR5e/Robotiq,
PiPER and SO-101. Local 3D previews expose the native geometry; CPU benchmarks
check gravity and support contact with matched controls. Properties are labelled
as synthetic assumptions. Shoe grasping, deformation and hardware transfer still
need their own evidence. The [broader asset roadmap](docs/robot-simulation-library.md#reusable-asset-library-planned-product-layer)
extends this foundation toward measured objects, fixtures, grippers and sensors.

The [PiPER grip-recovery candidate](docs/piper-grip-recovery.md) adds a short
test lift before carrying, one supported recovery attempt, and a paired
benchmark against the original controller. It uses current simulator state;
task completion, grip qualification, retries and aborts are recorded separately.

The [contact-history v0.3 candidate](docs/piper-contact-memory-v03.md) compares
the original controller, test lift alone, and bounded slowdown/closed-hold
reassessment. Its complete paired benchmark varies object mass and friction;
resuming a movement and recovering a completed placement are separate outcomes.
The [84-attempt confirmation](benchmarks/piper-contact-v03-20261011/README.md) completes
6/12 history placements versus 4/12 for each paired alternative, with all 48
negative controls failing. These are synthetic, privileged-state results.

The [contact-recovery v0.4 study](docs/piper-contact-recovery-v04.md) separates
a faster carry response from avoiding the same ineffective supported grasp.
The [fresh 84-attempt confirmation](benchmarks/piper-recovery-v04-20261011/README.md)
completes 4/12 placements in every mode, with 0/48 negative-control successes.
Two resumed placements finish 2.44–2.64 simulated seconds sooner, and four failed
grasps are refused 2.88–2.90 seconds sooner. No completion is gained or lost.
Eight failures per mode remain visible, including two held blocks that never
satisfy the unchanged resume gate.

The [reliability v0.5 study](docs/piper-reliability-v05.md) adds a bounded second
lift and a fixed alternate-grasp planner. Three frozen fresh cohorts complete
**15/36 placements with v0.4, 18/36 with second-lift recovery, and 23/36 with the
revised grasp**, with no lost baseline successes and all 144 negative controls
failing. All 252 attempts are retained in the [complete evidence](benchmarks/piper-reliability-v05-20261012/README.md).
The revised grasp takes about 0.77 simulated seconds longer on shared successes;
thirteen failures remain. Loaded-host computation delays are reported, with no
physical real-time claim. This remains authored simulation, not a learned skill.

Train a small visual model from Skillspace footage on your own computer or server. The pipeline checks actual videos and source groups, fixes training/development/test splits, fits weights on a CPU, and exports a result with held-out measurements and integrity hashes.

The base install needs **Python 3.12+, NumPy and FFmpeg**. It needs no GPU, CUDA, account or API key. Training is offline by default; downloading software or public footage is a separate, explicit step.

**Visual learning is not an acquired robot skill.** Human videos do not directly supply robot joint commands or contact forces. An optional movement branch requires compatible, validated action supervision; installation and closed-loop qualification remain separate gates.

## Install and open the studio

Install Python 3.12+ and FFmpeg, including `ffprobe`, first. On macOS, `brew install python@3.12 ffmpeg`; on Debian/Ubuntu with Python 3.12+ available, install `python3-venv` and `ffmpeg`. See [platform instructions](docs/installation.md) for Windows PowerShell, disconnected servers and Docker.

macOS / Linux:

```sh
git clone https://github.com/Dvidia-Inference/dvidia-training.git
cd dvidia-training
python3.12 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/dvidia-training-studio --directory runs/studio
```

Open [http://127.0.0.1:8270](http://127.0.0.1:8270). Select a local Skillspace folder or upload a ZIP, check its footage and groups, train, then inspect and download the result. Larger folders can be supplied by local path. The studio binds only to the local machine; use SSH port forwarding when it runs on a remote server.

For Windows, use `.venv\Scripts\python.exe` and `.venv\Scripts\dvidia-training-studio.exe`; [complete PowerShell commands](docs/installation.md#windows-powershell) are provided.

## Plan the next useful recording

Open [the collection planner](http://127.0.0.1:8270/coverage) in the same local studio. Define the skill's goal, operating envelope, selected conditions and simulation benchmark. The planner suggests a capture brief for a gap and tallies declared independent reviewed groups. Its example records are fabricated; it does not inspect footage or establish training readiness. Export the plan to keep it.

[Collection design and proposed incentives](docs/skillspace-coverage.md) explain complete attempts, review and rights declarations, source grouping, failures and recoveries. Every skill starts with a benchmark description; an unsupported task environment keeps qualification blocked. Collection targets and measured competence are separate.

The [episode tape recorder](docs/episode-tape.md) preserves the existing simulated arm's measured state, requested/applied commands, contacts and sensor ages. Recording needs the optional arm extra; validation and inspection need only Python's standard library. These tools are available from the current checkout and do not train a new policy or change the older Hugging Face pilot.

## Review what is visible in a video

The new [local observation lab](docs/observations.md) uses optional CPU object and
palm detectors to propose timestamped image observations. Open the original video,
inspect its boxes and timeline, correct labels, and save a separate review revision.
Raw evidence stays unchanged. No account, cloud inference, or GPU is needed.

[Observation Lab release](https://github.com/Dvidia-Inference/dvidia-training/releases/tag/observation-lab-v0.1.0) · [Research paper](https://research.dvidia.org/papers/dvidia-observation-lab/) · [Synthetic review demo](https://huggingface.co/spaces/Dvidia/observation-lab)

Use the `observation-lab-v0.1.0` tag for this feature. The hosted demo uses
authored synthetic clips and keeps edits in your browser; actual detection runs
locally with the optional observation runtime and explicitly provisioned models.

The [first feasibility run](benchmarks/observation-20261010/README.md) completed
six public excerpts. This establishes the integration, not accuracy: labels can
be wrong, outcomes remain unknown, and there is no 3D reconstruction, automatic
action understanding, or human-video-to-robot control bridge.

## Estimate and forecast image motion

The [first image-motion alpha](docs/motion-observations.md) is available in the
candidate source checkout. Manually seed one first-frame region; causal CPU
feature tracking estimates its image-anchor speed and forecasts a path at
100, 250 and 500 ms. Lost tracks stay lost. Actual integer video timestamps,
tracking support, units, source/implementation hashes and failed attempts are
preserved. This feature trains no model and requires no detector weights.

```sh
.venv/bin/python -m pip install '.[observations]'
.venv/bin/dvidia-motion benchmark --output runs/image-motion-001
.venv/bin/dvidia-motion inspect runs/image-motion-001
.venv/bin/dvidia-motion studio runs/image-motion-001 --port 8286
```

Use the [candidate source installation instructions](docs/motion-observations.md#install-from-source),
not the frozen `observation-lab-v0.1.0` tag, for `dvidia-motion`. The authored
benchmark and read-only local replay test image tracking and prediction, not
real detector accuracy. Speeds are analysis-image pixels/second; physical speed,
moving-camera compensation, live robot latency and interaction prediction remain
unqualified. This is one component toward [the full Observation Lab v0.2](docs/observation-v0.2-plan.md).
See the [measured authored benchmark](benchmarks/motion-alpha-20261010/README.md),
including conditional errors, failed tracking and forecast tails.

## Learn a compact command-to-movement model

The [laptop skill pilot](docs/laptop-skill-pilot.md) connects episode tapes to a
small action-conditioned predictor. It learns the next 20 ms of joint and jaw
movement from current state and issued commands, using CPU ridge regression.
Separate simulation groups test it against no movement, current velocity,
training-mean movement and a matched model with commands removed. A local,
offline report exposes prediction errors, artifact size and model-only timing.

```sh
.venv/bin/python -m pip install '.[arm]'
.venv/bin/dvidia-transition corpus --output runs/actuator-corpus
.venv/bin/dvidia-transition run runs/actuator-corpus/corpus.json --output runs/actuator-pilot
.venv/bin/dvidia-transition inspect runs/actuator-pilot
```

Use the `feat/laptop-skill-pilot-v0.1` candidate source branch for these commands,
then open `runs/actuator-pilot/report.html`. Generation uses the existing native
rigid-box simulator; fitting and inference need no language model, GPU or remote
service. Simulator measurements are privileged, even though the selected inputs
resemble signals a robot could expose. The predictor issues no robot commands
and establishes neither grasp success nor shoe organization.

The [retained CPU benchmark](benchmarks/actuator-pilot-20261011/README.md) includes
the trained 6.3 KiB predictor, frozen protocol and complete attempt receipts.
Its warm model computation measured 0.207 ms at p95 on one laptop; sensing and
actuation are excluded. Command inputs reduce test RMSE, while some stationary
gripper and worst-error results regress. Read all error measures before reusing it.

The [first shoe collection plan](examples/shoe-organizing-coverage.json) defines
one shoe, one marked slot and complete attempts. Its benchmark remains blocked
until a validated shoe environment and compatible physical adapter exist.
Recording quotas are proposed study sizes, not known sufficiency thresholds.

## Try the pipeline

The [hardware reference bench](docs/hardware-reference.md) assembles a pinned
UR5e arm and coupled Robotiq gripper, audits actual joint/actuator/collision
structure, and compares CPU physics at three timesteps. Optional left/right
five-finger references have separate contracts. A paired replay scorer measures
joint, tool and jaw errors once aligned physical recordings are available.
Export actual mesh geometry and recorded native poses to an offline interactive
3D replay with orbit controls, six-joint readouts and collision-proxy inspection.
The [hardware list](docs/hardware-roadmap.md) tracks implemented PiPER/SO-101
profiles and upcoming YAM, OpenArm, ARX X5 and PAROL6 integrations.
These tools use the `feat/reference-task-benchmark` candidate branch; the new
reference has no learned task policy or measured physical fidelity.

Generate an explicitly synthetic example, then train and verify its output:

```sh
.venv/bin/dvidia-train example --clips 10 --output examples/visual-demo
.venv/bin/dvidia-train run examples/visual-demo --output runs/visual-demo
.venv/bin/dvidia-train inspect runs/visual-demo
```

The example consists of moving-disc videos. It checks the software path and does not demonstrate human-video learning or shoe organization. Each run needs a fresh output directory.

For your own footage:

```sh
.venv/bin/dvidia-train prepare /path/to/skillspace --output runs/checked
.venv/bin/dvidia-train run runs/checked/dataset/dataset.json --output runs/trained
```

A folder of videos is accepted, with lineage/completeness warnings. Prefer an explicit [`skillspace.training.json`](docs/skillspace-format.md). A public catalog page containing only a demonstration count cannot supply training footage.

## Hardware

These are **provisional planning targets for the small default visual learner**, not verified minimum specifications or guarantees for arbitrary datasets.

| Resource | Proposed starting point | Recommended |
| --- | --- | --- |
| CPU | 1 modern CPU core | 2–4 cores |
| RAM | 2 GiB | 4–8 GiB |
| GPU / VRAM | None required | None required for this learner |
| Free scratch disk | 1 GiB for a small example, plus original footage and installed tools | At least 2 GiB, scaled to your dataset |

Allow space for roughly two additional source copies when checking and then training, plus intermediates and exports. Intake is bounded to 500 MiB per source and 128 MiB per video. At the intake ceiling, the extra copies alone can approach 1 GiB; the starter scratch allowance does not cover those copies plus all intermediates. The studio ZIP upload limit is 24 MiB. [Hardware details and checks](docs/hardware.md) explain the estimates.

Development was tested on a macOS Apple M5 machine with 24 GiB RAM. That machine is not the minimum. [All eight checks passed on runtime commit `d1f0cbc`](https://github.com/Dvidia-Inference/dvidia-training/actions/runs/37740260434): visual training and studio tests on Linux/macOS/Windows with Python 3.12 and 3.13, the optional Linux arm pilot, and actual Docker training with networking disabled. Hardware requirements for larger video models or shoe motor policies remain unmeasured. Run `dvidia-training-doctor` to check installed dependencies and local capacity without training.

## What the result means

The default learner fits a training-only PCA representation and a ridge next-frame predictor. Development footage selects regularization; test footage is scored afterward against last-frame and training-mean baselines. Pixel/latent prediction error is a visual diagnostic, not task success, calibrated 3D understanding or robot competence.

`training-result.zip` contains weights, model contracts, measurements, a trimmed dataset receipt and an integrity manifest. Original footage stays in the local prepared dataset and is excluded from the download. Hashes check consistency; they do not authenticate capture history or measured truth.

Related recording/session/shoe-pair IDs and identical video hashes stay in the same connected source group. Group declarations cannot detect every near duplicate. Five-second cuts from one recording are not independent examples, and no video count is a proven sufficiency threshold.

## Optional arm simulation

```sh
.venv/bin/python -m pip install ".[arm]"
```

This adds MuJoCo 3.15.0 for the existing bounded simulation placement adapter, native telemetry examples and `dvidia-skill-capsule`. A movement candidate still needs an explicit compatible task contract and independent closed-loop qualification. It is not a physical robot driver or a learned shoe policy. See [the movement data contract](docs/skillspace-format.md#movement-supervision).

To measure a candidate before improving it, use the [sample-count and simulation benchmark](docs/benchmarks.md). It compares nested sets of whole recordings, keeps development/test footage fixed, freezes candidate weights before evaluation, and reports new-layout outcomes alongside matched open-jaw controls. It runs locally on a CPU and retains failures and qualification receipts. Install from the current source checkout to get `dvidia-training-benchmark`; the frozen v0.1.0 release predates this tool.

The [reaction framework v0.1](docs/reaction-framework.md) adds a reusable monitor and native linear-actuator/parallel-jaw fixtures for predicted clearance, contact, slip and sensor faults. Its local benchmark reports detection errors with uncertainty, continued-physics braking time/distance, contact and load outcomes, and CPU cost against a matched disabled-intervention baseline. It uses synthetic sensed state; camera perception, whole-arm protection and physical stopping accuracy need further adapters and calibration. It preserves every attempted trial and the existing frozen placement evidence.

[The first measured run](benchmarks/reaction-v0.1-20261009/README.md) preserves 78 primary trials and eight timestep checks, including late-impact diagnostic failures and payload drops. Run `dvidia-reaction benchmark --output runs/reaction --repeats 3 --convergence`, then `dvidia-reaction inspect runs/reaction` to generate and verify your own offline report.

The [motor-memory framework v0.2](docs/motor-memory.md) fits a compact actuator model on separate native calibration episodes, then compares slow feedback, a fixed fast approach and predictive adaptive control on “approach, settle and retain.” It measures completion time against native error at the completion claim, overshoot, false/revoked completion, retained load and CPU cost. Commanded goals are exact; observations of the moving carriage can be delayed, noisy or missing. The movement profile and slowdown rules are authored. This is a CPU simulation experiment with learned dynamics, and does not train a task policy from human video or drive a physical robot.

```sh
.venv/bin/dvidia-motor-memory train --output runs/motor-memory
.venv/bin/dvidia-motor-memory benchmark --memory runs/motor-memory --output runs/motor-benchmark --repeats 3
.venv/bin/dvidia-motor-memory inspect runs/motor-benchmark
```

Each command uses a fresh output directory where applicable. The model and complete attempted matrix are frozen before evaluation; development and confirmation partitions remain separate. [The measured v0.2 run](benchmarks/motor-memory-v0.2-20261009/README.md) preserves the candidate, compact attempt receipts and limitations. Complete calibration transitions and scoring traces are retained in local runs and CI artifacts.

## Public pilot on Hugging Face

[Recorded demo](https://huggingface.co/spaces/Dvidia/dvidia-training) · [Synthetic datasets](https://huggingface.co/datasets/Dvidia/dvidia-training-examples) · [Pilot models and measurements](https://huggingface.co/Dvidia/dvidia-training-pilot)

The demo presents saved candidate and open-jaw control playback from one exact simulation scene. Training runs locally; the dataset and model cards document reproducible inputs, CPU measurements and the limits of this research alpha. [Maintainer publishing tools](publishing/huggingface/README.md) stage only the already-public synthetic release and use a credential from macOS Keychain without printing or caching it.

Original DVIDIA code is MIT licensed; observation detector adaptations retain
Apache 2.0 notices and upstream models retain their own licenses. Input footage
has separate rights. Research and measured limitations: [research.dvidia.org](https://research.dvidia.org).
Contact: [hello@dvidia.org](mailto:hello@dvidia.org).

![DVIDIA Training — toward installable robot skills. Open source, offline, CPU.](assets/branding/github-header.png)

# DVIDIA Training

[Install locally](docs/installation.md) · [Hardware guide](docs/hardware.md) · [Execution map](docs/roadmap.md) · [Official site](https://dvidia.org/) · [Research](https://research.dvidia.org/papers/skillspace-footage-training/) · [Frozen benchmark evidence](benchmarks/sample-count-19047/README.md)

[Recorded demo](https://huggingface.co/spaces/Dvidia/dvidia-training) · [Synthetic datasets](https://huggingface.co/datasets/Dvidia/dvidia-training-examples) · [Pilot models](https://huggingface.co/Dvidia/dvidia-training-pilot)

**Research alpha.** The recorded demo covers the original one-scene simulation pilot; the frozen benchmark is a separate experiment. Training runs locally. No human-video action bridge or hardware skill is established.

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

## Try the pipeline

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

MIT licensed. Research and measured limitations: [research.dvidia.org](https://research.dvidia.org). Contact: [hello@dvidia.org](mailto:hello@dvidia.org).

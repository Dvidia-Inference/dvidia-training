# Hardware and dependencies

Visual training runs on the CPU. A GPU is not required by this implementation, and the local studio runs one active job. We have not established a measured minimum CPU or RAM configuration. More GPUs, multiple machines, and additional simultaneous workers are not qualified execution modes.

## Functional requirements

| Component | This project's contract | How it is checked |
| --- | --- | --- |
| Python | 3.12 or later | Running interpreter version |
| NumPy | `2.5.3` supported package baseline | Installed package metadata |
| FFmpeg and ffprobe | Executables on `PATH`; ordinary MOV/MP4, Matroska/WebM and AVI demuxers | Version and enabled-demuxer listings |
| Video codecs | A supplied clip must decode within the bounded training workflow | Actual per-clip probing and decoding |
| MuJoCo | `3.15.0`, optional for visual training; required for the arm simulation tools | Package metadata when `--arm` is selected |
| libx264 | Optional; required to create this project's synthetic MP4 examples | FFmpeg encoder listing |

The doctor does not import NumPy or MuJoCo. A metadata check can pass even when a native package cannot load, so run a small example to verify the environment. NumPy's official installation guide describes environment-based installation; the pinned version here is the project's supported baseline. [NumPy installation](https://numpy.org/install/).

FFmpeg builds differ in enabled codecs and containers. The enabled-demuxer and encoder checks use the documented `-demuxers` and `-encoders` queries; existing-footage training decodes frames and does not need libx264. [FFmpeg options](https://ffmpeg.org/ffmpeg.html#Main-options), [ffprobe documentation](https://ffmpeg.org/ffprobe.html).

MuJoCo is a separate optional native dependency. Its upstream documentation describes Python installation and supported platform builds; this does not establish that every upstream platform has been tested by this project. [MuJoCo Python installation](https://mujoco.readthedocs.io/en/stable/python.html).

## Provisional small-example sizing

These estimates apply to a small collection of short clips at the default 12 frames per clip, 16 × 16 RGB resolution and eight latent dimensions. They are engineering starting points, not benchmarked minimums or success guarantees.

| Resource | Starter estimate | More comfortable small-example setup |
| --- | --- | --- |
| Logical CPU cores | 1 | 2–4 |
| RAM | 2 GiB | 4–8 GiB |
| Free scratch space | 1 GiB | At least 2 GiB |
| GPU | None required | None required |

Retain additional space for your original recordings and Python/FFmpeg installation. Intake copies footage into the prepared dataset. The current intake bounds each video to 128 MiB and total footage plus sidecars to 500 MiB. Training additionally bounds decoded RGB arrays to eight million elements, aligned sidecar JSON to 64 MiB and movement samples to 100,000. These are rejection limits, not predictions of peak RAM: numerical fitting and decoding create intermediate arrays and processes. Larger datasets or settings can need substantially more memory and time.

The doctor reports total physical RAM, logical CPU count and free space at the requested directory or its nearest existing parent. Physical RAM is not the memory available to the job; other applications and container resource limits can reduce it. Unknown or small capacities produce notes rather than a fabricated hardware certification.

## Check before training

```sh
dvidia-training-doctor
dvidia-training-doctor --json --directory ./runs
dvidia-training-doctor --arm
```

The command reads package metadata and system listings. It does not train, create output directories, access the network or inspect your footage. Missing required dependencies return a nonzero exit code. Missing optional simulation support only fails the check when `--arm` is requested; a missing fixture encoder does not block training from existing clips.

## Tested scope

Development and local integration checks have used a macOS Apple M5 system with ten logical CPU cores and 24 GiB RAM, Python 3.12.9, NumPy 2.5.3 and FFmpeg 9.0.1. This is a tested development configuration, not the minimum required machine.

Three repeated standalone runs on ten synthetic moving-disc clips, totaling 60 seconds of footage, used seven training, two development and one test episode at default visual settings. Complete pipeline times were 0.5315, 0.5106 and 0.5102 seconds, with a median of **0.5106 seconds**. The Python process lifetime peak RSS was **50,462,720 bytes (48.13 MiB)**. This memory figure excludes FFmpeg/ffprobe child processes; installation, imports and fixture creation are excluded from pipeline time. These results describe this small synthetic visual task, rather than a minimum machine or an arm-training benchmark. [Benchmark receipt](../benchmarks/macos-m5.json).

Reproduce the benchmark after installation with `python benchmarks/run.py --output runs/benchmark --repeats 3`. Original source clips and every run's learned weights and metrics remain local; the benchmark reports the stated task and measurement scope.

[All eight CI jobs passed on runtime commit `d1f0cbc`](https://github.com/Dvidia-Inference/dvidia-training/actions/runs/37740260434). Visual footage fitting, export checks, the loopback studio tests and installed entry points passed on Linux, macOS and Windows with Python 3.12 and 3.13. The Linux arm job also collected ten synthetic native recordings, fitted a candidate, installed it and qualified one exact scene. A separate Linux Docker job built the image and ran synthetic example creation, training and inspection with `--network none`.

These jobs establish the stated software paths on hosted runners, not minimum hardware or offline installation on a new air-gapped machine. Other Python versions, architectures and codecs need local checks. Native arm simulation remains unqualified on Windows/macOS by CI. Downloaded skills remain simulation candidates until qualified for their exact local scene; no physical robot readiness follows from installing dependencies.

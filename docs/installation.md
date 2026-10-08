# Installation

The visual base uses Python 3.12+, pinned NumPy and the external `ffmpeg`/`ffprobe` executables. The optional `arm` extra adds MuJoCo 3.15.0. No CUDA runtime or GPU is required for the current visual learner.

The commands use the virtual environment's executable directly, so activation and PowerShell execution-policy changes are unnecessary. [Python's venv documentation](https://docs.python.org/3/library/venv.html) describes the platform-specific executable paths.

## macOS

With Homebrew installed:

```sh
brew install python@3.12 ffmpeg git
git clone https://github.com/Dvidia-Inference/dvidia-training.git
cd dvidia-training
python3.12 -m venv .venv
.venv/bin/python -m pip install .
ffmpeg -version
ffprobe -version
.venv/bin/dvidia-training-studio --directory runs/studio
```

Open `http://127.0.0.1:8270`. Python 3.13 can also be selected when installed; the configured CI matrix covers 3.12 and 3.13.

## Linux

On a Debian/Ubuntu distribution that provides Python 3.12+:

```sh
sudo apt-get update
sudo apt-get install -y python3 python3-venv ffmpeg git
python3 --version
git clone https://github.com/Dvidia-Inference/dvidia-training.git
cd dvidia-training
python3 -m venv .venv
.venv/bin/python -m pip install .
ffmpeg -version
ffprobe -version
.venv/bin/dvidia-training-studio --directory runs/studio
```

If `python3 --version` is older than 3.12, install a supported Python before creating the environment. Use your distribution's equivalent package manager on other Linux distributions.

## Windows PowerShell

Install Python 3.12+ from [python.org](https://www.python.org/downloads/windows/) with the Python launcher, plus Git. FFmpeg is a separate dependency: when Chocolatey is already installed, run the following in an Administrator PowerShell window:

```powershell
choco install ffmpeg -y
```

Alternatively, choose a Windows build linked from [FFmpeg's download page](https://ffmpeg.org/download.html) and put the folder containing both `ffmpeg.exe` and `ffprobe.exe` on `PATH`. Chocolatey's [FFmpeg package page](https://community.chocolatey.org/packages/ffmpeg) documents its package installation. Open a new PowerShell window after updating `PATH`.

Run these commands in a regular PowerShell window:

```powershell
git clone https://github.com/Dvidia-Inference/dvidia-training.git
Set-Location dvidia-training
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install .
ffmpeg -version
ffprobe -version
.\.venv\Scripts\dvidia-training-studio.exe --directory runs\studio
```

Open `http://127.0.0.1:8270`. If you installed Python 3.13 instead, use `py -3.13` when creating the environment.

```powershell
.\.venv\Scripts\dvidia-train.exe example --clips 10 --output examples\visual-demo
.\.venv\Scripts\dvidia-train.exe run examples\visual-demo --output runs\visual-demo
.\.venv\Scripts\dvidia-train.exe inspect runs\visual-demo
```

## Remote server

Start the studio on the server using its platform's commands. Keep its `127.0.0.1` binding, then forward the port from your own computer:

```sh
ssh -L 8270:127.0.0.1:8270 user@your-server
```

Visit `http://127.0.0.1:8270` locally. Paths entered in the studio refer to the server's filesystem. The interface can read local files and is intended for a trusted local user; this repository does not provide public hosting or multi-user authentication.

## Offline installation

Obtain Python, FFmpeg and a compatible wheelhouse on a connected machine **matching the target OS, CPU architecture and Python version**. The commands below build the package wheel and download its Python dependencies; they do not package Python itself or FFmpeg.

From a clone of this repository on that connected machine:

```sh
python -m pip wheel --wheel-dir wheelhouse .
```

For the optional arm dependencies, use `python -m pip wheel --wheel-dir wheelhouse ".[arm]"` instead. Also obtain offline Python/FFmpeg installers or binaries for the target. Transfer the wheelhouse, source or release wheel, tools and your data to the disconnected server.

Create a virtual environment on the target, then install entirely from the local wheelhouse:

macOS / Linux:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --no-index --find-links wheelhouse dvidia-training
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-index --find-links .\wheelhouse dvidia-training
```

For an arm wheelhouse, replace `dvidia-training` with `"dvidia-training[arm]"`. Keep a record of the exact wheel versions and hashes you transfer. `--no-index` prevents consulting package indexes; the local wheelhouse must contain every dependency. See [pip's local-package instructions](https://pip.pypa.io/en/stable/user_guide/#installing-from-local-packages).

After Python, FFmpeg, dependencies and data are present, run `dvidia-train run` or the studio without `--allow-online`. No login or hosted inference is needed. Offline installation on a new air-gapped machine is a deployment procedure, separate from any measured offline training receipt.

## Optional arm features

From the checkout:

```sh
.venv/bin/python -m pip install ".[arm]"
```

On Windows, use `.\.venv\Scripts\python.exe`. The extra enables native simulation and simulation capsule tooling; it does not install physical robot drivers. Read [movement supervision](skillspace-format.md#movement-supervision) before fitting or exporting a movement candidate.

## Docker command-line use

The provided image uses `python:3.12-slim`, FFmpeg and the visual base package. Building the image normally needs network access to retrieve the base image, OS packages and Python dependencies. After building, a local-data training invocation can run with network access disabled.

From the repository on Linux/macOS:

```sh
docker build -t dvidia-training:local .
mkdir -p docker-output
docker run --rm --network none --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$(pwd)/my-skillspace,target=/data,readonly" \
  --mount "type=bind,source=$(pwd)/docker-output,target=/output" \
  dvidia-training:local run /data --output /output/run-001
```

`my-skillspace` must already exist. The output directory must be writable by the container user; use a fresh subdirectory for every run. A Windows Docker Desktop invocation can use resolved absolute paths:

```powershell
docker build -t dvidia-training:local .
New-Item -ItemType Directory -Force docker-output
$trainingInput = (Resolve-Path .\my-skillspace).Path
$trainingOutput = (Resolve-Path .\docker-output).Path
docker run --rm --network none --mount "type=bind,source=$trainingInput,target=/data,readonly" --mount "type=bind,source=$trainingOutput,target=/output" dvidia-training:local run /data --output /output/run-001
```

The Dockerfile is for the command line. The studio's loopback binding is inside the container, so publishing a host port does not expose it; these instructions do not change that binding. Run the studio natively or through the SSH forwarding path above.

## Resources and verification

Plan initially for 1 CPU core, 2 GiB RAM and 1 GiB free scratch disk for a small example, **plus** original recordings and installed tools. These are proposed starting targets; use 2–4 cores and 4–8 GiB RAM for headroom. Budget roughly two extra source copies for a checked-then-trained workflow plus intermediates and exports. Sources may reach 500 MiB, so the extra copies alone can approach 1 GiB and exceed the starter allowance once intermediates are included. Dataset size, codec, resolution and parallel numerical libraries change the actual cost. See [hardware details](hardware.md).

The development host is an Apple M5 macOS machine with 24 GiB RAM, not a minimum-hardware test. The CI workflow defines six visual platform/Python combinations and a separate Linux arm job. Consult actual workflow results before claiming a platform passed. Docker build and disconnected installation similarly require their own execution evidence.

Verify your local installation:

```sh
python -m dvidia_training.footage_pipeline --help
python -m pip check
dvidia-training-doctor --json --directory runs
ffmpeg -version
ffprobe -version
```

Use the virtual environment's Python executable in these commands. `dvidia-train inspect runs/your-run` verifies the export after training. A visual error score is not a robotics success score.

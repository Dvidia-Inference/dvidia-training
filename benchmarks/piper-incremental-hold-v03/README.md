# DVIDIA · PiPER incremental hold v0.3

One bounded pose-error accumulator is compared with the unchanged measured-joint,
issued-joint and Cartesian controls across eight fresh synthetic scenes. The
primary outcome requires native grip **and** TCP stationarity for five seconds.
Active controller qualification is a separate conjunction, so a fallback that
retains an object cannot count as active regulation.

This is an authored CPU simulation experiment. It does not train a policy or
evaluate videos, camera perception, complete placement or physical hardware.

## Run locally

Python 3.12, NumPy and MuJoCo are used. No GPU, API key or language model is needed.
The first dependency and pinned-asset installation needs network access;
subsequent simulation runs locally. This source preview is separate from the
stable local application. Minimum memory/CPU requirements have not been qualified.

```sh
git clone --branch piper-incremental-hold-v0.3-preview https://github.com/Dvidia-Inference/dvidia-training.git
cd dvidia-training
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install '.[arm]'
dvidia-robot-sim fetch piper --output runs/robot-profile-upstream/assets
```

Use your absolute asset path and a new output directory:

```sh
python -m dvidia_training.piper_incremental_benchmark prepare \
  --assets /absolute/path/to/robot-profile-upstream/assets \
  --output runs/piper-incremental-hold-v03
python -m dvidia_training.piper_incremental_benchmark run \
  --assets /absolute/path/to/robot-profile-upstream/assets \
  --output runs/piper-incremental-hold-v03
python -m dvidia_training.piper_incremental_benchmark inspect \
  --output runs/piper-incremental-hold-v03 > runs/piper-incremental-hold-v03/inspection.json
python benchmarks/piper-incremental-hold-v03/render_report.py \
  runs/piper-incremental-hold-v03/report.json
python benchmarks/piper-incremental-hold-v03/independent_audit.py \
  --study runs/piper-incremental-hold-v03 --repo . \
  --output runs/piper-incremental-hold-v03-audit/audit.json
python benchmarks/piper-incremental-hold-v03/package_evidence.py \
  --study runs/piper-incremental-hold-v03 \
  --audit runs/piper-incremental-hold-v03-audit \
  --output runs/piper-incremental-hold-v03-release
```

The run refuses changed source/runtime/assets and refuses overwriting evidence.
Each scheduled scene/controller pair runs once. No automatic outcome-based retry
or tuning occurs. The independent audit reads saved arrays with NumPy and the
standard library, without importing the controller, scorer or simulator.

The package contains all run and audit files with a deterministic inventory and
standalone archive verifier. Original DVIDIA code and synthetic records are MIT;
upstream assets and dependencies retain their licenses. No third-party arm
meshes, private footage, credentials or pretrained model weights are bundled.

## Why this experiment

The prior Cartesian update nearly cancelled measured joint drift against its
pose-error step. This candidate accumulates the current error step around the
admission-issued target, with fixed gain, rate, total-correction and pose bounds.
Review caught a zero-error clipping edge case; measured joints outside the target
margin now cause refusal before solving. Jaw requests and native gains are unchanged.

See the [frozen design](../../docs/designs/piper-incremental-hold-v03.md) and
[EGM/Harness² workflow research](../../docs/research/agent-workflows-20261031.md).
These frameworks inspired the review and comparison process; this is not a run
of either framework and does not claim their benchmark gains for robotics.

## Frozen result

Source freeze: `3f30847a7285a180f93b8b0beac0def1a7ceb912`. All 32 attempts
completed valid recordings: 640,000 native 1 ms rows and 32,000 commands.
Each variant admitted four scenes with zero false initial admissions.

| Controller | Native grip 5 s | Native stationary grip 5 s | Active stationary grip 5 s |
| --- | ---: | ---: | ---: |
| Measured joint | 0/8 | 0/8 | 0/8 |
| Issued joint | 4/8 | 0/8 | 0/8 |
| Cartesian | 4/8 | 0/8 | 0/8 |
| Incremental | 1/8 | 1/8 | 0/8 |

The candidate terminally refused every admitted hold after 0.32 s, when its
unlimited correction proposal exceeded 0.03 rad. Its one native stationary-grip
result includes fallback; it is not a successful active controller. Four
lower-friction scenes lacked admission in every variant. No controller is promoted.

[Research note](https://research.dvidia.org/papers/piper-incremental-hold-v03/) ·
[GitHub evidence](https://github.com/Dvidia-Inference/dvidia-training/releases/tag/piper-incremental-hold-v0.3-preview) ·
[Hugging Face](https://huggingface.co/datasets/Dvidia/piper-incremental-hold-v03)

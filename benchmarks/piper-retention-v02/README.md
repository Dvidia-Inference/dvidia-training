# Checking grip retention without changing the squeeze · PiPER v0.2

This frozen CPU study compares three authored ways of requesting a PiPER grasp hold across 12 fresh synthetic scenes: measured-joint, issued-joint and Cartesian. Each variant is scheduled once per scene (36 attempts). The primary outcome is whether the object remains continuously held for five simulated seconds after the policy's admission decision, checked against false confirmation. A second outcome also requires the TCP to stay within 4 mm and 0.05 rad of its admission pose for every step of that five-second pause; it applies the same threshold to every variant. Every valid failure and invalid attempt remains in the denominator.

This is not a learning study, full transport or placement task, physical-sensor test, hardware qualification or reliability estimate for real objects. The controllers and simulator use authored privileged state. A five-second simulator hold does not establish safe physical retention.

## Reproduce and inspect

Use Python 3.12 and a separate source checkout. This experiment is a source preview; the existing dev3 wheel does not include it. No GPU, language model or API key is required. Fetching dependencies and the pinned PiPER assets needs a network connection; the simulation then runs locally. Disk use is about 180 MiB for this recorded study, plus dependencies/assets; hardware minimums beyond this CPU run have not been qualified.

```sh
git clone --branch piper-retention-v0.2-preview https://github.com/Dvidia-Inference/dvidia-training.git
cd dvidia-training
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install '.[arm]'
dvidia-robot-sim fetch piper --output runs/robot-profile-upstream/assets
```

Use the absolute path to that asset directory below, and a fresh study output directory:

```sh
python -m dvidia_training.piper_retention_benchmark prepare \
  --assets /absolute/path/to/robot-profile-upstream/assets \
  --output runs/piper-retention-v02
python -m dvidia_training.piper_retention_benchmark run \
  --assets /absolute/path/to/robot-profile-upstream/assets \
  --output runs/piper-retention-v02
python -m dvidia_training.piper_retention_benchmark inspect \
  --output runs/piper-retention-v02 > runs/piper-retention-v02/inspection.json
python benchmarks/piper-retention-v02/render_report.py \
  runs/piper-retention-v02/report.json
python benchmarks/piper-retention-v02/independent_audit.py \
  --study runs/piper-retention-v02 --repo . \
  --output runs/piper-retention-v02-audit/audit.json
```

The renderer uses only Python's standard library. It reports all 36 scheduled outcomes, including failures, and marks unreported fields as unknown rather than inferring them.

## Evidence package

After the run and independent audit are complete, package the full run directory and audit directory without rerunning physics:

```sh
python benchmarks/piper-retention-v02/package_evidence.py \
  --study runs/piper-retention-v02 \
  --audit runs/piper-retention-v02-audit \
  --output runs/piper-retention-v02-release
```

The package is deterministic and retains every file from both inputs with a checked inventory. It produces the six-file dataset payload (`README.md`, `LICENSE`, ZIP archive, `report.json`, `protocol.json`, `inspection.json`) plus local package receipts. The package command refuses missing, mismatched or failed audit evidence. It does not publish.

## Frozen results

The source protocol was frozen at `6ab5e34ecbdd39ba58f4bb143135d0bfd0db3202`, before the 36 fresh attempts. All completed valid recordings. Each variant admitted the same four high-friction scenes, with zero false initial admissions; all eight lower-friction scenes lacked admission.

| Hold | Native grip for 5 s | Grip + stationary TCP for 5 s | Controller qualified at 5 s | Final controller + native |
|---|---:|---:|---:|---:|
| Measured joint | 2/12 | 2/12 | 2/12 | 0/12 |
| Issued joint | 4/12 | 0/12 | 4/12 | 4/12 |
| Cartesian | 4/12 | 0/12 | 0/12 | 0/12 |

Issued targets retained the four initially qualified grips, but TCP displacement reached 4.68–4.80 mm, beyond the common 4 mm bound. Cartesian refused all four admitted holds after 1.62–1.74 seconds; its native grip score includes the measured-position fallback. It is not a successful active stationary hold. No controller is promoted.

The complete archive also retains one earlier selected Cartesian development smoke under `development-smoke/`, excluded from these 36 attempts. It refused after 1.60 seconds beyond the same pose bound. No trial or parameter was retried or tuned after observing these results.

[Research paper](https://research.dvidia.org/papers/piper-retention-v02/) · [GitHub evidence release](https://github.com/Dvidia-Inference/dvidia-training/releases/tag/piper-retention-v0.2-preview) · [Hugging Face dataset](https://huggingface.co/datasets/Dvidia/piper-retention-v02)

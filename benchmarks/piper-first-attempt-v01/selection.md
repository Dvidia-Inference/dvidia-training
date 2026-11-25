# PiPER first-attempt policy handoff

Frozen November 25, 2026 at 14:32 UTC. The policy developer has not inspected the confirmation roster or outcomes. The two controller names are `corner-v8` and `corner-v8-timed`. The frozen policy SHA-256 is `b8ddb90c7c49c7426d3a39f60dd744b1983b97568957fc7d7049f44d2becede2`.

The public observation is one surveyed 320×240 RGB-D estimate of the initial cube XY, a specified goal on the same level support, 50 Hz encoders and robot-only FK. The policy is authored. It uses a diagonal pinch, a slowly ramped 8-mm test lift and quintic motion profiles. The timed ablation omits only the encoder aperture checks.

Both completed 12 of 12 development placements. The selected source reproduced all 12 development action digests exactly. The two policies also issued identical commands in each unperturbed development scene. Median strict completion took 14.3755 simulated seconds with 4.1433 mm terminal XY error, zero critical contacts and a 1.0099 N peak pad force.

## Evidence

- `benchmarks/piper-first-attempt-v01/development-history.json` records all eight iterations, the frozen replay and jaw ablation.
- `benchmarks/piper-first-attempt-v01/development-policy-snapshots.zip` contains the archived selection-time source.
- `benchmarks/piper-first-attempt-v01/grip-loss-diagnostic/` contains a separate, registered six-trial development fault study and its executable driver.
- `runs/first-attempt/policy-dev/frozen-corner-v8/` and `frozen-corner-v8-timed/` contain per-episode receipts and exact command tapes.

The confirmation runner should use the training venv, `PYTHONPATH=src`, the pinned PiPER assets, and `benchmarks/piper-first-attempt-v01/workspace/nominal.json` as its calibration. Follow the original plan for the untouched confirmation groups and policy ordering. Any supplemental rotated-cube or induced-fault work is development-only, with separate commands and receipts.

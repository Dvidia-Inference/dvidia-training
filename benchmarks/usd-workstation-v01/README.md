# DVIDIA OpenUSD workstation v0.1

A real OpenUSD parser maps one original rigid cube and a tabletop into a pinned native PiPER/MuJoCo workstation. All **45/45 passive physics checks pass**: 27 normal drops, nine normal slides and nine controls disabling gravity, support or friction. The arm holds home commands; this is not a learned manipulation success rate.

Independent task goals accept 27/27 normal drops and 6/9 normal slides. The other three slides stop outside the original XY goal. Every scheduled case is retained. Maximum early freefall position error is 0.2453 mm in this authored setup. The matrix took 47.613 seconds including scene construction on a shared Apple host; this is neither isolated throughput nor a hardware minimum.

The read-only audit re-scores all 45 cases and checks 7,845 saved frames with zero mismatches. It cannot reconstruct unrecorded 1 ms goal intervals or authenticate physical calibration. Full inputs, native poses, traces, source snapshots and license notices are in the release ZIP.

[Reproduction guide](../../docs/usd-workstation-v01.md) · [Source and recorded evidence release](https://github.com/Dvidia-Inference/dvidia-training/releases/tag/usd-workstation-v0.1) · [Research and replay](https://research.dvidia.org/papers/openusd-workstation-v01/) · [Hugging Face dataset](https://huggingface.co/datasets/Dvidia/usd-workstation-v01)

Frozen protocol: `383cd32d445365e66fbc74f4ec37d5385459b95a30d80051a46f70fdc6a0243b`. ZIP SHA-256: `037a7fb053c317ed4900771321dc3498a83ed8a92ce83f9a648e20b7f29947e6`. Original code and scene: MIT. Upstream dependencies retain their licenses.

This does not import arbitrary SimReady assets or USD articulations, estimate hidden materials from video, learn a policy or qualify a physical robot.

A separate [clean-install smoke](clean-install-smoke.json) on 30 October validates the installed package outside the checkout: CLI inspection/build pass and 1,200 native ticks reach the passive settling goal with zero native warnings. It uses an isolated Python 3.12 venv, pip 26.2.2 with bounded download resumption, no injected native environment and previously verified local robot assets. This single smoke is outside the frozen 45-case matrix and does not certify other hosts or minimum hardware.

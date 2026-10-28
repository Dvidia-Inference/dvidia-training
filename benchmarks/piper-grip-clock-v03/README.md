# DVIDIA PiPER grip clock v0.3

Simulation-only research: learn when to hold or advance a frozen manipulation reference. This release does not qualify a physical robot skill.

| Controller | Strict completions | Nominal | Assigned post-grip condition |
| --- | ---: | ---: | ---: |
| clock | 44/128 | 22/64 | 22/64 |
| blind_clock | 26/128 | 13/64 | 13/64 |
| fixed | 72/128 | 37/64 | 35/64 |
| teacher | 67/128 | 34/64 | 33/64 |

All 512 scheduled attempts remain, across 32 paired XY groups. Full sensing versus blind is +14.06 percentage points, with group interval -8.52 to +36.64 points; this does not establish a positive benefit. Versus fixed timing it is -21.88 points, interval -36.77 to -6.98 points. These are marginal 97.5% t intervals with Bonferroni simultaneous 95% coverage for the two contrasts. The model fails the declared qualification gates.

The clock uses 58 causal privileged simulator features and 64,000 synthetic hold/advance labels from all 64 teacher attempts, including unsuccessful demonstrations. The same virtual progress drives the six arm references and original jaw reference. The blind clock masks its live/history features. No new grasp geometry, human-video training, continuous learned slip recovery or physical transfer is established.

[Research and replay](https://research.dvidia.org/papers/learned-piper-grip-clock-v03/) · [Model](https://huggingface.co/Dvidia/piper-grip-clock-v03) · [Complete evidence](https://huggingface.co/datasets/Dvidia/piper-grip-clock-demonstrations-v03) · [Release](https://github.com/Dvidia-Inference/dvidia-training/releases/tag/piper-grip-clock-v0.3). Exact public revisions are recorded in the release publication receipt. `report-summary.json`, `paired-comparisons.json` and `audit-summary.json` preserve the measured results and audit scope without full native arrays.

The independent audit reconstructed 768 attempts with zero mismatches. Sampled readiness uses six 20 ms observations spanning 100 ms; 1 ms interval diagnostics are reported separately. A sampled readiness signal does not certify a continuously secure physical grip.

The known installed smoke is one nominal start (0.312, -0.037) m, goal 0; both installed models completed it. It was selected as an installation check and is outside the 128-case confirmation denominator. Its two successes are not an accuracy estimate. The portable parity and smoke receipts explicitly record inherited pinned dependencies rather than a fresh package resolver or Linux qualification.

Measured study host: Apple M5, ten logical CPUs, 24 GiB memory, no GPU for training/native simulation. Full-clock inference p95 was 1.916 ms; complete simulator-step p95 9.594 ms, maximum 265.909 ms, with 728 of 128,000 steps exceeding 20 ms. These exclude camera, communication, rendering and real actuators. The host is a reference, not a minimum requirement or hard real-time certification.

Use the compact release kit for wheel installation and the complete dataset for reproduction. DVIDIA code/models/evidence use MIT; bundled components retain their original license notices. This directory adds no new experiments.

# PiPER reliability v0.5 — three frozen fresh cohorts

The revised grasp completes **23/36 placements**, compared with **15/36** for the
frozen v0.4 controller and **18/36** with second-lift recovery alone. It gains eight
completions and loses none, reducing failures from **21 to 13** (38.1% fewer in
this narrow simulation comparison). The fifteen shared successful placements
average **0.772 simulated seconds slower** with the revised grasp. This is a
reliability improvement with a task-speed tradeoff, not faster motion everywhere.

| Fresh seed | Original v0.4 | Second lift | Revised grasp | Matched controls |
| --- | ---: | ---: | ---: | ---: |
| 832847 | 4/12 | 6/12 | 7/12 | 0/48 |
| 227037 | 6/12 | 6/12 | 8/12 | 0/48 |
| 862186 | 5/12 | 6/12 | 8/12 | 0/48 |
| Total | 15/36 | 18/36 | 23/36 | 0/144 |

All **252 attempts** are numerically valid, with zero solver warnings and zero
prohibited native contact ticks. All 144 negative controls fail. Twelve remaining
full-candidate failures never acquire a load-bearing low-friction grasp. One
moderate-friction 60 g case reaches full lift but loses retention and fails.
Neither a short lift, a resumed movement nor an earlier refusal counts as success.

Three bounded second probes occur in the lift-only candidate; all three qualify
and complete their tasks. Its fifteen shared successes have unchanged completion
clocks. The full revised-grasp candidate triggers no second probes in these fresh
cohorts, so its gains here do not demonstrate a contribution from the second lift.

## Complete evidence

Every cohort below retains its original **90-file** sealed bundle: 84 full
compressed attempts, report, protocol, integrity and original MIT model notices.
All valid episodes retain 20 simulated seconds, 1,000 commands, 1,001 native states
and every raw software computation duration. No failed attempt was discarded.

- **832847:** [Report](receipts/832847/report.html) · [JSON](receipts/832847/report.json) · [Protocol](receipts/832847/protocol.json) · [Seal](receipts/832847/integrity.json) · [Execution](executions/832847.json)
- **227037:** [Report](receipts/227037/report.html) · [JSON](receipts/227037/report.json) · [Protocol](receipts/227037/protocol.json) · [Seal](receipts/227037/integrity.json) · [Execution](executions/227037.json)
- **862186:** [Report](receipts/862186/report.html) · [JSON](receipts/862186/report.json) · [Protocol](receipts/862186/protocol.json) · [Seal](receipts/862186/integrity.json) · [Execution](executions/862186.json)
- [Complete aggregate, failures, forces and motion](analysis.json) · [Analysis source](analyze.py) · [Installed runtime check](installed-runtime.json) · [Loaded-host context](host-load.json) · [Publication provenance](provenance.json)
- [Verified Hugging Face publication](huggingface-publication.json) · [GitHub checks for the frozen code/results head](github-validation.json)

All 563 public Hugging Face payload files were downloaded and matched to the
prepared bytes at revision [d10b2c8a3b2484600585c16f897befc1ff0374b0](https://huggingface.co/spaces/Dvidia/robot-simulation-lab/tree/d10b2c8a3b2484600585c16f897befc1ff0374b0).
The GitHub snapshot records 28 passing checks for code/results commit
`0545ac47aedcd7c197341986533aa3e74c263ac6`; later publication-only commits have
their own checks. These outer publication records do not change the sealed runs.

The original 270 receipt files total **178,935,621 stored bytes**; with compressed
attempts decoded, seal-listed contents total **1,022,818,422 bytes**. Storage is
not runtime RAM. Execution, analysis and publication records sit outside the
original seals. OS network-denial self-checks and progress are retained verbatim
as text in `execution-logs/`; private local CLI output paths are not published.

## Frozen comparison and inspection

Runtime [e76b2e41256604d5b4314e45ca45959cab9aa571](https://github.com/Dvidia-Inference/dvidia-training/tree/e76b2e41256604d5b4314e45ca45959cab9aa571)
and all three protocols were committed before fresh execution. An installed wheel
matches all sixteen source/lock hashes. Every emitted protocol exactly matches
its preregistered JSON values; whitespace differs, so the original protocol file
hashes are reported separately. No controller or evaluation setting changed
between these cohorts, and no outcome-triggered substitute reruns were used.

Each scene preserves the same model, gains, force limits, starting state and
independent task scorer across seven variants. Each placement mode has its own
exact arm-command replay with open jaws, plus a common open-jaw hold. The native
1 ms integration, 20 ms observation/action cadence and 20-second budget remain.
The full task still requires 60 mm load-bearing lift, retained carry, supported
placement, release and retreat; the original contact-loss latch remains final.

Portable inspection checks the complete bounded seal, task evidence, matched
worlds/tapes, causal second-probe prerequisites, budgets and timing. The benchmark
also reconstructs native forward kinematics and every policy action, IK result,
diagnostic and event before returning success. It uses a recorded native lift
latch checked against sampled evidence. This is recorded consistency, not
independent physics reintegration or publisher authentication.

The [separate development archive](../piper-reliability-development-20261011/README.md)
retains 200 earlier attempts, including failed candidates and 24 exact v0.4
trajectory regressions. The fresh cohorts contain only six separately sampled
XY layouts reused across mass/friction strata. Counts are descriptive for this
rigid-block distribution; they do not estimate physical-robot reliability.

## Performance and force tradeoffs

The three offline installed commands run on a macOS arm64 laptop with ten logical
CPUs, Python 3.12.9, MuJoCo 3.15.0 and NumPy 2.5.3. No GPU is used. OpenBLAS and
OpenMP each use one thread. Dependencies and pinned assets were provisioned first.
OS process-level network denial is checked with an `EPERM` socket test.

| Measurement | Recorded result |
| --- | ---: |
| Full commands, including evaluation, compression and native policy audit | 964.796 wall seconds for 5,040 simulated seconds |
| Raw software computation windows | 252,000 |
| Pooled p50 / p95 / p99 | 1.813 / 7.453 / 14.050 ms |
| Maximum computation window | 3,082.138 ms |
| Windows exceeding 20 ms | 1,035 / 252,000 (0.411%) |
| Maximum child lifetime peak RSS including inspection | 816.27 MiB |
| Maximum observed pad force, original / second lift / revised grasp | 1.261 / 3.872 / 1.448 N |

The host was heavily loaded: one snapshot recorded a one-minute load average of
23.30 and 9,409.38 MiB of swap in use. All timings and overruns remain in the
receipts. The three whole-command times were 568.949, 218.985 and 176.862 seconds;
that variation prevents an isolated-host speed claim. These are software
computation windows, not physical sensor-to-actuator latency. A low pooled p95
does not excuse a three-second outlier or establish a real-time guarantee.

Peak contact forces are simulator diagnostics, not calibrated damage limits. The
second-lift variant's higher observed peak is retained even though its placement
count improves. Holding a target does not stop native inertia instantly; the
analysis reports sampled motion during each recovery and active probe.

[Install and run](../../docs/piper-reliability-v05.md) · [Research article](https://research.dvidia.org/papers/piper-reliability-v05/) · [Robot Simulation Lab](https://huggingface.co/spaces/Dvidia/robot-simulation-lab)

Original DVIDIA controller code and pinned PiPER assets retain MIT terms. This
is authored privileged-state simulation, not footage-trained competence,
deformable-object modeling, sensor robustness or physical hardware qualification.

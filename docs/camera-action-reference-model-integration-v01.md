# Reference-model integration results

Two reference-model candidates now complete the demonstration-to-action training, serialization and native-policy interface. Neither completes this task reliably: the separately registered development smoke achieved **0/4 strict placements for each candidate**. Both files and all failures are retained as unqualified research artifacts.

| Measurement | Absolute reference (Cabs) | Prior-reference delta (Cdelta) |
|---|---:|---:|
| Training rows | 96,000 | 96,000 |
| Fit wall time | 5.230713 s | 5.097628 s |
| Model JSON bytes | 539,345 | 538,021 |
| Recorded-trajectory callback p95 | 0.204840 ms | 0.207383 ms |
| Save/load output parity | 1,000/1,000 bitwise equal | 1,000/1,000 bitwise equal |
| Complete native development episodes | 4/4 | 4/4 |
| Bilateral grasp contact seen | 3/4 | 4/4 |
| Qualified lift seen | 1/4 | 0/4 |
| Carry completed / strict placement | 0/4 / 0/4 | 0/4 / 0/4 |
| Episodes with unexpected contacts | 2/4 | 4/4 |
| Unexpected-contact native ticks | 7,606 | 19,751 |
| Visual estimate coverage | 2,396/4,000 (59.9%) | 2,148/4,000 (53.7%) |
| Longest missing visual interval | 13.36 s | 6.16 s |
| Native callback plus guard p95 | 0.295460 ms | 0.284297 ms |
| Full offline boundary p95 | 22.362537 ms | 22.049043 ms |
| Qualification | No | No |

The common recipe is seed 17, two 64-unit tanh hidden layers, 50 epochs, batch 512 and Adam learning rate 0.003. Both fits use the same sealed 64,000 authored-teacher demonstration rows plus 32,000 earlier mixed-control correction rows, normalization and initial parameters. No new demonstrations, confirmation data, model selection or parameter tuning were used.

The 1,000-step integration replay is open loop along one recorded **training** teacher trajectory. Predictions never determine its later observations or references. The separate native smoke uses the first two protocol-ordered development groups, both destinations and both unchanged models, with no teacher query, intervention or fallback. All eight completed 1,000 control intervals with 501 RGB-D pairs per episode. Full artifact bytes, authority, timing contracts and strict scores were independently checked offline.

Native metric consistency passed for all eight episodes. That is distinct from the strict trajectory contract: all four delta episodes additionally return `invalid_state_value` because finite tool-centre coordinates cross its unchanged ±0.5 m component envelope. No NaN or native physics warning caused those rejections. Cabs achieved one qualified lift, but neither model completed carry or placement. Zero post-lift retention-loss events is not evidence of dependable grip when only one episode reaches qualified lift. Long visual dropouts and unexpected contacts remain important observed failure signals; these observations do not establish a causal diagnosis or justify selecting a model from four cases.

The original first eight attempts were refused during MuJoCo's macOS `sysctl` import probe. A separately registered startup correction passed that probe but the next eight were refused by CoreGraphics. A tiny static-frame preflight isolated graphics-session access; the third batch changed only the execution access, preserving source/models/cases/guards. All sixteen infrastructure refusals remain separate, rather than becoming a misleading 0/24 manipulation score.

Timing is measured on a shared macOS 26.5.2 arm64 host with Python 3.12.9, NumPy 2.5.3 and declared BLAS threads 1. Fit includes source/derived-input validation and normalization. Whole-process integration peak RSS was 1,345,404,928 bytes. Callback timing includes encoding, inference and reference projection; the native callback also includes its storage guard. Full-boundary timing includes file writes and alternate camera/tracker work, excluding the separate initial acquisition. The simulator pauses for this work. Its 22 ms p95 boundary is not a 50 Hz real-time delivery claim or a physical hardware measurement. The renderer warns that missing `ARB_clip_control` limits depth accuracy for its requested depth mode; no backend depth calibration was established.

This is an eight-attempt engineering smoke, not the proposed 164-attempt v0.2 screen. It contains no fixed, teacher or blind comparator; no negative-control qualification; and no generalization or confidence-interval claim. Frozen v0.1 remains 5/64 strict learned placements, unqualified.

[Benchmark, both model files, cards and evidence](../benchmarks/camera-action-reference-model-integration-v01/README.md) contain source, model and dataset identities. The candidates load through the Python `camera_action_reference_model` API. They are not installable through the qualified-skill CLI and provide no hardware interface. Training reproduction additionally requires the original independently pinned data exports; the compact benchmark does not redistribute those arrays or raw recordings.

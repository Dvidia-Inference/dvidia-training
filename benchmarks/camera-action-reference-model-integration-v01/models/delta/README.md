---
license: mit
language:
- en
tags:
- robotics
- simulation
- numpy
- research-candidate
---

# DVIDIA PiPER reference model — Cdelta

**Research candidate; not a qualified skill or hardware controller.** This card documents the retained `delta` candidate from the separately registered training/inference integration benchmark. A separately registered four-case development smoke completed for this candidate and achieved 0/4 strict placements. It remains unqualified.

The two candidates share the same 269-input, two-layer 64-unit tanh network, feature normalization, seeded initialization and optimizer recipe. Training uses 96,000 rows from 64 authored-teacher simulation demonstrations and 32 earlier mixed-control correction episodes, all within the original training split. Both candidates are retained; no model selection was performed.

Inputs include an explicitly historical initial visual estimate, destination, elapsed time, joint/gripper measurements, encoder-derived tool pose, previous applied commands and 160 ms of causal observation history with availability masks and age. Current teacher commands, teacher phase, native object/contact truth and future observations are excluded. Exact float64 prior normal servo references are separate decoder state; they are not reconstructed from float32 features.

The labels are reachable normal-reference endpoints computed by the original 20 limiter updates. Mixed-control teacher queries are counterfactual supervision; they are not relabeled as commands that were actually executed. This experiment does not recover actions from ordinary human videos.

The decoder is `r + D*g(x)`. `r0` is the fixed reset reference; `r` is the actual retained previous normal servo reference. `D` is 0.03 rad per joint and 0.001 m for the single-finger jaw coordinate. Runtime projection uses native bounds and the original 20 normal reference updates. Low training error does not establish stable feedback behavior.

## Recorded integration measurements

| Measurement | Value |
|---|---:|
| Training rows | 96,000 |
| Fit wall time | 5.097628 s |
| Model JSON bytes | 538,021 |
| Initial training loss | 8.72223173973e-05 |
| Final training loss | 3.53132778774e-06 |
| Open-loop callback median | 0.182729 ms |
| Open-loop callback p95 | 0.207383 ms |
| Serialized policy outputs | Bitwise identical at all 1,000 replay steps |

One fit used seed 17, 50 epochs, batch size 512 and Adam learning rate 0.003. Loss is the mean squared unprojected endpoint error normalized by scales of 3 rad per joint and 0.035 m for the jaw; it is a training-set descriptive metric. The loss includes no task-success scorer.

The 1,000-step replay follows the first protocol-ordered training teacher trajectory. Every next observation and reference remains the recorded teacher value. Predictions never enter simulation or feed the next state. Callback timing includes causal encoding, NumPy inference and reference projection; it excludes camera acquisition, rendering, physics and transport. Fit time includes source/derived-input validation and normalization. The shared host ran macOS 26.5.2, arm64, Python 3.12.9 and NumPy 2.5.3 with declared BLAS threads 1. Whole-process peak RSS across both fits and replays was 1,345,404,928 bytes; this is not a hardware minimum or model-only memory requirement.

## Artifact and source identity

- Artifact: [model.json](model.json), copied byte-for-byte from the retained integration result.
- File SHA-256: `fece6d056e50a1bde41bdb9960bf7cbc81d983c4bfdb03a07632e30b712b1489`.
- Canonical model content SHA-256: `4545be31f50267c9da29b211510c7a32df500678ba107c7a100604bb0c4c18ab`.
- [Pinned source](https://github.com/Dvidia-Inference/dvidia-training/tree/78ea4fbb4b2733e7738eb425eeecd7cb67a52195); model source identity `12ea9757cb0d76573c2e67f80b56ab08271f95e43f18b344b6481cc9153d70bc`.
- [Integration receipt](../../evidence/integration-result.json) and [training input identity](../../evidence/training-input-identity.json).

The Python API is `dvidia_training.camera_action_reference_model.load_model` / `ModelPolicy` at the pinned source. Existing qualified-skill installers and arbitrary-arm deployment are not validated for this new model type. The model's `scientific_registration` is intentionally null; the external benchmark registration identifies this operational integration test.

## Native evaluation and limitations

Two earlier eight-attempt batches remain retained separately: import-guard refusals, then CoreGraphics renderer refusals. A tiny static-frame preflight isolated the graphics access requirement. A third, separately registered batch used the same models, source, cases, cadence and guards with scoped graphics access. All eight 20-second simulated attempts completed, four per candidate, with zero teacher queries or fallbacks. No training or parameter selection followed these outcomes.

| Native development measurement | This candidate |
|---|---:|
| Complete independently checked attempts | 4/4 |
| Strict placements | 0/4 |
| Bilateral grasp contact seen | 4/4 |
| Qualified lift seen | 0/4 |
| Carry completed | 0/4 |
| Episodes with unexpected contacts | 4/4 |
| Unexpected-contact native ticks | 19,751 |
| Available visual estimates | 2,148/4,000 (53.7%) |
| Longest missing visual interval | 6.16 s |
| Callback including storage guard p95 | 0.284297 ms |
| Full offline boundary p95 | 22.049043 ms |

The delta candidate made bilateral contact in all four episodes but never reached a qualified lift. All four had unexpected contacts and the strict trajectory scorer returned `invalid_state_value`: finite tool-centre coordinates exceeded its unchanged component envelope of ±0.5 m. These were not NaNs or solver warnings. Native metric consistency alone does not imply that the strict trajectory contract passed. [Exact diagnosis](../../evidence/native-smoke-diagnosis.json).

These are four development cases per model, with no fixed/teacher/blind comparator, negative controls or uncertainty intervals in this engineering batch. They cannot establish generalization or relative superiority. Zero recorded retention-loss events cannot establish grip reliability when so few episodes reach the scorer's post-lift window.

The shared-host callback distribution includes the storage guard; the model's diagnostic timer excludes it. Full-boundary wall timing includes file writes and alternate camera/tracker work. The simulator pauses for that work; this is not a real-time robot delivery measurement. The renderer warned that `ARB_clip_control` was unavailable for the requested depth mode, limiting depth accuracy. This backend was not depth-calibrated by the smoke test.

[Exact eight-attempt audit](../../evidence/native-smoke-v3-audit.json), [all three batches](../../evidence/development-smoke-status.json), [renderer warnings](../../evidence/renderer-warnings.json) and [graphics preflight](../../evidence/graphics-preflight.json) retain the evidence and limitations.

No confirmation-set evaluation, generalization claim, candidate selection, physical robot action or qualification has occurred. The frozen v0.1 result and the proposed 164-attempt screen remain unchanged. This model applies only to the bounded simulated PiPER block-placement setup and its exact sensing/reference contracts.

## License

DVIDIA candidate weights and accompanying DVIDIA code/documentation are offered under [MIT](../../LICENSE). Upstream dependencies and robot assets retain their own licenses; this benchmark does not bundle them.

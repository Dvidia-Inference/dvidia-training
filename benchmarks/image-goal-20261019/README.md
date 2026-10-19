# Jev image-goal evaluation: first measured run

DVIDIA research · 19 October 2026 · measured follow-up to protocol v0.1

All **64 planned Jev 1.14.0 requests returned valid responses**, with no retries,
API errors or omitted trials. The 32 native fixtures, prompts, thresholds and
scorer are unchanged from the [published protocol](../image-goal-20261018/README.md).
The original unrun release remains available; this follow-up adds measurements.

- [Research article](https://research.dvidia.org/papers/image-goal-jev-v02/)
- [Inspect every measured case on Hugging Face](https://dvidia-robot-simulation-lab.static.hf.space/image-goals/v02/index.html)
- [Metrics and provenance](metrics.json)
- [Reproduce the benchmark](../../docs/image-goal-benchmark.md)

## Results

| Measure | Images only | Images + native telemetry |
| --- | ---: | ---: |
| Valid responses | 32/32 | 32/32 |
| Observability-aware verdict accuracy | 24/32 (75%) | 25/32 (78.125%) |
| Physical false completion | 0/24 | 0/24 |
| Actual complete states called complete | 0/8 | 1/8 |
| Correct abstention when inputs are ambiguous | 8/16 | No ambiguous reference cases |
| Explicit decisions | 24/32 | 29/32 |
| False presence on absent targets | 0/8 | 0/8 |
| Expected presence decision accuracy | 28/32 | 27/32 |
| Completion-probability Brier score | 0.185772 | 0.117438 |
| Online latency p50 | 1.585 s | 1.914 s |
| Online latency p95 | 3.548 s | 3.221 s |

Zero false completion does not establish a useful completion detector by itself.
The telemetry condition recognized only one of eight physically completed states;
an always-incomplete baseline scores 24/32 on that condition, versus Jev's 25/32.
No RGB-only complete verdict was justified by the frozen reference: a single
frame cannot establish hidden velocity, contact or sustained support.

The RGB and telemetry accuracy columns have different expected labels. Their
difference is not a controlled estimate of a general accuracy improvement.
The paired cases provide a more specific result: all four moving controls changed
from a justified RGB abstention to a correct incomplete verdict with telemetry.
All four RGB settled controls also abstained correctly, but three still abstained
with telemetry. The physically complete occluded target was called incomplete
in every layout under both input conditions. Visible unreleased grippers were
called incomplete in both conditions; the RGB reference required unknown because
an image cannot prove the declared contact-force threshold.

All eight absent-target trials per condition received incomplete verdicts.
Presence probabilities did not always cross the frozen absence threshold:
zero false presence is not equivalent to perfect absence detection.

## Speed, cost and interpretation

Across all 64 requests, online latency was **1.776 s p50, 3.287 s p95,
3.950 s p99 and 4.091 s maximum**. Timing includes request construction, upload,
network, hosted response and parsing from preloaded images. It excludes rendering,
camera capture and robot motion. A single serial run cannot establish production
tail latency or a real-time deadline.

The input-token estimate is **$0.007644756** across the run, using the price frozen
on 18 October and returned usage for all 64 requests. It is not a billing receipt.

These results support investigating Jev as a supervisory image-goal evaluator.
They do not justify putting this hosted call in a fast grip, contact or actuator
loop. A local verifier can check measured physical constraints; the visual layer
still needs testing for object identity, occlusion, uncertainty and evidence use.
This is a small diagnostic with four correlated layouts, privileged simulator
telemetry and a narrowed 30 g rigid shoe proxy. It demonstrates neither a learned
controller nor real-robot transfer.

## Reproducibility and credential handling

The frozen generator, client and scorer are commit
`19acf0734d63ecebc07140c8d84302a54baee4c4`. The evaluation plan and original result
hashes are in `metrics.json`. Publication tooling is versioned separately.
No prompts, cases or thresholds were tuned after observing these responses.

The credential was supplied through a hidden terminal prompt to process memory.
It was not placed in shell arguments or environment files. An exact-value scan
of 1,677 tracked repository and run files found zero matches. Public exports omit
raw provider responses and pass the publication scanner before upload.

Original DVIDIA code and authored fixtures: MIT. PiPER retains its upstream
notice. Hosted Jev remains subject to TypeSafe's separate terms; no model weights,
customer footage, training or distillation are part of this release.

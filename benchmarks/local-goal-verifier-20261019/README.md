# Local physical-goal verifier v0.3

DVIDIA research · 19 October 2026 · 200 fixed cases · zero provider calls

The local verifier recognized **16/16 completed native scenes** and rejected
all **48 incomplete native scenes**. It checks explicit measured conditions
rather than asking an image model to interpret physical thresholds. This is a
verification component, not a learned perception system or robot controller.

- [Interactive evidence on Hugging Face](https://dvidia-robot-simulation-lab.static.hf.space/image-goals/v03/index.html)
- [Research article](https://research.dvidia.org/papers/local-goal-verifier-v03/)
- [Run locally and inspect the contract](../../docs/local-goal-verifier.md)
- [Metrics and provenance](metrics.json) · [Frozen protocol](protocol.json)
- [Prior Jev response forensics](jev-forensics.json)
- [Independent audit](INDEPENDENT-REVIEW.md) · [Machine-readable audit](independent-review.json)

## Why the prior run recognized only one of eight

We reconstructed all 64 original requests and reparsed the saved responses.
The images, telemetry, request hashes and typed answers match the frozen run.
There was no missing telemetry or client completion-probability cutoff: the
completion verdict came directly from the provider's Choice answer.

Of eight physically complete scenes, three visible ones received `unknown`,
four covered ones received `incomplete`, and one visible scene received
`complete`. Every positive's supplied native measurements satisfied the physical
requirements. Covered and visible controls had identical measured-state payloads.
This demonstrates unreliable use of sufficient telemetry on those cases. The
model's internal reason is unknown; we did not evaluate an alternative prompt,
new model, or repeated provider samples.

The new verifier computes pose error and checks motion, support and release
over the latest 100 observations at a declared 2 ms cadence. It requires trusted
identity, coordinate frame, goal and current clock, and explicit uncertainty
bounds. Definite physical violations return incomplete. Missing, invalid, stale
or insufficient evidence returns unknown. Passing every condition returns complete.

## Fresh benchmark

| Cohort | Cases | Complete / incomplete / unknown | Physical false completion |
| --- | ---: | ---: | ---: |
| Fresh native simulation | 64 | 16 / 48 / 0 | 0/48 |
| Perturbed evidence on completed states | 128 | 0 / 8 / 120 | Not applicable: no physical negatives |
| Consistent sensor forgeries on moving states | 8 | 8 / 0 / 0 | **8/8** |
| All scheduled cases | 200 | 24 / 56 / 120 | **8/56** |

Native positives include eight visible and eight occluded completed states.
The other six scenarios per layout cover missing targets, a decoy object,
incorrect position/orientation, unreleased contact and motion. Moving controls
share their completed control's exact image and pose, but have an explicitly
assigned native velocity. No artificial object attachment, gravity compensation
or external force was introduced.

All 128 fault cases agree with the prespecified evidence verdict: 120 return
unknown for insufficient or invalid evidence, and eight fabricated force spikes
produce incomplete. Their underlying physical states remain complete. These are
conservative evidence checks, not 128 extra robot tasks completed successfully.

All eight self-consistent forged observation streams fool the verifier. They
are retained as **physical false completions**, even though complete is the
arithmetically correct verdict for the supplied packet. The benchmark therefore
has 200/200 packet-contract agreement but eight physical false completions.
This distinction is the reason to publish separate cohorts. A trusted sensor
path and independent calibration remain necessary.

All 64 native cases constructed successfully; no cases were discarded or
retried. Physical truth is derived by the frozen native measurement code,
independently of the new verifier. The protocol uses eight seeded layouts and
fixed thresholds; one default-seed layout was inspected in an initial unit check
without parameter tuning. The full development preflight used seed 20261018;
the measured run uses 20261019. This is not claimed as a fully unseen test set.

## Compute and scope

There are 100 timed verifier calls per case after ten warm-ups: 20,000 timed
calls total. Native cases take **0.295 ms p50, 0.317 ms p95, 0.386 ms p99 and
0.799 ms maximum**. Across all cohorts: **0.267 ms p50, 0.318 ms p95,
0.398 ms p99 and 0.963 ms maximum**. Early rejection paths and corrupted
packets make the aggregate timing different from native timing.

Timing surrounds the verifier with parsed inputs already in memory. It excludes
sensing, the required **200 ms history accumulation**, rendering, serialization,
networking and actuation. This is a repeated-call microbenchmark on an arm64
Mac with ten logical CPUs, Python 3.12.9 and Darwin 25.5.0; it is not a real-time
deadline or an end-to-end reaction measurement. It must not be presented as a
speedup over the prior Jev online-request measurement, which timed different work.
Native fixture generation, rendering and control-file writing took 13.101 seconds.
No GPU or hosted model was used by the verifier.

The narrow 30 g rigid proxy and privileged native state remain substantial
limits. This does not validate deformable real shoes, sensor identity estimation,
force measurement, calibrated uncertainty, manipulation success or hardware
transfer. Zero uncertainty means exact-native-state assumptions, not a claim
about real sensors. A sampled window also does not establish continuous safety.

## Reproduce and audit

Benchmark source commit: `b4c76b13ccbdaff4f01726bee50fae5fcc1e4e4a`.
The source and twelve dependency hashes are frozen in `protocol.json`. Native
assets retain PiPER upstream commit `0059d4335f8156206f63a35662313385f7ad6d74`.
The earlier generator, Jev client and scorer remain unchanged. Protocol,
inputs and results have separate hashes in `metrics.json`.

The public evidence includes every packet, native state and image, all 20,000
timings, complete decisions, the input seal and publication seal. Generation
failures abort scoring with sealed diagnostics; category mismatches with valid
packets stay in the measured inventory. No credential is needed or included.

Original DVIDIA source and authored fixtures are MIT. Upstream PiPER and MuJoCo
notices remain applicable. No Jev weights, new training or raw customer footage
are redistributed. The next step is to test estimated observations and calibrated
uncertainty against independent truth, then connect the verifier to bounded
controller recovery—not to treat simulator truth as a solved perception system.

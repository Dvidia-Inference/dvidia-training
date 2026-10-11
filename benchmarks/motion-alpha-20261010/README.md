# DVIDIA image-motion alpha — 10 October 2026

**Six authored fixtures processed successfully; real-world accuracy and live
robot latency remain unmeasured.** This evaluates sparse tracking of a manually
seeded image region and passive constant-velocity forecasting. It evaluates
neither an object detector nor a trained interaction or robot-control model.

The [frozen protocol](protocol.json) and [aggregate results](result.json)
preserve parameters, input/reference hashes, software, attempted counts,
conditional errors, abstentions and timing scope. They contain no input footage,
model weights, private data or credentials. The generated fixtures and these
DVIDIA artifacts are MIT licensed. The existing project notices continue to
apply to upstream dependencies.

## Frozen inputs and causal separation

Each case is a four-second, 640 × 360, 30 fps video: 120 source frames, 720 frames
and 24 seconds across the batch. One 80 × 64 pixel textured rectangle translates
over a uniform background. The seed is manually authored inside that rectangle
on the first frame. Its center is the image anchor; it is not a measured physical
centroid. The low-texture negative uses an untextured rectangle. The occlusion
case removes the rectangle for frames 45–64 while its authored trajectory
continues.

The deterministic generator freezes trajectories, encoded source hashes and
actual frame presentation timestamps before tracking. Reference timestamps come
from integer video PTS multiplied by its rational time base. The tracker receives
only the manifest, initial seed and processing cadence. Future reference
positions are opened by the evaluator after processing and checked against their
frozen hashes. No future reference, simulator state or video frame enters an
earlier motion estimate or forecast.

The tracker reports the initial seed center plus accumulated robust feature
translation. Its current estimate is the observed anchor in this unsmoothed
baseline. Velocity uses two contiguous actual observations and their actual
time difference. A lost track stays lost, with no replacement, interpolation or
automatic reacquisition. Missing evidence becomes an abstention.

## Tracking results

| Authored case | Tracked / visible reference frames | Visible track coverage | Anchor error p95, px | Velocity error p95, px/s |
| --- | ---: | ---: | ---: | ---: |
| Stationary | 120 / 120 | 100% | 0 | 0 |
| Constant velocity | 120 / 120 | 100% | 0.00691 | 0.0574 |
| Stop/start | 120 / 120 | 100% | 0.00698 | 0.0621 |
| Direction change | 120 / 120 | 100% | 0.00714 | 0.0627 |
| Brief occlusion | 45 / 100 | 45% | 0.00722 | 0.0582 |
| Low texture | 0 / 120 | 0% | Unavailable | Unavailable |

All 720 source frames have attempted observation rows. All four textured cases
without occlusion retained their anchors. The occlusion case lost the track
when the object disappeared and remained lost after reappearance: 75 lost rows,
including 55 visible rows. The low-texture case abstained on all 120 frames.
Neither negative case is a perfect prediction result; the accuracy sample
counts become zero where no valid anchor/forecast exists. Velocity scores
exclude the two-frame neighborhood of authored speed discontinuities.

## Forecast results and failure tails

Errors are image-plane endpoint distances in pixels. Hold-current and
constant-velocity forecasts below share the same evaluable current anchors,
reference targets and sample counts. Scores are conditional on producing a
forecast; their answer coverage is reported separately.

| Authored case | 100 ms CV mean / hold mean, px | 500 ms CV mean / hold mean, px | 500 ms CV p95 / worst, px | Forecast samples at 100 / 500 ms |
| --- | ---: | ---: | ---: | ---: |
| Stationary | 0 / 0 | 0 / 0 | 0 / 0 | 116 / 104 |
| Constant velocity | 0.00675 / 9.005 | 0.0252 / 45.005 | 0.0337 / 0.0914 | 116 / 104 |
| Stop/start | 0.317 / 5.903 | 6.937 / 27.699 | 38.531 / 45.007 | 116 / 104 |
| Direction change | 0.317 / 8.897 | 6.945 / 41.769 | 59.109 / 89.975 | 116 / 104 |
| Brief occlusion | 0.00664 / 9.006 | 0.0252 / 45.006 | 0.0342 / 0.0914 | 44 / 44 |
| Low texture | Unavailable | Unavailable | Unavailable | 0 / 0 |

Every case has 117 available future reference frames at 100 ms and 105 at
500 ms. Terminal forecasts are censored: three and fifteen rows respectively.
The first tracked row has no velocity and therefore no forecast. The four
continuous textured cases answer on 116/117 and 104/105 available targets.
Occlusion answers on 44/117 and 44/105; low texture answers on zero. The remaining
rows are explicit abstentions, not zero-error samples. All 250 ms metrics and
denominators are also retained in the result file.

Current anchors match the same acquisition PTS within a 2 μs numeric
serialization tolerance. Off-grid observations cannot borrow nearby reference
truth. Future targets use the nearest actual source frame within half a frame
(1/60 second), with the earlier frame chosen on an exact tie. The 250 ms horizon
lands between 30 Hz frames; its reference is therefore approximately 16.67 ms
from the nominal target. The constant-velocity case consequently has about
1.49 px mean error there even though its 100/500 ms error is much smaller.
These are nearest-frame endpoint scores, with offsets disclosed, rather than
exact-time 250 ms accuracy. No backward interpolation produces new references.

The worst direction-change forecast misses by approximately 90 px at 500 ms.
Its low whole-case mean does not make it reliable at the turn. Passive
extrapolation cannot know an unannounced future reversal or contact action.
Interaction-conditioned models and evaluation slices remain subsequent work.
The propagated uncertainty radii are heuristic and uncalibrated; their empirical
coverage and size are recorded without probability claims.

## Processing and reproducibility

Six of six attempted cases completed with no pipeline failures. Processing took
**1.454 seconds**, plus **1.514 seconds** for fixture generation. The offline
batch-time/source-duration ratio is **0.0606**. Processing includes intake,
probing, streaming decode, causal tracking/forecast, writes and scoring; it
excludes fixture generation and provisioning. The ratio is withheld for any
failed batch so early failure cannot appear to improve throughput.

This was one run on Darwin 25.5.0, ARM64, ten logical CPUs, Python 3.12.9,
OpenCV 5.0.0 and NumPy 2.5.3. One OpenCV thread was requested, while OpenCV
reported ten. Actual peak thread concurrency, enforcement of the requested
limit and a minimum hardware requirement are unmeasured.
The process-lifetime resident high-water mark was **83,591,168 bytes
(79.72 MiB)**, including fixture generation; it is not a per-stage limit.
No GPU or paid inference API is used in this baseline. A fast batch result does
not establish live capture-to-action latency, worst-case deadlines, long-running
stability or robot-control benefit.

From an installed checkout containing the motion alpha, with the observations
extra and FFmpeg/ffprobe installed:

```sh
dvidia-motion benchmark --output runs/my-motion-benchmark
dvidia-motion inspect runs/my-motion-benchmark
dvidia-motion studio runs/my-motion-benchmark --port 8286
```

Use a new output directory each time. Complete local replay bundles retain
generated videos, actual-clock frame metadata, causal rows, seed evidence and
all attempted receipts. This public aggregate is not itself a replay bundle.
Generator and implementation hashes bind the protocol; encoder/platform
differences may change encoded-file hashes, so compare declared software and
parameters before interpreting reproduction differences.

These six cases share one authoring recipe and texture. They are engineering
checks rather than six independent real-world task attempts or a held-out
generalization cohort. Next evidence should include unseen permitted footage,
camera movement, blur, deformation and separate live stream measurements.

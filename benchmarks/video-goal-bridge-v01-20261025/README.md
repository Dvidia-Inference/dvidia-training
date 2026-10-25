# Video destination grounding: PiPER pilot v0.1

This executable bridge extracts a destination from encoded video and hands it to
the existing PiPER controller. It trains **no motor policy**. Inputs show a known
green rigid block in simulation with a declared fixed orthographic camera, not
human footage or a shoe skill.

[Open the local report](evidence/report.html) · [Frozen protocol](evidence/protocol.json)
· [All attempted outcomes](evidence/report.json) · [Post-run error analysis](analysis.json)
· [Integrity manifest](evidence/integrity.json) · [Implementation and references](../../docs/video-goal-bridge-v01.md)

## Frozen run

Four seeded layouts produced four successful native source transfers. Each
recorded trajectory was rendered with actual MuJoCo geometry into a 320×240,
5 fps, 20.2-second MP4. The destination marker was hidden. Inference received
only pixels and calibration, without native poses, commands, intended goals or
outcome labels. Native traces remain separate. The controller still receives
privileged live simulator state.

| Variant | Intended-task successes / scheduled | Native attempts executed |
| --- | ---: | ---: |
| Oracle intended destination | 4/4 | 4 |
| Video-derived destination | 3/4 | 4 |
| Destination from another clip | 0/4 | 4 |
| Video with 60 mm calibration offset | 0/4 | 3 |
| Exact video-goal arm tape, gripper held open | 0/4 | 4 |

All 20 scheduled controller attempts remain in the denominator. One
miscalibrated destination was rejected before physics because it did not meet
the task's minimum transfer distance. In total the run executed four source
episodes and 19 controller episodes. Wall time was **122.54 seconds**, including
native physics, rendering, encoding, inference, controls and report construction,
after protocol/source snapshots and asset verification.

The common task score checks valid physics, retained lift/transfer,
supported/released/settled placement within 25 mm of the original intended task
center, and tool retreat. Controller self-success cannot replace it. One
self-reported success failed this reference in the video variant, four in the
shuffled variant, and three in the executed calibration controls. A controller
can successfully execute the wrong instruction.

## Perception and fault checks

| Layout | Proposal error vs observed source endpoint | Video task error | Task result |
| --- | ---: | ---: | --- |
| 00 | 5.13 mm | 13.75 mm | Pass |
| 01 | 1.78 mm | 22.02 mm | Pass |
| 02 | 2.70 mm | 27.79 mm | Fail |
| 03 | 4.01 mm | 15.74 mm | Pass |

All proposals met the predeclared 10 mm diagnostic tolerance. Only **3/4**
fell inside the caller-declared 5 mm position bound; that bound is unestablished.
A color centroid is neither authenticated identity nor a general pose estimator.

Four clips per pixel intervention were encoded and decoded. The observer
returned `unknown` for **4/4 terminal occlusions, 4/4 extra green candidates and
4/4 terminal camera-motion clips**. These are engineered controls, not population
accuracy or robustness estimates. Five-Hz sampling leaves unobserved motion
between frames and a 200 ms tail in these clips; continuous physical settling,
depth, support, identity and brief terminal changes remain unestablished.

Each clean 20.2-second clip took **0.133–0.572 seconds** for local file checks,
probing, decoding, color observation and hashing. These are four one-shot timings
including cold/warm runtime effects, not a latency distribution or live deadline.
They exclude camera acquisition, terminal observation dwell, networking and
physical actuation. Vision ran on CPU without a hosted model; native rendering
used macOS graphics. No GPU training or language-model inference was performed.

## Why layout 02 failed

The retained vectors show three offsets largely in the same world-X direction:

| Stage | Offset magnitude |
| --- | ---: |
| Demonstrated endpoint vs original intended center | 12.04 mm |
| Pixel destination vs demonstrated endpoint | 2.70 mm |
| New execution endpoint vs pixel destination | 13.05 mm |
| New execution endpoint vs original intended center | 27.79 mm |

Vectors add; their magnitudes do not generally add. These measurements locate
the accumulated offsets but do not establish the physical cause of the source
or execution bias. The failed attempt retained its grasp, released, retreated
and declared success against its own destination; the common reference caught it.

The controls have an information difference: the oracle receives the exact
intended center, while video shows an imperfect demonstrated outcome. A labeled,
exploratory post-run check asks whether the new execution matches that
**observed endpoint** under the same 25 mm criteria. That secondary
destination-replay score is **4/4**. It does not replace the predeclared
**3/4 intended-task score** or establish new policy learning.

The next problem is a reusable placement skill with a measured error budget and
corrective feedback, so it avoids adding its placement bias to an imperfect
demonstration. New motor learning then needs paired robot actions or a validated
action-inference bridge, with fixed-label-budget and no-video controls.

## Reproduce and inspect

Exact source snapshots/hashes are retained in [evidence/sources/](evidence/sources/).
Install project requirements and the optional `arm` extra, FFmpeg/ffprobe and
the existing pinned PiPER assets. Native rendering needs a graphics context.
After dependency and asset provisioning this experiment uses no network or API key.

```sh
python -m dvidia_training.video_goal_benchmark run \
  --assets /absolute/path/to/robot-profile-upstream/assets \
  --output runs/video-goal-fresh --count 4 --seed 2026102507
python -m dvidia_training.video_goal_benchmark inspect \
  benchmarks/video-goal-bridge-v01-20261025/evidence --reprocess-video
python -m dvidia_training.video_goal_analysis \
  benchmarks/video-goal-bridge-v01-20261025/evidence --output /tmp/new-video-goal-analysis.json
```

All **91 sealed files** passed integrity checks; source/fault videos were
reprocessed and native-trace scores recomputed. Inspection does not rerun native
physics. Fifteen focused tests passed, including real MP4 decode, calibration,
terminal refusal, and rejection of wrong-goal controller self-success. Four
layouts are a small mechanism test, not a general reliability estimate, shoe
competence or physical qualification.

Original DVIDIA code and generated evidence are MIT. Rendered PiPER assets
retain their upstream MIT notice in [PIPER-ASSET-LICENSE.txt](PIPER-ASSET-LICENSE.txt).
MuJoCo and other dependencies retain their existing licenses. No upstream model
weights, private media or physical robot recordings are included.

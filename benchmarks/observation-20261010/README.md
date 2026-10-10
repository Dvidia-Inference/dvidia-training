# Local observation feasibility — 10 October 2026

**Engineering integration passed on six public excerpts; accuracy not assessed.**
This is not a cup-placement benchmark, 3D reconstruction or trained robot skill.
The [plan](../../docs/observation-pilot-plan.md) was written before the model run.

## Inputs and execution

The fixed order was shirt, stir, pack, onion, melon, jeans, selected from existing
local public examples before examining model output. Five are attributed
EgoAnnotate excerpts; stir is a CoMind excerpt. Licenses and public source links
are preserved in both protocol files. Exact excerpts were hash-verified. The
full upstream originals were not downloaded or hash-verified for this pilot.
No raw media, sampled frames or private user metadata are included in this report.

Six excerpts total 19 seconds, 2,753,287 bytes and 38 sampled frames. They are not
six independent complete attempts; unknown session grouping remains one
conservative shared group. No train/test split or learning run took place.
Publisher titles were never fed to the detector.

| Attempt | Complete / attempted | Processing time | Failures |
| --- | --- | ---: | ---: |
| Initial recipe v1 | 6 / 6 | 4.835 seconds | 0 |
| Hardened recipe v1.1 | 6 / 6 | 4.635 seconds | 0 |

Timing covers intake, copy/hash, frame decoding, detection, association and
episode writes. It excludes model provisioning/loading, final receipt writing,
UI startup and human review. Single runs, warm machine; this is neither a p95
latency measurement nor a production throughput guarantee. Peak RAM, energy and
dollar cost were not measured. No paid inference API was called.

Runtime: Python 3.12.9, OpenCV 5.0.0.93, NumPy 2.5.3, macOS ARM64 and one
inference worker. Four OpenCV threads were requested, but the saved model
metadata records `cv2.getNumThreads()` as ten after model loading. Actual peak
thread concurrency was not measured; a four-thread cap is not established.
Both pinned ONNX models total approximately 40 MB. OpenCV
5 printed that target selection is unsupported by its new graph engine; this
build uses the OpenCV CPU backend and no GPU path was requested or measured.

## What changed after the first run

Review identified a possible browser-clock shift for videos with nonzero start
timestamps. Recipe v1.1 rejects nonzero container/video/decoded starts rather
than moving boxes relative to the unchanged original. The initial artifacts are
preserved alongside the subsequent run, which processed the same six inputs.
All six inputs have zero starts and were accepted in both runs.

The review server was also hardened against same-size media replacement,
malformed review IDs and oversized run metadata. These are workflow fixes,
not improvements in detector quality.

## Observed limitations

- The three-frame preliminary adapter check included a saucepan incorrectly
  labeled cup and storage jars labeled cups. Hands were missed in the stir
  sample. These are qualitative findings, not scored accuracy estimates.
- COCO labels do not cover every task object; palm rectangles do not cover full
  hands or prove contact. Two frames/second can miss short actions.
- Association uses same-class image overlap. It is not optical flow and can
  fragment or switch identities. Overlap proposals do not establish grasping.
- Outcomes remain unknown. Model scores are uncalibrated, and the pilot creates
  no automatic natural-language action captions or robot commands.
- The intended permission-cleared cup-placement cohort and independent reference
  annotations have not been collected. Accuracy, false-completion rate and
  correction time remain unmeasured. No model ranking is established.

## Evidence

- `initial-protocol.json`, `initial-run.json`: original frozen inputs and receipt.
- `hardened-protocol.json`, `hardened-run.json`: later timing policy and receipt.
- `episode-summary.json`: durations, proposal counts, timing and excerpt hashes.
- Local ignored run directories retain media, frames and raw episode annotations.
  Run and source hashes bind the receipts to those local artifacts; this report
  alone cannot reproduce a visual inspection without obtaining permitted clips.

Browser interaction testing uses a separate copied run. Test review edits do not
count as independent human annotations and do not alter the original run's
unreviewed status.

## Implementation validation

The repository suite passed 307 tests, followed by 24 focused observation tests
including four additional detector-admission/suppression tests. These exercise
real FFmpeg decoding, timestamps, source tampering, failed attempts, malformed
reviews, revision conflicts, local-origin enforcement and video byte ranges.

Fifteen browser checks passed on an isolated review copy: playback, timestamped
overlays, correction save/reload, preservation of raw labels, manual events,
export, unsaved-edit protection, completion validation and 390/320px layouts.
There were no console/page errors. See `ui-validation.json`. Headless browser
checks are not physical-phone or Safari qualification.

The wheel built successfully; its final HTML, command entry points and both
upstream license files were verified. Raw footage/model files are Git-ignored.
Production uploads, accounts, storage and website deployment were not modified.

## Next decision

Collect 30–50 complete cup-placement attempts with permission and explicit
session/person grouping, including failures, occlusion and recovery. Establish
blinded reference labels first. Then compare object-only, temporal caption-only,
combined proposals and manual annotation at matched quality and total cost.
Only after that should an app integration claim reliable task labeling.

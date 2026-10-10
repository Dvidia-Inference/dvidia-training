# Local video observation lab

This is a runnable research feature in DVIDIA Training. It does not replace
the upload flow or activate a production analysis service. The implementation
owns intake, timestamps, proposal association, reviews and provenance. The two
optional pretrained detectors retain their own licenses; no new foundation
model has been trained and no SpatialLM weights are included.

## Setup

From this repository checkout, with Python 3.12+ and FFmpeg/ffprobe installed:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install '.[observations]'
.venv/bin/python scripts/download_observation_models.py --download --output runs/observation-models
```

The explicit provisioning command downloads about 40 MB from pinned official
OpenCV Zoo artifacts and verifies sizes and SHA-256 hashes. Inference never
downloads anything. Install software and models before disconnecting from the
network. The tested runtime is Python 3.12.9, OpenCV 5.0.0.93, NumPy 2.5.3 on
macOS ARM64. Other platforms have not been benchmarked. Model loading and
software installation are excluded from the published six-clip processing time.

## Supply permitted videos

Keep the following manifest next to your own files. Replace every example value,
including the hash; it is deliberately not a runnable dataset. Do not add raw
recordings, private metadata or local paths to Git.

```json
{
  "kind": "dvidia.observation-source.v1",
  "task": "Move a cup onto a marked surface",
  "episodes": [{
    "id": "attempt-001",
    "path": "attempt-001.mp4",
    "title": "Cup placement, first attempt",
    "session_id": "participant-a-session-1",
    "source_recording_id": "consented-original-001",
    "kind": "user_authorized",
    "credit": "Consenting contributor; local analysis only",
    "license": "Private; local analysis permission recorded",
    "source_url": "local-authorized-recording",
    "permission": "local_analysis",
    "sha256": "REPLACE_WITH_THE_ORIGINAL_FILE_SHA256"
  }]
}
```

The permission field records a declaration; it does not obtain consent or grant
rights. Use the same session/person/original groups for related clips. Unknown
grouping must remain unknown, not become a new independent sample per excerpt.
Publisher titles are display metadata, never model inputs.

Inputs are limited to 50 clips per run, 60 seconds and 128 MiB per clip, 512 MiB
per batch and 4096 pixels per source side. Formats follow the existing footage
reader. For this pilot, video and container timestamps must start at zero;
offset recordings fail with a saved reason rather than silently shifting boxes.
An explicit normalized derivative can be supplied separately with its provenance;
the tool never rewrites the original.

## Analyze, review, inspect

```sh
.venv/bin/dvidia-observe analyze /path/to/source.json --models runs/observation-models --output runs/observations-001
.venv/bin/dvidia-observation-studio runs/observations-001 --port 8280
```

Open http://127.0.0.1:8280. Choose an episode, play or scrub, select an observation,
correct its description/timing and approve or reject it. Add a missed event when
needed. Save a draft or explicitly finish the review. Source proposals remain
separate; edits create append-only revisions with stale-tab conflict detection.
Download exports the last saved review. Outcomes default to unknown.

```sh
.venv/bin/dvidia-observe inspect runs/observations-001
```

This verifies the source/protocol/episode hashes and revision chain, then reports
review progress. Without independent references it returns `accuracy: null` and
`accuracy_status: not_assessed`. A review is an assessment by the local operator,
not an authenticated identity or independent accuracy measurement.

The lab listens only on 127.0.0.1, checks the local origin for writes, supports
bounded video byte ranges, and performs no uploads. It serves no arbitrary files
or model URLs. Run it only with bundles you trust; local users can read files in
the same account. Hashes detect accidental changes but are not digital signatures.

## What the observations mean

- YOLOX suggests COCO object labels and image rectangles. Scores are uncalibrated.
- A MediaPipe palm detector suggests palm rectangles, not full hand segmentation.
- Greedy same-class image-overlap association joins nearby detections; it does
  not prove physical identity and splits gaps. No boxes fill an occlusion.
- Hand/object box overlap yields an explicit 2D proximity hypothesis. It cannot
  establish touching, grasping, force, depth, task completion or robot commands.
- Sampling is two frames/second, up to 120 frames and 640 pixels per side. The UI
  displays boxes only within 0.25 seconds of a sampled timestamp. This is a
  recorded evidence viewer, not a smooth live hand tracker.

The run stores copied originals, sampled frames, raw proposals, failures, recipe,
model hashes, source rights and review revisions. Allow scratch space beyond the
input size. Four CPU threads and one inference worker are used; this is not a
hard RAM/container limit. Do not install it on a production database machine
without a separate resource-constrained deployment test.

## Next gates

1. Collect the [planned cup-placement cohort](observation-pilot-plan.md), including
   complete successes, failures, occlusions and recoveries from several setups.
2. Obtain independent human references before assessing labels or timing.
3. Add a separate temporal caption recipe to describe observable actions; compare
   it with detector-only and combined proposals, including correction effort.
4. Add depth/geometry only with explicit scale and validation; then test a
   separately supervised robot-action bridge and closed-loop safety/competence.
5. Integrate queued analysis into the app only after quality, cost, retries,
   permissions and mobile review are validated on consented DVIDIA recordings.

See [model licenses](observation-models.md), [UI contract](observation-ui.md) and
the [actual feasibility report](../benchmarks/observation-20261010/README.md).

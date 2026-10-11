# DVIDIA observation pilot v1

Planned 10 October 2026, before model execution. This is an engineering feasibility
pilot, not a trained robot skill, production deployment, or accuracy benchmark.

## Deliverables

1. Local, bounded video decoding with actual sampled presentation timestamps,
   original/frame hashes, source rights and session provenance.
2. Optional local detector and temporal caption adapters. DVIDIA owns the
   orchestration, association, annotation contract and review workflow; pretrained
   components retain their own licenses. No SpatialLM weights are used.
3. A mobile-friendly video/timeline review with editable proposals, explicit
   outcomes, immutable review revisions and optimistic concurrency.
4. A runnable pilot receipt with time, resource scope, failures and limitations.
   An independent reference-label evaluator must report not assessed when no
   references exist; model agreement is not human review.

## Fixed experiment scope

- First candidate task: move a cup to a marked surface. Available public sample
  footage may instead establish adapter feasibility; record task mismatches.
- Begin with five existing attributed EgoAnnotate excerpts (shirt, pack,
  onion, melon, jeans) and one CoMind excerpt (stir), without selecting by model outputs. These are
  excerpts, not independent complete cup-placement attempts.
- At most six initial clips, 60 seconds per clip, 128 MiB per clip, 120 frames
  per clip, four CPU threads, one inference worker, no concurrent model jobs.
- Decode at most two samples/second, maximum image side 640. Proposals reference
  actual decoded timestamps. No off-device media transfer or paid compute.
- The first run uses object/palm detections and 2D overlap proposals. Temporal
  language captions, if added, receive a separate frozen recipe: at most three
  windows per clip and four frames per window, maximum 96 generated tokens per
  window. Record model/load/inference time separately.
- Download only official model artifacts and registry packages, capped at 2 GiB
  of model weights for this first run. Inference is offline after preparation.
- Model failure remains a failed attempt. Never fabricate detections, timestamps,
  grasp states, successful completion, metric geometry, contact or robot actions.
- Freeze the recipe before the initial run; adjustments produce another recipe
  and result directory. Preserve the original attempt and failures.

## Proposed cup cohort (not collected)

Collect 30–50 complete permission-cleared attempts from several people and
setups, including occluded cups, wrong targets, drops, incomplete attempts and
recoveries. Group by original/session/person/object before a development/test
split. Obtain independent references for object boxes, event boundaries and
observable outcome, blinded to proposals. A marked target and known-size
reference enable later geometry checks; no metric coordinates are implied now.

Compare detector-only proposals, caption-only proposals, their combination and
manual review at matched quality. Report detection precision/recall and IoU,
boundary timing error, missed/extra events, false completion, correction time,
abstention and full cost. Keep model scores separate from measured accuracy.
Physical skill qualification and action learning require separate experiments.

## Gates

The engineering gate is readable real video, hash-bound finite annotations,
successful correction/reload/export, offline execution, explicit unknowns,
resource limits and tests. The accuracy gate remains open until independent
references exist. Commercial release additionally requires review of exact
model/data licenses and evaluation on consented DVIDIA footage. Repeated scans
may later support a 3D view, but v1 supplies no 3D reconstruction.

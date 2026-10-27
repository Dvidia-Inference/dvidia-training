# Learned PiPER placement v0.1

The frozen learned primitive completed **5/12** unseen-start placements. The authored teacher completed **12/12**; open-jaw, hold and shuffled-label controls each completed **0/12**. All 60 attempts were valid and retained. All twelve learned final poses qualify, but seven fail stricter native retention/contact-history checks.

| Variant | Full completion |
| --- | ---: |
| Learned feedforward primitive | 5/12 |
| Authored teacher | 12/12 |
| Learned arm with jaw open | 0/12 |
| Reset arm targets held, jaw open | 0/12 |
| Same-recipe shuffled-label learner | 0/12 |

Thirty-two successful synthetic teacher episodes supply 32,000 causal state/action samples. Training, development and confirmation use 16, 4 and 6 disjoint starts with the same two goals. The selected cubic/Gaussian primitive trained in 1.42 s and occupies 2,031,879 bytes. Two feedback MLPs and a lower-resolution primitive each scored 0/8 on development; the selected candidate scored 1/8 before it was frozen. Every failed candidate remains in the complete bundle.

This is time-conditioned joint/jaw trajectory distillation for a 40 g rigid block in a 30 × 30 mm start region. It uses reset XY, known goal and elapsed time, with live joints only supplying generic command limiting. No teacher, IK, task phases or tape lookup run at inference. Native servos remain authored. Contact recovery, visual action learning, physical calibration and transfer to other arms are open.

The independent audit recomputed every fixed-goal score and checked all tape, model, source and schedule bindings. Its [per-case review](independent-review.md) explains the seven failures. CPU inference measured 0.148 ms median / 1.039 ms p95 across 12,000 calls; 344 complete control steps exceeded 20 ms. These are compute measurements, excluding sensing, communication and physical actuation; they are not hard real-time qualification.

The original frozen driver pauses after saving a tape because of a tuple/list identity comparison. A separately checked bounded adapter resumes through the unchanged driver, scores persisted tapes and advances without rerunning completed physics. All 60 recovery records are in the full bundle; [the recovery receipt](driver-recovery.json) is separate from outcome counts.

- [Run and reproduce locally](../../docs/piper-learned-placement.md)
- [Research paper](https://research.dvidia.org/papers/learned-piper-placement-v01/)
- [Recorded 3D replay](https://dvidia-robot-simulation-lab.static.hf.space/learned-placement/v01/viewer/)
- [Model](https://huggingface.co/Dvidia/piper-block-placement-v01) · [Dataset and full traces](https://huggingface.co/datasets/Dvidia/piper-block-placement-demonstrations-v01)
- [Wheel, model and full sealed evidence bundle](https://github.com/Dvidia-Inference/dvidia-training/releases/tag/piper-imitation-v0.1)

This directory keeps the full report, audit, protocol, frozen hashes and complete-bundle integrity inventory. Compressed demonstrations, all development and confirmation tapes, recorded geometry, model weights and source snapshots are distributed in the release bundle and Hugging Face. [Release artifact hashes](release-artifacts.json) bind the downloads. Original code, generated data and model weights use MIT; upstream notices remain in the bundle. The research site's writing uses its own published license.

The public export excludes two generated Python bytecode caches. Its integrity manifest records the exclusion and original seal hash; every native tape, model and plaintext source remains byte-identical.

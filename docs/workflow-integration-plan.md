# Local workflow integration plan

**Status: proposed.** DVIDIA already has local video import and inspection, footage intake and preparation, CPU visual-model training, simulation profiles, separate video-goal and training studios, and separate PiPER learned-grip research commands. The proposed work connects these pieces with a small shared run record; it does not introduce a general workflow engine.

The existing video-goal path verifies a video and builds a destination candidate around an authored controller. It does **not** train a PiPER motor policy from arbitrary uploaded video. Footage training can fit a visual predictor and, when action supervision exists, a movement head. The learned PiPER study consumes preserved synthetic robot-teacher data through its own workflow. Keep these inputs and outputs distinct in user-facing language.

## Proposed integrations

1. **Shared manifest and preflight.** Add a versioned local manifest that references existing source packages and records the task, split, recipe, benchmark, and intended output type. A read-only preflight should reuse existing validators and say which supported stages can run, what is missing, and whether an action-supervised or PiPER route is actually available. Preflight must not download assets, start training, launch confirmation, or imply that missing robot actions can be inferred from video alone.

   A compact proposed schema could look like this:

   ```yaml
   schema: dvidia.run/v1
   run_id: lift-video-001
   input:
     package: ./imports/session.zip
     sha256: <content-hash>
     split_manifest: ./prepared/splits.json
   workflow: footage-visual
   task: block-placement
   recipe: cpu-visual-baseline
   benchmark: heldout-prediction
   output: visual-model-candidate
   ```

   This is an example proposal, not a current supported file format or recipe registry. The plan must resolve every referenced input to its content hash and bind applicable robot/profile, assets, calibration and runtime identities. It should reference existing intake and split records rather than replace them. A visual prediction benchmark cannot qualify a manipulation skill.

2. **One durable run record for UI and CLI.** Proposed UI and Terminal commands should read and update the same local record: immutable plan digest, ordered stages, stage inputs and outputs, attempt IDs, and terminal results. Persist updates atomically so the UI can recover status after restart. A future `status --json` and `resume` can reuse verified completed stages only when their inputs still match; changed video, split, calibration, assets, model recipe, or runtime must invalidate dependent outputs. If a confirmation attempt was interrupted, retain it and require an explicit new attempt; never silently retry confirmation. Start with simple local files and one CPU worker.

3. **Evidence-gated install output.** Present a video-goal candidate, visual model, movement candidate, and evaluated simulation skill as different artifact types. Route existing compatible capsules through their current qualification gate. Add a separately versioned PiPER adapter if its format needs installation; do not equate its model JSON with a qualified capsule. An install receipt should bind the exact model, task/profile, runtime and assets, evaluation protocol, and all trial outcomes. A successful process exit alone must not grant qualification.

## Boundaries and sequence

Keep the default workflow local and CPU-based, with no cloud account or GPU required for the existing supported path after required dependencies and assets are available. Optional heavier backends can be added later behind the same data and evaluation contracts. The workflow additions do not make the existing video-goal path train PiPER from arbitrary footage.

Preserve the current PiPER research result: v02 records strict success **6/24** versus blind **8/24**, with contact and timing unresolved. It remains a research candidate; workflow integration adds reproducibility and recovery, not evidence of better learning. Continue comparing trained and blind policies on the same finite held-out cases, with failure categories and reviewable traces.

Implement in this order: manifest/preflight, shared durable run record, then evidence-gated artifact handoff. Treat the schema and commands above as proposals until implemented and documented.

## Source notes

- [DVIDIA local pipeline and skill workflow](pipeline.md), [local Skill Studio handoff](skillspace-local-handoff.md), and [Skillspace format](skillspace-format.md) describe current paths.
- [Nebius Physical AI Workbench repository](https://github.com/nebius/nebius-physical-ai) documents a cloud control plane, workflow artifacts, run identity, and recovery patterns; its [workflow guide](https://github.com/nebius/nebius-physical-ai/blob/main/docs/workbench/npa-workflow-guide.md) is a reference for explicit stage contracts, not a dependency for DVIDIA.
- [Nebius Workbench announcement](https://nebius.com/blog/posts/physical-ai-workbench) reports customer experience but does not provide an independent benchmark for its engineering-time claim.

# PiPER alternate-grasp development — 2026-10-11

This is development evidence, including every failed candidate. It is not fresh confirmation, physical-robot validation, learned grasp selection or a force-calibrated model. The fixed candidate uses privileged native poses/contact observations and the existing authored controller. Both source cohorts were already inspected before these experiments.

The selected experimental candidate requests a **−45° world yaw for the downward grasp pose**, leaves the original approach/carry yaw, and uses a **0.10 rad orientation-error clamp only for the requested grasp quaternion** in bounded differential IK. It preserves the existing 15 mm translation-error bound, 0.06 rad joint-delta bound, servo gains, control/force limits, target rates, contact geometry/friction, gravity, task scorer and complete 20-second episode budget. It does not select a grasp using mass, friction, outcome labels or future observations. Changing IK request limits is a planner change, not a native actuator-gain change.

`piper_grasp_strategy.GraspPlanningEnvironment` is a read-through policy view: original environment configuration/model/state remain unchanged. The combined `GraspReacquisitionPiperTaskPolicy` uses this view and the v0.5 second-probe controller. Every compared raw world/config/model/start field was identical to the respective v0.4 baseline. **No second probe triggered in the selected grasp candidate's 24 final development scenes**, so these particular gains do not demonstrate the second-probe contribution.

## All rounds retained

| Round | Fixed candidates / scope | Attempts | Successes | Attempts with prohibited contact |
|---|---|---:|---:|---:|
| r0 | Original IK, fixed 0/+45/−45° yaw for both tool and carry | 12 | 2 | 0 |
| r1 | Original IK, tool-only ±45° yaw | 8 | 0 | 0 |
| r2 | Tool-only 0/−45° yaw, larger 0.20 rad orientation clamp for all poses | 8 | 0 | 8 |
| r3 | Tool-only 0/−45° yaw, 0.20 rad clamp only for grasp target | 8 | 0 | 6 |
| r4 | Tool-only −30/−45° yaw, 0.10 rad grasp-target clamp | 8 | 5 | 0 |
| r5 | −45° yaw retained through postapproach carry, 0.05/0.10 rad grasp-target clamp | 8 | 2 | 0 |
| r6 | Selected −45°/0.10/original-carry candidate, all 12 published 941827 scenes | 12 | 7 | 0 |
| r7 | Parameters frozen, all 12 historical 813407 scenes | 12 | 8 | 0 |

**76 native attempts; 76 valid integrations; 24 successful placements.** Valid integration does not imply safe motion or task success: aggressive discarded candidates produced prohibited contacts and remained failures. All 76 raw attempts, per-round startup source snapshots, plans, and source hashes are retained. No warning was recorded. r1's plan declared two candidates and four scenes before execution but retained an incorrect `planned_attempts:12` formula; its separate correction explains the actual eight-attempt grid without editing original evidence or omitting attempts.

## Selected candidate versus published v0.4

| Known cohort | v0.4 successes | Candidate successes | Gains | Losses | Common-success mean completion change |
|---|---:|---:|---:|---:|---:|
| 941827 | 4/12 | 7/12 | 3 | 0 | +0.841 s (slower) |
| 813407 | 6/12 | 8/12 | 2 | 0 | +0.7335 s (slower) |

The selected candidate preserves every baseline success across these two known cohorts. It completes all nominal/high-friction scenes and three of four moderate-friction/heavy scenes. One moderate-friction/heavy scene still fails carry reassessment, and all eight low-friction scenes fail acquisition and retain the closed refusal. There are no prohibited native contacts or solver warnings in these final 24 scenes. The wider diagonal grip supplies more observed pad normal force from the unchanged position servo; it does not change force caps or establish physical gripper capacity.

A diagonal grasp alone was insufficient. The original IK took too long to reach that pose; larger orientation corrections regressed approach or placement. The narrower 0.10 rad grasp-target change improved acquisition while preserving the native constraints. Retaining diagonal yaw during carry did not improve the difficult pair and was not selected.

No parameter tuning was performed after the r7 historical regression. Remaining evidence must come from a separately frozen, multi-seed fresh protocol with matched open-jaw controls and the unchanged scorer. Do not pool these selected/known scenes with fresh confirmation or call them unseen.

## Reproduction and audit

Each round includes `run.py`, `plan.json`, `summary.json`, a `sources/` snapshot and every complete `*.json.gz` episode. Run `PYTHONPATH=src .venv/bin/python runs/piper-v05-development/grasp/analyze.py` from the repository to regenerate `analysis.json` against the published baseline receipts. Source snapshots preserve the original experimental versions; the current selected source intentionally differs from discarded candidate versions. The analyzer found zero startup-snapshot/raw-source hash mismatches.

The final source tests cover strict input validation, normalized yaw composition, unchanged native/config/state, original-IK equivalence at the old clamp, original approach IK, native action bounds, phase-aware optional carry yaw, and combined-controller contracts. Seven tests passed, including two native tests using the pinned asset bundle.

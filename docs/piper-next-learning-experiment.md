# Next experiment: learn corrective actions from demonstrations

**Proposed and unrun.** The [native action capture preview](piper-action-capture.md) now preserves actual rendered RGB-D, measured joint/jaw state, and issued/applied actuator references. It has no implemented object tracker or operator corrections; its repeated authored trajectory cannot qualify as corrective training data. The urgent steps are validating a frozen camera estimator, collecting reviewed corrective segments, and validating their replay evidence before training. A video without synchronized action evidence does not supply these motor labels.

The [proposed RGB-D perception gate](piper-perception-gate-v01.md) specifies the
causal estimator inputs, abstention, uncertainty and separate held-out evaluation
needed before these recordings can supply visual action-learning inputs.

## Why change actions instead of adding another waiting rule

In [v0.3](../benchmarks/piper-grip-clock-v03/report-summary.json), fixed progression achieved 72/128 strict placements versus 44/128 for the learned clock. In the different [v0.4 cohort](../benchmarks/piper-grip-wait-v04/recovery-20261030/report.json), bounded waiting achieved 42/128 scheduled successes, learned timing 32/128, and fixed progression 44/128; both primary comparisons were inconclusive. Four interrupted outcomes remained unknown and were failure-imputed only for the scheduled analysis.

A plausible hypothesis is that stopping early can block the centering motion that creates a grasp. Bounded eligibility permits that motion, but cannot change a poor grasp or carrying path. Its 31 retention failures and 52 critical-contact attempts suggest unresolved manipulation problems; they do not prove their causes. Unequal pulse exposure also prevents attributing differences to recovery quality. Test whether demonstrated corrective actions improve contact and retention, rather than teaching another clock to wait longer.

## Candidate and evidence boundary

Train one small goal-conditioned residual action policy: a two-layer, 64-unit MLP predicting six joint-reference corrections and one correction in the native jaw coordinate every 20 ms. Bound corrections to ±0.03 rad per joint and ±2 mm in the jaw coordinate before the existing actuator limits and rate caps. Operator demonstrations must remain inside this envelope. The candidate changes motor actions; it does not add a hand-authored grasp/recovery phase machine or silently invoke a teacher.

Primary inputs are the current and preceding 160 ms of joint position/velocity, measured jaw position/velocity, previous applied commands, reference progress, goal, and object/tool-relative motion estimated by a frozen calibrated rendered RGB-D tracker. Include observation age and uncertainty. These are causal measurements or estimates; no future frames, future success, or simulator object/contact truth may enter the primary policy. Do not assume a real PiPER supplies tactile force sensing.

Privileged native pose, contact forces and complete outcomes may label demonstrations and score tests offline. A separately labeled oracle-pose ablation can diagnose perception limits, but cannot replace the primary result. All proposed camera and actuator signals remain simulated until measured on selected hardware.

## Fixed data budget and held-out groups

Register 72 fresh XY groups before collection: seed 304031, uniform within the existing 30 mm square, at least 0.75 mm from every historical v0.1–v0.4 start and each other. Assign the first 32 groups to training, next eight to development, and last 32 to confirmation. Keep every frame, condition, correction and retry from a group in its split.

Collect 128 training demonstration episodes: 32 groups × two known goals × nominal/pulse conditions. Collect another 32 development demonstrations, excluded from fitting. Each lasts at most 20 seconds. An operator demonstrates approach corrections, test lift, carrying and controlled regrasp; failed trials remain retained. Fit action targets from successful operator-controlled segments, explicitly reporting usable coverage and excluded failure segments. Never replace failures with unlimited extra demonstrations. If corrected successful examples do not cover every training start/goal pair, stop at the data-quality result.

Fix architecture, optimizer and 50-epoch budget before development. Select checkpoints using development action loss, then run 32 additional development rollouts to freeze all acceptance/uncertainty thresholds. No confirmation tuning or automatic extra collection.

## Two separate evaluations

Run 512 end-to-end confirmation attempts: 32 groups × two goals × two assigned conditions × four controllers. Compare the learned adapter, an equally sized feedback-blind adapter, frozen fixed progression, and v0.4 bounded waiting. The blind adapter retains initial scene/goal and reference progress but masks subsequent sensor feedback during fitting and inference. Use identical starts, seeds, limits and observation estimates.

Separately schedule 64 common test-lift checkpoints, one per confirmation start/goal pair. Generate each with a frozen common routine and verify the existing native bilateral/lift dwell criteria. Retain failed checkpoint attempts; never substitute easier starts. From each qualifying checkpoint, fork all four controllers with identical state, causal observation history and nominal/pulse assignments. Report eligibility K/64 and all 8K follow-on outcomes separately. This conditions the retention comparison on the same successful lift; it does not replace end-to-end acquisition measurement. The complete proposal permits at most 1,280 episodes, including demonstrations, development checks, checkpoint probes and forks.

## Decisions, failures and limits

Keep the independent strict scorer: grasp, sustained lift, retention during carry, release, settled placement and retreat, with unintended contacts checked. Report acquisition failure, retention loss, critical contact, aborts, intervention assigned/eligible/effective counts, and terminal-only false completion. Missing-object and no-action verifier controls must remain unsuccessful. Measure perception age, full-step latency, deadline misses, memory and training time on the named shared host.

Predeclare learned-minus-fixed and learned-minus-blind comparisons using whole-XY paired intervals and the existing two-comparison correction. Claim an improvement only when supported by the interval; report contact tradeoffs regardless of placement rate. Refuse missing calibration/channels, nonfinite observations, sensor age above 100 ms, and unsupported profiles. Enforce the 20-second deadline and at most one learned regrasp; aborted attempts remain failures, without hidden fallback.

This is a proposed simulator action-learning study for one rigid block and one embodiment. It does not establish human-video action recovery, shoe handling, arbitrary-arm installation, physical safety, or universal one-shot learning.

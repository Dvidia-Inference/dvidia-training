# Frozen sample-count pilot — 8 October 2026

This offline CPU experiment fits and installs three movement candidates from
the public synthetic native Skillspace, then evaluates them on twelve new
object/target layouts. The 2/4/7 conditions use nested whole recording groups;
the two development recordings and one test recording remain fixed. All models
are frozen before any benchmark physics is evaluated. This is one composition
order, not a law relating video count to acquired skill.

| Training groups | Aligned training actions | Nominal qualification | Actuator-stress qualification | Nominal mean target error, including failures | Movement test MSE, rad² |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 1,135 | 8/8 | 3/4 | 0.677 mm | 1.120 × 10⁻⁶ |
| 4 | 2,281 | 7/8 | 3/4 | 31.641 mm | 1.109 × 10⁻⁶ |
| 7 | 4,029 | 8/8 | 3/4 | 0.267 mm | 1.325 × 10⁻⁶ |

The authored teacher completes 8/8 nominal and 3/4 stress scenes. Every one of
the 36 matched open-jaw controls fails; there are no execution errors. All four
conditions fail stress scene 11. The four-group candidate additionally fails
nominal scene 04, which its teacher and the other two candidates complete.
Failures remain in all planned denominators and distance summaries.

The seven-group candidate's nominal maximum target error is 0.483 mm. Its eight
successful nominal episodes simulate 106.263 seconds in 7.813 seconds of native
episode wall time, or **13.60 simulated seconds per wall second**. The complete
benchmark takes **102.727 seconds** on an Apple M5 host with 24 GiB RAM and no
GPU use, including input copies, fitting, installation, 84 native episodes and
artifact output. Process startup, top-level imports and final report writing
are excluded. This is a one-host timing observation, not a minimum hardware
requirement, speedup over another implementation, or physical-accuracy result.

Pipeline timers for the 2/4/7 fits are 0.592/0.840/1.252 seconds, with their
narrower scope recorded in the report. The movement test has one recording and
621 correlated labels. Its prediction error does not rank closed-loop success:
the four-group candidate has the lowest test MSE yet an additional task failure.
More recordings did not monotonically improve execution in this pilot.

## Inspect and reproduce

- [Exact report](report.json), [frozen protocol](protocol.json),
  [protocol hash](protocol.sha256), and [frozen model identities](models-frozen.json).
- Installed-candidate payloads: [2 groups](groups-002.skill-capsule.json),
  [4 groups](groups-004.skill-capsule.json), [7 groups](groups-007.skill-capsule.json).
  These are simulation candidates, bound to the existing runtime and adapter.
  Downloading one does not authorize a physical arm or an untested scene.
- [Full evidence download](https://github.com/Dvidia-Inference/dvidia-training/releases/tag/benchmark-2026-10-08)
  includes native traces, controls, qualification receipts and replays.
  [Archive size and SHA256](archive.json) and the
  [individual file manifest](evidence-manifest.json) identify the exact bytes.
- [Reproduction commands and method](../../docs/benchmarks.md) and
  [Grok 4.7 method review](../method-review-2026-10-08.md).

Source data came from an anonymous, hash-verified download of the public
[DVIDIA dataset](https://huggingface.co/datasets/Dvidia/dvidia-training-examples/tree/d351a32923d54e3166e8ff780fd827c80174a903).
Training and evaluation then ran with network access restricted. The source
archive contains ten synthetic native recordings, totaling 116.5 seconds, with
aligned numerical simulator labels. These are not five-second human shoe clips.
Repeating this seed is a repeat of observed scenes; subsequent learner changes
need a fresh, frozen evaluation suite.

## What this establishes

The product can now measure whether a fitted, portable candidate survives
installation and new-layout simulation qualification, preserving both successful
and failed cases. Exact layouts from all ten source recordings are excluded
relative to their hash-bound native declarations. That checks consistency, not
independent capture history or out-of-distribution transfer.

The RGB predictor does not control the arm. Movement uses privileged simulator
state and an authored phase/waypoint/gripper/recovery scaffold. No raw
human-video action bridge, shoe skill, sufficient demonstration count, contact
accuracy on hardware, or physical robot qualification follows from these results.

The next optimization targets are measurable: investigate the four-group
closed-loop failure, distinguish movement-head behavior from the shared actuator
stress limit, then test any revised learner across new composition orders and
fresh layouts. Human footage also needs a validated observation-to-action bridge
for a defined arm and gripper before it can enter this movement benchmark.

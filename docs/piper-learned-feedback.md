# Learned PiPER feedback v0.2

This work tests whether present and recent grip measurements help a simulated arm finish a rigid-block placement when its gripper is briefly interrupted. The current prototype learns the jaw command while preserving every six-axis arm command from the frozen v0.1 movement model. Earlier candidates that learned all seven command residuals remain separate evidence under their original protocol. This remains research using synthetic authored-expert labels and privileged simulator measurements.

The frozen 120-attempt confirmation completed **6/24** with live feedback, **8/24** with the matched blind learner, **5/24** with the original movement primitive, **24/24** with the authored teacher and **0/24** with an open jaw. All attempts were valid. The feedback learner completed **0/12** interrupted cases. This experiment does not demonstrate a benefit from live grip features.

[Research and recorded replay](https://research.dvidia.org/papers/learned-piper-grip-v02/) · [Model](https://huggingface.co/Dvidia/piper-block-placement-feedback-v02) · [Complete evidence](https://huggingface.co/datasets/Dvidia/piper-block-placement-feedback-demonstrations-v02) · [GitHub release](https://github.com/Dvidia-Inference/dvidia-training/releases/tag/piper-feedback-v0.2).

A successful local installation does not establish a robust learned skill, human-video learning, camera perception, physical robot transfer, or general slip recovery.

## Run a downloaded skill locally

Download the wheel from the [GitHub release](https://github.com/Dvidia-Inference/dvidia-training/releases/tag/piper-feedback-v0.2), then install it from the directory containing the download:

```sh
python -m pip install './dvidia_training-0.1.0-py3-none-any.whl[arm]'
```

The smaller installation kit on that release also contains the models and a complete command example. If using a source checkout instead, run this from the repository root:

```sh
python -m pip install '.[arm]'
```

Fetch the pinned PiPER assets once. This is the only download needed by the simulation asset command:

```sh
dvidia-robot-sim fetch piper --output ./robot-assets
```

Download two finite data-only JSON files from the model repository: `base-model.json` and `candidates/jaw4/model.json`, saving the latter as `jaw-model.json`. For reproducible use, select the exact Hub commit recorded in the publication receipt rather than a moving branch.

```sh
dvidia-piper-feedback-skill \
  --base-model ./base-model.json \
  --model ./jaw-model.json \
  --assets ./robot-assets \
  --start-x 0.286 --start-y -0.031 --goal 0 \
  --output ./my-first-feedback-attempt
```

The runner identifies the residual or jaw-only architecture and full/blind sensing mode from the strictly validated model document. Its manifest records the architecture and the corresponding runtime hashes. It creates a fresh output directory, copies the exact model bytes, runs one native 20-second attempt, and preserves the episode, dense numeric native telemetry, independent score and SHA-256 receipts. It never overwrites an earlier attempt. The normal installed run adds no artificial faults and collects no expert labels.

Exit status is `0` for strict independently reconstructed task success, `1` for a valid recorded failure, and `2` for invalid input or an invalid recorded attempt. A failure is evidence to inspect, not a reason to silently rerun until success.

## Bounded task and inputs

The current task uses the pinned six-axis PiPER arm, coupled jaw and a rigid 40 g block. Starting X is restricted to `0.284–0.314 m`, and Y to `−0.045–−0.015 m`. Goal `0` is `(0.35, 0.08) m`; goal `1` is `(0.40, −0.08) m`. These are trained destinations. Other objects, masses, surfaces, goals and physical hardware require separate evaluation.

The frozen base predicts movement from reset object XY, the known goal and the physical episode clock. The current adapter returns its original six arm targets unchanged. A small NumPy classification tree chooses an absolute jaw target of `0` or `0.035 m`. Its 56-feature contract contains fixed context plus current, 40 ms-old and 120 ms-old grip snapshots: object position relative to the tool, destination relative to the object, object velocity, measured jaw positions, pad forces and contact counts. There are no joint, joint-velocity, absolute tool-pose or actuator-control features. These inputs are privileged simulator values. The complete ordered normalization contract is `FEATURE_NAMES` in `piper_grip_adapter.py`.

At inference the student receives no teacher, IK, task phase, intervention ID, success flag or future frame. The original generic joint caps and native servo/rate limits still apply. The matched blind tree uses the same feature dimensions, tree training recipe, teacher episodes and sample budget, with all live measurement/history slots zeroed. Its fixed context still includes the original model's jaw reference and episode clock.

The earlier seven-command residual family uses a different 183-feature contract in `piper_feedback_model.py`. Those models and failed development attempts are retained for comparison. The installed manifest distinguishes `learned_seven_command_residual` from `learned_absolute_jaw_over_frozen_six_joint_path`.

## Refit the jaw prototype from preserved evidence

The jaw family has its own prepared study folder and protocol. It imports all 64 teacher episodes from its verified parent evidence without recollecting physics. All 64 of these teacher attempts succeeded strictly; the collection contract also preserves finite unsuccessful prefixes when they occur. The parent's mixed-policy data and failed development candidates are retained for transparency and excluded from jaw-only fitting.

With that prepared lineage-bound folder available, fit the selected tree recipe and its matched blind control into fresh candidate names. A completed frozen study refuses retraining; preserve it and create a separate development study:

```sh
dvidia-piper-learned-grip prepare ./jaw-study \
  --parent ./evidence/prior-studies/study01
python -m dvidia_training.piper_learned_grip_benchmark train ./jaw-study \
  --name jaw4 --max-depth 4 --min-leaf 64 --max-bins 31
python -m dvidia_training.piper_learned_grip_benchmark train ./jaw-study \
  --name blind4 --blind --max-depth 4 --min-leaf 64 --max-bins 31
```

These commands produce candidate model JSON; they do not certify a skill. Development evaluation, model selection, source/protocol freeze and fresh confirmation remain separate steps. Starting another architecture creates another study lineage, preserving prior receipts.

## Evaluation and evidence

All starting positions are new relative to v0.1 and are split by whole starting position across training, development and confirmation. Both known destinations and both nominal/interruption conditions remain together within a split. Development positions were reused adaptively while investigating the earlier residual family; they are tuning evidence. Confirmation begins only after the selected jaw/blind models, protocol and runtime hashes are frozen.

The primary endpoint remains the original strict native task success, including lift, retained carry, supported placement, release and settling. Terminal geometry and restored contact are reported separately. Restoring loaded contact for 100 ms does not by itself prove task recovery or airborne regrasp. Every scheduled attempt and every unsuccessful expert prefix is retained.

The selected 25-node tree uses all 64,000 teacher labels from 64 episodes; five unsuccessful residual fits, 32 mixed-policy episodes and all development candidates remain in the dataset. An independent numeric audit checks all 360 attempts and reports zero native-metric mismatches. It verifies evidence consistency, not external physical fidelity.

Feedback inference p95 was **0.501 ms** and the full control step p95 was **5.648 ms**, including dense native telemetry. **34/24,000** steps exceeded 20 ms. Camera capture, communication and real actuator latency are excluded; this is not a hard real-time guarantee. No GPU is required by this numeric policy or its CPU simulation. CPU and memory minimums have not been established by a hardware sweep.

Fourteen feedback trials reached a settled destination, but only six passed the full manipulation endpoint. Finger-table contacts occurred in twelve feedback trials. Ten of twelve nominally interrupted feedback trials actually received a positive applied jaw-target disturbance, all before established loaded grip. These data principally test acquisition disturbance, not post-grasp recovery. The next study must couple grip decisions to clearance-aware movement timing and verify disturbances after an established grip.

# First-attempt PiPER placement after a camera survey

Can a small, locally installed skill place a block on its first attempt after the robot has had time to learn the workspace? We surveyed the camera once with a removable marker fixture, then located each new block with one RGB-D frame. The robot received a known destination, measured joint positions and its gripper encoders. The controller issued arm and jaw targets at 50 Hz for up to 20 simulated seconds. Native physics independently checked grasp, lift, retention, release, settling and retreat every millisecond.

[Try the learned skill or watch a placement](local-skill/README.md) · [Supplementary learned-model evaluation](supplemental-learned/) · [Camera survey](workspace/README.md)

## What the downloadable model learned

An unchanged 195 kB model trained on 32 successful synthetic demonstrations completed **36/36 first-attempt placements** on the 18 new starting positions. Both goals passed at every position, including every 60 g, lower-friction case. Median completion was **14.331 simulated seconds** and median final XY error was **4.224 mm**. Peak force on either pad was 1.172 N, with zero critical-contact episodes and zero teacher calls. Independent headless replays reproduced all 36 native outcomes. This separately registered test used the same unseen positions as the primary comparison. The demonstrations supplied the motor targets. The package also contains an authored wrist-angle adapter and encoder-based command bounds. [Learned-model receipts and replay](supplemental-learned/README.md)

## Frozen comparison

We registered 18 new starting positions and two destinations per position. Each controller received the same camera images, cube mass, friction, geometry, robot and time budget in each pair. The operating envelope spans 24 mm per horizontal axis and uses a known upright orange cube. No controller received material labels, object truth, fresh object tracking during motion, teacher intervention or a second attempt. The baseline is the earlier learned motor primitive. The new teacher is an authored diagonal grasp with a gradual test lift. A matched controller omits its jaw checks.

| Frozen controller | First-attempt placements | Episodes with critical contact | Median successful placement | Median final XY error on successes |
|---|---:|---:|---:|---:|
| Earlier learned primitive | 8/36 | 22 | 15.54 s | 10.75 mm |
| Authored diagonal grasp and gradual lift | 36/36 | 0 | 14.33 s | 4.14 mm |
| Same trajectory, timed jaw control | 36/36 | 0 | 14.33 s | 4.14 mm |

The improvement over the earlier primitive was 77.8 percentage points. A paired bootstrap over the 18 starting positions gives a descriptive 95% interval of 61.1 to 91.7 points. The timed controller issued identical commands in all 36 pairs, so these ordinary placements provide no evidence that jaw monitoring caused the gain. The heavier, lower-friction condition improved from 0/12 to 12/12. Each material condition combines a mass and a friction coefficient.

The full 12-case development comparison with the old primitive was completed retrospectively after selection. It scored 4/12 versus 12/12 for the selected candidate. The controller freeze preceded confirmation. Missing initial perception refused before movement, and forced-open actuation failed the native grasp check. A headless reconstruction of all 36 old-primitive actuator tapes reproduced every score, as did the sampled corner-controller tape.

[Registration](plan.json) · [Selection record](selection.md) · [All eight development iterations](development-history.json) · [Audited confirmation metrics](results.json) · [Native receipts and actuator tapes](confirmation-evidence.zip)

A separate development fault experiment opened the jaw during transport. Checking aperture cut median tool travel after confirmed loss from 107 mm to 14 mm. All six placements still failed. Recovery after the early warning remains an open experiment. [Fault experiment](grip-loss-diagnostic/results.json)

## Reproduce the comparison

Use the training library with the arm dependencies and offscreen MuJoCo graphics. From the checkout, fetch the existing PiPER assets once. The earlier baseline weights and their pinned SHA-256 are available in the [arm install guide](../../docs/arm-interfaces-v01.md).

```sh
dvidia-robot-sim fetch piper --output robot-assets
study=benchmarks/piper-first-attempt-v01
PYTHONPATH=src python -m dvidia_training.piper_first_attempt_eval \
  --cases "$study/confirmation.json" --controller corner-v8 \
  --assets robot-assets --calibration "$study/workspace/nominal.json" \
  --output runs/my-first-confirmation
```

For the timed comparison use controller `corner-v8-timed`. For the learned baseline use `fixed --fixed-model model.json`. Each evaluation requires a fresh output. To audit the published receipts and recompute grouped differences, use the following commands from this directory.

```sh
unzip -q confirmation-evidence.zip -d evidence
python summarize.py evidence
```

This study used a local arm64 Mac. The native renderer reported limited depth precision. Wall times include local computation and concurrent simulation workers. Simulated placement time measures the physical trajectory as scored. Physical sensing, actuator response, deformable objects and other arms require new measurements.

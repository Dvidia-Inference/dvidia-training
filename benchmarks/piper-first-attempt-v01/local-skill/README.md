# PiPER placement from a camera observation

A 195 kB CPU model learned six joint targets and one gripper target from 32 simulated demonstrations. At the beginning of each placement, a surveyed camera locates a known upright 40 mm block. The skill uses its position, the requested destination and elapsed time to reconstruct the motion. An authored wrist adapter uses the measured block angle to choose a diagonal grasp. Measured joints bound each command. In 24 development trials with turned blocks the adapted skill placed all 24, compared with 16 using the same weights and a fixed wrist.

**Status: local simulation research candidate.** The demonstrations came from an authored simulator controller. The starting positions in these development trials were excluded from model fitting and had been used to develop that controller. This report covers one simulated PiPER, one upright orange cube, a known support plane, a calibrated camera and two goals. Ordinary human video, arbitrary objects or arms, physical transfer and an independent qualification test remain open.

[Watch one placement at double speed](learned-piper-placement.mp4) · [Download the Terminal skill](piper-camera-yaw-skill-0.1.0.zip) · [Training records and base model](piper-camera-xy-distillation-v1.zip) · [Benchmark evidence](evidence.zip)

## Try the skill in Terminal

From this directory, extract the skill into a new folder. Use Python 3.12 with NumPy 2.5.3. These commands run offline on a CPU once those dependencies are installed.

```sh
unzip -q piper-camera-yaw-skill-0.1.0.zip -d piper-skill
python -I piper-skill/run_yaw.py --self-test
python -I piper-skill/run_yaw.py < piper-skill/example-observations.jsonl
```

The self-test compares 24,000 seven-coordinate commands against saved observations. The example emits simulation targets in radians and metres. The archive includes the model and data cards, license, runtime, observation schema and timestamp requirements.

## Run one physical simulation locally

Use the DVIDIA training source at `e13ff91fe2466cf630fe49c0dd0993dfafdd8e0e` or a matching simulator installation. From a training checkout, install its arm dependencies using `python -m pip install '.[arm]'`. The native run uses MuJoCo 3.15.0 and requires offscreen rendering. Fetch the pinned PiPER assets once, or point the command at an existing asset directory. For the following commands, return to this report directory and retain `simulate.py`, the skill archive and the evidence archive together.

```sh
dvidia-robot-sim fetch piper --output ./robot-assets
python simulate.py --assets ./robot-assets --output ./first-placement
```

The default case uses the heavier 60 g cube at +24°. The runner settles the scene, renders an initial RGB-D pair, runs the measured starting pose through the packaged model, applies commands over 20 simulated seconds and independently scores physical outcomes at 1 ms. It writes a receipt and the actual command tape. To choose another registered scene, use `python simulate.py --list` and pass its ID using `--case`. Each run takes a fresh output directory. The default scene reproduced its original 1,000 commands and achieved strict placement at **14.373 simulated seconds** when run from a separately built wheel outside the checkout. The test reused the installed native dependencies. [Verification receipt](one-run-receipt.json)

## What changed the outcome

| Development experiment | Result | Interpretation |
|---|---|---|
| Fitted camera-position policy versus nearest demonstration | Both 12/12. Median peak pad force 0.855 N fitted versus 6.094 N retrieved. Median terminal XY error 4.28 versus 3.30 mm. | Learning gave gentler contact on all 12 matched cases. Retrieval remained competitive on placement. |
| Frozen model with and without initial camera yaw | 24/24 versus 16/24. The adapter recovered all eight heavier-block cases. Both had zero critical-contact episodes. Maximum pad force rose from 1.533 to 1.816 N. | Choosing grip direction from a single observed pose helped the harder material. The fitted weights were identical. |
| Independent check on an authored, kinematics-ranked grasp | 34/36 versus 8/36 for its fixed grasp, across six additional angles on three other development positions. Two cases lost retention. | Grasp geometry remained important across more orientations. This is a separate controller from the downloadable learned policy. |
| Material limit, fixed authored controller | 26/40 strict placements over 20 mass/friction combinations at two positions. Every failure occurred during lift. | The simple static calculation declared four infeasible trials feasible. |
| Short test lift using encoders | 17/20 correct lift forecasts versus 11/20 using only the pre-close snapshot on held-out material combinations. Both falsely predicted two infeasible lifts. | 1.32 simulated seconds of interaction improved the forecast. A permissive success threshold would still admit failures. |
| Grasp-loss monitoring | 14/14 induced losses detected. Median delay was 423 ms for a broad aperture limit, 125 ms for a learned aperture envelope, 100.5 ms for a 10 Hz camera and 73 ms for the earliest of the last two. | The camera had zero unconfirmed warnings in ten episodes without sustained loss. Combining cues had one. All detections followed loss. |

The visual monitor computes the distance from an identifiable cube surface to the encoder-derived tool centre. It returned an estimate on 852/864 frames. The stricter upright pose tracker returned a complete pose on 244/864. Retention was independently scored from native bilateral contact, sustained height and separation from the table. These monitors were evaluated in replay and have yet to demonstrate recovery. The supplemental contact and material studies have separate denominators. [Sensor outcomes and timing](monitor-summary.json)

## Local performance

On one arm64 Mac with Python 3.12.9 and NumPy 2.5.3, the standalone process took **46.3 ms median from launch to first response**. A preloaded worker answered its first request in **1.36 ms median**. For subsequent requests, synchronous JSONL serialization, inference and response parsing took **0.042 ms p95**. Peak worker memory was **39.1 MiB**. All 48,000 tested commands matched their fixtures. These figures measure the CPU policy process. In the 10 Hz camera experiment, rendering and encoding took **9.38 ms median, 25.84 ms p95** and checking the visible surface took **0.608 ms median**. Physical sensing, communication and actuator response require separate measurements.

The native renderer on this Mac reported reduced depth precision for its OpenGL path. Calibration, pose visibility and transfer should be measured again on a physical camera. A productive next experiment would test whether the earliest reliable warning can trigger a controlled set-down or regrasp that increases strict completions.

## Reproducibility

The evidence archive contains the registered supplemental designs, source overlays, calibration, action tapes, independent simulation receipts, replay checks and timing records. Extracting it into a fresh copy of the training checkout restores the relative study paths. The base-model archive includes the demonstrations, trainer and byte-identical refit check. Archive checksums and sizes are recorded in [the artifact manifest](sha256.json). The demonstration and policy studies ran locally. This package contains simulated targets and exposes no physical robot interface.

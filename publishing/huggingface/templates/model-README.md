---
license: mit
datasets:
  - "{{HF_ORG}}/dvidia-training-examples"
tags:
  - numpy
  - video
  - robotics
  - simulation
  - synthetic
  - dvidia-training
---

![DVIDIA Training](assets/branding/github-header.png)

# DVIDIA Training: synthetic CPU pilot

Small fitted NumPy artifacts from the DVIDIA Training v0.1.0 synthetic pilot. **The moving-disc visual predictor and native simulation movement candidate have separate inputs, training objectives and evidence.** Visual prediction does not supply a robot policy. The native candidate uses numerical action supervision, authored control logic and privileged simulator state.

[Source datasets](https://huggingface.co/datasets/{{HF_ORG}}/dvidia-training-examples) · [Demo](https://huggingface.co/spaces/{{HF_ORG}}/dvidia-training) · [Training software](https://github.com/Dvidia-Inference/dvidia-training) · [Research report](https://research.dvidia.org/papers/skillspace-footage-training/)

## Artifacts and architecture

| Branch | Files | What was fitted |
| --- | --- | --- |
| `visual/` | `training/model.json`, `training/visual_model.npz`, reports, receipts and `training-result.zip` | A training-only randomized PCA representation and affine ridge next-frame predictor for moving-disc videos |
| `native/` | A second visual predictor, `movement/movement_head.json`, movement report, `candidate.skill-capsule.json`, receipts and `training-result.zip` | A normalized radial-basis ridge movement head fitted to aligned native numerical supervision |

Both visual models sample 12 RGB frames per video, resize them to 16 × 16, normalize pixels to [0, 1], and use 8 latent dimensions. PCA and transition weights use training recordings only. Development selects ridge regularization from `[1e-6, 1e-4, 1e-2, 1]`: `1e-6` for the moving-disc model and `1` for the native schematic visual model. The NPZ arrays are finite float32 `mean` (768), `basis` (8 × 768) and `transition` (9 × 8).

The native movement head selects regularization `1e-6` on development labels. It uses the authored arm profile `dvidia-authored-6dof-parallel-jaw-v0` and this feature contract:

```text
q6_object_minus_tcp3_target_minus_tcp3__capped_goal_delta3_upright_orientation_delta3_v1
```

Its context contains six joint positions plus object-minus-tool and target-minus-tool positions. Goal translation and upright orientation errors supply six additional values. Prediction produces bounded joint deltas; the runtime converts them to joint targets within its joint limits. Phase selection, waypoints, gripping, contact handling and recovery remain authored. Control does not use the visual predictor or camera perception.

The candidate capsule binds the model to an exact source/runtime hash. Loading weights, installing a capsule and qualifying a scene are distinct operations. The training receipts retain `robot_skill_acquired: false` and `physical_robot_ready: false`.

## Data and selection

The [dataset card](https://huggingface.co/datasets/{{HF_ORG}}/dvidia-training-examples) documents the authored moving-disc videos and native MuJoCo recordings. Each branch has ten recording groups. Seed `17` produces seven training groups (`example-01`, `02`, `04`, `05`, `07`, `08`, `09`), two development groups (`example-00`, `03`) and one test group (`example-06`). IDs abbreviated here retain the `example-` prefix.

The moving-disc data total 60 seconds; native schematic footage totals 116.5 seconds. Native action fitting uses 4,029 training, 1,154 development and 621 test samples. Labels are simulator state and authored inverse-kinematics targets, with media/sample/evidence hashes and supplier-declared timestamp alignment. They are not actions inferred from human video. The intake receipt labels numerical action contracts unvalidated; the later trainer checks the strict native sidecar contract and declared alignment. Hashes establish declared consistency, not independent provenance attestation.

## Measured evaluation

### Visual prediction

Both rows below are measurements on **one held-out recording and eleven correlated next-frame transitions**. RGB mean squared error uses normalized uncalibrated pixels; lower values indicate better prediction on those image samples.

| Visual model | Learned RGB MSE | Last-frame baseline | Training-mean baseline |
| --- | --- | --- | --- |
| Moving discs | 0.000303879284 | 0.001771413838 | 0.001330730870 |
| Native schematic footage | 0.002342320281 | 0.003554971423 | 0.003859011473 |

Exact values and per-recording results are in [visual report](visual/training/report.json) and [native visual report](native/training/report.json). These metrics do not measure calibrated 3D geometry, grasping, task completion or robot competence. Eleven transitions are not eleven independent trials.

### Offline movement prediction

[Native movement report](native/movement/report.json) scores six joint-delta outputs against numerical labels from **621 correlated samples in one held-out recording**.

| Predictor | Test joint-delta MSE |
| --- | --- |
| Learned movement head | 5.767923533e-7 |
| Zero-delta baseline | 8.172739843e-5 |
| Training-mean delta baseline | 7.842909795e-5 |

This is offline imitation error in squared joint-angle units. It supplies no closed-loop success rate or independent scene-generalization estimate.

### Separate native qualification

The [qualification receipt](native/qualification/qualification.json) records one exact scene corresponding to test recording `example-06`, seed `8006`. The candidate completed placement in **12.952 simulated seconds**, with **0.310932 mm** final target distance. Replaying the identical arm-target sequence with open jaws failed at the **18-second simulated horizon**. [View the measured replay](native/qualification/replay.html).

This local execution supports contact-dependent placement in that single scene under the unchanged success gate. It adds no independent held-out scene, generalization result, comparison against the earlier controller or physical robot evidence. The receipt's status is `validated_simulation_scene`; its verification scope is caller-owned native local execution, without authenticity or hardware attestation.

## Load and inspect

These are custom JSON and numeric NPZ artifacts. Use DVIDIA Training's validated readers; there is no Transformers `AutoModel` or hosted inference integration. Python 3.12+ is required. System FFmpeg/ffprobe is needed to train from videos; loading the saved visual arrays uses NumPy.

```sh
python -m pip install huggingface_hub
python -m pip install https://github.com/Dvidia-Inference/dvidia-training/releases/download/v0.1.0/dvidia_training-0.1.0-py3-none-any.whl
```

```python
from pathlib import Path
from huggingface_hub import snapshot_download
from dvidia_training.footage_train import load_visual_model

snapshot_download(
    repo_id="{{HF_ORG}}/dvidia-training-pilot",
    local_dir="model",
)
metadata, arrays = load_visual_model(Path("model/visual/training/model.json"))
print(metadata["scope"])
print({name: array.shape for name, array in arrays.items()})
```

The reader checks the schema, array hash, dimensions and numeric payload, and loads with `allow_pickle=False`. For a flattened 16 × 16 RGB frame `x` normalized to [0, 1], the prediction calculation is:

```python
import numpy as np

def predict_next_frame(x, arrays):
    pixels = np.asarray(x, dtype=np.float32).reshape(1, 768)
    latent = (pixels - arrays["mean"]) @ arrays["basis"].T
    predicted_latent = np.column_stack([latent, np.ones(len(latent))]) @ arrays["transition"]
    return np.clip(predicted_latent @ arrays["basis"] + arrays["mean"], 0, 1).reshape(16, 16, 3)
```

For the saved native head, install the optional engine and load its validated numerical contract:

```sh
python -m pip install mujoco==3.15.0
```

```python
import json
from dvidia_training.arm_distill import MovementHead

head = MovementHead(json.loads(Path("model/native/movement/movement_head.json").read_text()))
print(head.model["scope"], head.model["arm_profile"])
```

`head.predict(observation, goal)` requires the exact simulator observation fields and authored arm contract; it is not an API for an arbitrary robot. The supplied `loader.py` exposes `load_visual(root, lane="visual")`, `predict_next_rgb(root, normalized_rgb_16x16, lane="visual")` and `load_movement(root)`. The movement helper returns a validated `MovementHead`; it does not provide a hardware driver. Verify the complete result exports with:

```sh
dvidia-train inspect model/visual
dvidia-train inspect model/native
```

These inspections check artifact integrity, not the truth of measurements or a new scene's execution. Retraining instructions are in the [dataset card](https://huggingface.co/datasets/{{HF_ORG}}/dvidia-training-examples); the [native pilot procedure](https://github.com/Dvidia-Inference/dvidia-training/blob/v0.1.0/benchmarks/native_pilot.py) documents the separate qualification workflow. Existing capsules require their exact runtime; retraining against different runtime files creates a different candidate.

## Compute, provenance and license

[Mac M5 benchmark](benchmarks/macos-m5.json) records a median **0.510619 seconds** across three complete tiny visual-pipeline runs, on an Apple M5 host with 24 GiB RAM, Python 3.12.9, NumPy 2.5.3 and FFmpeg 9.0.1. Wall time includes intake, fitting and export; dependency installation/imports are excluded and integrity inspection is timed separately. Python-process lifetime peak RSS is **50,462,720 bytes**, excluding FFmpeg/ffprobe children. These are measurements on one host, not minimum hardware requirements, hosted Space performance or independent task evidence.

Artifacts originate in the [2026-10-08 synthetic release](https://research.dvidia.org/downloads/topics/physical-grounding/runs/2026-10-08-footage-pipeline-v1/README.md), with [release ledger](https://research.dvidia.org/downloads/topics/physical-grounding/runs/2026-10-08-footage-pipeline-v1/release-ledger.json) and the [v0.1.0 training release](https://github.com/Dvidia-Inference/dvidia-training/releases/tag/v0.1.0). DVIDIA contributors distribute these authored examples, training artifacts and accompanying code under MIT; see `LICENSE` for the notice. Original footage is supplied separately in the dataset repository and excluded from `training-result.zip`.

Use this pilot to inspect small CPU learning, portable artifact contracts and bounded simulation integration. Its narrow synthetic imagery, authored trajectories and privileged features do not demonstrate human-video learning, shoe organization, unseen-scene competence or hardware transfer. No physical robot is qualified by these artifacts.

---
license: mit
viewer: false
tags:
  - video
  - robotics
  - simulation
  - synthetic
  - dvidia-training
---

![DVIDIA Training](assets/branding/github-header.png)

# DVIDIA Training: synthetic examples

Two authored datasets for reproducing the DVIDIA Training v0.1.0 CPU pipeline: **moving-disc videos for visual prediction** and **native simulation recordings with numerical movement labels**. These are software research fixtures. They contain no human footage, shoe demonstrations, physical robot recordings or calibrated camera perception.

[Training artifacts and measurements](https://huggingface.co/{{HF_ORG}}/dvidia-training-pilot) · [Demo](https://huggingface.co/spaces/{{HF_ORG}}/dvidia-training) · [Source and installation](https://github.com/Dvidia-Inference/dvidia-training)

## Contents

| Dataset | Source | Scale | Supervision |
| --- | --- | --- | --- |
| `visual/source/` | Procedurally drawn moving-disc MP4 videos | 10 recordings, 6 seconds each; 60 seconds total | Visual next-frame targets; no action labels |
| `native/source/` | Ten separately executed native MuJoCo placement recordings with schematic MP4 videos | 10 recordings; 116.5 seconds total | Timestamped privileged simulator state and authored inverse-kinematics joint targets |

Videos are H.264, 160 × 128 RGB schematics sampled at 10 Hz. The native videos depict a simulated six-joint arm and rigid box; they are not camera renderings. The native control labels were collected at 20 ms control timestamps when an authored target was requested. Source seeds are `8000` through `8009`.

Each source directory contains `skillspace.training.json`, `example-receipt.json` and `videos/`. Native source also contains `actions/`, `evidence/` and `task.skill.json`. Each native sidecar contains a timestamp, 12 context values, 6 bounded goal/orientation error values and 6 joint-delta targets per sample. Its feature contract is:

```text
q6_object_minus_tcp3_target_minus_tcp3__capped_goal_delta3_upright_orientation_delta3_v1
```

The action labels come from simulator state and authored targets, not from recovering actions from pixels. Media, sample and evidence hashes record declared consistency; the alignment declaration is not an independent attestation. The prepared intake receipt retains `actions_validation: "unvalidated_numeric_contract"`; the later movement trainer checks the strict sidecar contract, hashes and declared alignment. Neither stage independently attests the underlying evidence.

`archives/visual-skillspace.zip` and `archives/native-skillspace.zip` preserve the original source exports. `visual/prepared/` and `native/prepared/` preserve the measured `dataset.json`, `source_manifest.json` and every referenced `data/…` video or action sidecar from the published evidence. These prepared datasets can be supplied directly to DVIDIA Training; their recording split and media hashes remain inspectable.

## Recorded splits

Both datasets use seed `17` and connected groups formed from shared recording IDs, session IDs, shoe-pair IDs when present, and exact video hashes. In these examples, each recording has its own declared recording/session group. The protocol's target ratios are 0.70/0.15/0.15; the recorded allocation is 7/2/1 groups.

| Split | Recording IDs in each dataset | Visual duration | Native movement samples |
| --- | --- | --- | --- |
| Training | `example-01`, `02`, `04`, `05`, `07`, `08`, `09` | 42 seconds | 4,029 |
| Development | `example-00`, `03` | 12 seconds | 1,154 |
| Test | `example-06` | 6 seconds | 621 |

Abbreviated IDs in the table retain the `example-` prefix. Training fits weights, development selects ridge regularization, and test is scored after selection. The visual learner samples 12 frames per recording: 84/24/12 frames and 77/22/11 within-recording transitions across training/development/test.

The ten native recordings total 5,804 movement samples. Samples within a recording are correlated. Separate recording groups and supplier-declared `complete` flags do not establish independent task trials, novelty or a sufficient number of demonstrations. `example-06` is also the exact native scene used for the separate one-scene qualification; that qualification adds no new held-out scene.

## Download and reproduce

This repository is an archive and JSON/MP4 dataset, without a packaged Hugging Face `datasets.load_dataset` builder. The tabular Dataset Viewer is disabled because the files include nested simulator records, provenance and training contracts; they are intended for DVIDIA's validated intake. Download it with `huggingface_hub`, then use DVIDIA Training's source intake. The following requires Python 3.12+ and system FFmpeg, including `ffprobe`.

```sh
python -m pip install huggingface_hub
python -m pip install https://github.com/Dvidia-Inference/dvidia-training/releases/download/v0.1.0/dvidia_training-0.1.0-py3-none-any.whl
```

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="{{HF_ORG}}/dvidia-training-examples",
    repo_type="dataset",
    local_dir="examples",
)
```

Train and verify the visual result, using a fresh output directory:

```sh
dvidia-train run examples/archives/visual-skillspace.zip --seed 17 --output runs/visual
dvidia-train inspect runs/visual
```

To reuse the published prepared split and referenced media, replace the source argument with `examples/visual/prepared/dataset.json` (or `examples/native/prepared/dataset.json` for the native branch).

For the native movement branch and candidate capsule, add the optional simulation engine and supply the explicit task source:

```sh
python -m pip install mujoco==3.15.0
dvidia-train run examples/archives/native-skillspace.zip --seed 17 --task-source examples/native/source/task.skill.json --output runs/native
dvidia-train inspect runs/native
```

Software installation and repository download require network access. Training on the downloaded files is offline by default. Fitting and artifact inspection do not perform closed-loop qualification. See the [native pilot procedure](https://github.com/Dvidia-Inference/dvidia-training/blob/v0.1.0/benchmarks/native_pilot.py) for the separate collect, train, install and exact-scene test. Numerical reproducibility can depend on installed decoders and numerical libraries.

## Provenance, license and intended use

DVIDIA contributors authored the source generator and the synthetic recordings. These files originate in the [2026-10-08 footage-pipeline release](https://research.dvidia.org/downloads/topics/physical-grounding/runs/2026-10-08-footage-pipeline-v1/README.md). The [release ledger](https://research.dvidia.org/downloads/topics/physical-grounding/runs/2026-10-08-footage-pipeline-v1/release-ledger.json) declares MIT code and footage licenses and records an initial export-integration failure before the completed runs. The repository's `LICENSE` preserves the MIT copyright and permission notice.

Use these examples to inspect video intake, provenance records, grouped splits, small CPU fitting and simulation-only movement supervision. Their simple imagery, narrow authored trajectories and privileged numerical labels do not represent human behavior or the variety of real robot tasks. No human-video action bridge, shoe-organizing skill, unseen-scene generalization or physical robot qualification has been demonstrated. Reported model metrics and the single native qualification are documented separately in the [model card](https://huggingface.co/{{HF_ORG}}/dvidia-training-pilot).

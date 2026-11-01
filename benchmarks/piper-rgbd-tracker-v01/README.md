# DVIDIA · Known-block RGB-D tracker v0.1

A local NumPy estimator for one upright, orange, 40 mm cube. It consumes the
current RGB image, aligned optical-Z depth, camera calibration and frame timing.
It supplies robot-base position, yaw modulo 90 degrees, and explicitly
uncalibrated geometric uncertainty scales—or an explicit missing result.

This is a static simulated perception benchmark. It does not recover motor
actions from human video, train a policy, execute a robot, or qualify hardware.
The native action recorder and strict learning export are unchanged. Successful
perception is one prerequisite for the proposed observation-to-action study.

## Local use

Use Python 3.12 and the repository's pinned NumPy dependency. Estimation itself
uses the CPU and requires no network, hosted model, API key or dedicated GPU.
Generating these fixtures additionally requires MuJoCo and a working OpenGL
context. macOS arm64/CGL is the evaluated rendering host; other graphics backends
and minimum memory/CPU specifications remain unqualified. Install dependencies
and pinned robot assets once before an offline run.

```sh
git clone --branch piper-rgbd-tracker-v0.1-preview https://github.com/Dvidia-Inference/dvidia-training.git
cd dvidia-training
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install '.[arm]'
dvidia-robot-sim fetch piper --output runs/robot-profile-upstream/assets
python -m dvidia_training.piper_rgbd_benchmark prepare --output runs/rgbd-study
python -m dvidia_training.piper_rgbd_benchmark render --output runs/rgbd-study \
  --assets /absolute/path/to/robot-profile-upstream/assets --split development
python -m dvidia_training.piper_rgbd_benchmark estimate --output runs/rgbd-study --split development
python -m dvidia_training.piper_rgbd_benchmark score --output runs/rgbd-study --split development
python -m dvidia_training.piper_rgbd_benchmark freeze --output runs/rgbd-study
python -m dvidia_training.piper_rgbd_benchmark render --output runs/rgbd-study \
  --assets /absolute/path/to/robot-profile-upstream/assets --split confirmation
python -m dvidia_training.piper_rgbd_benchmark estimate --output runs/rgbd-study --split confirmation
python -m dvidia_training.piper_rgbd_benchmark score --output runs/rgbd-study --split confirmation
python benchmarks/piper-rgbd-tracker-v01/render_report.py runs/rgbd-study
python -m http.server 8361 --bind 127.0.0.1 --directory runs/rgbd-study
```

Every output is exclusive: use a new directory for a new experiment. Read the
report's coverage, refusals and gates together. A returned pose is not proof
that it is correct. `freeze` prevents subsequent changes to the candidate,
configuration, protocol and evaluated renderer binding during confirmation;
it does not certify the estimator or automatically allow learning export.

## Registered evaluation

Seed 311301 assigns eight development and sixteen confirmation pose groups.
Each group is rendered at 160×120 and 320×240 with ten conditions: clear,
partial occlusion, 15-degree tilt, 30-degree tilt, absent, fully occluded, an
ambiguous second orange cube, invalid depth, stale frame and future availability.
All variants and resolutions from a group remain in the same split. Positive
coverage is measured separately from absent/unsupported false availability.

The provisional gate requires at least 90% clear coverage, available-estimate
position p95 ≤10 mm and upright yaw p95 ≤10 degrees, zero negative false
availability, complete metrics and no pipeline failures. These are task-specific
research thresholds, not a confidence guarantee or a manipulation qualification.
Partial-occlusion results are reported separately. The two resolutions are
explicit comparisons; there is no hidden adaptive resolution or retry.

The fixtures use the actual native PiPER renderer. Object poses are authored
without physics advancing; some elevated blocks are intentionally unsupported
render poses. Occlusion cards and extra cubes are render-only geometry with
valid segmentation IDs so they also exist in depth. Native truth is written
separately and joined only after prediction receipts are sealed.

Synthetic 0.2-second exposure and stale/future times test clock rules; they are
not physical delivery measurements. CPU and wall estimator times exclude file
I/O and are descriptive measurements on a shared host. CGL renderer warnings
remain in the evidence; `depth_accuracy_verified` remains false. Error within
a heuristic scale is reported as a descriptive count, not calibrated coverage.

Original DVIDIA code and generated records are MIT. Upstream PiPER assets and
MuJoCo retain their notices; no robot meshes, private footage, credentials or
model weights are bundled in this evidence release.

## Frozen result

Source freeze: `846a1d9b04745a516ff060b4efa931e5e9898fbf`. Sixteen confirmation
pose groups × two resolutions × ten conditions: all 320 fixtures rendered and
scored once.

| Resolution | Clear estimates | Center error p95 | Yaw error p95 | Estimator wall p95 | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| 160×120 | 12/16 | 2.34 mm | 11.31° | 7.73 ms | Coverage and yaw failed |
| 320×240 | 16/16 | 1.98 mm | 6.36° | 7.58 ms | Clear-scene gate passed |

Both resolutions withheld all 256 negative estimates. Partial occlusion returned
0/32 estimates and remains unsupported. These are correlated static fixtures,
not 320 independent trials, hardware accuracy or learned placement.

Independent audit: zero mismatches across 320 rows, 20 summaries and two gates;
2,578 evidence files and 32 occluder-depth checks. The initial sandbox CGL failure
and development sources/results are retained. The first candidate dropped
shadowed orange faces; brightness floors 65/20 → 25/8 recovered coverage without
changing geometry gates. No confirmation tuning followed. The native action
recorder, strict learning export and earlier learned placement 5/12 versus its
authored teacher 12/12 remain unchanged.

# Camera-to-action skill v0.1

This local research workflow pairs rendered RGB-D with explicitly synthetic
PiPER teacher commands, fits a small CPU policy and compares complete tasks on
unseen starts. A model or successful installation does not establish qualification.
The four diagnostic trajectories and teacher-data gate must pass before fitting.
Failures are preserved and block dependent stages rather than triggering retries.

## Local setup

Python 3.12, the pinned NumPy and MuJoCo dependencies, existing verified PiPER
assets and a working OpenGL context are required. The tested renderer is macOS
CGL. No hosted inference or CUDA GPU is required; other operating systems and
hardware minimums remain unqualified. Native stepping is CPU work; rendering
uses the graphics driver. After dependencies/assets are provisioned, the workflow
is offline. The study reserves 30 GiB of free disk and caps its evidence at 120 GiB.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install '.[arm]'
dvidia-train robot prepare --study runs/camera-action --fixed-model /path/to/piper-base-model.json
dvidia-train robot collect --study runs/camera-action --assets /path/to/robot-assets --stage diagnostic
dvidia-train robot inspect --study runs/camera-action
dvidia-training-studio --directory runs/studio --port 8363
```

Open `http://127.0.0.1:8363/robot`, choose the prepared study and assets folders,
check the inputs, then run the remaining registered stages. The UI and Terminal
read the same immutable stage receipts. The UI stops when a gate blocks progress.
It installs a model only after a completed benchmark; installation remains a
simulation-candidate action. Ordinary video without paired actions is not an
accepted motor-learning input.

For manual execution after a passing diagnostic:

```sh
dvidia-train robot collect --study runs/camera-action --assets /path/to/robot-assets --stage demonstrations
dvidia-train robot train --study runs/camera-action --stage initial
dvidia-train robot evaluate --study runs/camera-action --assets /path/to/robot-assets --stage development-initial
dvidia-train robot collect --study runs/camera-action --assets /path/to/robot-assets --stage corrections
dvidia-train robot train --study runs/camera-action --stage corrected
dvidia-train robot evaluate --study runs/camera-action --assets /path/to/robot-assets --stage development-corrected
dvidia-train robot evaluate --study runs/camera-action --assets /path/to/robot-assets --stage confirmation
dvidia-train robot evaluate --study runs/camera-action --assets /path/to/robot-assets --stage negative
dvidia-train robot package --study runs/camera-action --output runs/camera-action-summary
```

The CLI package contains summary evidence and any fitted model JSON; it explicitly
does not include raw recordings or the executable runtime. Published research
archives additionally carry the independently audited evidence available for the
reported stage. Existing destination directories are never overwritten.

## What the learner sees

The frozen RGB-D tracker receives current pixels, depth and calibration only.
The policy receives a timestamped initial visual position, goal, elapsed time,
encoders, encoder-derived tool pose, previous applied servo references and 160 ms
of causal history. Missing live estimates retain availability masks and source
frame age; their numerical tensor fills are not measurements. Normalization uses
training data, including these explicit missing fills and masks.

The policy never receives native object/contact truth, teacher phase, current
teacher target or future frames. Historical initial position is never copied into
the current-object channel. Native truth is used only by the teacher and offline
scorer. The first tracker only recognizes a known upright block: tilt and gripper
occlusion remain measured gaps, not invented observations.

Teacher queries and issued commands are separate: one registered 50% mixed-control
round can request teacher labels on student-visited states. Those queries are
synthetic suggestions, not proven recoveries or operator corrections. Confirmation
has no teacher fallback. All candidates retain the native actuator and task bounds.

## Registered comparison

72 fresh initial-XY groups, split 32/8/32, share two known goals within each group.
Starts use a 3 mm inset from the original primitive's operating envelope to buffer
camera error, with no clipping or simulator-truth substitution. The fixed comparator
gets the same initial visual position. Comparisons also require at least 24 common
admitted groups so comparator refusals alone cannot establish an improvement.

The acceptance gates are at least 58/64 strict placements, positive 97.5% grouped
paired intervals versus fixed and blind controls, no increase in critical-contact
frequency, all sixteen negative controls behaving correctly and no privileged
learner inputs or hidden intervention. All failures remain in scheduled denominators.
These are proposed acceptance thresholds; the study report records whether they
were actually met. Offline host timings are not physical sensor-to-actuator latency.

## Candidate installation

For a fitted model from a completed study:

```sh
dvidia-arm check --model /path/to/model.json --robot piper
dvidia-arm install --model /path/to/model.json --robot piper --assets /path/to/robot-assets --output installed-camera-skill
dvidia-arm run installed-camera-skill --assets /path/to/robot-assets --output runs/installed-camera-attempt --start-x .30 --start-y -.03 --goal 0
```

Only the exact local PiPER simulation interface is accepted. No physical driver,
arbitrary-arm transfer, shoe skill, human-video motor recovery or real-time guarantee
is introduced. Original code and generated evidence are MIT; upstream asset notices
remain in the repository and assets are provisioned separately.

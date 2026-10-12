# DVIDIA robot simulation library

Each robot gets an explicit simulation profile and its own evidence. A profile
bundles source revisions, asset hashes, licenses, joints, command units, gripper
couplings and model assumptions. Switching profiles does not automatically
transfer a trained skill.

## Run the additional profiles

This candidate source is on `feat/reference-task-benchmark`; it is newer than
the frozen Observation Lab release. Clone that branch, create a Python 3.12+
virtual environment and install the optional CPU physics runtime:

```sh
git clone --branch feat/reference-task-benchmark https://github.com/Dvidia-Inference/dvidia-training.git
cd dvidia-training
python3.12 -m venv .venv
.venv/bin/python -m pip install '.[arm]'
.venv/bin/dvidia-robot-sim list
.venv/bin/dvidia-robot-sim fetch piper --output runs/robot-assets
.venv/bin/dvidia-robot-sim fetch so101 --output runs/robot-assets
.venv/bin/dvidia-robot-sim audit piper runs/robot-assets
.venv/bin/dvidia-robot-sim audit so101 runs/robot-assets
.venv/bin/dvidia-robot-sim benchmark piper runs/robot-assets --output runs/piper-smoke
.venv/bin/dvidia-robot-sim benchmark so101 runs/robot-assets --output runs/so101-smoke
```

On Windows, use `py -3.12 -m venv .venv`, `.venv\Scripts\python.exe`
and `.venv\Scripts\dvidia-robot-sim.exe`. Use a fresh output directory for each
benchmark. Fetch is the explicit network step; audit and native benchmarking
use the verified local files.

| Profile | Arm axes, excluding gripper | Source model | Current scope |
| --- | --- | --- | --- |
| `piper` | 6 | [PiPER Menagerie model](https://github.com/google-deepmind/mujoco_menagerie/tree/0059d4335f8156206f63a35662313385f7ad6d74/agilex_piper), MIT | Model audit, unloaded motion, an [authored contact pilot](piper-contact-pilot.md), [test-lift/recovery candidate](piper-grip-recovery.md), and [contact-history v0.3 comparison](piper-contact-memory-v03.md). |
| `so101` | 5 | [SO-101 Menagerie model](https://github.com/google-deepmind/mujoco_menagerie/tree/0059d4335f8156206f63a35662313385f7ad6d74/robotstudio_so101), Apache-2.0 | Model audit and bounded unloaded native motion. |

The native PiPER gripper command is one finger's sliding-joint target in meters,
with a mirrored-joint equality constraint; it is not a calibrated total jaw-gap
request. The SO-101 gripper command is a hinge target in radians. Both use native
joint-position servos. LeRobot's normalized or hardware command values require
an explicit calibrated adapter before they can be exchanged with these profiles.

DVIDIA owns the profile tooling and benchmark integration; the models retain
their upstream attribution and licenses. SO-101 has an openly published
[build/design repository](https://github.com/TheRobotStudio/SO-ARM100) and
[LeRobot guide](https://huggingface.co/docs/lerobot/so101). PiPER has an open
[SDK](https://github.com/agilexrobotics/piper_sdk); this does not establish open
manufacturing designs. Availability, pricing and delivery depend on your region.

## Qualification stages

| Stage | What it establishes | Required evidence |
| --- | --- | --- |
| Model loaded | The pinned native model compiles and matches the declared interfaces. | Asset verification, joint/actuator inventory, units, limits and model identity. |
| Motion measured | The declared bounded unloaded motions run in the native simulator. | Every attempted trajectory, numerical warnings, command tracking and timing boundaries. |
| Task measured | A particular controller completes a declared object interaction. | Frozen starts/materials, independent success rules, failures and matched controls. |
| Learned skill measured | A frozen learned policy works in new declared task conditions. | Data provenance, grouped splits, fresh closed-loop evaluation and artifact identity. |
| Hardware compared | Simulated and measured physical responses have been compared. | Serial-specific calibration, synchronized telemetry, uncertainty and retained failures. |

A clean-looking replay is useful for inspection but does not advance a profile
through these stages. The hosted lab plays recorded native results. Local Python
runs perform the actual physics, after explicit model provisioning.

## What the first task teaches us

The [native grasp benchmark](native-task-benchmark.md) measures a free rigid block
on the UR5e/Robotiq reference. Its authored placement controller, matched
open-jaw replay and hold control are separate from any learned policy. Its
complete receipt retains the timing outliers as well as successful placements.
Other arms need their own task geometry, feasible workspace, grasp settings and
fresh benchmark; the current six-joint/native-jaw command vector cannot be
reused simply because another robot also has six axes.

## From Skillspace to an installable skill

1. Select an exact robot and gripper profile.
2. Collect complete attempts, including failures, with synchronized observations
   and issued robot commands. Review reconstructed actions from human footage
   separately; object boxes and hand tracks are not actuator labels.
3. Freeze grouped training/development splits and declare the task's success,
   failure and coverage conditions.
4. Train the compact DVIDIA policy locally, then measure it in fresh native
   simulation attempts. Include varied starts, contacts, occlusions, materials
   and stop conditions appropriate to that skill.
5. Package weights, processors, units, observation history, compatibility and
   the exact benchmark receipt together. Reject incompatible installation.
6. Compare against the selected physical arm before describing hardware success.

Shoe organization will need explicit shoe shape, pair identity, orientation,
reachable grasp regions, slip and deformation coverage. Dexterous and bimanual
profiles add their own action spaces and collision/handoff benchmarks.

## Hardware and offline operation

The initial numerical benchmarks use CPU MuJoCo and do not require a discrete
GPU. Provisioning dependencies and pinned assets requires network access; local
physics can then run offline. Browser replay additionally needs a compatible
WebGL browser. A replay's smooth playback is not a native simulation speed test.

No universal minimum RAM, laptop speed or real-time guarantee has been
established. Publish CPU/OS/runtime, load time, stepping throughput, control-loop
percentiles and misses, process memory and rendering costs with their exact
measurement boundaries. More GPUs do not automatically accelerate a serial CPU
physics loop; parallel environments and perception/training need separate
measurements.

See the [hardware roadmap](hardware-roadmap.md) for source/licensing differences
and the [learning protocol](native-grasp-learning.md) for the next DVIDIA policy.

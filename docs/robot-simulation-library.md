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

## Reusable asset library: planned product layer

The next library layer is a collection of versioned assets that people can take
into a local simulator, compose into a task, and reuse across Skillspaces. This
extends the current pinned arm profiles with grippers, objects, environments and
replaceable camera/sensor profiles.
The [first shoe asset release](asset-library.md) implements a local catalog,
strict versioned scene loader, six synthetic rigid shoe proxies and native 3D
previews for all three supported arms. Its gravity/support benchmark checks this
bounded composition layer. A public package registry, measured physical
compatibility resolver, sensor profiles and shoe manipulation remain **planned**. Adding many meshes will grow the visual catalog; tested interaction
models and per-skill evidence will grow its useful manipulation coverage.

| Package kind | Reusable contents | Evidence needed before a stronger claim |
| --- | --- | --- |
| Arm profile | Joint hierarchy, frames, units, limits, collision geometry, inertia, actuator/control contract and supported simulator version. | Pinned source audit, native motion/limit tests, tracking and runtime measurements; separate measured hardware response. |
| Gripper or hand profile | Mount frame, finger geometry, joint couplings, command units, jaw/finger limits and declared sensing interfaces. | Coupling/closure tests, grasp workspace and retention trials; measured force, contact and latency before physical transfer. |
| Object family and instances | Stable instance IDs, visual and collision geometry, scale, mass/inertia assumptions, candidate grasp regions and task-relevant variations. | Geometry/scale checks, physical-property evidence and interaction tests for each claimed range. An instance does not establish its entire family's behavior. |
| Environment and fixture | Tables, shelves, bins, racks, supports, collision surfaces, lighting/camera configurations and reset distributions. | Reachability, support/contact and reset checks for compatible embodiments; task success rules independent of the controller. |
| Camera/sensor profile | RGB/depth camera intrinsics and placement, observation units, sampling clocks, delay/noise/dropout assumptions, and optional joint/force/tactile interfaces. | Recorded calibration and timing evidence for the claimed sensor; test missing/stale observations. A simulator pose channel is not a physical sensor. |
| Interaction benchmark | Asset versions, robot/gripper adapter, starts and disturbances, controller baselines, success/failure rules and complete result receipts. | All planned attempts, matched controls, untouched confirmation conditions and named compute/latency boundaries. |

Visual variations serve recognition: lighting, textures, views, backgrounds and
occlusion. Physical variations serve manipulation: mass, center of mass,
friction, stiffness, damping and contact geometry. Declare and benchmark these
axes separately, then test their combinations. Changing a texture must not
silently change friction, and rendering more views does not establish contact
accuracy. A replaceable sensor profile binds what the policy actually observes
to its frame, units and timing assumptions; switching sensors requires a
compatibility check and new evidence for the affected skill.

### Proposed package contract

Every future package should carry a human-readable card and machine-readable
manifest. The manifest needs an immutable package ID and version, content hashes,
source/derivation lineage, original attribution and license files, permitted
redistribution status, engine/importer version, units and coordinate frames.
Unknown redistribution rights block public distribution while local review can
continue. A conversion must retain its upstream notices and record the converter
and settings; a new export does not establish equivalent behavior in another
simulator.

Physical properties need evidence **per field**, rather than a single “realistic”
badge. Record each value or range with one of these states:

- **Measured:** method, instrument/calibration, tested specimen, conditions,
  sample count, date, uncertainty and original measurements.
- **Estimated:** estimation method, source observations or literature, assumed
  range and uncertainty; keep estimates distinguishable from measurements.
- **Untested:** an explicit assumption or placeholder, with its source and the
  experiment needed to replace it. Omit unknown values instead of inventing
  precision.

For objects this includes dimensions, mass, center of mass, inertia and surface
regions. Friction belongs to a contact pair and conditions: shoe sole against a
shelf, or upper against a particular pad, may require different measured ranges.
Stiffness, damping, deformation and damage thresholds require their own loading
conditions and observations. For robots, apply the same evidence states to
actuator response, force limits, backlash, compliance, sensing and delay.
Geometry reconstructed from footage supplies appearance and shape evidence;
it does not supply these physical measurements by itself.

### Versioned fidelity and compatibility

A proposed asset card should show the following capabilities independently, each
with its model version, test envelope and evidence links. Higher visual detail
must not silently promote contact or deformation fidelity.

| Fidelity claim | What users can rely on within the declared scope |
| --- | --- |
| Geometry available | Importable visuals/collisions with declared scale and frames; physical parameters may still be untested. |
| Native mechanics tested | The declared rigid-body/articulation configuration passes numerical and interface checks in a named engine. |
| Contact compared | Contact/load/retention responses have been compared with measurements for named material pairs and conditions, with residual error reported. |
| Deformation compared | A specific flexible-body model has measured force/deformation and recovery evidence within its stated range. Rigid proxies remain separately identified. |
| Task benchmarked | A named policy, embodiment, asset set and task version have complete simulation receipts; this claim belongs to that combination. |

These claims complement the existing qualification stages; they do not replace
physical robot qualification. A package may have measured mass and untested
friction, or tested rigid collisions and no deformation model. Keep that mixed
status visible in search and installation.

Compatibility should declare the arm/gripper mount and reachable workspace,
joint/action units and couplings, object size/mass range, required observations,
simulator features and supported runtime cost. A dependency lock should bind
these versions to the Skillspace benchmark and eventual skill capsule. A
mismatched adapter, missing contact capability or unsupported deformation model
should produce a clear incompatibility result. Downloading an asset supplies
simulation components; installing a learned skill additionally requires its
policy, processors, calibrated adapter and qualification evidence.

### First shoe collection

Start with shoe organization on one compatible arm/gripper and a tabletop or
low rack. Keep pairing, orientation/alignment and placement as separate task
contracts with independently observable outcomes. Shoe tying and dexterous
manipulation remain later deformable/bimanual work.

1. Collect several physical shoe instances across a few useful families, such as
   structured shoes, flexible uppers and slippers. Capture complete attempts,
   failures, multiple views, pair identity and persistent instance/session IDs.
   Reserve physical instances and capture sessions for confirmation before fitting.
2. Build versioned visual/collision proxies and a table/rack fixture. Record
   measured dimensions and mass where available; label unknown contact and
   flexibility assumptions. These first rigid proxies must be described as
   geometry/task-development assets, not validated shoe physics.
3. Collect controlled grip, lift, hold and release measurements for the actual
   gripper pads and shoe regions. Compare friction, load response and recoveries;
   add compliant models only where their measured behavior supports a task need.
4. Publish each asset's package, notices, evidence status and compatible task
   receipts. Measure success, false completion, slip/drop, recovery, damage where
   observable, placement error and runtime on declared hardware. Retain failures
   and report simulation-to-measurement discrepancies.
5. Add new instances, object families and arms through the same import, provenance,
   compatibility and benchmark gates. Community submissions should include a
   reproducible audit and evidence gaps, so catalog growth does not erase which
   interactions remain untested.

The catalog UI should let users preview an asset, inspect the physics evidence,
filter by arm/gripper and simulator compatibility, and download a pinned package
with a runnable example and benchmark command. Skillspace should link the exact
asset/benchmark versions it needs and expose missing coverage as collection tasks.
The first release provides these browsing, scene-download and local-run basics
for the three arms and synthetic shoe proxies. The broader registry and
interaction-qualified package workflow remain part of the roadmap.

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

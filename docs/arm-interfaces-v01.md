# Arm interfaces and skill installation v0.1

One local command now describes three native simulation adapters, checks a
learned policy against its exact embodiment, installs its data, and runs an
independently scored attempt. The same package includes a common tracking
benchmark and a small environment-conditioned lift-planning assessment.

This is a research preview. The three adapters are PiPER, SO-101, and UR5e with
a Robotiq 2F-85. Only PiPER has a learned policy runner in this installer. No
physical arm is connected, calibrated or qualified by these commands.

## Install locally

Use Python 3.12 and a fresh virtual environment. This preview is source branch
`feat/arm-interfaces-v01`; the existing stable package and previous preview do
not contain `dvidia-arm`.

```sh
git clone --branch feat/arm-interfaces-v01 https://github.com/Dvidia-Inference/dvidia-training.git
cd dvidia-training
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install '.[arm]'
dvidia-arm list
dvidia-arm describe piper
```

The CPU simulator needs no CUDA GPU. Setup requires network access; commands
use local, verified assets after provisioning. Hardware minimums and other OS
performance are unqualified. A browser UI can call the same Python functions;
this preview adds Terminal commands and does not change the public website.

## Three explicit command contracts

| Adapter | Arm axes | Gripper command | Available here |
| --- | --- | --- | --- |
| `piper` | 6, radians | One coupled finger slide target in metres; not total jaw gap | Native mechanics and existing learned block policy |
| `so101` | 5, radians | Gripper hinge angle in radians | Native mechanics; no learned task runner |
| `ur5e-2f85` | 6, radians | Robotiq native actuator input, 0–255; not metres or newtons | Native mechanics; no learned task runner in this installer |

Asset identity, ordered actuator names, units, couplings, limits, native version
and compiled model identity travel with the adapter. Unknown commands, wrong
units/order, nonfinite values and limit violations are rejected. Named command
inputs require the complete exact name set. No driver is selected from a model
file and no downloaded code is executed.

Provision the already pinned models with `dvidia-robot-sim fetch piper` and
`dvidia-robot-sim fetch so101`, each with `--output robot-assets`. The UR5e and
Robotiq reference uses `dvidia-embodiment fetch-assets --output reference-assets`,
documented in [hardware-reference.md](hardware-reference.md). Different roots can be passed
to the shared benchmark using `--reference-assets`.

## Check, install, then run

Use the published [PiPER block policy](https://huggingface.co/Dvidia/piper-block-placement-v01).
The exact initial release model is:

```sh
curl --fail --location 'https://huggingface.co/Dvidia/piper-block-placement-v01/resolve/10edc19c63f04d1428e051787a601e5b69d8a3f5/model/model.json?download=true' --output model.json
printf '%s\n' '2976a605575d9edefb5db366e72c5e006c0ce85a5531ad3d61512436864c8ded  model.json' | shasum -a 256 -c -
dvidia-arm check --model model.json --robot piper
dvidia-arm install --model model.json --robot piper --assets robot-assets --output installed-placement
dvidia-arm inspect installed-placement --assets robot-assets
dvidia-arm run installed-placement --assets robot-assets --start-x .309 --start-y -.04 --goal 1 --output placement-attempt
```

`install` verifies and copies data without taking physics steps. `run` checks
the installation again, then uses the existing bounded PiPER runner and retains
the trace and independent score. A failed attempt remains a failed attempt. The
output directory must be new. Changing the policy, runtime, assets or contract
requires a new installation. The same policy on `--robot so101` or
`--robot ur5e-2f85` is rejected, even though UR5e also has six arm axes.

This policy was distilled from synthetic controller demonstrations. The frozen
held-out placement result remains 5/12, versus 12/12 for the authored teacher.
One installation smoke does not update that benchmark. The fresh installed dev3 wheel passed the selected example once, but its full control-step compute p95 was 34.924 ms with 179/1,000 nominal 20 ms budgets exceeded on the shared host; this is not real-time qualification. Its operating envelope
is a 40 g rigid block, initial x=0.284–0.314 m and y=−0.045–−0.015 m, and two
known destinations. The published feed-forward primitive uses initial native
object position, goal and time. It has no learned camera or contact recovery.

## Measure accuracy and compute separately

```sh
dvidia-arm benchmark --assets robot-assets --reference-assets reference-assets --output tracking-study
```

The shared protocol exercises hold, smooth arm motion and gripper cycling, at
10, 20 and 40 ms control updates with a fixed 1 ms physics step. It preserves
all attempts and compares joint tracking error, gripper tracking in its native
units, terminal-link tracking and loop compute percentiles. Compilation and
loop compute are separate costs. These are unloaded simulated mechanics, not
object-placement success, sensor-to-actuator latency or physical accuracy.

The [first frozen study](../benchmarks/arm-interfaces-v01/README.md) retained
54/54 numerically valid attempts, 21,654 state rows and 1,260 issued commands.
An independent audit reconstructed 648 RMS/max values without a mismatch.
There were 21 offline compute-budget exceedances. The two repeats share identical
initial conditions; they measure timing variability, not task generalization.
The host was shared, not a dedicated real-time machine. The receipt field
`native_step_compute_s` includes the full observed-step wrapper (forward update,
state snapshot and reference forward kinematics), not isolated physics cost.

Do not increase a learned policy's speed merely by replaying it faster. Its
training clock, acceleration, contact, braking and recovery requirements need
a new task benchmark. Optimize on development scenes, freeze the choice, then
evaluate untouched confirmation scenes. Report the accuracy/compute tradeoff
and failures rather than collapsing unlike units into a single ranking.

## A skill needs a suitable scene

For state `x`, estimated physical properties `theta`, embodiment `r` and task
`g`, applicability is a set `A(g, r, theta)`. The controller should transport
only while its estimated state and uncertainty satisfy the task's conditions.
Probing changes the estimate of `theta`; it is not evidence that the original
estimate was correct. A new object, pose or grasp can invalidate earlier evidence.

The first planning gate uses a deliberately limited vertical, two-finger
friction model:

```text
required force = mass upper bound × (gravity + upward acceleration) × safety factor
friction capacity = 2 × friction lower bound × allowable normal force per finger
```

It also checks declared reachability, gripper fit, pose uncertainty, freshness,
slip, calibration identity and a pose-specific payload ceiling. Unknown mass or
friction remains unknown. Supplied measurements are caller assertions; the tool
does not authenticate a calibration or measure a robot.

```sh
dvidia-arm task-template > lift-conditions.json
dvidia-arm assess-task lift-conditions.json
```

Results are `blocked`, `probe_required`, `test_lift_required`, or
`ready_for_simulated_transport`. This assessment performs no sensing or motion
and is not wired into the existing learned block controller. It omits moments,
center of mass, deformation, contact damage dynamics and impacts. The input
normal-force budget must already respect the object's allowable load and the
gripper's calibrated capability. A real runtime needs time-aligned evidence,
force limits, slip monitoring and bounded recovery before using this decision.

The next experiment is to vary object mass, friction, center of mass and sensor
delay independently, then compare an unconditional lift with probe → small lift
→ transport/regrasp/abstain. Measure drops, false secure-grip decisions, overload,
completion, recovery time and observation-to-command latency. Preserve failure
conditions, especially when probing cannot resolve uncertainty.

## Expansion roadmap

Add adapters by interface family, then qualify each exact arm/gripper pairing.
The following are integration candidates, not supported DVIDIA hardware:

| Priority | Candidate family | Upstream interface to build on |
| --- | --- | --- |
| First | SO-100, SO-101; later Koch | [LeRobot robot interface](https://github.com/huggingface/lerobot/blob/main/README.md), with distinct motor maps and per-device calibration |
| First | PiPER and its firmware variants | [AgileX CAN SDK](https://github.com/agilexrobotics/pyAgxArm) and [ROS 2 driver](https://github.com/agilexrobotics/agx_arm_ros) |
| Next | OpenArm, later bimanual configurations | [LeRobot OpenArm guide](https://huggingface.co/docs/lerobot/openarm), CAN and distinct arm/hand calibration |
| Next | UFACTORY xArm variants | [Vendor ROS 2 packages](https://github.com/xArm-Developer/xarm_ros2) |
| Next | Franka research arms | [Vendor ROS 2 integration](https://github.com/frankarobotics/franka_ros2) |
| Next | Interbotix / WidowX family | [Vendor manipulator packages](https://github.com/Interbotix/interbotix_ros_manipulators) |

[ros2_control](https://control.ros.org/jazzy/doc/ros2_control/hardware_interface/doc/hardware_interface_types_userdoc.html)
provides a common hardware-component vocabulary. It does not erase joint limits,
transmission, firmware, tool, sensor, or task differences. Published software
interfaces also do not establish that every robot's manufacturing design is
open source. Review each dependency's actual license before integration.

Every new adapter needs a pinned asset/driver inventory, named units and frames,
calibration mapping, stale-state/limit/stop tests, a native mechanics receipt,
and a task-specific policy evaluation. Physical support requires measured
hardware evidence. Cross-arm transfer remains a separate learning and retargeting
problem; it cannot be solved by renaming joint fields.

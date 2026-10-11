# Upcoming hardware support

Updated 11 October 2026. This is DVIDIA's planned support list, not a purchase
decision, delivery schedule or declaration of compatibility. PiPER and SO-101 now have [pinned CPU simulation profiles](robot-simulation-library.md)
and unloaded numerical benchmarks. PiPER also has an [authored contact pilot](piper-contact-pilot.md).
Their object-task learning and physical
adapters remain unqualified. The other additions below remain upcoming.

## Planned targets

Arm-axis counts below exclude the gripper. Fingers, coupled joints and independent
motor commands need their own inventory; a model never becomes compatible by
padding or truncating its action vector.

| Priority | Target | Intended role | Source availability and integration work |
| --- | --- | --- | --- |
| 1 | **YAM — 6 arm axes + gripper** | First candidate for a full-size, single-arm shoe grasp-and-place benchmark. | MIT control/model repository supplies MuJoCo, gripper control and teleoperation. Complete open manufacturing designs are not established by that repository. Pin the exact arm/gripper revision, audit its dynamics and fit physical response. [Official SDK/models](https://github.com/i2rt-robotics/i2rt). |
| 2 | **OpenArm — 7 arm axes + gripper per side** | Later two-arm coordination, with dexterous hands evaluated as separate end effectors. | Published hardware uses CERN-OHL-S-2.0; MuJoCo/control packages use Apache-2.0. Native LeRobot support currently connects hardware on Linux. Keep v1/v2 profiles distinct and account for hand mass, mounts, power and remaining object payload. [Hardware](https://github.com/enactic/openarm_hardware), [MuJoCo](https://github.com/enactic/openarm_mujoco), [LeRobot](https://huggingface.co/docs/lerobot/main/openarm). |
| 3 | **AgileX PiPER — 6 arm axes + gripper** | Alternative six-axis community/research target. | The pinned MIT Menagerie profile now has a DVIDIA loader, audit, unloaded CPU benchmark and authored rigid-block contact pilot. Full manufacturing openness is not established. Physical firmware, offsets, control modes and gripper calibration remain future adapter work. [SDK](https://github.com/agilexrobotics/piper_sdk), [model](https://github.com/google-deepmind/mujoco_menagerie/tree/main/agilex_piper), [manufacturer](https://global.agilex.ai/products/piper). |
| 4 | **ARX X5 — 6 arm axes + gripper** | Research adapter for joint position, velocity and torque evidence; later multiple-arm control. | MIT research SDK and community LeRobot integration. Hardware licensing and an exact X5 simulation still need qualification; a differently named ARX model cannot substitute for it. [Research SDK](https://github.com/real-stanford/arx5-sdk), [LeRobot plugin](https://github.com/villekuosmanen/lerobot-arx5). |
| 5 | **PAROL6 — 6 arm axes + separate gripper** | Community-build desktop simulation target. | GPLv3 mechanics/software, Python and ROS tooling. The documented open-loop variant has limit-switch position sensing; verify actual joint feedback before claiming contact or tracking accuracy. Complete electronics openness is not established. [Project](https://github.com/Source-Robotics/PAROL6-Desktop-robot-arm), [specifications](https://source-robotics.github.io/PAROL-docs/page2_2/). |

The accessible **SO-101** entry profile now has a pinned native model, audit and
unloaded CPU benchmark in the [simulation library](robot-simulation-library.md).
It has five arm axes plus its gripper and official LeRobot collection support;
it requires a different action/observation contract from a six-axis arm.
[Official build](https://github.com/TheRobotStudio/SO-ARM100),
[LeRobot guide](https://huggingface.co/docs/lerobot/main/so101).

Our current [UR5e/Robotiq reference](hardware-reference.md) is already compiled and
numerically characterized. Its physical response, shoe-task competence and
serial-specific calibration remain unmeasured.

## What each addition must deliver

1. **Pinned simulation profile:** source revision and notices, named arm/gripper
   joints, independent actuators and couplings, units, mounting/TCP frames,
   geometry/collision coverage, masses, sensors and controller assumptions.
2. **Physical adapter:** calibration, requested/applied commands, measured feedback,
   actual acquisition/arrival clocks, observation age and fault reporting.
3. **Simulation benchmark:** native mechanical checks, numerical sensitivity,
   latency/throughput/memory with stated measurement boundaries, and all attempted
   nominal, stress, failed-grasp and recovery task trials.
4. **Physical comparison:** matched command replays, independent calibration and
   confirmation groups, pose/gap/contact/slip residuals, actual control latency
   and retained failures. Importing a mesh cannot satisfy this step.
5. **Installable skill evidence:** a robot-specific policy and supported conditions,
   with separate simulation and physical results tied to the exact artifact.

For the first shoe task, choose one actual arm and gripper before broadening
hardware work. Opening, finger geometry, slip, workspace and usable payload drive
that choice. Five-finger hands require a separate geometry, sensing, actuation and
benchmark contract.

Support follows the [execution map](roadmap.md). LeRobot can supply recording and
device connections while DVIDIA develops its own action model. Support for a
device does not establish that any existing skill model transfers to it.

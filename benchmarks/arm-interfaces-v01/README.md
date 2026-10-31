# DVIDIA arm interface tracking evidence v0.1

Three explicit native simulation interfaces: **PiPER**, **SO101**, and **UR5e + Robotiq 2F-85**.

This is a local CPU mechanics and interface benchmark. It contains no trained policy, object manipulation task, physical robot connection or hardware-latency qualification.

## What was run

- 54 fixed attempts: three profiles × hold / bounded arm perturbation / gripper cycle × 10 / 20 / 40 ms updates × two repeats.
- Every attempt simulates 0.4 seconds at a native 1 ms timestep, without hidden settling or automatic retries.
- 54/54 attempts were numerically valid, with zero native warnings; this is not a task-success count.
- 21,654 recorded state rows and 1,260 issued command events are retained.
- **21 of 1,260 offline control blocks exceeded their nominal compute budget**: PiPER 13, SO101 1, UR5e/Robotiq 7. These are unpaced computation diagnostics, not measured physical deadlines.
- The independent read-only audit recomputed 648 RMS/max values with zero trace/source digest or command-alignment mismatches.

All cases and repeats are retained. Repeats use identical deterministic initial states and trajectories for timing variability; they do not establish generalization, confidence intervals or statistically independent task trials. No profile or update period was selected automatically.

## Accuracy and speed mean specific things

Arm tracking is measured in radians. PiPER gripper tracking uses its single-finger slide command in metres, SO101 uses a moving-jaw hinge target in radians, and Robotiq uses its native 0–255 servo coordinate. These are reported separately.

Robotiq feedback is derived from the pinned affine servo model as `-(bias0 + bias1 * actuator_length) / gain`. It is not an aperture, force or tactile measurement. Raw state and the mapping parameters are retained.

Tracking errors compare observed state with both the held issued target and the continuous reference at the observation time. Terminal-link position/orientation errors use native forward kinematics at the desired arm configuration; the link origin is not a calibrated fingertip tool centre point.

`compile_and_initialize_wall_s` includes pinned asset verification, model compilation and initialization. `control_loop_compute_s` includes the complete control block. **`native_step_compute_s` measures the full observed-step wrapper: native stepping, forward update, state checks/snapshot and reference FK. It is not pure physics-engine execution time.** Trace serialization is outside the loop; total attempt timing stops before writing the final result receipt.

The run used Python 3.12.9, MuJoCo 3.15.0 and NumPy 2.5.3 on Darwin arm64. The machine was an interactive shared host, not an isolated real-time test rig. Upstream gravity compensation and mechanics differ by profile. Do not interpret this table as a hardware quality ranking, minimum-system specification or sensor-to-actuator latency measurement.

## Read the evidence

[Interface and installation guide](../../docs/arm-interfaces-v01.md)

- [Bounded summary](summary.json): all 27 condition summaries, settings, source hashes and timing scope.
- [Independent audit](audit.json) and [original audit source](audit.py).
- [Prepared artifact identities](publication-artifacts.json): archive and full inventory hashes.

The companion evidence ZIP retains every raw trace; large traces are excluded from Git. The portable verifier is inside the ZIP. This directory records prepared artifacts and does not itself claim a completed publication.

Archive: `arm-interfaces-v01-release-20261031.zip` · 11,159,124 bytes · SHA-256 `48200eb5c647badff7db5ee14272571b93d9c31003d9f5f92dd6e28d062b272b`.

Original DVIDIA source and synthetic records: MIT. PiPER simulation assets: MIT; SO101: Apache-2.0; UR5e: BSD-3-Clause; Robotiq 2F-85: BSD-2-Clause. Full upstream notices are preserved in the companion archive.

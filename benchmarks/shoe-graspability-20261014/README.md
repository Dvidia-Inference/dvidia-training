# Shoe graspability v0.3 — frozen diagnostic

Source: [`73057dc259c6a19344906f4a43a283553e17ade0`](https://github.com/Dvidia-Inference/dvidia-training/tree/73057dc259c6a19344906f4a43a283553e17ade0).

**All 24 native attempts are valid. The 30 g controls complete 8/12 lifts; all 12 attempts at 300 g fail.** Removing the rake gains one narrow-shoe result and loses one block result, with no net completion gain. The full rigid shoe completes 1/2 lightweight cases with either tool condition and 0/2 at 300 g. This is a direct-grasp diagnostic, not a shoe retrieval success rate or an improvement claim over v0.2.

[Research](https://research.dvidia.org/papers/shoe-graspability-v03/) · [Complete recorded trials](https://research.dvidia.org/downloads/topics/physical-grounding/runs/2026-10-14-shoe-graspability-v03/lab/) · [Hugging Face](https://dvidia-robot-simulation-lab.static.hf.space/prerequisites/v03/index.html) · [Run locally](../../docs/shoe-graspability.md)

| Rigid shape | 30 g, rake mounted | 30 g, rake absent | 300 g, mounted | 300 g, absent |
| --- | ---: | ---: | ---: | ---: |
| 40 mm block | 2/2 | 1/2 | 0/2 | 0/2 |
| Narrow shoe proxy | 1/2 | 2/2 | 0/2 | 0/2 |
| Current shoe proxy | 1/2 | 1/2 | 0/2 | 0/2 |

The two replicates use paired layout/friction combinations. The controller has fixed shape-specific grasp offsets and shares its trajectory, gains and command limits across the tool and load conditions. The rake is present or absent from reset. Its removal changes collision geometry and wrist inertia; neither autonomous detachment nor the preceding retrieval is tested. Shape changes also change inertia and contact geometry. A 30 g shoe shape is a deliberate lightweight control, not a measured physical shoe.

## Evidence

- [Frozen protocol](protocol.json), [all-attempt benchmark index](benchmark.json), [complete export inventory](integrity.json).
- [Derived counts, timing and force summary](analysis.json).
- [Independent native audit](frozen-audit.json), [portable audit script](audit_frozen.py), [recorded jaw-limit diagnostic](jaw-limit-audit.json) and [diagnostic script](audit_jaw_limits.py).
- [Source-bound release acceptance](acceptance.json) and [local browser checks](browser-qa.json).

This directory contains compact evidence. The complete **107-file, 55,750,637-byte** export includes every full trial, all 79,200 native ticks, geometry and motion, replay UI and upstream licenses on the linked research site and Hugging Face. The source was frozen before this single confirmation execution. Replaying or rerunning these cases does not make them unseen evaluation data again.

The independent scorer verifies world-supported, robot-clear settling at 0.4 s over the previous 100 ms, with at most 1 mm position variation on each axis. Success needs object-origin rise **at least 35 mm** relative to that reference, both finger normal forces **above 0.01 N**, and world/rake/other-robot support **below 0.01 N** on all 250 final native ticks (500 ms). The object's lowest collision surface is reported separately. These are synthetic scoring conventions, not physical force or safety limits.

All 12 tool pairs retain the same robot actuator fingerprint. Full compiled-model hashes differ because object shape, mass and tool geometry differ. Trial validity means numerical/evidence validity, not successful manipulation or hardware fidelity.

## What changed in our understanding

The current scene does not establish that the shoe alone is the problem. Even the block fails every 300 g condition. The pinned gripper position-drive gain is 40 N/m, damping is 5 N·s/m, and its actuator cap is ±10 N. A force cap does not cause the actuator to apply that force. The highest absolute gripper actuator force across this cohort is **1.7024 N**, a transient observation rather than a guaranteed gripping capacity. Actual fingertip contact forces are separate trace columns. No setting was increased to make the benchmark pass.

The full shoe is a rigid, solid proxy: no hollow collar, flexible upper, laces or heel tab. Its 100 mm sole and 94 mm upper exceed the pinned model's nominal 70 mm slide-coordinate span, while the controller targets a narrower region. This geometry omits useful physical grasp strategies and does not prove all shoe grasps impossible. Both retained lightweight current-shoe attempts also exceed the nominal 70 mm span throughout their final 250 native ticks: **71.003–71.135 mm mounted; 71.597–71.768 mm absent**. Sampled issued commands remain within their limits, while the simulated joint-limit constraints permit actual overtravel. The frozen score did not include a joint-limit tolerance, so those results are unchanged. This does not establish a physical aperture or prove the same lift works with hard travel stops. Next work should check drive calibration and reachable contact geometry against named hardware.

## Timing

On the recorded macOS arm64 host, Python 3.12.9 and MuJoCo 3.15.0, the 24 attempts total **158.40 simulated seconds**, **8.4043 s native wall time**, and **20.0317 s separate audit/export time**. Each episode is 6.60 simulated seconds. Native timing includes initialization, control, physics and recording, excluding rendering, downloads and physical sensing/actuation.

There are 7,920 timed target-solver calls: p95 **0.0443 ms**, maximum **4.2008 ms**, zero above 20 ms. These calls do not measure the complete observation/control loop or establish real-time hardware latency. CPU-native simulation needs no discrete GPU; browser replay requires WebGL2. Minimum RAM remains unmeasured.

Authored policy, privileged simulator observations, rigid synthetic contacts and uncalibrated hardware parameters. Original code and synthetic primitives are MIT; upstream notices are bundled. No learned skill or physical-robot qualification is claimed.

# Video to an installable robot skill: evidence gates

The product target is to upload a demonstration, produce a reusable skill, and
reproduce its outcome quickly with a supported arm. The current code establishes
only a narrow portion of that path. Treat support as an independently tested
combination of task, object, camera, robot, gripper and operating conditions.

## Current boundary

| Component | What exists | What it does not establish |
| --- | --- | --- |
| Video destination | Encoded pixels → terminal planar XY for one green proxy and a supplied orthographic calibration | Arbitrary object identity, metric depth or human action inference |
| Saved candidate | Local data-only bundle, runtime/profile checks, fresh-process simulation execution | A trained motor policy or compatibility with every arm |
| PiPER execution | Authored contact-based block pickup, transport, release and independent native scoring | Sensor-driven hardware control, shoe grasping or calibrated contact |
| Fabric appearances | Five NVIDIA diffuse-map variants with physics-invariance checks | Measured fabric friction, softness or improved trained perception |
| Other arms | Pinned mechanics profiles and separate existing reference tasks | A shared qualified manipulation adapter for the video candidate |

See [destination evidence](../benchmarks/video-goal-bridge-v01-20261025/README.md),
[appearance evidence](../benchmarks/nvidia-material-appearance-v01-20261025/README.md)
and the [SimReady integration](nvidia-simready-integration-v01.md).

## Next experiments, in order

1. **Ground the same object despite appearance changes.** Replace the green rule
   with a selected-object tracker or compact detector/segmenter. Freeze object
   identities, material families, camera layouts and source groups before
   fitting. Evaluate unseen appearances, lookalikes, occlusion, absent targets,
   camera motion and uncertain depth. Report metric endpoint error, identity
   switches, abstentions and false proposals. A supplied reference annotation or
   calibration must be recorded as supervision, not claimed as inferred.
2. **Reconstruct the task state that matters.** Record object pose, size,
   support, articulation, reachable grasp regions and uncertainty. A single RGB
   view usually underdetermines geometry behind objects and their physical
   properties. Use calibrated depth/multiple views or bounded hypotheses; let
   failed hypotheses remain visible. Use SimReady Central as a source of known
   asset hypotheses, with exact licenses and profiles, rather than assuming its
   library covers the observed object.
3. **Learn actions from measured demonstrations.** Capture synchronized RGB or
   RGB-D, monotonic timestamps, calibration, joint position/velocity, commanded
   actions, gripper state and available force/contact signals. Keep failed
   grasps, recoveries and interruptions. Compare a compact action model trained
   on robot demonstrations alone against the same label budget augmented with
   video-derived supervision. A goal inferred from a video is not an action label.
4. **Prove one additional robot adapter.** Map the shared task frame into that
   arm's base/TCP, joint units, limits, gripper aperture and sensor contract.
   Replay the same goals with fresh starts. Each arm needs independent contact
   completion, collision, calibration-offset and no-grip controls. Unreachable
   or unsupported cases remain in the denominator or explicitly refuse execution.
5. **Qualify and package a learned skill.** Bind model weights, observation and
   action schemas, robot adapter, operating envelope, recovery policy, source
   split and frozen evidence. Install into a fresh offline process and rerun
   before claiming compatibility. Small model size and laptop speed are measured
   properties, not consequences of calling an artifact a skill.

## Measurements for every milestone

Keep task completion separate from proposal accuracy and simulator validity.
Report every scheduled attempt, uncertainty/refusal coverage, retained lifts,
placement error, peak contact/actuator response, collision and slip events,
recovery count, end-to-end latency and compute-only latency. Separate capture,
observation dwell, decode, inference, planning and actuation times. Describe CPU,
memory and any GPU use. Compare matched oracle, absent/shuffled video, calibration
fault and open-gripper conditions without changing the success reference.

“One shot” means one predeclared execution per new scene, with any internal
recovery explicitly counted. Passing a synthetic block test advances the path;
it does not prove transfer to an unseen shoe, unmeasured gripper or real hardware.

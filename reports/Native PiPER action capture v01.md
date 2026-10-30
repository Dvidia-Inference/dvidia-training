# Preserve what the camera saw and what the arm actually received

DVIDIA's native PiPER capture preview connects rendered RGB-D observations to measured joints and explicit actuator commands. Two 20-second runs preserve 1,000 complete control intervals each. All 1,001 paired RGB/depth frames, applied controls and measured motion match exactly between those runs. A declared interruption retains two complete intervals and an unknown outcome. This establishes repeatable collection for one authored simulator trajectory. It does not establish a newly learned skill.

## The missing link between footage and motor learning

An ordinary video shows the consequences of movement. It usually omits the motor command, measured gripper response, camera calibration and time at which each observation became available. A language description cannot fill those channels reliably.

The first capture tool preserves that evidence where we can measure it: inside native MuJoCo simulation. Each 20 ms boundary records the before-observation, authored decision, requested and issued targets, final ramped reference, actual native actuator controls, and after-observation. Native physics advances at 1 ms. The seven commands remain six radian arm references and one finger-slide reference in metres. The two measured, signed finger positions and velocities remain distinct from that command.

The controller still uses privileged native object and contact state. Its diagnostic inputs live outside the primary camera/encoder observations. No operator corrections were collected and no model was fitted.

## Actual pixels with an explicit clock

Each boundary includes a real 160×120 RGB PNG and an aligned float32 depth array in metres along the camera's optical forward axis. Intrinsics, optical axes, pixel-centre convention and camera-to-base transformation are explicit. Measurements use a shared simulation clock; the simulator pauses during rendering. These are instantaneous synthetic exposures, not measurements of physical camera delivery latency.

The four object-pose and uncertainty channels remain missing/null. Rendering a scene does not implement an object tracker, and native object truth cannot be silently renamed as a visual estimate. Strict learning-target export refuses the full captures for missing evidence and the partial recording for its interrupted status. This is a requirement of the present action-learning input contract, not a claim that every possible robotics policy must use an explicit pose tracker.

## A full recording, a repeat and an interruption

The engineering protocol scheduled two identical 20-second captures from one fixed synthetic start and a short interrupted capture. Independent readback decoded every registered RGB image and depth array, checked byte identities and shapes, and checked causal camera boundaries. Both full runs contain 1,000 control intervals and 1,001 camera observations; the interrupted run contains two intervals and three observations. All missing tracker values remain explicit.

The two full runs have identical issued commands, final applied controls, measured joint/finger motion and 1,001/1,001 RGB/depth frame pairs. They belong to the same source group. Repeated frames and an identical trajectory are not additional independent training scenarios. This comparison is separate from the stricter data-contract replay checker, which correctly refuses missing tracker channels.

The frozen native task scorer reports successful placement in both full runs with no native solver warnings. The recording-level task result remains unverified. These repeated authored-controller diagnostics are not a learned success rate, held-out test, physical experiment or independently qualified corrective demonstration.

The interrupted capture stops before its third action, retains two complete transitions and marks its outcome unknown. Setup failures and unsealed directories are refused. Development also retained one sandbox graphics refusal and a five-interval smoke; they are listed separately rather than pooled into the registered results. A hard termination can leave an unsealed directory, so capture is not a power-loss-safe service.

## Speed and accuracy have separate limits

Acquisition plus recording write took 13.516 and 13.298 seconds for the two 20-second simulated captures on a shared macOS arm64 host. That timing excludes final receipt and integrity sealing. Eleven and six of their 1,000 measured control boundaries exceeded 20 ms; maxima were 82.8 and 100.3 ms. The simulator pauses for acquisition. These measurements do not establish a hard real-time controller or minimum hardware requirements.

The CGL renderer reported: “depth accuracy will be limited.” The complete warning is preserved with the raw captures. Depth values were finite and byte-consistent; depth accuracy, physical calibration and tracker quality remain unverified. The native rigid block, synthetic contact and fixed camera do not establish shoe deformation or real-world fidelity.

## Download and build the next learner

The separate **0.1.3.dev1 native preview** adds `dvidia-piper-capture run` and `inspect`. Its installed-wheel smoke runs outside the source checkout with isolated Python, verifies five genuine native intervals and 12 frame files, and uses previously verified local robot assets. All 37 focused tests pass, including 27 recorder and 10 native-capture tests. The stable 0.1.2 data-only SDK remains a separate release.

[Download the native preview and complete evidence](https://github.com/Dvidia-Inference/dvidia-training/releases/tag/piper-action-capture-v0.1-preview). [Read the capture guide](https://github.com/Dvidia-Inference/dvidia-training/blob/piper-action-capture-v0.1-preview/docs/piper-action-capture.md). [Inspect the public evidence dataset](https://huggingface.co/datasets/Dvidia/piper-action-capture-v01).

The next experiment needs a frozen camera estimator with measured errors and reviewed corrective action segments. We can then fit and compare a small action policy on genuinely held-out starting groups. Uploading arbitrary human videos, transferring to another arm and reliably executing on hardware remain open.

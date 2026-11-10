# Godot robotics workstation: simulation roadmap

## Purpose and boundary

Build a small desktop workstation that makes robot scenes and physical assets easier to inspect, assemble, and replay. Keep the existing MuJoCo runtime as the single source of simulation truth for training, qualification, and benchmark receipts. Godot supplies the desktop scene editor, asset browser, camera views, and interaction layer; it does not replace or silently approximate the physics authority.

This boundary keeps skill learning central. The workstation should make it easier to create useful training episodes, inspect failure, and run more episodes. It should not turn visual similarity into a claim of physical fidelity. No engine reproduces every contact, flexible material, sensor, actuator, or real-world effect. Physical transfer still requires measured assets, hardware-matched dynamics, and real-robot evaluation.

## v0.1 anchor

The first integration should wrap the existing live workstation behavior: MuJoCo at a fixed 2 ms physics step, 50 Hz differential-IK command updates, the current PiPER profile, and its rigid shoe proxy. Preserve the current controls, initial conditions, reset behavior, contact diagnostics, and receipt fields. Treat Godot as a new front end for this existing scene and command interface.

This milestone is a workstation integration, not a new 1 ms learner scene, a trained manipulation skill, calibrated real-object physics, or evidence of real-robot transfer. Keep the headless MuJoCo CLI as the repeatable training and benchmark path. The desktop interface must not be required for data generation or qualification.

## Architecture

Use one versioned scene manifest as the boundary between editor and simulator. It should identify the robot profile and asset revisions, task, units and axes, object visual and collision assets, body transforms, mass properties, contact model, actuator/controller parameters, sensor setup, domain-randomization distributions, and provenance. Distinguish every value as measured, manufacturer-specified, estimated, or authored. Store uncertainty and measurement conditions alongside values; do not present an authored guess as a physical measurement.

The Godot adapter reads and edits that manifest, displays GLB/USD-derived visual geometry and collision proxies, and sends bounded commands or reset requests to the MuJoCo authority. MuJoCo owns integration, contacts, constraints, actuator response, and authoritative state. Return timestamped state snapshots and diagnostics for display. Keep conversions explicit for meters, Z-up/world frames, joint order, quaternion convention, and radians. Record hashes for scene, assets, profile, code, and controller in every run receipt.

Import visuals and physics separately. A GLB is a useful render asset; it does not by itself establish a safe or accurate collider, mass, center of mass, inertia tensor, friction, compliance, or articulated joint. Prefer simple, named collision shapes for training. Retain the source mesh, scale, transforms, and license, and record how each collider and physical property was obtained. For articulated SimReady assets, represent each rigid body and joint explicitly and verify the compiled model rather than trusting import success.

## Training and real-time execution

Run training headlessly with rendering disabled unless the policy consumes camera observations. When cameras are required, render only the declared sensors at their configured rate and resolution; do not render a desktop viewport per training step. Keep physics time fixed and independent of display refresh. Keep controller update rate, action hold duration, camera exposure/sampling, and policy rate as separate declared clocks. Use integer simulation ticks for scheduling. A slow renderer may delay presentation, but must never change simulated elapsed time or skip control updates.

The desktop view interpolates between authoritative snapshots for smooth display. User actions enter through a bounded command queue and are applied on declared simulation ticks. Reset, pause, single-step, and catch-up behavior must be explicit. If the UI falls behind, show the lag and preserve simulation state; never hide dropped commands or reset history.

## Asset and physical calibration

For each task-critical asset, establish measured or defensible values for mass, center of mass, inertia, dimensions, contact geometry, friction, compliance/stiffness, damping, and relevant actuator or joint behavior. Include latency and observation noise where the training loop depends on them. Record instrument, procedure, units, sample count, conditions, uncertainty, and source. Where measurement is unavailable, label the value as authored or estimated and carry a plausible bounded distribution into domain randomization.

Randomization ranges need evidence and a named purpose. Separate visual appearance changes from physical changes. Record the sampled values and seed for each episode. Randomization can improve robustness to the declared variations; it does not prove that the ranges represent reality or guarantee transfer. Never randomize away a mismatch discovered by calibration without documenting it.

## Qualification gates

Advance in small, reviewable stages. Every gate uses pinned scenes, seeds, and tolerances; preserve failures and per-tick traces.

1. **Scene and asset integrity:** verify hashes, units, axes, transforms, hierarchy, joint mapping, and separation of visual meshes from collision geometry. Reject missing or ambiguous physics metadata.
2. **Simple rigid-body behavior:** compare free drop and settling, sliding, and toppling against MuJoCo reference trajectories. Check contact onset, pose, velocity, support state, and settled time.
3. **Constraints:** add a hinged object and compare hinge limits, constraint motion, and impacts. Validate opening/closing direction and joint effort conventions.
4. **Robot contact:** use a fixed gripper and rigid test object for grasp, lift, hold, release, and slip cases. Check finger/object/table contact ordering, joint motion, force/impulse diagnostics, and task outcome. Add a compliant object only after its model and measurements are available.
5. **Exact scene and command parity:** replay the same initial state and command sequence in desktop and headless paths. Verify scene hashes, action timing, state traces within declared tolerances, reset reproducibility, scorer decisions, and receipt completeness. MuJoCo remains authoritative; a Godot/Jolt-only path would need its own full qualification before it could produce separate benchmark claims.
6. **Performance and capacity:** measure the same workloads with headless simulation, camera rendering only, and the full desktop view. Report hardware and OS, runtime/build versions, scene and camera configuration, warm-up, episode count, simulated duration, and concurrent workers. Measure p50/p95 step and episode time, real-time factor (simulated seconds divided by wall seconds), render overhead, CPU/GPU use, peak memory, and missed-deadline/lag rate. Make no speed claim until these measurements are recorded and reproducible.

Cover representative failure cases at each applicable gate: dropped objects, low/high friction, edge support, toppling, hinge collision, incomplete grasp, slip during lift, release and settle, and robot/table interference. Define success and tolerances before running the confirmation cohort.

## Performance strategy

The optimization objective is the least wall time and memory that keep declared
task-relevant errors below their tolerances. A faster run that changes grasp,
slip, release, or success decisions fails this constraint. Frame rate, physics
throughput, camera generation, and policy-loop latency need separate numbers.

First remove unnecessary work: keep training headless, avoid per-step UI updates, batch state reads, use compact observation buffers, and render only cameras needed by the policy. Profile before optimizing. Keep a measured CPU baseline on the supported desktop hardware and preserve an equivalent deterministic benchmark workload.

An optional batched backend such as Newton or MuJoCo Warp is a later investigation, not an implemented capability. First check official support for the target operating system, CPU/GPU, Python/runtime versions, and required sensor and contact APIs. In particular, do not assume a GPU backend runs on the current Mac CPU baseline. Compare it against the MuJoCo reference on the qualification suite and document any unsupported features or semantic differences before using it for training.

## Release criteria

The v0.1 desktop integration is ready for internal use when the existing live workstation can be opened, reset, controlled, and inspected through Godot while MuJoCo remains the only physics authority; a headless run remains reproducible; and scene/asset provenance plus timing and lag are visible in receipts. Make no claim of improved training throughput until the performance gate passes. Make no claim of physical calibration or transfer until measured asset and hardware evidence supports it. Skill success must come from a frozen task benchmark, not from a visually convincing replay.

## Fast sim-to-real priorities

Optimize elapsed time and compute required to reach a specified **real-task
success rate**, while enforcing separate contact/force/failure limits. Simulation
steps per second and training loss are supporting metrics. They cannot replace
real task outcomes. No hardware transfer rate has been measured for this work.

1. **Reuse a calibrated arm profile.** Once an exact arm/gripper is selected,
   measure encoder zero offsets, base and camera transforms, end-effector geometry,
   command delay, velocity response and loaded tracking. Use instrumented,
   conservative identification procedures with the hardware's native limits.
   Keep calibration versioned and reusable across skills. The current prototype
   issues no physical commands and has no hardware identification implementation.
2. **Identify only task-critical object properties.** For rigid block placement,
   start with dimensions, mass, table/finger friction, compliance and contact
   geometry. Use simple drop/slide/grip observations with known conditions; retain
   uncertainty and distinguish identifiable properties from confounded ones.
   Appearance-only footage cannot determine all these values. Deformable shoes
   require a separate material model and measured deformation/contact tests.
3. **Train across measured uncertainty.** Start from demonstrated robot actions,
   then vary starts, contact parameters, latency, calibration error and observation
   dropout within declared ranges. Add the specific failures seen in development.
   Freeze held-out tests before tuning. Use short action predictions plus current
   feedback so a changing grip can interrupt the next movement; do not treat a
   long predicted action sequence as permission for blind execution.
4. **Scale useful rollouts.** Keep the Godot viewport optional; render only policy
   cameras at their declared rates. Reuse compiled models, buffer observations,
   and profile data loading, rendering, inference and physics separately before
   adding parallel workers. Evaluate any new CPU batch API or GPU backend on a
   separate versioned experiment; do not upgrade the pinned 3.15.0 benchmark in
   place. A host's supported parallelism and memory, not its core count alone,
   determine the worker budget.
5. **Deploy a compact local policy with explicit checks.** Preserve units, frame
   conventions, action rates and calibration identity between simulation and the
   adapter. Measure complete observation-to-command p95/p99 latency and stale
   sensor behavior. Apply independent joint/rate/contact checks supported by the
   actual sensors; motor current is not automatically a calibrated force sensor.
   Slow near uncertain contact and verify grip before transport. Local software
   checks do not replace the arm's physical safety mechanisms.
6. **Close the loop with reality.** Once hardware is available, use a fixed budget
   of controlled real trials to compare tracking, grip/retention, placement,
   recovery and failures against the same simulated cases. Fit physical parameter
   distributions from calibration/development data, then evaluate on held-out
   trials. Update reusable profiles rather than repeatedly hand-tuning each skill.

The immediate learning bottleneck remains acquisition/progress: the frozen
camera-to-action candidate achieved 5/64 strict simulated placements, and the
two reference-action development candidates achieved 0/4 each. The Godot adapter
does not change those scores. Complete the separately specified acquisition and
grip-progress experiment before treating additional simulation throughput as
evidence that the learned skill is ready for a physical robot.

This strategy follows the idea of updating simulation distributions with real
observations in [SimOpt](https://arxiv.org/abs/1810.05687). The effectiveness and
required real-data budget still need measurement for our arm and task. Action
sequence models such as [ACT](https://arxiv.org/abs/2304.13705) are a relevant
baseline to compare; their published results do not establish our performance.

## Primary references

- [MuJoCo overview](https://mujoco.readthedocs.io/en/stable/overview.html): articulated contact dynamics, actuator models, and separation of model and state for parallel sampling.
- [Godot Jolt integration](https://docs.godotengine.org/en/stable/tutorials/physics/using_jolt_physics.html): joint-interface and reported multi-body contact-impulse limitations relevant to a future Jolt comparison.
- [SimReady GLB dataset card](https://huggingface.co/datasets/codepawl/simready-glb): separate engine-format visuals, physics metadata, collision hulls, source rights and documented conversion limitations.
- [Newton project](https://github.com/newton-physics/newton): optional GPU backend research; its documented macOS path is CPU-only. It is not installed or used by this prototype.

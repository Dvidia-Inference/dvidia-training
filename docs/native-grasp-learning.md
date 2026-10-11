# Next experiment: a native grasp action policy

This is a proposed learning experiment after the native rigid-block grasp/carry/
release benchmark. It reports no newly trained policy, learned task success,
minimum hardware requirement or physical qualification.

The first useful result is a small DVIDIA policy choosing the complete sequence
of native arm and jaw commands from causal observations, with success measured
in fresh closed-loop simulations. The authored baseline establishes the task and
its contact-dependent failure controls before a learner is compared with it.

## Freeze the task before fitting a model

Use the pinned UR5e/Robotiq reference and an independently moving rigid block.
Object movement must result from native dynamics and contact. Do not attach,
teleport or write the block pose during a rollout. Freeze geometry, collision
pairs, mass/friction, controller settings, command limits, timestep, 20-ms
command cadence, initial/goal states, timeout and success predicates.

Success must include acquisition, lift, retention during carry, release and
stable placement with tool clearance. A reach or a transient visit to the goal
does not complete the task. Report these intermediate outcomes separately.
Record contact/drop evidence at native ticks so short events between command
boundaries remain visible.

Retain all declared attempts, including failures, validation errors and native
warnings. Save the nominal issued joint commands and replay exactly those arm
commands with the jaw open as the matched negative control. A controller that
replans its arm movement after failing to grasp is a different intervention.
Numerical convergence and valid contact-dependent task outcomes are separate
gates; unloaded timing does not establish loaded throughput.

## Observations and actions

Keep an explicit versioned observation contract. Proposed first features are:

- Six ordered joint positions/velocities in radians and radians per second.
- Named gripper driver position/velocity, retaining the coupled linkage profile.
- Current tool pose and block/goal relative poses, with declared frames and
  rotation representation.
- Previous issued native commands, per-feature acquisition age/availability and
  a history-validity mask.

All of these begin as **privileged native simulation observations**. Simulated
pose, contact and force fields must retain that origin. A later camera/encoder
adapter must prove that it can supply the trained features; it cannot silently
replace simulator truth with an unvalidated estimate. Known mass/friction,
authored phase, scene/case ID, future state, terminal outcome and teacher IK goal
must not enter policy inputs. Optional contact/velocity features need named
ablations and an explicit sensing contract.

The model predicts six issued position targets in radians and one native
Robotiq command in **0–255**. If joint deltas are predicted, store how they are
reconstructed against the current joint state and bounded. The reference's eight
jaw linkage coordinates are not eight independent commands. Pad-center distance
is not a calibrated inner jaw gap. The older authored profile's metric-gap
transition model cannot be reused as this action policy.

Training labels are the teacher's issued requests. Applied servo forces, next
measured positions and post-step outcomes are distinct records. Preserve any
limiter/governor intervention and do not relabel it as unmodified model output.
The low-level servo and explicit safety governor may remain authored, but no
authored task-phase, contact, grip or teacher-motion fallback may choose the
learner's task actions unnoticed.

## Small local models to compare

Start with a DVIDIA behavioral-cloning MLP with two hidden layers of 64 units and
an eight-boundary causal window: 160 ms at the proposed 50-Hz command cadence.
Compare it with a single-boundary MLP and a four-action chunk head that replans
each boundary and initially executes only its first action. These are proposed
compact designs, not implementations or measured reproductions of ACT. If
release/retreat states remain ambiguous, compare an explicitly reset recurrent
state before introducing privileged phase information.

Fit normalization and model weights on successful nominal training attempts.
Retain failed nominal/intervention attempts in the complete corpus manifest;
they may support separately declared outcome/monitor learning. They must not
quietly become successful imitation labels or disappear from data denominators.
Normalize joint-radian and native 0–255 jaw targets separately. The authored
benchmark runs to a fixed timeout after first successful dwell, so predeclare a
moving/contact/release sampler or an idle-tail exclusion based on the independent
first-success measurement. Keep the complete original tape and report excluded
or reweighted training rows. Sampling labels and final outcomes are offline
curation metadata, never deployment inputs; history windows stay within episodes.

Use source/reset-group partitions; all cuts, replays and sibling interventions
stay together. Freeze training, development and confirmation IDs before fitting.
Select architecture and checkpoint using development rollouts only. Freeze the
selected model, processors and protocol before confirmation; report every
confirmation attempt. Random frame splits do not establish episode independence.
Simulation reset groups are not independent physical sessions.

A proposed 8/16/32-group learning curve can assess workload and sensitivity to
collection order on development cases. These counts do not establish how many
human clips a skill needs. Fresh sealed confirmation evaluates the selected
candidate, after the development curve and model selection have finished.

## Decisive held-out cases and metrics

Declare reachable new object/goal poses, permitted new size/mass/friction
combinations, small grasp offsets, prescribed carry disturbances, matched open
jaws, release after lift, unreachable/infeasible tasks and missing/stale input
cases. Keep nominal, stress, recoverable and impossible/fault strata distinct.
Never turn an impossible task or sensor fault into an invented successful
placement. Define whether each condition expects recovery, controlled abort or
refusal before running it.

Report complete placement and intermediate phase rates with all-attempt
denominators and group counts; placement errors and duration; retention drift,
slip/drop and false completion; unplanned contact; requested/applied command
limits and limiter intervention; numerical warnings/invalid outcomes; and every
fallback. Offline action loss is a diagnostic rather than a mastery score.

Measure cold load, warm policy p50/p95/p99/max, peak memory, full loop work and
20-ms deadline misses on named CPU/runtime hardware. Separate acquisition,
feature construction, inference, postprocessing, recording, physics and render
costs. A proposed warm policy p95 below 2 ms is a compute target, not an achieved
measurement, a sensing-to-actuation budget or a hard real-time guarantee.
Compare loaded outcomes and trajectories at prescribed physics timesteps with
the same commanded boundaries; retain runtime/architecture and model hashes.

## What an installable candidate contains

Package bounded data-only weights, normalization, history/reset rules, ordered
observation/action units and frames, named hardware/controller profile, runtime
versions, source/dataset/protocol hashes, limits and failure behavior, licenses
and the frozen all-attempt receipt. A candidate requires an explicit compatible
adapter and qualification record. Keep `physical_robot_ready: false` until
separate physical evidence supports a different release.

Human videos supply useful task, object and goal evidence. Action learning also
needs synchronized robot demonstrations or a reviewed metric trajectory/jaw
reconstruction with uncertainty, kinematic feasibility and clock alignment.
UMI is a primary precedent for that bridge. A rigid block does not model shoe
uppers, loose laces, sole compliance or varied grasp geometry. Shoe organization
and five-finger/bimanual control require their own data, task contracts and
benchmarks. A successful seven-command block artifact cannot qualify them.

## Primary references

- [robomimic algorithms](https://github.com/ARISE-Initiative/robomimic/blob/master/docs/introduction/implemented_algorithms.md)
  and [offline-demonstration study](https://arxiv.org/abs/2108.03298) for BC
  baselines and evaluation sensitivity.
- [ACT](https://tonyzhaozh.github.io/aloha/) for action-chunking motivation.
- [UMI](https://umi-gripper.github.io/) for reconstructable action demonstrations
  and latency-aware interfaces.
- [LeRobot custom policies](https://huggingface.co/docs/lerobot/main/en/bring_your_own_policies),
  [dataset format](https://huggingface.co/docs/lerobot/main/en/lerobot-dataset-v3)
  and [actual cadence](https://huggingface.co/docs/lerobot/main/en/il_robots#cadence-reporting)
  for independently packaged models, temporal data and clock evidence.
- [MuJoCo computation](https://mujoco.readthedocs.io/en/stable/computation/)
  for contact, state-computation and reproducibility limits.
- [Current hardware reference](hardware-reference.md),
  [episode tapes](episode-tape.md) and [laptop pilot](laptop-skill-pilot.md)
  for the contracts and limitations already implemented by DVIDIA.

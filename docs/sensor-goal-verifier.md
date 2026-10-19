# Camera and actuator corroboration

The v0.4 layer checks whether independently processed camera pixels and drive
readings contradict a primary physical-goal packet. The existing v0.3 verifier
still checks that packet's explicit placement, motion, support and release
requirements. A primary `complete` now also needs compatible camera and drive
evidence. Missing, stale, invalid, obscured or contradictory auxiliary evidence
returns `unknown`. Primary `incomplete` and `unknown` results remain unchanged.

This is a synthetic corroboration experiment. All channels originate in one
MuJoCo simulation. They have separate processing paths, but are not physically
independent sensors or authenticated sources. The checks do not command a robot.

## Inputs and limits

The trusted caller supplies the goal, clock and calibration. Camera packets
contain eleven 640×480 RGB frames sampled every 20 ms over 200 ms. A small CPU
observer finds one green elongated candidate, estimates its XY centroid and
axis, and compares it with the primary packet at matching timestamps. It gets
pixels only: no simulator body identities, position, segmentation or truth labels.

The camera's declared XY radius is 41.875 mm, larger than the 25 mm goal tolerance.
Compatibility therefore corroborates a supplied position; it does not certify
precise placement from vision. Position intervals propagate into displacement
and average-speed intervals. Broad bounds can reveal large motion but cannot
prove zero or instantaneous speed. A covered or missing candidate is unknown,
not a positive observation that the target is absent.

One hundred drive samples arrive at 2 ms intervals. They contain arm and finger
joint positions and velocities, servo commands, and measured simulated actuator
efforts. The release checks require the entire uncertainty interval to satisfy:

- Jaw aperture at least 60 mm and opening command at least 30 mm per finger.
- Absolute drive effort no greater than 0.10 N and aperture rate no greater than 10 mm/s.
- Servo force consistency and finger symmetry within the declared bounds.

Those are thresholds for this particular simulated PiPER/proxy configuration.
Actuator effort is drive output, not total target contact force. It does not
identify what is loading the gripper. The RGB and actuator paths cannot establish
table support, authenticated shoe identity, a 180-degree axis reversal, or the
absence of motion between camera samples. Self-consistent replay of all channels
can still pass. Completion remains conditional on the primary assertions.

## Reproduce the benchmark

Use Python 3.12+, the project's pinned NumPy, and a CPU for evidence evaluation.
Native scene generation additionally needs pinned MuJoCo/PiPER assets and a
working graphics context. No API key, downloaded model weights or dedicated GPU
is required for the observer. No measured minimum hardware specification is claimed.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[arm]'

# Provision the pinned PiPER assets through docs/robot-simulation-library.md.
dvidia-sensor-goal generate --assets /absolute/path/to/robot-profile-upstream/assets \
  --output runs/sensor-goal-fresh --seed 2026102004 --count 8
dvidia-sensor-goal evaluate --run runs/sensor-goal-fresh
python -m dvidia_training.sensor_goal_verifier_report --help
```

Generation requires committed sources. It freezes the source fingerprints,
protocol and seeded layouts before generating 96 native states, 80 primary
forgeries, 64 auxiliary evidence faults and 32 cross-channel controls. Evaluation
refuses changed inputs or source and writes one result for all 272 scheduled
cases. A hard generation failure seals diagnostics and stops scoring. Valid
category mismatches remain in the denominator. No outcome-based retries or
filtering are permitted. Use the run's frozen source commit to reproduce it.

The v0.3 baseline and v0.4 gate receive exactly the same primary packet and
decision time. Native physical truth remains a separate frozen measurement of
the simulator. Forged packets retain their actual underlying physical label.
The report publishes false completions prevented, true completions lost,
abstentions, and residual false completions by cohort. Returning unknown is a
refusal to confirm, not proof that the task was incomplete.

Every consumed image, packet and native state is sealed and downloadable. The
reader rebuilds decisions from these files before publishing. Timing covers
file reads, decoding, RGB observation, drive observation, baseline and fusion
separately and together. Repeats may use a warm OS file cache. The 200 ms acquisition
window, rendering, network and physical actuation are excluded from evaluation
latency; native generation wall time is reported separately. This is not a
hard-real-time robot response measurement.

Original DVIDIA code is MIT. Existing runtime and asset licenses remain in
force; see the repository licenses and pinned asset notices. This component
does not redistribute Jev weights or call a hosted provider.

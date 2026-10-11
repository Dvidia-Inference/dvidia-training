# PiPER native contact pilot

This pilot adds an authored rigid-block grasp, carry and release task to the
pinned PiPER CPU profile. It uses simulator state, not camera observations or a
learned policy. DVIDIA owns the task/controller/benchmark; the original MIT robot
model, servos, force limits and gravity compensation remain attributed upstream.

The action is six native joint-position targets in radians plus one finger-slide
target in metres. Both actual finger slides are recorded. An equality constraint
couples the opposite finger; the issued target does not establish measured jaw
opening or grip force. The test block is a free body under gravity and contact.
There is no grasp weld or post-reset object pose assignment.

## Run locally

Install the candidate source with its `[arm]` optional dependencies, then fetch
PiPER assets once. Subsequent benchmarks run on a CPU without network access.
Use a fresh output directory for every run. This is the candidate research source
on `feat/reference-task-benchmark`; it is newer than the Observation Lab release.

```sh
.venv/bin/dvidia-robot-sim fetch piper --output runs/robot-assets
.venv/bin/dvidia-piper-task benchmark runs/robot-assets --output runs/piper-contact
.venv/bin/dvidia-piper-task inspect runs/piper-contact
```

The protocol freezes scene configurations and source identities before the first
attempt. Each scene retains placement, the exact placement arm targets replayed
with the jaw open, and a hold control. Failures and unavailable matched controls
remain in the declared denominator. The full timeout is retained even after an
early success event.

## What counts as success

The scorer measures the free object independently of the authored policy phase:

- Sustained bilateral contact while lifting clear of the support.
- Contact retention during transfer until supported near the destination.
- Actual opening of both fingers, no remaining pad contact, a settled destination
  dwell and tool retreat.
- Valid native physics and no disqualifying unexpected contacts.

A changed wrist orientation during carry is declared in the configuration. The
workspace and support height are part of the task, not a claim about an arbitrary
floor or shoe rack. The block is rigid; no shoe deformation or laces are modeled. This pilot scores
destination position and settled motion; it does not require a target object orientation.

These small synthetic cohorts do not measure physical accuracy, general task
reliability, learned skill, perception latency or hardware safety. A numerical
success is a useful test case for an eventual DVIDIA policy and hardware
comparison; it is not permission to execute the tape on a physical robot.

# Local physical-goal verification

The v0.3 verifier turns a trusted placement goal and timestamped observations
into `complete`, `incomplete` or `unknown`, with a reason for each condition.
It runs locally on a CPU without a provider key. It does not infer object pose,
identity, speed or contact force from an image: those measurements must come
from a separate simulator or calibrated sensing layer.

The previous Jev diagnostic recognized only one of eight completed states.
Offline inspection ruled out missing telemetry, a parser error and a client
completion cutoff. Three visible positives were returned as unknown; four
covered positives were returned as incomplete. The model's internal cause is
unknown. This follow-up moves explicit physical checks into local code rather
than asking a model to interpret numeric requirements reliably.

## Contract

A caller supplies `GoalContract(object_id, frame_id, goal_xy_m, goal_yaw_rad)`
and a trusted current clock. The observation packet cannot set its own goal,
thresholds or verdict. Every sample identifies its object and coordinate frame.
Position is metres, yaw radians, speeds m/s and rad/s, contact forces newtons,
and timestamps seconds on the same clock as the caller.

The latest 100 samples, at a declared 2 ms cadence, must establish:

- Target presence throughout the window.
- XY error no greater than 25 mm and yaw error no greater than 15 degrees.
- Linear speed no greater than 10 mm/s and angular speed no greater than 0.05 rad/s.
- Table support force greater than 0.01 N and robot contact no greater than 0.01 N.

The newest sample must be at most 100 ms old. Missing, malformed, stale,
future, irregular or insufficient history returns unknown before physical
scoring. A valid definite physical violation returns incomplete. Otherwise,
uncertainty overlapping a threshold returns unknown; all intervals passing
returns complete. Absolute uncertainty bounds are mandatory. Zero bounds in
this native benchmark explicitly assume exact simulator state, not noiseless
hardware. Unknown target presence returns unknown; a measured absence returns
incomplete.

The code validates the full packet (at most 500 samples), then scores only the
latest 100. Each timestamp ends a 2 ms interval. This sampled contract does not
prove what happened between samples. It also cannot authenticate a self-consistent
fabricated sensor stream. Freshness, identity and confidence declarations are
useful only when their sources and clocks can be trusted.

## Run locally

Use Python 3.12 or later. The verifier itself imports only the standard library
and needs no GPU. Native benchmark generation additionally uses the project's
pinned NumPy/MuJoCo dependencies, pinned PiPER assets, and a working graphics
context. These are development prerequisites, not a measured minimum hardware
specification. Full CPU and runtime details are retained in each result.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[arm]'

# Provision the PiPER assets using the existing robot profile workflow.
# See docs/robot-simulation-library.md for pinned assets and third-party licenses.
dvidia-local-goal generate --assets /absolute/path/to/robot-profile-upstream/assets \
  --output runs/local-goal-fresh --seed 20261019 --count 8

dvidia-local-goal evaluate --run runs/local-goal-fresh
```

Generation writes the protocol before producing scenes, then seals all input
files. Evaluation refuses changed inputs or changed benchmark/verifier sources.
It refuses to overwrite existing results. Retain the source commit and input
seal when comparing runs. A verifier execution can also be used alone:

```python
from dvidia_training.local_goal_verifier import GoalContract, verify_goal

contract = GoalContract('green-shoe', 'workstation', (.4, -.1), 0.)
result = verify_goal(packet, contract=contract, now_s=clock_now)
print(result['verdict'], result['reasons'])
```

`contract_description()` documents the exact packet fields, bounds and time
semantics. The CLI `dvidia-local-goal verify --packet observations.json
--contract goal.json --now-s 0.6` checks a stored packet without MuJoCo.

## What the benchmark measures

Eight fresh seeded layouts each have eight native cases: complete, missing
target, wrong object, wrong position, wrong orientation, unreleased, moving
alias and occluded. Native physical truth comes from the frozen simulation
measurement code independently of the verifier. A moving alias has the same
rendered pose as its completed control but an explicitly assigned velocity.
These diagnostic states do not demonstrate a successful placement controller.

Sixteen input perturbations per layout exercise evidence faults. Their underlying
physical states remain complete; a conservative unknown on bad evidence is
expected. A fabricated contact spike instead supplies a definite failure.
Eight additional controls pair a physically moving state with a self-consistent
complete-state sensor stream. Those must remain visible as physical false
completions even when the packet verdict agrees with the supplied evidence.

Report native scenes, evidence faults and consistent forgeries separately.
Packet-contract agreement is not perception accuracy. Overall physical metrics
include every cohort, so a favorable native score cannot hide forged-sensor
failures. Category mismatches with valid packets remain in the scored inventory. A hard
scene-construction failure seals a failed-run manifest and stops evaluation; it
never produces a partial success score. One default-seed
layout was inspected during an initial complete/moving-alias unit check without
parameter tuning; the full development preflight uses a different seed. These
are fresh generated layouts, not a claim of a fully unseen test set.

Repeated timing measurements surround only `verify_goal`, with parsed data
already in memory. They exclude the required 200 ms observation accumulation,
sensing, I/O, rendering and actuation. Do not compare that number directly with
the earlier Jev full online-request latency or call it end-to-end robot reaction
time. No new Jev requests are needed for this benchmark.

## Integration direction

Use an image model to propose object identity, geometry and a task goal. Obtain
pose, motion, support and release evidence through a calibrated sensing path.
Use the local verifier to decide whether that explicit goal is established.
A separate controller handles movements, recovery and independent stop logic.
The next useful benchmark varies estimated pose/force error and calibration
against independent physical truth. It must measure false completion under
sensor faults before connecting the verifier to real actuators.

Original DVIDIA code is MIT. Existing asset and runtime licenses remain in force;
this component does not redistribute Jev weights or imply a license to them.

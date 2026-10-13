# Skill prerequisites: two small simulations

The next step toward an installable skill is an explicit account of when the
action is possible, what can go wrong, and what a recovery actually achieves.
This pilot uses a native MuJoCo PiPER arm in two controlled environments:

- A hinged shoe box tests a moving lid, a gripper in its closing path, and a
  bounded withdrawal or retry.
- An out-of-reach shoe tests whether a reachable tool can change the scene so
  that the shoe enters the arm's usable workspace, followed by a separate
  retained-lift check. The rake is pre-mounted; the controller does not discover
  or acquire it. The no-tool baseline is a capability control, not a policy-only
  comparison.

These are authored controllers and synthetic mechanical assumptions. They do
not contain learned weights, train a robot from footage, estimate properties
from a camera, or qualify a physical robot. The purpose is to make those later
comparisons executable and measurable.

## One small contract

Each environment should expose the current observation, accept a bounded robot
command, advance native physics, and return the resulting observation and
events. The outcome scorer should remain separate from the policy. A visible
controller phase is useful for inspection but is not evidence of task success.

Record the observation, the requested command, what was applied, and the next
observation. Keep interventions and human hints distinct from autonomous
decisions. Mark simulator ground truth as such; a contact force supplied by
MuJoCo is not a physical tactile measurement.

The scene owns object motion. After reset, policies must not teleport the shoe,
weld it to the arm, or apply an undisclosed object force to manufacture success.
Any attached tool, pre-established grasp, or externally triggered lid release
must be stated as part of the initial conditions or intervention protocol.

## What the benchmark must preserve

| Question | Evidence |
| --- | --- |
| Did the task finish? | An independent geometric and temporal completion condition, not a policy phase name. |
| Was a hazard avoided? | No disallowed contact over the relevant physics steps; ending outside the hazard is insufficient. |
| Did a recovery work? | A recorded failure or risk trigger, the issued recovery, and the subsequent measured result. |
| Was the tool necessary? | Matched direct-action or no-tool controls, with the same initial scene and task deadline. |
| What remains impossible? | Unsuitable-tool or unreachable conditions retained as failed tasks or explicit deferrals. |
| How fast was it? | Simulated task time, computation time and policy decision time reported separately. |
| Does it generalize? | A frozen confirmation cohort with new parameter combinations after development tuning ends. |

Success, recovery, avoidance and correct refusal are different outcomes. A
refusal can be appropriate while still counting as an unfinished task. The
publication should show both, without increasing the completion denominator
after observing the results or removing difficult cases.

Brief contacts need physics-step aggregation or event records. Recording only
the final contact at each control interval can miss an impact between samples.
Report peak normal force, contact duration or impulse where implemented, with
their units and sampling interval. They are simulated contact quantities, not
validated injury thresholds or measured material damage.

CPU measurements must identify the tested machine, engine version, timed
region, cold versus warm setup, and timing tails. Physics timestep is simulated
time; it is not a measured camera-to-actuator latency. No universal minimum
hardware requirement follows from one successful laptop run.

## Evidence before release

Freeze the scenario matrix and implementation hashes before the confirmation
run. Keep every planned attempt, failure and computation overrun. Bind the
results to a source revision, engine version, robot asset revision, scene
parameters and licenses. A new policy tuned after that run needs a fresh
confirmation cohort; the old failures remain evidence.

The new release belongs in its own versioned namespace. GitHub should contain
the runnable code and compact benchmark receipts. Hugging Face and the research
site should link to the same frozen artifacts, with a clear distinction between
a recorded replay and a live physics environment. Old workstation results and
their licenses remain intact.

Original DVIDIA code and synthetic assets are MIT. Redistributed PiPER model
assets retain their upstream MIT notice. MuJoCo retains Apache-2.0 and its
dependency notices. Any replay renderer and bundled font need their respective
notices. Research writing on the DVIDIA research site is CC BY 4.0.

## Run fresh native attempts

Use the exact source revision identified by the publication. From that checkout:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install '.[arm]'
.venv/bin/dvidia-robot-sim fetch piper --output runs/robot-assets
.venv/bin/dvidia-prerequisites --assets runs/robot-assets \
  --output runs/my-prerequisites --split development
.venv/bin/python -m http.server 8000 --bind 127.0.0.1 \
  --directory runs/my-prerequisites
```

Open `http://127.0.0.1:8000/` to inspect the recorded replay. The benchmark has
already run in native Python; moving the replay timeline does not run physics.
Use `--task hinged-box` or `--task tool-reach` to run just one environment.
`--split heldout` selects the frozen parameter holdouts. A fresh run is a
reproduction of that cohort, not new independent confirmation evidence. Use
`--source-commit FULL_GIT_SHA` to identify the checked-out source in your receipt.
The output directory must not already exist.

On Windows use the virtual environment's `Scripts` executables and place each
continued command on one line. Installing dependencies and fetching the pinned
arm assets needs a network connection. Native simulation and replay can run
locally after those files have been provisioned. The replay needs WebGL2; the
native benchmark needs the optional MuJoCo dependency and a CPU. These are
dependency requirements, not measured minimum hardware specifications.

The export contains a small `benchmark.json` index, the frozen `protocol.json`,
every full trial JSON, geometry and recorded motion, local viewer code, license
notices and `integrity.json`. Dense native contact traces stay in the full trial
receipts so that the initial viewer does not download every physics tick.
Failed attempts remain in the index and the sealed inventory.

Measured outcomes and immutable publication links will be added after review.
The protocol above does not itself claim successful task completion.

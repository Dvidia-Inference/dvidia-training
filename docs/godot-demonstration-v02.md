# Record demonstrations locally

This prototype connects a new Godot demonstration UI to the existing PiPER
rigid-block learning scene. A person can command XYZ and gripper opening,
record paired observations/actions, inspect the attempt and export selected
rows for the existing CPU learner. The tool orientation is fixed.

This release implements capture and export. It does not add hand tracking,
physical teleoperation, a Train button, a newly trained policy or a qualified
skill. The existing learned placement result remains 5/64. The older Godot
rigid-shoe workstation and all previous study evidence remain unchanged.

## Run

Use Python 3.12+, the pinned NumPy/MuJoCo dependencies, Godot 4.7.2 and a working
native graphics context. The tested host is Apple M5/macOS; other platforms
require their own rendering checks. Installation alone does not include Godot
or the separately licensed arm assets. Keep the checkout for the Godot project.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[arm]'
dvidia-robot-sim fetch piper --output ./robot-assets
dvidia-demonstrate prepare --assets ./robot-assets --output ./demo-scene
dvidia-demonstrate serve --assets ./robot-assets --scene ./demo-scene --output ./demo-session
```

In a second terminal, replace the scene path with your absolute directory:

```sh
godot --path integrations/godot-demonstration -- --scene /absolute/path/to/demo-scene --port 8431
```

Choose the target and press **Record**. Move the XYZ and gripper controls;
**Pause** holds simulation time, **Finish** seals the current prefix, and
**New episode** preserves the previous recording and resets into a new directory.
Home changes the target, without resetting or erasing the recording. The
default budget is four episodes, each at most 20 simulated seconds. Finishing
early produces an interrupted recording; recording completion and task success
are separate. Closing the client retains the committed prefix.

The service listens only on localhost, accepts one client and exits on its
disconnect. No physical commands or cloud service are used. Provisioning the
dependencies/assets requires networking; subsequent capture and export are local.
The capacity guard reserves 10 GiB plus the declared capture budget. This small
capture service does not override the older full study's separate disk guard.

Use `prepare --case case.json` for explicit starts and either existing destination.
The file has `id`, `group_id`, `seed`, `object_position_m`, and `goal_position_m`.
Changing seed alone does not change the object position. The default case is
the original block at [0.30, -0.03, 0.082] and goal [0.35, 0.08, 0.08] metres.
Keep related starts, retries and corrections in the same train/development/test
group. A different case needs its own prepared scene and session.

## Inspect and export

```sh
dvidia-demonstrate inspect ./demo-session/episode-0001
dvidia-demonstration-export --capture ./demo-session/episode-0001 --review ./review.json --output ./reviewed-data
```

Example `review.json`:

```json
{"segments": [[0, 50]], "reason": "Reviewed useful approach segment, before the unsuccessful grasp."}
```

Segments are half-open control-row ranges; `[0, 50]` selects one simulated second.
Alternatively supply `indices` and `reason`. Duplicate or overlapping selections
are rejected. Selection is an unauthenticated review assertion, not an automatic
quality certification. Preserve failures, but only select actions that are useful
for the intended learning experiment. The exporter does not train automatically.

`operator_export.load_dataset(path)` returns the existing camera-action dataset
contract, suitable for its Python training API. It verifies the saved arrays and
provenance. UI/Terminal captures use the same data; the existing fixed-budget
study orchestrator does not silently accept these as authored-teacher episodes.
A new human-demonstration study still needs a registered split, fit budget and
held-out evaluation before a model can be presented as a useful skill.

After an abrupt process loss, `dvidia-demonstrate recover PATH` reads the
durable committed prefix without changing the directory. It never resumes an
uncertain native interval, fabricates task success or produces a training export
from an unsealed attempt. `inspect` is for normally sealed attempts.

## Data and clocks

- MuJoCo alone advances physics: twenty 1 ms ticks per 50 Hz control interval.
- Native RGB-D is 320x240 at 25 Hz, with calibration and frame hashes. The Godot
  orbit view is operator visualization, not the learner's camera.
- Missing perception stays missing. Historical observations retain acquisition
  times and masks. Future frames, native object/contact truth and current action
  labels are excluded from learner inputs.
- The journal distinguishes input requests, the issued seven-channel command,
  the final applied reference and measured joints. The gripper command is one
  finger-slide target in metres, not measured full aperture or force.
- A separate encoder-only FK/IK component maps the input target to joint
  commands. It receives measured joints, target and jaw, never object/contact
  truth. Original limits and native rate limits remain active.
- Native evidence independently checks grasp/lift/retention/release/settling/
  retreat. A successful recording can contain a failed task.

The source kind is `simulation_human_input` by default, a producer declaration
with `human_authenticated=False`. Automated checks explicitly use
`simulation_scripted_input`. Neither is a physical teleoperation recording.
Every package remains `physical_robot_ready=False`.

Simulation timestamps and monotonic wall timestamps are separate. A slow camera
or disk slows simulation progress; it does not skip physics ticks. The software
does not establish a 50 Hz wall-clock deadline or real-robot reaction latency.
Capture includes durable writes, rendering and perception costs that the older
unrendered throughput benchmark excluded.

## Verify

```sh
python -m unittest tests.test_operator_capture tests.test_operator_export tests.test_godot_demonstration_bridge -v
godot --headless --path integrations/godot-demonstration --script tests/test_coordinates.gd
godot --headless --path integrations/godot-demonstration --script tests/test_recording.gd
python scripts/verify-operator-replay.py --assets ./robot-assets --capture ./demo-session/episode-0001 --output ./native-replay.json
```

The native replay script independently executes issued commands and compares
dense native arrays plus the independently reconstructed task score. It measures
reproduction of simulation behavior, not successful manipulation or sim-to-real.
Use `--source-kind simulation_scripted_input` when serving a scripted Godot smoke
or TCP lifecycle check; never label those runs human demonstrations.

No new learning result is claimed by this release. The next milestone is genuine
operator recordings across starts and goals, followed by the bounded, separately
evaluated learning experiment.

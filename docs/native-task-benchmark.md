# Native grasp task benchmark

This source alpha adds a free rigid object and a placement task to the pinned
UR5e / Robotiq 2F-85 assembly. It is the next test environment for DVIDIA's
installable-skill pipeline. The initial controller is authored and reads
privileged simulator state. This release does not learn a task from footage or
control a physical arm.

## Run on a local CPU

Use the `feat/reference-task-benchmark` source branch until this component is
released. Python 3.12+, NumPy 2.5.3 and MuJoCo 3.15.0 are required. Provision the
[pinned reference assets](hardware-reference.md) once; subsequent simulation,
inspection and replay export need no account, API, model download or network.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install '.[arm]'
.venv/bin/dvidia-embodiment fetch-assets --output runs/reference-assets
.venv/bin/dvidia-reference-task benchmark runs/reference-assets \
  --output runs/grasp-development --seed 411000 --nominal 4 --stress 2
.venv/bin/dvidia-reference-task inspect runs/grasp-development
.venv/bin/dvidia-reference-task export-replay runs/reference-assets \
  runs/grasp-development --output runs/grasp-replay
.venv/bin/python -m http.server 8290 --bind 127.0.0.1 --directory runs/grasp-replay
```

Open `http://127.0.0.1:8290/` for the recorded 3D task. Use
`.venv\Scripts\COMMAND.exe` in Windows PowerShell and put continued commands on
one line. Help and artifact inspection do not require the optional native
simulator. From a checkout, use `PYTHONPATH=src .venv/bin/python -m
dvidia_training.reference_task_benchmark` in place of the installed command.

Headless simulation needs no GPU or renderer. The browser needs WebGL2 for
viewing; rendering cost is excluded from native CPU timings. RAM and minimum
hardware guarantees require measurements on each supported workload. The new
source component is separate from the earlier tagged release and Hugging Face
Observation Lab demonstration.

## What the task asks

Acquire one free rigid block from a support table, lift it clear, carry it to a
different marked position, open the fingers, retreat and leave the object
settled. Object movement comes from native contact and gravity. No weld, direct
object-force controller or object-position assignment carries it through the
episode. A full six-axis pose solver controls both tool position and orientation.

The reference retains the arm's six hinge coordinates and the gripper's eight
coupled hinge coordinates with seven actuator commands. The block adds a free
body, with three position and four quaternion coordinates and six velocity
components. Arm targets use radians; the gripper uses its native 0–255 command,
not meters or calibrated fingertip force. The task has its own profile and
compiled-model identity. Older simplified-arm capsules are incompatible.

The success evaluator is independent of the controller's phase. It requires a
sustained lift, arrival near the target, release from the pads, retreat clearance
and a dwell of settled object state. After the sustained lift, losing bilateral
pad contact for 40 ms before supported delivery at the target disqualifies the
attempt. A later numerical warning also invalidates an earlier successful dwell.
A controller announcing completion cannot
make those measurements true. Runs retain the complete declared timeout,
including what happens after the first successful dwell.

The existing reference asset's geometry, inertia, collision masks and exclusions
remain the basis of the model. This does not calibrate its motor firmware,
mounting, pad friction, force, backlash, compliance or serial-specific dimensions.
The rigid block and authored material values are a bounded contact fixture;
they do not establish shoe deformation, sole friction, laces or camera sensing.

## Baselines and failure accounting

Each scenario runs three controls with the same physical reset and time budget:

1. An authored grasp/carry/release controller.
2. The recorded issued arm targets from that attempt, with the jaws requested
   open. This checks whether the commanded arm path alone suffices.
3. An initial-pose hold, which checks accidental task completion after reset.

Nominal scenarios vary the initial and target planar positions. A separate
low-friction stratum changes the declared contact coefficient. Scenarios share
one object geometry and environment family; this is not broad task or object
generalization. Stress failures remain in the denominator. If a baseline cannot
produce a replayable command tape, its dependent control is recorded invalid.

`protocol.json` is written before execution and binds every planned scene,
controller, configuration, source hash and seed. Code changes during execution
stop the run, leaving partial evidence rather than continuing under a stale
identity. Each completed attempt is saved immediately. `report.json` retains
all planned attempt summaries, with success and numerical validity separate.
`attempts/*.json` retains the full states and issued commands, outcome metrics,
warnings and timing; `integrity.json` seals the exact files.

The inspector verifies bounded artifact hashes, protocol consistency and attempt
denominators. It does not authenticate the producer or rerun physics. Replay
export checks the pinned model and recorded geometry before reconstructing poses
with forward kinematics. Native builds can differ by platform; regenerate a
separate local receipt if its compiled-model identity differs. An exported
browser replay is portable.

Development runs may guide changes. After freezing the candidate, use a new
seed and `--cohort confirmation` for a separately declared evaluation. The tool
cannot determine whether you have already seen that seed or related layouts.
After inspection, those scenarios become regression evidence for future tuning.
Changing a folder name does not restore their independence.

## How this becomes a downloadable skill

The native task environment provides an action contract, a baseline, full tapes
and a measurable outcome for the next learning experiment. The next small
DVIDIA action policy should consume only the declared current observations and
history, output bounded arm/jaw commands and own the task transitions it claims
to learn. Authored phases, future state and terminal labels must stay outside its
deployment inputs. Compare its closed-loop completion, false-completion and
recovery outcomes with the baseline on new scenes; a low action-prediction error
alone cannot qualify it.

Skillspace footage will contribute appearances, goals, phases, failures and
coverage. Synchronized robot commands and feedback, or an independently tested
action bridge, supply movement supervision. A learned policy can then be bundled
with its observation/action contract, exact embodiment, supported conditions,
dependency hashes and evaluation evidence. Hardware installation additionally
needs the actual robot adapter, calibration and separate physical trials.

The [next learning experiment](native-grasp-learning.md) defines causal history,
native action units, grouped data, idle-tail sampling, compact model comparisons
and full closed-loop evaluation. These are proposed learning steps, not a trained
policy included in this task benchmark.

See the [execution map](roadmap.md), [episode evidence](episode-tape.md) and
[hardware contract](hardware-reference.md). DVIDIA task code is MIT; upstream
UR5e, Robotiq and viewer notices keep their original licenses.

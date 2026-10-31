# DVIDIA task environments v0.2

Build a small, reproducible practice environment before training a skill. This
preview adds several object instances, an open bin, explicit local OpenUSD,
native PiPER physics and a completion check that considers the whole object.
The local editor and Terminal commands use the same recipe and native runner.

The browser edits a **draft** and replays **recorded native physics**. Simulation
runs in the local Python process; the browser does not run a physics engine.
The arm holds its home commands during these environment checks. This release
does not train a placement policy or infer motor actions from a human video.

## Install the preview

Use Python 3.12 and a fresh virtual environment. Install the wheel from the
[task environment release](https://github.com/Dvidia-Inference/dvidia-training/releases/tag/task-environments-v0.2-preview)
with its optional USD/native dependencies:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install './dvidia_training-0.1.3.dev2-py3-none-any.whl[usd]'
dvidia-robot-sim fetch piper --output robot-assets
dvidia-environment studio --assets robot-assets --output local-scenes
```

Open the displayed `http://127.0.0.1:8346/` address. Select an object, drag it in
the table plane or edit its metric position, then run a preview. Invalid layouts
are explained before stepping physics. Export/import the recipe to share a
layout. Run outputs, including incomplete attempts, stay in `local-scenes`.
Stop the server with Ctrl-C. It binds only to loopback and accepts no remote
credentials, uploaded Python or robot commands. After dependencies and pinned
robot assets are installed, these native runs need no network service.

Physics runs on CPU without CUDA. The browser needs WebGL2, JavaScript and
DecompressionStream; integrated graphics are sufficient for our tested preview.
The tested host is Apple arm64 with Python 3.12.9, NumPy 2.5.3, MuJoCo 3.15.0 and
OpenUSD 26.11. Other operating systems and hardware minima are not qualified.
For planning, allow several GB for the Python environment and verified robot
assets. Timings are shared-host measurements, not hard real-time guarantees.

## Run the same workflow in Terminal

```sh
dvidia-environment init --output my-scene
cp my-scene/recipe.json draft.json
# Edit draft.json; the original authored scene remains unchanged.
dvidia-environment validate --recipe draft.json
dvidia-environment author --recipe draft.json --output edited-scene
dvidia-environment run --recipe edited-scene/recipe.json --assets robot-assets --output my-run
dvidia-environment export --study my-run --output my-viewer
python -m http.server 8000 --directory my-viewer --bind 127.0.0.1
```

Open `http://127.0.0.1:8000/` to inspect the saved native motion. Output commands
refuse existing destinations. `workstation.usda` is deterministically authored
from the recipe and re-read with the real USD SDK before native compilation;
editing the USD without changing the recipe causes rejection. This is a bounded
recipe compiler, not an arbitrary USD or NVIDIA asset importer.

## What the recipe means

All distances use metres, masses kilograms, angles radians and quaternions wxyz.
Z points up, gravity is 9.81 m/s² downward, physics steps every 1 ms, and robot
control boundaries are 20 ms. Object positions refer to their **bottom centre**;
the bin position refers to its **outside bottom centre**. Sizes are full XYZ
dimensions. Contact properties are explicitly labelled `authored_assumption`.

- One pinned six-axis PiPER with its coupled parallel gripper and seven controls.
- A horizontal static tabletop with its upper surface at Z=0.
- One open-top bin built from a floor and four separate wall colliders.
- One to four rigid objects: boxes or an original stepped-box shoe surrogate.
- Versioned source, catalog and robot asset locks, with exact USD byte identity.

Validation rejects duplicate IDs, nonfinite or unsupported fields, overlapping
initial object/bin colliders, table overhangs and objects that cannot fit upright
inside the bin. It allows an explicitly bounded initial drop. It does not prove
collision-free arm reachability, grasp feasibility or a safe insertion path.
The bin remains static. Shoe geometry and mass distribution are synthetic;
deformation, fragile shells and measured material properties remain separate.

## What counts as completed

For **every object**, every corner of every collision piece must fit in the
rotated bin interior. The object must contact the bin floor, have no robot or
gripper contact and stay below the recipe's linear/angular speed limits for the
entire dwell interval. All required objects must pass together. The default
dwell is 250 ms; bounded contact penetration permits a declared 1 mm floor
tolerance. Time reversals, observation gaps and invalid native states clear
history and remain explicitly unknown.

This verifier reads native ground truth for simulation evaluation. It is not a
camera estimator and is not supplied to a robot as an unmeasured sensor.
Likewise, a passive object settling in a bin is an environment fixture, not a
learned robotic placement. Later training can use this contract to evaluate
executed actions without relying on whether an image merely looks finished.

## Reproduce the frozen checks

```sh
dvidia-environment benchmark prepare --output scene-study
dvidia-environment benchmark run --output scene-study --assets robot-assets
dvidia-environment export --study scene-study --output study-viewer
```

Preparation fixes source identities, thresholds, case roster and inputs before
execution. A run writes an intent and attempt evidence; it refuses to repeat an
already started study. Interrupted attempts remain incomplete, never silently
retried or counted as successful. The registered study separates accepted
completion fixtures, negative completion controls and rejected layout inputs.
Their agreement rate is not a robot skill success rate.

## Toward video-to-skill training

A video can propose object identities, relationships and a destination. To turn
that into a useful training environment we must also recover metric scale,
camera/robot frames, contact mechanics and the actual action history. This
release supplies the bounded scene and independent evaluation portion. The next
learning experiment needs causal object pose estimates and corrective action
demonstrations, then held-out start positions and perturbations. It should beat
the fixed controller before becoming an installable skill.

Original DVIDIA code, authored scenes and synthetic rigid assets: MIT. Pinned
PiPER assets retain their MIT notices; Three.js and other dependencies retain
their licenses. No new learned model weights are published with this study.
DVIDIA research · [@iammrriver](https://x.com/iammrriver) · hello@dvidia.org.

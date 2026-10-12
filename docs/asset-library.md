# DVIDIA asset library v0.1

Load a supported arm and synthetic shoes into local CPU simulation, inspect
their native 3D geometry, and save a portable scene descriptor. The first release
includes UR5e + Robotiq 2F-85, PiPER, SO-101, and left/right instances of trainers,
ankle boots and slippers.

These shoes are original **rigid proxies with untested physical properties**.
This release supplies scene assets and an asset benchmark. It does not install
a trained shoe-organizing policy, calibrate deformation, or qualify a physical
robot. SO-101 has **five arm axes plus a gripper command**; UR5e and PiPER have
six arm axes plus a gripper command.

## Install and provision once

Clone the asset-library branch, or check out the exact source commit linked
from a published benchmark before installing its arm extra:

~~~sh
git clone --branch feat/shoe-asset-library-v01 https://github.com/Dvidia-Inference/dvidia-training.git
cd dvidia-training
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install '.[arm]'
dvidia-assets list
dvidia-assets fetch ur5e --output ./robot-assets
dvidia-assets fetch piper --output ./robot-assets
dvidia-assets fetch so101 --output ./robot-assets
~~~

Fetch UR5e first when sharing one root: it requires a fresh output directory.
PiPER and SO-101 then add their own fresh robot folders. Run each fetch once. Fetch downloads only the
packaged, hash-pinned upstream allowlist. Keep the resulting model files and
Python dependencies; scene, load, benchmark and inspect then run offline.
Descriptors cannot introduce URLs, plugins, arbitrary XML or overrides to the
physical catalog.

## Load shoes into the simulator

~~~sh
dvidia-assets scene --robot piper --assets ./robot-assets \
  --pair trainer --output ./trainer-scene
python -m http.server 8000 --bind 127.0.0.1 --directory ./trainer-scene
~~~

On Windows use `py -3.12 -m venv .venv`, `.venv\Scripts\python.exe`
and `.venv\Scripts\dvidia-assets.exe`; activation is optional.

Open http://localhost:8000. The selector offers only the exported shoe families.
Use `--pair all` with a fresh output directory to include trainers, ankle boots
and slippers in one preview.
Use a fresh output directory for each export.

The exporter compiles verified native arm assets, adds shoe bodies and an
infinite flat support plane, and runs two seconds of physics with arm commands
held at initialization. The offline browser shows compiled arm meshes and shoe
collision primitives, recorded every 20 ms. Orbit the camera, scrub initial and
settled states and switch to collision geometry. **The browser replays saved
poses; it does not run physics.** Pair selection filters the recording visually.

Download a descriptor and rerun exactly that selection:

~~~sh
dvidia-assets load ./trainer-scene/scene.json \
  --assets ./robot-assets --output ./reloaded-scene
dvidia-assets inspect ./reloaded-scene
~~~

Descriptors contain robot ID, ordered shoe IDs, timestep, catalog version and
exact catalog hash. They contain no machine-specific paths. Incompatible
catalogs and changed or unknown descriptor fields are rejected.

## Run the benchmark

~~~sh
dvidia-assets benchmark --assets ./robot-assets \
  --seed 612047 --repeats 3 --output ./asset-benchmark
dvidia-assets inspect ./asset-benchmark
~~~

For an existing separate UR5e asset root, add --ur5e-assets PATH. Add
--source-commit FULL_COMMIT_SHA to record a frozen code revision. The benchmark
produces machine-readable receipts and a readable report, retains failures,
and measures its defined native asset protocol rather than a learned skill.
Read scenario definitions, denominators, negative controls and timing scope
before comparing results.

## Hardware and offline use

MuJoCo 3.15.0 runs on the CPU. No CUDA GPU, model weights, cloud inference service
or physical robot connection is required. The optional arm dependencies and
upstream model files require network access during provisioning.

The browser needs JavaScript, WebGL2 and DecompressionStream. Serve the complete
exported directory through a local HTTP server; file URLs cannot reliably load
the data. All scripts, geometry, motion and notices are bundled locally. Browser
graphics support is separate from GPU compute for native simulation.

Minimum RAM and CPU targets have **not** been established across machines.
The benchmark host's measured peak memory and timing are observations, not
universal minimum specifications or a real-time guarantee.

## Evidence and licenses

Each catalog property carries units, evidence status and source. Shoe geometry,
dimensions, mass and friction are authored assumptions. Left/right variants
are procedural instances, not independently captured real objects. The flat
plane, gravity and contact settings are simulation choices. No laces, fabric
dynamics, compressible soles, liquid contents or fracture are modeled.

DVIDIA code, catalog and original shoes are MIT licensed. Arm models retain
their upstream licenses: UR5e BSD-3-Clause, Robotiq 2F-85 BSD-2-Clause, PiPER MIT
and SO-101 Apache-2.0. Exports retain original model licenses and READMEs under
licenses/. Three.js and OrbitControls are MIT; their notices accompany every
viewer.

The preview integrity manifest records every file's byte count and SHA256.
Inspection checks the complete inventory, exact catalog and descriptors,
bounded compressed geometry and motion, finite poses, mesh indices and original
notices. Hashes establish consistency, not authenticity, accurate physics or
real-world transfer.

## Rebuild the viewer

The checked-in browser bundle uses pinned dependencies:

~~~sh
npm ci --prefix viewer
npm run build:assets --prefix viewer
~~~

Next steps are measured object families, calibrated contact and robot/task
compatibility benchmarks. Appearance cannot establish friction, stiffness,
safe squeeze force or graspability.

# Live PiPER shoe workstation v0.1

[Open the workstation](https://dvidia-robot-simulation-lab.static.hf.space/workstation/v01/index.html)
or read the [research and complete benchmark evidence](https://research.dvidia.org/papers/live-piper-shoe-workstation-v01/).

The six-axis PiPER and its native gripper respond to dragged or typed tool-position
targets. A free rigid trainer falls, settles on the worktop and moves under a
bounded physical nudge. Pause, reset, camera views, collision geometry and a local
session receipt make the mechanics inspectable. No account or inference service
is needed to run the hosted workstation.

## Run a complete copy locally

Use Python 3.12 or newer and the published source revision. Provision the package
and upstream robot assets once:

```sh
git clone --branch feat/live-piper-workstation-v01 https://github.com/Dvidia-Inference/dvidia-training.git
cd dvidia-training
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install '.[arm]'
dvidia-robot-sim fetch piper --output ./robot-assets
dvidia-workstation --assets ./robot-assets --output ./live-workstation
python -m http.server 8000 --bind 127.0.0.1 --directory ./live-workstation
```

Open http://localhost:8000. Keep the exported folder together. All meshes,
JavaScript, WASM, geometry, catalog and license notices are local. Later sessions
work offline. On Windows use `py -3.12 -m venv .venv` and the corresponding
executables under `.venv\Scripts`; activation is optional.

The browser fetches one gzip bundle for all 84 native robot meshes, concurrently
with the physics engine and scene. It verifies the bundle, bounds decompression,
and checks every original mesh hash before compiling the model. The original
individual meshes remain in the export for reproduction. This changes delivery
only; the scene, controller, contacts and engine binary are unchanged. First-load
time still depends on the connection and device; it is measured separately from
steady-state simulation speed.

The export includes an integrity inventory and hashed upstream assets. To inspect:

```sh
python -c "from dvidia_training.workstation import inspect_export; print(inspect_export('./live-workstation')['file_count'])"
```

## Controls and mechanics

- Drag a colored target axis or enter X/Y/Z in millimeters. Height and keyboard
  controls provide alternatives to pointer dragging.
- The controller holds one declared tool orientation while solving six arm joint
  targets. The allowed input box is a command bound, not a promise that every
  point is reachable at that orientation. The measured target error stays visible.
- Finger travel spans 0–35 mm for one side of the coupled gripper. It is a native
  position command, not a calibrated jaw gap or force.
- Nudge applies at most 0.12 N·s for one physics tick. It is an explicit user
  disturbance, never a hidden lifting or attachment force.
- Reset restores the declared initial state. Hidden tabs pause; resume explicitly
  after returning. Session receipts retain a bounded recent history and disclose
  dropped samples and resets.

MuJoCo advances at a fixed 2 ms physics timestep. Differential IK runs every
20 ms. Joint-command rates and original servo limits remain enforced. Browser
state updates and rendering run separately, with bounded catch-up and disclosed
lag when a device cannot keep up. There is no shoe weld, object teleport during
interaction, neural policy or physical robot connection.

## Reproduce the benchmarks

```sh
dvidia-workstation-benchmark --assets ./robot-assets --output ./workstation-benchmark
```

The protocol records two resets for eight target positions, jaw close/open travel,
a shoe impulse, disabled-table gravity control and 600 simulated seconds of
continuous native holding. Each tracking attempt has a four-second deadline and
requires both 5 mm position and 3° orientation tolerances. Every attempt and raw
control-window duration remains in the output, including failures. Native timing
excludes browser rendering and network. This evaluates target tracking and
contact diagnostics; it does not score an autonomous shoe-organizing skill.

Browser acceptance and native/WASM parity are separate:

```sh
PYTHONPATH=src python scripts/workstation-parity-fixture.py --scene ./live-workstation --output ./native-parity.json
node tests/browser/workstation_parity.mjs ./live-workstation ./native-parity.json ./wasm-parity.json
```

The actual-browser tests use the same exported folder served over HTTP:

```sh
npm ci --prefix viewer
viewer/node_modules/.bin/playwright install chromium
node tests/browser/workstation_ui.cjs http://localhost:8000/ ./ui-evidence
DVIDIA_BENCH_SECONDS=600 DVIDIA_BENCH_RENDERER=device DVIDIA_BROWSER_CHANNEL=chromium node tests/browser/workstation_live.cjs http://localhost:8000/ ./browser-evidence
```

The device renderer is requested, then identified in the receipt. The live benchmark defaults to
SwiftShader software rendering. Neither render-call counts nor worker state
updates measure GPU presentation completion or hardware actuator latency.

The parity runner requires `npm ci --prefix viewer`. It retains numerical,
loaded-contact and strict raw contact-count comparisons separately. A strict
contact-count mismatch exits nonzero even when state tolerances agree; inspect
all result fields. Floating-point differences around zero-load contacts must not
be presented as exact contact parity.

## Hardware and licensing

Physics runs on a CPU. Browser rendering requires WebGL2; JavaScript,
WebAssembly and DecompressionStream must be available. No CUDA training GPU or
large model download is required. Minimum RAM and CPU specifications have not
been established across devices. Published measurements name their browser,
rendering backend and timing scope; software-rendered browser results are not
hardware GPU measurements.

DVIDIA code, original shoe shapes and procedural visual details are MIT. PiPER
assets retain their pinned upstream MIT notices. Three.js is MIT. MuJoCo is
Apache-2.0 with dependency notices retained in `workstation.third-party.txt`.
The header uses DVIDIA's existing mark and wordmark. The embedded IBM Plex Sans
wordmark font retains its SIL Open Font License 1.1 notice in the HTML; the
software license does not grant rights to DVIDIA trademarks.
Visual shoelaces and surface details are cosmetic; they do not simulate cloth or
change collision geometry. Shoe mass, friction and contact parameters remain
authored assumptions. Retained shoe lifting and real-world transfer are open.

To rebuild the browser runtime with pinned dependencies:

```sh
npm ci --prefix viewer
npm run --prefix viewer build:workstation
```

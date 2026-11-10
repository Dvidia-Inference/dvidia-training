# DVIDIA Godot workstation

A local desktop interface for the existing PiPER workstation. Godot renders the native compiled robot geometry and authored rigid shoe. MuJoCo is the only physics authority; both Godot physics servers are disabled. This does not add a learned shoe policy, calibrated materials, deformable shoes, or physical robot control.

Run the Python bridge's `prepare` and `serve` stages first, then open this project with Godot 4.7.2:

```sh
godot --path integrations/godot -- --scene /absolute/path/to/prepared-scene --port 8430
```

The interface connects only to `127.0.0.1`. Move the three target sliders and gripper-opening slider. Drag empty space to orbit and scroll to zoom. **Home** changes the commanded target through the existing controller. **Reset** explicitly reinitializes the native scene. **Pause** waits for any outstanding control interval and then sends no step commands; a heartbeat keeps the local connection open.

The viewer verifies the scene manifest, each declared file's size and SHA-256, the compiled model identity, and the 2 ms physics / 20 ms control clocks before enabling controls. Each accepted step advances exactly one 20 ms native control interval. Rendering does not determine the number of physics ticks. One request is outstanding at a time; network delays do not create catch-up steps. A disconnection or protocol error stops requests and appears in the interface.

Only native mesh triangles, geom dimensions, transforms and RGBA colors are currently rendered. The project does not load arbitrary GLBs, PBR textures or physics extensions. The MuJoCo Z-up coordinate system is consistently mapped into Godot Y-up by `[x, y, z] → [x, z, -y]`; body poses remain world-space and geom transforms remain body-local.

## Verification

```sh
godot --headless --path integrations/godot --check-only --script main.gd
godot --headless --path integrations/godot --script tests/test_coordinates.gd
godot --headless --path integrations/godot -- \
  --scene /absolute/path/to/prepared-scene --port 8430 \
  --smoke-steps 100 --smoke-output /absolute/path/to/fresh-receipt.json
```

For a normal-window smoke run, omit `--headless` and optionally add `--screenshot /absolute/path/to/fresh-image.png`. Smoke mode executes three deterministic bounded target/jaw settings. Its receipt preserves initial state, up to 256 exact request/response records, model and scene identities, applied simulation time, native compute time and wall-clock elapsed time. Existing output files are preserved. Native compute time excludes rendering and socket transport; process frame count includes headless process ticks and is not a graphics benchmark.

DVIDIA source is MIT licensed. Prepared scene exports carry the original pinned PiPER asset license and attribution alongside DVIDIA's license. No external asset license is replaced by the application license.

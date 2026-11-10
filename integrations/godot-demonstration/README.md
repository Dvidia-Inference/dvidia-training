# DVIDIA Demonstration studio

A standalone Godot 4.7 project for recording simulated PiPER block demonstrations. It renders the native MuJoCo geometry and exposes target X/Y/Z and the single-finger jaw target. Tool orientation is fixed. The existing `integrations/godot` workstation is a separate project.

Prepare the matching scene, start the local recorder, then open the viewer. Use fresh output directories:

```sh
python -m dvidia_training.godot_demonstration_bridge prepare \
  --assets /absolute/path/to/piper-assets \
  --output /absolute/path/to/demo-scene --duration 20

python -m dvidia_training.godot_demonstration_bridge serve \
  --assets /absolute/path/to/piper-assets \
  --scene /absolute/path/to/demo-scene \
  --output /absolute/path/to/demo-session \
  --source-kind simulation_human_input --port 8431 --max-episodes 4

# In another terminal:
godot --path integrations/godot-demonstration -- \
  --scene /absolute/path/to/demo-scene --port 8431
```

`simulation_human_input` identifies a session where a person operates the controls. Automated actions must use `simulation_scripted_input`; smoke mode refuses any other source. The interface displays the recorder's source, acknowledged step count, recorded duration, and output directory.

- **Record** begins the prepared episode. No recording control intervals run before it; preparation performs the native reset and settling.
- **Pause / Resume** changes recording state without advancing simulation time. A click waits for an outstanding step to finish.
- **Finish** finalizes the current episode. Reaching the configured step limit also finalizes it.
- **New episode** finalizes the current episode and resets into a fresh output directory. It is disabled after the session's episode limit.
- **Home** returns the commanded target and jaw to the scene's initial values. Drag empty space to orbit; scroll to zoom.

The orbiting viewport is an operator view, not the policy camera. The Python recorder captures native RGB-D and action/state records. This client has no training or physical robot controls.

MuJoCo is the only physics authority; both Godot physics servers are disabled. Each acknowledged step advances one 20 ms control interval comprising 1 ms native physics ticks. The client requests steps at a nominal 50 Hz with exactly one outstanding request. It does not issue catch-up steps after delayed responses, so elapsed wall time can exceed simulated duration. Recorder errors and disconnections remain visible and stop further commands.

The viewer verifies the exported manifest, each declared file's size and SHA-256, compiled model identity, and simulation clocks. Meshes and transforms use the coordinate rotation `[x, y, z] → [x, z, -y]`. Body transforms are world-space and geom transforms are body-local. The line-delimited JSON protocol is `dvidia.godot-demonstration/v1`; numbered commands start at 1, while hello and heartbeat ping are unnumbered.

## Checks and scripted smoke

```sh
godot --headless --path integrations/godot-demonstration --check-only --script main.gd
godot --headless --path integrations/godot-demonstration --script tests/test_coordinates.gd
godot --headless --path integrations/godot-demonstration --script tests/test_recording.gd
```

Start a fresh recorder session with `--source-kind simulation_scripted_input`, then run:

```sh
godot --headless --path integrations/godot-demonstration -- \
  --scene /absolute/path/to/demo-scene --port 8431 \
  --smoke-steps 100 --smoke-output /absolute/path/to/fresh-smoke.json
```

The prepared duration must permit the requested number of steps. Smoke sends Begin, bounded scripted XYZ/jaw steps, and Finish, including an idempotent Finish after automatic completion. It exits successfully only after the final acknowledgement. Its diagnostic receipt contains initial state, up to 256 request/state/recording snapshots, final recording metadata, model and scene digests, server compute timing (including native camera capture and disk writes), and socket/process timing. These are diagnostics, not a training dataset or a graphics benchmark. Existing output files are preserved.

For a windowed smoke run, omit `--headless` and add `--screenshot /absolute/path/to/fresh-image.png`. The screenshot shows the operator viewport; native policy camera images are written by the Python recorder.

DVIDIA source is MIT licensed. Prepared exports include the original pinned PiPER asset license and attribution alongside DVIDIA's license.

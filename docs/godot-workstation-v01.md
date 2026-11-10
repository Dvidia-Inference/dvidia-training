# Godot workstation v0.1

This desktop prototype displays the existing PiPER workstation through Godot while
MuJoCo alone computes motion and contact. It adds a local command bridge and a
native-mesh viewer; it does not replace the dynamics solver or train a skill.

The scene uses the pinned upstream arm model and an authored, rigid shoe proxy.
Its physical properties are uncalibrated. The visual export currently contains
compiled geometry and RGBA colors, not GLB materials, textures, or a general asset
importer. Physical robot control is not implemented.

## Run locally

Use a checkout containing this prototype, Python 3.12+, the repository's pinned
NumPy/MuJoCo dependencies, and Godot 4.7.2. The Python package alone does not
include the Godot project; keep the checkout. This initial version has been
developed on macOS/arm64. Other platforms need their own runtime checks.

From the repository root, install the simulation dependencies and fetch the
checksum-pinned arm assets once. These two setup operations need networking.
The simulation then runs locally, without a cloud API or account.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[arm]'
dvidia-robot-sim fetch piper --output ./robot-assets
python -m dvidia_training.godot_bridge prepare --assets ./robot-assets --output ./godot-scene
python -m dvidia_training.godot_bridge serve --assets ./robot-assets --port 8430
```

In a second terminal, open the viewer. Replace the scene path with the absolute
path printed by your shell's `pwd` plus `/godot-scene`.

```sh
godot --path integrations/godot -- --scene /absolute/path/to/godot-scene --port 8430
```

The target sliders move the end-effector through bounded authored inverse
kinematics. The gripper slider changes the commanded finger opening. Pause stops
new simulation commands; Home changes the target; Reset reinitializes the entire
scene at time zero. Orbit and zoom affect only the viewer. Stop the server with
Ctrl-C. Each new connection starts a new scene; it does not resume an old run.

The service binds only `127.0.0.1`. It has no hardware adapter. A viewer presents
the exact prepared model digest before sending commands. Every accepted command
advances ten 2 ms physics ticks, with the existing 50 Hz controller and limits.
There is only one pending command in the viewer. A slow client reduces wall-clock
progress instead of skipping simulation ticks. Paused keepalives carry no control
authority; an idle or disconnected client cannot cause background stepping.

## Reproduce the checks

```sh
PYTHONPATH=src python -m unittest tests.test_godot_bridge -v
godot --headless --path integrations/godot --script tests/test_coordinates.gd
PYTHONPATH=src python scripts/benchmark-godot-bridge.py --assets ./robot-assets --output ./cpu-parity.json
```

The benchmark compares three paired repetitions of the same six-segment tracking
sequence. Each path advances 12 simulated seconds per repetition. It compares
all 1,800 paired returned states, including contacts, exactly. The direct path is
the native Python call; the bridge path additionally encodes, validates and
decodes JSON. Its timing therefore measures the complete JSON request path,
not just physics or socket latency. Initialization and compilation are reported
separately. No viewer or camera is rendered during that benchmark.

With the server running, an automated actual Godot client can collect a separate
receipt:

```sh
godot --path integrations/godot -- --scene /absolute/path/to/godot-scene --port 8430 --smoke-steps 100 --smoke-output /absolute/path/to/godot-smoke.json --screenshot /absolute/path/to/godot-smoke.png
```

Use fresh output paths. The smoke receipt records commands and authoritative
states, permitting independent replay against the original workstation. This
is an integration and performance test, not a grasp or placement success rate.
Headless Godot tests omit actual viewport rendering and must be reported
separately from windowed execution.

## What this establishes

A passing comparison establishes that the adapter preserves this exact native
scene's simulated behavior under the tested commands. It does not validate that
the scene matches a real shoe or arm, that new assets are accurate, or that a
policy transfers to hardware. Reference-to-reference parity and comparison with
physical measurements are different gates.

The next physical-asset work and performance criteria are specified in
[the simulation roadmap](godot-simulation-roadmap.md). Keep the existing
camera-to-action benchmark and its qualification result separate.

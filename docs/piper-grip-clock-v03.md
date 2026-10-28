# PiPER grip clock v0.3

This simulation study learns when to pause a frozen movement reference while grasping a 40 g rigid block. The same virtual progress drives the arm and original jaw reference, so a wait also delays release. It does not learn new grasp geometry. A matched blind clock trains from the same demonstration rows and capacity recipe with live grip/history inputs masked.

The exact frozen result, model revision, selected recipe and complete benchmark will be linked here after confirmation. The acquisition teacher checks sampled bilateral pad contact/force and small tool-relative drift over 100 ms. That signal supplies causal training labels before command issue; it is not an inference-time teacher or physical grasp certification. After acquisition its latch stays released, so this version does not establish continuous learned slip recovery.

The new study retains 64 synthetic teacher demonstrations, four finite candidate fits, 192 development attempts and 512 confirmation attempts from 32 fresh XY groups. Both trained goals and nominal/post-grip conditions remain together within each group. Failed finite teacher prefixes are included in training; every scheduled evaluation remains in its denominator. The strict endpoint requires lift, retained carry, supported placement, release, settling and retreat without unintended contact.

## Run one installed model

Python 3.12 or newer is required. Install the version 0.1.1 wheel with its `arm` extra, or install the exact release checkout:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[arm]'
dvidia-robot-sim fetch piper --output ./robot-assets
```

Download `base-model.json`, `jaw-model.json` and `selected.json` from the model repository at the exact revision in the publication receipt. The jaw file is a manifest binding the original primitive's jaw reference; it is not the v0.2 jaw adapter. Run with a fresh output directory:

```sh
dvidia-piper-clock-skill \
  --base-model ./base-model.json \
  --jaw-model ./jaw-model.json \
  --model ./selected.json \
  --assets ./robot-assets \
  --start-x 0.312 --start-y -0.037 --goal 0 \
  --output ./my-clock-attempt
```

The runner copies exact JSON bytes, uses privileged native simulator observations and saves a complete 20-second attempt, 1 ms numeric telemetry, independent strict score and hashes. It collects no expert labels and adds no artificial disturbance. It refuses an existing output directory, preserving failures and interrupted work. Exit 0 means strict success, 1 a valid failed attempt, and 2 invalid input or invalid evidence. The separately fitted blind model can run through the same command; its validated document selects masked input mode.

The task envelope is starting X 0.284–0.314 m, Y −0.045–−0.015 m. Goal 0 is (0.35, 0.08) m and goal 1 is (0.40, −0.08) m. Different objects, arms, surfaces, goals, sensors and physical hardware require separate evaluation.

## Reproduce the study

The lifecycle CLI uses the same immutable attempt runner as the installer:

```sh
dvidia-piper-grip-clock prepare ./new-study \
  --base-model ./base-model.json --jaw-model ./jaw-model.json
dvidia-piper-grip-clock collect ./new-study --assets ./robot-assets
dvidia-piper-grip-clock train ./new-study --name full4 --max-depth 4
dvidia-piper-grip-clock train ./new-study --name full6 --max-depth 6
dvidia-piper-grip-clock train ./new-study --name blind4 --blind --max-depth 4
dvidia-piper-grip-clock train ./new-study --name blind6 --blind --max-depth 6
```

Evaluate each of the four declared candidate names with `development ./new-study --assets ./robot-assets --name NAME`. Select one matched depth pair explicitly only after reviewing all development results. `freeze ./new-study --name full4 --blind-name blind4` is an example, not a claim that depth four is the selected release. Then `confirm ./new-study --assets ./robot-assets` runs the entire frozen interleaved roster, and `report ./new-study` verifies the saved results without rerunning physics.

A prepared study binds source files, models, labels, cases, weights, teacher configuration and native scorer. Changing a bound source requires a new lineage. A frozen study cannot be retuned or recollected. These deterministic roster commands reproduce the published positions; changing training design requires a separately versioned experiment and new unseen confirmation positions.

## Hardware and limits

No GPU, model service, camera or account is required after installation and one pinned-asset download. Training is a small NumPy classification tree. The study host is an Apple M5 with ten logical CPUs and 24 GiB installed memory; that is a measured host, not a minimum requirement. Shared-host latency distributions include dense native logging and exclude camera capture, communication, rendering and real actuators. Hard real-time performance and physical transfer remain unqualified.

See the [development lineage](piper-grip-clock-v03/development-lineage.md) and [host record](piper-grip-clock-v03/host.json). DVIDIA code, models and authored numeric evidence use MIT; upstream PiPER geometry and renderer notices remain included with their assets and replay.

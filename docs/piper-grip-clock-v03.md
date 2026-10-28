# PiPER grip clock v0.3

This simulation study learns when to pause a frozen movement reference while grasping a 40 g rigid block. The same virtual progress drives the arm and original jaw reference, so a wait also delays release. It does not learn new grasp geometry. A matched blind clock trains from the same demonstration rows and capacity recipe with live grip/history inputs masked.

[Research and recorded comparison](https://research.dvidia.org/papers/learned-piper-grip-clock-v03/) · [Model](https://huggingface.co/Dvidia/piper-grip-clock-v03) · [Complete evidence](https://huggingface.co/datasets/Dvidia/piper-grip-clock-demonstrations-v03) · [Release and installer](https://github.com/Dvidia-Inference/dvidia-training/releases/tag/piper-grip-clock-v0.3). The release’s [publication receipt](https://github.com/Dvidia-Inference/dvidia-training/releases/download/piper-grip-clock-v0.3/publication-receipt.json) records exact public revisions and artifact hashes. The acquisition teacher checks sampled bilateral pad contact/force and small tool-relative drift over 100 ms. That signal supplies causal training labels before command issue; it is not an inference-time teacher or physical grasp certification. After acquisition its latch stays released, so this version does not establish continuous learned slip recovery.

The new study retains 64 synthetic teacher demonstrations, four finite candidate fits, 192 development attempts and 512 confirmation attempts from 32 fresh XY groups. Both trained goals and nominal/post-grip conditions remain together within each group. Failed finite teacher prefixes are included in training; every scheduled evaluation remains in its denominator. The strict endpoint requires lift, retained carry, supported placement, release, settling and retreat without unintended contact.

## Frozen outcome

The learned clock completes **44/128** strict tasks, versus **26/128** for the matched blind clock, **72/128** for fixed timing and **67/128** for the sampled acquisition teacher. All 512 confirmation attempts are valid. The paired full-minus-blind difference is +14.1 percentage points with an approximate simultaneous 95% interval of −8.5 to +36.6 points; it does not establish a reliable feedback advantage. Full-minus-fixed is −21.9 points, interval −36.8 to −7.0. Intervals use 32 XY groups, keeping the two goals and two conditions together.

The learned clock fails acquisition in 40/128 attempts; fixed timing fails acquisition in none. The independent audit reconstructs 768 main-study attempts with zero mismatches and finds 33,418 learned hold decisions where the sampled teacher would advance. These are observations of the saved rollout, not proof of a single causal failure mechanism. All promotion gates together fail, and the model remains unqualified.

Policy-call CPU p95 is 1.916 ms and maximum 256.394 ms; full control-step p95 is 9.594 ms and maximum 265.909 ms. There are 728/128,000 full steps over the 20 ms decision period on the shared four-worker host. These results do not establish hard real-time operation.

## Run one installed model

Python 3.12 or newer is required. Install the version 0.1.1 wheel with its `arm` extra, or install the exact release checkout:

```sh
git clone --branch piper-grip-clock-v0.3 https://github.com/Dvidia-Inference/dvidia-training.git
cd dvidia-training
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

The controller issues six joint targets in radians and one coupled jaw slide target in metres. The original jaw command ranges from 0 to 0.035 m; that is not a measured full finger aperture. Native physics uses 1 ms ticks, while decisions and observations use 20 ms boundaries. Force values are uncalibrated simulator readings.

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

Evaluate each of the four declared candidate names with `development ./new-study --assets ./robot-assets --name NAME`. Select one matched depth pair explicitly only after reviewing all development results. The published selection is `full4` and `blind4`: both full-sensing recipes completed 12/32 development tasks, so we selected the smaller tied model before seeing blind results. Use `freeze ./new-study --name full4 --blind-name blind4` to freeze that matched pair. Then `confirm ./new-study --assets ./robot-assets` runs the entire frozen interleaved roster, and `report ./new-study` verifies the saved results without rerunning physics.

A prepared study binds source files, models, labels, cases, weights, teacher configuration and native scorer. Changing a bound source requires a new lineage. A frozen study cannot be retuned or recollected. These deterministic roster commands reproduce the published positions; changing training design requires a separately versioned experiment and new unseen confirmation positions.

## Hardware and limits

Training and native CLI simulation need no GPU, model service, camera or account after installation and one pinned-asset download. The recorded 3D browser viewer uses WebGL. Training is a small NumPy classification tree. The study host is an Apple M5 with ten logical CPUs and 24 GiB installed memory; that is a measured host, not a minimum requirement. Shared-host latency distributions include dense native logging and exclude camera capture, communication, rendering and real actuators. Hard real-time performance and physical transfer remain unqualified.

See the [development lineage](piper-grip-clock-v03/development-lineage.md) and [host record](piper-grip-clock-v03/host.json). DVIDIA code, models and authored numeric evidence use MIT; upstream PiPER geometry and renderer notices remain included with their assets and replay.

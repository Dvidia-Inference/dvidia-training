# PiPER test lift and bounded recovery

This candidate adds a grip check to DVIDIA's authored PiPER rigid-block task.
The controller briefly lifts before committing to a carry. A supported pinch
can register two finger contacts without supporting the object's weight, so
contact alone does not qualify the grasp.

The original model's servos, force limits, mass, collision geometry and gravity
settings are preserved. The original phase-independent task scorer still
requires a retained lift, carry, supported release, settled destination and
retreat. A refused carry is recorded as an incomplete task, even when the grip
check correctly detects a problem.

## Run locally

Use the candidate branch, Python 3.12 or newer, and the optional CPU simulation
dependencies. No GPU, model API or account is required for local simulation.
Provision the software and pinned PiPER assets once, then run offline.

```sh
git clone --branch feat/piper-grip-recovery https://github.com/Dvidia-Inference/dvidia-training.git
cd dvidia-training
python3.12 -m venv .venv
.venv/bin/python -m pip install '.[arm]'
.venv/bin/dvidia-robot-sim fetch piper --output runs/robot-assets
.venv/bin/dvidia-piper-grip benchmark --assets runs/robot-assets --output runs/piper-grip
.venv/bin/dvidia-piper-grip inspect runs/piper-grip
```

On Windows use `py -3.12 -m venv .venv` and executables under
`.venv\Scripts\`. Every run requires a new output directory; failed attempts
and incomplete controls remain visible in the declared denominator.

## What is compared

Each scene runs the original placement controller and the guarded controller.
Each gets its own identical requested-arm tape replayed with the jaw open.
A common stationary-arm control completes the five attempts per scene.
The default protocol varies friction across 0.8, 0.55 and 0.15 with two seeded
start/target layouts per surface: thirty attempts in total.

The guard uses current simulator observations and its past state. It does not
read the declared friction coefficient to decide whether a grasp will work.
After closure, it tests object following and loss of table support. Excessive
relative motion triggers a hold of the measured joint positions. A supported
object permits at most one open/recenter/retry; an unsupported object does not
permit an opening recovery. Issuing a hold does not stop physical velocity
instantly, so the benchmark records actual motion after detection.

## Reading the evidence

Completion, acquisition decisions, retries and aborts are separate outcomes.
Threshold agreement is not a detector accuracy measurement. Any later-window
check is explicitly an offline evaluation of recorded simulator truth and is
not available to the causal controller. Trials with no opportunity for a
decision must not disappear from the report.

The full timeout, action tape, contacts and native states are retained even
when the controller aborts or completes early. A source/protocol freeze precedes
confirmation runs. The inspector checks evidence consistency and declared
controls; it does not authenticate a publisher or recompute physics.

This remains an authored policy using privileged simulator state. Camera,
force or tactile observations on hardware require calibrated sensing, bounded
latency and independent validation. There is no learned shoe skill or physical
robot qualification. Retry cannot be assumed to repair inadequate friction;
a different grasp or separately declared actuator/material experiment may be
needed.

The measured host and timing scope accompany each report. These CPU results do
not establish universal minimum RAM or a sensor-to-actuator real-time guarantee.
DVIDIA's original tooling is MIT; the pinned upstream PiPER model retains its
original MIT notice, distributed separately with the benchmark.

## Frozen confirmation result

The [complete 30-attempt receipt](../benchmarks/piper-grip-20261011/README.md)
records a regression: **guarded placement 2/6 versus original placement 4/6**.
Both controllers complete the normal cases, but at friction 0.55 the original
completes 2/2 while the guard aborts 0/2 completed tasks. At friction 0.15 the
guard withholds qualification, retries once per scene, then aborts; neither
controller completes those tasks. No successful recovery is demonstrated.

The measured runtime is commit `3ee94706f436690c6d1d8554eda9202d23620e3c`,
seed 731209, executed from an installed wheel with network denied. The full
30-attempt command takes 104.261 wall seconds for 600 simulated seconds,
including compression and final inspection on the recorded CPU host. All
53 over-budget computation windows remain visible; real-time control is not
established. The regression identifies a concrete next experiment, rather than
a qualified replacement for the original controller.

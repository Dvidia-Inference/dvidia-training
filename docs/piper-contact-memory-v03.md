# PiPER contact history v0.3

This candidate compares the original placement controller, the same controller
with an acquisition test lift, and a controller that also uses recent contact
history to slow down, hold closed and reassess apparent slip. Version 0.3 names
this controller experiment; the training package version remains separate.

The previous guard completed 2/6 placements versus the original controller's
4/6. Its fixed displacement threshold stopped intermediate-friction tasks that
the original controller completed. A probe-only comparison is needed to
separate the effect of the probe motion from the effect of intervening.

The [frozen confirmation](../benchmarks/piper-contact-20261011/README.md) completed
6/12 history placements versus 4/12 original and 4/12 probe-only placements.
All 84 attempts were valid, and all 48 negative controls failed. The two added
successes were heavier blocks at friction 0.8; lower-friction failures remain.

## Run on a local CPU

Install Python 3.12 or newer and provision the pinned model once. Simulation
then runs locally without a GPU, model API or account.

```sh
git clone --branch feat/piper-contact-memory-v03 https://github.com/Dvidia-Inference/dvidia-training.git
cd dvidia-training
python3.12 -m venv .venv
.venv/bin/python -m pip install '.[arm]'
.venv/bin/dvidia-robot-sim fetch piper --output runs/robot-assets
.venv/bin/dvidia-piper-contact benchmark --assets runs/robot-assets --output runs/piper-contact
.venv/bin/dvidia-piper-contact inspect runs/piper-contact
```

On Windows use `py -3.12 -m venv .venv` and the executables under
`.venv\Scripts\`. Use a new output directory for each run. All attempted
episodes, including refusals and failures, stay in the declared denominator.

## Experimental contract

The benchmark compares three placement modes. `baseline` preserves the
original authored motion. `probe_only` adds the bounded acquisition probe and
supported retry, without post-qualification intervention. `history` uses the
same acquisition stage and adds the contact-history response. Each placement
has its own exact requested-arm tape replayed with open jaws, and every scene
also has an initial-arm hold control.

Two seeded start/target layouts are shared across friction 0.8, 0.55 and 0.15
and object masses 40 g and 60 g. The twelve scenes each run seven variants,
for 84 planned attempts. These synthetic conditions are not measured shoe
materials. Native robot mechanics, actuator gains, force caps, free-object
dynamics and the independent task scorer remain unchanged. Episodes retain
the original 20-second duration, including aborts and completed tasks.

The history controller observes current and past tool/object poses, jaw state,
native contacts and support. It does not read the scene's declared friction or
mass to decide whether to intervene. Short-window relative motion and contact
persistence produce settling/slip hypotheses, not independently validated
material labels. Insufficient history must remain distinguishable from a
stable grasp. A slowdown changes requested movement; the receipt measures the
actual movement that follows. A requested hold does not instantly stop inertia.
The original motion teacher retains its original lift flag; the new contact
history check supplements that controller and does not replace its acquisition
or task logic with a learned model.

Recovery remains bounded. Opening and recentering require a currently supported
object; holding an unsupported object keeps the jaws closed. Qualification,
slowdown, reassessment, resumption, retry, abort and eventual successful placement
are separate events and measures. Resumption alone is not successful recovery.

## Evidence and scope

Development may inform the controller, but confirmation uses a fresh reserved
seed after the implementation and protocol are frozen. Preserve every complete
compressed trajectory, command tape, native state, warning, timing outlier and
original model notice. The inspector checks stored and decoded hashes and the
recorded evidence contract. It does not authenticate a publisher or independently
recompute physics.

This is an authored controller using privileged simulator observations. It does
not train a policy from footage, estimate deformable materials, implement a
physical sensor pipeline or qualify a robot for hardware operation. Sensor
noise, latency, calibration and compliant-object behavior need separate tests.
Timing covers CPU computation under the stated measurement boundary; it is not
sensor-to-actuator response time or a universal minimum hardware specification.

Original DVIDIA code is MIT. The pinned PiPER model keeps its original MIT
notice. Installation and evidence remain linked to explicit source revisions.

# Shoe graspability v0.3

The earlier shoe experiment combined retrieval, wrist-tool clearance and pickup.
This diagnostic puts the object directly within reach and varies three explicit
factors: rigid shape, object load and whether the rake is mounted at episode start.
It asks what this particular simulated hand can hold before adding more recovery.

## Run the study

```sh
git clone https://github.com/Dvidia-Inference/dvidia-training.git
cd dvidia-training
python -m venv .venv
# Activate .venv using your shell.
python -m pip install '.[arm]'
dvidia-robot-sim fetch piper --output robot-assets
dvidia-graspability-benchmark --assets robot-assets --output results --split development
python -m http.server 8000 --directory results
```

Use the source commit recorded in a published receipt to reproduce that version.
The `heldout` split is a sealed evaluation cohort; once inspected or used to tune a
controller, it becomes regression data. Published fresh results are one execution
of the frozen implementation, not a collection of the best reruns.

The native simulation uses MuJoCo 3.15.0 on CPU, a 2 ms physics timestep and a
20 ms controller interval. No discrete GPU or inference server is required.
Install dependencies and pinned arm assets once; runs and the self-hosted replay
can then work offline. Replay requires WebGL2. Minimum RAM and cross-machine
throughput have not been established. Reported wall time includes native model
initialization, controller and trace collection, and excludes rendering/export.

## Controls and evidence

| Factor | Conditions |
| --- | --- |
| Shape | 40 × 40 × 40 mm block; 140 × 50 × 60 mm narrow shoe proxy; existing compound shoe proxy |
| Mass | 30 g lightweight control; 300 g reference load |
| Wrist tool | Rake mounted; rake absent from episode start |
| Fresh replication | Two paired layout/friction settings per load, 24 attempts total |

Every shape uses an authored, fixed grasp offset; the pose is the same with and
without the tool. Joint drives, force caps and controller rates stay unchanged.
Removing the rake changes both clearance and wrist inertia. Changing shape also
changes inertia and contact geometry. The study cannot isolate those effects
further, nor does it test tool removal, retrieval, learned grasp selection or a
trained shoe skill.

An independent scorer checks every native 2 ms sample. At 0.4 s it verifies a
settled reference: world support, no robot contact and at most 1 mm axis drift
over the previous 100 ms. The final 500 ms must maintain at least 35 mm rise from
that reference, contact on both fingers, and no world, rake or other robot support.
The exact lowest collision-surface height is reported separately. Invalid traces
are distinct from valid physical failures. Full traces, visual/collision geometry,
source hashes, actuator fingerprint, all successes and all failures are retained.

## What the model leaves out

The full shoe is a **rigid, solid proxy**. Its 100 mm sole and 94 mm upper are wider
than the pinned gripper's nominal 70 mm slide-coordinate span. That does not prove
all grasps impossible: the controller targets a narrower region. However, a hollow
collar, flexible upper, heel tab and laces are absent, so several real grasping
strategies cannot be evaluated here.

The pinned gripper is a coupled position drive with gain 40 N/m and an actuator
force cap of ±10 N. A cap is not a command to apply that force. For example, a
30 mm position error contributes about 1.2 N before damping and transmission
coupling. See the [MuJoCo position-actuator definition](https://mujoco.readthedocs.io/en/3.3.1/XMLreference.html#actuator-position).
The viewer reports actual simulated drive force and finger contact forces
separately; neither is a validated measurement of physical PiPER hardware.

The next fidelity work is to measure the actual gripper's force-versus-command
response, fit a named hardware profile, and add shoe geometry with a real collar
and measured material behavior. Keep this diagnostic as a regression test when
those calibrated models are introduced. More retries cannot substitute for
checking those physical assumptions.

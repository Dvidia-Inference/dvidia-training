# Continuous shoe grip v0.2

Reaching an object is only the first part of acquiring it. The v0.1 held-out
mounted-rake experiment brought three of four shoes into a retrieval checkpoint,
but none passed the subsequent retained lift. This experiment compares that
fixed sequence with a controller that checks acquisition and responds to grip
loss while the task is running.

Both policies use the same PiPER, mounted rake and rigid synthetic shoe. In this
comparison, `baseline` means the earlier **tool-enabled** controller, not the
no-tool capability control from v0.1. A compiled-model hash accompanies each
trial so matched geometry and dynamics can be checked independently.

## Run an experiment

Use the source revision identified by the research publication when reproducing
its results. From a checkout:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install '.[arm]'
.venv/bin/dvidia-robot-sim fetch piper --output runs/robot-assets
.venv/bin/dvidia-grip-benchmark --assets runs/robot-assets \
  --output runs/my-grip-development --split development
.venv/bin/python -m http.server 8000 --bind 127.0.0.1 \
  --directory runs/my-grip-development
```

Open `http://127.0.0.1:8000/` for recorded 3D replays of those native runs. Moving
the replay timeline does not execute physics. The Python command runs fresh
physics locally; it does not connect to a physical arm.

Use `--split regression` to inspect the earlier v0.1 development and held-out
cases together. They are now known regression cases, not unseen confirmation
data. `--split heldout` reproduces the separately declared v0.2 parameter cohort.
Re-running that cohort does not create independent confirmation evidence. Add
`--source-commit FULL_GIT_SHA` to bind the result to an exact source revision.
The output directory must be new.

Windows uses the virtual environment's `Scripts` executables; put continued
commands on one line. Initial package installation and asset provisioning need
network access. Simulation and replay can run locally after provisioning.

## What the result means

The policy sees privileged simulator position and contact information. It is
authored code, with no learned weights, camera perception or physical tactile
measurements. The rake starts mounted. Shoe geometry is rigid, contact
parameters are assumptions, and the usable reach envelope belongs to this
controller rather than all possible PiPER configurations.

Keep these outcomes separate:

- A retrieval checkpoint records whether the rake has brought the shoe into
  the configured working region at that time.
- Acquisition records measured bilateral finger contact, not merely closed
  jaws or a requested grasp pose.
- Retained lift requires height, bilateral contact and absence of world, rake
  and non-finger robot support throughout the final declared window.
- Recovery requires a recorded loss, issued response and subsequent independent
  result. A recovery attempt or help request does not count as completion.

The shoe height is its authored sole-bottom **frame**, not its lowest physical
surface after rotation. Independent support checks prevent a tilted shoe resting
on the table from counting as a retained lift. Dense physics-step records are
needed to see losses between the less frequent replay frames.

The records distinguish the independent condition-loss witness, controller
detection, recovery request and first native command packet. The trajectory
records subsequent movement, but this release does not derive the first
physical response or stopping time. Simulated command delay is not
camera-to-actuator latency, and a bounded position-command change is not proof
of a physical stopping distance.

An independent loss episode requires prior qualifying contact and lift, held
for 100 ms; a sustained loss then lasts at least 20 ms. A controller event that
has no matching independent witness may reflect a different conservative
predicate. It is not automatically a false alarm, and an unqualified contact
gap is not a measured missed slip. A planned release can also break the scored
condition; uncovered condition-loss episodes alone do not establish missed
alarms.

## Evidence and limits

The export includes the frozen protocol, a small benchmark index, all full
trial receipts, independent scores, dense native traces, model and motion
assets, the replay viewer, licenses and a file integrity inventory. Only the
dense `native_trace` is omitted from the index; it remains in each full receipt.
Invalid attempts and failed tasks stay in the denominator.

Controller changes and the new protocol are frozen before the confirmation
cohort runs. Development, historical regression and new confirmation results
remain separate. The old v0.1 artifacts are immutable. Publisher checks bind the
new `prerequisites/v02` namespace to the `dvidia.continuous-shoe-grip/v2` schema,
source commit, protocol and complete attempt inventory.

The native benchmark requires a CPU and the optional MuJoCo dependency. The
viewer requires WebGL2. No GPU or language model is on the control path.
Published timing should identify the tested host and report native execution,
audit/export, control computation and simulated task duration separately. A
single successful laptop run does not establish minimum RAM or real-time
hardware performance.

DVIDIA code and synthetic objects are MIT. PiPER and bundled dependencies retain
their notices; original research writing is CC BY 4.0. Branding does not grant
trademark rights. Neither success in these synthetic scenes nor an integrity
hash qualifies a physical robot or validates contact properties.

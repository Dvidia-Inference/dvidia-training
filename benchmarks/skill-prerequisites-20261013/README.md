# Skill prerequisites v0.1

Two small native MuJoCo experiments expose whether an action is possible before
attempting it: a passive shoe-box lid, and a shoe beyond the controller's usable
workspace. These are authored PiPER controllers with simulator observations,
synthetic rigid objects and real native contacts. They are not learned skills
or hardware qualification.

[Research and limitations](https://research.dvidia.org/papers/skill-prerequisites-v01/)
· [Inspect every trial](https://research.dvidia.org/downloads/topics/physical-grounding/runs/2026-10-13-skill-prerequisites-v01/lab/)
· [Hugging Face](https://huggingface.co/spaces/Dvidia/robot-simulation-lab)
· [Run locally](../../docs/prerequisite-simulations.md)

## Frozen result

Source was committed as `756306a2beda60ac54567971626080a56227e781` before this
single held-out run. All **16/16 attempts were valid** and retained. Four scene
conditions per task were run with both approaches. These are declared parameter
holdouts, not independent real-world reliability trials. No controller was
tuned after viewing these results.

| Experiment | Baseline | Feedback and recovery |
| --- | --- | --- |
| Hinged box: task completion | 2/4 | 2/4 |
| Hinged box: lid contact over 0.5 N | 2/4 | 0/4 |
| Shoe: reaches retrieval checkpoint | 0/4 | 3/4 |
| Shoe: unsupported retained lift | 0/4 | 0/4 |

The lid task is an inspection/access waypoint followed by withdrawal, **not
shoe extraction**. The guard avoids the two baseline lid contacts (75.15 N and
51.78 N peak normal force) by withdrawing and requesting help. Those tasks stay
unfinished. In the transient case it withdraws, waits, resumes, and completes in
2.30 s; the baseline already completed without contact in 1.60 s, so the guard
adds a 0.70 s nuisance delay. The 0.5 N threshold is a synthetic evaluation
criterion, not an injury limit. Hazards begin at reset; zero logged response
delay is an initial-state decision, not measured sudden-hazard or sensor latency.

The rake is explicitly **pre-mounted**; selection and pickup are not tested.
The baseline has no tool, so this compares combined capability and strategy,
not an isolated policy effect. “Out of reach” means beyond a configured 0.54 m
controller x envelope, not a proof of the arm's global kinematic reach limit.
The short-tool case defers. Retrieval is a checkpoint at 6.6 s, not final
retention in the workspace. No shoe is welded, teleported or externally forced
during rollout.

## What failed after retrieval

- **Light shoe:** both fingers contact it, but it never meaningfully leaves the
  table. Bilateral contact ends at 10.944 s; this is failed acquisition.
- **Offset shoe:** it rotates roughly 29 degrees during retrieval, only one
  finger contacts during acquisition, and it ends at x=0.5624 m outside the
  configured envelope. Reaching the earlier checkpoint did not remain true.
- **Heavy shoe:** it achieves an unsupported bilateral hold for roughly 606 ms,
  then tips and loses the grip. World support returns at 12.454 s. Its raised
  origin alone would misleadingly suggest a lift while its toe is supported.

The scorer requires the **final 500 ms** to satisfy height, bilateral finger
contact, and absence of rake/world support. None passes. The current controller
checks lift success at the end, not continuously: the help request follows
the first bilateral-contact break by 1.656 s for the light case and 204 ms for
the heavy case. These simulated intervals do not measure hardware latency.

The next development step is to recheck pose and workspace margin after clearing
the rake, require bilateral acquisition before a short test lift, verify that
the shoe follows while table support unloads, and monitor the grip throughout
the lift. New policies need a new untouched confirmation cohort; this result
remains a regression record.

## Speed and evidence

On the tested Darwin arm64 host, Python 3.12.9 and MuJoCo 3.15.0, 108.4 simulated
seconds took 5.496 s of summed native execution time, including initialization
and recording (about **19.7× simulated time**). Export added 12.970 s. Rendering,
network downloads, real perception and actuator latency are excluded. This is
one sequential run, not a universal hardware minimum or performance guarantee.
Native execution is CPU-only; browser replay requires WebGL2. Initial package
and model downloads need network access; native reruns can then be offline.

Physics observations are retained every 2 ms, commands/replay poses every
20 ms. Trial receipts contain full native traces; motion files retain joint
state, issued controls, requested targets and body poses. Every failed task
remains visible. The browser plays recorded native poses; it does not run
physics. No raw human footage or credentials are included.

This directory contains byte-identical compact [results](benchmark.json),
[frozen protocol](protocol.json), and [full export inventory](integrity.json).
All trial receipts, compressed geometry and motion are in the linked research
and Hugging Face export. Checksums establish artifact consistency, not physical
accuracy or authenticity.

Validation: 19 native/evidence tests, 13 publication tests, pinned reproducible
viewer build, desktop and 390/320 px browser checks. GitHub CI reruns development
cases only and retains unsuccessful attempts. DVIDIA code and synthetic objects
are MIT; upstream PiPER, MuJoCo, renderer and font notices are preserved.

## Reproduce

```sh
git clone https://github.com/Dvidia-Inference/dvidia-training.git
cd dvidia-training
git checkout 756306a2beda60ac54567971626080a56227e781
python3.12 -m venv .venv
.venv/bin/python -m pip install '.[arm]'
.venv/bin/dvidia-robot-sim fetch piper --output robot-assets
.venv/bin/dvidia-prerequisites --assets robot-assets --output results \
  --split heldout --source-commit 756306a2beda60ac54567971626080a56227e781
.venv/bin/python -m http.server 8000 --bind 127.0.0.1 --directory results
```

Use `--split development` when changing a controller. Never overwrite the frozen
results or use this confirmation cohort as an unseen test after tuning on it.

---
title: DVIDIA Robot Simulation Lab
emoji: 🤖
colorFrom: gray
colorTo: green
sdk: static
app_file: index.html
license: mit
pinned: false
tags:
  - robotics
  - mujoco
  - simulation
  - open-source
---

# DVIDIA Robot Simulation Lab

Explore recorded native robot simulation, inspect every attempted task, and run
the open-source tools on your own computer or server.

This Space contains the actual recorded UR5e + Robotiq 2F-85 rigid-block task:
native meshes, full body poses, requested commands, grasp/carry/release attempts,
matched open-jaw and hold controls, and the complete sealed source receipt.
The browser replays recorded state; **it does not rerun physics**.

{{BENCHMARK_SUMMARY}}

These are small synthetic scenarios using an authored controller with privileged
simulator state. They do not establish learned skills, human-video action recovery,
shoe manipulation, hardware reliability, real-time camera reaction, or physical
transfer. Replay selects an explicitly declared excerpt; the original full
denominator and every attempt remain under `replay/receipt/`.

## Install and reproduce locally

```sh
python3 -m venv .venv
.venv/bin/python -m pip install "dvidia-training[arm] @ git+https://github.com/Dvidia-Inference/dvidia-training.git@{{SOURCE_COMMIT}}"
.venv/bin/dvidia-embodiment fetch-assets --output runs/reference-assets
.venv/bin/dvidia-reference-task benchmark runs/reference-assets --output runs/grasp-development
.venv/bin/dvidia-reference-task inspect runs/grasp-development
.venv/bin/dvidia-reference-task export-replay runs/reference-assets runs/grasp-development --output runs/grasp-replay
```

Use Python 3.12 or later and a platform supported by the pinned MuJoCo 3.15.0 and
NumPy dependencies. The published native task ran on a CPU with no GPU. Native
physics and learning are separate workloads; no minimum RAM or speed guarantee
has been established for this task. WebGL2 is required only for the interactive
browser viewer. Fetch dependencies and pinned assets once, then native simulation
and exported replay work offline through a local HTTP server. The complete
[local task guide](https://github.com/Dvidia-Inference/dvidia-training/blob/{{SOURCE_COMMIT}}/docs/native-task-benchmark.md)
describes controls, scoring and measured timing boundaries.

## Hardware profiles

UR5e + 2F-85 has the recorded native contact-task evidence above. Additional arms
are listed with explicit implementation and evidence status in the gallery;
an upcoming listing is not a verified simulator or a hardware adapter. No
Space entry sends commands to a physical robot.

{{PROFILE_SUMMARY}}

{{PIPER_TASK_SUMMARY}}

{{PIPER_GRIP_SUMMARY}}

{{PIPER_CONTACT_SUMMARY}}

The reviewed publication now permits at most **256 MiB of stored files**, so the v0.3
comparison can retain all previous evidence alongside its 84 new full attempts.
The **32 MiB per-file limit** and bounded single-member gzip decoding remain in place.
This is a packaging limit, not a hardware or training memory requirement.

## Provenance and licensing

The native export, its complete receipt, model license files, upstream notices,
and original integrity bytes are retained without rewriting. See
`provenance.json`, `replay/replay-integrity.json`, `replay/receipt/integrity.json`,
`replay/licenses/` and `replay/reference_replay.third-party.txt`. Hashes establish
artifact consistency, not authenticity or physical qualification.

DVIDIA task/exporter/viewer software and bundled Three.js are MIT. The native
UR5e model is BSD-3-Clause; the Robotiq 2F-85 model is BSD-2-Clause. Component
license notices apply to their own files; the Space card's MIT label does not
relicense upstream model assets. Additional profile assets retain their own notices.
No neural policy weights, private footage, credentials or paid server compute
are included.

[DVIDIA](https://dvidia.org/) · [Research](https://research.dvidia.org/) ·
[GitHub](https://github.com/Dvidia-Inference/dvidia-training) ·
[@iammrriver](https://x.com/iammrriver) · [hello@dvidia.org](mailto:hello@dvidia.org)

# DVIDIA task environments v0.2

A bounded environment compiler for a pinned PiPER, one to four rigid objects and
an open-top bin. The local editor and `dvidia-environment` Terminal commands
author the same recipe, explicit USD and native MuJoCo scene.

| Frozen cohort | Expected behavior observed |
|---|---:|
| Passive completion fixtures | 6/6 accepted |
| Incomplete/false-completion controls | 6/6 rejected |
| Invalid initial layouts | 4/4 rejected before simulation |

All 12 native attempts are valid and retained. An independent read-only auditor
reconstructs **13,262 observations at 1 ms and 1,337 replay frames at 10 ms with
zero mismatches**. Its largest object-position Float32 roundoff is
`2.8971703702751483e-08` m. This checks recorded evidence consistency, not physical
truth or sensor accuracy.

The six accepted cases cover one box, one original rigid shoe surrogate, two
boxes, a box and shoe, a rotated bin, and four separate objects. The six negative
cases cover outside placement, overhanging volume, missing floor support,
excessive speed, insufficient dwell and one required object remaining outside.
Two controls explicitly disable gravity; the speed control starts at 0.3 m/s
with zero authored friction. No state overwrite occurs during stepping. All five
weakened verifiers produce at least one false completion in this fixed roster.

The native run records **13.25 simulated seconds in 16.550 wall seconds**,
including native compilation, goal checks, recording, compression and output.
This is 0.801 simulated seconds per wall second on a shared Apple arm64 host;
preparation, cameras, training and publication are excluded. It is not an
isolated physics-engine speed or hard real-time result. Identical robot mesh
geometry is stored once by content hash.

The compiler/protocol were published at commit `14759c5` before confirmation
execution. [protocol.json](protocol.json), [input-inventory.json](input-inventory.json)
and [preparation-seal.json](preparation-seal.json) preserve the exact frozen
identities. Protocol SHA-256:
`7414b6576ce2e57b77622f9173f39b9ecf320ee445ae6390d8714848659923e2`.
The first prepared roster was superseded before execution to deduplicate mesh
storage; it was not a scored run. Development previews and browser fixtures are
separate from the 16-case registered confirmation.

The arm holds its home controls. These are **environment and verifier checks**,
not learned robot placements. Shoe geometry and mechanics are synthetic rigid
assumptions; arbitrary USD composition, calibrated deformation, camera-based
control and physical transfer remain open.

[Interactive recorded checks](https://dvidia-robot-simulation-lab.static.hf.space/task-environments/v02/index.html) ·
[Complete evidence](https://huggingface.co/datasets/Dvidia/task-environments-v02) ·
[Install and reproduce](../../docs/task-environments-v02.md) ·
[Research note](https://research.dvidia.org/papers/task-environments-v02/).

# PiPER contact confirmation · 11 October 2026

This folder is a **summary mirror**, not a standalone inspectable task bundle.
The original integrity file names all source files, including attempts that are
kept in the [complete Hugging Face bundle](https://huggingface.co/spaces/Dvidia/robot-simulation-lab/tree/b07b6e0e95a19158384734d140abb36a9adb670b/tasks/piper).
Do not drop missing attempts or rewrite that seal to inspect this subset.

Frozen source: `c9a72584b4f35c38ad2616894f56da719bba0182`. The installed wheel
matched source hashes and ran with network denied: seed 642103, two nominal
scenes and one low-friction scene, three controls per scene. All nine attempts
are numerically valid; each retains 20 simulated seconds, 1,000 issued commands
and 1,001 state boundaries.

| Condition | Placement | Open-jaw replay | Hold |
| --- | ---: | ---: | ---: |
| Nominal friction 0.8 | 2/2 | 0/2 | 0/2 |
| Low friction 0.15 | 0/1 | 0/1 | 0/1 |

The block is 40 mm wide and 40 g, with a support surface 60 mm above the arm's
base plane and about 120 mm of required horizontal transfer. Native servos,
force limits and robot gravity compensation are unchanged. Grasp/carry/release
is authored with privileged simulator state. There is no object weld or
post-reset pose assignment. Position, not final object orientation, is scored.

Nominal final position error was 4.643–4.682 mm in simulator truth; maximum
lift was 97.842–97.911 mm. The first successful dwell completed at
15.112–15.132 simulated seconds. The slippery block failed to lift; its maximum
height change was 0.132 mm. The full 20-second traces remain even after success.

Complete attempts took 1.386–2.285 wall seconds on this host, including asset
checks, native compilation, reset, planning, physics and trace construction.
Per-attempt control computation p95 ranged 1.755–2.986 ms. **One of 9,000
control windows exceeded 20 ms** (28.182 ms in the first nominal placement).
That outlier remains. Rendering, disk serialization, sensor/transport latency
and learned inference are outside these timings; host load was uncontrolled.
The process lifetime peak RSS before final inspection/export was 759,169,024
bytes. This is not minimum RAM or a real-time guarantee.

Runtime: macOS 26.5.2 arm64, 10 logical CPUs, Python 3.12.9, MuJoCo 3.15.0,
NumPy 2.5.3; CPU workload with one requested BLAS/OpenMP thread and no GPU.

Development preceded this frozen cohort. The first high-support/downward-grasp
attempt failed at a wrist limit and contacted the table. A low support and an
explicit pitched carry orientation enabled later nominal and perturbed runs;
low-friction failure was retained. Those tuned development runs are not part of
this nine-attempt denominator, and the small synthetic confirmation cohort is
not proof of population reliability or independent physical trials.

[Run the pilot](../../docs/piper-contact-pilot.md) ·
[Report](report.json) · [Frozen protocol](protocol.json)

No learned policy, shoe manipulation or physical hardware is qualified.

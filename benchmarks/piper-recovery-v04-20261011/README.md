# PiPER contact recovery v0.4 — frozen confirmation

Fresh seed **941827**, runtime [`ccf8f3d1c2a1977da855ba8f601499aa543a7b6c`](https://github.com/Dvidia-Inference/dvidia-training/tree/ccf8f3d1c2a1977da855ba8f601499aa543a7b6c).

The fresh 84-attempt confirmation completes **4/12 placements in each mode**:
frozen v0.3 history, v0.4 carry-only and v0.4 full. All attempts are numerically
valid and all 48 negative controls fail. The two completed placements that
resume after a hold finish **2.440 and 2.640 simulated seconds sooner** in v0.4;
two other completions are unchanged. Four low-friction failures are refused
**2.88–2.90 seconds sooner**, while remaining failed placements. There are no
gained or lost completions in the matched comparison.

| Friction | Mass | v0.3 history | v0.4 carry-only | v0.4 full | Matched open-jaw + hold controls |
| --- | ---: | ---: | ---: | ---: | ---: |
|0.8|40 g|2/2|2/2|2/2|0/8|
|0.8|60 g|0/2|0/2|0/2|0/8|
|0.55|40 g|2/2|2/2|2/2|0/8|
|0.55|60 g|0/2|0/2|0/2|0/8|
|0.15|40 g|0/2|0/2|0/2|0/8|
|0.15|60 g|0/2|0/2|0/2|0/8|
|Total|—|4/12|4/12|4/12|0/48|

The new nominal-friction 60 g failures regain bilateral contact during the hold,
but the object settles below the unchanged 6 mm resume floor before sustaining
100 ms of eligible dwell. Both remain suspended and time out without resuming.
Their full native states and commands are equal across all three modes. These
fresh failures expose layout sensitivity; the earlier 6/12 v0.3 result is a
different cohort and is not reused as a confirmation result here.

The two successful resumed tasks keep the same 220 and 240 ms holds; the improvement
comes after settling, from restoring requested movement. The mean paired time
change over all four shared successes is −1.270 simulated seconds. Four low-friction
refusals skip the unchanged retry, but do not become successful placements. The
moderate-friction 60 g cases retain the original unsuccessful retry.

## Complete original evidence

`receipt/` is the complete, byte-identical 90-file benchmark bundle: 84 compressed
attempts, protocol, report JSON/HTML, original seal and MIT model notices.
All 84 attempts retain 20 simulated seconds, 1,000 commands and 1,001 native states.
Every raw per-decision computation time is included; 0/84,000 recorded windows
exceed 20 ms on this host. All warnings and prohibited-contact counts are zero.

- [Readable paired report](receipt/report.html)
- [Every outcome and paired timing](receipt/report.json)
- [Frozen protocol](receipt/protocol.json)
- [Stored and decoded hashes](receipt/integrity.json)
- [Execution boundary](execution.json)
- [Installed runtime source check](installed-runtime.json)
- [Publication provenance](provenance.json)

Seal-listed files total 58,450,596 stored bytes and 327,061,224 bytes with attempts
decoded. Stored report SHA-256 is
`899a05b44e9b76a597fb2683d433e445d84ae485ed95d15de463f3dab3173164`;
original seal SHA-256 is
`b0f6c0302fce5a44f4afe4bbe2451875b1c7a049340f08911465171958ec5156`.
Execution, installed-runtime and publication records are outside the original
bundle seal. Their presence does not rewrite or selectively shorten any attempt.

The installed source archive runs with OS process-level network denial and local
pinned assets. Whole-command wall time is 133.629 seconds for 1,680 simulated seconds;
it includes evaluation, compression and inspection. Attempt wall times are
1.175–1.728 seconds, per-attempt control p95 is 1.087–3.181 ms, and the worst raw
control window is 13.473 ms. The host is macOS 26.5.2 arm64, 10 logical CPUs,
Python 3.12.9, MuJoCo 3.15.0 and NumPy 2.5.3; no GPU is used, and OpenBLAS/OpenMP each
use one thread. Process high-water RSS is 727.19 MiB before inspection; the child
lifetime including inspection reaches 738.41 MiB. Host load is uncontrolled.

These are software computation times and measured memory peaks, not physical
sensor-to-actuator latency or universal minimum hardware requirements. Simulation
contacts, geometry and material values remain uncalibrated; this is an authored
privileged-state controller, not a learned Skillspace skill or hardware qualification.
Numerical validity does not imply successful placement. Original DVIDIA code and
the pinned PiPER model retain MIT terms.

[Install and inspect locally](../../docs/piper-contact-recovery-v04.md) · [Original six-failure diagnosis](../piper-recovery-forensics-20261011/README.md) · [Separate development archive](../piper-recovery-development-20261011/README.md) · [Research publication](https://research.dvidia.org/papers/piper-contact-recovery-v04/) · [Robot Simulation Lab](https://huggingface.co/spaces/Dvidia/robot-simulation-lab)


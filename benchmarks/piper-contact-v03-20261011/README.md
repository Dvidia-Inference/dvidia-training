# PiPER contact history v0.3 · 11 October 2026

In this frozen synthetic cohort, contact history completed **6/12 placements**,
versus **4/12 original** and **4/12 probe-only** placements. Both additional
successes were 60 g blocks at friction 0.8. All **84 native attempts** are valid;
all **48 negative controls** fail. These related scenes do not estimate physical
reliability or settling/slip detector accuracy.

| Friction | Mass | Original | Probe only | Contact history |
| --- | ---: | ---: | ---: | ---: |
| 0.8 | 40 g | 2/2 | 2/2 | 2/2 |
| 0.8 | 60 g | 0/2 | 0/2 | 2/2 |
| 0.55 | 40 g | 2/2 | 2/2 | 2/2 |
| 0.55 | 60 g | 0/2 | 0/2 | 0/2 |
| 0.15 | 40 g | 0/2 | 0/2 | 0/2 |
| 0.15 | 60 g | 0/2 | 0/2 | 0/2 |

Each scene has three separate matched requested-arm/open-jaw replays and one
initial-arm hold. Their complete command tapes and all failures are retained.
The new controller keeps the original native mechanics, gains, force caps and
scorer. Probe-only uses the identical acquisition test/retry as contact history,
then removes postqualification carry monitoring only.

Four history attempts resumed after closed reassessment and completed the task.
They contain five holds and five resume events; one heavier-block task needed
two holds. This is observed recovery within those attempts. Only two placements
improve over both paired alternatives; the two moderate-friction 40 g tasks
already succeed without intervention. They take 2.837 and 2.903 additional
simulated seconds versus probe-only. Six history acquisition retries all fail.

All five resume windows show loaded, unsupported contact. Four pass a separate
recorded absolute TCP peak-speed threshold of 10 mm/s. The first resume in the
heavier-block layout01 case fails that low-motion check and is followed by the
second hold. Relative settling and low absolute movement are different measures.

Requested target scaling is not a physical speed limit. Two slowdown phases end
early after 80 ms and 20 ms on bilateral contact loss; their 100 ms after-windows
include part of the hold. Across five holds, actual mean TCP speed during the
first 100 ms is 11.248–33.074 mm/s, and mean speed across the whole 220–240 ms
hold is 7.211–20.503 mm/s. Native inertia remains. Full before/after and intervening
TCP/object motion windows are recorded rather than assuming an instantaneous stop.

## Reproduce and inspect

[Installation and local run guide](../../docs/piper-contact-memory-v03.md).
Inspect the complete original bundle, not this parent directory:

```sh
dvidia-piper-contact inspect benchmarks/piper-contact-v03-20261011/receipt
```

Frozen runtime: `c8868c873f075116b71d908d464890e09778726b`.
Fresh confirmation seed: **813407**. Two shared XY layouts, three frictions,
two masses and seven variants produce 84 attempts. Each retains 20 simulated
seconds, 1,000 issued commands and 1,001 native states. All eleven installed-wheel
runtime sources matched the frozen source before execution with OS process-level
network access denied. The complete byte-identical 90-file receipt includes all
84 deterministic gzip JSON trajectories, protocol, report, original seal and MIT
model notices. Execution and publication records remain outside that seal.

The entire installed command simulated 1,680 seconds in **213.620 wall seconds**,
including startup, evaluation, compression, report output and final inspection.
Individual attempts took 1.269–5.017 seconds. Per-attempt control p95 ranged
1.369–8.316 ms, but **75 of 84,000** computation windows exceeded 20 ms; the
maximum was **225.096 ms**. No warnings or prohibited-contact ticks occurred.
These are CPU computation measurements, not sensor-to-actuator reaction times.

Measured host: macOS 26.5.2 arm64, ten reported logical CPUs, Python 3.12.9,
MuJoCo 3.15.0, NumPy 2.5.3, one OpenBLAS/OpenMP thread, no GPU workload.
Process peak RSS was 808,239,104 bytes; the whole-command child-process measure
reported the same high-water value. This is not a minimum RAM specification.
Host load was uncontrolled. Every attempt retains timing summaries, maxima and
miss counts; raw per-decision wall-time samples are not retained.

The [13-attempt development appendix](../piper-contact-development-20261011/README.md)
retains every exploratory trace and source snapshot separately. No controller
threshold was tuned on this confirmation cohort. This controller uses privileged
simulator poses/contact forces; it does not learn a skill from footage, infer
material properties, simulate deformation, implement calibrated sensing or
qualify a physical robot.

[Full readable report](receipt/report.html) · [Exact report](receipt/report.json) ·
[Execution measurement](execution.json) · [Source provenance](provenance.json) ·
[Research article](https://research.dvidia.org/papers/piper-contact-history-v03/) ·
[Hugging Face lab](https://huggingface.co/spaces/Dvidia/robot-simulation-lab)

[Immutable hosted original bundle](https://huggingface.co/spaces/Dvidia/robot-simulation-lab/tree/a6bf57fd64e61486dafaa93fd96168f953ccff80/tasks/piper-contact-v03) · [Verified publication receipt](huggingface-publication.json)

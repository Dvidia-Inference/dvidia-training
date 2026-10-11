# PiPER grip check and recovery · 11 October 2026

The new guard **did not improve placement completion** in this frozen cohort:
it completed 2/6 tasks, versus 4/6 for the original controller. All 30 native
attempts are valid. Both intermediate-friction baselines succeeded while the
guard aborted both corresponding tasks; this regression is retained.

| Friction | Original placement | Guarded placement | Each open-jaw control | Common hold |
| --- | ---: | ---: | ---: | ---: |
| 0.8 | 2/2 | 2/2 | 0/2 | 0/2 |
| 0.55 | 2/2 | 0/2 | 0/2 | 0/2 |
| 0.15 | 0/2 | 0/2 | 0/2 | 0/2 |

The guard uses a slow 10 mm test lift, continued relative-motion checks, a hold
of measured arm targets on alarm and at most one supported open/recenter/retry.
The two low-friction runs never qualify acquisition. Each retries once and
then aborts; **no successful recovery is demonstrated**. The two intermediate-
friction runs qualify the short test but later abort during full lift.

Four guard qualification decisions are followed by the independent short-window
following proxy. Two of those tasks subsequently fail, so 4/4 proxy agreement
is not a claim of detector accuracy or grasp reliability. Six alarms have no
preceding sampled loss event under the separate evaluation definition, leaving
its alarm-delay field null. Actual TCP path in the following 100 ms is
6.407–6.429 mm after intermediate-friction alarms and 0.366–0.376 mm after
low-friction alarms. Holding does not instantly stop native motion.

This is the **complete original benchmark bundle** in `receipt/`, including all
30 deterministic compressed JSON trajectories, protocol, report, original seal
and MIT model notices. Inspect that subdirectory, not this parent folder:

```sh
dvidia-piper-grip inspect benchmarks/piper-grip-20261011/receipt
```

Frozen runtime source: `3ee94706f436690c6d1d8554eda9202d23620e3c`.
Confirmation seed: 731209. Two XY layouts are shared across all three friction
settings. Every attempt retains 20 simulated seconds, 1,000 commands and 1,001
states. The installed wheel matched every frozen runtime source hash and ran
with OS process-level network access denied. Original robot mechanics, forces
and the native task scorer remain unchanged. This is an authored controller
using privileged simulator observations; it establishes no learned policy,
shoe capability, sensor implementation or physical robot qualification.

Complete attempts took 1.916–7.577 wall seconds. Per-attempt control p95 ranged
2.173–16.071 ms; 53 of 30,000 computation windows exceeded 20 ms, with a maximum
of 210.045 ms. All timing outliers remain. The full installed command, including
all 600 simulated seconds, compression, report output and final inspection,
took 104.261 wall seconds. Neither timing measures sensor-to-actuator reaction.

Runtime: macOS 26.5.2 arm64, ten reported logical CPUs, Python 3.12.9,
MuJoCo 3.15.0 and NumPy 2.5.3. No GPU workload. Peak RSS was 756,400,128 bytes
before final inspection; the separate child-process measurement including the
whole command reports 878,264,320 bytes. These scopes differ and neither is a
minimum RAM specification. Host load was uncontrolled.

Development informed the controller. One retry value-handling error (an
`AttributeError`) was retained with its
source snapshot, then corrected; subsequent development included nominal
success, intermediate-friction abort and low-friction retry/abort. Those
attempts are separate from this fresh confirmation denominator.

Successful intermediate-friction baselines also exceed the guard's 3 mm drift
threshold during full lift, then retain the block and complete. The guard's
fixed displacement limit appears over-conservative, although a comparison with
the identical probe motion and no abort is still needed to isolate its effect.
The two guarded "sampled drop" events are losses of the following/height proxy;
both blocks remain held roughly 6.5 mm above their initial height at the end.
They are not verified physical drops. The next experiment should isolate the
probe from the abort intervention, then evaluate bounded slowdown, closed hold
and reassessment using drift rate and persistent contact on fresh layouts.

[Run the candidate](../../docs/piper-grip-recovery.md) ·
[Readable complete report](receipt/report.html) · [Exact report](receipt/report.json) ·
[Whole-command execution measurement](execution.json) ·
[Research article](https://research.dvidia.org/papers/piper-grip-validation-and-recovery/) ·
[Hugging Face lab](https://huggingface.co/spaces/Dvidia/robot-simulation-lab) ·
[Immutable hosted evidence](https://huggingface.co/spaces/Dvidia/robot-simulation-lab/tree/bf64e4c3592058d3cf30616f0202e7ffc6cc2b4e/tasks/piper-grip) ·
[Verified publication receipt](huggingface-publication.json)

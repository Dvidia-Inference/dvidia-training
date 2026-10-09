# Recorded motor-memory v0.2 experiment

This snapshot preserves a CPU-native **approach, settle and retain** experiment.
The learned component is a small actuator-dynamics fit. The movement profile,
speed governor, goals, completion and contact criteria are authored.

Four separate two-second calibration episodes produce **8,000 native 1 ms
transitions**. These are not 8,000 independent demonstrations or human videos.
Three stopping coupons supply an empirical braking assumption. Candidate,
protocol and runtime source hashes were frozen before the complete evaluation.
The six confirmation combinations were not executed during development; after
this run they are regression evidence for future revisions.

The measured run contains **324 attempts**: 12 cases × 3 controllers × 3 speed
tiers × 3 repeats. Every attempt ran the four-second horizon through native
physics. All 324 were numerically valid; runtime validity is separate from task
success and physical accuracy. The goal tolerance is 3 mm, speed tolerance is
5 mm/s and completion dwell is 50 ms.

## Confirmation outcomes

| Controller | All attempts | True task successes | False completion claims | Dropped loads | Median completion among verified successes | Median native error at that claim |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Slow feedback | 54 | 36 | 0 | 9 | 2.005 s | 0.101 mm |
| Fast fixed | 54 | 36 | 0 | 9 | 1.730 s | 0.0109 mm |
| Adaptive predictive | 54 | 36 | 0 | 9 | 1.275 s | 0.181 mm |

The completion and error medians use the same 36 successful authored trials per
controller. Blackout and severe-slip cases contribute the remaining 18 attempts;
they remain incomplete. Slip drops nine loads per controller. An appropriate
latched hold is not credited as task success. Development has 54 attempts per
controller, with 45 successful completions and nine missing-input holds.

Adaptive control trades some native error at the claim for shorter completion in
the aggregate. It does not win every condition. Configurations combine an
authored speed profile, predictor and governor; component ablations are needed
before attributing gains to learned prediction alone. These exact-goal native
errors are not measurements of a physical robot's accuracy.

## CPU measurements and boundaries

The serial run simulated 1,296 seconds in 205.855 summed trial wall seconds:
**6.30 simulated seconds per wall second**. The complete benchmark loop including
JSONL writing took 207.158 seconds, excluding imports, calibration and final
report serialization. Median per-trial p95 controller-call cost on confirmation
was approximately 117 microseconds for adaptive control and 50 microseconds for
the feedback baselines. These measurements are not execution-time guarantees.

The host was Darwin 25.5.0, arm64, 10 logical CPUs, Python 3.12.9, NumPy 2.5.3 and
MuJoCo 3.15.0; no GPU was used. Process-lifetime peak RSS was 111,706,112 bytes,
including imports and the run. CPU model and total RAM were not identified by
this receipt, and no minimum hardware specification was established.

## Files and reproduction

- `memory.json`: frozen fitted candidate and calibration/braking identities.
- `summary.json`: compact protocol, source/host identities, split summaries,
  frontier points and the complete run's evidence hashes.
- `attempts.jsonl`: all 324 scalar attempt receipts and trace lengths.

This folder is a compact snapshot, **not a complete inspectable run**. The full
training directory retains calibration rows; the full benchmark directory retains
every 10 ms scoring trace and controller-call timing sample. CI retains both as
the `motor-memory-approach-tasks` artifact. Reproduce from the current checkout:

```sh
.venv/bin/python -m pip install ".[arm]"
.venv/bin/dvidia-motor-memory train --output runs/my-memory
.venv/bin/dvidia-motor-memory inspect runs/my-memory
.venv/bin/dvidia-motor-memory benchmark --memory runs/my-memory --output runs/my-benchmark --repeats 3
.venv/bin/dvidia-motor-memory inspect runs/my-benchmark
```

Fresh output directories prevent replacing earlier evidence. A
`--development-only` benchmark excludes confirmation cases during development.
Cross-platform/runtime costs can differ; hashes establish consistency rather than
authenticated provenance. Repeated authored cases are correlated.

This coupon has one horizontal travel axis and pre-established parallel-jaw grip.
Commanded goals are exact, while robot-state observations can be delayed/noisy or
missing. It does not learn goal recognition, acquire a task from video, protect a
whole arm, perform recovery, or establish hardware qualification. The 0.28 m/s
tier is beyond the highest 0.24 m/s braking-calibration target; that is tested
extrapolation under an empirical envelope. Contact-force convergence remains
unresolved from v0.1. See [the framework contract](../../docs/motor-memory.md).

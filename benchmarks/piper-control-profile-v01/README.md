# PiPER contact-path profiling v0.1

This development experiment reduces allocations inside `PiperTaskEnv._contacts`:
one private scratch force array per scan, cleared before each native query, and
membership sets computed once per scan. Every 1 ms physics, force, safety and
outcome check remains active. No contact result is cached between calls.

Four uninstrumented ABBA runs on the same previously selected synthetic command
tape measured **15.70% lower pooled mean environment-step wall compute** and
**12.60% lower process CPU compute**. Both clocks improved in both matched pairs.
This is two pairs on one shared host, not a general speed estimate. Timing
excludes policy inference, setup, serialization, sensors and physical actuators.

| Run | Source | Mean wall, ms | Mean CPU, ms | Wall p95, ms |
| --- | --- | ---: | ---: | ---: |
| A1 | Baseline | 1.468496 | 1.412751 | 2.542359 |
| B1 | Candidate | 1.232893 | 1.220514 | 1.886604 |
| B2 | Candidate | 1.305327 | 1.293750 | 1.986173 |
| A2 | Baseline | 1.542341 | 1.463971 | 2.606366 |

One baseline step exceeded its nominal 20 ms budget. These are unpaced simulator
compute windows, not hardware deadlines. The diagnostic profiler identified
contact scans as the largest recorded self-time hotspot; inclusive profiler
totals overlap, and profiler timings are not the performance comparison.

## Equivalence and retained failure

Ten separate diagnostic processes cover the original tape, open-jaw, hold,
feedback telemetry without a fault, and telemetry with a scheduled jaw-opening
pulse. All 14 scheduled processes completed. Each diagnostic retained an initial
snapshot and 20,000 native ticks: 200,010 snapshots total. Independent streaming
readback exactly matched all five baseline/candidate pairs, including 19 numeric
array fields, raw and active contacts, force thresholds, cumulative contact
counts, boundary observations and independently recomputed outcome state.

The original comparison reader rejected three dense pairs because their expanded
JSON was about 74.8 MB, over its 64 MiB limit. Its `all_passed: false` receipt is
preserved. A separate bounded streaming audit checked the **same recorded files**;
there were no simulation or timing retries. Its SHA-256 is
`1a4f4a1093961b59c03a3a244d123afed35f5a5f77c98d1467a393a20c2d6fd4`.

Original and telemetry cases passed the unchanged scorer; open-jaw and hold
failed. The pulse did not cause a task failure, so this is not recovery evidence.
Recorded equivalence does not establish real-world accuracy. The prior held-out
learned result remains **5/12**, versus **12/12** for the authored teacher.

## Inspect and reproduce

The [evidence dataset](https://huggingface.co/datasets/Dvidia/piper-control-profile-v01)
contains source snapshots, protocols, every recorded attempt, original and
supplemental reports, and a guarded archive verifier/reconstructor. The archive
uses content-addressed blobs to avoid storing identical evidence twice. No robot
meshes, credentials, footage or learned weights are included.

Use Python 3.12 and the repository's `arm` optional dependencies. Provision pinned
PiPER assets using the existing robot simulator instructions. Baseline source is
`7d9faaf4962f0afd36425399c190457cce86c2b6`; the candidate is this branch. After
verifying and reconstructing the archive, use the actual evidence paths in:

```sh
python benchmarks/piper-control-profile-v01/compare.py \
  --baseline-root ../baseline --candidate-root . \
  --python .venv/bin/python --episode evidence/comparison/preserved-episode.json.gz \
  --assets robot-assets --output runs/new-control-comparison
```

`compare.py` is the frozen experimental harness, including its original 64 MiB
reader limit. A repeat may therefore retain a reader refusal after all native
attempts finish. Run `readback_audit.py --help` for the separate streaming audit;
use a new audit output path and preserve both receipts. Neither tool overwrites
prior evidence. The archive also retains the exact verifier used in this study.

For the diagnostic cProfile run, use `profile.py --help` with `PYTHONPATH` pointing
to the baseline source. It requires the exact published prior model and selected
episode digests. Raw `.pstats` files stay local because they contain machine
paths; portable profiles and complete JSON function tables are included.

This is an experimental source patch, not a replacement wheel or production
rollout. The published dev3 wheel remains unchanged. Do not reuse an installed
skill's old runtime receipt after changing its source. Broader paired scenarios
and a fresh package qualification remain necessary before promoting the patch.

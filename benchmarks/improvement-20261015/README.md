# Measured data-selection experiment

These are compact, byte-preserved receipts from the local native simulation run
on 15 October 2026. They contain no user-uploaded video, account data or API
credentials. The [result report](../../docs/skillspace-improvement-pilot-20261015.md)
records scope, provenance, actual timing and the negative improvement result.

| Training set | Nominal qualified | Stress qualified |
| --- | ---: | ---: |
| Baseline, four groups | 12/12 | 3/4 |
| General addition, five groups | 9/12 | 2/4 |
| Targeted addition, five groups | 12/12 | 3/4 |

There was no demonstrated task-completion gain over baseline. The experiment
took 138.563 seconds with zero external inference API spend; local compute
cost was not priced. These are sixteen distinct paired scenes, not 112
independent trials. No model is automatically promoted.

Open the archived result from this checkout:

```sh
PYTHONPATH=src .venv/bin/python -m dvidia_training.improvement_studio \
  benchmarks/sample-count-19047 \
  --evaluation benchmarks/improvement-20261015/benchmark
```

The archive includes the top-level summary and selection plan, frozen benchmark
protocol and candidate identities, complete per-case result rows, and prepared
source metadata needed to validate exact cohort membership. It supports receipt
inspection in the viewer. It does **not** include raw simulator traces, video,
learned weights, action sidecars or a self-contained replay bundle. References to
those files retain their original paths and will not resolve in this archive.
Full artifacts remain in the original local run directory; see the result
report for reproduction requirements and receipt hashes.

These are simulation outcomes with aligned numerical supervision. They do not
establish training from ordinary human video or physical robot competence.

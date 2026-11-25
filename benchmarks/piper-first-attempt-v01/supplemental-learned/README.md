# Frozen learned PiPER skill on unseen starts

The locally packaged learned skill completed **36/36 strict placements**, covering 18 held-out starting positions with both destinations at every start. Each attempt ran once, with one initial surveyed RGB-D frame, 1,000 decisions and 20,000 native physics ticks. The independent scorer counted zero critical-contact episodes. Median time to strict success was **14.331 simulated seconds** and median final XY error was **4.224 mm**. Maximum pad force was **1.172 N**. A headless, action-only replay reproduced all 36 command streams and physical outcomes. Each material stratum contributed 12 successful attempts.

The model came from 32 authored simulator demonstrations at 16 training positions. This supplementary test ran the existing frozen model, wrist adapter, input validation and initial-pose eligibility checks from the downloadable package. It used the same two destinations, rectangular workspace, known 40 mm upright cube and three material strata as the parent study. The motion was inferred from the initial visual position, known goal and time. The adapter chose wrist orientation from the initial camera quaternion. Encoders bounded the requested targets. There were zero teacher calls or retries.

The code and ordered roster were recorded before running this supplementary study. This is a separate evaluation of the frozen learned package alongside the original authored versus fixed-controller experiment. It establishes transfer to unseen starts in the measured simulator envelope. It does not establish placement on unfamiliar geometry, passive human-video learning, physical transfer, or an advantage over a matched controller. The same camera calibration and simulator were used in training and evaluation.

[Registration](registration.json) · [Outcomes and timing](results.json) · [Exact actuator tapes, native receipts and headless checks](receipts.zip) · [Local skill](../local-skill/README.md)

## Reproduce

From the training checkout, with pinned PiPER assets in `$PIPER_ASSETS` and a working offscreen renderer, use fresh output directories. The first command loads the exact installable terminal prediction entrypoint and validates every input before emitting the simulated targets. The second independently replays the 36 tapes using the native simulator and task scorer without rendering.

```sh
study=benchmarks/piper-first-attempt-v01/supplemental-learned
PYTHONPATH=src python "$study/run.py" --assets "$PIPER_ASSETS" --output ./learned-holdout
PYTHONPATH=src python "$study/check.py" --assets "$PIPER_ASSETS" \
  --receipt-dirs ./learned-holdout --output ./learned-holdout-check
```

The reported run used two local workers with disjoint, ordered groups; each episode received one first attempt. The frozen package ran on Python 3.12.9, NumPy 2.5.3 and MuJoCo 3.15.0 on an arm64 Mac. Timing includes the portable entrypoint's per-observation validation. Hardware, load and process startup timing can differ from the standalone Terminal benchmark.

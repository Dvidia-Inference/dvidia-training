# Faster PiPER cube tracking on a laptop

The simulated PiPER arm completed ten seconds of scripted commands in **24.20 seconds of control time**, down from 31.85 seconds on average. All three paired comparisons ran faster with the revised tracker. The tracker produced exactly the same block pose, uncertainty, timestamps and availability on the recorded images. **All 750 intervals that captured a camera frame still exceeded the 20 ms control budget.** [Native results](results.json) · [Frame replay](replay.json)

The tracker estimates the pose of a known upright orange cube from calibrated RGB and depth. The change batches the cross products for 96 candidate planes, stops scoring when a plane includes every point and compares the cube’s three coordinate slabs across all pixels. It takes the first winning plane under the original tie rule. The arm followed a scripted path above the block, so the experiment measures capture and perception during motion. A separate grasp probe measures how far the block moves when the gripper holds or lifts. [Tesla Reality View and PiPER](../../docs/reality-view-and-piper.md)

## Six fresh simulator runs

Each process recorded 500 commands, 251 RGB-D frames at 320 × 240 and 10,000 physics steps. Control time includes target computation, inverse kinematics, physics, camera rendering, tracking and durable recording. Startup and finalization are measured separately. The registered run order was baseline, candidate, candidate, baseline, baseline, candidate. [Protocol](plan.json)

| Pair (baseline / candidate run) | Baseline control | Candidate control | Time saved | Intervals over 20 ms, baseline / candidate |
|---|---:|---:|---:|---:|
| [01](run-01-baseline.json) / [02](run-02-candidate.json) | 37.56 s | 29.85 s | 20.5% | 413 / 405 |
| [04](run-04-baseline.json) / [03](run-03-candidate.json) | 33.81 s | 23.80 s | 29.6% | 393 / 333 |
| [05](run-05-baseline.json) / [06](run-06-candidate.json) | 24.18 s | 18.94 s | 21.7% | 344 / 290 |

| Measurement | Baseline | Candidate |
|---|---:|---:|
| Mean control time per episode | 31.85 s | 24.20 s |
| Mean time in tracker during control per episode | 7.25 s | 4.32 s |
| All control intervals, p50 / p95 / maximum | 54.26 / 167.20 / 868.46 ms | 40.50 / 113.41 / 729.31 ms |
| All intervals over 20 ms | 1,150 / 1,500 | 1,028 / 1,500 |
| Camera intervals, p50 / p95 / maximum | 83.87 / 214.14 / 868.46 ms | 66.88 / 137.35 / 729.31 ms |
| Camera intervals over 20 ms | 750 / 750 | 750 / 750 |
| Process peak memory range across runs | 703–853 MiB | 594–849 MiB |

Percentiles pool intervals within each variant. Each episode repeats the same motion on the same Apple M5 laptop. The registered sources and scene match within each pair. Other work on the laptop and thermal conditions were uncontrolled, and baseline control times ranged from 24.18 to 37.56 s. The 24.0% reduction in mean control time describes these six runs. The earlier recorder experiment had substantially different absolute timings. [Earlier measurement](../godot-capture-latency-v02/README.md)

## Perception and data fidelity

Replaying one previously recorded episode isolates the tracker function. Both versions received the same 251 decoded frames twice. Each version ran four warmup calls. The two trackers took turns going first on successive frames. The order for each frame reversed on the second pass. Disk reads, decoding and result comparison were outside the timer. Tracker time per frame averaged **13.24 ms baseline and 8.52 ms candidate**, a reduction of **35.6%**. The p95 fell from 32.72 to 23.06 ms. All 502 calls per version returned available estimates. Every estimate matched the one stored during the original capture, across all fields except the processing timer. The replay covers one visible block at one starting position. [Replay receipt](replay.json)

Across the three fresh baseline/candidate pairs, comparisons matched **1,530,000 native scalar values**, all 1,506 RGB/depth artifacts including encoded bytes, and 1,503 perception records. Each pair reproduced the same issued and applied commands, 500 × 269 learning features, 500 × 7 action targets, masks and split groups. Each native placement scorer reported an unsuccessful placement, as expected for a path above the block. The focused tracker and recorder suites passed 98 tests, including abstention and comparison against the original geometry equations. [Pairwise evidence](results.json) · [Test receipt](tests.json)

## Run locally

From this revision of `dvidia-training`, activate a Python 3.12 environment with the `arm` extra, provision the pinned arm assets and use a working MuJoCo graphics context. These runs used macOS 26.5.2, Python 3.12.9 and NumPy 2.5.3 with Apple Accelerate. The runner requests one native library thread and records each request. Effective thread counts were unmeasured. [Installation and graphics setup](../../docs/godot-demonstration-v02.md)

```sh
python -m pip install '.[arm]'
dvidia-robot-sim fetch piper --output ./robot-assets
git worktree add --detach ../piper-baseline 065a10f44da5d011ee67cd8e7c22e7b71fc57881
mkdir -p runs
python scripts/benchmark-piper-perception.py --baseline ../piper-baseline \
  --assets ./robot-assets --output ./runs/perception-study
```

The runner preserves each capture locally and writes the summary to `runs/perception-study/results.json`. To repeat the tracker-only benchmark, pass any complete operator capture produced by the matching baseline tracker. The command below reuses the first baseline run just generated. The published replay used confirmation run 02 from the November 11 study. The six profile receipts linked above contain individual control-interval timings and hashes of the local recordings.

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python scripts/replay-tracker-latency.py \
  --baseline ../piper-baseline/src/dvidia_training/piper_rgbd_tracker.py \
  --candidate src/dvidia_training/piper_rgbd_tracker.py \
  --capture runs/perception-study/run-01-baseline/capture \
  --output runs/perception-study/tracker-replay.json
```

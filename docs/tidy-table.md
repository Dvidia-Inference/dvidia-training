# Remember this table

Save a tidy arrangement once and use it as the goal for a simulated PiPER. Three colored, 40 g blocks share four zones on a known table. One zone acts as a spare surface. You can give each block its own home or allow A and B to exchange places.

## Run locally

Use Python 3.12 with a working MuJoCo offscreen OpenGL context. Download the preview wheel and its checksum from the [v0.1 release](https://github.com/Dvidia-Inference/dvidia-training/releases/tag/tidy-table-v0.1).

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install './dvidia_training-0.1.3.dev6-py3-none-any.whl[arm]'
dvidia-robot-sim fetch piper --output robot-assets
dvidia-training-studio --table-assets robot-assets
```

Open the Table page at `http://127.0.0.1:8270/table`. Arrange the three blocks, leave the buffer free, and select **Remember this table**. Change the arrangement and select **Tidy this table**. The Studio displays what the simulated camera observed, which moves PiPER attempted, and whether the objects settled in approved zones. OpenGL and native physics run on your computer. Later runs can reuse the downloaded arm assets offline.

Terminal and Studio read the same local records. The layout argument lists the zones for A, B and C in that order.

```sh
dvidia-train table remember --assets robot-assets --layout left,center,right --exchange-ab
dvidia-train table tidy --assets robot-assets --layout center,left,right
dvidia-train table tidy --assets robot-assets --layout left,center,buffer
dvidia-train table inspect
```

The first tidy request allows the exchanged pair. The second asks PiPER to return C from the buffer to its home. A failed simulation returns a nonzero exit code and saves the attempt.

## Benchmark

We recorded one RGB reference and registered 24 fresh layouts before confirmation. The controller received a fresh overhead RGB observation to plan each move. Breadth-first search found a shortest sequence of moves among the four zones. An authored controller attempted the next move in one persistent MuJoCo scene. The independent scorer checked contact, lift, release, settling and the whole table. All attempted moves and outcomes are retained.

| Registered condition | Strict successes | Median time to first arm motion |
| --- | ---: | ---: |
| One block in the spare zone | 8/8 | 214 ms |
| Swaps or cycles | 2/8 | 277 ms |
| Already tidy, exact reference | 4/4 | No motion |
| Already tidy, permitted A/B exchange | 4/4 | No motion |

Across the 16 shuffled tables, first measured motion took **242 ms median, 307 ms p95** on an Apple M5. The clock includes scene construction, camera rendering, planning and the first 0.001 radians of actual simulated joint movement. Asset download, interpreter startup, UI transport and historical reference capture are excluded. The eight no-op decisions took **235 ms median, 278 ms p95** and issued zero task control steps.

The six failures all timed out during descent to pick up a block in the right-hand zone, before registering a sustained bilateral lift. These are fresh object starts with up to 4 mm of per-axis jitter on one table. Scene geometry, block appearance and camera calibration are fixed. Room reconstruction, learning from human footage and physical-arm performance require separate experiments.

From a source checkout with the arm dependencies installed, reproduce the registered experiment in an empty output directory or audit the published evidence:

```sh
python scripts/benchmark-tidy-table.py --assets robot-assets --out my-table-benchmark
python scripts/audit-tidy-table.py benchmarks/tidy-table-v01/evidence
```

[Registration](../benchmarks/tidy-table-v01/registration.json) · [Per-attempt evidence](../benchmarks/tidy-table-v01/evidence/attempts.jsonl) · [Hugging Face dataset](https://huggingface.co/datasets/Dvidia/piper-tidy-table-v01) · [DVIDIA research report](https://research.dvidia.org/papers/piper-tidy-table-v01/)

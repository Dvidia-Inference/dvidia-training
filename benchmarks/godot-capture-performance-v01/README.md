# Capture performance: fixed camera geometry

We added a small, bounded cache for the RGB-D tracker's fixed camera geometry.
Across three paired runs, average control-loop time fell **4.2%**, with one pair
slightly slower. This is a preliminary optimization result, not a reliable
speedup or real-time qualification. All compared observations, commands,
physics traces and learning arrays were preserved exactly.

The task is a small scripted movement of the simulated PiPER above a rigid
block. It does not demonstrate grasping, human teleoperation or a learned skill.
Every attempt is retained; all task scorers correctly report no placement.

## What changed

The tracker used to rebuild every pixel's optical ray and transform it into the
robot base frame on every image. Those directions depend only on image size,
camera intrinsics and camera rotation. We now reuse them with exact calibration
keys. The cache contains no images, depth, object position, task state or previous
estimates. Current observations still determine visibility, occlusion, missingness
and the estimated object pose.

Two immutable arrays are retained at most: **3.516 MiB** at 320×240 if both entries
are occupied, or **14.063 MiB** at the supported 640×480 maximum. Temporary
allocations and total process memory are separate and were not measured here.
No camera cadence, native tick, command rate limit, durable write or journal
commit barrier was removed.

## Measurements

Each confirmation run contains 100 control intervals, 51 RGB-D pairs and 2,000
native ticks: two simulated seconds at 50 Hz control, 25 Hz 320×240 cameras and
1 ms physics. The order was frozen as baseline, candidate, candidate, baseline,
baseline, candidate. Each run starts a fresh process and cache.

| Pair | Baseline control time | Cached control time | Time reduction |
|---|---:|---:|---:|
| 1 | 6.287 s | 5.920 s | 5.83% |
| 2 | 6.012 s | 6.093 s | −1.34% |
| 3 | 6.159 s | 5.668 s | 7.96% |
| Mean | 6.152 s | 5.894 s | 4.20% |

Control time includes context retrieval, encoder-only IK, native advancement,
camera/perception and durable recording. It excludes startup, finalization,
inspection, sockets, Godot and deliberate real-time pacing. Full per-run
percentiles, startup and finalization times are retained in the JSON sidecars.
`total_seconds` includes kinematics construction through capture finalization;
it excludes subsequent inspection and writing the profiling receipt.

A single actual windowed check per variant completed without client errors.
Two simulated seconds took **10.700 s baseline / 10.469 s cached** after handshake;
RPC p95 was **111.520 / 113.571 ms**. Tail latency did not improve in that check.
These runs demonstrate integration, not sustained throughput or a physical robot
reaction bound. The software does not yet meet a 20 ms wall-clock control deadline.

The paired and windowed runs use one worker process with
`OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`. Earlier exploratory profiles
inherited the host thread settings, so their absolute timings are not directly
comparable. In that diagnostic profile, rendering/encoding consumed about
0.700 s and tracking 0.534 s of a 2.177 s profiled control loop; ray checks
accounted for 0.262 s within tracking. These cumulative scopes overlap.
The unprofiled diagnostic took 1.839 s; one comparison cannot isolate profiler
overhead from host variation. A separate-process warmup warms host/driver caches,
not the next process's Python or renderer instance.

Host: Apple M5, 10 logical processors, macOS 26.5.2 arm64, Python 3.12.9,
NumPy 2.5.3, MuJoCo 3.15.0 and Godot 4.7.2 compatibility OpenGL. This shared
host was not isolated from other work. Three pairs support a directional
observation, not statistical confidence or a hardware minimum recommendation.

## Correctness and installation

- **114 local test executions passed**, including randomized exact ray-depth
  parity, cache eviction and immutability, calibration changes, missing/occluded
  frames, causal export and interrupted-journal recovery.
- The three confirmation pairs matched **306,000 native scalar values**,
  **306 RGB/depth files**, and **303 perception records** exactly, excluding only
  explicitly identified source/run identities and wall-time diagnostics.
  Calibration, timestamps, masks, issued/applied actions, lifecycle, source groups,
  data types, 100×269 feature arrays and 100×7 target arrays also matched per pair.
- The windowed pair independently matched another **102,000 native values**,
  102 frame files and 101 perception records, with the same full export checks.
- A separate native replay of one candidate recording matched all **102,000
  values** and independently reconstructed the task score.
- A fresh base-package installation outside the checkout inspected the recording
  and exported/reloaded 100 rows. This tests CLI/export installation; native
  graphics used the existing simulation environment and remain tested only on Mac.

All eleven planned captures completed. The raw local evidence, including scene,
receipts and windowed screenshot, occupies about 201 MiB. No raw capture is bundled
in this small source publication. Integrity hashes bind the retained recordings.
There were no fits, human demonstrations, hardware commands or new skill claims.
The previous learned placement result remains 5/64 and unqualified.

## Reproduce

Start with the [local capture guide](../../docs/godot-demonstration-v02.md).
Use baseline commit `4ca9804c79e330221106217eac4faca582b4efa2` in a separate checkout.
The candidate source and script hashes are fixed in `candidate.json`; the
registered attempt budget and order are in `plan.json`.

From the candidate checkout, supply the baseline or candidate source directory:

```sh
PYTHONPATH=/absolute/baseline/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/profile-operator-capture.py --assets /absolute/robot-assets \
  --output /absolute/fresh/baseline-run --duration 2

PYTHONPATH=/absolute/candidate/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/profile-operator-capture.py --assets /absolute/robot-assets \
  --output /absolute/fresh/candidate-run --duration 2

python scripts/compare-operator-captures.py \
  --baseline /absolute/fresh/baseline-run/capture \
  --candidate /absolute/fresh/candidate-run/capture --output /absolute/fresh/parity.json
```

Use `--profile` only for diagnostic attribution; never combine profiled and
unprofiled samples into a speed claim. Keep all six confirmation attempts in
the declared order. The capture guard reserves 10 GiB plus each episode budget;
the original full-learning study's larger guard remains separate.

Next: isolate rendering, thread settings and file-finalization costs under a
longer, separately registered workload before claiming useful live teleoperation
latency. Human demonstration collection and held-out skill evaluation follow.

Added DVIDIA source is MIT; existing upstream robot notices remain unchanged.

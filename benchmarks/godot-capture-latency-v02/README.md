# Lossless capture latency: faster encoding and sealing, mixed control timing

We reduced repeated work in the local PiPER demonstration recorder. **Episode
finalization was 19.4% shorter on average across three pairs**, and the diagnostic
PNG encoder spent 65.5% less time compressing the same pixels. However, **mean
control-loop time was 1.3% longer overall**. This is a useful recording improvement
and a negative result for reliable live-control speed. It is not a real-time release.

All twelve registered captures completed and remain retained. Exact decoded RGB,
depth, native physics traces, commands, perception and learning exports matched.
The workload moves a simulated PiPER above a rigid block; it does not grasp or
place it. No new skill was trained, and all placement scorers correctly report failure.

## Changes

- PNG compression uses lossless level 1 instead of level 6. Pixel values, channels,
  row order and resolution remain exact; compressed files and their hashes change.
- The recorder retains the renderer's freshly allocated arrays without copying
  them again. Separate RGB and depth passes remain: the pinned depth renderer
  uses different scene flags to preserve its depth behavior.
- Each step constructs its returned context once. The top-level estimate remains
  independently copied, so callers cannot mutate sibling results or recorder state.
- Bound JSON artifacts are read once, then those exact bytes are hashed and parsed.
  This also removes the possibility of parsing different bytes on a second read.

No camera frames, native checks, journal commits, fsync barriers, validation gates
or rate limits were removed. The camera remains 320×240 RGB-D at 25 Hz, commands
at 50 Hz and native physics checks at 1 ms. No new dependency was introduced.

## Fixed experiment

The [plan](plan.json) and [candidate source hashes](candidate.json) were fixed
before collection. Baseline: `439c54966fbf1ab51aaf0654dbd122169c5e13ca`.
Four two-second diagnostic captures precede six unprofiled ten-second captures
in B–C–C–B–B–C order, followed by one actual five-second windowed Godot run per
variant. There are no replacement runs or confirmation-driven changes.

Each ten-second capture contains 500 control intervals, 251 RGB-D pairs and
10,000 native ticks. Control time includes context retrieval, encoder-only IK,
physics, camera/perception and durable recording. It excludes startup,
finalization, sockets, Godot and deliberate pacing.

| Pair | Baseline control | Candidate control | Time reduction | Baseline sealing | Candidate sealing |
|---|---:|---:|---:|---:|---:|
| 1 | 12.911 s | 12.112 s | 6.19% | 2.377 s | 2.124 s |
| 2 | 12.848 s | 12.281 s | 4.41% | 2.713 s | 2.191 s |
| 3 | 8.224 s | 10.045 s | −22.14% | 2.364 s | 1.691 s |
| Mean | 11.327 s | 11.479 s | −1.34% | 2.485 s | 2.002 s |

The third pair reverses the control-time result. Shared-host variation is material;
these three pairs do not establish a reliable speedup. No run was discarded.

| Full control interval, pooled descriptively across three runs | Baseline | Candidate |
|---|---:|---:|
| p50 | 20.39 ms | 25.98 ms |
| p95 | 45.31 ms | 43.25 ms |
| Maximum | 71.16 ms | 89.71 ms |
| Intervals over 20 ms | 751 / 1,500 | 752 / 1,500 |

Every camera-bearing control interval exceeded 20 ms in both variants. We have
not achieved sustained 50 Hz wall-clock control. The pooled intervals are
correlated observations, not 1,500 independent experiments.

In one diagnostic profile per variant, 51 PNG encodes took 0.226 s baseline and
0.078 s candidate. Combined rendering/encoding took 0.907 / 0.680 s; the native
rendering algorithm itself was not changed. Profiles carry instrumentation
overhead and overlap parent scopes; these totals must not be added together or
pooled with confirmation timings. Render/encode includes the initial frame;
the separately reported tracking series starts with control camera updates.
Diagnostic file-write timers span startup/control/finish and exclude journal appends.

Capture storage increased **2.70%**, from about 98.09 MB to 100.73 MB per ten-second
recording. This is the tradeoff for lighter lossless compression. Whole-process
peak RSS ranged from about 0.93 to 0.96 GiB across confirmation captures; it is
not incremental cache memory, GPU memory accounting or a minimum system requirement.

## Actual Godot integration

Both windowed runs completed 250 commands, saved their episodes and reported
zero client errors. The same client, scene, compiled model and project hashes
were used. Screenshot inspection confirmed the finished episode in the native UI;
manual mouse/button interaction was not tested in this study.

| Five-second simulated recording | Baseline | Candidate |
|---|---:|---:|
| Ready-to-completion wall time | 14.175 s | 13.555 s |
| Ordinary step RPC p95, excluding final sealing step | 57.48 ms | 50.30 ms |
| Ordinary step RPC maximum | 67.21 ms | 72.28 ms |
| Final step RPC, including sealing | 1,295.90 ms | 1,032.83 ms |

This single pair suggests an integration benefit but does not establish a latency
guarantee. The existing UI's final “Server step” value includes episode sealing;
the report separates that request from ordinary control. Godot frame intervals
measure process updates, not GPU render time. The interactive wall time includes
client pacing and IPC, while startup is reported separately in the receipts.

## Accuracy and reproducibility

The comparator first validates both original recordings and their integrity
bindings. Its explicit `--lossless-png` mode then checks every PNG CRC and bounded
decoded RGB value, retaining exact NPY depth bytes. Only verified PNG references
and their consequent snapshot digests are normalized. Original configuration
digests are checked before normalizing implementation hashes; all configuration
values and source inventories remain checked. Default comparison still requires
exact artifact bytes.

- Three confirmation pairs matched **1,530,000 native scalar values**, 1,506
  RGB/depth artifacts and 1,503 perception records. Each pair also matched
  500×269 features and 500×7 targets, including data types, masks, ages and groups.
- The windowed pair matched another **255,000 native values**, 252 frame artifacts
  and 251 perception records. The diagnostic pair independently passed the same checks.
- An independent native replay matched 510,000 candidate values with maximum
  absolute error zero and reconstructed the same unsuccessful task score.
- **138 local test executions passed**, including pixel/depth tampering,
  calibration changes, missing observations, branch mutation and interrupted
  journal recovery. An initial test attempt lacked a tracked example in the
  sparse checkout; it is retained, and the complete suite passed after restoring
  that fixture without changing runtime code.
- Godot coordinate/recording checks passed 7 and 29 assertions. Sandbox log and
  system-certificate warnings are retained separately; actual windowed clients
  reported no errors.
- A fresh offline base-package installation outside the checkout passed CLI,
  inspection and 500-row export/reload checks. Native graphics used the existing
  arm environment; this does not establish fresh native installation on every OS.

CI exposed an older exporter's deliberate frozen-source check. The follow-up
[compatibility change](../../docs/reference-source-compatibility.md) preserves
its original pins and admits only the exact reviewed capture revision under a
distinct identity. Historical caller pins and unknown changes still fail.
All 65 reference/cache/preflight checks passed locally; a separate
[installation check](compatibility-install.json) verified the updated helper
outside the checkout. These are additional compatibility checks, not new timing
runs. Measured runtime files and original benchmark receipts are unchanged.

Host: Apple M5, ten logical processors, macOS 26.5.2 arm64, Python 3.12.9,
NumPy 2.5.3, MuJoCo 3.15.0 and Godot 4.7.2 compatibility OpenGL. NumPy reports
Apple Accelerate. All new runs request `OPENBLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1` and `VECLIB_MAXIMUM_THREADS=1`; effective native thread count
was not measured. There is one worker process. Earlier studies used different
thread settings and workloads, so their absolute timings cannot establish an
additional speedup here. The host was shared with other work.

Raw synthetic evidence occupies about 751 MiB locally, within the registered
2 GiB budget. The 10 GiB storage reserve and original full-study resource guard
remain unchanged. Compact receipts are public; raw recordings are not bundled.
Hashes bind retained evidence. No credentials, model weights or account footage
are included. Added DVIDIA source is MIT; pinned upstream notices are unchanged.

## Reproduce and next step

Follow the [local capture guide](../../docs/godot-demonstration-v02.md) for native
dependencies and pinned assets. Use separate baseline and candidate checkouts.
Run the candidate profiling script against either source directory:

```sh
PYTHONPATH=/absolute/baseline/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 python scripts/profile-operator-capture.py \
  --assets /absolute/robot-assets --output /absolute/fresh/baseline --duration 10

PYTHONPATH=/absolute/candidate/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 python scripts/profile-operator-capture.py \
  --assets /absolute/robot-assets --output /absolute/fresh/candidate --duration 10

python scripts/compare-operator-captures.py --lossless-png \
  --baseline /absolute/fresh/baseline/capture \
  --candidate /absolute/fresh/candidate/capture --output /absolute/fresh/parity.json
```

The complete run names/order are in `plan.json`. After all registered comparisons,
`scripts/summarize-capture-latency.py` generates the compact report from retained
receipts. Use `--profile` only for diagnostic attribution.

The next architectural experiment should separate camera work from command
delivery while explicitly preserving observation timestamps, availability and
stale-frame handling. That requires a new causal scheduling contract and its
own benchmark; it was not implemented here. The existing 5/64 learned placement
result remains unqualified. Faster capture alone is not a learned or physically
qualified skill.

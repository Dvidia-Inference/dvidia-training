# Verified camera-action export cache v0.1

Repeated local training preparation can reuse the exact derived arrays from a strict export instead of rereading thousands of RGB-D and sidecar files. This optional cache preserves the existing features, issued or separately queried action labels, missing-value masks, groups and provenance. It does not change the frozen v0.1 model, study, task scorer or result. No training or native rollout was run for this change.

The interface is `dvidia-train cache`; `python -m dvidia_training.camera_action_cache` remains available. Existing `dvidia-train robot` behavior remains unchanged; a future, separately registered study must choose this cache explicitly.

## Build once, retain the pins independently

Install the checkout with the existing NumPy dependency. The study must have sealed, complete demonstration/correction stages, an exact registered runtime and no worker lock. Supply its independently retained protocol SHA-256. Use a fresh output directory outside the study; the parent must already exist.

```sh
dvidia-train cache build /path/to/sealed-study /path/to/cache \
  --protocol-sha256 APPROVED_PROTOCOL_SHA256 \
  --split train --stage demonstrations > /path/to/trusted-run-record/cache-pins.json
```

By default this selects every eligible episode in that split, in protocol order. Repeat `--attempt ID` to select an explicit ordered subset; an ineligible, duplicate or reordered ID is rejected. Demonstrations require complete episodes with a bound initial camera and successful authored-teacher outcome, matching the unchanged study exporter. Corrections require `--stage corrections --split train`, use separate teacher-query suggestions as labels, and retain actual control authority in provenance. `--blind` uses the same exporter with its existing feedback masking. Development demonstrations can be cached separately using `--split development`; confirmation cannot be exported through this interface.

The build calls the original strict exporter, checks source and episode/stage seals, and writes only:

- `features.npy`: finite C-order float32 features, including the existing availability masks and ages.
- `targets.npy`: finite C-order float64 absolute targets within the original joint/jaw bounds.
- `manifest.json`: exact protocol, split, stage and ordered episode identities; provenance; array shapes, dtypes, hashes and dataset identity; implementation and feature-contract hashes.

The single-episode exporter already produces float32 features. The existing multi-episode combiner promotes them to float64; the cache stores those originally float32 values as float32 again, preserving numeric values and the model dataset digest exactly. It does not promise identical ndarray dtype or NPY storage bytes to a combined strict export. Targets stay float64.

Files are linked atomically, their writable temporary names are removed, and their complete targets are sealed read-only; this ordering permits cleanup on Windows. The manifest is linked last. An interrupted directory is incomplete and is never overwritten or treated as a cache hit. Cache size is bounded to 128,000 samples. Before writing, the builder requires enough space for the bounded output plus a 30 GiB reserve. It does not delete recordings or allocate cloud resources.

The returned manifest, protocol, split and source pins must be kept in an independently trusted run record. A checksum stored only beside untrusted data is not authorization. Changing both a cache and its own self-reported hashes cannot satisfy the original caller pins. Replacing the independent pins with values from the changed cache defeats that guarantee.

## Load or inspect with explicit expectations

```sh
dvidia-train cache inspect /path/to/cache \
  --manifest-sha256 APPROVED_MANIFEST_SHA256 \
  --protocol-sha256 APPROVED_PROTOCOL_SHA256 \
  --split train --source-sha256 APPROVED_SOURCE_SHA256
```

`inspect` and `load` perform the same full derived-cache integrity check. Neither starts or authorizes training. The caller remains responsible for choosing eligible data and keeping training/development group sets disjoint in a newly registered experiment. The `source` command reports the current exact implementation identity; it cannot bless a cache with a different source identity.

For Python callers:

```python
from dvidia_training.camera_action_cache import load_cache

dataset = load_cache(
    "/path/to/cache",
    expected_manifest_sha256=approved_manifest_sha256,
    expected_protocol_sha256=approved_protocol_sha256,
    expected_split="train",
    expected_source_sha256=approved_source_sha256,
)
```

`dataset` has the unchanged mask-aware exporter format and is suitable for explicit downstream selection. Returned arrays are read-only. Loading verifies every cached byte, dtype, shape, finite-value bound, mask, causal episode row order and dataset hash. Unknown files, symlinks, hard links, changed/reordered/truncated arrays, split mismatches, code mismatches and wrong caller pins are rejected. NPY loading forbids pickle and bounds headers.

**A cache hit does not reverify raw recordings.** It verifies the derived snapshot that was created when the strict exporter checked those recordings. Raw source bytes can only be assumed unchanged if the caller preserves them immutably after that verification; build/load checks are not a producer signature, exclusive filesystem lock or hardware attestation. Rebuild from raw recordings when fresh raw-source validation is required. Source/feature-code changes invalidate prior caches automatically.

## Measured local preparation cost

The measured case was the first protocol-ordered training demonstration, `train-00-goal0-teacher`: 1,000 control samples. The original episode contains 6,011 files totaling 197,398,789 bytes. The derived cache is 1,137,058 bytes; raw data remains intact.

| Operation | Wall time |
|---|---:|
| Unchanged strict export | 1.636 s |
| Strict export plus fresh cache construction | 1.053 s |
| Verified cache load 1 | 2.952 ms |
| Verified cache load 2 | 2.831 ms |
| Verified cache load 3 | 3.214 ms |

Median validated load was 2.952 ms, about 554× faster than this one measured strict export. Features, targets, groups, provenance and the complete dataset digest matched exactly on all three loads. The build also exports strictly; its shorter observed time is a warm-cache/order effect, not an optimized strict exporter.

This was one episode on a shared arm64 macOS laptop, Python 3.12.9 and NumPy 2.5.3, with existing filesystem caches and no dedicated CPU reservation. CPU branding/memory sysctl metadata was unavailable in the sandbox. Process peak RSS was 163,463,168 bytes, including strict export and construction; it is not isolated cache memory. Timing includes validation/source hashes and excludes Python imports; explicit equality comparisons follow each timed load. The first reporter attempt reached cache creation and equality checks but failed on restricted host metadata; that directory is retained. The corrected reporter produced the [prior-source measurement](../benchmarks/camera-action-export-cache-v01/results-prior-source.json), bound to cache module `ad79c142b51798281c3f2ce0645662e24da8a21e778ae7785ce9c2c44102c185`. Review then corrected Windows cleanup ordering: the temporary hard-link name must stay writable until it is removed. The [current measurement](../benchmarks/camera-action-export-cache-v01/results.json) reran the same first training episode in a fresh directory, bound to module `9191f2eb9f96bd09d4f711c31966a583bd938853a874e6018f8ddebfb0bc00d8`. All prior artifacts are preserved. These are three disclosed data-preparation attempts, not a controlled speed comparison between implementations. No benchmark scenes were replaced, no models fitted and no native simulation ran.

A fresh offline `0.1.3.dev5` wheel install outside the checkout matched the current benchmark source identity, loaded/inspected its cache with independent pins, rejected the wrong split and preserved the existing robot and footage CLI help. The focused CLI/cache/export suite passed 25 tests on macOS, including the portable cleanup order and original low-disk refusal. CPU CI includes the cache tests on Ubuntu, macOS and Windows; this local verification does not substitute for a Windows CI run.

This measures repeated data preparation only. It is not evidence of faster policy inference, better grip/placement success, whole-study throughput or physical-arm readiness. The v0.1 learned result remains 5/64 strict unseen placements and unqualified.

## Complete training-data validation

A separate, single construction pass validated **all 64 teacher demonstrations and 32 mixed-control correction recordings** from the registered training split: 96,000 action samples across 32 starting-position groups. The cache used the unchanged `cc6746a95669cfd69a49e32a3e6c2e5a42ce2546` implementation. No development or confirmation recordings were exported and no model was fitted.

The four original model files provide independent, pinned dataset references. On each of three loads, the initial/full, initial/blind, corrected/full and corrected/blind dataset hashes, ordered groups and complete provenance matched those sealed references: **12/12 checks passed**. The blind variants use the unchanged exporter mask; corrections retain separately queried teacher labels. This validates the complete existing training dataset, beyond the earlier one-episode check.

| Operation | Wall time |
|---|---:|
| Build 64-demonstration cache, including strict source validation | 105.621 s |
| Build 32-correction cache, including strict source validation | 55.050 s |
| One-time construction total | 160.671 s |
| Validated load of both caches plus ordered combination, rounds 1 / 2 / 3 | 0.347 / 0.280 / 0.279 s |
| Separate four-reference checks and blind projection, rounds 1 / 2 / 3 | 0.596 / 0.491 / 0.490 s |

The derived caches occupy **108,804,852 bytes**. Their source inventory contains 577,056 files and 18,971,932,592 bytes; this is the unique corpus size, not a physical-I/O measurement. Whole-process peak RSS was 1,631,633,408 bytes across construction, loads, combination and reference checks. Total validation took 163.738 s. The original raw recordings remain intact, with the 30 GiB reserve preserved.

These are shared-host, warm-filesystem preparation measurements on the same arm64 macOS / Python 3.12.9 / NumPy 2.5.3 environment; CPU model and physical-memory metadata were unavailable. No matched strict-export baseline was rerun, so **there is no whole-pipeline speedup ratio**. The roughly 0.28 s repeated load-and-combine measurement excludes model fitting and the separately reported golden checks. It does not change inference speed, task success or qualification: the learned candidate remains **unqualified** (`qualified: false`). Cache hits still do not revalidate raw recordings.

The [full-training receipt](../benchmarks/camera-action-export-cache-v01/full-training-results.json) records every timing, cache pin and original model identity. The [portable helper](../benchmarks/camera-action-export-cache-v01/validate_full_training.py) changes only host paths into required CLI inputs; its [provenance record](../benchmarks/camera-action-export-cache-v01/full-training-script-provenance.json) distinguishes its hash from the exact original script that produced these measurements. The portable helper was not used for another measurement. It requires the exact pinned study and the independently verified dev5 environment; it refuses changed sources, wrong models, missing training recordings and an existing output directory.

```sh
cd /absolute/neutral-work-directory
env -u PYTHONPATH /absolute/dev5/bin/python -I -B \
  /absolute/dvidia-training/benchmarks/camera-action-export-cache-v01/validate_full_training.py \
  /absolute/sealed-study /absolute/fresh-output \
  --expected-venv /absolute/dev5 \
  --working-directory /absolute/neutral-work-directory
```

Keep the study sealed and give the process exclusive write access to its fresh output. This helper performs two strict cache builds and three validation rounds; it does not start training, execute a policy, run physics or download anything. Interrupted output is retained and never silently overwritten. The released dev5 wheel and tag remain unchanged by this benchmark-only addition.

## Reproduce the bounded measurement

```sh
python benchmarks/camera-action-export-cache-v01/benchmark.py \
  /path/to/sealed-study /path/to/fresh-proof-directory \
  --protocol-sha256 APPROVED_PROTOCOL_SHA256
python -m unittest tests.test_camera_action_cache tests.test_camera_cache_cli tests.test_camera_action_export tests.test_camera_action_model
```

The helper always selects the first registered training demonstration. It writes progress after each measured stage, independently retained pins, exact dataset/source identities and all three hit times. It cannot fit models or execute native simulation. An unavailable optional host metadata query is recorded without discarding measurements.

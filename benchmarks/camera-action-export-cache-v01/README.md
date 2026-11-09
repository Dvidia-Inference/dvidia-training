# Camera-action export cache: bounded local benchmark

`results.json` is the byte-identical result from one first-protocol-ordered training episode (`train-00-goal0-teacher`). No fitting, native simulation or confirmation-episode data was used.

| Operation | Wall time |
|---|---:|
| Unchanged strict export | 1.636 s |
| Strict export + cache preparation | 1.053 s |
| Three validated cache loads | 2.952 / 2.831 / 3.214 ms |

All 1,000 feature/target samples, groups, provenance and model dataset identity match exactly for every load. This episode's strict-export features are float32, as stored in the cache. Multi-episode strict export promotes float32 values to float64; caching those values as float32 preserves numeric values and model identity but does not preserve the combiner's array dtype or NPY bytes.

The median ratio was 554× for repeated preparation of this one warm shared-host episode. The 1,137,058-byte cache replaces repeated validation of an episode containing 6,011 files and 197,398,789 bytes for this use; original recordings remain intact. A cache hit does **not** reverify those raw files. No inference-speed, grip-success, task-success or hardware claim follows from these timings.

Host scope: arm64 macOS, Python 3.12.9, NumPy 2.5.3. CPU and memory sysctl metadata was unavailable in the sandbox. Peak process RSS was 163,463,168 bytes including strict export and preparation. Validation is timed; Python imports and explicit post-load equality comparison are excluded. There was no OS file-cache flush or exclusive CPU reservation. The shorter cache-build time is an order/cache effect, not a faster strict exporter.

Three data-preparation measurements are retained. The first reporter attempt completed data operations but failed on restricted optional host metadata; its local cache/pins directory remains intact. The corrected reporter persisted progress and produced the prior-source measurement in `results-prior-source.json` (SHA-256 `5e28976319740bc6da85c833084c87b0df4a48aea62f0c249a1007c9cddbc922`, cache module `ad79c142b51798281c3f2ce0645662e24da8a21e778ae7785ce9c2c44102c185`).

Review then identified a Windows portability issue: setting the temporary hard link read-only before unlinking its temporary name can prevent cleanup. The writer now unlinks the still-writable temporary name before sealing the complete target read-only. The current measurement reran the same episode in a fresh directory, bound to cache module `9191f2eb9f96bd09d4f711c31966a583bd938853a874e6018f8ddebfb0bc00d8`. Times across these shared-host runs are not a controlled before/after comparison or evidence that this cleanup fix accelerated export. There were no native/model trials or benchmark-scene replacements. Original source bytes and local directories are preserved.

The focused CLI/cache/export suite passes 25 tests, including cleanup order, no-overwrite behavior, disk-space refusal and fixtures that skip unavailable platform link privileges. The cache tests are included in the existing Ubuntu/macOS/Windows CI matrix; local execution here was macOS only. A fresh offline `0.1.3.dev5` wheel install outside the checkout loaded the new cache using independently retained benchmark pins and matched all 19 frozen source hashes plus the cache source. It also passed wrong-split refusal, original robot stages, and footage help checks without installing native simulation dependencies.

- Result SHA-256: `a4b4b1675e8d157ee9808e556e37c6c697134f529a084488b930299028084353`.
- Dataset SHA-256: `856205dec60ad1c452373fabfd1a75e72a875ff76075971ac2ed26ada8a8bdf3`.
- Protocol SHA-256: `576851e321034a0f6d3cb4d0818b3a5ca97975fcc6f2c7fa30b07ffd203a2d76`.
- `cli-proof.json` records fresh installed-wheel source, inspection/load, wrong-split refusal and existing CLI checks, outside the three timed loads. `cli-proof-prior-source.json` retains the earlier source-bound checks.
- [Usage, trust boundary and reproduction](../../docs/camera-action-export-cache-v01.md).

# Shoe asset library v0.1 — complete native evidence

Frozen benchmark source: [`441ee2871620104246ab9caeb89825f12a14b889`](https://github.com/Dvidia-Inference/dvidia-training/tree/441ee2871620104246ab9caeb89825f12a14b889).

- Six original MIT rigid shoe proxies: trainer, ankle-boot and slipper left/right pairs.
- UR5e/Robotiq, PiPER and SO-101 native scenes, portable descriptors and local 3D previews.
- 54/54 numerically valid attempts. Supported scenes: **27/27**. Disabled-table support: **0/27**; all 27 fall as required. All 54 gravity checks pass.
- No warnings or loaded non-table shoe contacts in the 20 ms recorded samples. This sampling does not rule out shorter transient contacts.
- Complete denominator: 3 robots ×3 shoe pairs ×3 seeded layouts ×2 support conditions; 108 repeated synthetic shoe instances, not independent physical specimens.

[Browse the library](library/index.html) · [Full benchmark](library/benchmark/report.html) · [Protocol](library/benchmark/protocol.json) · [Every sampled trajectory and native step duration](library/benchmark/attempts.json) · [File seal](library/library-integrity.json)

## Frozen execution

A wheel built from the exact Git source archive was installed into a fresh Python environment. Fourteen installed runtime/catalog/viewer/lock files were byte-compared with the archive before execution. `sandbox-exec` denied network access throughout the benchmark and three exports; a denied socket-bind self-check was required before attempt zero. `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` were set.

[Installed runtime](installed-runtime.json) · [Execution receipt](execution.json) · [Complete execution log](execution-log.txt) · [Reusable execution script](run-frozen.py) · [25 native/contract tests](native-tests.txt).

The source declares seed 612047 and the 54-attempt protocol before execution; the emitted protocol is written before the first attempt. Development smoke used seed 109731 separately. No outcome-triggered reruns or omitted attempts were used for this result.

## Runtime boundaries

Apple M5, 10 logical CPUs, 24 GiB RAM, macOS arm64; Python 3.12.9, MuJoCo 3.15.0, NumPy 2.5.3. Native stepping is CPU-only.

54 simulated seconds took **12.3616 wall seconds**, about **4.37×** simulated throughput including each fresh asset verification, compilation and state extraction. Run wall stops before serialization; rendering is separate. The slowest complete attempt took **6.7047s**, retained in the receipt.

Across 27,000 native `mj_step` calls, median **8.50µs**, p95 **15.75µs**, p99 **23.42µs**, maximum **111.92µs**. These exclude observation extraction, rendering, sensors, transport and actuator response. Median compilation was **65.66ms**. Peak process RSS **1,111,916,544bytes** includes allocations in this process and is not a minimum RAM specification. No real-time guarantee is made.

## Scope and reproduction

These are synthetic rigid proxies with untested mass/contact assumptions. Native loading, gravity and passive support do not establish shoe grasping, deformation, learned competence or physical transfer. Geometry is intentionally outside the held robot's immediate contacts; reachability and graspability are unqualified.

See the [installation and scene-loading guide](../../docs/asset-library.md) and [benchmark protocol](../../docs/asset-library-benchmark-protocol.md). Fetch UR5e into a fresh root first, then add PiPER and SO-101. After provisioning:

```sh
dvidia-assets benchmark --assets ./robot-assets --seed 612047 --repeats 3 --source-commit 441ee2871620104246ab9caeb89825f12a14b889 --output ./asset-benchmark
dvidia-assets inspect ./asset-benchmark
dvidia-assets scene --robot piper --assets ./robot-assets --pair all --output ./shoe-preview
python -m http.server 8360 --bind 127.0.0.1 --directory ./shoe-preview
```

The library includes exact upstream model license notices. DVIDIA tooling and original shoe proxies are MIT; robot models retain their individual licenses. The binary viewer uses recorded native body poses and performs no physics or hardware control in the browser.

## Corrected viewer source

The installable viewer and publication tooling use [`17a7c81784362697f09757712efbe00d98743501`](https://github.com/Dvidia-Inference/dvidia-training/tree/17a7c81784362697f09757712efbe00d98743501). Browser acceptance caught an animation timestamp race that could request a negative frame. Playback now clamps indices and cannot move backwards on stale callbacks. A deterministic test forces a timestamp 10 seconds old before checking start, resume and pause.

The original benchmark was **not rerun**. Only the browser bundle and its preview seals changed; all native geometry, motion, scene descriptors and model notices are byte-identical to the frozen exports. [Installed corrected wheel and preservation proof](viewer-build.json). The library records both benchmark and publication source revisions.

[Final browser acceptance](browser-acceptance.json): all three arms passed at 1440, 390 and 320 pixels, with rendered geometry, scene downloads, controls and stale-animation-timestamp regression checks. No JavaScript errors, unexpected remote imports or horizontal overflow were observed.

The final packaging and clarified setup guide use [`45bef6315c9632b33915e1ce993babd2926b8f7d`](https://github.com/Dvidia-Inference/dvidia-training/tree/45bef6315c9632b33915e1ce993babd2926b8f7d). Viewer code and all native evidence remain unchanged from the revisions above.

Static-host navigation was corrected in [`d291ee11f28b0d3c7a0b9ed062174e46fe764d15`](https://github.com/Dvidia-Inference/dvidia-training/tree/d291ee11f28b0d3c7a0b9ed062174e46fe764d15): robot and overview links use explicit `index.html` paths for Hugging Face. [Navigation proof](navigation-build.json) preserves the prior viewer receipt and binds only changed HTML and seals; all native and JavaScript bytes are unchanged.

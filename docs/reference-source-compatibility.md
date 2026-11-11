# Exact source compatibility after the capture latency change

The reference-label exporter admits two explicitly reviewed source sets. Its
original `SOURCE_PINS` remain unchanged. The second set differs only in the
exact `piper_action_capture.py` SHA-256:

| Named source revision | Capture source SHA-256 |
|---|---|
| `frozen-reference-v01` | `da037b2198cf7d2f3a9aa40fc19a4c0f5cc375a22c5dd75606e2c1856f6efc7f` |
| `capture-latency-v02` | `a2613dca764c6f1d687e00b7773a83d571e1a2449efe3140363e9ee7ee2bd0ec` |

All seven other source pins remain exact. Unknown hashes, changed source
inventories, and changed limiter/model/export code are refused. The capture
revision changes lossless PNG compression and redundant array copies; the
[latency study](../benchmarks/godot-capture-latency-v02/README.md) retains the
independent image, depth, native trace and learning-array comparisons.

Every new reference identity records the actual source hashes, its named source
revision, the current helper hash and the actual parent-cache source identity.
The parent cache still requires the original independently supplied protocol,
manifest, stage, split and implementation pins. New exports therefore have new
identities. This is no alias for an old digest and performs no migration or
requalification of existing caches, models, studies or benchmark receipts.

Use the original pinned checkout/package to reproduce or load historical
artifacts with their original caller pins. A historical expected source digest
must fail against the new helper. The training-only, causal reference contract,
original float64 servo updates and qualification labels are unchanged. This
compatibility change does not fit a policy or claim improved skill success.

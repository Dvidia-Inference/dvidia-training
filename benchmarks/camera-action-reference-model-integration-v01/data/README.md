---
license: mit
language:
- en
tags:
- robotics
- simulation
- demonstrations
- research
---

# DVIDIA PiPER reference training data

This card describes the derived training inputs for both retained reference-model candidates. This data directory contains an identity receipt; the benchmark does not redistribute raw videos, caches or reference arrays. Both candidate model files are included separately.

The data comprises 96,000 control rows from 96 existing simulation episodes in 32 training starting-position groups: 64 authored-teacher demonstrations covering both destinations, followed by 32 previously collected mixed-control corrections. These are synchronized simulated camera/action data, not recovered actions from human video. Development and confirmation episodes are excluded from fitting.

Each parent row contains 269 causal features, the issued-teacher or separately recorded teacher-query target, group ownership and provenance. Each reference export adds two C-order float64 `(N, 7)` arrays: exact previous normal servo references and feasible target endpoints. Six dimensions use radians; the seventh is the single-finger jaw coordinate in metres. Reset references come from the recording reset; later references come from the preceding step's actual normal ramp state. The original 20 updates are applied to teacher labels. For mixed control this is counterfactual normal-reference supervision, not a statement that those labels were executed.

The parent cache was built through the original strict exporter. The reference-export construction verified sealed recording metadata and caller-pinned parent data; cache/reference loads validate derived bytes without re-reading raw RGB-D frames. This integration experiment did not revalidate raw recordings. Array, protocol, stage, source, parent dataset and manifest identities are in [training-input-identity.json](../evidence/training-input-identity.json).

Source: [DVIDIA training at the benchmark commit](https://github.com/Dvidia-Inference/dvidia-training/tree/78ea4fbb4b2733e7738eb425eeecd7cb67a52195). Training uses only training-data normalization. Current teacher actions, future observations and native object/contact truth are excluded from features. Missing visual observations remain masked, and historical estimates retain their age semantics.

These bounded synthetic demonstrations do not establish transfer to shoes, arbitrary videos, other arms or physical hardware. Model training losses and a single training-trajectory replay do not establish generalization. The original dataset ownership and failure accounting are preserved.

DVIDIA-generated derived data and accompanying documentation use [MIT](../LICENSE). Upstream simulator and robot asset licenses remain separate. No raw human footage, credentials or upstream model weights are included in this benchmark.

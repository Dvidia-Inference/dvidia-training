# PiPER feedback v0.2 publication

This maintainer tool publishes three fixed public destinations:

- Model: `Dvidia/piper-block-placement-feedback-v02`.
- Dataset: `Dvidia/piper-block-placement-feedback-demonstrations-v02`.
- Existing static Space: `Dvidia/robot-simulation-lab`, only inside `learned-placement/v02/`.

Preparation and validation are offline. They do not import the Hugging Face SDK,
read credentials, call the network or modify the completed study. Use a fresh,
cache-free export of the complete study and a separate viewer projection.

The study must include `protocol.json`, `report.json`, `frozen.json`,
`base-model.json`, every attempted demonstration/development/DAgger/confirmation
directory, frozen `sources/`, `audit.json`, and `integrity.json`. The full seal
has schema `dvidia.piper-feedback-integrity/v0.2` and a `files` mapping of every
file except itself to `{bytes, sha256}`. The independent audit must have schema
`dvidia.piper-feedback-audit/v0.2`, `status: "passed"`, and exact
`protocol_sha256`, `report_sha256`, `frozen_sha256` fields. The audit must be
performed separately; this publisher does not manufacture a passing audit or
rerun native physics.

For the imported-training lineage, active collection directories are absent.
The full, unfrozen first study lives byte-for-byte in `prior-studies/study01/`,
including 64 original teachers, 32 DAgger attempts, all five candidates and their
80 development attempts. `training-parent.json` has schema
`dvidia.piper-feedback-training-parent/v1`, the old `protocol_sha256`, old
`runtime_sha256`, an exact `files` receipt map, and `teacher_sample_files`
mapping the 64 original `demonstrations/<id>/samples.npz` paths to SHA256 strings.
The active protocol's `training_parent_manifest_sha256`, the audit's
`training_parent_sha256`, and both selected candidate receipts'
`training_parent_sha256` bind that manifest. Original attempt receipts retain
their original protocol and runtime hashes. No old confirmation data may exist.
This lineage does not imply the new model trains on the preserved DAgger data;
the candidate's training recipe must disclose which original labels it uses.

The viewer must have its own `dvidia.piper-feedback-viewer/v0.2` seal, match the
same report/protocol/frozen bytes, and contain no native NPZ arrays. Native
arrays stay in the dataset; the viewer links to those files with their hashes.
Every individual upload must fit 8 MiB. Large viewer data should be divided
into bounded chunks by its exporter rather than dropping failed attempts.

```sh
python publishing/huggingface/publish_piper_feedback.py stage \
  --run runs/complete-cache-free-export \
  --viewer runs/feedback-viewer \
  --source-commit EXACT_40_HEX_SOURCE_COMMIT \
  --space-baseline EXACT_40_HEX_REVIEWED_SPACE_HEAD \
  --output runs/feedback-hf-stage
python publishing/huggingface/publish_piper_feedback.py check \
  runs/feedback-hf-stage/publication-plan.json
```

Only the following explicit command reads the existing named Keychain entry
(`dvidia-huggingface`, account `dvidia-publisher`) through the shared client:

```sh
python publishing/huggingface/publish_piper_feedback.py publish \
  runs/feedback-hf-stage/publication-plan.json
```

Model and dataset IDs must be absent on the initial run. The shared Space must
match the reviewed baseline exactly and have an unused v0.2 namespace. Existing
Space files, including its root entry point, remain byte-identical. Binary Space
files follow the pinned canonical `.gitattributes` LFS rules. Batches contain at
most 25 files and 8 MiB, and each is verified through anonymous pinned downloads
before continuing.

The local journal is atomically saved before remote writes and records each
exact pending batch. After an interrupted or uncertain result, rerun the
**identical plan**. Only that pending batch's exact public bytes may reconcile;
unexpected files, modified baseline files or unrelated head changes stop the
publisher. Never delete the journal and restart against already published IDs.
A completed `publication-receipt.json` records all final commits and verified
file counts. Publication verification is distinct from browser playback QA.

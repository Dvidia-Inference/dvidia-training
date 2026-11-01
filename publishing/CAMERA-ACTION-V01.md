# Camera-to-action evidence publication

This workflow publishes a stopped, sealed simulation study to
`Dvidia/piper-camera-action-v01`. It does not train a model, run physics, alter a
Space, or turn a failed study into a qualified skill.

Build the wheel and a `git archive` tar.gz from the reviewed source commit.
The archive must retain Git's PAX commit comment. Both artifacts must contain
the exact registered runtime bytes, including unchanged native scorers.
Provide the reviewed dataset card and the independently generated diagnostic
audit. The audit's scope is explicitly the four diagnostic episodes.
The optional `--phase-coverage` argument includes the diagnostic-only phase
coverage proof after verifying its protocol and audit bindings.

After the study stops, independently audit every sealed stage with the numeric
scorer and saved evidence. This command does not import or execute the native
environment, tracker, teacher or learned policy. It verifies frame identities,
50 Hz command and 25 Hz camera clocks, 1 ms native traces, command authority,
teacher-query separation, model identities and split membership. It reconstructs
data-quality gates and, if confirmation exists, grouped bootstrap comparisons
and the final qualification gates. Missing stages remain unrun.

```sh
PYTHONPATH=src:benchmarks/camera-action-v01 python \
  benchmarks/camera-action-v01/study_audit.py /absolute/stopped-study \
  > /absolute/full-study-audit.json
```

Pass `--study-audit /absolute/full-study-audit.json` to preparation. It must
pass, cover every currently sealed stage and match the protocol, model, stage
and episode identities. This proof supplements the diagnostic audit. An audit
passing means reported outcomes agree with saved evidence, including failures;
it does not mean the skill passed its qualification gates.
The reviewed source archive must include the exact full-study auditor and its
diagnostic byte-reader helper; use the final tooling commit while preserving
all registered runtime bytes.

Add `--hardware /absolute/hardware-native.json` to include the measured host,
software versions, native worker count and graphics/timing scope as
`hardware.json`. The field whitelist permits public CPU/model/memory details
and rejects hostname, serial, account or other arbitrary fields. The entire
JSON must fit within 16 KiB. These details describe the measured run; they do
not establish minimum hardware requirements or physical sensor latency.

```sh
PYTHONPATH=src python publishing/prepare_camera_action_v01.py \
  --study /absolute/stopped-study \
  --output /absolute/new-publication-directory \
  --audit /absolute/diagnostic-audit.json \
  --card /absolute/README.md \
  --wheel /absolute/dvidia_training.whl \
  --source-archive /absolute/reviewed-source.tar.gz \
  --source-commit EXACT_40_CHARACTER_COMMIT

python publishing/huggingface/publish_camera_action_v01.py stage \
  /absolute/new-publication-directory --plan /absolute/reviewed-plan.json
```

Review the public roster and returned plan digest before publication. The
explicit `publish` command reads only the existing macOS Keychain entry
`dvidia-huggingface`, account `dvidia-publisher`. No credential argument,
environment variable, credential file, or login cache is required.

```sh
python publishing/huggingface/publish_camera_action_v01.py publish \
  --plan /absolute/reviewed-plan.json --sha256 REVIEWED_PLAN_DIGEST \
  --cache /absolute/publication-cache

python publishing/huggingface/publish_camera_action_v01.py verify \
  --plan /absolute/reviewed-plan.json --sha256 REVIEWED_PLAN_DIGEST \
  --cache /absolute/anonymous-readback-cache
```

Preparation streams one deterministic ZIP64 archive per sealed episode,
checks each file against the original integrity manifest, and decompresses
every member to verify its bytes. It refuses a worker lock, unsealed episode,
unexpected study file, symbolic link, missing source binding, or detected
credential-shaped text. The 30 GiB disk reserve is preserved. Existing output
directories and files are never overwritten.

The publisher freezes reviewed bytes into a private temporary directory before
credential lookup, uses a single commit bound to the empty dataset's exact
parent revision, and refuses existing nonempty or private repositories.
Anonymous verification downloads every published file at the exact resulting
commit and checks its byte identity. Large files are streamed, not loaded into
one large in-memory payload.

To reconstruct the recorded study, verify `SHA256SUMS`, then extract
`study-metadata.zip` and every `episodes/**/*.zip` into one new directory.
The resulting layout contains the original protocol, stage receipts, camera
frames, encoders, issued controls, perception and teacher-query sidecars,
native traces, and each episode's original `integrity.json`. Raw files inside
an episode must not be edited or supplemented before inspection.

The dataset includes separate model JSON files only when sealed training
receipts bind those exact files. A blocked data-quality gate therefore
publishes evidence and source without inventing a learned policy. The package
retains the study's qualification result; installation itself does not award
task qualification or physical-hardware readiness.

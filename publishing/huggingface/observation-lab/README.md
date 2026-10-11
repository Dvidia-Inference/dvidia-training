# Publishing the Observation Lab workflow demo

This preparation package is separate from the original training-pilot publisher.
It stages **only an authored synthetic static Space**, intended for
`Dvidia/observation-lab`. It has no upload function, credential reader or model
dependency. Do not pass its plan to the older three-repository publisher, which
correctly refuses a different destination/layout.

## Prepare and inspect

From the training repository, using installed Python3.12+ and FFmpeg/libx264:

```sh
python3 publishing/huggingface/observation-lab/prepare.py --org Dvidia --output runs/observation-space-publication
python3 -m http.server 8294 --bind 127.0.0.1 --directory runs/observation-space-publication/space
```

Open http://127.0.0.1:8294. Check playback, sampled boxes, label/time edits,
approval/rejection, manual events, save/reload, review JSON download and reset.
Check a second clean browser context starts at revision 0, while another tab in
the same browser reports a stale revision instead of overwriting it. Blocked
browser storage should show the temporary-memory notice.

The preparation script has a fixed source allowlist; it never scans private
`runs/`, user footage, model directories or recursively copies `src/`. It copies
the exact review UI to `source/` and makes a labeled browser-only API derivative.
It generates two 8-second 640×360 schematic clips using stdlib RGB drawing and
FFmpeg/libx264. All demo state stays in browser storage or temporary memory.
Nothing is posted to a server; media requests read only the included fixtures.

Review `space/README.md`, `index.html`, `demo-adapter.js`, `provenance.json`,
`demo/provenance.json` and the exact `publication-plan.json` inventory/hashes.
The preparation budget is 2 MiB. No model weights, secrets, private clips or
commercial server dependencies are bundled. A static Space has no container;
therefore there is no container/image validation or model throughput claim.

## Maintainer publication

Use the official Hub SDK in an isolated maintainer environment and the existing
secure credential path. Create public `Dvidia/observation-lab` with
`repo_type='space'` and `space_sdk='static'`; upload only the reviewed `space/`
directory with the plan's exact inventory. Do not upload the outer staging
directory or local token files. No application token/secret, GPU allocation or
persistent shared review store is needed.

Before a remote mutation, freeze the reviewed bytes to a separate bounded
snapshot, verify every digest/inventory entry, check destination organization
membership and preserve any differing existing remote files rather than
overwriting them silently. After publication, anonymously fetch every artifact
at the returned exact commit and compare hashes; confirm the live Space starts,
playback/download work and synthetic/local-state labels remain visible.

The user explicitly authorized this public publication, but this preparation
script performs no remote calls. Actual local ONNX tool installation and model
notices are linked from the demo; its own code/data remain MIT.

The dedicated publisher validates this single-Space plan, reads the named
Keychain credential through the existing in-memory adapter, refuses differing
existing remote bytes, and anonymously verifies every file at the new commit:

```sh
python3 publishing/huggingface/publish_observation.py check runs/observation-space-publication/publication-plan.json
runs/huggingface-tools/.venv/bin/python publishing/huggingface/publish_observation.py publish runs/observation-space-publication/publication-plan.json
```

Provision the maintainer environment and Keychain entry using the parent
[publishing guide](../README.md). Never paste a token into a command or the plan.

```sh
python3 -m unittest discover -s publishing/huggingface/observation-lab -p 'test_*.py'
node publishing/huggingface/observation-lab/test_adapter.cjs
```

Official references: [Static Spaces](https://huggingface.co/docs/hub/spaces-sdks-static),
[configuration](https://huggingface.co/docs/hub/spaces-config-reference),
[Hub uploads](https://huggingface.co/docs/huggingface_hub/guides/upload).

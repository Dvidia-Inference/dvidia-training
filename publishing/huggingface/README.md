# Publishing the synthetic training pilot

These maintainer tools publish the **already-public authored synthetic v0.1.0 evidence** as a dataset, custom numeric model repository and static replay Space. They do not scan private recordings or upload arbitrary local training runs. The model card describes the single-scene evidence and limits; the demo runs saved playback, while actual training runs locally.

The training package has no Hugging Face dependency. Publishing uses an isolated environment with the official Hub SDK; the current maintainer credential adapter uses macOS Keychain.

## Credential

Create a fine-grained write token limited to the intended publishing repositories and organization. Store it in your own Terminal:

```sh
security add-generic-password -a dvidia-publisher -s dvidia-huggingface -w
```

Leave `-w` last so the token is entered at a prompt, not in command history or process arguments. Never paste the token into chat, source files or the publication plan. The publisher reads this exact Keychain item into memory, suppresses SDK debug/progress output, passes the token directly to the official SDK, and never calls `login` or writes a token cache. Keychain may request access when the publisher reads the item.

## Prepare and review

From the training repository, with the published `dvidia-research` source checkout alongside it:

```sh
python3.12 -m venv runs/huggingface-tools/.venv
runs/huggingface-tools/.venv/bin/python -m pip install -r publishing/huggingface/requirements.txt
python3 publishing/huggingface/prepare.py --org Dvidia --output runs/huggingface-publication
python3 publishing/huggingface/publish.py check runs/huggingface-publication/publication-plan.json
```

Alternatively pass `--evidence /path/to/2026-10-08-footage-pipeline-v1`. Staging verifies the frozen release inventory and hashes, copies only listed public evidence, preserves complete prepared datasets and original source archives, and substitutes the organization in reviewed card/demo templates. It corrects old display wording and adds the published same-scene open-jaw control to the Space replay at 100 ms sampling. Candidate states and recorded physics values remain unchanged; a derivative receipt records both source hashes. No new experiment is run.

Review the staged `dataset/README.md`, `model/README.md`, `space/index.html` and `publication-plan.json`. The output directory must be fresh. Publishing that plan creates public repositories; there is no paid compute request or secret in the static Space.

```sh
runs/huggingface-tools/.venv/bin/python publishing/huggingface/publish.py identity
runs/huggingface-tools/.venv/bin/python publishing/huggingface/publish.py publish runs/huggingface-publication/publication-plan.json
```

The publisher checks organization membership, refuses private or unrelated existing repositories and differing existing files, freezes the reviewed bytes before network calls, and uploads only those bytes. It reads every published file without credentials at its exact commit and checks its hash. Progress receipts contain only public URLs, commits and verification counts. API errors expose only an error type and HTTP status. SDK binary/download caches use the staging directory; a credential is never written there.

Retrying an interrupted publication is supported when destination files still match the same plan. Existing files with changed content require a separately reviewed update workflow; this initial-publication tool refuses to overwrite them.

To complete independent repositories while another download is delayed, add `--only model space` (or `--only dataset`). The entire reviewed plan still passes local inventory and hash checks; only the selected destinations are read or changed. Each invocation's receipt records the repositories it actually verified.

If the accelerated transfer backend stalls, retry the same command with `HF_HUB_DISABLE_XET=1` in its environment to use the SDK's documented fallback. Verification reads use up to four concurrent downloads at fixed commits.

Official references: [Hub uploads](https://huggingface.co/docs/huggingface_hub/guides/upload), [fine-grained token scopes](https://huggingface.co/docs/hub/security-tokens), [SDK environment settings](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables).

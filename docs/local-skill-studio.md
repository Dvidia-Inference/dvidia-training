# DVIDIA Skill Studio: local alpha

Skill Studio adds a small browser interface to the same video-goal commands used
in Terminal. It is ready for **local research use with the documented PiPER
green-block task**. It is not a production integration for arbitrary platform
videos, a learned motor policy, or a hardware controller.

## Install and launch

From a checkout that includes this feature, with Python 3.12+ and FFmpeg/ffprobe
on PATH:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[arm]'
dvidia-skill-studio --fetch-assets
```

On Windows, activate with `.venv\Scripts\activate`. FFmpeg and ffprobe must be
installed separately. `--fetch-assets` explicitly downloads the pinned public
PiPER simulation assets once. Existing valid assets are reused. Subsequent
launches can omit that flag and operate offline:

```sh
dvidia-skill-studio
```

Open `http://127.0.0.1:8290/`. The default workspace is `./dvidia-skills`.
Choose a different directory, port or existing asset root when needed:

```sh
dvidia-skill-studio --directory ./my-skills --assets ./robot-assets --port 8290
```

Use the supplied synthetic example first. Its camera calibration is included.
For an uploaded clip, select its calibration JSON; the app never silently uses
the example camera for another video. Creation detects a bounded planar goal.
Review it, run the PiPER simulation, and download the data-only candidate ZIP.
Import that ZIP to reopen it; raw footage is not inside it. Uploaded videos and
native attempt receipts remain in the local workspace. Sessions can reopen a
saved candidate through Import after a server restart.

## Terminal interface

The UI and CLI call the same candidate loader and executor:

```sh
dvidia-video-skill create --video ./clip.mp4 --calibration ./calibration.json \
  --output ./candidate
dvidia-video-skill inspect --bundle ./candidate --profile piper
dvidia-video-skill run --bundle ./candidate --assets ./dvidia-skills/robot-assets \
  --object-x .294 --object-y -.025 --seed 1 --output ./attempt
```

Unzip a downloaded bundle into a new directory before using `--bundle` in the
CLI. Runtime versions/source hashes and the supported profile must match.
See [candidate format and limits](video-goal-install-v01.md).

## Compute and delivery

The launcher uses Python's standard HTTP server and one spawned CPU worker. The
front end is one HTML/CSS/JavaScript file with system fonts and no framework,
CDN, telemetry or browser build step. Video uploads stream to disk in bounded
chunks; native video previews support HTTP ranges. Polling runs while a worker
job is active and pauses in a hidden tab. Candidate export is roughly 22 KB for
the included example and includes no model weights. The measured wheel for this
tested build is 4,850,292 bytes; the UI is under 36 KiB before HTTP compression.

Requirements are Python 3.12+, NumPy 2.5.3, MuJoCo 3.15.0 and FFmpeg/ffprobe.
No GPU, CUDA, cloud key or account is required. The wheel and pinned dependencies
were installed into a fresh Python 3.12.9 environment and ran the packaged
example through the installed CLI from outside the repository. That is not a
RAM or CPU minimum certification; Windows/Linux and offline provisioning still
need their own install checks. The desktop/mobile layout check is a browser
test, not native-mobile support.

## Platform integration boundary

DVIDIA's account-owned video detail under `/videos?tab=mine` is the appropriate
entry point. Its same-origin contribution-video route enforces ownership; keep
that authorization in the platform. A first integration can provide an explicit
authorized download plus this local setup guide. Users select the local clip and
its camera calibration. Offer this experimentally for compatible captures; most
platform footage does not yet satisfy the green-object/orthographic contract.

The local service intentionally rejects other origins, does not accept platform
cookies/tokens, and cannot be embedded as a hosted iframe. Do not loosen those
checks to make a website call localhost. A future bridge needs an explicit
consented export bound to the contribution ID/revision, video bytes/hash,
calibration and permission to process locally. That contract is **not implemented**
by this alpha. A hosted version would instead wrap the same Python candidate API
in the platform's authenticated, resource-limited job service, with authorized
artifact access. There is no production platform change in this release.

The UI's completion result is scored against the **detected goal**, not a
separately verified human intention. The [paired benchmark](../benchmarks/video-goal-install-v01-20261025/README.md)
keeps those references distinct. The next research gates are color-independent
object grounding, synchronized action data and a second qualified robot adapter.

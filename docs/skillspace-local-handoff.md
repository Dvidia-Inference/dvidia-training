# From Skillspace to your computer

Open one of your saved videos in DVIDIA and choose **Run locally**. Download
the video package, open Skill Studio, and choose or drop that ZIP. The selected
video and its skillspace context appear together; no second video search is
needed. The download route checks the current account and source revision before
and after streaming the original. The local importer verifies the saved bytes.

The package is a private local copy. It does not share the video, change its
review status, authorize a public training release, or contain a trained policy.

## Open in the UI

Install the [local alpha](local-skill-studio.md), then run:

```sh
dvidia-skill-studio --import ~/Downloads/dvidia-video-VIDEO_ID.zip --fetch-assets
```

Use the filename you actually downloaded. Renamed downloads also work. If
Skill Studio is already running, choose the ZIP in its existing window instead
of starting a second server on the same port.

The source video opens immediately. Current platform exports have **no camera
calibration**: supply the mapping for that exact capture before creating a
candidate. The included example's calibration is never silently reused. The
current observer/controller covers a calibrated green rigid proxy in PiPER
simulation; general human footage and arbitrary robot arms remain research work.

## Use Terminal

```sh
dvidia-skillspace import-video --package ./dvidia-video-VIDEO_ID.zip --output ./my-video
dvidia-skillspace inspect-video --directory ./my-video
```

The JSON output includes the verified source path, SHA-256, skillspace reference
and `calibration_required: true`. To run a compatible calibrated capture:

```sh
dvidia-video-skill create --video ./my-video/video.mp4 --calibration ./camera.json --output ./candidate
dvidia-video-skill run --bundle ./candidate --assets ./dvidia-skills/robot-assets \
  --object-x .294 --object-y -.025 --seed 1 --output ./attempt
```

MP4, MOV and WebM keep their matching extensions. The first asset setup needs
network access; source import and later simulation are local. Import never
starts training or sends hardware commands.

## Package contract

`dvidia.skillspace-video.v1` is an uncompressed ZIP with exactly `manifest.json`
and `video.mp4`, `video.mov` or `video.webm`. It carries:

- Contribution ID, source revision, skillspace ID and task title.
- Original filename, MIME type, byte count and SHA-256.
- Export time, the fixed platform origin, an explicit local-processing scope,
  and `calibration: null`.

The importer accepts at most 100 MiB of video and a 16 KiB manifest, verifies
the actual video digest, rejects extra files/fields, paths, symlinks, encrypted
or compressed entries, and installs into a fresh directory. Failed imports
preserve the current studio source and existing files. A candidate ZIP and a
video-source ZIP have different contracts and controls.

Hashes detect changed bytes; these exports are not cryptographic platform
signatures. Importing a manually constructed package does not independently
prove ownership or a license. A local copy also cannot be remotely recalled
after its platform original is removed.

The streaming download reuses the existing authenticated same-origin video route. No
platform cookie, bearer token, signed download URL, remote-fetch instruction or
executable code enters the package or local service. The local service keeps
its original loopback Host/Origin restrictions and is not embedded in the site.

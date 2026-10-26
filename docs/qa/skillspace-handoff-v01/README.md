# Skillspace local handoff verification

The video-source export and candidate bundle are separate formats. This release
adds strict intake for the source export, a `dvidia-skillspace` Terminal command,
`dvidia-skill-studio --import`, and the existing UI's source chooser/drop area.

Thirty Python tests pass: ten source-package cases, fifteen Studio HTTP/setup
cases, and five existing candidate compatibility cases. They check integrity,
ZIP metadata bounds, malformed files, failed-import isolation, origin/Host
isolation, renamed downloads, metadata filenames, stale local source edits,
calibration requirements and unchanged candidate loading.

An actual streamed ZIP produced by the platform Worker implementation imported
through the Python implementation. The checks cover the two entries, streaming
data descriptor, central-directory/EOCD offsets, CRC and actual video SHA-256.
The Worker carries the stored receipt digest; local intake checks actual bytes.
No account credential or temporary remote download URL travels in the package.

The wheel was installed into the existing isolated Python 3.12.9 environment.
From outside the source repository, its installed commands imported a source,
created a calibrated candidate and completed a valid native PiPER attempt
(`success: true`, observed wall time 4.08 s). This uses the existing authored
controller and privileged simulator state. One observed duration is not a speed
guarantee. macOS is the qualified installation in this check.

The local browser opened the synthetic Skillspace source, required its own camera
calibration, created a candidate and completed native simulation. The displayed
error against the detected destination was 5.0 mm, with 5.38 s terminal dwell.
The synthetic source is explicitly labeled; this does not validate human intent,
general video understanding or physical robot operation.

See [package digest and checks](checks.json), [local result](local-result.png),
and [the handoff guide](../../skillspace-local-handoff.md). Platform publication
and its browser acceptance are recorded separately in the private platform
release receipt.

The alpha 3 visual correction gives the preview a definite full-width grid and
height, so the intrinsic 4:3 video fits inside the player instead of being cropped.
The complete media controls remain visible. Measured desktop video/frame bounds
match at about658 ×180 CSS pixels; the page has no horizontal overflow.

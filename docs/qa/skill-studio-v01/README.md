# Local Skill Studio verification — 2026-10-25

[Checks and package digest](checks.json) · [Desktop screenshot](desktop.png) ·
[Narrow screenshot](narrow.png) · [Setup and platform boundary](../../local-skill-studio.md)

The [experimental GitHub installer](https://github.com/Dvidia-Inference/dvidia-training/releases/tag/skill-studio-v0.1-alpha.1)
is published. An anonymous download matched the exact size and SHA-256 of the
wheel tested below. No platform or Hugging Face deployment is part of this release.

Sixteen tests pass: eleven local HTTP/import/setup cases and five candidate
contract cases. They include real example decoding, exact ZIP round trip,
cross-origin/Host rejection, media ranges, failed-job isolation, source-change
invalidation, user-calibration restoration, one-worker exclusion, archive limits,
and interruption-safe asset provisioning without overwriting invalid existing data.

The 4,850,292-byte wheel was installed with pinned runtime dependencies into a
fresh Python 3.12.9 environment. The installed CLI created and ran a candidate
from its packaged video while the working directory was outside the repository.
The installed UI also completed its native PiPER example. Both use the authored
controller; no model was trained. The displayed detected-goal error was **5.0 mm**
with **5.38 s** terminal dwell. This UI score does not verify original task intent.

Browser checks covered the example, normal local upload, invalid-calibration
feedback and retry, candidate creation, actual native execution, ZIP download,
reimport, and clearing old source/result state. Desktop review is above Run;
Create and Run were at approximately 659 and 669 CSS pixels from the top in the
recorded sample state. A narrower layout was checked at the browser's observed
444-pixel CSS width with no horizontal overflow. No 320-pixel device test or
native-mobile qualification is claimed. Temporary viewport overrides were reset.

The front end is 35,873 bytes with no external assets. A worker keeps the local
HTTP service available during processing. Native run time varied with the host:
2.05 s and 3.02 s in source-server checks, 6.14 s in the fresh installed CLI, and
12.76 s for the installed browser's 20-s episode. These are individual observations,
not minimum hardware or real-time guarantees. Other operating systems, a hosted
account integration, arbitrary footage and physical robots remain unqualified.

# PiPER and SO-101 native CPU profile checks

11 October 2026. These are unloaded numerical mechanics checks, not learned
skills, object-task success, physical accuracy or a real-time hardware result.
The source is frozen at `928fed79eca09a1a5081aae7720ab78c2e5b817c`.

| Profile | Arm axes + gripper commands | Numerically valid attempts | Simulated / full-attempt wall time | Per-attempt control computation p95 |
| --- | --- | --- | --- | --- |
| AgileX PiPER | 6 + 1 | 6/6 | 4.38–14.62× | 0.446–0.500 ms |
| The Robot Studio SO101 | 5 + 1 | 6/6 | 14.73–20.17× | 0.396–0.744 ms |

Each profile retains hold, arm sweep and gripper cycle with two deterministic
repeats per case: six attempts, two simulated seconds each, a 1 ms physics step
and 20 ms command boundary. Repeats are not independent scene coverage. Every
attempt includes its complete native state, issued command tape and original
license. Both groups completed 600 measured command boundaries with zero
computation windows above 20 ms; this excludes real sensors, transport,
rendering and learned inference. The first PiPER attempt's slower wall time is
retained. Full-attempt wall time includes asset checks, native compilation and
state snapshots; file serialization is outside that measurement. Host load
was not controlled. There is no discrete-GPU workload.

Measured runtime: macOS/Darwin arm64, Python 3.12.9, MuJoCo 3.15.0 and NumPy
2.5.3. No universal minimum memory or speed requirement is established.
The serial benchmark process ran with network access denied after provisioning.

PiPER has six arm hinges, one independently commanded finger slide in metres,
and an equality-coupled second finger. SO-101 has five arm hinges and a hinge
jaw in radians. Native servo limits and upstream gravity compensation are
recorded in each attempt; neither command is a calibrated real jaw-gap or
force command. Model licenses are MIT and Apache-2.0 respectively, retained
byte-for-byte in each bundle. DVIDIA integration and upstream asset authorship
are distinct.

[Run locally](../../docs/robot-simulation-library.md) ·
[PiPER report](piper/report.html) · [PiPER JSON](piper/report.json) ·
[SO-101 report](so101/report.html) · [SO-101 JSON](so101/report.json)

The twelve focused profile tests also compare the commanded motion with a
matched hold trace, verify units/topology, and reject incompatible or corrupt
assets/receipts. Each complete subdirectory can be checked with
`dvidia_training.robot_profiles.inspect_benchmark` without native physics.

An installed wheel also passed SO-101 native audit and a separate three-case PiPER smoke run. Its imported profile source hash matched the frozen source above. This installation check is separate from the six-attempt timing groups.

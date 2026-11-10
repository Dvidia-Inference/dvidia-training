# Godot workstation v0.1: integration and timing

**Result: native-state parity passed. Physical accuracy and learned-task success
were not evaluated.** Godot displays the pinned PiPER workstation while the
unchanged MuJoCo implementation owns all dynamics. The shoe is an authored rigid
proxy, and the renderer uses solid colors rather than imported GLB textures.

Hardware: Apple M5, 10 logical processors, macOS 26.5.2 arm64. Python 3.12.9,
NumPy 2.5.3, MuJoCo 3.15.0, Godot 4.7.2 OpenGL compatibility renderer. Shared host;
these short development measurements are not isolated hardware qualification.

## Native CPU comparison

Three paired repetitions of six authored target-tracking segments were run.
Each repetition contains 600 control intervals and 12 simulated seconds. Both
paths preserve ten 2 ms native ticks per 20 ms control interval.

| Path | Simulated seconds / wall second | Per-interval p95 |
|---|---:|---:|
| Direct native Python call | 58.63–59.81 | 0.353–0.374 ms |
| Encode request → bridge → encode/decode response | 49.71–50.15 | 0.426–0.444 ms |

All **1,800 paired returned states matched exactly**, including contacts. The
six runs advanced 36,000 native ticks with zero solver warnings. These timing
numbers exclude socket transport, camera rendering, Godot, model learning, and
initialization. The whole benchmark process peaked at 875,184,128 bytes RSS;
this includes retained comparison traces and is not a minimum RAM requirement.
The JSON path measures its additional serialization and validation work; it is
not a physics speedup. Source digests and exact timings are in `cpu-parity.json`.

## Actual windowed Godot execution

The final client sent 100 commands across three bounded target/jaw settings.
It advanced exactly 2.00 simulated seconds in **2.264 seconds** after the handshake.
This viewer deliberately paces commands around 50 Hz. Its wall time must not be
compared with unpaced CPU throughput as a rendering slowdown ratio.

- 100 accepted commands, zero protocol errors, zero hidden retries.
- Godot process frame interval p50 8.00 ms, p95 10.91 ms; this is **not GPU time**.
- Socket request-to-parsed-response latency p50 6.109 ms, p95 16.826 ms.
- Independent native replay checked 101 states and 12,387 numeric values;
  maximum absolute discrepancy was 5.11e-15, below the declared 1e-9 tolerance.
- The receipt pins the exact client script, project, model and scene integrity.

Two earlier windowed development runs took 2.441 and 2.062 seconds respectively.
Their summaries remain in `development-timing-summary.json`; the latest run is
reported above rather than selecting the fastest. Lighting and command pacing
were corrected between the first and second runs. Shared-host variation and
changes between these runs prevent a controlled speedup claim.

The initial sandboxed headless client could not open a localhost connection and
advanced zero ticks (`headless-client.json`). Scoped local graphics/network access
enabled the subsequent real windowed runs. The external native UI automation
could not attach to the Godot application, so manual mouse/button interaction was
not independently verified. Automated windowed stepping and actual screenshots
were verified; server reset and no-step keepalive behavior are covered by unit tests.

## Verification and limits

19 focused Python tests passed: 16 bridge tests, plus unchanged actuator/mass/
inertia contracts, deterministic reset, and native gravity/support-force checks.
Seven Godot coordinate assertions passed. Independent read-only code review
checked protocol authority, frame conversion, clock and digest validation.

These are numerical/integration checks against our existing simulator. They do
not measure agreement with physical hardware, validate shoe deformation, prove
successful grasp/placement, or change the existing learned-skill qualification.
The next useful gate is a calibrated rigid-object contact suite followed by a
matched learned-control benchmark. See `docs/godot-simulation-roadmap.md` and
`docs/godot-workstation-v01.md` for scope and reproduction commands.

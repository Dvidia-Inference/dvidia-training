# Frozen continuous-grip v0.2 audit

The fresh cohort is **2/10 for both controllers, on the same two cases**
(`fresh-05` and `fresh-08`). There are zero paired gains and zero paired losses.
Both controllers appropriately defer the two declared short-tool cases. Deferral
does not count as task completion. All 20 attempts remain valid and retained.

The new evidence demonstrates monitoring and bounded response, not better task
completion:

- Nine bilateral-contact-loss alarms follow a trace-verified native loss origin
  by **20–52 ms, median 30 ms**. All have three failed decision samples on the
  20 ms observation clock. Native contacts can briefly recover between those
  samples; the timing origin is the latest uninterrupted native loss segment.
- Three additional alarms report failure to acquire bilateral contact. They
  have no acquired-grip loss origin and are excluded from the latency figures.
- All 12 responses have separate recovery-request and actual native packet
  events. Each packet matches the recorded native actuator ramp. The packet is
  issued at the detector's simulation timestamp, before the following 2 ms
  physics step. This is **not zero physical stopping or response time**.
- All 12 attempts to lower and reopen the jaws record world support before
  opening. None produces a subsequent completed retained lift in this cohort.

For four first-attempt losses, baseline and v0.2 native trajectories are
identical through the intervention. The v0.2 alarm occurs 26–48 ms after the
shared loss origin; baseline performs its scheduled terminal check 1.460–1.610 s
after that origin. The latter is a task-check interval, **not the latency of a
baseline loss detector**, and does not measure stopping distance or recovery
speed. New retries also make the failed v0.2 episodes longer.

`fresh-05` has an unalarmed strict-scoring gap: 82 native ticks (164 ms) of
non-finger robot support beginning at 19.076 s. Bilateral finger contact, height,
world-support and rake-support conditions do not fail during that gap. This
must not be described as undetected bilateral slip or a certified missed safety
alarm. The shoe later requalifies, and all final 250 ticks pass strict retention.

Audit coverage: 148,610 native ticks, 14,861 replay frames, 86 export files and
17 decoded binary payloads. Every full receipt was rescored; replay positions,
contact telemetry, stored controls, source hashes, protocol seal and file
checksums were checked. Each pair has identical exported geometry and dynamics.
Ten models were recompiled without stepping physics and matched the recorded
compiled-model hashes. No fresh simulation was rerun, and no source policy was
changed.

Frozen source: `af8933994b4626e28a0554abc782ae9d22cff06c`.
Machine-readable detail: `native-audit.json`. Reproducible audit: `native_audit.py`.
These are deterministic synthetic cases with privileged native observations,
not a measured camera-to-actuator system, learned policy or hardware safety test.

## Reproduce this audit

From a checkout of the frozen source commit, install `.[arm]` and fetch the
pinned assets with `dvidia-robot-sim fetch piper --output runs/robot-assets`.
Keep this script outside the checkout if switching revisions would remove it.
With the frozen export downloaded locally, run:

```sh
PYTHONPATH=src python path/to/native_audit.py \
  --assets runs/robot-assets \
  --input path/to/frozen/lab \
  --output runs/native-audit
```

The command reads the complete export and recompiles its models without stepping
physics. It does not run the cohort again. The `--input`, `--assets` and
`--output` options accept absolute paths or paths relative to the current working
directory. Running elsewhere also works when `PYTHONPATH` points to the frozen
checkout's `src` directory. An alternate input path changes only the audit's
`input_root` metadata; the measured results remain unchanged.

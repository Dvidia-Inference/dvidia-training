# Native contact audit receipt

**14 frozen replays verified; 249,502 native physics ticks.** A separate portable
rerun reproduced all 14 compressed native traces byte-for-byte and every
physical summary exactly. These are repeated measurements of seven controller
tapes and seven open-jaw controls from two already inspected scenes, not 28 new
policy trials. No controller or model was promoted.

- [Interpretation and local reproduction](../../docs/native-contact-audit.md)
- [Goal images, Jev and the actuator interface](../../docs/goal-images-to-actuators.md)
- [Published research](https://research.dvidia.org/papers/goal-images-physical-feedback/)
- [Verified complete-bundle downloader](https://research.dvidia.org/downloads/topics/physical-grounding/runs/2026-10-18-native-contact-audit/download-audit.py)
- [Complete public file inventory](https://research.dvidia.org/downloads/topics/physical-grounding/runs/2026-10-18-native-contact-audit/audit/integrity.json)

Implementation source freeze: `ff40985ec076eaed8421821b1244217d8502777e`.
The runtime, physical limits, actions and original success criteria are unchanged.
Ten focused tests include an exact-state native probe and tests for contact
thresholds, sampling, dwell, tampering and portable-input paths. An independent
source and result review found no high-impact methodological defect.

This compact folder preserves the original protocol, report, protocol checksum
and full file manifest. The published 59-file bundle additionally includes all
compressed original input tapes, native traces, per-episode receipts, runtime
source snapshots and upstream experiment metadata: **151,876,079 bytes**.
`integrity.json` lists the other 58 files; its own hash is recorded below.
The separate `qa.json` records tests, execution environment and portable
reproduction; it is not part of the sealed primary bundle.

| Artifact | SHA-256 |
| --- | --- |
| Protocol | `f5ded43bc66dd512e223ffea76ad51897ffea15bb016fa2c29dcd1b4fb6601e2` |
| Report | `e1c8ad0fafd2aa040d215601dcdc235a9fd010dd212400a43a450bb3bf7eec63` |
| Integrity manifest | `dc7b6091b1c680b9479ea65157c7d553043af688e159463db6174696ad5949fa` |

Baseline and smoothed students sustain the native lift predicate for at most
75 ms and 89 ms, below the unchanged 100 ms requirement. Compliance sustains it
for 947 ms but still fails placement. Missing loaded contacts in these tapes
correspond to simulated geometric separations. A successful comparator also has
brief separations, so this does not establish a one-factor failure cause or
real-world slip.

The primary audit took 54.18 wall seconds; the portable repetition took 204.60
seconds under different host load. Both include instrumentation and export;
neither is an uninstrumented throughput benchmark. Tested on macOS arm64,
Python 3.12.9, MuJoCo 3.15.0 and NumPy 2.5.3 using CPU simulation. No external
inference was used. Cross-platform exact numerical reproduction remains a gate
to test, not an established guarantee.

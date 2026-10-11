# Native grasp task: frozen confirmation

11 October 2026 · source `ac920cbf70147a7edeb0c706a9b71047f318e12f`.

The authored native UR5e/Robotiq controller completes **4/4 nominal and 2/2
low-friction placements**. Recorded-arm/open-jaw controls complete **0/6** and
hold controls **0/6**. All 18 attempts are numerically valid, with no recorded
unexpected task contacts, carry-retention loss or native warnings in the six
placement attempts. They run the full 20 seconds, including after first success.

Final simulated placement error ranges from **0.489 to 0.551 mm**. These are
native simulator measurements against its authored target, not physical accuracy.
The six synthetic layouts vary initial/target coordinates within ±15 mm of a
single configuration; the lower-friction stratum uses 0.15 rather than 0.8. This
does not establish broad manipulation, shoe competence or sufficient video data.

The confirmation process ran with network access denied. Headless stepping,
native state/contact scoring, authored pose control and trace construction used
CPU. No learned policy, rendered observations or hardware control was involved.

## Timing includes the slow attempt

Placement attempt wall time ranges from 2.97 to **29.51 seconds** for 20 simulated
seconds. The slow nominal-03 attempt has a 78.14 ms control-boundary p95, a
1.479 s worst boundary and 251/1,000 boundaries above the 20 ms computation
budget. Other placement attempts also retain their deadline misses. The host was
not controlled for unrelated activity, and the cause of this variation was not
measured. These clocks do not establish real-time operation or a minimum laptop.

The recorded process high-water resident memory is 531,054,592 bytes, before
final bundle inspection and replay export. It includes imports, native models
and traces; it is neither per-episode memory nor a minimum RAM recommendation.

## Read and reproduce

- [Public task replay and complete receipt](https://research.dvidia.org/downloads/topics/physical-grounding/runs/2026-10-11-native-grasp/replay/)
- [Exact summary](report.json), [frozen protocol](protocol.json),
  [original receipt manifest](receipt-integrity.json), [execution provenance](provenance.json).
- [Local instructions](../../docs/native-task-benchmark.md) and
  [proposed next action-learning experiment](../../docs/native-grasp-learning.md).

This compact directory keeps summaries and the original manifest. Full raw
attempt JSON is in the public replay's `receipt/attempts/` directory. To inspect
the bundle, download the entire `receipt/` directory, retain its original
`integrity.json` name, and run `dvidia-reference-task inspect PATH`. This compact
directory alone is not a complete inspectable bundle.

Reproduction uses seed 521047, four nominal and two low-friction scenes, and
`--cohort confirmation` with the recorded source/runtime and pinned assets.
Those scenes are now known regression evidence. Use fresh scenarios after
changing a candidate; a directory or seed label alone cannot establish novelty.
The simulator's compiled-model hash may differ across platforms, so retain a
new platform-specific receipt instead of replacing this result.

The 537-test suite passed before this publication. The code remains an alpha
source candidate; earlier releases and Hugging Face Observation Lab are distinct.

# Retention experiment receipts — 16 October 2026

Two frozen local simulation comparisons tested object retention before recovery,
using the original per-scene force limits, native success predicates and
18-second horizon. These files are exact copies of the completed protocols and
reports; phase two binds the phase-one protocol and report hashes.

| Folder | Candidate | Episodes | Complete wall clock |
| --- | --- | ---: | ---: |
| `phase-one` | Smooth loaded lift/transfer goal increments | 68 | 98.311345 s |
| `phase-two` | Bounded gravity-aligned compliant transfer path | 68 | 67.524272 s |
| Combined | Same 16 previously inspected cases | **136** | **165.835617 s** |

Each phase runs all 16 student cases and the teacher on known failing scene 15,
baseline versus candidate, with each trajectory's own open-jaw control. All
attempts completed without errors; all 68 controls across both phases failed.
All original baseline trajectories reproduced exactly and all candidate actions
matched before their intervention. The low-headroom gate activates only on
scene 15; the other 15 cases remain exact no-ops.

**Neither candidate repairs task completion.** Student baseline and candidates
all score 15/16; paired gains and regressions are both zero. Smoothing delays
student recovery by 0.42 s; compliant transfer delays it by 1.10 s and permits
native sustained lift, but still reaches the unchanged slip guard. Both teacher
candidates also fail. No new qualification, promotion or fresh confirmation
followed. These are repeated inspected cases, not independent generalization
trials.

Read the [measured report](../../docs/retention-experiment-20261016.md) for settings,
timestamped evidence and limits. External API spend is $0; local compute remains
unpriced. Full synthetic traces, replay pages, model weights and frozen sources
stay under ignored `runs/` directories. This compact archive contains no human
uploads, private account records or credentials.

| Artifact | SHA256 |
| --- | --- |
| Phase-one protocol | `cfcf9199220cbad3e9dc5acc50e00e95c5b449b3c4cc3ee6f714c422a5f85551` |
| Phase-one report | `4ca816d0c1f3a03aebbceb64bf2bb116b969129b07777b8f3ecb9e9428bf26ed` |
| Phase-two protocol | `cc13853612a01d2f4301a9db0dbe7b2a702577f77d396ef0c4cc3af8c4b656fc` |
| Phase-two report | `adcace792306828353dc93e4857db6cda56e7f01692b83f2d8ba19b3f84ba5fb` |

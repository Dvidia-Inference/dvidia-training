# Controlled recovery receipts — 15 October 2026

These are byte-identical protocol and result copies from two completed local
simulation experiments. The second experiment explicitly references the first
protocol and report hashes. Both plans and their source snapshots were frozen
before running.

| Receipt folder | Intervention | Native episodes | Complete wall clock |
| --- | --- | ---: | ---: |
| `phase-one` | Lower to settled table support before commanding the jaws open | 68 | 66.698673 s |
| `phase-two` | Also hold the hand still until measured jaw clearance and settled support | 68 | 70.969774 s |
| Total | Same 16 previously inspected cases, two revisions | **136** | **137.668447 s** |

Each phase compares baseline and candidate across all 16 original student cases,
plus the teacher on the known failing case. Every run has its own recorded-arm,
open-jaw control: 34 controller episodes and 34 controls per phase. All attempts
completed, all controls failed, all 17 baseline trajectories exactly reproduced
their originals, and all 17 candidate action prefixes matched before recovery.

The student still completes 15/16 cases in both revisions and both baselines.
There are no task-completion gains or regressions. The remaining case still
fails the original 18-second deadline. The second revision removes unsupported
opening commands in the sampled target recovery and waits for physical jaw
clearance before withdrawal. That is a bounded recovery-behavior improvement,
not a qualified robot skill or a demonstrated task-completion improvement.

The frozen confirmation gate was not met. No fresh confirmation suite was run,
no model was retrained and no production controller was changed. These repeated,
previously inspected cases are not independent generalization trials.

Read the [measured report](../../docs/controlled-recovery-experiment-20261015.md)
for exact settings, timing, observations and limits. External inference spend
was $0; local compute is unpriced. Full synthetic traces, replay HTML, model
weights and source snapshots remain in ignored `runs/` directories. These
compact receipts include no human video, credentials or private account data.

| Artifact | SHA256 |
| --- | --- |
| Phase-one protocol | `d35eec1f71c0154f75078b313338a0627487731057a3582c4afb04e9377bfe8d` |
| Phase-one report | `8d9d5162fd9904e3d4770ff335d6156414247267ad15d500bfcdc1f236b1b2d3` |
| Phase-two protocol | `e4736ec7895ce2f445dfece37eb943470b628c2287e7fd92af1a8ab11619f882` |
| Phase-two report | `acd006c62adc9749d2e038c117714ecdd5b1d79eed7c55287dd226d7f193a659` |

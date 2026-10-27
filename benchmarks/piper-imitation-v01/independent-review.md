# PiPER learned placement v0.1: independent confirmation audit

## Audit result

I independently recomputed the fixed-reference score for each of the 60 scheduled confirmation tapes and compared it with both the report row and saved score file. All tape hashes, case identities, seeds, goals, model/runtime bindings, and recomputed scores matched. The existing release audit also completed with the same roster and outcomes: 60 verified attempts and no split overlap.

The run is bound to report SHA-256 `52ae9f37e845c0949879012fe9e5279192a543dc5f898a8a8d2fe9ab0f51d4fe`, frozen manifest SHA-256 `8b514fa449e32e1f2da401176f9b7513d2b7b080daf9d3154fc02d8edd5073a3`, protocol SHA-256 `a7869a80ed801ab0ee85ab699a3d6231a17f4aa70109e519e0e03129ad30969f`, and selected model SHA-256 `2976a605575d9edefb5db366e72c5e006c0ce85a5531ad3d61512436864c8ded`. The selected artifact is `primitive401`, schema `dvidia.piper-motor-primitive/v2`. The current runtime hashes, training dataset and sample hashes, selected and shuffled model hashes, and protocol hash all match `frozen.json`.

The confirmation plan contains six held-out starts with both known goals, giving 12 scenes per arm and 60 attempts across five arms. The confirmation-opened marker contains the exact frozen 60-attempt schedule. There is no overlap among the 16 training starts, 4 development starts, and 6 confirmation starts. All 32 scheduled training demonstrations were successful, used, and contributed 32,000 frames. Development selection was based on the existing eight dev cases: `mlp160` 0/8, `absolute240` 0/8, `primitive201` 0/8, and selected `primitive401` 1/8. The held-out outcomes were not used for model selection.

| Arm | Successes / scheduled | Independently recomputed failures |
|---|---:|---:|
| Learned primitive | 5 / 12 | 7 native-outcome failures |
| Authored teacher | 12 / 12 | 0 |
| Open jaw | 0 / 12 | 12 |
| Hold | 0 / 12 | 12 |
| Shuffled labels | 0 / 12 | 12 |

All 12 learned episodes passed the independent terminal reference predicate, including the seven that failed overall native outcome gates. Thus the reported 5/12 primary success rate is not a terminal XY-only result: the terminal pose can be close while the native history gates reject the episode. The teacher and controls were scored against the same per-case goals.

The shuffled-label arm is the declared full frame-wise label permutation. Its training receipt correctly describes this as a weak temporal negative control, not a whole-trajectory association shuffle; its 0/12 result should be interpreted within that limit.

## Seven learned failures

Every learned failure was a valid 20-second timeout. All seven ended with the object supported by the table, both finger contact counts at zero, and the jaw fully open. Their final XY errors were 3.57–11.01 mm, inside the configured 25 mm tolerance. The independent terminal predicate found 4.50–5.62 seconds of qualifying terminal dwell, and reconstructed 1.38–1.96 seconds of sustained table-free bilateral lift from the dense trace, above the 0.10-second lift requirement. Native `lift_dwell_seconds` was zero at the end because it is a live streak counter that resets after release; it was not treated as evidence that no lift occurred.

The failures were instead rejected by native recorded history: native dwell remained zero in all seven; four had `retention_lost=true`; three had `carry_complete=false`; and five recorded unexpected-contact ticks. These categories overlap. The raw contact-pair IDs below come from each episode's `contact_pair_native_ticks`; they match the corresponding unexpected-contact count. The episode traces do not carry a geometry-ID-to-body-name mapping, so this audit does not assign those IDs to a particular robot part or infer a collision cause.

| Case | Goal | Final XY error | Carry complete | Retention lost | Carry-loss events; max gap / tolerance | Unexpected ticks (recorded pair IDs) |
|---|---|---:|:---:|:---:|---:|---:|
| `confirmation-00-goal1` | (0.40, −0.08) | 10.91 mm | yes | yes | 1; 63 ms / 40 ms | 0 |
| `confirmation-01-goal0` | (0.35, 0.08) | 4.83 mm | no | yes | 1; 6.904 s / 40 ms | 199 (`1:88`) |
| `confirmation-01-goal1` | (0.40, −0.08) | 10.08 mm | yes | no | 0; 3 ms / 40 ms | 117 (`1:85`) |
| `confirmation-02-goal0` | (0.35, 0.08) | 5.23 mm | no | yes | 2; 6.143 s / 40 ms | 92 (`1:88`) |
| `confirmation-02-goal1` | (0.40, −0.08) | 10.80 mm | no | yes | 1; 5.816 s / 40 ms | 0 |
| `confirmation-05-goal0` | (0.35, 0.08) | 3.57 mm | yes | no | 0; 0 s / 40 ms | 84 (`1:88`) |
| `confirmation-05-goal1` | (0.40, −0.08) | 11.01 mm | yes | no | 0; 0 s / 40 ms | 60 (`1:85`) |

All seven also recorded `lift_seen=true`, `grasp_seen=true`, maximum lift of 95–109 mm, and a trace-reconstructed lift dwell above threshold. The two failures with no unexpected-contact ticks still failed native retention/carry history and native dwell. This evidence identifies the failed gates; it does not establish why the feed-forward trajectory produced those histories. The primitive has no learned contact feedback or recovery, and these traces do not test such behavior.

## Recovery and limits

The frozen driver hit its tuple-versus-JSON-list configuration identity guard after saving a new tape. The separate recovery receipt contains exactly 60 scheduled recovery records, each bound to the frozen run, adapter, and tape digest; it reports `completed_tapes_reexecuted=0` and `frozen_runtime_changed=false`. There is exactly one start marker, tape, and score file per scheduled case/arm. All records were then scored through the original frozen confirmation path. This is consistent evidence of resuming saved tapes rather than rerunning physics. The receipt and simulator-authored traces are not an external attestation of process execution or authenticated sensor data.

The evidence supports a preliminary, simulation-only result for synthetic authored-controller distillation and interpolation to unseen starts with known goals. Five held-out successes show some task completion, while the teacher's 12/12 and the seven learned native-history failures show a substantial reliability gap. The 12 fixed scenes are not a broad reliability sample. This is not evidence for human or video imitation, visual perception, contact-reactive control, physical PiPER readiness, or transfer to other objects, goals, or robots.

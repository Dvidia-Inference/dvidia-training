# PiPER grip clock v0.3 development lineage

27 October 2026. Simulation only. This document records design choices made before fresh training/development/confirmation collection. Earlier proposals describe a physical-time jaw; that architecture was investigated and superseded for the main study.

The original pose-projection label idea had ambiguous path alignments. We instead log the collection teacher's causal binary hold/advance decision before issuing each command. All seven frozen v0.1 primitive references share virtual progress in the main architecture B. The separately declared architecture A used the unchanged v0.2 jaw4 on physical time and is not pooled with B.

The first A pilot found that world-frame object/tool displacement could label rigid tool rotation as slip. Readiness now uses the object displacement in the tool's rotating coordinate frame. Continuous gating also held at destination/release; this study narrows the target to acquisition readiness. Once released, the teacher latch remains released. This is not continuous learned recovery.

| Preserved development run | Scheduled native attempts | Finding |
|---|---:|---|
| `piper-clock-v03-feasibility-20261027` | 16 | World-frame readiness and incomplete issued-command metadata. |
| `piper-clock-v03-feasibility02-20261027` | 16 | Corrected rotation/latch; command metadata still omitted decision details. |
| `piper-clock-v03-feasibility03-20261027` | 16 | Exact logger; physical-time jaw teacher 3/8 versus fixed 8/8. |
| `piper-clock-v03-coordinated-feasibility-20261027` | 32 | Shared arm/jaw progress: fixed 4/8; teacher gate offsets 0, .16, .32 seconds scored 4/8, 6/8, 5/8. Offset .16 selected. |
| `piper-grip-clock-v03-runner-smoke-20261027` | 1 | Maintainer supplied an unsupported destination; zero-action invalid trace retained. |
| `piper-grip-clock-v03-runner-smoke02-20261027` | 2 | Supported goal, nominal and post-grip teacher attempts valid/successful; exact completed-attempt reuse verified. |
| `piper-grip-clock-v03-parallel-smoke-20261027` | 4 | Four spawn-isolated native controllers completed valid technical attempts on an already known nominal position. This check is excluded from candidate selection. |

All starts above were already known v0.2 points and are excluded from the new roster. A's first two runs retain original traces and explicit metadata-only derived projections reconstructed from saved actual commands. Those projections are not new physics attempts. The later pilots use the corrected original logger. No pilot is used for model fitting or counted as fresh confirmation.

The teacher requires bilateral contact, at least .20 N per pad, and at most 2 mm tool-relative drift across sampled observations spanning 100 ms. These are six causal control observations, not a guarantee of continuously secure contact at every native millisecond. Table support is allowed before the lift. The external disturbance separately requires airborne loaded grip and audits its native timing numerically.

The prepared protocol fixes 16 fresh training XY groups, eight development groups, and 32 confirmation groups, each with two known goals and nominal/post-grip conditions. The same cases/labels/weights and matched depth recipe are used for full and blind models; only the clock's live/history features are masked. Training retains failed finite prefixes. Four candidate fits are fixed in advance. Confirmation uses all 512 scheduled attempts across full, blind, fixed, and teacher controllers; failed/invalid attempts remain in denominators.

The two group-level comparisons use approximate t intervals with Bonferroni simultaneous coverage under their model assumptions. This exploratory sample is not a powered qualification study. A zero-variance interval does not establish certainty. Timing is shared-host simulator computation, excluding cameras, transport and physical actuators. No GPU is required for this bounded study. No hardware transfer or arbitrary-video learning is demonstrated.

Confirmation uses a separate four-worker execution driver bound into the study before model freeze. Each case dispatches all four variants and waits for all four before the next case. Sources, exact driver bytes, schedule and worker count are sealed. Timing describes concurrent simulator computation on a shared desktop. Training and development are sequential; their timing is not pooled with confirmation. The four-worker smoke is an infrastructure check, not additional unseen evaluation.

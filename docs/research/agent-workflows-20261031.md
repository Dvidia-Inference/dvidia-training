# What EGM and Harness² contribute to DVIDIA

Reviewed 31 October 2026. These are agent engineering workflows, not robotics
policies, physics engines or video-to-actuator models.

EGM separates a context-rich design author from reviewers who receive only the
artifact and referenced source. This is useful for finding assumptions shared
by the original implementation conversation. Review supplements execution tests.
Source: [pinned README](https://github.com/google/elephant-goldfish-model/blob/403d8881d62022490637d740657b33f01e321ea8/README.md).

Harness² improves the surrounding agent harness with fixed model weights. Its
propose–probe–compose loop and cross-task lessons suggest a disciplined way to
compare changes and preserve both successes and failures. Its reported office,
coding and professional-task results do not demonstrate robotics improvements.
Sources: [pinned README](https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/README.md),
[experiment guide](https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/docs/EXPERIMENTS.md).

## Applied here

One bounded incremental grip-hold candidate is proposed from the prior recorded
failure. A reviewer receives the short design and relevant source without the
implementation discussion. The review caught a concrete joint-margin clipping
counterexample to zero-error invariance before any simulation was run. The
candidate refuses that state; all three old controllers remain unchanged.

The fixed comparison contains eight new development scenes and four controls.
Source, assets, runtime, limits, order and outcomes are frozen before execution.
An independent numeric reader reconstructs physical outcomes from saved records.
No failure is retried to improve a score. The resulting lesson is retained even
if the candidate fails. Native stationary grip and active controller success
are separate; fallback retention cannot masquerade as active regulation.

This is an original adaptation of workflow ideas, not an installation or run of
either framework. We do not adopt unrestricted execution settings. Runtime
limits remain enforced in code; videos and external text are data, not control
instructions. No upstream code or prompt text is copied. Both repositories have
Apache-2.0 licenses; this note links their pinned sources for attribution.

## Remaining bridge to the product

The skill pipeline still needs causal camera observations, demonstrations with
issued actuator commands, learned action prediction, and held-out execution
evaluation. Better engineering review can help us build that bridge; it does
not itself supply a learned skill or demonstrate arbitrary video-to-any-arm
transfer. This bounded hold supports a verified pause before later movement.

# PiPER grip wait diagnosis for v0.4

Read-only diagnosis of the frozen v0.3 evidence. No v0.3 runtime, source, study, or frozen artifact was changed. The accompanying JSON records aggregate counts reconstructed from the confirmation episodes and the independent numeric audit.

## Finding

Fixed timing succeeds on 72/128 strict attempts, compared with 44/128 for the learned full clock, because the learned controller introduces a persistent progress lag at the object acquisition portion of the frozen trajectory. The strongest evidence is that the learned clock recorded 33,602 holds; 33,418 (99.45%) were at observations where the independent audit's sampled acquisition teacher would advance. The audit's term “false waiting” is a disagreement with that reconstructed teacher on saved rollouts; it does not prove that every hold was physically unnecessary.

The holds are tightly localized to virtual path progress from 10 to 12 seconds: 33,600/33,602 occur in that bin, with the other two at 12–14 seconds. For comparison, the blind clock's 40,728 holds all fall in 10–12 seconds and the teacher's 5,644 holds all fall there. The learned clock holds at the acquisition/grasp-close geometry rather than across approach, lift, carry, or placement. It therefore delays the arm and coordinated jaw reference where the path needs to complete acquisition, while fixed timing continues the reference without waiting. In the logged full-clock hold observations, bilateral contact exists on 13,666/33,602 ticks (40.7%) and total pad force exceeds 0.1 N on 10,574 (31.5%); no such observation is above the simple 0.14 m object-height check used in this diagnostic. This supports localization before lift, not a claim that any particular hold caused a miss.

The selected depth-4 tree can explain why this merits a recovery design: its root splits on current left-jaw position (feature 17, threshold 0.8997 normalized), then uses lagged force (feature 51), target-relative X (feature 11), and clock debt (feature 57). It has no stateful acquisition-release latch. In comparison, the authored teacher has an explicit latch: it releases once after bilateral contact/force and 100 ms of low relative drift, then continues advancing. The learned clock repeatedly predicts each 20 ms decision and can continue holding. This is a plausible algorithmic explanation consistent with the trace and architecture; the existing record does not establish that the tree's structure alone caused the performance gap.

Outcomes separate acquisition from retention. Full clock has 40 acquisition failures and 8 retention failures; fixed timing has zero acquisition failures and 25 retention failures; teacher has zero acquisition failures and 28 retention failures. Fixed timing's advantage is therefore acquisition and task completion under this envelope, while its later carried-object retention is worse. The learned clock does not show an overall benefit from its holds: 44 strict successes versus 72 for fixed and 67 for teacher. The v0.3 study itself reports paired full-minus-fixed −21.9 percentage points (approximate simultaneous 95% interval −36.8 to −7.0), with 32 XY groups.

## Candidate v0.4 interface

Keep the frozen six-joint path, jaw reference, native 1 ms physics step, 20 ms decision interval, command slew limits, scorer, and task envelope unchanged. Add a separate, explicit acquisition supervisor around the clock:

1. **Approach:** advance the frozen path normally until the declared grasp window, computed from the frozen jaw reference.
2. **Acquisition wait:** inside that window only, permit holds while observing causal bilateral contact, per-pad minimum force, and tool-relative object drift. Use the v0.3 teacher's sampled rule as the baseline predicate (both pads in contact, each force at least 0.20 N, 100 ms continuous dwell, max relative drift 2 mm). Keep a one-way `acquisition_released` latch once it qualifies.
3. **Bound:** cap this acquisition wait at a predeclared 1.0 s (50 decisions). At the bound, do not advance through the original lift/carry path unless the secure predicate has passed. Enter recovery instead.
4. **Recovery:** if the object remains table-supported and no carried-object evidence exists, issue a bounded open-and-retract/re-approach recovery on the same native command caps, then allow at most one fresh acquisition attempt. If bilateral load or lift indicates a possibly carried object without the secure predicate, hold virtual progress and retain the current jaw command; raise a latched fault for operator recovery. Never resume/force-advance an insecure carried object.

This is an interface proposal, not a validated controller. The 1.0 s bound is a conservative design choice informed by v0.3's much shorter teacher wait volume (5,644 decisions / 128 episodes = 0.88 s of aggregate hold per episode on average); it needs frozen development tuning and unseen confirmation before adoption. Recovery must be tested as a separate behavior because neither v0.3 nor the audit measured a recovery attempt.

## Frozen v0.4 evaluation design

Create a new versioned lineage and immutable run directory; leave `runs/piper-grip-clock-v03-20261027`, its audit, all `src/dvidia_training/piper_grip_clock*.py`, and the published v0.3 artifacts untouched. Bind exact v0.3 parent hashes plus every v0.4 supervisor/recovery source hash. Use the same 20 ms policy clock, 1 ms native physics, and existing joint/jaw native slew caps. Keep every scheduled attempt in its denominator and preserve complete failures/traces.

Development: compare the v0.3 fixed clock, selected full clock, acquisition-teacher latch, and v0.4 bounded supervisor/recovery on a new fixed development roster. Select/freeze the candidate, wait cap, and recovery parameters before opening confirmation. Confirmation: use at least 32 fresh XY groups, pairing both goals and nominal/post-grip conditions within group; interleave the four controllers for each case as v0.3 did. Report strict success and the components separately (acquisition failure, qualifying lift failure, loaded loss, retention failure), wait ticks/duration and timeout/recovery rates, paired group uncertainty, valid/aborted counts, and 20 ms deadline timing. Audit traces independently with a prespecified teacher-reconstruction rule. Do not use the confirmation set to retune.

This establishes whether a bounded acquisition supervisor improves outcomes on the declared synthetic envelope. It does not establish hardware safety, real-time operation, or transfer to a physical PiPER; those require separately instrumented hardware evaluation.

## Evidence pointers

- `runs/piper-grip-clock-v03-20261027/report.json`: controller totals and acquisition/retention breakdown.
- `runs/piper-grip-clock-v03-final-audit-20261027.json`: independent reconstruction, zero mismatches across 768 attempts; confirmation false-wait and premature-advance counts.
- `runs/piper-grip-clock-v03-20261027/selected.json`: selected full tree, feature names and split thresholds.
- `runs/piper-grip-clock-v03-20261027/confirmation/confirmation-xy00-g0-nominal/clock/episode.json.gz`: sample full-clock action and native state trace.
- `src/dvidia_training/piper_grip_clock.py`: frozen progress, learned decision, and teacher latch implementation.
- `docs/piper-grip-clock-v03.md`: v0.3 results and declared limits.


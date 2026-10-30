# Bounded grip waiting v0.4: recovered confirmation results

The authored supervisor bounds acquisition waiting while retaining the frozen
v0.3 arm/jaw references and advisory learned predictions. **No new motor model
or weights were trained. This is simulation evidence, not a qualified robot skill.**

| Controller | Strict successes / scheduled | Completed, valid | Interrupted, unknown |
| --- | ---: | ---: | ---: |
| Bounded waiting | 42/128 | 127 | 1 |
| Learned clock | 32/128 | 127 | 1 |
| Fixed progression | 44/128 | 127 | 1 |
| Teacher | 41/128 | 127 | 1 |

Four started attempts were interrupted in the first session. Recovery preserved
all 72 completed receipts and four partial folders byte for byte, then executed
only the 436 never-started requests. There are **508 scored, valid confirmation
attempts and four unevaluable outcomes**, rather than 512 valid attempts. The
primary scheduled-denominator analysis imputes the four unknowns as failures;
the sensitivity analysis excludes their whole XY group, leaving 31 groups.

Bounded-minus-learned was +7.81 percentage points (adjusted interval −7.25 to
+22.87); bounded-minus-fixed was −1.56 points (−12.44 to +9.32). Both comparisons
are inconclusive. The supervisor latched secure acquisition in 105/127 completed
attempts and waited at most 200 ms. Its 22 explicit aborts comprised eight
exhausted budgets and fourteen support losses before readiness.

Post-grip pulse exposure differed: 53/64 assigned bounded attempts, 22/64 learned,
64/64 fixed, and 58/64 teacher attempts received an effective pulse. Fewer
retention failures therefore do not establish superior slip recovery. Critical
contacts remain frequent, and there is no physical safety or real-time claim.

The independent audit reconstructed all 48 development and 508 complete
confirmation attempts: **556 audits, zero mismatches**. Original and resumed
CPU timings are separate; the sessions ran on different dates under shared load.

- [Compact report, all 512 scheduled rows and both analyses](report.json)
- [Audit identities and summary for all 556 complete attempts](audit-summary.json)
- [Four interruption byte manifests and retained started files](interruptions.json)
- [Source, policy, recovery and test lineage](source-lineage.json)
- [Research note](../../../reports/Bounded%20grip%20waiting%20v04.md)

These summaries omit native arrays, full decision traces and per-step timing
samples. They bind the 35,327,217-byte full local recovery report by SHA256:
`87d252d2ffe2aea5a1bca6b50d926abbd2181b4f51695471dfb5747af91e4e56`.
Repository-relative raw paths identify the source evidence; this compact bundle
does not itself contain the full raw report or native dataset.

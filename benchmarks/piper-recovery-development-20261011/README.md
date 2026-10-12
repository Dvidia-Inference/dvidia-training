# PiPER contact recovery v0.4 — retained development

These experiments use the already published v0.3 layout seed **813407**. They
are development and regression evidence, not fresh confirmation or hardware
validation. Every named native experiment retains its complete 20-second
action/state record, including failed attempts and closed holding after abort.

| Revision | Purpose | Native attempts |
| --- | --- | ---: |
| `r0` | Runner setup failed before simulation: a configuration mapping was passed instead of `PiperTaskConfig`. Error and controller snapshots retained. | 0 |
| `r1` | Initial stable-evidence speed restoration and selective refusal of an unchanged supported grasp retry. Twelve full-mode placements and two carry-only failed-grasp controls. | 14 |
| `r2` | Same controller parameters and decisions. Restoration events now record the resulting flags and requested targets; raw control-window timing samples retained. Same 14 placements. | 14 |
| `v03-regression` | Re-run all 12 historical v0.3 placements through the updated task runner. Compare every requested action, native qpos/qvel state, outcome and task metric to the sealed original. | 12 |
| `r3` | Final review: an exhausted retry budget cannot count as a newly refused retry; hold events reset the restoration diagnostic fields. Three selected full-mode scenes plus one carry-only control. | 4 |
| `paired-benchmark` | Separate full benchmark integration on the same historical seed: three placement modes, each own open-jaw replay, and shared hold for each of twelve scenes. | 84 |

Each directory contains the relevant controller/task source snapshots and a
manifest with exact raw-file hashes. Gzip files decode to the full result from
`piper_task.run_episode`. `comparison.json` reports outcome and timing
comparisons directly from the retained records. The source snapshots preserve
the exact code used even when a later diagnostic improvement supersedes it.

The two complete full-mode development sweeps each retain **6/12 successful
placements**. Four successful post-hold placements complete about 2.46–2.58
simulated seconds sooner than v0.3. The settling hold and its qualification
thresholds are unchanged: the improvement comes after resuming, when bounded
speed restoration replaces permanent reduced joint-delta scale.

Three of the six failed full-mode acquisitions refuse the unchanged supported
retry at its first failure, roughly 2.9 simulated seconds before the old final
abort. Three preserve the old retry because their recent force/contact evidence
is ambiguous. All six remain failures. A failed side grasp does not prove a
different grasp is impossible. The controller receives no mass, friction or
future outcome to make this refusal decision.

All twelve historical v0.3 regressions match the original actions, qpos, qvel,
success flags and task metrics exactly. Timing and source identifiers can
differ. The original v0.3 controller, native mechanics, task scorer and old
sealed results remain unchanged.

`r1`'s manifest has a null `completion` field because its summarizer requested
the wrong metric key; the complete raw results retain the correct
`metrics.first_success_time_seconds`. This summary error is preserved rather
than rewritten. `r2`, `r3` and `comparison.json` use the correct raw field.

These runs use MuJoCo's privileged state on rigid synthetic blocks, not sensed
material properties, learned robot policies or real-robot stopping guarantees.
Native unit-test fixtures are separate regression checks; the table inventories
the retained development experiments only. No controller threshold was tuned
between these revisions.

The **44 controller development/regression attempts** above are separate from
this **84-attempt historical benchmark integration**. The latter is retained
byte-for-byte in `paired-benchmark/`, including all 84 complete gzip trajectories,
original report, protocol, integrity seal and model notices. Every attempt is
valid. Each placement mode completes 6/12; all 48 negative controls fail. The
six common successes average 1.670 simulated seconds sooner for either v0.4
mode; this average includes the two unchanged successful scenes. No outcomes
are added to the future fresh-confirmation denominator.

`paired-benchmark-sources/` retains all thirteen exact runtime files used for
that integration run. The first post-run inspection exposed a JSON tuple/list
comparison error after all complete evidence had been saved. Only the inspector
was repaired; the existing sealed bundle passes independent reinspection.
The original error/progress and repair record remain in
`paired-benchmark-progress.jsonl` and `paired-benchmark-review.json`, with the
successful `paired-benchmark-inspection.json`. Further inspection checks verify
restoration event diagnostics, finite native completion timestamps corroborated
by sampled release dwell, and raw computation windows against whole-episode
wall time. These consistency checks do not reconstruct the exact first 1 ms
native success tick or authenticate a physical robot.

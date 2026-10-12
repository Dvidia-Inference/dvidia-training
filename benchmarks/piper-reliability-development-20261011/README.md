# PiPER reliability v0.5 — development archive

These experiments use already published layouts (813407 and 941827). They are development data, not fresh confirmation. Every native attempt and discarded candidate is retained with its source snapshot. The independently reserved fresh seeds are in `confirmation-reservation.json`; they were reserved before execution and must run unchanged after the selected runtime/protocol is committed.

- `lift/`: 40 attempts, 19 successes, including two rejected probe settings. The selected second lift improves known 941827 from 4/12 to 6/12 and reproduces all historical 813407 trajectories unchanged.
- `grasp/`: 76 attempts, 24 successes. Fourteen discarded aggressive-planner attempts have prohibited contacts and remain failures. The final 24 selected-candidate attempts improve known counts from 4 to 7 and 6 to 8, with zero lost successes, prohibited contacts or warnings. Shared successes are slower on average; the candidate is fixed before each episode and does not learn or select by material labels.
- `original-regressions/`: 24 exact old-v0.4 reproductions. Commands, full qpos/qvel, task metrics and outcomes all match their original receipts.
- `paired-known-941827/`: a separate 60-attempt two-mode integration, including 36 matched negative controls. Its original sealed source snapshot is in `paired-known-941827-sources/`. The final three-mode release is a different frozen protocol; inspect this intermediate bundle with its own archived source, not by substituting the later protocol.

There are **200 full native attempts** across these distinct development and regression runs. Counts include negative controls and failures; selected development success rates are not estimates of unseen performance. Unit-test and preflight fixture executions are separate. `inventory.json` seals this retained archive; original per-run receipts and seals remain untouched.

Native dynamics, model, gravity, force/gain limits, scorer and 20-second task budget remain fixed in the selected comparison. All code is authored and uses privileged simulator state. No claim of learned skills, shoe manipulation, sensor robustness or physical robot qualification follows.

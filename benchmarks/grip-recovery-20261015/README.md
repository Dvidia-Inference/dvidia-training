# Grip/recovery diagnostic receipts

These are exact JSON-byte copies of two completed local **simulation-only**
experiments on the already inspected `scene-15` from the Skillspace improvement
pilot. Phase one contains 28 native episodes; phase two contains 20. There is
only **one distinct scene**, with changed conditions and deterministic repeats.

Read [the interpretation](../../docs/grip-recovery-diagnostic-20261015.md).

- `phase-one`: seven one-factor conditions, authored teacher and frozen student,
  each with a paired recorded-arm/open-jaw control.
- `phase-two`: the 2 × 2 jaw-capability/time-budget diagnostic and one separate
  exploratory loaded-goal-cap condition; same two policies and paired controls.

Protocols preserve original source/runtime/model hashes and explicit conditions.
Reports retain every completed outcome, errors, instrumentation summaries,
wall clocks and unpriced local-compute limitations. The copy operation did not
rewrite JSON or its hashes.

Full traces, actions, replay HTML, original-run snapshots and movement weights
remain in ignored local directories:

```
runs/grip-diagnostic-20261015/
runs/grip-followup-20261015/
```

This compact archive is sufficient for the read-only results viewer, **not**
independent native replay. The archive includes no uploaded human footage,
account information, credentials, or model weights. No runtime defaults or
qualification gates changed, and no candidate was promoted.

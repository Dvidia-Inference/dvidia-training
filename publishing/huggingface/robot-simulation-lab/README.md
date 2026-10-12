# Robot Simulation Lab publication

This static Space publishes recorded simulation evidence and the existing UR5e replay. Native simulation and policy auditing run locally; the browser does not run live PiPER physics. The uploader is restricted to the public `Dvidia/robot-simulation-lab` Space and uses the existing in-memory credential adapter.

## Add v0.5 reliability receipts

The preparer accepts `--piper-reliability-benchmark PATH` up to three times. Each argument must point to a complete sealed v0.5 benchmark directory. The portable inspector validates recorded consistency and explicit second-lift prerequisites; complete native policy reconstruction with the pinned assets is a separate release check. Preparation does not run simulations or invent results.

Continue supplying every existing replay, profile and benchmark argument so all previously published evidence remains in the staged inventory. For example, add these arguments to the existing complete preparation command:

```sh
--piper-reliability-benchmark runs/reliability-cohort-a \
--piper-reliability-benchmark runs/reliability-cohort-b \
--piper-reliability-benchmark runs/reliability-cohort-c
```

Use the unchanged `--replay`, `--profile-benchmark`, `--piper-task-benchmark`, `--piper-grip-benchmark`, `--piper-contact-benchmark`, and `--piper-recovery-benchmark` arguments for prior evidence. Supply a fresh `--output` directory, the committed source revision in `--source-commit`, and the exact current public Space parent revision in `--expected-remote-commit`.

A reliability directory is copied byte for byte into `tasks/piper-reliability-v05/{cohort}-{seed}/`, using the validated original protocol fields. Duplicate cohort/seed identities and more than three supplied cohorts fail before staging writes. Each copy includes the complete protocol, report, every compressed raw attempt, stored/decoded integrity records, and original model notices. Missing, extra, changed, private, oversized, malformed or incorrectly scoped evidence is rejected.

The cards show each cohort separately: every placement denominator, matched control counts, gained and lost completions, shared successes, timing only among shared successes, and descriptive pilot gates. A cohort label does not prove data independence. Recovered contact is not necessarily successful placement; simulation results do not qualify a physical robot or establish learned competence.

## Bounds and verification

The stored Space payload cap is **512 MiB**, increased from 256 MiB to retain existing evidence alongside three complete v0.5 cohorts. The **32 MiB per-file cap**, bounded single-member gzip decoder, JSON depth limit, exact inventory checks, and compressed/decoded privacy checks remain unchanged. The existing maximum inventory remains 600 files; three 90-file v0.5 receipts plus the previous 293-file payload total 563 files. These are publication bounds, not minimum hardware requirements.

Both staging and the standalone publisher enforce the three-cohort limit. The publisher rejects a plan that would omit existing remote files, except its separately bounded original starter stylesheet cleanup. It freezes the reviewed local payload, updates only the exact expected parent, then publicly downloads and verifies every file at the resulting immutable commit. `publication-receipt.json` is the completed publication evidence; preparing a plan does not upload anything.

```sh
PYTHONPATH=src python publishing/huggingface/robot-simulation-lab/test_prepare.py
PYTHONPATH=src:publishing/huggingface python publishing/huggingface/test_publish_robot_sim.py
```

The publisher tests require the optional maintainer Hugging Face environment and use mocked credentials/network calls only. They test full evidence preservation, denominator and seal checks, failed/invalid outcomes, cohort collisions, stored/per-file/decoded bounds, scoped compressed paths, escaped credentials/private paths, remote parent pinning and immutable upload snapshots.

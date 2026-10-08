# Frozen sample-count and simulation qualification benchmark

`dvidia-training-benchmark` compares nested sets of **whole connected training
groups** from an actual prepared dataset. It keeps the original development and
test episodes fixed, trains visual predictors and aligned-label movement heads,
builds and installs candidate capsules, and qualifies each candidate in a new
seeded suite of bounded native simulation scenes.

This is a simulation experiment. It does not recover actions from human video,
establish a sufficient demonstration count, qualify a physical arm, or broaden
the installed capsule's existing exact-scene execution gate.

## Run locally

Install the optional arm dependencies first:

```sh
.venv/bin/python -m pip install ".[arm]"
```

Use a prepared `dataset.json` containing real encoded media and validated action
sidecars. The default comparison uses 2, 4, and 7 training groups when available;
smaller datasets use the eligible counts. Every episode needs aligned movement
supervision, and the existing fitter requires at least 32 distinct nonzero
training action examples. Related clips remain together.

```sh
.venv/bin/dvidia-training-benchmark path/to/native/prepared/dataset.json \
  --task-source path/to/native/source/task.skill.json \
  --source-evidence path/to/native/source/evidence \
  --counts 2 4 7 --nominal 8 --stress 4 --seed 19047 \
  --output runs/sample-count-19047
```

From a source checkout, `.venv/bin/python -m dvidia_training.footage_benchmark`
exposes the same command. Use a fresh output
directory. A benchmark cannot overwrite or resume a frozen experiment. To make
a small smoke run, use one count with `--nominal 1 --stress 0`; a smoke run is not
general scene evidence.

The default task source is the bundled authored rigid-box placement envelope.
Supply `--task-source` and, if needed, `--grounding` explicitly for your supported
task. The benchmark does not infer a task adapter from videos.

## What is frozen

Before any fitting or physics evaluation, `protocol.json` and `protocol.sha256`
record source dataset and training code hashes, the exact runtime, nested group
and episode identities, unchanged development/test episodes, full scene inputs
and hashes, model settings, development-only regularization selection, and the
existing success/control rules. Copies of the prepared inputs are retained.

All models are fitted and installed before any benchmark scene is evaluated.
`models-frozen.json` then binds the exact candidate bytes. Source, protocol,
input, runtime, and model checks repeat before subsequent fitting/evaluation.
Results cannot silently redefine the plan. New tuning requires a new experiment
and fresh evaluation scenes; choosing another output directory with the same
seed merely repeats known cases.

One authored teacher baseline runs per exact scene. Each installed candidate is
evaluated through the existing `capsule_worker(qualify=True)` path, including its
own recorded-joint-target/open-jaw control. The original `run.json`,
`controls.json`, `qualification.json`, and replay remain under each candidate's
scene directory. Raw student completion, control completion, valid
qualification, and execution errors are reported separately. A student that
completes while its control also completes does not earn qualification.

## Source-scene exclusion

Prepared datasets retain evidence hashes but do not always contain complete
source scene declarations. When `--source-evidence` is supplied, every action
sidecar must bind a corresponding native telemetry evidence JSON. Its declared
zero-jitter scene is validated, and evaluation layouts exclude object/target
layouts from **all** source training, development, and test episodes. Changing
only a seed or material cannot make a source layout new.

These checks establish consistency relative to hash-bound supplier declarations,
not independent attestation of simulation or capture history. Without the
evidence directory, the report explicitly marks source-scene disjointness as
unestablished. Do not present that run as independent scene generalization.

## Interpret the result

`report.json` gives training-group counts, training-episode and numerical-action
sample counts, fixed held-out prediction errors, per-scene teacher/student/control
outcomes, nominal/stress qualification totals, paired teacher comparisons, and
scoped clocks. Every recorded teacher, student, and control also retains final
target error, simulated time, wall time, and throughput. Aggregate throughput is
total simulated time divided by total episode wall time, rather than an average
of individual speed ratios. Error means/maxima count only available finite
numeric target distances. `measured_error_scenes` is separate from `timed_scenes`;
rejected, invalid, and execution-error cases remain in planned-scene counts. The same test
recordings and evaluation scenes are shared across
counts; they are not repeated independent studies. A single nesting order
confounds count with which groups are added. This is a bounded composition-order
pilot, not a universal learning curve or performance guarantee. Nominal and
actuator-stress outcomes and paired teacher comparisons are stratified. Combined
clocks/counts are execution accounting, not a pooled performance estimate. Exact
source layout exclusion establishes neither out-of-distribution robustness nor
human-to-robot transfer.

Stress cases use reduced actuator limits. A shared teacher/student grasp failure
may reflect that constraint rather than too few training labels. Privileged
simulator observations and authored phases, grasp/contact logic and recovery
remain in use. Camera-based control, human-video action recovery, hardware
transfer, and broader embodiments remain unmeasured.

The existing published one-scene receipt recorded about 0.82 seconds for its
candidate and 0.91 seconds for its open-jaw control on one host. A default
three-count, twelve-scene run therefore adds 72 candidate/control episodes and
twelve teacher episodes. Allow a few minutes on a comparable machine, plus
installation and output I/O; this is an estimate, not a requirement or guarantee.
Full native traces, controls, copied media, and replays can require hundreds of
MiB. The report measures the actual run. Dependencies and data must be installed
before offline execution.

## Published pilot and reproduction

The [8 October frozen result](../benchmarks/sample-count-19047/README.md) records
the completed 2/4/7 comparison, including every failed case. Repeating seed 19047
reproduces known layouts; use a new, recorded seed for a subsequent evaluation
after changing a learner.

To reproduce the published source intake, download the fixed
[native Skillspace archive](https://huggingface.co/datasets/Dvidia/dvidia-training-examples/resolve/d351a32923d54e3166e8ff780fd827c80174a903/archives/native-skillspace.zip?download=true)
before disconnecting. Its SHA256 is
`5091233dbc21ea84022e57101a5e2036be44bb6b226adf2610265f12263349e9`.
Extract it into `examples/native-source`, then prepare it with:

```sh
.venv/bin/dvidia-train prepare examples/native-source --output runs/native-prepared
```

Use `runs/native-prepared/dataset/dataset.json`,
`examples/native-source/task.skill.json`, and
`examples/native-source/evidence` in the benchmark command above. Keep the
default preparation split seed; changing the source split creates a different
experiment. Training and qualification need no network after dependencies and
these inputs are installed. The source archive contains simulator recordings
and aligned numeric labels, rather than human shoe demonstrations.

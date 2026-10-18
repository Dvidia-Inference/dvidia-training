# One measured improvement experiment

This local pilot connects an existing development failure to a frozen data
selection experiment. It fits the **simplified native rigid-box placement
movement head** from aligned numerical simulator actions. It does not learn
robot commands from human video, train the newer PiPER/UR5e task controller,
collect new demonstrations or qualify physical hardware.

The historical `sample-count-19047` benchmark provides the diagnosis: on nominal
`scene-04`, the four-group candidate timed out while its paired authored teacher
succeeded. This is already-inspected development evidence. A shared failure on
the actuator-stress case is not treated as evidence that more videos would fix
an actuator or controller limit.

## Frozen hypothesis and comparison

The hypothesis is that an unused training group whose initial object and target
layout is nearer the development failure may help the learned movement head.
Distance is Euclidean over the six object/target XYZ coordinates, in metres;
it does not measure contact dynamics, coverage sufficiency or causal benefit.

Before fitting, `improvement_experiment.prepare_plan` verifies the historical
report/protocol bindings, checks the exact prepared source hash, derives the
original four groups using seed 19047 and resolves hash-bound native scenes.
It creates three explicit whole-group cohorts:

| Cohort | Membership rule |
| --- | --- |
| `baseline4` | Original four training groups |
| `generic5` | Baseline plus one other unused TRAIN group, selected by seeded SHA256 order |
| `targeted5` | Baseline plus the nearest unused TRAIN group; distance ties use group ID |

The generic selection excludes the targeted group. Both additions have the same
one-group budget; numerical rows and video durations are **not** forced equal.
Those quantities are recorded explicitly. All members of a connected group move
together. Development/test episodes remain fixed, and evaluation groups cannot
become candidates for the additions. No new evaluation result enters selection.

The learner seed remains 19047 for all variants. The existing fixed development
episodes select ridge regularization; architecture and recipe stay the same.
Every model is fitted and its identity frozen before any new simulation is
evaluated. The default new scene seed is 381527, with twelve nominal and four
actuator-stress cases. All declared source layouts and all twelve known
development benchmark layouts are excluded by object/target position, even if
their seed or material differs. A new seed alone is not proof of non-exposure.
Historical source test prediction errors remain regression diagnostics.

## Inspect and run

From a source checkout with the existing optional arm dependencies and local
public native dataset already present:

```sh
PYTHONPATH=src .venv/bin/python -m dvidia_training.improvement_experiment plan \
  --dataset runs/huggingface-publication/dataset/native/prepared/dataset.json \
  --prior-protocol benchmarks/sample-count-19047/protocol.json \
  --prior-report benchmarks/sample-count-19047/report.json \
  --source-evidence runs/huggingface-publication/dataset/native/source/evidence \
  --output runs/improvement-plan

PYTHONPATH=src .venv/bin/python -m dvidia_training.improvement_experiment run \
  --dataset runs/huggingface-publication/dataset/native/prepared/dataset.json \
  --prior-protocol benchmarks/sample-count-19047/protocol.json \
  --prior-report benchmarks/sample-count-19047/report.json \
  --source-evidence runs/huggingface-publication/dataset/native/source/evidence \
  --task-source runs/huggingface-publication/dataset/native/source/task.skill.json \
  --seed 381527 --nominal 12 --stress 4 \
  --output runs/improvement-experiment
```

Each output must be fresh. `plan` validates and records selection without
training or evaluating; `run` recomputes and freezes the plan and explicit scene
budget before fitting. It does not resume or edit an existing experiment.

The default run performs 112 native episodes: sixteen shared authored-teacher
episodes and sixteen candidate/control pairs for each of three cohorts. The
earlier 84-episode benchmark took 102.7 seconds on an Apple M5; allow a few minutes
for this experiment on comparable hardware. This is an estimate, not a timing
result for the new pilot. No external inference API is used. Actual wall time
is recorded, while hardware/electricity/operator dollar costs remain unknown.

## Artifacts and interpretation

- `experiment-plan.json`: exact memberships, reserve ranking, selection rules,
  source hashes, historical diagnosis, training/evaluation seeds and group budget.
- `benchmark/protocol.json` and its hash: recipe, explicit cohorts, source and
  known-development exclusions, full new scene declarations and task/runtime.
- `benchmark/models-frozen.json`: every candidate identity before evaluation.
- `benchmark/qualification/<cohort>/<scene>/`: original run, identical-arm-target
  open-jaw control and exact-scene qualification receipt, including failures.
- `benchmark/report.json`: per-scene results, separate nominal/stress summaries,
  fixed-source prediction diagnostics, actual sample counts and scoped clocks.
- `report.json`: compact cohort summaries, artifact bindings and cost scope.

Compare targeted against generic and baseline on the same scenes. Preserve all
attempted denominators, teacher failures, open-jaw controls, errors and negative
results. One composition per condition does not establish a general learning
curve, significance or an attribution/payment score. A result does not trigger
automatic promotion. Future tuning makes inspected cases regression evidence;
new claims require another frozen confirmation suite.

The underlying benchmark remains backward compatible with nested count runs.
Its optional Python API accepts `cohorts=[{'cohort_id': 'name',
'train_group_ids': [...]}]`, `excluded_scenes`, `experiment` provenance and a
separate `training_seed`. It derives episode membership from validated source
groups, rejects unknown/nontraining/duplicate groups and unsafe or duplicate
names, and preserves `groups-002` paths for default count-based calls.

Validation:

```sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_footage_benchmark.py'
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_improvement_experiment.py'
```

The selection and dispatch tests are authored contract fixtures, not an executed
training result or measured robot improvement.

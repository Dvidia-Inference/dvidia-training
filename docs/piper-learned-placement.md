# A first learned PiPER placement primitive

**Frozen confirmation: learned 5/12, authored teacher 12/12, open-jaw/hold/shuffled-label controls 0/12 each.** All twelve learned terminal poses qualify; seven fail stricter native contact-history gates. This is preliminary evidence, not a reliable released robot skill. [Independent audit](piper-imitation-independent-audit.md).

This study distills **synthetic robot demonstrations into learned joint and gripper commands**. It is a step beyond the earlier video-goal installer, whose movement came from an authored controller. It does not learn actions from human video. The input demonstrations contain simulator state synchronized with the teacher's commands.

The task is deliberately small: one 40 g rigid block, a native six-axis PiPER with a coupled parallel gripper, two known destinations, and starting positions in a 30 × 30 mm region. New starting positions are excluded from training by complete episode. No physical arm is connected.

## What is learned

Development first tests two feedback MLPs using 43 privileged state features: relative-delta prediction and standardized absolute-target prediction. Both complete **0/8 development placements**. Their weights, recipes and all sixteen failed attempts remain in the evidence. Offline supervised error does not establish closed-loop competence: the first learner drifts outside demonstrated states and closes too early; the absolute-target learner tends to stall near its initial configuration.

The selected candidate is a conditional movement primitive. A ridge fit learns coefficients for joint and jaw references from the reset block XY, one of the two known goals, and elapsed time. A 201-time-basis quadratic candidate scored 0/8 on development. A higher-resolution cubic candidate with 401 time bases scored 1/8 and was frozen before confirmation. It trained in 1.42 seconds and occupies 2,031,879 bytes. Spatial polynomials and normalized Gaussian time bases describe the regression; they do not contain an authored approach/grasp/release sequence. All six joint targets and the jaw target come from learned coefficients. Native servos and generic joint/rate limits track those references.

This is **feedforward trajectory distillation**, not learned contact correction. After reset, the primitive does not adapt its reference to slipping, object displacement or an obstruction. It contains no demonstration lookup, teacher call, IK, task-phase machine, success-triggered release or fallback. Generic command limiting still uses measured joint positions. The model's schema, inputs, basis, weights and training recipe are explicit JSON.

## Evidence and selection

The fixed schedule contains 32 teacher episodes from sixteen training starts, eight development cases from four other starts, and twelve confirmation cases from six more starts. Both destinations are shared across splits. All 32 teacher attempts succeeded under the independently checked contract and supply 32,000 causal state/action pairs. No human-video labels or physical sensor data are included.

Model-family choices use development only. Confirmation starts after weights, protocol, native runtime and scorer hashes are frozen. Every scheduled attempt counts, including exceptions or interruption; completed attempts cannot be overwritten or retried. Teacher, learned model, open-jaw execution, hold control and a learner fitted with permuted action labels share the same final scenes and scoring rule. The label-shuffle control is a basic supervision check; it is not a nearest-demonstration baseline or evidence of optimality.

Success requires valid native physics, an actual bilateral unsupported lift, retained transfer, open-jaw release, supported settling within 25 mm of the external reference goal, and tool retreat. The dense saved trace must independently contain the sustained lift and final settled interval. The scorer checks recorded native evidence; it does not authenticate physical sensors.

## Run a downloaded model locally

Use Python 3.12 or later and the arm-enabled release wheel. The numerical path needs NumPy and native MuJoCo; it needs no language-model service, account or GPU after installation and asset download. CPU/RAM minima, Windows support and physical latency are not certified by this pilot.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install --force-reinstall 'dvidia_training-0.1.0-py3-none-any.whl[arm]'
dvidia-robot-sim fetch piper --output robot-assets
dvidia-piper-learned --model model.json --assets robot-assets \
  --start-x 0.30 --start-y -0.03 --goal 0 --output my-attempt
```

The output preserves the installed model, complete native episode, independent score and an installation manifest. Exit status is 0 for an independently successful attempt, 1 for a preserved failed attempt, and 2 for rejected input. Output directories must be fresh. Coordinates stay within x=0.284–0.314 m, y=−0.045–−0.015 m; goal 0 is (0.35,0.08) m and goal 1 is (0.40,−0.08) m. This command executes simulation only.

The fresh installed-wheel check used the previously successful development example `--start-x .309 --start-y -.04 --goal 1`. It is an installation check selected after evaluation, not an additional independent confirmation trial. Other starts, including the generic example above, may fail.

## Reproduce the study

```sh
dvidia-piper-imitation prepare --run my-study
dvidia-piper-imitation collect --run my-study --assets robot-assets
dvidia-piper-imitation train --run my-study --name primitive401 --primitive --high-resolution
dvidia-piper-imitation development --run my-study --name primitive401 --assets robot-assets
dvidia-piper-imitation train --run my-study --name shuffled401 --primitive --high-resolution --shuffle
dvidia-piper-imitation freeze --run my-study --name primitive401 --shuffled-name shuffled401
dvidia-piper-confirm --run my-study --assets robot-assets
```

The final command uses the bounded resume adapter for the frozen driver's tuple/list serialization guard. It validates and scores each saved tape through the original driver and advances without repeating completed physics. The published run retains all 60 recovery records. Do not change a frozen model or rerun its failed confirmation cases. The published development-selection history includes all four candidate families; this recipe reproduces the selected candidate directly.

CPU inference across 12,000 confirmation calls measured 0.148 ms median and 1.039 ms p95. The full control step missed its 20 ms compute budget 344 times. These exclude sensing/transport/real actuation and are not a hard real-time guarantee.

## What comes next

Demonstrating a learned primitive would establish a bounded motor-memory result. Progress toward video-installed physical skills still requires an action-data bridge, camera-derived state, learned feedback under disturbances, broader object/start coverage, and hardware calibration with independent completion evidence. The [Cosmos observation-layer proposal](cosmos-observation-layer.md) addresses a possible visual augmentation experiment; it does not supply missing PiPER actions or qualify physical transfer.

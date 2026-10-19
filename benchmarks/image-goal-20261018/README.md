# Image-goal benchmark v0.1: native validation and Jev protocol

**Jev has not been run.** The 32 native cases and evaluation tools are tested;
all 64 scheduled Jev trials remain unrun while secure API configuration is
pending. There are no measured Jev accuracy, latency or cost results here.

- [Research note](https://research.dvidia.org/papers/image-goal-jev-v01/)
- [Inspect every case on Hugging Face](https://dvidia-robot-simulation-lab.static.hf.space/image-goals/v01/index.html)
- [Local run instructions](../../docs/image-goal-benchmark.md)
- [Independent native validation](native-validation.json)

## Executed local checks

The frozen cohort has four layouts and eight conditions, with 8 physically
complete and 24 incomplete states. Eight cases lack the intended target, counting
the missing-target and orange-decoy conditions. Four moving/settled pairs have
identical PNGs and different physical completion labels. Native generation took
6.297 seconds on this host, including model construction, stepping and rendering.
That is not model inference or robot reaction time.

The independent verifier checks all 67 original files (5,019,443 bytes), derives
the physical predicates and dwell from recorded numeric measurements/history,
cross-checks raw poses and velocities, and validates telemetry extrema. All
32 cases pass these consistency checks. Repeating verification gives identical
JSON. A separate native test suite rebuilds the scenes and renders the controls.

The shoe is a 30 g rigid proxy narrowed to half the catalog width. Real simulated
finger contact establishes the unreleased negative control under unchanged
PiPER actuator gains. This is not evidence of full-size shoe pickup, learning,
deformation or physical robot transfer.

## Interpretation before any model calls

Images alone have 16 visibly incomplete cases and 16 cases whose physical
completion is unobservable from one frame. The telemetry condition has 8 complete
and 24 incomplete cases. These class balances imply simple reference baselines:

| Fixed answer | RGB expected-verdict accuracy | RGB + telemetry expected-verdict accuracy |
| --- | ---: | ---: |
| Always incomplete | 16/32 | 24/32 |
| Always unknown | 16/32 | 0/32 |
| Always complete | 0/32 | 8/32 |

These are arithmetic baselines from the frozen labels, **not Jev results**.
Accuracy must be read alongside false completion, coverage, justified abstention
and request errors. Four layouts are correlated clusters; this cohort does not
establish generalization or production tail latency.

## Frozen provenance

- Native generator/evaluator source: `19acf0734d63ecebc07140c8d84302a54baee4c4`.
- Base PiPER study: `191856b`.
- Protocol SHA-256: `d48e363d24cc3c9f3be67ffb87390aa3f9b37272bd28d4b9b22424ccf336fba4`.
- Cases SHA-256: `f8b1b205b16e9833df7bd7063e0572355794dcede49a024f5d73ae6f65bbb1aa`.
- Evaluation-plan SHA-256: `5f02006190d97bdaca43f139f5bf221cece23bba868a5e9ee0f424618ea87d8f`.

The presentation, independent verifier and publisher are later additions in this
branch. Their source and artifact hashes are retained separately; they do not
change the frozen cases or Jev prompts. Native GPU rendering can vary slightly
between runs; the four within-run alias pairs must remain identical.

```sh
python scripts/verify-image-goal-fixtures.py \
  --fixtures PATH_TO_DOWNLOADED_FIXTURES --output native-validation.json
```

The verifier reads frozen native observations; it does not re-run physics or
contact Jev. Use the documented generator command for a fresh native run.

Original DVIDIA code and authored fixtures: MIT. PiPER assets preserve their
upstream MIT notice. Hosted Jev remains subject to TypeSafe's separate terms;
no weights, credentials or customer footage are distributed.

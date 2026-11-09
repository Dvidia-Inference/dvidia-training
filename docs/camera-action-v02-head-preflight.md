# Camera-to-action v0.2: matching initial action heads

**9 November 2026 — a standalone numerical prerequisite passes.** Two proposed
decoders give identical outputs at the pinned PiPER reset reference, before
projection, and use the same endpoint-loss units. This does not train a policy
or change the v0.1 result. The five-candidate v0.2 learning screen remains unrun
and unregistered. This parameterization choice needs explicit registration
before any learning or rollout.

## The comparison

The original v0.1 feature vector stores normalized previous references as
float32. It cannot reconstruct exact float64 limiter state. A later controller
must obtain `r` from its exact preceding applied reference, separately from the
learned features. This preflight uses declared synthetic references only.

Both candidates share the same seeded float32 269→64→64→7 network, including
both hidden layers and the final head. Let its output be `g(x)`:

```text
D = [0.03 rad × 6, 0.001 single-finger m]
S = [3 rad × 6, 0.035 single-finger m]
Cabs endpoint   = r0 + D * g(x)
Cdelta endpoint = r  + D * g(x)
loss           = mean(((endpoint - target_endpoint) / S)²)
d(loss)/d(g)    = 2 * (endpoint - target_endpoint) * D / (7 * S²)
```

`Cabs` predicts an absolute endpoint through a fixed affine parameterization;
its offset is always the original reset reference `r0`. `Cdelta` uses the
current reference `r`. Both decoding and loss arithmetic use float64. The
source-bound reset is `[0, 1.57, -1.3485, 0, 0, 0, 0.035]` in the declared
joint/jaw order. The script parses those literals and the reference-copy
assignment from the hash-pinned `PiperTaskEnv.reset`; it imports no native code.

At reset, `r = r0`, so both decoded functions match for any shared head value.
Later, their output difference is `r - r0`. That difference is intentional.
Using common `D` also gives both heads the same endpoint Jacobian and loss
scaling. This avoids comparing heads with different parameter scales merely
because one output represents an increment. It does not prove equal later
gradients, optimizer trajectories, function classes or manipulation performance.
No Adam update or full network-backpropagation comparison is performed.

## Frozen numerical evidence

The script writes the complete cases, random initialization and protocol before
measurement. Parameter seed is 17; case seed is 171711. There are 64 synthetic
feature vectors and two reference conditions per vector: reset and later.
These are bounded numerical inputs, **not** captured observations or additional
training episodes. Native-unit output comparisons use absolute tolerance
`1e-12`, zero relative tolerance; central-difference derivative checks use
step `1e-4` and absolute tolerance `1e-8`.

| Check | Result |
|---|---:|
| Reset decoded outputs, before projection | 64/64 bitwise equal |
| Reset loss and output-head gradients | 64/64 equal within tolerance |
| Later-reference offset identity | 64/64 within tolerance |
| Largest offset-identity error | 2.22e-16 native units |
| Largest central-difference loss-gradient error | 5.29e-13 |
| Later cases with different decoded outputs | 64/64 |
| Changed-contract/reference-kind checks refused | 9/9 |

The comparison applies **no projection or joint clipping**. In 36 reset cases,
the random head produces an output outside native bounds; the matching test
still checks those original values. Thus saturation cannot manufacture an
initialization pass. These outputs must not be executed. The existing separate
[limiter preflight](camera-action-v02-preflight.md) covers numerical reference
evolution; this head preflight does not run microsteps or physics.

All 15 focused tests passed, including changed source/reset rejection, altered
units/order/reference authority, loss derivatives, a mislabeled later state,
unknown decoder and preservation of existing output. The numerical run took
0.131 s on shared arm64 macOS, Python 3.12.9 and NumPy 2.5.3. That timing includes
comparisons, central differences, negative checks and measurement writes;
it excludes imports, source validation, case/parameter/protocol writes and the
final report. It is not an inference or training performance measurement.

## Reproduction and limits

```sh
python -B benchmarks/camera-action-v02-head-preflight/preflight.py \
  --output /absolute/new-numerical-evidence
python -B -m unittest discover \
  -s benchmarks/camera-action-v02-head-preflight -p test_preflight.py -v
```

Only NumPy and the three pinned local source files are required. Run tests in
their own interpreter: the no-native-import test intentionally rejects a
process that already imported `dvidia_training` or MuJoCo. No recording, trained
model, cloud service, renderer or physical arm is used. The retained random
initialization is synthetic and never fitted; it is not a skill package.

The [protocol](../benchmarks/camera-action-v02-head-preflight/evidence-20261109/protocol.json),
[cases](../benchmarks/camera-action-v02-head-preflight/evidence-20261109/cases.json),
[per-case measurements](../benchmarks/camera-action-v02-head-preflight/evidence-20261109/measurements.json),
[report](../benchmarks/camera-action-v02-head-preflight/evidence-20261109/report.json)
and [test log](../benchmarks/camera-action-v02-head-preflight/tests-20261109.log)
retain the exact scope. Protocol SHA-256:
`72cf2af974193300315b3db477ecd983300e8ead75d983cd6e8c55dcbc135a84`.
Report SHA-256:
`cc2e81e6a3922461e36aa6659754c128aff763c75080ca0b441fef2a3177ff81`.

Before integration, separately register this common-scale parameterization,
validate exact reference ancestry, and test decoding with real registered reset
contexts and the chosen serialization/inference dtype. Use identical final
feasibility projection and native-unit scoring for both candidates. Equal
initial commands alone are not evidence of improved training or a qualified
skill; the frozen v0.1 candidate remains unqualified.

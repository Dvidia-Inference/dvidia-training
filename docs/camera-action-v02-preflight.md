# Camera-to-action v0.2: reference-limiter preflight

**9 November 2026 — the numerical reference-equivalence gate passes.** This
checks one prerequisite for the proposed Cabs/CΔ action-representation contrast.
It does not train a policy, improve the v0.1 result, or qualify an installable
skill. The frozen v0.1 source and evidence were not changed.

The [executable check](../benchmarks/camera-action-v02-preflight/preflight.py)
compares the original 20 reference updates against issuing their projected
endpoint. It executes only the two NumPy update statements extracted from the
hash-pinned `PiperTaskEnv.step`; no simulator, renderer, policy, teacher or
recording is loaded. The pinned upstream PiPER XML is parsed to check joint
ranges, actuator order and the inherited reference ranges. The XML is not
compiled into a physical model in this check.

## What was frozen and measured

The original defaults remain 1 ms per update, 20 updates per decision,
1.5 rad/s arm-reference slew and 0.05 m/s single-finger jaw-reference slew.
Thus a 20 ms interval allows at most 0.03 rad per arm reference and 0.001 m in the
jaw reference. These are command-reference limits, not guarantees of actual
joint speed, jaw separation, contact force or stopping distance.

Before comparison, the script wrote the complete case list and protocol:

- 146 explicit hold, native-boundary, mixed-direction, within-one-tick,
  exact-tick, interval-edge and one-ULP-from-boundary cases;
- 2,048 independent uniform bounded reference/target pairs;
- 2,048 bounded pairs near the 20 ms slew allowance;
- a fixed NumPy PCG64 seed of 171709;
- a fixed absolute tolerance of 1e-12 in each native coordinate, with zero
  relative tolerance.

These are synthetic numerical vectors reserved for development checks. They
are **not** collected robot episodes, extra training examples, or confirmation
scenes. No v0.1 training, development or confirmation recording was read.

| Comparison | Cases within the frozen tolerance | Bitwise-identical cases | Largest arm error | Largest jaw error |
| --- | ---: | ---: | ---: | ---: |
| Closed-form endpoint issued through original updates | 4,242/4,242 | 883/4,242 | 1.33e-15 rad | 2.78e-17 m |
| Original 20-update endpoint issued again | 4,242/4,242 | 4,242/4,242 | 0 | 0 |
| CΔ numerical encode/decode, then original updates | 4,242/4,242 | 883/4,242 | 1.33e-15 rad | 2.78e-17 m |

Each comparison checks all **593,880 reference entries**, not only the final
endpoint. Closed-form projection and repeated floating-point addition are not
bitwise equivalent in general. The pass is explicitly numerical equivalence
under the predeclared tolerance; a consumer requiring exact byte reproduction
must preserve the original sequence of 20 updates and its dtype/operation order.

All 24 negative contract/input cases were rejected on both preflight paths.
They include changed rates, timestep, interval, microstep count, joint/jaw
limits, reordered joints, nonfinite or nonnumeric values, out-of-bound references,
and measured/future/teacher-target state substituted for a previous applied
reference. These are **preflight input guards**, not claims that the native
simulator implements a telemetry provenance check.

Twenty unit tests passed, including changed source/XML rejection, preservation
of an existing output directory, and a deliberately reordered microstep history
that reaches the same endpoint but must fail equivalence. Source-parser mistakes
found while developing the check were fixed before the frozen numerical run;
the retained development test log is separate from the final passing log.

The measured comparison section took 3.370 seconds with Python 3.12.9,
NumPy 2.5.3 and ARM64 macOS 26.5.2. This single local wall clock includes
comparisons, negative checks and measurement-file writing; it excludes initial
source/XML validation, vector generation, protocol/case writes and final report
writing. It is not a robot inference, training or dedicated hardware speed
benchmark. No CPU model, energy use or monetary compute cost was measured here.

## Why the endpoints agree

For each coordinate, let `r` be the preceding **applied servo reference**, `a`
the fixed teacher target, `d` the unchanged per-tick reference allowance, and
`N = 20`. Exact-real arithmetic gives the original reference sequence

```text
r[k] = r + sign(a-r) * min(abs(a-r), k*d),  k = 1..20
P    = r + clip(a-r, -20*d, +20*d)
```

If `a` is reached inside the interval, `P = a`. Otherwise `P` is exactly the
last allowed reference in the original direction. For every `k ≤ 20`, issuing
`P` therefore traverses the same reference sequence. Because `r` and `a` are
both within each original coordinate bound, this monotone sequence also stays
within it. The numerical check measures the small departures introduced by
float64 subtraction, repeated addition and CΔ encode/decode.

This argument depends on a constant target, constant independent slew limits,
the preceding applied reference and an unchanged update interval. It does not
cover a measured-joint-relative target, an actuator-opening pulse, variable
rates, acceleration limiting, force control or coupled physical dynamics.
A typed `previous_applied_servo_reference` field can reject a declared wrong
state type; it cannot prove that a dishonest producer supplied authentic
telemetry. Future episode integration must bind that reference to the preceding
sealed actuator command exactly as the v0.1 export does.

## Decision and next bounded screen

The reference-projection prerequisite in the existing
[v0.2 proposal](camera-action-v02-proposal.md) can proceed to implementation of a
**separately versioned** Cabs/CΔ model/export contract. The remaining prerequisites
are still outstanding: matching initial decoded model heads, identical feasible
endpoint labels and loss units, training-only normalization, exact preceding
reference bindings, and fresh predeclared evaluation groups.

Keep the proposed contrasts separate: learner-state coverage versus B0,
physical-scale absolute targets versus B0, and CΔ versus Cabs. Do not combine
favorable choices or tune against v0.1 confirmation. The proposal's 164-attempt
screen remains **unrun** and requires storage/resource preflight and registration
before collection. It awards no qualified-skill label by itself. Fewer clipped
commands, smaller prediction loss or fewer contacts without task completion
would not establish progress toward an acquired skill.

## Evidence and reproduction

[Full protocol](../benchmarks/camera-action-v02-preflight/evidence-20261109/protocol.json),
[all numeric vectors](../benchmarks/camera-action-v02-preflight/evidence-20261109/cases.json),
[every comparison](../benchmarks/camera-action-v02-preflight/evidence-20261109/measurements.json),
[report](../benchmarks/camera-action-v02-preflight/evidence-20261109/report.json),
and [passing test log](../benchmarks/camera-action-v02-preflight/tests-20261109.log)
are retained. The source base is
`ab9082e1ee2b2cc91c7c5fd3f1358727b3789c66`; six source-file hashes and the upstream
PiPER XML hash are included in the protocol. No credentials, model weights or
human media are part of this evidence.

| Artifact | SHA-256 |
| --- | --- |
| Protocol | `55a32578740170841a26bf3c06a36778c8c5e56595cd5003967a2c67dd3fffc1` |
| Cases | `5ffc870ddb0b6e7b7187a41558b17b3b47c9a3df479cdd4df949bc814648f766` |
| Measurements | `3b0f235bdee6f5ef3add2e793fcb4d625f926eb28de1fb6f348dc79df2ad345c` |
| Report | `b27b758bd591196aa7f37c788fe12ca70548eab2d3adf1d830f1601ff8eeb87f` |

Use a fresh output path. The only extra input is the existing separately pinned
PiPER XML; no mesh or native renderer is needed:

```sh
python benchmarks/camera-action-v02-preflight/preflight.py \
  --piper-xml /path/to/verified/assets/agilex_piper/piper.xml \
  --output /path/to/new/numerical-evidence

DVIDIA_PREFLIGHT_PIPER_XML=/path/to/verified/assets/agilex_piper/piper.xml \
  python -B -m unittest discover \
  -s benchmarks/camera-action-v02-preflight -p test_preflight.py -v
```

The script binds to the exact original six source files. A changed source or XML
fails closed rather than silently becoming a different numerical experiment.

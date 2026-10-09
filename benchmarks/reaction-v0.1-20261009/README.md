# Reaction v0.1 developmental measurements — 9 October 2026

[Recorded summary](summary.json) and [all primary attempt summaries](attempts.jsonl) bind these observations to source, protocol,
engine and fixture identities. These are synthetic mechanical coupons on one
CPU host, with no camera perception, learned task policy or physical qualification.

The run contains 13 authored cases × 3 repeats × 2 intervention variants:
**78 primary attempts**, plus **8 timestep-comparison attempts**. None emitted
native numerical errors. All attempted trials, diagnostic failures and drops
remain in their declared denominators.

| Observation | Result and scope |
| --- | --- |
| Detector with braking enabled | 24/24 authored positive conditions detected; 0/15 negative cases flagged. Three repeats of the same case families are correlated; this is not broad detection accuracy. |
| Modeled command delay | 20 ms from request to application. |
| Application to confirmed rest | Median 293 ms; maximum 391 ms, including the declared 50 ms quiet dwell. This is modeled mechanical motion, not hardware timing. |
| Native episode throughput | 109.2 rollout seconds / 8.812 per-trial wall seconds = 12.39× real time. Wall includes compilation, warmup, sensor priming, monitor, scoring and trace construction; rollout numerator excludes warmup/priming. |
| Primary benchmark wall | 8.872 s including JSONL writing; additional timestep work 1.228 s. Imports and final report serialization are excluded. |
| CPU monitor calls | Median of per-episode p95 values: 9.79 µs on this host. A per-episode observed percentile is not a hard real-time guarantee. |
| Contact diagnostics | Late-entry attempts exceed declared penetration/force diagnostics in both intervention variants. Finite engine state does not establish physically credible impact forces. |
| Retention | Stable-grip cases retain their load. Slip cases drop it in both variants: stopping horizontal motion does not supply a regrip/set-down recovery. |
| Timestep comparison | Detection/rest timing agrees within declared tolerances at 1 ms and 0.5 ms. Contact-force/impulse diagnostics fail agreement; precise impact forces remain unqualified. |

Anticipatory braking avoids native contact in the crossing and static-obstacle
cases; matched no-brake controls collide. The late occluded entry still contacts
the fixture. The controller runs in shadow in disabled controls, preserving its
detection result while withholding the intervention.

The recorded host is Darwin arm64 with 10 logical CPUs and no GPU use. CPU model
and total RAM were not identified in this run. Process lifetime peak RSS was
75,350,016 bytes, including imports and all trials; it is not a minimum requirement.

Reproduce from the current source with Python 3.12+ and the optional arm extra:

```sh
dvidia-reaction benchmark --output runs/reaction-new --repeats 3 --convergence
dvidia-reaction inspect runs/reaction-new
```

This committed snapshot omits full traces and compiled scene declarations. A
fresh run writes complete native traces, attempted-trial JSONL, frozen protocol,
source identities, convergence receipts, readable HTML and integrity manifest.
The CI arm job also runs the framework and retains its diagnostic artifacts.
The inspector consumes a complete run directory, rather than this summary.

These inspected scenarios are developmental/regression evidence. Changes to a
learner, controller or physics profile need fresh confirmation for qualification.
See [framework contracts](../../docs/reaction-framework.md).

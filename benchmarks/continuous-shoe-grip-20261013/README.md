# Continuous shoe grip v0.2 — frozen comparison

Source: [`af8933994b4626e28a0554abc782ae9d22cff06c`](https://github.com/Dvidia-Inference/dvidia-training/tree/af8933994b4626e28a0554abc782ae9d22cff06c).

The fresh comparison retains **20/20 valid native attempts**. Both authored controllers complete **2/10**, on the same two cases; there are no gains or losses. Both enter the retrieval region in 8/10 and correctly defer two short-tool cases. The new controller issues 12 bounded lowering responses across six unsuccessful grasp cases. Those cycles add 34.02 simulated seconds and produce no additional completed tasks.

[Research paper](https://research.dvidia.org/papers/continuous-shoe-grip-v02/) · [Recorded comparison and full receipts](https://research.dvidia.org/downloads/topics/physical-grounding/runs/2026-10-13-continuous-shoe-grip-v02/lab/) · [Hugging Face](https://huggingface.co/spaces/Dvidia/robot-simulation-lab) · [Run locally](../../docs/continuous-shoe-grip.md)

## Scope and evidence

`baseline` is the original **mounted-rake** v0.1 controller. It is not the earlier no-tool capability control. All ten paired compiled-model hashes agree. The new controller uses continuous acquisition/loss checks, a 40 mm observed TCP checkpoint during continued lifting, and a bounded supported retry. The checkpoint is not independent shoe-lift success.

- [Fresh benchmark index](benchmark.json), [protocol](protocol.json), [full export inventory](integrity.json).
- [Historical regression index](regression-benchmark.json), [protocol](regression-protocol.json), [source export inventory](regression-integrity.json): 18/18 valid attempts, 1/9 completed under either controller on the same known development case. These are previously seen cases.
- [Development ledger](development-ledger.json): all nine tuning rounds are summarized. First-five source hashes were not recorded; no retroactive provenance is claimed. Its first iteration's peak height included initialization and is not an earned lift.
- [Timing and outcome derivation](analysis.json).
- [Independent native audit](native-audit.md), [machine-readable findings](native-audit.json), [portable reproduction script](native_audit.py) and [local release acceptance](acceptance.json).

This directory contains compact results, not the full replay exports. `integrity.json` inventories the complete fresh 87-file export on the research site and Hugging Face. Regression replay paths refer to the original local export; regression trajectories are not included in this compact directory. The development ledger refers to local historical iteration artifacts by name and hash, rather than bundling those full trajectories.

The independent audit verifies nine bilateral-contact-loss alarms at **20–52 ms, median 30 ms**, after the latest uninterrupted native contact-loss segment. Three separate acquisition-failure alarms have no acquired-grip loss origin and are excluded. Three failed 20 ms samples can straddle brief native contact reacquisition. Recorded request-to-packet timing is not physical stopping or response time. All twelve lowering responses establish world support before reopening; none completes afterward.

## Remaining failures

The independent scorer requires the shoe's sole-bottom frame above 35 mm, both finger contacts above 0.01 N, and world/rake/non-finger robot support below 0.01 N for every tick of the final 500 ms. A tilted high pose or earlier good hold cannot pass by itself.

Both policies have an uncovered non-finger-support interval in completed case `fresh-05`: 82 failed ticks from 19.076–19.238 s, with 100 ms requalification complete at 19.338 s. This is support contamination, not evidence of physical slipping. Neither controller detects it; the final window passes after it clears. All 12 new controller detections are unmatched to the independent loss witness, which requires prior 100 ms qualifying retention. They are not automatically false alarms or measured missed-slip rates.

The [geometry probe summaries](geometry-probes/probe.json), including [the lower-envelope diagnostic](geometry-probes/lower-envelope.json), retain 32 unsuccessful probes across seven files. The diagnostic lowered the software TCP floor to 35 mm; the release retains its original 55 mm floor. One diagnostic reached a 181.6 mm shoe frame height while carrying 1.816 N of rake support, so it correctly failed. Finite probes do not prove grasp infeasibility; they motivate examining gripper clearance and the mounted tool's interference before adding more retries.

## Compute

On Apple M5 / 10 logical CPUs / 24 GiB RAM, macOS arm64, Python 3.12.9, MuJoCo 3.15.0:

| Measurement | v0.1 mounted-rake baseline | v0.2 recovery |
| --- | ---: | ---: |
| Simulated task duration, summed | 131.60 s | 165.62 s |
| Native execution wall time | 4.9601 s | 5.7716 s |
| Separate audit and export wall time | 8.9476 s | 8.7451 s |
| Recorded target-solver calls | 6,580 | 8,281 |
| Target-solver p95 / maximum | 0.0305 / 0.5635 ms | 0.0316 / 0.3753 ms |

Native timing includes setup, control, physics and recording. Target-solver samples exclude most monitoring, contact observation, physics, network and hardware latency. None of the recorded solver calls exceeds 20 ms; that is not a whole-loop or hard-real-time guarantee. No GPU is required for this native benchmark. Browser replay needs WebGL2; minimum RAM is unmeasured.

These are authored policies with privileged observations, rigid synthetic shoes and assumed contact. No footage-trained model, force calibration or physical-robot qualification is claimed. Code and synthetic assets are MIT; upstream notices remain with the complete export.

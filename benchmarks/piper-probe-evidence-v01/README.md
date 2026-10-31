# PiPER probe evidence v0.1

This local CPU simulation study compares two authored short-lift admission rules on the same 12 synthetic scenes: an evidence gate and a fixed timer gate. Each scene is run once per variant (24 scheduled attempts total). The object masses are 40 g and 60 g, friction settings are 0.80, 0.55 and 0.15, and two authored XY layouts are used.

Both policies command a TCP probe target up to 10 mm above probe start at 20 mm/s, then request a hold at measured arm joints and the last issued jaw target. Actual object/TCP motion can continue under native dynamics and target rate limits. They do not continue to a full transport or placement. The evidence policy admits only after bilateral native pad force above 0.01 N, no table support, at least 6 mm object rise, and no more than 3 mm TCP-local offset drift continuously across at least 100 ms of sampled control boundaries. The timer policy admits at 600 ms after probe start without consulting that predicate. Unexpected contacts and bounded timeouts still stop the attempt. One-millisecond native records audit the policy claims; they are privileged simulator state, not physical sensors.

The paired attempts are closed-loop runs. Early holds, alarms and supported retries can change subsequent paths, so differences are not an isolated classifier-causality estimate. All scheduled attempts and claim disagreements remain in the report. Each final planning gate retains declared assumptions; the shared 10 N per-finger value is an arbitrary synthetic admission ceiling, not a measured damage limit or force controller. Unknown quantities remain unknown.

## Frozen run

All 24/24 native attempts were valid. Evidence admission occurred in **6/12** attempts; reconstruction over the recorded 1 ms rows classified **zero false initial qualifications** and **zero conservative abstentions** among the supplied claims. Four of those six correctly admitted probes later lost the required probe height; two ended in a qualified hold. The timer admitted **12/12** attempts, with **8 false initial qualifications** and zero conservative abstentions. Ten timer attempts were invalidated in total: the eight false admissions plus two of the four initially correct admissions. Six invalidations followed support return and four followed loss of probe height. Two attempts ended in a timer hold. At the final observation, each variant had 2/12 `ready_for_simulated_transport` planning gates and 10/12 `test_lift_required`. A planning status is not transport execution or actuation permission.

An independent readback audit reconstructed the native predicate from the saved arrays without importing receipt derivation or rerunning physics. It verified all 24 attempts and 480,000 native rows, with frozen source and raw evidence unchanged. Eight timer admissions lacked a qualifying 100 ms native window at the decision; four initially valid evidence admissions later lost the hold condition. Both variants finish with 2/12 qualified. Closed-loop trajectories differ, so these results do not isolate classifier causality. The audit receipt SHA-256 is `b333f788d388f5e43a035b6bcd85bb6517e197d24a18c92c12efbd6ae3ccbe8f`.

| Variant | Primary admissions | False qualifications | Conservative abstentions | Later invalidations | Ended in admission hold | Final planning gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Evidence | 6/12 | 0 | 0 | 4/6 admitted | 2/12 | 2 ready · 10 test lift required |
| Timer | 12/12 | 8 | 0 | 10 total (2/4 correct admissions) | 2/12 | 2 ready · 10 test lift required |

The audit categories keep false positive admission separate from conservative abstention. These are finite synthetic scene outcomes, not estimated rates for physical objects or robots.

Across the twelve attempts per variant, mean policy-call CPU compute was 0.045 ms for evidence and 0.044 ms for timer. Mean whole-step CPU compute was 1.707 ms and 1.722 ms, respectively; mean per-attempt step-compute p95 was 2.872 ms and 2.559 ms. Across recorded step wall samples, 5/12,000 exceeded 20 ms for each variant. These are offline timings on an unpaced shared host; complete-step values include native step work and dense recording, and exclude evidence serialization, physical sensing, transport and actuation. They are not real-time guarantees.

This is not a learned policy, new learning result, arbitrary-video-to-robot transfer, complete placement benchmark or hardware qualification. An earlier PiPER learned policy result of 5/12 remains unchanged; this study does not train or reevaluate it.

## Reproduce and inspect

Use Python 3.12 or newer. This CPU-only study needs no GPU, account or API key. Provisioning downloads the pinned public PiPER assets once; subsequent runs are offline. The experimental source branch leaves the stable release unchanged.

```sh
git clone --branch feat/piper-probe-evidence-v01 https://github.com/Dvidia-Inference/dvidia-training.git
cd dvidia-training
python3.12 -m venv .venv
.venv/bin/python -m pip install '.[arm]'
.venv/bin/dvidia-robot-sim fetch piper --output runs/piper-assets
.venv/bin/python -m dvidia_training.piper_probe_benchmark prepare \
  --assets runs/piper-assets --output runs/my-probe-study
.venv/bin/python -m dvidia_training.piper_probe_benchmark run \
  --assets runs/piper-assets --output runs/my-probe-study
.venv/bin/python -m dvidia_training.piper_probe_benchmark inspect \
  --output runs/my-probe-study
.venv/bin/python benchmarks/piper-probe-evidence-v01/render_report.py \
  runs/my-probe-study/report.json
```

Use the corresponding `.venv/Scripts/` executables on Windows. Each experiment requires a fresh output directory; the runner refuses to overwrite or resume a frozen run. The protocol binds local source and asset bytes. Reproduction timings and compiled model hashes may differ by machine. The published study used Python 3.12.9, NumPy 2.5.3 and MuJoCo 3.15.0. About 113 MB of raw evidence was produced; no minimum RAM or real-time target has been qualified.

The renderer uses only Python's standard library and writes a self-contained `report.html` beside the report JSON. The report, protocol, independent audit and inspection links in the rendered page resolve to colocated files.

## Publication links

- Research article: [piper-probe-evidence-v01](https://research.dvidia.org/papers/piper-probe-evidence-v01/)
- Source release: [piper-probe-evidence-v0.1-preview](https://github.com/Dvidia-Inference/dvidia-training/releases/tag/piper-probe-evidence-v0.1-preview)
- Evidence dataset: [Dvidia/piper-probe-evidence-v01](https://huggingface.co/datasets/Dvidia/piper-probe-evidence-v01)

Original DVIDIA source and authored synthetic records are MIT. Simulation assets and dependencies retain their upstream licenses and notices.

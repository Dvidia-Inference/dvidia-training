# Learned PiPER grip feedback v0.2

The new 25-node jaw adapter preserves the frozen six-joint motion primitive. All 120 confirmation attempts were valid.

| Policy | Strict success | Nominal | Interrupted |
| --- | ---: | ---: | ---: |
| Learned jaw feedback | 6/24 | 6/12 | 0/12 |
| Matched blind jaw learner | 8/24 | 7/12 | 1/12 |
| Frozen v0.1 primitive | 5/24 | 4/12 | 1/12 |
| Authored teacher | 24/24 | 12/12 | 12/12 |
| Open jaw control | 0/24 | 0/12 | 0/12 |

Live feedback did not demonstrate improved reliability. Fourteen feedback trials passed terminal-only settling, but only six passed the complete manipulation endpoint. Finger-table contacts occurred in twelve feedback trials. Ten of twelve interrupted feedback trials received an effective applied-target disturbance, all before established grip. There were no qualified feedback restoration events.

The audit independently rescored all 360 attempts, including 64 original teachers, 32 mixed-policy attempts, 80 unsuccessful residual-development trials, 64 jaw-development trials and 120 confirmation trials. It found zero native-metric mismatches. The five original residual candidates each completed 0/16 development cases and remain under their original source/protocol bindings.

Feedback inference p95 was 0.501 ms and the full step p95 was 5.648 ms; 34/24,000 full steps exceeded 20 ms. These CPU timings exclude cameras, transport and physical actuators. This is privileged-state synthetic simulation research, not human-video learning, hard real-time qualification or physical robot transfer.

[Paper and replay](https://research.dvidia.org/papers/learned-piper-grip-v02/) · [Model](https://huggingface.co/Dvidia/piper-block-placement-feedback-v02) · [Full evidence](https://huggingface.co/datasets/Dvidia/piper-block-placement-feedback-demonstrations-v02) · [Local installation](../../docs/piper-learned-feedback.md)

## Evidence bindings

- `protocol.json`: `c35622d4ddc7e6868fd609a5665e931d45788345cd9ed2238fbb9ebac5b24fc9`
- `frozen.json`: `b49c20ec464581a3bd3bfea5b6962ae4fe219966fbf998fbdc2c0774d8d042ca`
- `report.json`: `b2417ba4733afeb8239eb768a3daac827f4faf271a8b4df5384b7454e827e6ec`
- `audit.json`: `029bb4e3eb39afe2f1933a7935842dacf2f27f2e204ecf985a7ba3e988576366`
- `training-parent.json`: `75779fc36cdfdf790a810d588c45fbb836773e73922e57202790d7a7cdcd0d44`

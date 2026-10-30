# Native PiPER action capture v0.1

Engineering collection validation for one fixed synthetic start. Two full native captures contain 1,000 control intervals and 1,001 RGB/depth observations each; all paired frames, measured motion and applied controls match exactly. A declared interruption preserves two complete intervals with an unknown outcome. The two full runs are repeats of one trajectory, not independent skill examples.

Independent readback decodes all 2,005 registered RGB images and depth arrays and checks their hashes and causal boundaries. All four object tracker channels remain missing, so strict training-target export refuses all three captures. Native scorer success in the full runs is an authored-controller diagnostic; recording task results remain unverified. No correction learning, hardware command or generalization result is claimed.

The shared macOS arm64 host records 20 simulated seconds in 13.516/13.298 seconds, excluding final receipt/integrity sealing. Boundary overruns above 20 ms are 11/1,000 and 6/1,000. Simulator time pauses for capture; depth precision has an explicit CGL warning. These are not hard real-time or minimum-hardware qualifications.

The native preview is **0.1.3.dev1**, separate from the stable 0.1.2 data-only SDK. All 37 focused tests pass (27 recorder +10 capture). A fresh isolated wheel installation captures five genuine native intervals outside the source checkout. See the [capture guide](../../docs/piper-action-capture.md), [paper](../../reports/Native%20PiPER%20action%20capture%20v01.md), and compact JSON proofs here. The complete immutable capture kit is available in the [preview release](https://github.com/Dvidia-Inference/dvidia-training/releases/tag/piper-action-capture-v0.1-preview).

# DVIDIA reference-model training and inference integration

Two small PiPER research candidates now train and serialize successfully from the same 96,000 training rows. The absolute-reference fit took 5.230713 s and the prior-reference delta fit took 5.097628 s on the measured shared CPU host. Both saved models reproduced all 1,000 outputs in the recorded-trajectory replay. This validates the learning/inference bridge, not task completion.

- [Absolute candidate card](models/absolute/README.md)
- [Delta candidate card](models/delta/README.md)
- [Derived-data card](data/README.md)
- [Compact integration receipt](evidence/integration-result.json)
- [Training reference identity](evidence/training-input-identity.json)
- [Initial native smoke refusal accounting](evidence/initial-smoke-accounting.json)
- [All three engineering batches](evidence/development-smoke-status.json)
- [Independent eight-attempt native audit](evidence/native-smoke-v3-audit.json)
- [Renderer warnings](evidence/renderer-warnings.json)
- [Strict-state failure diagnosis](evidence/native-smoke-diagnosis.json)

The third separately registered native batch completed eight attempts: **0/4 strict placements for each candidate**. Cabs made bilateral contact in 3/4 and a qualified lift in 1/4; Cdelta contacted in 4/4 and lifted in 0/4. Neither completed carry. Unexpected contacts occurred in 2/4 Cabs episodes and 4/4 Cdelta episodes. All four delta traces crossed the strict scorer’s finite TCP coordinate envelope of ±0.5 m; this was not a NaN or physics-warning failure. Full per-case evidence is retained, together with the sixteen earlier infrastructure refusals. No candidate was selected or qualified; the proposed 164-attempt screen and frozen v0.1 result are unchanged.

The two candidate model files are included under `models/absolute/model.json` and `models/delta/model.json`. They are usable through the pinned Python API; this new model format is not supported by the qualified-skill CLI installer. Raw training recordings and derived reference arrays are not redistributed here. Original artifacts and source commit `78ea4fbb4b2733e7738eb425eeecd7cb67a52195` are unchanged. Exact original result SHA-256: `6d44d98488ef26f949412aadc94c32174bd5c4ce879442191b9784deff5b325f`.


## Inspect a retained candidate

Use a source checkout containing this benchmark and the unchanged runtime bytes identified below. Install the package and its existing NumPy dependency with `python -m pip install -e .`, then run Python from the repository root:

```python
from dvidia_training import camera_action_reference_model as reference_model

model = reference_model.load_model(
    "benchmarks/camera-action-reference-model-integration-v01/models/delta/model.json"
)
assert not model["task_qualified"]
assert not model["physical_robot_ready"]
policy = reference_model.ModelPolicy(model)
print(model["variant"], model["training"]["samples"])
```

Loading does not start a simulator, read a camera or command hardware. Calling `policy(context)` requires the recorder's exact causal context and retained float64 previous reference, beginning at the registered reset; an arbitrary image or joint vector does not satisfy that contract. Any next native experiment requires its own registration; these two candidates remain unchanged. Do not infer operational task success from successful loading or a training-trajectory replay.

Source at measurement: [`78ea4fbb4b2733e7738eb425eeecd7cb67a52195`](https://github.com/Dvidia-Inference/dvidia-training/tree/78ea4fbb4b2733e7738eb425eeecd7cb67a52195). See [the model bridge contract](../../docs/camera-action-reference-model-v01.md) and [integration notes](../../docs/camera-action-reference-model-integration-v01.md).

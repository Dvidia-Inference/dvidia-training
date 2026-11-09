# Reference-endpoint model bridge

This additive Python module connects the verified training export to a trainable
model and a callable local simulation policy. It implements the proposed Cabs
and Cdelta comparison; **no real-study model has been fitted or evaluated with
it**. The frozen v0.1 result remains 5/64 strict learned placements and
unqualified. The common-scale parameterization and the 164-attempt v0.2 screen
still need registration before real learning or rollouts.

The module is `dvidia_training.camera_action_reference_model`. It changes none
of the existing recorder, trainer, model, installer, actuator limits or scorer.
It provides no hardware commands, new CLI command or installation qualification.
Its policy has the same callable interface that the unchanged
`camera_action_capture.run_rollout` already accepts. That interface compatibility
is tested with synthetic contexts; no native rollout was run for this bridge.

## From sealed training inputs to actions

`train(sources, variant='absolute' | 'delta')` accepts one independently pinned
demonstration reference export, optionally followed by the correction export.
Each source includes its reference directory, parent cache and all independently
retained manifest, protocol, stage, split and source pins. The existing reference
loader verifies both derived artifacts. Development and confirmation data are
refused; the API has no evaluation-data argument and does not accept substituted
unverified feature/label arrays. Both stages must share a protocol. Duplicate
recordings are rejected by the existing parent-data combiner.
The combined 128,000-row cap is enforced before concatenation, normalization
or parameter initialization, including when each input fits the cap separately.

Training uses the original causal 269-dimensional full-feedback features. It
keeps their missing-value masks, historical initial visual anchor and 160 ms
history. Exact previous normal servo references and feasible endpoints are
separate float64 arrays. Each training episode must start at the source-bound
reset reference. Issued teacher targets and mixed-control teacher suggestions
retain their separate parent provenance; a suggestion is not a claim of actual
executed control. No original frames, native traces or query bodies are read by
this module. Raw-data validation is inherited from the pinned parent cache.

The fixed recipe is seed 17, two 64-unit tanh hidden layers, 50 epochs, batch size
512 and Adam learning rate 0.003. Both variants initialize every parameter
identically and use the same example shuffles, float32 weights and Adam moments,
gradient clipping and parameter bounds. Feature mean and standard deviation
come only from the loaded training set. Training and policy normalization both
use float64 arithmetic and then cast the normalized features to float32; saving
and loading preserves this arithmetic exactly.

For network output `g(x)`, the two decoders are:

```text
r0 = [0, 1.57, -1.3485, 0, 0, 0, 0.035]
D  = [0.03 rad × 6, 0.001 single-finger m]
S  = [3 rad × 6, 0.035 single-finger m]
Cabs endpoint   = r0 + D * g(x)
Cdelta endpoint = r  + D * g(x)
loss = mean(((unprojected_endpoint - feasible_target) / S)²)
```

The decoder, loss and backpropagation chain rule use float64. The final network
output remains float32. Training differentiates the **unprojected** endpoint
loss; it does not silently optimize a saturated/clipped output. At execution,
both variants clip to the same original native joint/jaw bounds and apply the
same exact 20-update normal-reference projection. The existing native runtime
then retains its own rate, force and task checks. This bounds reference changes;
it does not guarantee contact safety or bound actual physical joint motion.

At initialization, for the same reset inputs and targets, both variants have
the same decoded outputs, complete-network gradients and first Adam update.
Later references intentionally change the
decoder offset and can change losses, gradients and actions. This is a controlled
parameterization, not evidence that one representation learns better.

## Exact state and a separate model format

`ModelPolicy` reads `context['previous_applied']` directly as double precision;
it never reconstructs that state from normalized float32 features. The first
reference must equal the original reset. Reduced-precision NumPy arrays are
refused. The existing feature encoder enforces a fresh initial visual estimate,
20 ms increasing observation times, fixed destination and causal camera/signal
timestamps. Extra teacher labels, future snapshots and native object/contact
truth do not enter model features or decoding.

The reference's authority comes from the recorder's existing field and its
source contract. The callable interface cannot authenticate an arbitrary caller
that deliberately supplies rounded or false values as ordinary Python floats.
Tests demonstrate that two exact references sharing the same float32 feature
representation still produce distinct delta outputs. They do not pretend a
number alone proves where it came from.

The strict model schema is `dvidia.camera-action-reference-mlp/v1`. It stores the
variant, complete contract and recipe, source identity, normalization, exact
float32 parameters and training provenance. The latter binds parent/reference
datasets, independent input pins, groups, sample counts, initialization and the
fixed optimizer budget. Save uses exclusive creation; load rejects changed
fields, sources, units, bounds, order, recipe or float32 parameter encoding.
Models remain `physical_robot_ready=false`, `task_qualified=false`, with
`scientific_registration=null`. A model file is not an experiment registration
or a qualification receipt, and these hashes are not publisher authentication.

For a future separately registered experiment, the API is:

```python
from dvidia_training import camera_action_reference_model as reference_model

# sources: independently reviewed list of {directory, parent_cache, pins}.
# Each pins object is the argument set for load_reference_export, including
# expected_manifest_sha256; never replace independent pins with file contents.
candidate = reference_model.train(sources, variant="delta")
reference_model.save_model(candidate, "/absolute/new/model.json")
policy = reference_model.ModelPolicy(
    reference_model.load_model("/absolute/new/model.json")
)
# The unchanged recorder can invoke policy(context). Running a new simulation
# is a separate registered experiment, not part of loading this model.
```

## Verification and remaining gate

Run the focused synthetic suite locally:

```sh
python -B -m unittest tests.test_camera_action_reference_model -v
```

Each suite invocation performs exactly **two synthetic 12-row fits**, one per
variant, at the complete 50-epoch recipe. It also checks complete-network
gradient and single-update reset parity, finite-difference derivatives across
all six parameter tensors, legitimate later-state differences, exact shared
normalization, save/load policy parity, missing live observations, causal-input
refusal, independent-pin refusal, label exclusion and native-reference
projection against the original two numerical limiter statements. Those
statements are evaluated as arithmetic only; no physics engine executes.

The full-chain derivative check uses a smooth float64 finite-difference oracle.
Production float32 forward/parameter/moment behavior is checked separately by
bitwise cross-variant reset parity and saved-model inference parity. These are
implementation tests, not task benchmarks. Synthetic fit losses are not
evidence of manipulation progress.

Before the real v0.2 screen: register this parameterization and source, verify
shared initialization on the registered reset contexts, bind the exact dataset
and resource budgets, and then run the declared fits and fresh closed-loop
attempts with independent scoring. Existing confirmation data must remain
untouched. The installer needs a separate explicit extension before this new
model format can be offered as an installable simulation candidate.

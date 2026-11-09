# Exact references for the next camera-to-action experiment

This additive Python API preserves the exact prior servo reference needed to
compare future absolute and reference-relative action heads. It writes two
float64 arrays beside a caller-pinned v0.1 training cache. It does not change
v0.1 features, targets, models, CLI, runtime, released wheel or qualification.
Only synthetic fixtures have exercised this new exporter; no original study
recording or trained model was read for this milestone.

The existing feature cache stores normalized command history as float32. That
history cannot recover the exact float64 servo state. For row zero of each
episode, `previous-references.npy` uses `initial_applied_targets`. For later
rows, it uses the preceding row's `applied.ramped_targets`. These are six joint
references in radians and one **normal single-finger jaw reference** in metres.
Measured joint positions, current issued targets and pulse-overridden native
controls are different quantities and are not used as that reference.

`feasible-targets.npy` contains the endpoint after the original twenty 1 ms
reference updates toward the parent training label: 1.5 rad/s per joint and
0.05 m/s for the single-finger jaw. Each subtraction, clip and addition uses
float64 in the original order; a closed-form projection is not substituted.
Teacher demonstration labels remain issued commands. Mixed-control correction
labels remain separately recorded teacher suggestions, even when the student
actually controlled the arm. Exact prior references follow **actual executed
commands**, never those suggestions.

## Construction and authority

Import `dvidia_training.camera_action_reference_export` from this source branch.
The public API accepts the following explicit caller expectations:

```python
from dvidia_training import camera_action_reference_export as references

pins = {
    "expected_parent_manifest_sha256": parent_manifest_digest,
    "expected_parent_source_sha256": parent_source_digest,
    "expected_protocol_sha256": registered_protocol_digest,
    "expected_stage": "demonstrations",  # or "corrections"
    "expected_split": "train",
    "expected_source_sha256": reviewed_reference_helper_digest,
}
seal = references.build_reference_export(
    immutable_study_directory, verified_parent_cache, fresh_output_directory,
    **pins,
)
# Retain this digest separately from the output directory.
loaded = references.load_reference_export(
    fresh_output_directory, verified_parent_cache,
    expected_manifest_sha256=seal["manifest_sha256"], **pins,
)
```

`source_identity()` and `source_digest()` expose the implementation binding for
review. Supply independently reviewed digests; reading expectations from an
untrusted manifest does not establish authority. The source identity includes
the new helper bytes, unchanged parent cache contract, eight fixed source
hashes, original limiter AST hash, native units and rates. A changed helper
version needs a new reviewed source pin. The parent must be a full-feedback
training cache bound to the exact registered stage and ordered episode roster.
Development, confirmation and blind caches are refused.

Construction revalidates parent **derived bytes** and reads only the registered
study/stage metadata and the recording, integrity and rollout JSON files bound
by that parent. It verifies complete row indices, causal snapshot/clock chains,
episode resets, teacher-query identities and actual-control authority. It also
checks that recorded normal references follow the twenty updates toward actual
issued commands. Metadata, source and parent bindings are checked again before
sealing. No frames, teacher-query bodies, native trace, trained model or physics
are loaded. This is **metadata verification, not a fresh strict raw export**:
the parent's original raw verification and subsequent source immutability are
prerequisites. Hashes detect changed bytes; they do not authenticate a producer
or prove physically truthful metadata. Keep exclusive write access during a
build; filesystem checks do not provide transactional isolation.

The output is exactly a manifest and two C-order little-endian float64 `[N,7]`
NPY arrays. The manifest binds every episode's reset boundary, row offset and
count to the parent dataset, provenance, target array and caller expectations.
No pickle is accepted. Files are sealed read-only, with the manifest written
last. A load revalidates those derived arrays and the pinned parent, recomputes
the twenty-update labels, and returns read-only arrays plus the unchanged
parent dataset. It does not read source recordings or reverify raw footage.
The returned `feasible_targets` are **supervision**, never decoder input.

Use a fresh directory outside both study and parent cache. Existing outputs,
extra files, symlink/hardlink paths, malformed/reordered/truncated arrays,
changed sources and active study workers are refused. Writes preserve a 30 GiB
free-space reserve. Interrupted prefixes remain for diagnosis and are not
accepted or overwritten. The bound is 128,000 rows; a larger experiment needs
an explicitly reviewed contract change.

## Synthetic verification and remaining gate

Run the focused tests using the repository's existing NumPy environment:

```sh
PYTHONPATH=src python -B -m unittest tests.test_camera_action_reference_export -v
```

The tests use tiny authored metadata fixtures. Only fixture parent-cache
construction substitutes the original native recording reader; the new
reference-export build and load operate normally. An independent oracle
extracts and executes only the two hash-bound reference-update statements
from the original source, with no simulator. Tests compare **all twenty
microsteps** and endpoint reissue, preserve two distinct float64 resets that
collapse to the same float32 feature, and check mixed-label separation,
future/truth exclusion, before-seal mutation, stale pins, broken ancestry,
file tampering, split isolation, reserve enforcement and interrupted writes.

The separate [head initialization preflight](camera-action-v02-head-preflight.md)
checks a proposed common-scale absolute/delta parameterization. Integration
still requires a registered learning experiment, real metadata admission,
serialization/inference tests and a runtime path that supplies the actual
retained limiter state. Neither this exporter nor that numerical preflight
fits or executes a policy. There is no new evidence of placement improvement,
physical transfer or a qualified skill; the frozen v0.1 result is unchanged.

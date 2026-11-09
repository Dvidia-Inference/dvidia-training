# Exact references for the next camera-to-action experiment

This additive Python API preserves the exact prior servo reference needed to
compare future absolute and reference-relative action heads. It writes two
float64 arrays beside a caller-pinned v0.1 training cache. It does not change
v0.1 features, targets, models, CLI, runtime, released wheel or qualification.
Synthetic tests and one bounded run on all 96,000 training metadata rows have
now exercised this exporter. Original recording JSON metadata was read; raw
frames, native traces and trained models were not. This validates preparation,
not a learned policy.

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

## Recorded training metadata validation

On 9 November 2026, one run built both artifacts from the previously verified
training caches: 64 teacher episodes and 32 mixed-control episodes, each with
1,000 rows. Exact float64 reset/prior-reference ancestry and every recorded
normal endpoint after the twenty updates passed. All six per-stage loads and
three combined-parent checks retained the independently pinned dataset and
provenance identities. Exactly 294 original metadata files were read: six
study/stage files and three files per training episode. An audit guard refused
other original-study reads and writes and native physics imports.

| Measurement | Observed value |
|---|---:|
| Rows / training episodes | 96,000 / 96 |
| One-time teacher reference construction | 7.253 s |
| One-time correction reference construction | 3.545 s |
| Total construction | 10.798 s |
| Total derived artifact size | 10,832,870 bytes |
| Both validated loads, three successive rounds | 0.303 / 0.258 / 0.233 s |
| Parent combination, separately measured | 0.107 / 0.097 / 0.096 s |
| Parent identity checks, separately measured | 0.254 / 0.207 / 0.183 s |
| Whole validation wall time | 12.575 s |
| Peak process RSS | 1,238,597,632 bytes |

These shared-host measurements used arm64 macOS 26.5.2, Python 3.12.9 and NumPy
2.5.3 with one worker. Construction includes metadata, source and parent-cache
validation. Load timings include derived-byte and parent-cache validation plus
endpoint recomputation. The separately listed identity checks include both
stage datasets and their combination. Whole wall time includes preflight and
validation orchestration, excluding imports, initial output setup and the final
report write. There is no repeated raw export comparator, inference
latency, fitting time or placement-success measurement here.

The [exact receipt](../benchmarks/camera-action-reference-export-v01/full-training-results.json)
and [reproduction instructions](../benchmarks/camera-action-reference-export-v01/)
retain the full scope and source pins. Source helper commit is `a9340b7`; later
CI and test portability changes preserve those helper bytes. The original
per-microstep native traces were not reread. Mixed feasible endpoints describe
what the normal reference would reach toward the teacher suggestion, while
reference ancestry follows the actual executed action. No teacher-control or
physical-truth claim is inferred from those counterfactual labels.

The separate [head initialization preflight](camera-action-v02-head-preflight.md)
checks a proposed common-scale absolute/delta parameterization. Integration
still requires a registered learning experiment, new-data admission,
serialization/inference tests and a runtime path that supplies the actual
retained limiter state. Neither this exporter nor that numerical preflight
fits or executes a policy. There is no new evidence of placement improvement,
physical transfer or a qualified skill; the frozen v0.1 result is unchanged.

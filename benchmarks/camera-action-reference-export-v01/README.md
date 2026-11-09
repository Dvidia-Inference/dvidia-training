# Exact reference export: full training metadata validation

One bounded run validated the 96,000 rows from 64 teacher demonstrations and
32 mixed-control corrections. It built exact float64 prior-reference and
feasible-label artifacts without reading raw images, depth, teacher-query
bodies, native traces, held-out recordings or trained models. No policy was
fitted or executed. The original candidate remains unqualified.

- [Exact result receipt](full-training-results.json)
- [Portable reproduction helper](validate_full_training_references.py)
- [Executed versus portable script provenance](script-provenance.json)
- [Contract, metrics and remaining integration work](../../docs/camera-action-reference-export-v01.md)

Reproduction requires the pinned source checkout at or containing `a9340b7`,
the unchanged v0.1 study metadata, and both independently verified parent caches
from the [full training cache validation](../camera-action-export-cache-v01/).
The reference helper reuses their strict source/data pins; it cannot create
trusted inputs from arbitrary metadata. Parent-cache creation performs the
separate raw verification step. Preserve those source bytes after verification.

Use an existing NumPy environment and a fresh output outside the study/cache.
The measured run used Python 3.12.9 and NumPy 2.5.3 on arm64 macOS, one worker,
with at least 30 GiB free space retained. Example using absolute paths:

```sh
cd /absolute/work-directory-outside-source
PYTHONPATH=/absolute/source/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 /absolute/environment/bin/python -B \
  /absolute/source/benchmarks/camera-action-reference-export-v01/validate_full_training_references.py \
  --source-root /absolute/source \
  --study-root /absolute/sealed-study \
  --parent-cache-root /absolute/parent-cache-root \
  --output /absolute/fresh-reference-validation \
  --python-env /absolute/environment \
  --execution-cwd /absolute/work-directory-outside-source
```

The parent root must contain `demonstrations-cache` and `corrections-cache` with
the exact pinned manifests. The script builds each derived artifact once and
loads each three times. It refuses existing output and retains interrupted
prefixes. It makes no network request. The portable variant changes path
bindings and normalizes platform-specific RSS reporting only; its numerical
and validation operations are unchanged. RSS is null where unavailable.
The portable variant has not run the real benchmark again.

The receipt is the exact original result, SHA-256
`68908bb9111ca92c4d672c25aa58c55417b4197f60a50d89bc96cd4e46bd799e`.
Original script SHA-256:
`3eb816e1145a9e3ad24bda0468ef9312e9cae9f08325948ead8c7d41981b7e45`.
The measured run verifies normal reference endpoints after twenty updates,
not a new replay of each original native microstep. Mixed feasible labels are
counterfactual normal endpoints toward teacher suggestions, not proof that
those suggestions controlled the arm. These are data-preparation measurements;
there is no inference, training-speed or placement-success comparison.

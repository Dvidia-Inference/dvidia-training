# Camera/action v0.2 numerical limiter preflight

The original constant-target reference limiter and the projected endpoint agree
within the predeclared 1e-12 native-unit tolerance on all 4,242 bounded numerical
cases. Closed-form projection is not universally bitwise identical. No robot
skill, model training or physical motion was tested.

See [scope, results and reproduction](../../docs/camera-action-v02-preflight.md).
The v0.1 source, recordings, models and results are unchanged.

`evidence-20261109/` preserves the protocol and complete case manifest written
before the numerical comparisons, all per-case results, 24 negative checks and
the final report. `tests-20261109.log` records 20 passing tests;
`tests-source-parser-development.log` retains an earlier parser failure found
and corrected before the frozen evidence run. The proof binds the exact script,
tests, original source files and original PiPER XML. It contains only synthetic
numerical vectors and receipts.

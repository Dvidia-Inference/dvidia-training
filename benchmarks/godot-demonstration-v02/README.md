# Godot demonstration capture: integration evidence

This is a scripted recording/export check, not a manipulation benchmark or a
new learned skill. The final recording contains ten 50 Hz command intervals,
six 320x240 RGB-D pairs, and 200 native 1 ms ticks. It does not attempt a complete
placement; the independent task scorer correctly returns failure, with no
critical contact ticks. No human demonstrations or model fits were performed.

The original evidence below is pinned to commit `b8b96cd`. Windows CI then
identified a local inventory serialization bug: backslash paths were correctly
rejected by the portable recording validator. The follow-up fix uses POSIX
paths for locally generated inventory entries, retaining strict validation of
imported references. `portability-tests.json` records **97 passing test executions**
including Windows/POSIX path cases. `portability-checks.json` records a fresh
windowed capture, exact replay of another 10,200 native values and a reinstall/
export check after the fix. This second short run took **0.482349 wall seconds
for 0.2 simulated seconds**; it is a compatibility check, not a speedup claim.
Native rendering remains tested on Mac; cross-platform data tests do not
qualify Windows rendering. Original receipts are retained unchanged.

## Results

| Check | Result |
|---|---|
| Python regression tests, including existing capture/export contracts | 96 test executions passed |
| Godot coordinate and recording-contract assertions | 7 + 29 passed |
| Actual windowed Godot: Begin, ten steps, Finish | Passed; zero client errors |
| Independent native replay | All 10,200 scalar values exactly matched; maximum difference 0 |
| Independently reconstructed task score | Matched; task unsuccessful |
| Fresh installation outside checkout | Both CLI entry points work; installed source hashes match |
| Reviewed direct-action export and reload | 10 rows x 269 causal features, 10 x 7 issued target labels |
| Learner inputs | Existing feature contract; no teacher calls or native object/contact truth |

Fault-injection tests cover interrupted intent/outcome writes, fsync before
in-memory journal cursor updates, and truncated trailing records. Finalization
reconstructs the committed prefix before sealing. A truncated journal remains
unsealed and recoverable. Pause/resume/reset/disconnect were also exercised over
real localhost TCP on an earlier source revision; those retained development
captures are not presented as evidence pinned to the final recorder source.
Their final-source behavior remains covered by the Python regressions above.

## Timing scope

Final windowed capture: **0.2 simulated seconds in 0.406678 wall seconds** after
the initial handshake. The server reports 201.570 ms aggregate command handling,
including controller, camera/perception, file writing and finalization. Twelve
RPC samples include Begin and Finish; their p95 is 53.432 ms. This small sample
does not establish a tail-latency bound, sustained throughput or 50 Hz real-time
operation. Physics, wall-clock delivery and GUI frame rate are different metrics.

Host: Apple M5, 10 logical processors, macOS 26.5.2 arm64, Python 3.12.9, NumPy 2.5.3,
MuJoCo 3.15.0, Godot 4.7.2. The standalone viewer uses the compatibility OpenGL
renderer. No GPU-training speed, RAM minimum or physical reaction latency was
measured. Godot headless assertions passed with sandbox log/certificate-access
warnings; the actual windowed run completed without client errors.

Earlier development windowed receipts measured 0.511076 and 0.595010 seconds
for the same short simulation duration, on earlier recorder/UI revisions. They
remain local alongside the initial graphics-sandbox refusal and a development
capture rejected after the source-identity contract changed. Do not infer a
controlled speedup from these development runs.

## Evidence and reproduction

- `tests.json` pins tested source and reports the regression executions.
- `windowed-client.json` retains the scripted request/state trace and client
  source hashes. Its local output directory is replaced with a portable label;
  other values match the original receipt.
- `native-replay.json` records independently executed native array comparisons.
- `capture-summary.json` retains the independent task score, source identities
  and exact original capture inventory. The synthetic pixel/native capture is
  retained locally; it is not bundled in this small source/evidence publication.
- `fresh-install.json` records installed source hashes and exported array shapes.
- `workstation.png` is the real final window screenshot, not a design mockup.

Follow [the capture guide](../../docs/godot-demonstration-v02.md). To reproduce
the short engineering check, prepare with `--duration 0.2`, serve with
`--source-kind simulation_scripted_input --max-episodes 1`, then run Godot with
`--smoke-steps 10 --smoke-output /absolute/fresh/receipt.json` and an optional
`--screenshot /absolute/fresh/workstation.png`. Run the native replay script on
the resulting episode. Use fresh directories; failed attempts are preserved.

MIT applies to the added DVIDIA source. Upstream arm notices remain under
`third_party/robot-models` and in prepared scene exports. MuJoCo is an external
Apache-2.0 dependency. No credentials, real footage or model weights are included.

Next: collect actual operator demonstrations, review useful recovery segments,
and register a separate learning comparison on unseen starts. The existing
5/64 learned placement result and its unqualified status remain unchanged.

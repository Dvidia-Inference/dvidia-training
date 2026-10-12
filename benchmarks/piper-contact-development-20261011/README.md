# PiPER contact-memory development appendix

This archive retains all 13 scripted exploratory development attempts, including four unsuccessful tasks, before the separately reserved confirmation cohort. These repeated fixtures are development evidence and do not estimate reliability or detector accuracy. Every attempt was numerically valid, used its original fixed 20-second window, and retains all 1,000 commands and 1,001 native states. No attempt was removed or overwritten.

The new controller inherits the exact v0.2 acquisition probe and supported one-retry limit. `probe_only` removes carry monitoring only. `history` uses a causal short TCP-local offset window and may request reduced joint deltas, hold closed, reassess and resume. Native forces, gains, dynamics and the outcome scorer are unchanged. Contacts and object poses are privileged simulator observations; this is not a hardware safety controller or a learned model.

| Candidate | Recorded attempts | Change |
| --- | ---: | --- |
| r0 | 6 | Initial finite parameter design: probe-only/history at three friction values, 40 g |
| r1 | 6 | Correctness checks: maximum pairwise excursion prevents endpoint-canceling oscillation from qualifying as settled; unscaled teacher targets are recorded for command audit. Four history scenes plus current/frozen v0.2 baseline comparison |
| r2 | 1 | Clear completed settling dwell after resume; controller motion is unchanged. Final-source 40 g moderate-friction check |

No controller threshold was tuned between these candidates. Complete exact source snapshots are under `sources/r0`, `sources/r1`, `sources/r2`, and `sources/frozen-v02`. `summary.json` binds every attempt to its own snapshot and includes stored/decoded hashes, events, timing and task outcomes. Original decoded JSON bytes are preserved inside deterministic gzip files.

The nominal 40 g fixture completed in both new modes. At friction 0.55, the 40 g probe-only fixture completed at 15.632 seconds; history completed at 18.421 seconds after one requested slowdown, one closed hold and one resume. The 60 g nominal-friction history fixture also completed after one hold/resume. The 60 g moderate-friction fixture failed acquisition, and low-friction fixtures failed after the inherited bounded probe/retry behavior. These results remain exploratory; they are separate from confirmation denominators.

The retained current baseline and frozen v0.2 baseline have exactly equal issued commands and exactly equal every-row native `qpos` and `qvel` values. Their provenance/recording fields naturally differ with source identity.

Requested slowdown is not proof of immediate physical slowdown. For the 40 g moderate-friction fixture, actual mean TCP speed in the 120 ms before/after the slowdown request increased from 79.8 to 88.9 mm/s. In the first 120 ms after the measured-target hold it was 12.1 mm/s; in the 120 ms before resume it was 3.6 mm/s, and after resume 34.3 mm/s. The 60 g nominal-friction fixture showed the same direction of change. Actual speed/path windows are retained in `summary.json`; this archive makes no instantaneous arrest claim.

The PiPER simulation asset notices are preserved in `licenses/agilex_piper`. DVIDIA source code is MIT; its license is retained in `licenses/DVIDIA-MIT.txt`. No model mesh assets, raw video, weights or credentials are included. Fetching and simulating assets is a separate pinned operation; inspecting these recorded files requires no network or native simulation.

`integrity.json` lists every archive file with byte counts and SHA-256, plus decoded byte counts and SHA-256 for gzip attempts. This seal checks recorded file consistency; it is not cryptographic authentication of physical robot competence.

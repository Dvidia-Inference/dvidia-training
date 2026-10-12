# PiPER bounded second-lift development

These are known-layout development results. Seed 941827 was previously v0.4 confirmation and is now development; it must not be described as fresh v0.5 confirmation. No native mechanics, gains, forces, material parameters, or task success scorer were changed.

The candidate only acts after an existing qualified grip enters a settling hold, remains loaded and unsupported with known stable relative history, but falls below the original 6 mm resume floor. After 100 ms of this low stable state, it performs at most one 8 mm vertical probe at 10 mm/s with its current measured orientation. It keeps the jaw closed. Any unknown history, loaded-pad loss, support return, unexpected contact, or renewed slip aborts closed. It must recover the original >=6 mm height and 100 ms stable dwell, and the selected revision additionally requires >=4 mm of measured second-probe TCP travel. The requested probe times out after 1.2 seconds, checked at 20 ms observation boundaries. The original full task lift, retention, placement, release, and retreat checks still decide success.

| Revision | Native attempts | Successes | Change |
| --- | ---: | ---: | --- |
| r0 | 12 | 5 | Original floor and settling dwell after second probe; rescued one of two low holds. Other case later lost contact. |
| r1 | 2 | 0 | Added 6 mm measured second-probe travel. Both timed out because servo lag kept travel below that bound. |
| r2 | 2 | 2 | Measured second-probe travel 4 mm. Both previously failed heavy low holds completed. |
| r3 | 12 | 6 | Same r2 physics/parameters, explicit runtime integration and causal diagnostic fields; guarded against supported-arrival bypass during a probe. |
| r3 historical regression | 12 | 6 | Seed 813407: every action and native position/velocity trajectory exactly matches the existing v0.4 reference. |

All 40 native attempts are valid and retained, including 21 failures. `analysis.json` records hashes, source verification, native trajectory comparisons and all outcomes. Source snapshots are in each revision's `sources` directory. The initial helper configuration error happened before native simulation and is recorded in `preflight-error.json`.

The selected r3 candidate completed 6/12 placements on the known 941827 layouts, versus v0.4's 4/12, with no lost successes. The two added completions occurred at 17.014 and 16.923 simulated seconds. Each second probe lasted 0.86 seconds before qualification. All ten unaffected scenes preserve exact actions, native positions and native velocities; the original six friction-limited acquisition failures remain failures. Every native model/gain digest comparison matches the original reference; there are no warnings or unexpected contact ticks.

Ten causal unit tests cover invalid configuration, preserved v0.4 acquisition refusal, known history and dwell prerequisites, support/unloaded/jitter rejection, preserved height and added travel gating, fail-closed renewed loss/slip/unknown/contact, supported-arrival bypass prevention, bounded timeout/travel, and unchanged high stable hold behavior. Physical robot transfer and sensor latency remain untested.

`run_development.py` uses the explicit integrated `reacquire` mode for r3. r0-r2 injected the new subclass into `run_episode` without changing the native environment; their raw records disclose this injection. Historical results, failed revisions and all raw samples remain available rather than selecting only successful trials.

Before the fresh freeze, a deadline-ordering fix ensures qualification exactly at 1.20 s remains eligible, while qualification afterward aborts. A regression test reproduced the former late-qualification edge. The recorded selected development probes qualified at 0.86 s; their original source snapshots and raw evidence are preserved.

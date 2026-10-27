# PiPER grip feedback v0.2 public replay checks

These screenshots document independent UI checks of a recorded simulation replay. They do not establish physical robot transfer, live browser physics, camera perception, a learned grip advantage, or universal download reliability.

## Browser-managed resources

The checked research deployment was worker `4da363f3-356d-4006-ac7e-5498650d90da`, research commit `237a1fdf2f74c93ce7a03d6e822550f5386e5694`. Its derived viewer integrity digest was `c303e9da299ce12eba214adcd091d17faa6b396a82b0ca7046021300529c56d0`, with original parent `2731ec8df376fde096ed2ff30540fd4de1fabd698fb1a1be650345d5bdf73d04`.

One intentional reload after deployment loaded the first scheduled feedback replay. Play advanced the frame counter; pause and timeline scrubbing reached frame 1001 at 20 seconds. Switching to the first scheduled jaw-interruption failure loaded a separate saved case without another reload. The matched blind learner and authored teacher also loaded independently. No warning/error entries were observed in that browser session. Timings were not instrumented by this check.

- `research-browser-resources-overview.png` shows the page branding, complete-cohort results and authored teacher comparison.
- `research-browser-resources-failure.png` shows the interrupted failure at its final frame: 128.1 mm terminal error, no restored loaded-contact event.
- `research-original-failure.png` is historical proof from the original viewer transport, before the browser-managed projection. Do not use it to validate the current loader.

The frozen benchmark remains feedback **6/24**, matched blind **8/24**, frozen primitive **5/24**, authored teacher **24/24**, open jaw **0/24**. Feedback interruption success is **0/12**. First-frame loading success does not change any scientific outcome.

That earlier experimental transport projection still used the explicitly unpinned native dataset links from its original viewer. The following final check supersedes its hosting and evidence-link status; scientific outcomes remain unchanged.

## Final pinned publication

The final derived viewer seal is `11b6ceae739efb38902ea17beb05b948cda8b2773b641b269d70fa8b3bbb21ce`. Both hosts retain the exact projection derived from original seal `2731ec8df376fde096ed2ff30540fd4de1fabd698fb1a1be650345d5bdf73d04`; all 120 rendered native evidence links pin dataset revision `52cfbdabb550beee7af4e2af41e3e8c534c59baf`. The entry script has content digest `c861d9974c143be15399a2c497f7e08d055a549529c3dad51a71448b511683a5` in its URL and a matching SHA384 integrity attribute.

Space revision `9eaeeae67def2697ca664b54a754e915f0741c78` loaded the first scheduled feedback case without a second reload. Play advanced from frame 1 to 11; pause and scrubbing reached frame 1001 at 20 seconds. A separate saved jaw-interruption case loaded with 128.1 mm terminal error, no restored contact and a failed outcome. The blind control independently loaded with 5.2 mm terminal error. All 120 visible native links were inspected for the final dataset pin. No browser warning/error entry was observed. These are controlled functional checks, not instrumented loading-time measurements.

Research commit `dc9e5b337a1fdfa6ce400f1e7a37e2daebff019a`, deployed as worker `3aa2246a-7cac-4a1a-ac95-9e1aae7dedd7`, serves the identical sealed projection. Its first fresh load remained in the loading state when checked. One intentional reload rendered the first case; play advanced from frame 1 to 32, pause and scrubbing reached frame 1001, the separate interrupted failure loaded, and all 120 native evidence links carried the final pin. This preserves the remaining first-load limitation rather than claiming a universal transport fix. No network root cause or cold-load latency is established by the screenshots.

The research-host blind control then independently loaded with the expected 5.2 mm error and completed recorded outcome; no warning/error log was observed. The final paper visibly includes publication receipts and the separately scoped TRAINING-only clock-label diagnostic.

The public dataset card visibly shows the MIT license, complete five-variant outcome table, scientific README preservation and unchanged-seal mapping. Public model/dataset/Space byte checks and installation asset receipts are linked from the [research paper](https://research.dvidia.org/papers/learned-piper-grip-v02/).

- `huggingface-mit-dataset-card.png` records the public card and license.
- `huggingface-final-pinned-hero.png` records the final replay branding and complete-cohort counts.
- `huggingface-final-pinned-overview.png` records the blind control and playback interface.
- `huggingface-final-pinned-failure.png` records the interrupted failure at its last frame.
- `research-final-pinned-first-load.png` retains the initial unresolved loading state before the single reload.
- `research-final-pinned-failure.png` records the independently loaded research-host failure.
- `research-final-pinned-overview.png` records the research-host blind control and complete-cohort counts after the single reload.

The browser replays saved simulation states. It does not execute live physics, fit a policy, or demonstrate physical robot sensing. The current learned feedback result is still 6/24 versus blind 8/24.

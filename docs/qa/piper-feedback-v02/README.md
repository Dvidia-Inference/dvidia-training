# PiPER grip feedback v0.2 public replay checks

These screenshots document independent UI checks of a recorded simulation replay. They do not establish physical robot transfer, live browser physics, camera perception, a learned grip advantage, or universal download reliability.

## Browser-managed resources

The checked research deployment was worker `4da363f3-356d-4006-ac7e-5498650d90da`, research commit `237a1fdf2f74c93ce7a03d6e822550f5386e5694`. Its derived viewer integrity digest was `c303e9da299ce12eba214adcd091d17faa6b396a82b0ca7046021300529c56d0`, with original parent `2731ec8df376fde096ed2ff30540fd4de1fabd698fb1a1be650345d5bdf73d04`.

One intentional reload after deployment loaded the first scheduled feedback replay. Play advanced the frame counter; pause and timeline scrubbing reached frame 1001 at 20 seconds. Switching to the first scheduled jaw-interruption failure loaded a separate saved case without another reload. The matched blind learner and authored teacher also loaded independently. No warning/error entries were observed in that browser session. Timings were not instrumented by this check.

- `research-browser-resources-overview.png` shows the page branding, complete-cohort results and authored teacher comparison.
- `research-browser-resources-failure.png` shows the interrupted failure at its final frame: 128.1 mm terminal error, no restored loaded-contact event.
- `research-original-failure.png` is historical proof from the original viewer transport, before the browser-managed projection. Do not use it to validate the current loader.

The frozen benchmark remains feedback **6/24**, matched blind **8/24**, frozen primitive **5/24**, authored teacher **24/24**, open jaw **0/24**. Feedback interruption success is **0/12**. First-frame loading success does not change any scientific outcome.

This experimental transport projection still used the explicitly unpinned native dataset links from its original viewer. Final cross-platform publication must additionally check the dataset-pinned projection and Hugging Face Space. Those checks are recorded separately when complete.

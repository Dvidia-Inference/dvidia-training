# PiPER imitation replay verification

Verified in the Codex in-app browser on 27 October 2026. Screenshots show the local recorded replay, not a live robot or fresh physics execution.

- Native PiPER geometry loads with 12 recorded bodies and 1,001 frames per case.
- Play advances the frame, clock and measured joint/jaw readouts; pause and scrubbing to frame 1,001 show 20.00 seconds.
- Changing to failed learned case 00-goal1 displays “Did not complete” despite only 10.9 mm final error.
- The open-jaw control for case 00-goal0 displays “Did not complete”, 128.5 mm error.
- All twelve learned cases and the first scheduled control replays remain available; all sixty original traces are linked.
- The landing view states learned 5/12 and teacher 12/12, and labels the sidebar value “Episode outcome”.
- The Hugging Face link points to Dvidia. DVIDIA uses a plain text wordmark.
- A real 390 × 844 pixel iframe viewport renders the canvas and wraps the playback controls without observed horizontal overflow. This is a responsive viewport check, not a mobile-device qualification.

The exporter independently reconstructed 16,016 recorded frames using `mj_forward`, with zero maximum object/TCP residual. It did not step physics. Original Float64 evidence remains separate from Float32 browser body poses.

See [desktop](desktop.png) and [390-pixel preview](narrow-390.png). The latter includes the surrounding test harness.

## Public hosting verification

On 27 October 2026, the deployed research viewer loaded native meshes and all 1,001 saved frames. Play advanced from frame 1 to frame 1,001; timeline scrubbing updated the clock and joint readouts. Failed learned case 00-goal1 remained failed at 10.9 mm, while the teacher and open-jaw comparisons showed completed and failed outcomes respectively. All 60 trace links were present. The research transport projection preserves the original model, geometry, motion, native tapes and scores. Its separately recorded frontend adds bounded downloads and exact size/SHA-256 verification.

The Hugging Face static host requires the explicit `viewer/index.html` entry. An initial binary request returned a network error; a reload after anonymous byte verification played through 20 seconds. This motivated a separately recorded fallback update. Final verification of that update is recorded in the public publication provenance. No browser loading latency or network-reliability benchmark is claimed.

See [the public research replay](public-research.png).

Final Hugging Face hosting patch `73390d05bc8f8b82333f4de060b77076ccd603d3` was checked through the corrected namespace landing page: it redirected to the explicit viewer entry, loaded native meshes, played to frame 1,001 / 20.00 seconds, and retained the failed case at 10.9 mm. Its separately derived frontend passed 17 transport tests. All 25 PiPER publication tests passed, and the hosting adapter verified unchanged tree entries plus anonymous readback at the final revisions. See [the hosted failure case](public-huggingface.png).

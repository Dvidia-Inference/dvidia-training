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

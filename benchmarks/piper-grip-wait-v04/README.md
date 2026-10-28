# Bounded PiPER grip waiting v0.4 — preregistered experiment

This study tests an authored acquisition supervisor around the frozen v0.3 path. It is not a newly trained manipulation model. Learned predictions are advisory. All v0.3 results and sources remain frozen.

Supported bilateral contact starts a stability wait. After 100 ms of bilateral force and small tool-relative drift, a one-way latch resumes the original path. A bounded unsuccessful wait or observed support loss produces an explicit abort and capped measured-pose holding for the rest of the 20 s episode. Abort is a failure. Acquisition-only readiness does not certify continuous retention or physical safety.

Three wait budgets (200, 360, 500 ms) are evaluated on four fresh development XY groups (16 cases each). Select by strict successes, then fewer aborts, critical-contact attempts, total wait ticks and shorter budget. Freeze selection before confirmation. Confirmation uses 32 additional unseen XY groups, two known goals and nominal/post-grip conditions: 128 cases each for the selected supervisor, frozen learned clock, fixed timing and authored v0.3 teacher. All 512 cases are retained. The unchanged native strict scorer remains authoritative; aborts additionally fail.

Two primary paired contrasts compare the selected supervisor with the frozen learned clock and fixed timing. Whole XY groups are the independent unit, with Bonferroni simultaneous 95% intervals. No optional stopping or confirmation retuning. Simulation-only privileged state, one rigid 40 g block, a narrow start envelope, no camera or real-arm qualification.

## Run locally from this source revision

```sh
python -m dvidia_training.piper_grip_wait_v04_benchmark prepare --run runs/my-grip-wait-v04 --previous runs/piper-grip-clock-v03-20261027
python -m dvidia_training.piper_grip_wait_v04_benchmark development --run runs/my-grip-wait-v04 --assets /path/to/verified/robot-assets
python -m dvidia_training.piper_grip_wait_v04_benchmark freeze --run runs/my-grip-wait-v04
python -m dvidia_training.piper_grip_wait_v04_benchmark confirmation --run runs/my-grip-wait-v04 --assets /path/to/verified/robot-assets
python -m dvidia_training.piper_grip_wait_v04_benchmark report --run runs/my-grip-wait-v04
```

Inputs are the exact v0.3 native-study policy files. This command does not infer actions from an arbitrary video. Protocol and preparation seals are published before native development attempts. Results will be added after the frozen confirmation run.

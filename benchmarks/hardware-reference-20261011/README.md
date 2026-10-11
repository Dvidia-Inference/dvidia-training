# Hardware reference mechanics — 11 October 2026

Frozen CPU diagnostics from runtime [c95a438](https://github.com/Dvidia-Inference/dvidia-training/commit/c95a4385dae4b654b01ef75f56cbcc144fde2f5c). All **18/18 attempted runs are numerically valid**, with no native warnings. These are deterministic unloaded motion/closure experiments, not eighteen independent skill trials or physical qualification.

The compiled UR5e/Robotiq reference has fourteen scalar joint coordinates, seven actuator commands, three equality objects, one split tendon and 24 collision geoms. All six moving arm links have active collision proxies; the upstream fixed base has none. Both separately audited Shadow references have 24 joint coordinates, 20 commands and four summed-angle tendons. The bimanual assembly is not implemented. No native sensor channels, learned task policy, task object or physical calibration is supplied.

## CPU cost at a 1 ms physics step

| Authored motion | Throughput, two repetitions | 20 ms chunk p95, two repetitions (ms) |
| --- | --- | --- |
| hold_open | 44.41×, 47.29× | 0.574125, 0.562250 |
| arm_sweep | 44.32×, 40.80× | 0.711583, 0.708542 |
| jaw_cycle | 41.79×, 42.83× | 0.685666, 0.682750 |

Hardware: macOS-26.5.2-arm64-arm-64bit, Python 3.12.9, MuJoCo 3.15.0. No GPU used. Throughput is simulated duration divided by native stepping/instrumentation/boundary-forward wall time; it excludes verification/compile, settling, sampling, rendering, perception, policy, communication and physical actuation. It is not total training throughput, physical reaction speed or a measured minimum machine. CPU load/power state may change these timings.

## Numerical sensitivity and contact coverage

Each of three cases runs four seconds with 2, 1 and 0.5 ms native timesteps and two repeats. Issued commands stay fixed at 20 ms boundaries. Adjacent-step comparisons retain all 201 matching boundaries, including post-settling starts. For arm sweep, TCP translation RMSE drops from 0.074238 mm (2→1 ms) to 0.037023 mm (1→0.5 ms). For jaw cycling, pad-center separation RMSE drops from 0.034480 mm to 0.017331 mm. This compares simulation with simulation, not reality, and does not prove solver convergence across contact-rich tasks.

The static sphere-overlap diagnostic records one upper-arm contact pair with ordinary masks and zero with arm masks disabled. It tests geometry participation, not avoidance, stopping or physical contact accuracy. Pad-center separation is a geometric proxy, not inner jaw gap or calibrated force. Repeated samples/cases are correlated.

## Artifacts and reproduction

- `report.json`: settings, source hashes, every attempted run, full native trajectories, inventory, collision diagnostic and numerical comparisons.
- `report.html`: offline schematic replay and errors/timing; no remote dependencies or physical commands.
- `integrity.json`: exact hashes/sizes for the two report artifacts; internal consistency only.
- `source.json`: supplemental post-run source attribution, outside the artifact seal.
- `LICENSE`: MIT for DVIDIA's original software/report. Third-party assets are not included in this bundle; their BSD-3-Clause/BSD-2-Clause/Apache-2.0 source paths and digests are retained in the reference lock.

```sh
.venv/bin/dvidia-embodiment inspect benchmarks/hardware-reference-20261011
.venv/bin/dvidia-embodiment fetch-assets --output runs/reference-assets --hands
.venv/bin/dvidia-embodiment benchmark runs/reference-assets --hands --output runs/new-reference-bench
```

[Install and calibration protocol](../../docs/hardware-reference.md). Physical mismatch can be scored from complete aligned physical/simulated traces; the included analytic paired fixture is synthetic and is not this arm's hardware data. Images alone do not supply calibrated geometry, actuator response, grip/friction, actual sensing or a trained action policy. LeRobot can be the future collection/device layer while DVIDIA builds its own model; it is not integrated into this version.

# Independent camera and actuator corroboration v0.4

This release adds a CPU RGB observer and PiPER drive-channel checks to the v0.3
physical-goal verifier. The checks can withhold a `complete` verdict when the
visible object position/motion conflicts with the primary packet, or when the
gripper drive does not corroborate release. They do not authenticate sensors,
identify arbitrary objects, reconstruct depth or control a robot.

## Fresh paired result

The frozen run contains 272 cases across eight layouts: 96 native states, 80
primary-packet forgeries, 64 auxiliary-channel faults and 32 cross-channel
controls. Native generation completed with **0/96 invalid cases**. The v0.3
baseline and v0.4 gate received the exact same primary packets.

| Measure | v0.3 primary only | v0.4 camera + drive |
| --- | ---: | ---: |
| Scheduled / evaluated | 272 / 272 | 272 / 272 |
| Complete verdicts | 192 | 50 |
| Physically true completions recognized | 80/80 | 7/80 |
| Physically false completions | 112/192 (58.3%) | 43/192 (22.4%) |
| Baseline false completions prevented | — | 69/112 (61.6%) |
| Baseline true completions lost to abstention | — | 73/80 (91.3%) |
| Unknown verdicts | 0/272 | 142/272 (52.2%) |

The true-completion denominator includes 64 repeated sensor-fault controls whose
physical scenes are complete but whose auxiliary evidence is deliberately
missing, stale, gapped or malformed. All 64 correctly become `unknown`. In the
96-case native cohort, seven of eight visible settled scenes are recognized;
all eight occluded scenes abstain, and one nominally visible target is obscured
by the simulated arm. The gate is conservative and has low positive coverage.

| Cohort | Cases | Primary complete | Fused complete | Fused unknown | Physical false completions after fusion |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native scenes | 96 | 16 | 7 | 9 | 0/80 negatives |
| Primary forgeries | 80 | 80 | 29 | 51 | 29/80 |
| Sensor faults | 64 | 64 | 0 | 64 | Not applicable: all are physical positives |
| Cross-channel controls | 32 | 32 | 14 | 18 | 14/32 |

The 43 residual physical false completions are retained by cause:

| Control | Residual false completions |
| --- | ---: |
| Final-instant motion alias | 7/8 |
| 180° yaw reversal | 8/8 |
| Same-color visual lookalike | 7/8 |
| Elevated support hidden from the top camera | 7/8 |
| Refreshed camera replay | 7/8 |
| Coherent replay of all channels | 7/8 |

The four controls with seven residuals have one layout where the camera could
not observe the green proxy and therefore abstained. Axial geometry cannot
distinguish a half-turn. A synthetic green proxy is not authenticated object
identity. An orthographic top view does not see elevation or table support.
Sampled RGB and drive data miss motion between samples. All channels share the
same simulator and can be replayed together.

## Processing time

| Stage | Median | p95 | Maximum |
| --- | ---: | ---: | ---: |
| Primary verifier | 0.42 ms | 0.58 ms | 1.15 ms |
| RGB camera observation | 14.32 ms | 20.10 ms | 27.85 ms |
| Actuator observation | 0.86 ms | 1.21 ms | 10.33 ms |
| Fusion | 0.005 ms | 0.018 ms | 0.047 ms |
| Entire processed window | 22.09 ms | 31.48 ms | 44.77 ms |

The timer covers local JSON reads, PNG decoding, the v0.3 baseline, camera
processing, actuator checks and fusion. It excludes the 200 ms observation
history, native rendering/acquisition, networking and robot actuation. The
optional OpenCV CPU kernel was installed; a NumPy-only fallback is available
but was not used for this timing run. These are not hard-real-time or
end-to-end robot response measurements. No GPU or hosted model call was used.

## Run it again

Use the source commit frozen in [`evidence/protocol.json`](evidence/protocol.json).
The reported run used `90e84701ce7d5bc330ef41b02793e2ce037d702d`.

```sh
python -m pip install -e '.[arm,observations]'
dvidia-sensor-goal generate \
  --assets /absolute/path/to/robot-profile-upstream/assets \
  --output benchmarks/sensor-goal-verifier-v04/evidence \
  --seed 2026102007 --count 8 --repeats 3 --warmups 1
dvidia-sensor-goal evaluate --run benchmarks/sensor-goal-verifier-v04/evidence
python -m dvidia_training.sensor_goal_verifier_report \
  --run benchmarks/sensor-goal-verifier-v04/evidence --output /tmp/sensor-goal-v04-report
```

Generation needs the pinned PiPER MuJoCo assets and an available graphics
context. Evaluation is local CPU work. `integrity.json` covers every native
input; the result stores one paired decision per case and all three timing
repetitions. See [`protocol.json`](evidence/protocol.json),
[`cases.json`](evidence/cases.json), [`benchmark.json`](evidence/benchmark.json), and
the complete [`native evidence directory`](evidence/).

## Scope

The RGB path receives eleven 640×480 images at 20 ms intervals and estimates
XY centroid and an undirected object axis from one green elongated proxy. It
has no depth, segmentation IDs or target identity. The drive path checks one
hundred 2 ms samples of six PiPER arm joints, signed finger positions and
velocities, gripper command and simulated motor effort. It cross-checks
servo/encoder consistency, release aperture and settling; motor effort is not
object contact force. Position bounds, timestamp calibration IDs and drive
calibration are synthetic declarations, not physical calibrations.

The result establishes neither independent physical sensors nor hardware
qualification, perception quality on ordinary objects, real-time controller
latency, action learning, training-data transfer, or robot safety. Unknown
means insufficient or contradictory corroboration, not physical failure.
Original DVIDIA code is MIT. Pinned simulator code and robot assets retain the
upstream licenses recorded in the repository.

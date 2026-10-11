# Hardware reference simulation v0.1

The first reference assembles the documented **UR5e six-joint arm and Robotiq
2F-85 gripper** in native MuJoCo. It preserves upstream geometry, inertias,
collision proxies and the gripper's coupled linkage. It is a replaceable software
reference while the actual arm is being selected, not a calibrated physical replica.

The existing `dvidia-authored-6dof-parallel-jaw-v0` box environment and its frozen
benchmarks stay separate. Its simplified links and disabled arm collision masks
cannot establish whole-arm contact behavior. Its learned transition model is not
an action policy for this new reference. Installing these tools does not install a
hardware controller or an acquired shoe skill.

## Install, provision once, then run offline

Use the candidate `feat/hardware-reference-v0.1` source branch until this feature
is released. Python 3.12+, NumPy 2.5.3 and the optional MuJoCo 3.15.0 runtime are
required for simulation. FFmpeg is unnecessary for this mechanics-only bench.
Use `.venv\Scripts\COMMAND.exe` in Windows PowerShell.

```sh
git clone --branch feat/hardware-reference-v0.1 https://github.com/Dvidia-Inference/dvidia-training.git
cd dvidia-training
python3.12 -m venv .venv
.venv/bin/python -m pip install '.[arm]'
.venv/bin/dvidia-embodiment fetch-assets --output runs/reference-assets --hands
```

Provisioning explicitly downloads a fixed public allowlist from MuJoCo Menagerie
commit `0059d4335f8156206f63a35662313385f7ad6d74`. The bundle is 34,193,501 bytes
for the arm/gripper or 37,746,863 bytes including the left/right Shadow hand
references. Each file's exact size and SHA256 are checked; upstream licenses and
derivation notes are retained. No account, API key or model weights are needed.
A failed download leaves a partial directory; verification rejects it. Retry
into a fresh directory. Fetching refuses to overwrite existing output.

After provisioning, these commands use no network:

```sh
.venv/bin/dvidia-embodiment verify-assets runs/reference-assets --hands
.venv/bin/dvidia-embodiment audit runs/reference-assets --hands --output runs/reference-audit.json
.venv/bin/dvidia-embodiment benchmark runs/reference-assets --hands --output runs/reference-bench
.venv/bin/dvidia-embodiment inspect runs/reference-bench
```

Open `runs/reference-bench/report.html` for a local schematic replay and timing /
numerical comparisons. Its poses come from native physics; it draws link and pad
centers, not the full mesh or contact surfaces. Optional native mesh viewing is
available with `dvidia-embodiment view runs/reference-assets`. On macOS launch
the module using `.venv/bin/mjpython -m dvidia_training.embodiment_cli view
runs/reference-assets`; MuJoCo's graphical viewer needs its supported desktop
graphics context. Headless benchmarking requires no GPU or graphical viewer.

`inspect` checks bounded regular files, hashes and attempt-count consistency.
That establishes artifact consistency, not authentic measurements or recomputed
metrics. The inspector, asset verifier and paired trace scorer do not import the
native simulator. Command help works without NumPy or MuJoCo.

## The mechanical and sensing contract

| Component | Modeled coordinates | Independent actuator commands | Important distinction |
| --- | ---: | ---: | --- |
| UR5e | 6 hinges | 6 | Position targets in radians; upstream affine position servos are not identified UR firmware. |
| Robotiq 2F-85 | 8 linkage hinges | 1 | One fixed split tendon and three equality objects; 0–255 input, not meters or measured fingertip force. |
| Combined reference | 14 scalar coordinates | 7 | No free object or learned task policy in this bench. |
| Shadow hand, each side | 24 joints | 20 | Four summed-angle tendons; fingers and joint/motor counts remain separate. |

Attachment uses the upstream arm `attachment_site` and a `gripper/` namespace.
The `gripper/pinch` site is the reference tool point; mounting adapter mass and
serial-specific/tool calibration are not supplied. Parent and gripper engine
options are reconciled explicitly: implicitfast integration, elliptic contact
cone and impratio 10. The default no-slip iteration count remains zero.
Geometry, include files and meshes are compiled from the exact verified bytes
in memory rather than reread from mutable model paths. Regular-file identity
checks are best effort on platforms without POSIX no-follow semantics.

Every moving arm link has an active collision proxy. The upstream fixed base
has none; self-contact also depends on upstream exclusions and the engine's
filtering. This is not complete geometric collision coverage. Both source models
and the hand references expose zero native sensor definitions/data channels.
Recorded positions, velocities, tool poses and contact counts are privileged
simulator state, not deployable camera, encoder or tactile observations.

The reported pad-center separation is the Euclidean distance between the two
upstream `pad1` geom centers. It is **not inner jaw gap** and cannot be silently
used as physical gripper calibration. The jaw's native tendon force cap is not
its real force setting. Firmware object detection, backlash, encoder-to-gap
mapping, wear, delays and material compliance remain unmeasured.

## What the benchmark measures

Three synthetic, unloaded cases hold the open gripper, sweep the base joint and
cycle jaw closure. Commands are held at declared 20 ms boundaries. Each case
runs with 2, 1 and 0.5 ms physics steps, with two deterministic repetitions by
default. Every attempted run is retained, including failures and warnings.

The report separates asset verification/compilation, per-native-tick p50/p95/max
and 20 ms chunk p50/p95/max. Chunk timing includes native stepping,
instrumentation and boundary `mj_forward`, while excluding settling, sampling,
rendering, perception, policy computation, transport and physical actuation.
Physics throughput is simulated duration divided by that chunk wall time; it is
not end-to-end training throughput or a robot's response rate.

Adjacent timestep comparisons use matching elapsed boundaries, including the
post-settling initial state. They report joint position, tool translation and
pad-center differences. These numerical comparisons do not measure hardware
accuracy. A matched static sphere overlap produces an arm contact, while an
otherwise matching scene with arm collision masks disabled removes it. This is
an overlap diagnostic, not path planning or a validated collision response.

Repetitions share commands and geometry. They are repeated numerical/performance
observations, not independent task or physical trials. Object acquisition,
contact/slip fidelity, shoe organization, sensor-to-stop response and the bimanual
assembly are not evaluated. CPU/GPU batching and minimum hardware requirements
need separate measured workloads.

## Measure physical mismatch with paired traces

The pure-Python scorer accepts `dvidia.paired-fidelity/v1`: an ordered set of
named joint positions, tool position, normalized wxyz tool quaternion and an
explicit inner gripper gap for simulator and physical observations at strictly
increasing shared elapsed times in the robot-base frame. It retains protocol,
hardware/source-group identifiers and model/command-replay hash declarations.
The adapter creating this file must review clock/frame alignment and supply the
same command history. The scorer does not infer or authenticate those claims.

```sh
.venv/bin/dvidia-embodiment score-paired tests/fixtures/paired_fidelity.json --output runs/paired-scoring-example.json
```

The supplied two-joint example is an analytic **synthetic fixture**, not physical
evidence or this reference arm's calibration. Real traces require a compatible
hardware adapter, independent pose/gap references and a defined physical gap
measurement. Do not rename pad-center distance to inner gap.

Metrics include per-joint and pooled RMSE/MAE/worst error, tool translation and
sign-invariant rotation error, jaw error and sample intervals. Missing fields,
stale/repeated times, booleans/nonfinite numbers, invalid quaternions, duplicate
JSON keys and files over 4 MiB are rejected. There is no interpolation, hidden
clock shift, angle wrapping or automatic passing threshold. Many rows from one
session remain correlated. Reports always retain `physical_robot_ready: false`.

## First hardware evidence, then hands

1. Identify the actual arm, gripper, mounting, serial calibration, controller
   settings and exposed sensor/command clocks. Measure base and tool transforms.
2. Fit unloaded and loaded response on development replays. Freeze the candidate
   before scoring separate confirmation poses/trajectories and gripper cycles.
3. Measure contact onset, pad force, slip, hysteresis and release with reference
   instrumentation and declared objects/materials. Add a shoe-specific scene.
4. Collect complete shoe attempts with synchronized commands and observations.
   Train a small action policy, retain failed attempts and measure task success
   under held-out conditions before supervised physical qualification.
5. Attach the chosen left/right hands and arms with explicit transmissions,
   mounting/payloads, shared frames/clocks and tactile adapters. Qualify cross-arm
   collision, grasp retention and coordinated handoffs separately.

The later two-arm system is not assembled by this release. Reference hand assets
are separately audited to expose the future interface, not to imply that an
existing six-joint/one-gap model transfers to them. A hand's summed distal angles
do not reproduce every physical transmission constraint.

## LeRobot collection infrastructure, DVIDIA policy

LeRobot is a useful open-source collection/device layer. Its
[custom policy interface](https://huggingface.co/docs/lerobot/main/bring_your_own_policies)
allows an independently published DVIDIA model; using its tools does not require
adopting a pretrained policy. The new reference bench does not yet integrate
LeRobot datasets or command physical LeRobot devices.

Keep that integration explicit: validate a pinned dataset revision, ordered
observation/action fields and units, camera calibration, actual acquisition /
arrival / issued-and-applied-command clocks, controller settings, source groups
and complete outcomes. The [SO-101](https://huggingface.co/docs/lerobot/main/so101)
has five arm axes plus the gripper, not six arm axes plus a gripper. Its degree /
normalized body-position modes and normalized jaw command cannot accept our
existing six-joint-radian/metric-gap artifact without a distinct adapter/profile.

The [recording guide](https://huggingface.co/docs/lerobot/main/il_robots#cadence-reporting)
also distinguishes nominal recording cadence from actual loop performance.
Nominal frame-index/FPS timestamps are insufficient evidence for actuator-delay
or physical-speed fitting when cadence is missed. Preserve actual clocks and
missing/stale observations in an evidence sidecar instead of inventing samples.

The next DVIDIA learning component is a history-aware, robot-specific action
policy with measured reach/grasp/transfer/release outcomes. The current compact
actuator model predicts movement; it does not choose a shoe task's actions.

## Sources and licensing

DVIDIA's new simulator/scorer code and analytic fixtures are MIT. Asset bytes are
downloaded separately, under the retained [UR5e BSD-3-Clause](https://github.com/google-deepmind/mujoco_menagerie/blob/0059d4335f8156206f63a35662313385f7ad6d74/universal_robots_ur5e/LICENSE),
[Robotiq BSD-2-Clause](https://github.com/google-deepmind/mujoco_menagerie/blob/0059d4335f8156206f63a35662313385f7ad6d74/robotiq_2f85/LICENSE)
and [Shadow Apache-2.0](https://github.com/google-deepmind/mujoco_menagerie/blob/0059d4335f8156206f63a35662313385f7ad6d74/shadow_hand/LICENSE)
licenses. Those licenses are not replaced by DVIDIA's brand or software license.
The package carries their source paths, digests and license identifiers, not the
third-party meshes. See [UR serial calibration](https://docs.universal-robots.com/Universal_Robots_ROS_Documentation/kilted/doc/ur_robot_driver/ur_calibration/doc/index.html)
and [MuJoCo attachment](https://mujoco.readthedocs.io/en/stable/python.html#attachment).

# PiPER encoder kinematics v0.1

`PiperEncoderKinematics` converts six measured joint angles into a tool pose and
one authored differential-IK joint-target update. It owns a private robot-only
model and accepts no live task environment, object state or contact observations.
This is a geometry component for future learning integration, not a learned skill.
It is not connected to the learner, installer or physical robot control.

## Local use

Use Python 3.12 or later with the package's `arm` extra (NumPy 2.5.3 and MuJoCo
3.15.0). From a checkout containing this module, install with
`python -m pip install '.[arm]'`. Supply the existing pinned PiPER assets; if you
need to download them, `dvidia-robot-sim fetch piper --output ./robot-assets`
fetches the locked upstream allowlist into a fresh directory. The kinematics
constructor itself is offline and verifies those bytes.

```python
from dvidia_training.piper_encoder_kinematics import PiperEncoderKinematics

robot = PiperEncoderKinematics("./robot-assets")
measured_q = [0.0, 1.57, -1.3485, 0.0, 0.0, 0.0]  # example encoders, radians
pose = robot.fk(measured_q)
desired_position = pose["tcp_position_world_m"].copy()
desired_position[0] += 0.005
target = robot.pose_action(
    measured_q,
    desired_position,
    pose["tcp_quaternion_world_wxyz"],
    0.035,
)
print(target)  # six joint targets and one single-finger target; nothing executed
```

The last argument is **one finger's slide target in metres**, from 0 closed to
0.035 open. It is not the full jaw gap. `jacobian(measured_q)` returns a detached
6×6 spatial Jacobian in world coordinates: three linear rows followed by three
angular rows. No renderer or graphics context is needed.

## Points, frames and limits

The authored TCP is link6-local `[0, 0, 0.12003]` metres with identity local
orientation. `fk` names its world-frame position and wxyz quaternion explicitly.
It also returns `link6_position_base_m` and `link6_quaternion_base_wxyz`: the
link6 **origin** in base coordinates. These are different points. The existing
camera-action encoder feature describes that link6 origin; this module does not
change it. For the pinned fixed base (identity world transform), TCP position is
`link6_position_base_m + R(link6_quaternion_base_wxyz) @ [0, 0, 0.12003]`.
Changing a real mounting transform or fingertip calibration requires a separate
contract. Neither pose is a hardware-calibrated fingertip measurement.

Measured angles must be finite six-vectors within the native joint ranges,
including their endpoints. Even a small measured overshoot is refused rather
than clamped. This is separate from the **issued-target** interior margin of
0.02 rad. The final joint-target change stays within 0.06 rad of accepted
measurements, including native-limit endpoints. Quaternions must already be
normalized wxyz; malformed shapes, booleans, nonfinite values and invalid jaw
targets are refused.

The update matches the existing `PiperTaskEnv.pose_action`: translation error
limited to 0.015 m, rotation error to 0.05 rad, angular weight 0.25, damping
0.012, and a maximum 0.06 rad joint update before the native-limit margin.
Measured joint angles are **not previous applied servo references**. The existing
rate limiter must still run downstream; this API does not execute it.

This local geometric update does not check reachability, collision avoidance,
contact, grip, joint velocity, actuator strength or task completion. Targets are
not authorization to command hardware. One instance per worker is required
because private scratch data is reused.

## Static verification

The focused test suite passed 10 tests using the existing local MuJoCo 3.15.0
environment. It compiled static robot/full-task models, without settling,
time integration, rendering, training or task rollouts. Python dynamics entry
points were patched to fail if called.

- Independent XML transform-chain FK and the 0.12003 m tool offset matched at
  17 fixed/seeded configurations, including native joint-limit endpoints.
- World-frame linear/angular Jacobians matched central finite differences at
  15 interior configurations, across all six joints.
- 34 static pose-target comparisons matched the unchanged original authored
  controller, using full-task snapshots with varied object/finger state only
  as an independent test oracle. No such state enters the new API.
- Quaternion sign equivalence, zero-error targets, bounds/refusal behavior,
  copied results and A→B→A statelessness passed.

Run these checks with existing pinned assets:

```sh
DVIDIA_ROBOT_PROFILE_ASSETS=./robot-assets PYTHONPATH=src \
  python -m unittest tests.test_piper_encoder_kinematics -v
```

Without the asset variable, eight static geometry tests are explicitly skipped;
only the two interface tests run. Passing these checks establishes geometric
agreement, not improved placement success or physical qualification. The asset
lock is `4b2ee2ca675aceb2f826983cb3082a3e08e9aa0ff419b23078abcff08ebcc6fe`;
the pinned PiPER simulation assets retain their MIT license.

## Static cost

A bounded [CPU timing probe](../benchmarks/piper-encoder-kinematics-v01/README.md)
measured 0.0305 ms median and 0.0324 ms p95 for one joint-target calculation
on the recorded shared macOS host. Initialization, including asset verification
and compilation, took 221.5 ms. The receipts, earlier prototype measurement and
portable probe are retained. No dynamics, learned inference, rendering or full
control loop is included, so these numbers establish no real-time guarantee.

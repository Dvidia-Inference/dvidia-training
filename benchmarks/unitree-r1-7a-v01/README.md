# Measuring the Unitree R1-7A arm frames

The standalone R1-7A has seven revolute joints. A portable skill needs to know how those joint angles place the end of its arm in space. We tested that mapping for both Unitree descriptions using every moving link, including the wrist joint that rotates the terminal frame without translating its origin.

[Open the live arm viewer](https://dvidia-robot-simulation-lab.static.hf.space/arms/unitree-r1-7a/v01/index.html) · [Left-arm results](viewer/benchmark-left.json) · [Right-arm results](viewer/benchmark-right.json)

## What we tested

From the seven joint angles, a NumPy implementation composes the URDF joint origins and local rotations. Separately, MuJoCo 3.15.0 imports the same pinned URDF after its visual and collision elements are removed. We set the seven joint coordinates, call `mj_forward` and compare every moving link's position and orientation. Both methods use the stationary URDF `base_link` as their reference. Each pose receipt includes a 4×4 `base_from_link7` matrix, with translation in metres and rotation in its upper-left 3×3 block. Coordinates in the bare `Link7` frame transform into base coordinates by left multiplication.

| Description | Poses | Moving-link transforms | Maximum position disagreement | Maximum orientation disagreement |
|---|---:|---:|---:|---:|
| Left V1.0-1027 | 20 | 140 | 1.42 × 10⁻¹⁶ m | 6.38 × 10⁻¹⁶ rad |
| Right V1.2-1023 | 20 | 140 | 1.42 × 10⁻¹⁶ m | 3.63 × 10⁻¹⁶ rad |

The 20 configurations include zero, the middle of the joint ranges, positive and negative deviations from the middle for each joint separately, both sets of range endpoints and two mixed poses. Each of the 280 link transforms satisfied both tolerances of 10⁻⁹ m and 10⁻⁹ rad. These errors measure agreement between two calculations using a common source. Hardware accuracy needs an independently measured reference. Contact, placement success, dynamics and motion latency require separate experiments.

## Run locally

From this source checkout, install the arm extra into a Python 3.12 environment. The URDFs are included in the package. Once Python, NumPy and MuJoCo are installed, the checks run offline on a CPU with no display or robot connection.

```sh
python -m pip install '.[arm]'
dvidia-arm list --include-references
dvidia-arm describe unitree-r1-7a --side right
dvidia-arm pose unitree-r1-7a --side right --joints 0 0 0 0 0 0 0
dvidia-arm benchmark-kinematics unitree-r1-7a --side left
dvidia-arm benchmark-kinematics unitree-r1-7a --side right
```

Select the actual arm side explicitly. Joint inputs follow J1 through J7 in radians. Among other differences, the vendor gives J6 a range of ±0.524 rad on the left and ±0.785 rad on the right. Both the range and frame axes matter when interpreting measured encoders. Each side’s canonical URDF, revision, archive path, hashes and BSD-3-Clause notice are bound to its output. [Official Unitree archive](https://github.com/unitreerobotics/r1-7a_sdk/blob/92f32514c6a9089c580ddf331eeb26075fcc5609/urdf/URDF_R1-7A.tar.gz) · [Vendor license](https://github.com/unitreerobotics/r1-7a_sdk/blob/92f32514c6a9089c580ddf331eeb26075fcc5609/LICENSE)

## From arm frames to a placement skill

The first physical comparison needs an independently surveyed base, known joint zero offsets and the mounted tool's transform relative to `Link7`. A gripper profile also needs jaw position, opening limits, pad geometry and measurement units. Synchronized camera frames, requested motor targets and measured joint states can then supply a robot-specific training set. Unitree's teleoperation project supports those recordings in MCAP. A fresh first-attempt benchmark on this arm would establish the performance of a fitted placement skill. [Unitree R1-7A teleoperation guide](https://github.com/unitreerobotics/R1-7A/blob/e2faf3ec17f589db2139c51b020592a4a05f836b/README_en.md)

# Surveying the PiPER camera before a placement

A simulated workspace can be surveyed once, then reused for different block arrangements. The PiPER placement study freezes one 3.1 kB camera survey before the first placement. A first task image still has to establish the block position for each arrangement.

Four surveyed coloured square tops determine the rigid transform from a fixed RGB-D camera to the robot base. The registration uses the measured colour interiors, metric optical-Z depth, known pinhole intrinsics and marker locations in metres. Two other marker positions check the result independently of the fit. Every fit and check must fall within 4 mm. The camera survey uses a separate scene with no task object, goal or robot trajectory.

| Fixed camera pose | Refit errors at two held-out positions, mm | Errors reusing nominal pose, mm | Acquisition and fit, ms |
|---|---:|---:|---:|
| Nominal | 0.23, 0.26 | — | 313.8 |
| Relocated | 0.47, 0.14 | 40.0, 34.1 | 15.9 |
| Left | 0.24, 0.32 | 84.0, 54.7 | 14.8 |
| Right | 0.26, 0.14 | 74.4, 57.0 | 14.2 |
| Lower | 0.47, 0.73 | 23.8, 31.9 | 13.6 |
| Higher | 0.27, 0.65 | 15.1, 11.7 | 13.5 |

Across the five shifted views, the median error at the two check locations was 0.30 mm after surveying and 37.06 mm with the stale transform. These ten observations repeat the same two locations. The table measures registration under clean synthetic conditions. Placement accuracy, object recognition, physical depth noise, fixture metrology and camera drift after surveying need separate measurements. A camera move requires another survey. The calculation assumes the markers are registered to the robot base.

**Timing.** On this arm64 Mac, the first process spent 313.8 ms in acquisition and fit, including 5.3 ms for marker detection and fitting. The five shifted surveys took 13.5–15.9 ms with the native imports warm. Each uses one simulated 320 × 240 RGB-D pair. Timings cover fixture scene creation, native rendering, readback and CPU fitting. They exclude interpreter startup, writing the evidence files and physical setup. MuJoCo emitted a depth-precision warning on the Mac renderer. The residuals reported above include that renderer's depth output.

**Development failure.** The first relocated view refused the green marker. An axis-aligned bounding box had a 77.3 mm diagonal while the farthest pair of observed points was 55.4 mm apart. The revised detector uses the actual diameter for borderline projections and keeps the original 75 mm support and 4 mm residual thresholds. Four additional camera poses tested this change. Reprocessing the nominal view reproduced its original pixels, depth and transform.

## Run locally

From the source checkout with the native simulation dependencies installed, choose a new output directory.

```sh
PYTHONPATH=src python scripts/survey-piper-workspace.py --output runs/workspace-survey
```

A renderer is required to generate the scene. The numerical audit can run on CPU with NumPy and no graphics context.

```sh
PYTHONPATH=src python -m unittest -v tests.test_piper_workspace_calibration
```

Each camera has a JSON receipt and a compressed RGB-D frame. The receipt stores separate digests for the decoded RGB and optical depth arrays, the fit and withheld marker lists, the measured optical points, the rigid transform and timings. The nominal receipt used by the placement study has SHA-256 `1a092b3145d754bbf2e4b584fc90822d878b66330f199547c78425f93ebf19f2`. Digest binding identifies this particular capture. A newly rendered receipt has its own timings and digest.

[Camera comparison](comparison.json) · [Nominal survey](nominal.json) · [Nominal RGB-D](nominal-frame.npz)

## How a moved fixture can silently spoil registration

We applied a 12 mm mounting offset in three fresh synthetic scenes with the camera fixed. Moving only the four fit markers left their fit errors below 0.19 mm. The two separately positioned checks missed by 11.97 and 12.21 mm and refused the survey. Moving only the two checks likewise refused it.

When all six markers moved together, their internal check errors fell below 0.19 mm and the survey passed. Applying that accepted transform to a separately captured base reference produced errors of 11.97 and 12.21 mm. This identifies a practical setup requirement for a physical trial. Anchor at least one verification reference independently to the robot base, and survey its position independently of the movable calibration fixture. This diagnostic used simulated mounting offsets and the same two check locations captured in a separate stationary reference scene.

```sh
PYTHONPATH=src python scripts/survey-piper-workspace.py --mounting-diagnostics --output runs/workspace-mounts
```

[Mounting test plan](mounting-plan.json) · [Mounting results](mounting/comparison.json)

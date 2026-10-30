# Native PiPER action capture

This producer connects the [action-demonstration contract](action-demonstrations.md)
to the existing native MuJoCo PiPER environment. It records simulator encoder
measurements, the authored controller's decisions, issued targets, ramped
references, actual final native controls, and actual rendered RGB-D frames.
It collects **no operator corrections or learned policy** and sends no physical
robot commands. It is a capture milestone, not a manipulation benchmark.

The native capture preview is packaged as **0.1.3.dev1**. Install its downloaded
wheel with the optional native dependencies:

```sh
python -m pip install './dvidia_training-0.1.3.dev1-py3-none-any.whl[arm]'
dvidia-piper-capture --help
```

Python 3.12 or later and a working local graphics context are required. Physics
runs on the CPU; rendering has a separate OpenGL driver requirement. The tested
host is macOS arm64 with CGL. Minimum RAM, dedicated GPU requirements, other
operating systems and headless rendering remain unqualified. The installed
smoke uses already verified local robot assets; it does not test a fresh asset
download. After dependencies and assets are available, capture uses no network
or hosted inference service.

Install the optional `arm` dependencies and obtain the pinned PiPER assets
separately. Capture reads and verifies existing local assets; it does not fetch
anything or require an API key. The native renderer needs an available OpenGL
context. Use a fresh destination for every attempt:

```sh
python -m dvidia_training.piper_action_capture run --assets /path/to/robot-assets --output runs/native-capture --duration 0.2
python -m dvidia_training.piper_action_capture inspect runs/native-capture
```

The default is a 0.2-second authored-task excerpt at 160×120. `--mode hold`
retains the starting ramped reference. `--duration` permits 0.1–20 seconds in
complete 20 ms intervals. A conservative frame budget refuses combinations
that could exceed the contract's 512 MiB byte-readback allowance. Neither
duration nor resolution is silently changed. This is an offline simulator
workflow; render and serialization costs may exceed a real-time deadline.

## What is recorded

`recording.json` retains contiguous before → decision/issued → applied → after
transitions. Measured jaw position **and velocity** come from native signed
joint7/joint8 coordinates. The single actuator target remains a finger-slide
coordinate in metres, alongside six radian arm references. The TCP estimate is
encoder-derived **link6** pose in the fixed `base_link` frame; it is not the
task controller's offset finger-centre site or a calibrated physical tool.

The unchanged task environment slews references every 1 ms, copies them to
the seven native position controls, steps physics, and scores native contact.
The producer records the final ramped references and `data.ctrl` separately.
Upstream gravity compensation is a body dynamics setting, not an inferred
force measurement. Endpoint records do not certify intermediate contact peaks.

`frames/rgb-NNNN.png` contains actual lossless uint8 RGB, top row first.
`frames/depth-NNNN.npy` contains little-endian float32 arrays in metres along
the optical forward axis, loaded without pickle. Depth is linear camera-Z,
not ray distance or the raw graphics z-buffer. Far-plane background remains
present; no validity mask, calibrated error bound or object segmentation is
claimed. The explicit classic renderer supplies both arrays from the same
paused state. Exposures are instantaneous synthetic exposures at that shared
simulation timestamp. Wall rendering delay is measured separately.
Renderer capability warnings are retained in
`capture.json` → `native_status.renderer_warnings`, separately from physics
solver warnings. On macOS CGL, unavailable `ARB_clip_control` can limit depth
accuracy. Finite metric depth remains a renderer output, not a verified depth
accuracy result; `depth_accuracy_verified` remains false.

Calibration derives from the actual centred perspective frustum and camera
basis, transformed into robot-base coordinates. Optical axes are right/down/
forward. Pixel centres use `(column + .5, row + .5)` and principal point
`(width/2, height/2)`. The source/configuration, compiled model, profile lock,
scene, limits, calibration and reference policy are bound by hashes. These
establish consistency, not authentic capture history or hardware accuracy.

## What remains deliberately missing

The authored policy uses privileged native object/contact state. Its inputs
and native outcome diagnostics are stored separately under `decisions/` and
`native/`, explicitly outside primary sensor-learning inputs. The four
RGB-D-tracker object pose/uncertainty channels remain **missing/null** because
we have not implemented that tracker. Native object truth is never relabeled
as a visual estimate. Strict learning-target export therefore refuses these
captures. Recording images does not create a pose estimator or corrective
demonstrations.

## Failures and inspection

Each decision journal is saved before the native action. Only a fully
completed, valid interval with its after-observation enters the tape. A
camera, solver, validation or interruption failure seals an explicit refusal
or an interrupted prefix with an **unknown** task result. Native time beyond
the last committed observation remains visible in the receipt. It is never
converted into an invented transition or retried automatically.

```sh
python -m dvidia_training.piper_action_capture run --assets /path/to/robot-assets --output runs/interrupted-capture --duration 0.2 --interrupt-after 2
```

That engineering negative control stops before the third action and retains
two complete transitions. The command returns a nonzero status. A hard process
termination can leave an unsealed directory; inspection refuses it. Earlier
decision/transition artifacts remain for investigation. Reusing any existing
destination is refused.

`capture.json` reports actual prefix counts, error scope and wall timings.
`integrity.json` binds the complete local file inventory. Offline inspection
checks file bytes, the action contract and referenced frame identities; it
does not initialize physics, decode image accuracy, replay a learned policy,
independently certify task success, or qualify hardware. The next learning
step still needs a frozen camera tracker and reviewed successful operator
correction segments under the [proposed experiment](piper-next-learning-experiment.md).

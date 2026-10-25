# Saved video-goal candidates v0.1

This connects an encoded video destination to a reloadable **data-only PiPER
simulation candidate**. Creation, inspection and execution can run in separate
processes. The candidate pins its input hashes, calibration, profile and runtime.
It contains no Python payload, model weights or motor policy. Runtime inspection
rejects changed files, stale code/dependencies and unsupported robot profiles.

The current input is one visible green rigid proxy under a supplied fixed
orthographic camera calibration, for a supported planar block placement. This is
not yet arbitrary human-video learning. No depth, grasp, force or action sequence
is inferred. The existing authored PiPER controller still observes privileged
native simulator state. No command is sent to robot hardware.

## Run locally

Use Python 3.12+, the project's pinned NumPy 2.5.3 and MuJoCo 3.15.0, plus
FFmpeg/ffprobe on PATH. These operations need no GPU, account, API or running
server. Download/install dependencies and provision pinned robot assets as
separate steps; candidate creation and native execution are offline.

```sh
python -m pip install '.[arm]'
python -m dvidia_training.robot_profiles fetch piper --output NEW_ROBOT_ASSET_DIRECTORY
python -m dvidia_training.video_goal_install create \
  --video PATH/DEMONSTRATION.mp4 --calibration PATH/calibration.json \
  --output NEW_CANDIDATE_DIRECTORY
python -m dvidia_training.video_goal_install inspect \
  --bundle NEW_CANDIDATE_DIRECTORY --profile piper
python -m dvidia_training.video_goal_install run \
  --bundle NEW_CANDIDATE_DIRECTORY --assets PATH/PINNED_ROBOT_ASSETS \
  --object-x .294 --object-y -.025 --seed 1 --output NEW_ATTEMPT_DIRECTORY
```

Calibration JSON uses the fields of `video_goal_bridge.PlanarVideoCalibration`;
see the [frozen example](../benchmarks/video-goal-bridge-v01-20261025/evidence/calibration.json).
The camera-to-PiPER task frame relation must already be known. The package uses
the existing task's support height (object center Z=.08 m), size/mass/contact
assumptions and workspace. `object-x/y` are supplied initial simulator state,
not measurements recovered from the video. An unobservable terminal window or
unsupported workspace refuses creation. Invalid starts refuse execution.

The candidate folder contains `manifest.json`, `proposal.json` and
`calibration.json`. It records the video digest without copying private footage.
Copy all three files to a machine with the identical runtime to inspect/run.
The execution directory retains `episode.json.gz` and `receipt.json`, binding
the source candidate, destination, assets, runtime and actual native trace.
Integrity hashes detect changes; they are not a signature proving authorship.
The receipt's controller success is a diagnostic; use independent benchmark
references before making performance claims.

## Evidence and next boundary

The [32-attempt benchmark](../benchmarks/video-goal-install-v01-20261025/README.md)
uses four earlier synthetic videos and two predeclared new starts each. Oracle,
installed-video, shuffled-video and matched open-jaw conditions share the same
original task reference. A separate actual-demo-endpoint score answers a
different question and does not replace the intended-task score.

Reproduce with fresh output, retaining failures:

```sh
python -m dvidia_training.video_goal_install_benchmark \
  --source benchmarks/video-goal-bridge-v01-20261025/evidence \
  --assets PATH/PINNED_ROBOT_ASSETS --output NEW_EVIDENCE_DIRECTORY
```

The source clips and authored controller are reused; this is an installation and
changed-start check, not a novel-video generalization test or a learned policy
evaluation. SO101 is deliberately refused because its mechanics profile has no
compatible placement executor here. Extending to another arm needs frame/TCP,
gripper, units, workspace, perception and contact qualification. The
[evidence gates](video-to-skill-milestones.md) lay out that progression.

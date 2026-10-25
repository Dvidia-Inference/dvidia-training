# Video-to-goal bridge v0.1

DVIDIA’s longer-term product vision is that a user uploads a skill video and receives a compact arm skill. The central research problem is the bridge from pixels to something executable: does the video specify an outcome the robot can identify, a sequence of actions it can reproduce, or both? Those are different learning problems. Ordinary video can communicate *what should happen* without giving robot joint commands, contact forces, or a robot-compatible action sequence. A reusable arm skill can sometimes carry out that goal already; learning a new motor policy from demonstrations is a later, distinct layer.

## What research approaches establish

**XIRL learns task progress as a reward.** It uses temporal cycle consistency across multiple videos of the same task, including different embodiments, to learn a visual embedding of task progress. Distance to a demonstrated goal in that embedding can reward robot learning. The videos need no action labels, but converting the reward into a policy still requires a target robot environment and policy-learning interaction. It is a route from video to a task objective, not direct video-to-motor commands. [XIRL project and paper](https://x-irl.github.io/)

**BCO infers actions from the target robot’s own experience.** Behavioral Cloning from Observation first learns the robot’s inverse dynamics from exploratory state/action/next-state transitions. Given a demonstration in a compatible state representation, it infers the missing robot actions and behavior-clones them. That requires robot interaction data and demonstrations expressed in state features the inverse model can use; arbitrary RGB alone does not provide those states. [BCO paper](https://www.ijcai.org/proceedings/2018/0687.pdf)

**UMI narrows the embodiment gap at data collection.** People demonstrate with a purpose-built handheld gripper and camera system that records robot-like relative trajectories and gripper information. The resulting data is designed to transfer to compatible robot policies. This is robot-free demonstration capture, but it is instrumented capture rather than an ordinary phone-video upload. [UMI project and paper](https://umi-gripper.github.io/)

Together, these approaches motivate a staged product: use ordinary video to identify a goal where that is visually supported; call an existing compatible skill to execute it; collect action-aligned robot demonstrations when a new motor behavior must be learned. The current experiment addresses only the first two stages.

## Implemented bridge and benchmark

[video_goal_bridge.py](../src/dvidia_training/video_goal_bridge.py) decodes a local video and proposes the stable terminal XY position of one visible green rigid cube under a user-declared, fixed orthographic camera calibration. It uses image color regions and timestamps. It does not identify arbitrary objects, infer depth or contact, reconstruct the demonstrated action, or train a policy. An ambiguous, occluded, clipped, or moving terminal candidate can produce an “unknown” result.

[video_goal_benchmark.py](../src/dvidia_training/video_goal_benchmark.py) creates synthetic native PiPER simulations, renders their recorded state traces into replay MP4s, and tests whether the proposed destination can be handed to the existing authored PiPER box-placement controller. The destination marker is hidden in the rendered clips. The controller still uses privileged live simulator state for grasping, transport, and release; the video supplies only its destination. These clips are simulated robot replays, not human demonstrations.

The benchmark pairs each layout’s variants and scores all controller attempts against the same task-goal reference from the frozen protocol. It records an oracle-goal control, the video-derived goal, a shuffled-video negative control, a deliberately miscalibrated-video control, and an open-jaw replay using the video-goal arm commands. Native scoring checks valid physics, lift and retained transfer, then supported, released, settled placement and tool retreat. Scheduled attempts remain in denominators when a proposal or execution fails. Pixel fault clips test terminal occlusion, an extra green candidate, and terminal image motion. These controls distinguish goal grounding, controller capability, and the effect of the gripper on this simulated task; they do not establish general video imitation.

The local video path depends on the project’s NumPy and FFmpeg/ffprobe installation. Running the native benchmark additionally needs the optional pinned MuJoCo dependency and the existing PiPER assets. Native rendering requires graphics access; a CPU-only rendering path has not been established. Reported inference timing covers local file checks, probing, decoding, pixel observation, and hashing. It excludes camera capture, the observation dwell in the video, network transfer, and physical robot actuation.

## Run and inspect

Use fresh output paths; the tools preserve existing outputs.

```sh
python -m dvidia_training.video_goal_bridge infer VIDEO --calibration JSON --output NEW_JSON
python -m dvidia_training.video_goal_benchmark run --assets PATH --output NEW --count 4 --seed 2026102507
python -m dvidia_training.video_goal_benchmark inspect RUN --reprocess-video
```

The [retained four-layout benchmark](../benchmarks/video-goal-bridge-v01-20261025/README.md) produced 3/4 intended-task successes with video goals and 4/4 with oracle goals. All 12 engineered pixel-fault clips caused abstention. Destination extraction took 0.133–0.572 seconds per 20.2-second clip on this host, including local probing/decoding/observation/hashing. These small synthetic counts are a mechanism test, not a reliability estimate.

The missed task reveals an error-budget problem: a 12.04 mm source-placement offset, a 2.70 mm perception offset and a 13.05 mm execution offset largely aligned, producing 27.79 mm error against a 25 mm task tolerance. Video saw the imperfect source endpoint; the oracle knew the original intended center. An exploratory score for matching that observed endpoint was 4/4, while the original 3/4 task score remains unchanged. Three of four proposals stayed inside the declared 5 mm uncertainty bound, so that bound has not been established as reliable.

## Next experiment

First complete the calibrated geometry-to-goal benchmark and review its retained clips, proposals, and paired outcomes. Then test actual action learning separately with action-aligned demonstrations: compare a fixed robot-demo label budget with no video input against the same robot data plus video-derived task information, and add increasing label-budget conditions. Keep task, controller, seeds, and held-out layouts matched. That experiment can ask whether video reduces the robot action labels needed for a new behavior. The current bridge does not answer that data-efficiency question.

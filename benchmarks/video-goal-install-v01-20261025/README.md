# Persisted video-goal candidates — 2026-10-25

This local simulation experiment tests **video → saved candidate → fresh-process
reload → authored PiPER execution**. It learns no motor policy. Four earlier
synthetic green-block videos and their trusted orthographic calibration are
reused; each is tested from two predeclared new starting positions. This is not
an unseen-video or arbitrary-human-footage evaluation.

[Complete report](evidence/report.html) · [All results](evidence/report.json) ·
[Frozen protocol](evidence/protocol.json) · [Evidence hashes](evidence/integrity.json) ·
[Run guide](../../docs/video-goal-install-v01.md) ·
[Original destination pilot](../video-goal-bridge-v01-20261025/README.md)

## All 32 attempts

| Condition | Executed/scheduled | Original intended task | Actual demo endpoint (secondary) |
| --- | ---: | ---: | ---: |
| Oracle intended goal | 8/8 | 8/8 | 8/8 |
| Installed video candidate | 8/8 | **6/8** | **8/8** |
| Candidate from shuffled video | 8/8 | 0/8 | 0/8 |
| Matched open-jaw replay | 8/8 | 0/8 | 0/8 |

All 32 attempts had valid native physics. The unchanged scorer requires retained
lift/carry, actual supported release, low linear/angular motion, gripper opening,
tool retreat and a terminal dwell inside 25 mm of the fixed reference. The
controller's own success flag cannot override that reference.

All **4/4** candidates were created and inspected in separate Python processes.
All **4/4** unsupported SO101-profile inspections refused, as did all **4/4**
occluded-terminal clips. Refusal reasons are retained, not inferred from any
arbitrary subprocess failure. Unknown proposals never became installed candidates.

Both installed-video failures were layout 02: final errors **27.389 mm** and
**27.387 mm** exceeded the intended-task tolerance. The source demonstration
itself ended approximately 12 mm from that task target; pixel inference and
controller execution then add error in the same direction. Both runs still met
the separate actual-demo-endpoint reference. Copying a demonstration's outcome
therefore does not prove it achieved the intended task. This confirms the
[earlier error analysis](../video-goal-bridge-v01-20261025/README.md), without
changing its denominator or success rule.

## Reproducibility and compute

The final frozen run took **75.192 seconds** locally for preparation and all
32 attempts. Candidate-creation subprocesses took **0.406–0.870 s**; inspection
subprocesses took **0.169–0.247 s**, including Python/runtime imports. Candidate
folders contain approximately **22 KB** each, with no raw footage or weights.
These small local measurements exclude acquisition and observation dwell before
upload; they do not qualify robot real-time control or minimum hardware.

A development run before inspection-reader hardening used the same frozen starts,
clips, controller and scoring, with identical counts (106.312 s on a differently
loaded host). The final run follows path-containment and gzip-expansion checks.
No controller parameters or success thresholds were tuned between runs.

Use Python 3.12+, NumPy 2.5.3, MuJoCo 3.15.0, FFmpeg/ffprobe and provisioned pinned
PiPER assets. No GPU, account, cloud request or physical-robot connection is used:

```sh
PYTHONPATH=src python -m dvidia_training.video_goal_install_benchmark \
  --source benchmarks/video-goal-bridge-v01-20261025/evidence \
  --assets PATH/PINNED_ROBOT_ASSETS --output NEW_EVIDENCE_DIRECTORY
PYTHONPATH=src python -c 'from dvidia_training.video_goal_install_benchmark import inspect_evidence; print(inspect_evidence("benchmarks/video-goal-install-v01-20261025/evidence"))'
```

Inspection checks retained bytes, source bindings, receipt-to-bundle/episode
hashes, complete scheduled denominators and recomputed trajectory scores. It
does not rerun physics. Source videos and trajectories are synthetic; the pinned
PiPER assets originate in [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/agilex_piper)
under [MIT](LICENSE-PIPER-MIT.txt). Asset provenance is frozen in the protocol.

This milestone packages a **calibrated goal with an authored controller**. It
does not infer arbitrary object identity, depth, contact or robot actions, and
it does not establish skill learning or transfer to other arms or hardware.
The [next evidence gates](../../docs/video-to-skill-milestones.md) address those gaps.

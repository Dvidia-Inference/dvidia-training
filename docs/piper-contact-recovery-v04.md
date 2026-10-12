# PiPER contact recovery v0.4

Version 0.4 separates two questions: can carry recovery finish sooner, and can
observed failed acquisition avoid a repeated grasp that changes nothing? The
training package version is separate from this controller experiment.

The fresh 84-attempt confirmation completes **4/12 placements in each mode**:
frozen v0.3 history, v0.4 carry-only and v0.4 full. All attempts are numerically
valid and all 48 negative controls fail. The two completed placements that
resume after a hold finish **2.440 and 2.640 simulated seconds sooner** in v0.4;
two other completions are unchanged. Four low-friction failures are refused
**2.88–2.90 seconds sooner**, while remaining failed placements. There are no
gained or lost completions in the matched comparison.

## Run locally

Use Python 3.12 or newer and a CPU. Provision the pinned robot assets once;
rollouts then run locally without a GPU, model API or account. Measured memory
and runtime below describe one host, not universal minimum requirements.

```sh
git clone --branch feat/piper-contact-recovery-v04 https://github.com/Dvidia-Inference/dvidia-training.git
cd dvidia-training
git checkout d5c1cc6adbf6566947aef6defb1bbc04207a96c4
python3.12 -m venv .venv
.venv/bin/python -m pip install '.[arm]'
.venv/bin/dvidia-robot-sim fetch piper --output runs/robot-assets
.venv/bin/dvidia-piper-recovery benchmark --assets runs/robot-assets --output runs/piper-recovery
.venv/bin/dvidia-piper-recovery inspect runs/piper-recovery
```

The command above runs the default development layouts. To rerun the published
confirmation layouts with the frozen implementation, use a fresh output directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/dvidia-piper-recovery benchmark --assets runs/robot-assets --seed 941827 --cohort confirmation --output runs/piper-recovery-941827
.venv/bin/dvidia-piper-recovery inspect runs/piper-recovery-941827
```

The environment-variable prefix is for POSIX shells. In PowerShell, set
`$env:OPENBLAS_NUM_THREADS='1'` and `$env:OMP_NUM_THREADS='1'` before running the
command. This is a reproduction of already published scenes, not a fresh unseen
confirmation. Runtime source hashes match the original freeze; timing and
serialized receipt hashes can vary by host and installed dependency versions.

On Windows use `py -3.12 -m venv .venv` and executables in `.venv\Scripts\`.
Use a fresh output directory for each run. Downloading software and assets is
separate from the offline simulation. Original compressed receipts can also be
inspected without executing MuJoCo; inspection checks stored consistency rather
than independently reconstructing physics or authenticating a publisher.

## Why the six earlier failures stayed failures

All six failed v0.3 history cases fail acquisition. Their object lifts only
0.040–0.148 mm over the full episode; the first probe raises the tool
3.428–3.849 mm. An unchanged supported retry spends another 2.90–2.92 seconds
before aborting. Both the history/probe-only native trajectories are identical,
and the original unguarded controller also fails these cases.

The pinned gripper position servo has proportional gain 40, a minimum closing
target 0 m and a measured finger slide around 0.0196 m. At rest, this means about
−0.784 N of drive force and roughly 0.39 N of normal force per linked finger.
The ±10 N actuator cap is not reached. A side-pinch friction approximation
cannot support the declared weights in these scenes. This diagnoses the pinned
servo and grasp, not the physical robot’s capacity or every alternate grasp.
The [complete forensic appendix](../benchmarks/piper-recovery-forensics-20261011/README.md)
retains all events, window statistics and the effective native friction check.

## Controller contract

The carry change begins **after** the unchanged v0.3 closed hold has qualified relative settling and resumed. It starts from the same 0.45 requested joint-delta scale. A further 100 ms of stable observed history enables a linear increase toward the original scale of 1, accumulated over 300 ms of additional stable observed time. The evidence remains the v0.3 gate: known history, endpoint displacement and maximum pairwise excursion within 1.5 mm, relative rate at most 8 mm/s, loaded bilateral contacts throughout, no support and at least 6 mm object rise. Unstable evidence pauses the increase at its current scale; it does not itself reduce that scale. The unchanged persistent-trend and bilateral-contact-loss responses remain active, and a new hold resets restoration. This tests faster task progress after settling, rather than shortening the settling test.

The full mode adds one acquisition decision. When the original two-observation `object_not_following` alarm fires while the object remains supported, it examines the past 120 ms with at least 100 ms of observed span. Every sample must have loaded bilateral contact, table support and an already-issued closed-jaw request. Each jaw may vary by no more than 0.5 mm and each pad normal force by no more than 0.05 N. If these observations support a stable closed grasp that still does not follow, the controller holds closed and declines the otherwise available unchanged retry. Short, unloaded, varying-force or otherwise ambiguous windows keep the original bounded retry. No hidden mass or friction is consulted, and this refusal does not claim that another grasp is impossible.

The controller uses recorded current/past tool and object pose, pad contact and
normal force, jaw position and support. Hidden scene mass/friction are not
inputs to its new decision. It is authored and uses privileged simulator
observations; it is not a learned sensor-based policy.

The acquisition qualification, independent success scorer, native servo gains,
force limits, scene geometry and free-object dynamics remain unchanged. A
refusal remains a failed placement. A resume event alone is not successful
recovery. Holding the measured arm targets still permits physical motion under
the native dynamics. Every attempt remains 20 simulated seconds, including
completed placements, controls and refusals.

## Paired comparison

Two shared seeded XY layouts cross friction 0.8, 0.55, 0.15 with masses 40 g and 60 g.
The twelve scenes each retain seven variants, for 84 planned attempts:

- Frozen `history` placement and its own matched open-jaw arm tape.
- `recovery_carry` placement and its own matched open-jaw arm tape.
- Full `recovery` placement and its own matched open-jaw arm tape.
- A shared hold control.

The carry-only mode separates carry changes from the full version’s acquisition
decision. Completion-time deltas are reported only when both paired tasks
complete. Abort-time deltas are reported only when neither succeeds and both
abort. Gained and lost completions remain visible outside those matched clocks.
The receipt retains actual motion around requests, whole held intervals,
unresolved recoveries, and completion outcomes.

## Frozen evidence and measurement boundary

The [complete confirmation archive](../benchmarks/piper-recovery-v04-20261011/README.md)
uses fresh seed 941827 after runtime commit
[`ccf8f3d1c2a1977da855ba8f601499aa543a7b6c`](https://github.com/Dvidia-Inference/dvidia-training/tree/ccf8f3d1c2a1977da855ba8f601499aa543a7b6c)
was frozen. An installed wheel from that archive matches all 13 runtime source/lock
hashes and runs under OS process-level network denial with local pinned assets
and one OpenBLAS/OpenMP thread each. Physics runs at 1 ms and controller boundaries
at 20 ms. Every valid attempt contains 1,000 commands and 1,001 states.

The matched mean completion change across all four shared successes is −1.270
simulated seconds; the two intervened successes account for the entire reduction.
Their holds still last 220 and 240 ms, unchanged from v0.3. This speeds task progress
after settling; it does not demonstrate faster grip settling. Full recovery and
carry-only have equal successful task clocks. Full recovery skips four unchanged
low-friction retries; both heavier moderate-friction cases retain their original
retry because observations are ambiguous.

The new cohort also exposes two extra failures compared with the older v0.3
cohort: nominal-friction 60 g blocks qualify acquisition, briefly lose bilateral
contact, regain it during a hold and remain suspended, but their object rise
falls below the unchanged 6 mm resume floor before completing 100 ms of eligible
settling dwell. They time out after 820 ms without resume. All three modes have
exactly equal native states and commands throughout these two failed episodes,
so the new restoration never gets to act. The old cohort’s 6/12 result does not
carry over to the fresh layouts. Eight placements fail in every mode.

On one macOS arm64 host with 10 logical CPUs, Python 3.12.9, MuJoCo 3.15.0,
NumPy 2.5.3 and no GPU, attempts take 1.175–1.728 wall seconds. Per-attempt control
computation p95 is 1.087–3.181 ms; the maximum recorded window is 13.473 ms, with
0/84,000 windows exceeding 20 ms. Every raw per-decision computation duration is
retained. The whole installed command takes 133.629 seconds including evaluation,
compression and final inspection. Process high-water RSS is 727.19 MiB before
inspection and 738.41 MiB for the wrapper’s child lifetime including inspection;
these are measured peaks, not minimum system RAM. Host load is uncontrolled.

The next test can examine a second controlled load-bearing probe from a stable
unsupported hold or gentler acceleration after the initial probe, while retaining
the original independent task and retention criteria. Neither is implemented in
v0.4; lowering a gate after seeing these failures would invalidate this confirmation.

Development attempts are separate from fresh confirmation. The known v0.3
cohort is development evidence for this version. All full native trajectories,
commands, warnings, outcomes, per-attempt timing summaries and model notices
remain retained. CPU times do not measure physical sensor-to-actuator latency;
state snapshots do not establish true-slip accuracy. Material calibration,
compliance, deformation/damage and qualified physical transfer remain open.

Original DVIDIA code is MIT; pinned upstream PiPER assets retain their original
MIT notice. Research writing has its separately declared publication license.

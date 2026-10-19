# Native contact audit — 18 October 2026

The new passive replay audit exposes contact events between the stored control
observations. It reproduced **14/14 frozen episodes exactly** and recorded
**249,502 physics ticks** at 1 kHz, without changing commands, physics, limits or
success scoring. This is diagnostic evidence from two previously inspected
scenes, not a new policy benchmark or a controller improvement.

The input is seven recorded controller tapes and their seven paired open-jaw
controls from the [retention study](retention-experiment-20261016.md). Six tapes
cover the baseline, smoothed and compliant student/teacher on failing scene15;
one is the known successful baseline student on scene14. The model is the
simplified rigid-box arm. PiPER shoe geometry and physical robots are separate
studies.

## What the additional measurements show

| Frozen controller tape | Longest qualifying lift streak | Largest loaded contact gap | Loaded intervals hiding a gap / eligible intervals | Complete task |
| --- | ---: | ---: | ---: | --- |
| Baseline student, scene15 | 75 ms | 6 ms | 151 / 306 | Fail |
| Smoothed student, scene15 | 89 ms | 2 ms | 130 / 305 | Fail |
| Compliant student, scene15 | 947 ms | 6 ms | 105 / 291 | Fail |
| Baseline teacher, scene15 | 898 ms | 6 ms | 187 / 374 | Fail |
| Smoothed teacher, scene15 | 962 ms | 5 ms | 175 / 358 | Fail |
| Compliant teacher, scene15 | 951 ms | 4 ms | 172 / 358 | Fail |
| Baseline student, scene14 | 1,618 ms | 5 ms | 4 / 245 | Success |

The unchanged lift predicate requires both pads to have an individual normal
contact force above 0.01 N, no table contact, at least 55 mm object rise and an
uninterrupted 100 ms dwell. The baseline and smoothed student never meet that
dwell. Compliance and the teacher do meet it but still fail final placement.
All seven recorded open-jaw controls fail, with no qualifying retained lift.

An eligible interval has bilateral contact at both stored boundaries. The audit
counts when an intervening native tick lacks it. Stored observations are
nominally 50 Hz; successful scene14 ends on a partial 2 ms interval. Loaded
statistics include all recorded lift, transfer and place phases, including
retries. They are not restricted to the first grasp or pre-recovery motion.

For these seven controller tapes, missing loaded pad contacts classify as
geometric separation, rather than positive contact force falling below the
scoring threshold. The baseline student's largest measured pad separation is
about 80 micrometres. This describes the simulator's geometry and contact
solution; it does not establish calibrated physical slip. A successful tape
also has brief gaps, so their existence alone is not a sufficient failure
explanation.

The recovery budget also matters. The baseline student's second lift begins at
16.06 s, leaving 1.94 s before the unchanged 18 s deadline. The compliant
student retries its lift at 17.46 s, leaving 0.54 s. Delaying recovery is not the
same as completing placement.

## What the tool records and checks

Each compressed native trace contains timestamps, phase/action identity,
collision candidates, geometry separation, full contact-frame wrenches,
scored contacts, jaw state and actuator force, object/tool motion, relative
offset and independent lift-dwell reconstruction. Candidate contact, positive
force and the original per-contact scoring threshold remain distinct.

Every stored physical observation, the full initial/final information, action
tape, reward, outcome and termination must exactly reproduce the original.
Independent dwell reconstruction is checked against the original scorer at
every recorded boundary. A mismatch invalidates the audit and preserves its
partial trace. Repeated reads of one native tick cannot inflate dwell.

The protocol freezes the complete attempt list, runtime source/dependency
hashes, original compressed inputs and upstream experiment metadata before
execution. `integrity.json` binds the resulting files. The implementation was
frozen at `ff40985ec076eaed8421821b1244217d8502777e` before the 14 replays.

The primary run took **54.18 seconds** for 249.502 simulated seconds, including
instrumentation and export. This is not uninstrumented throughput or a hardware
control-latency result. The Python socket guard recorded zero attempts; it does
not audit native dependency networking. No external inference was used.

A separate full rerun using only the public-format frozen inputs reproduces all
14 compressed native traces byte-for-byte and all physical summaries exactly.
It took 204.60 seconds under different host load. This is a reproducibility
check of the same cases, not another independent set of policy trials.

## Reproduce from public evidence

The [public research entry](https://research.dvidia.org/papers/goal-images-physical-feedback/)
provides the complete frozen bundle and a verified downloader. Downloading the
bundle requires a network connection; replay runs locally. The compact GitHub
[receipt](../benchmarks/native-contact-audit-20261018/README.md) links the same
evidence and preserves its protocol, report and file manifest.

Create a Python 3.12+ environment and install the pinned source with its arm
extra:

```sh
git clone https://github.com/Dvidia-Inference/dvidia-training.git
cd dvidia-training
git checkout ff40985ec076eaed8421821b1244217d8502777e
python3 -m venv .venv
.venv/bin/python -m pip install '.[arm]'
```

Download and inspect `download-audit.py` from the research entry, then use its
documented invocation to retrieve the complete evidence into a new directory.
With that directory substituted below:

```sh
PYTHONPATH=src .venv/bin/python -m dvidia_training.native_contact_audit \
  --from-audit /absolute/path/to/downloaded/audit \
  --output runs/native-contact-audit-reproduction
```

No original private run folders are needed. The loader verifies compressed and
decompressed input hashes and source compatibility; it never executes source
files bundled with the evidence. An existing output directory is refused.
Exact native reproduction is a gate, not a promised cross-platform numerical
invariance: different dependencies or floating-point behavior can fail it.

## Next experiment

First test timestep/solver convergence of these small separations while keeping
physical parameters, commands and time-based scoring fixed. Then predeclare a
mechanically distinct grip/motion change and compare full completion, retained
lift, relative drift, contact loss, recovery time and deadline use. Preserve the
failure cases and the successful comparator. Do not silently raise force or
relax scoring to obtain an apparent gain.

This contact evidence also informs the [goal-image architecture](goal-images-to-actuators.md):
a visual or typed decision layer can choose a subgoal, while the local physical
feedback loop decides whether the grip and motion actually support it.

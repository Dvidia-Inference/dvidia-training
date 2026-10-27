# Offline clock-label feasibility: TRAINING-only diagnostic

**Unfrozen TRAINING-only development diagnostic, 2026-10-27. No new policy, physics trials or confirmation evaluation.**

**Do not treat this as a qualified v0.3 result. Do not start clock fitting from these projected labels.** The existing teachers can be aligned approximately to the frozen movement curve, but this analysis does not validate a unique, reliable scalar progress target. Monotonicity removes impossible jumps by imposing a convention; it does not establish that the convention is the teacher's intended progress. This is a finding about this finite dataset and these diagnostic definitions, not a theorem that timing cannot be learned.

## Scope and reproducibility

Only the 64 synthetic **TRAINING** episodes named in `runs/piper-feedback-v02-20261027-study02/prior-studies/study01/protocol.json` were read. They comprise 16 reset-XY groups × two known destinations × nominal/interruption conditions, 1,001 recorded states and 1,000 commands per episode: **64,064 states / 64,000 command intervals**. Paired conditions and adjacent frames are correlated; the counts below are within-dataset diagnostics, not independent statistical samples. No development or confirmation trajectory was read. No model was fitted, no policy was changed, no new data was collected, and no native dynamics was stepped.

This is an **explicitly derived public projection** of the original diagnostic, not a byte-exact copy of the private run records. Only path/source strings were normalized in [report.json](report.json), [runtime.json](runtime.json) and [preserved-inputs.json](preserved-inputs.json). Every scientific number, source/data digest, model binding and recorded timing is unchanged. `src/…` references are repository-relative; `dataset:…` references point inside the full dataset snapshot; `original-diagnostic:…` references identify separately retained historical records by their original raw digests. Historical script/report hashes in those records intentionally continue to bind the original versions, not the derived public files.

The original nine-file package and original diagnostic remain unchanged in the private run archive. They are not republished because their provenance records contain machine-local paths. [public-derivation.json](public-derivation.json) binds all nine original file hashes, each exact JSON-pointer replacement (original value hashed, portable replacement explicit), and original/derived payload hashes. It exposes no private path strings. [verify_derivation.py](verify_derivation.py) can verify against that separately retained package that no other JSON values changed. [package-integrity.json](package-integrity.json) seals the public files.

The public [analysis.py](analysis.py) differs from the original script by **one string literal only**: the default for `--assets` is now `runs/clock-label-assets`. The algorithm, metrics, thresholds, dataset selection and DP are unchanged. [protocol-proposal.md](protocol-proposal.md) and [verify_dp.py](verify_dp.py) remain byte-exact copies. The DP-check record only updates the script hash to the derived script.

### Portable reproduction

Run from a DVIDIA Training source checkout containing the implementation sources listed in `runtime.json`. Use the **full downloaded dataset snapshot** from [Dvidia/piper-block-placement-feedback-demonstrations-v02](https://huggingface.co/datasets/Dvidia/piper-block-placement-feedback-demonstrations-v02), pinned to a reviewed dataset commit. The model-only repository or partial snapshot is insufficient. Its root must contain `base-model.json`, `candidates/jaw4/model.json`, `training-parent.json`, and `prior-studies/study01/protocol.json` plus all 64 TRAINING episode tapes. The script selects the TRAINING roster explicitly; it does not read the included development/confirmation tapes.

The CLI accepts `--study <downloaded full dataset>` and `--assets <fetched assets>`; replace the example dataset path below with your snapshot directory. Python 3.12 or newer is required; use the versions recorded in `runtime.json` when comparing numerical results. The source checkout's `[arm]` extra installs the pinned NumPy/MuJoCo versions. Assets are fetched explicitly once and then verified/used locally. No GPU is required for this diagnostic; no new dynamics is run.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[arm]'
.venv/bin/dvidia-robot-sim fetch piper --output runs/clock-label-assets
mkdir -p runs
```

The public script retains a hardcoded draft-protocol **input-binding path**. Copy the included proposal there before running. Use a fresh checkout or ensure that destination does not exist; the command below refuses to overwrite a file or symlink.

```sh
.venv/bin/python - <<'PY'
from pathlib import Path
source = Path('benchmarks/piper-clock-label-feasibility-20261027/protocol-proposal.md')
target = Path('runs/piper-feedback-v03-protocol-draft.md')
with target.open('xb') as stream:
    stream.write(source.read_bytes())
PY

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python benchmarks/piper-clock-label-feasibility-20261027/analysis.py \
  --study "/path/to/downloaded-full-dataset" \
  --assets runs/clock-label-assets \
  --output runs/piper-clock-label-feasibility-REPRO/report.json

.venv/bin/python benchmarks/piper-clock-label-feasibility-20261027/verify_dp.py
```

All source/data hashes must match the original bindings before claiming a reproduction of these inputs. The output directory must be fresh. The public script records local paths when rerun, so keep its raw output local or derive a separately documented public projection. A reproduced report has different local paths, script basename/path, wall time and potentially host-dependent floating-point last digits; do not claim byte-identical report reproduction. The packaged report/runtime are path-normalized derivatives, not the original raw evidence bytes.

Runtime was 38.35 seconds on the original host; this is analysis throughput, not robot/control-loop performance. The separate runtime binding was captured after the pass without replaying the analysis. [verify_dp.py](verify_dp.py) independently enumerates every legal path on tiny deterministic matrices and compares the DP objective; [dp-check.json](dp-check.json) records that separate zero-simulation check. It validates the recurrence and constraints, not the scientific correctness of progress labels.

Pinned assets were verified before use: Menagerie commit `0059d4335f8156206f63a35662313385f7ad6d74`, PiPER lock SHA256 `4b2ee2ca675aceb2f826983cb3082a3e08e9aa0ff419b23078abcff08ebcc6fe`, 90 files / 34,562,302 bytes. `build_task_model` compiled the pinned geometry; `mj_kinematics` computed TCP positions from joint poses. There were **zero calls to reset, mj_step, mj_forward, rollout or training**. Reconstructed measured-joint TCP matched logged TCP exactly on 21 states per episode (1,344 checks, maximum coordinate-distance error 0 m).

## What was projected

The reference is the frozen v0.1 high-resolution motor primitive (`dvidia.piper-motor-primitive/v2`) queried at 0–20 seconds in 20 ms increments. Each episode uses its measured reset XY and one fixed known goal. Its 16 cubic spatial terms contract the stored coefficients; 401 normalized Gaussian temporal bases (width 0.025 s) produce seven references. The first six are multiplied by 3 rad; jaw is clipped to [0, 0.035] m. The arm curve is the **raw nominal reference before live-q generic caps**, not a claimed realized trajectory. All sampled raw arm references stayed inside the existing joint-limit margin. Static FK supplies the reference TCP point; it ignores orientation in the TCP metric.

The candidate scalar is `p = reference_seconds / 20`, including pauses. Two distinct distance definitions were tested independently:

- Joint distance: RMS error over six joint angles, in radians. This avoids a TCP position alone hiding orientation/configuration differences, but is still a diagnostic projection, not an inference input.
- TCP distance: Euclidean distance in metres. This is the proposed simple position projection; no orientation or contact criterion disambiguates it.

Pointwise nearest projection searches the complete reference curve and breaks exact ties at the earliest reference index. Near-optimal ambiguity is the span from earliest to latest reference time with distance at most **minimum distance + 1 mm** (TCP) or **minimum distance + 0.01 rad RMS** (joints). A span of at least 0.5 s is counted. These tolerances were declared in the script before running as sensitivity probes, **not acceptance thresholds**. A near-optimal span is not necessarily one continuous feasible interval; it can bridge separate visits to similar poses.

The bounded monotonic projection uses a global dynamic program minimizing summed squared distance, starts at reference index zero, and allows hold or advancement by 1–3 reference ticks per observed 20 ms interval. Thus its maximum rate is 3× nominal. Earliest predecessor/index breaks ties. It does not force the last state to `p=1`. The global optimizer uses later recorded rows **offline**, so its labels must not be confused with a causal inference algorithm. Both the 3× bound and tie rule are arbitrary diagnostic assumptions requiring prospective validation before any label contract is frozen.

## Results

| Diagnostic | TCP projection | Six-joint projection |
|---|---:|---:|
| Pointwise nearest mismatch, median / p95 | 2.395 / 7.323 mm | 0.00871 / 0.01954 rad RMS |
| Bounded monotonic mismatch, median / p95 | 3.022 / 7.784 mm | 0.00876 / 0.01955 rad RMS |
| Same physical-clock mismatch, median / p95 | 7.623 / 41.443 mm | 0.02179 / 0.06893 rad RMS |
| ≥0.5 s near-optimal span, out of 64,064 states | 29,511 (46.06%) | 32,862 (51.30%) |
| Near-optimal span, median / p95 / maximum | 0.08 / 5.28 / 9.82 s | 0.50 / 4.52 / 5.14 s |
| Pointwise backtrack >20 ms, out of 64,000 intervals | 923 (1.44%) | 196 (0.31%) |
| Pointwise forward jump >60 ms, out of 64,000 intervals | 1,188 (1.86%) | 732 (1.14%) |
| Largest pointwise backward / forward jump | −9.52 / +9.82 s | −1.22 / +1.58 s |

These apparent jumps describe the **estimated path coordinate**, not measured robot motion. TCP-position projection can switch between geometrically similar portions of the curve. The joint metric reduces this switching, but does not remove multi-second uncertainty during slow or repeated configurations. The ambiguity tolerance is added to the best residual fit; it does **not** mean that all counted candidate poses are within 1 mm of the measured TCP. These full-horizon diagnostics include pauses and revisits. They do not assess, or rule out, a causal bounded-window projection combining joint configuration, orientation and contact, nor do they isolate moving/contact-critical intervals.

The frozen reference itself contains substantial ambiguity: averaged across episodes, **27.12%** of reference states have another pose at least 0.5 s away within both 1 mm TCP and 0.01 rad joint RMS (range 21.38–31.57%). **22.32%** of reference intervals are simultaneously below 1 mm/s TCP speed and 0.01 rad/s joint RMS speed (range 16.8–26.3%). A physical-time coordinate keeps advancing during these nearly stationary intervals, while pose provides little evidence for which instant is intended. Arc length would collapse pauses, but would still need a directed segment/history convention for revisits and an explicit hold decision.

Enforcing monotonicity does not solve the semantic issue. The TCP- and joint-derived monotonic coordinates differ by a median **0.38 s**, p95 **2.14 s**, maximum **3.34 s**; **29,545/64,064 states (46.12%)** disagree by at least 0.5 s. The TCP fit holds on a mean 54.07% of command intervals and uses the imposed 3× limit on 16.88%; the joint fit holds on 29.17% and uses the limit on 5.59%. Those are properties of the imposed optimizer, not discovered correct control decisions. At the final physical state, 56/64 TCP alignments and all 64 joint alignments stop before reference 20 s; the joint endpoint range is 15.68–17.40 s. Terminal repeated poses and unforced-end tie conventions make the last path-time especially unsuitable as a correctness label.

The improved distance after retiming is useful evidence that timing/servo lag contributes to error. It is **not evidence that a learned clock alone can complete the task**. Nearest TCP mismatch still reaches 17.61 mm, and the raw reference is a command curve rather than the realized contact-constrained path. Even perfect pose alignment would not certify a loaded grip, correct release or collision-free motion. An unforced-end DP stopping on a terminal plateau is an expected parameterization/tie effect, not proof of physical failure.

## Frozen jaw coupling

The jaw model must retain the actual physical-time feature and actual physical-time frozen nominal jaw reference. Feeding virtual path progress into those slots would silently change the frozen model contract. Retiming only the arm nevertheless changes when the robot reaches acquisition/carry/release poses relative to those physical-time cues, and changes the object-relative/force/contact histories seen by the jaw tree.

As a bounded **input-sensitivity** check, the script kept each recorded causal teacher grip snapshot unchanged and shifted only physical time plus its corresponding physical nominal-jaw reference by +0.5/+1/+2 s, within the 20 s domain. The frozen jaw decision changed on **782/62,464 (1.25%)**, **1,582/60,864 (2.60%)**, and **3,429/57,664 (5.95%)** eligible rows, respectively. This is a counterfactual feature test, not a rollout, performance estimate or proposed inference feature manipulation. Real arm delay also changes live grip inputs, so these rates are neither a failure probability nor a bound on coupling. They do establish that freezing model bytes alone does not make the jaw invariant to arm timing.

## Next development check

Do not export these nearest/DP coordinates as ground-truth increment labels. First create a separately versioned **same-frozen-path authored teacher** with an explicit, logged causal hold/advance state and a declared rate bound. It should query the same six-joint curve and retain the identical physical-clock jaw model; log its progress **before command issue**, both clocks, command bounds and actual contact response. That gives a direct teacher progress label without inferring intended time from a repeated pose. Its hold/advance state is training supervision, not a student observation or runtime task-phase input.

Before fitting anything, test whether that authored same-path timing controller can acquire, carry and release under the proposed eligible post-grasp intervention. If it cannot, separate the issue: geometry/clearance may need a new path version rather than a more elaborate clock. If it can, use fresh prospectively defined TRAINING/development data to check the label contract, arm–jaw coupling, causal feature sufficiency and matched blind comparison before selecting a learner. No new collection, teacher implementation, acceptance gates or protocol edits were performed in this diagnostic.

## Original and derived bindings

| File | Original raw SHA256 | Derived public SHA256 |
|---|---|---|
| `report.json` | `87126c306abb9644618329a15cad4a5d576a7f7fd45c83764f5052ae3f48c689` | `c0005c80970462dc8a836046beae41d791811195715378d0ce8bd0548e027ef4` |
| `runtime.json` | `2fbae13acf47f588094f30b81e4c5fb252ebbe587c844ac8ed360da7293315c5` | `a853d05b221af476e83af2e2cf473eb33d6dba2cb97be0fae66621863d367681` |
| `analysis.py` | `1c3cc0d1ffcec1a2a4314aaf45d3a5d90af7d014554aa20929269870e6257400` | `fc03d92bdf357bfe5d82983f4159455973e3aed7eabbaa78bd549f0c33a29e3f` |
| `protocol-proposal.md` | `1b2178c4382dcc24c29947e63e46f266a83abda73572a47dd059ab986be05012` | `1b2178c4382dcc24c29947e63e46f266a83abda73572a47dd059ab986be05012` |

To verify the derivation when the original archive is available locally:

```sh
.venv/bin/python benchmarks/piper-clock-label-feasibility-20261027/verify_derivation.py \
  --original-package "/path/to/original-nine-file-package"
```

This checks original/derived file digests, the exact allowed path-string changes, all other JSON field values, the one-literal script edit, unchanged protocol/checker bytes and the public package seal. It does not require or read robot footage, rerun the analysis, fit a policy or step physics.

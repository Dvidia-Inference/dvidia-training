# DVIDIA compact actuator pilot — frozen evidence, 11 October 2026

This CPU experiment learned a seven-output, next-20-ms actuator transition from
native simulation telemetry. Issued commands improved joint prediction and jaw
RMSE relative to state-only and simple baselines, but important jaw errors remain.
It is a research predictor, not a robot controller or a learned shoe skill.

The exact final run artifacts are retained without modification. These original
DVIDIA model artifacts, evidence and documentation use the [MIT license](LICENSE).
The source candidate is
[`feat/laptop-skill-pilot-v0.1`](https://github.com/Dvidia-Inference/dvidia-training/tree/feat/laptop-skill-pilot-v0.1).
The measured source is pinned to [commit `0d3f05f`](https://github.com/Dvidia-Inference/dvidia-training/tree/0d3f05f186423e18b15ed8362d772b7d74ec0207);
[source.json](source.json) retains supplemental code hashes without modifying
the frozen receipt bundle.
See the [pilot guide](../../docs/laptop-skill-pilot.md) for installation and the
[episode-tape contract](../../docs/episode-tape.md) for signal provenance.

## Corpus and evaluation boundary

The authored MuJoCo 3.15.0 task is rigid-box placement with a six-joint arm and a
parallel-jaw gripper. The 12 declared seed groups, 311000–311011, each contain
three fresh sibling recordings: nominal, open jaw and release after lift.
All siblings stay together: eight training groups, two development groups and
two test groups. These are related native simulation resets, not independently
captured physical scenes. The same authored controller generates every case.

The capture plan declares scene jitter and bounded mass/friction variation before
collection. Group mass spans 0.035–0.045 kg and friction 0.70–0.90; the ordered
split holds out the final groups. Configuration, seed, group, case, phase,
terminal outcome, object/contact state and after-interval applied controls are
not predictor inputs. The models receive contemporaneous native joint positions,
joint velocities and jaw gap, with issued joint/jaw targets added for the action
model. This is privileged simulator telemetry in sensor-shaped fields; it was
not extracted from egocentric footage.

| Split | Groups | Recordings | Attempted control steps | Retained 20-ms transitions | Excluded partial boundaries |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 8 | 24 | 18,244 | 18,228 | 16 |
| Development | 2 | 6 | 4,559 | 4,555 | 4 |
| Test | 2 | 6 | 4,557 | 4,553 | 4 |
| Total | 12 | 36 | 27,360 | 27,336 | 24 |

All 36 recording attempts completed and none were discarded. The authored
controller reached simulation success in 24 recordings; the 12 open-jaw
recordings reached their horizon without success. These outcomes describe the
recording controller and are not learned-model success rates. Each case has
12 recording attempts. The test contains 1,171 nominal, 1,800 open-jaw and
1,582 release-after-lift transitions. The 24 exclusions retain explicit reasons
rather than inventing a full 20-ms target from a partial terminal interval.

Both ridge models fit normalization only on training rows, with equal total
weight per retained episode. Each selects alpha from `[0.0001, 0.01, 1, 100]`
using the fixed development score `joint_rmse_rad / 0.01 + jaw_rmse_m / 0.001`.
Both selected `0.0001`. Model files and [freeze.json](freeze.json) were persisted
before test-tape parsing or transition-target extraction. The manifest's test
identities, hashes and recording outcomes were already available for preflight;
`held_out_transition_targets_read: false` refers specifically to the transition
targets at freeze time. Test results did not choose the models.

## Held-out prediction errors

All five predictors cover the same 4,553 test rows. Joint statistics flatten the
six joint errors per transition; worst joint error is the largest absolute
component across all rows. Jaw statistics concern the measured gap delta.
Radians and millimetres are separate quantities; there is no mixed-unit accuracy
percentage. JSON receipts retain jaw values in metres; tables convert to mm.

| Predictor | RMSE (rad) | MAE (rad) | Worst absolute error (rad) |
| --- | ---: | ---: | ---: |
| Action-conditioned ridge | 3.63056194e-05 | 7.24671574e-06 | 0.00162278889 |
| State-only ridge | 0.000410349272 | 7.96971967e-05 | 0.0131272116 |
| Persistence | 0.00260724634 | 0.0015489858 | 0.0141029159 |
| Constant joint velocity | 0.000577719745 | 0.000265731457 | 0.0120383224 |
| Training-mean delta | 0.00254929171 | 0.00169018742 | 0.0141142779 |

| Predictor | RMSE (mm) | MAE (mm) | Worst absolute error (mm) |
| --- | ---: | ---: | ---: |
| Action-conditioned ridge | 0.34160938 | 0.114485012 | 3.4245926 |
| State-only ridge | 0.551877626 | 0.188971488 | 3.55103368 |
| Persistence | 0.564417684 | 0.108018148 | 3.20004021 |
| Constant joint velocity | 0.564417684 | 0.108018148 | 3.20004021 |
| Training-mean delta | 0.564417684 | 0.108021865 | 3.20004778 |

Persistence predicts zero delta. Constant velocity predicts `joint_velocity ×
0.02` for the joints and zero jaw delta because jaw velocity is not an input.
Training mean uses the fixed weighted training delta. Exact case/group/slice
results remain in [report.json](report.json).

**The jaw result is mixed.** The action model improves aggregate jaw RMSE
(0.341609 mm versus 0.564418 mm for persistence), but its jaw MAE is worse
(0.114485 versus 0.108018 mm), and its worst absolute jaw error is worse
(3.424593 versus 3.200040 mm). On the 1,800 open-jaw test transitions, action-model
jaw RMSE is 0.057758 mm versus 0.002761 mm for persistence. A lower average squared
error does not imply universally better predictions or safe actuator behavior.

The predeclared jaw-changing slice, `abs(measured gap delta) > 0.01 mm`, contains
296 test transitions. Its action-model jaw RMSE is 1.295277 mm, MAE 0.957763 mm,
and worst error 3.424593 mm; persistence gives 2.213605, 1.647262 and 3.200040 mm.
The moving-joint slice, any `abs(measured joint delta) > 0.0001 rad`, contains
3,398 transitions. Its action-model joint RMSE is 0.0000418183 rad versus
0.000668735 rad for constant velocity. These slices are evaluation summaries,
not selection inputs.

## Matched intervention exposes an opening error

The first retained nominal/release pair with identical 13 before-state inputs
and identical issued joint targets occurs at reset-010, index 290, around 5.8 s.
The issued jaw target changes from 33.537947 mm to 80 mm.

| Next-20-ms jaw delta | Nominal | Release intervention |
| --- | ---: | ---: |
| Native observation (mm) | −0.000001286 | +0.010414352 |
| Action-model prediction (mm) | −0.073450757 | +3.429849401 |
| State-model prediction (mm) | −0.108892870 | −0.108892870 |

The action model responds to the changed command, while the state-only model
necessarily gives the same prediction for both inputs. However, the action model
substantially overpredicts the opening in this contact-bearing transition:
3.429849 mm predicted versus 0.010414 mm observed. This demonstrates command
sensitivity and also a serious transient error. It does not validate contact
physics, gripper opening or an accurate action-conditioned world model.

## Measured laptop costs

The final fit/evaluation run used macOS 26.5.2 on arm64, with ten logical CPUs,
Python 3.12.9 and NumPy 2.5.3. The receipt does not identify a specific Apple
chip model. No GPU or remote inference service was used. Fitting uses NumPy;
saved-model inference uses the Python standard library.

| Warm compute-only predictor | Median | p95 | Worst |
| --- | ---: | ---: | ---: |
| Action model | 0.198250 ms | 0.206752 ms | 0.231667 ms |
| State model | 0.144833 ms | 0.151550 ms | 0.184917 ms |

Each model received 100 warmup calls followed by 1,000 measured calls over
pre-extracted test feature vectors. The timed `predict_features` call includes
model validation and arithmetic. This excludes camera capture, perception,
feature extraction, transport and actuator response; it is not a measured robot
reaction time or an operating-system deadline guarantee. Cold saved-model
loading was not separately timed. Timing differences between repeated runs can
reflect scheduling and process history.

The original corpus collection took 41.129711 s for
547.09 simulated seconds. The final fit/evaluation pipeline took
19.292200 s before the final receipt write,
including 8.799433 s for fitting/development selection.
Tape load/validation and feature/target extraction are reported separately:

| Split | Tape load and validation | Feature/target extraction |
| --- | ---: | ---: |
| Training | 1.653895 s | 2.772606 s |
| Development | 0.445649 s | 0.709693 s |
| Test | 0.442433 s | 0.730566 s |

Process-lifetime peak RSS was 184,614,912 bytes
(176.062 MiB). This is the process high-water mark,
which includes earlier work in that process; it is not isolated learner memory
or a minimum hardware requirement. The two serialized model files total
11,445 bytes: 6,438 for action and 5,007 for state.

## Artifact use and reproducibility

| Artifact | Purpose |
| --- | --- |
| [report.json](report.json) | Exact final results, timings, exclusions and limitations |
| [corpus-manifest.json](corpus-manifest.json) | Exact manifest bytes, grouped provenance and original tape hashes; tapes omitted |
| [freeze.json](freeze.json) | Selected models, source hashes and transition-target boundary declaration |
| [action.model.json](action.model.json) | Original MIT action-conditioned 20-input, seven-output ridge artifact |
| [state.model.json](state.model.json) | Original MIT state-only baseline with 13 active inputs |
| [integrity.json](integrity.json) | Hash/byte receipts for the five retained artifacts |

The bundle was checked for private absolute paths, credential-shaped values and
non-native recording sources before publication. No raw tapes, private footage,
third-party pretrained model weights or credentials are included. Original
DVIDIA coefficients and native-simulation evidence are covered by [LICENSE](LICENSE).
MuJoCo and other dependencies retain their own licenses.

With the candidate source installed, inspect this bundle locally:

```sh
dvidia-transition inspect benchmarks/actuator-pilot-20261011
```

Inspection verifies internal hashes, model contracts, frozen selection and
source/count consistency. It cannot authenticate the origin of declared sources,
recompute these metrics without the omitted tapes, or establish physical accuracy.
The model contract explicitly records `is_controller: false`,
`predicts_task_success: false` and `physical_robot_ready: false`.
Intended use is offline predictor comparison, simulator research and API
experimentation. Hardware command generation, stopping guarantees, grasp success
and shoe organization require separate validated implementations and evidence.

The recorded generator originally emitted `manifest.json`; a byte-identical
`corpus.json` copy supplied the final CLI contract. The current generator emits
`corpus.json` directly. This filename correction did not regenerate the corpus
or change any tape bytes. The final fit reused that corpus and corrected the
freeze wording to distinguish transition targets from already-read manifest
outcomes. Both model SHA256 values and all prediction metrics are identical to
the first fit; timings and final receipt hashes reflect the final rerun.
Reproduction on another machine can change timings and numerical serialization;
compare source/configuration contracts and metrics rather than assuming every
native simulation or fit will be byte-identical.

The next research gap is calibrated actuator/contact prediction and measured
hardware feedback, followed by a collected shoe task with synchronized robot
actions and a learned policy. These results establish a local training/evaluation
building block for Skillspace; they do not show footage-to-shoe-control learning.

# Simulator episode tapes

Episode tapes preserve what the authored research arm observed, requested and
actually did at each control boundary. They are a separate versioned evidence
format. They do not change the existing movement-training sidecar, train a
supervisor, recover actions from human videos or qualify physical hardware.

The recorder runs locally and offline with the optional `arm` dependencies.
Validation and inspection use only the Python standard library. No renderer,
GPU, FFmpeg, account, API key or network is needed for these operations after
installation. These are dependency requirements; minimum RAM and throughput
have not been established across machines.

## Record fresh evidence

Install the simulator extra using the [installation guide](installation.md).
Use a fresh output path outside the frozen benchmark directories:

```sh
mkdir -p runs/episode-tapes
python -m dvidia_training.episode_tape record \
  --output runs/episode-tapes/nominal-19050.json \
  --episode-id nominal-19050 \
  --source-recording-id recording-nominal-19050 \
  --session-id simulation-session-19050 \
  --seed 19050 --case nominal

python -m dvidia_training.episode_tape record \
  --output runs/episode-tapes/open-jaw-19050.json \
  --episode-id open-jaw-19050 \
  --source-recording-id recording-open-jaw-19050 \
  --session-id simulation-session-19050 \
  --seed 19050 --case open-jaw

python -m dvidia_training.episode_tape validate runs/episode-tapes/nominal-19050.json
python -m dvidia_training.episode_tape inspect runs/episode-tapes/open-jaw-19050.json
```

The `nominal` case records the unchanged authored controller. `open-jaw` keeps
its joint requests while issuing an 80 mm jaw gap. `release-after-lift` requests
an 80 mm jaw gap at control boundaries falling within 0.35 simulation seconds
after the environment first reports a retained lift. With the default 20 ms
control interval, the final such request also governs the remainder of its
step, so the overridden command intervals span 0.36 seconds. The native target
ramp still limits opening speed, and measured aperture may lag the requested
gap. The recorder then returns command authority to the unchanged authored controller. The
interventions change jaw commands and leave native object dynamics intact.
Their outcomes depend on the authored scene and controller; these examples
are recording checks, not a new generalization benchmark.

Related recordings must stay in the same source/session split group. The two
examples above share a session because they reuse the same seeded scene.
Assign stable IDs when collecting evidence; changing an ID cannot make a
duplicate or counterfactual recording independent. The recorder preserves
these IDs but does not assign dataset splits or inspect an existing corpus.

Output creation refuses to overwrite an existing file. A malformed tape is
rejected before a write. A broken native clock or invalid numeric observation
cannot be exported as valid evidence. Inspection never modifies the input.

## What the v1 format preserves

| Component | Meaning |
| --- | --- |
| Provenance | Episode, recording and session IDs; seed and case; authored policy revision; MuJoCo version; scene, configuration and source-file hashes. |
| Clock | Simulation elapsed seconds, with zero after reset settling and a native-clock offset. Wall-clock timing and hardware clock synchronization are outside v1. |
| Initial / before / after | Measured simulator state at zero and at each control boundary. The previous after-state must equal the next before-state. |
| Policy request | Authored joint targets in radians and requested jaw gap in metres. |
| Issued command | The command passed to the environment, separately retaining any declared jaw intervention. Issue time equals the before-state time. |
| Applied command | The step’s start/end times, final ramped targets, final native servo controls and rate-limit flag. These are commands, not measured positions or velocities. |
| Measurements | Joint positions/velocities/actuator torques; jaw actuator forces and measured gap; per-pad normal forces and object contacts; table contacts; tool and object pose/velocity; target position. |
| Authored labels | Action phase and next phase captured after policy evaluation, before the step. These are controller evidence, not approved human labels or deployment inputs. |
| Outcome | Terminal reason, success, validity, simulation duration, step count, authored failure reason and recovery count. These must agree with the final transition. |

Each signal declares its dimensions, units and coordinate frame. Quaternions
use `w, x, y, z`. All recorded signals come from privileged simulator state;
the format does not claim that a physical arm has equivalent sensors. Contact
normal force in newtons is distinct from jaw actuator force. Requested closing
and a measured gap alone are not labels for a successful grasp.

A signal is `available`, `stale` or `missing`. Available and stale readings
retain acquisition time and age. The v1 freshness limit is 0.1 seconds, a
recording-contract choice rather than a validated hardware deadline. Missing
readings have null value, acquisition time and age. They are never fabricated
as zeros. Current native recordings acquire all available signals at the
observation boundary, so their age is zero; stale/missing support is validated
for future data producers but no sensor-fault simulator is added here.

The tape samples at the environment’s control interval, ordinarily 20 ms. It
does not retain the intermediate 1 ms native ticks. Final servo controls
include the environment’s gravity compensation. Applied-command timestamps
describe the entire transition and the final controls, not a claim that the
final command was active throughout that interval. Intra-step force peaks and
short contacts can therefore be missed by this tape.

Validation refuses unsupported versions/scopes, unknown or duplicate fields,
nonfinite/boolean numeric entries, malformed shapes or quaternions, inconsistent
sensor ages, contradictory missingness, reordered or discontinuous observations,
command/transition timing mismatches, unsupported actuator commands, inconsistent
terminal outcomes and changed protocol/configuration hashes. Limits are 10,000
control steps, one million native ticks per control or settling interval, and
64 MiB per tape. The clock origin must match the declared native settling ticks,
and both native jaw controls must equal half the final ramped gap. Hashes
establish consistency, not provenance
authenticity, sensor accuracy or task mastery.

## Next consumers

The first consumers should be a synchronized episode inspector and a frozen
training/confirmation split, followed by a compact state-based outcome model.
The current visual predictor and movement head do not consume these tapes.
A future learner must choose deployable inputs explicitly and hold privileged
object state and authored phase labels out of its deployment feature vector.

Every eventual skill candidate needs its own versioned simulator task,
observable success/failure conditions, fresh held-out scenarios, matched controls,
complete attempted-episode receipts, performance measurements and a supported
robot/gripper envelope. A validated tape is an input to that process; it is not
an installable qualified skill.

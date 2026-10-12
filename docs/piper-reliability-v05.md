# PiPER reliability v0.5

This experiment compares the frozen v0.4 controller with a bounded second lift and a fixed alternate-grasp planner. It is an authored, privileged-state simulation study of rigid blocks. It does not train from footage, infer materials or qualify a physical robot.

The full task still requires load-bearing lift, retained carry, supported placement, release and retreat. Every attempt keeps its original 20-second budget. A successful recovery decision is not automatically a successful placement.

## Frozen results

Across three preregistered cohorts, the original controller completes **15/36**
placements, the second lift **18/36**, and the revised grasp **23/36**. The full
candidate gains eight completions and loses none, reducing failures from 21 to
13 (38.1% fewer failures within this test). Shared successful placements average
0.772 simulated seconds slower. These are six sampled layouts shared across
mass/friction conditions, not 36 independent real-world trials.

All 252 native attempts pass their recorded-state/policy audits, and all 144
matched negative controls fail. The remaining thirteen failures are visible in
[the complete results](../benchmarks/piper-reliability-v05-20261012/README.md).
The heavily loaded laptop missed computation deadlines; this experiment does
not establish physical real-time performance.

## What changes

`recovery` preserves v0.4 behavior. `reacquire` adds at most one second vertical test lift when an already-qualified grip becomes stable, loaded and unsupported but settles below the original 6 mm resume height. It requests at most 8 mm at 10 mm/s, keeps the jaw closed and has a 1.2-second deadline. It must re-earn the original height and 100 ms stability gates plus at least 4 mm measured tool rise before resuming. Missing history, loss of contact/load, support return or renewed slip takes the closed abort path.

`reacquire_grasp` adds a fixed −45° world yaw for the downward grasp and a 0.10 rad orientation-error clamp for that grasp target in differential IK. The original approach/carry yaw and other target clamps remain. The native servo gains, force/control limits, geometry, friction, integration and task scorer are unchanged. This fixed strategy is chosen before the episode; it is not a learned grasp selector or a material-based decision.

## Reproduce locally

The frozen runtime is [e76b2e41256604d5b4314e45ca45959cab9aa571](https://github.com/Dvidia-Inference/dvidia-training/tree/e76b2e41256604d5b4314e45ca45959cab9aa571). Python 3.12+ and the pinned NumPy/MuJoCo dependencies are required. No GPU, hosted model or API key is used by simulation.

```sh
git clone https://github.com/Dvidia-Inference/dvidia-training.git
cd dvidia-training
git checkout e76b2e41256604d5b4314e45ca45959cab9aa571
python3.12 -m venv .venv
.venv/bin/python -m pip install '.[arm]'
.venv/bin/dvidia-robot-sim fetch piper --output runs/piper-assets
.venv/bin/dvidia-robot-sim audit piper runs/piper-assets
.venv/bin/dvidia-piper-reliability benchmark \
  --assets runs/piper-assets --output runs/reliability-832847 \
  --seed 832847 --cohort confirmation
.venv/bin/dvidia-piper-reliability inspect runs/reliability-832847 \
  --assets runs/piper-assets
```

Repeat the benchmark with seeds **227037** and **862186**, using fresh output directories. Each cohort has 12 scenes and seven variants: three placement modes, an individual open-jaw replay for each mode and a shared open-jaw hold. All **84 attempts** per cohort remain in the denominator, including failures and invalid attempts. Downloading dependencies/assets requires network access; after provisioning, the simulator and inspectors operate locally. On Windows use the corresponding `.venv\Scripts\` executables.

The release runs use an installed wheel built from the committed archive, with all sixteen runtime/lock hashes verified. Three protocols were committed before any fresh outcome was observed. OS process-level network denial is tested with an `EPERM` socket check on macOS. This does not qualify air-gapped dependency installation or promise identical wall times on other machines.

## Inspect the evidence

Each complete cohort contains all 84 compressed attempts, its original report/protocol/seal and MIT model notices. Valid episodes retain 1,000 issued commands, 1,001 native states and every raw computation duration. Download the entire cohort directory, not only its report.

Portable inspection checks recorded task consistency, paired models and starting states, every mode's own arm tape, timing samples, denominators and explicit second-lift prerequisites and bounds. With `--assets`, inspection also reconstructs TCP forward kinematics and every controller action, IK result, diagnostic and event from recorded states under the exact frozen source. It checks the recorded native lift latch, which the original teacher reads. This is a recorded-consistency audit, not independent physics reintegration or proof of authenticity.

The [development archive](../benchmarks/piper-reliability-development-20261011/README.md) retains all 200 prior native attempts and discarded candidates. It includes 24 exact original-controller regressions and a separate intermediate two-mode benchmark. Development scenes are not pooled with fresh confirmation. The [preregistered evaluation plan](piper-reliability-v05-plan.md) defines the comparison and limits.

## Next reliability work

First address the remaining failure mechanism, then freeze another comparison. Useful next experiments are supported alternative grasps for low-friction objects, gentler transitions for a held object that loses load during carry, and explicit native disturbances. The current tests do not establish robustness to external pushes, camera latency, missing observations, compliant shoes or material damage. Those require separately declared models and measurements.

A future installable skill should carry its compatible arm/gripper profile, observation requirements, allowed operating conditions, success/failure tests, bounded recovery behavior and complete benchmark evidence. It should refuse incompatible inputs before controlling hardware. Downloading a controller cannot substitute for calibrating the real gripper, sensors and contact model.

Original DVIDIA controller code and pinned PiPER model retain MIT terms. See the repository's existing third-party notices for unrelated optional components.

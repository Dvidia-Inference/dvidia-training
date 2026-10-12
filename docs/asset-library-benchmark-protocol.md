# Asset library v0.1: native loading and passive support

This is a bounded test of original procedural shoe proxies composed with the
existing UR5e + Robotiq 2F-85, PiPER, and SO-101 models. It does **not** test shoe
grasping, sorting, a learned policy, deformation, or physical robot transfer.
SO-101 has five arm axes plus a gripper axis. The other two arms have six arm axes.

## Frozen experiment

The default protocol uses seed `612047`, three layouts, three shoe pairs
(`trainer`, `ankle-boot`, `slipper`), three robot profiles, and two controls:
**54 attempted scenes / 108 shoe instances**. Each pair shares its seeded layout
across robots and controls. This does not represent 108 independent real shoes.
Repeated geometry and authored physical assumptions are disclosed in the catalog.

`make_protocol` deterministically constructs every initial position and yaw.
`benchmark` writes `protocol.json` before it runs any attempt. That file includes
the complete catalog, catalog digest, implementation digests, optional exact Git
commit, thresholds, denominator, and every planned initial pose. A fresh output
directory is required. No attempt is silently replaced or omitted.

Each scene starts with its robot's initialized held commands and two free shoes,
with bottom heights from 0.06 to 0.10 m, x around 0.40 m, y around 0.38 / 0.56 m,
and yaw from −0.25 to +0.25 rad. Every native model compiles afresh. Each attempt
runs 1.0 simulated second, using MuJoCo 3.15.0, a 0.002 s native timestep, and
gravity `[0, 0, -9.81] m/s²`. The benchmark samples complete native joint position,
velocity, held commands, and shoe state every 0.02 s, including the initial state.

The controls differ only in the support plane's collision masks:

| Control | Expected outcome |
| --- | --- |
| `table_support` | Both shoes fall under gravity and remain supported by the table. |
| `table_disabled` | Both shoes fall under gravity, have no table support, and drop at least 0.10 m. |

The table has no hidden lower support. The robot remains present for both
controls. A robot contact, numerical failure, or unexpected control success
remains in the evidence. No shoe motion command is applied.

## Scoring

A native attempt is valid only if compilation and stepping complete, states stay
finite, no native warning occurs, both catalog masses match, each shoe has one
free joint with zero gravity compensation, gravity/timestep match the protocol,
and all 500 native steps and 51 samples are recorded.

For each shoe, gravity is checked at 0.04 s against free fall: position tolerance
0.001 m and vertical-velocity tolerance 0.015 m/s. This check happens before the
minimum-height shoe can reach the table.

Support requires **every recorded sample in the last 0.20 s** to satisfy:

- At least one actual shoe–table contact with positive normal force above
  `1e-6 N`; merely touching the table's bounding region does not count.
- Lowest collision surface within `±0.008 m` of the table.
- Linear speed at most `0.03 m/s` and angular speed at most `0.2 rad/s`.

The lowest surface is calculated from the current transforms of the native box
and ellipsoid collision primitives. Both shoes must pass for pair support. The
disabled-table control additionally requires at least 0.10 m downward travel and
zero support contacts in every recorded sample. Loaded shoe contacts with any
non-table geometry are counted separately, with the contacted geometry names.
An expected outcome additionally requires zero such contacts in every recorded
sample across the entire attempt; even a shoe that eventually settles fails that
check if it touched the robot or another shoe. Contacts are sampled every 20 ms,
so absence in these records does not prove absence between samples.
Support successes and expected
control outcomes are reported separately to avoid calling a falling control a
successful supported scene. All failures and invalid attempts stay in each
robot/control denominator.

## Runtime and evidence

The five-file bundle contains `protocol.json`, `attempts.json`, `report.json`,
`report.html`, and `integrity.json`. The attempt file contains all sampled native
positions and velocities, every per-step wall duration, compilation timing,
warnings/errors, native mass/free-joint/gravity declarations, and per-shoe scores.
The report records p50/p95/p99/max timings, simulated throughput, and process peak
RSS. Native-step latency measures only `mj_step`; it excludes state extraction,
rendering, physical sensors, transport, training, and actuator response. Run wall
time includes compilation and sampling, and ends before report serialization.
Process peak RSS includes prior allocations and is not a hardware minimum.

`inspect_benchmark` validates bounded regular files, exact file digests, linked
protocol/catalog/attempt digests, the regenerated seeded plan, attempt identities
and denominator, time/state dimensions, initial poses, constant commands, native
free-joint pose/velocity consistency, model property declarations, recomputed
scores/aggregates/timing quantiles, and the generated report presentation. It
rejects changed outcomes even when an attacker recomputes the outer file seals.
It does not rerun native physics, validate a publisher's identity, establish that
contact forces match real footwear, or prove that the supplied source commit was
the executable used. Hashes provide reproducibility and tamper evidence, not
authentication.

Development smoke runs should use separate directories and be identified as
development. Publish the frozen implementation and full confirmation bundle,
including unexpected failures. This test qualifies loading and passive support
only; future grasp/carry/place benchmarks need a separate protocol and measured
physical shoe evidence.

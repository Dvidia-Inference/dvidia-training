# Why six v0.3 PiPER placements failed

All six failures occurred during acquisition, before the contact-history carry controller had anything to recover. All alarm snapshots were table-supported; 11 of 12 probe windows remained supported throughout, with one unsupported snapshot during the remaining retry. The object moved at most **0.040–0.148 mm** across each complete episode while the first probe raised the TCP **3.428–3.849 mm**. Bilateral contact occupied 94.7–100% of each probe window; occasional single-sample contact gaps did not lead to acquisition. The retry repeated the same grip and failed again.

| Scene | First alarm (s) | Object / TCP rise at alarm (mm) | Final abort (s) | Retry overhead (s) | Episode max rise (mm) |
|---|---:|---:|---:|---:|---:|
| friction015-mass040-layout00 | 11.52 | 0.050 / 3.428 | 14.42 | 2.90 | 0.080 |
| friction015-mass040-layout01 | 11.50 | 0.052 / 3.448 | 14.40 | 2.90 | 0.080 |
| friction015-mass060-layout00 | 11.54 | 0.018 / 3.849 | 14.44 | 2.90 | 0.044 |
| friction015-mass060-layout01 | 11.50 | 0.016 / 3.435 | 14.40 | 2.90 | 0.040 |
| friction055-mass060-layout00 | 11.56 | 0.104 / 3.644 | 14.48 | 2.92 | 0.148 |
| friction055-mass060-layout01 | 11.54 | 0.088 / 3.680 | 14.46 | 2.92 | 0.142 |

The first alarm is `object_not_following` in all six cases. The second alarm is the same in five; μ=.55, 60 g, layout01 reaches `relative_slip` first. These names indicate which condition won the guard’s priority ordering, not independent ground-truth slip labels. In that last case the object is still table-supported and rises only 0.133 mm.

## Why more waiting or the same retry cannot add grip force

The upstream gripper is a position servo: `F = 40(target − q) − 5 q̇`, before its ±10 N clamp. At a stationary finger slide around 0.0196 m, the legal minimum target is already zero, so the drive supplies about −0.784 N. The coupled pair carries roughly 0.39 N per side. The actuator force cap is **not** being hit; it is the unchanged position target / gain / grasp combination that establishes this squeeze.

For approximately vertical object sides, a useful static support estimate is `μ × (Nleft + Nright)`. At μ=.15 this is about 0.118 N against weights 0.392 N (40 g) and 0.589 N (60 g). At μ=.55 it is about 0.43 N against 0.589 N for the failed 60 g cases. The .55 / 40 g successes have just enough approximate static support; .8 / 60 g also qualifies but needs carry recovery.

The compiled native contact check confirms both tangential friction coefficients equal the scene value; the object’s priority1 supplies them. First-alarm contact normals have |z| ≤0.00204, consistent with nearly vertical side contacts. This check evaluates parameters and geometry at recorded poses only. It does not substitute reconstructed normal forces for recorded values.

This is a diagnostic approximation, not a proof that every alternate grasp is impossible. It omits sub-threshold logged forces and does not reconstruct tangential wrenches. Hardware squeeze control, reorientation, support-assisted manipulation or a different grasp could alter feasibility and would need their own declared interface and benchmark.

## What v0.4 can honestly improve

1. Recognize a supported failed probe with persistently loaded pads and the jaw already closed, then stop repeating an unchanged grasp. The old retry adds **2.90–2.92 simulated seconds** before reaching the same refusal. The controller should use only observed history, not hidden mass or friction.
2. Keep the acquisition and task thresholds unchanged. A quicker refusal is a useful diagnostic result, but remains a failed placement.
3. Improve carry recovery separately, where the six successful cases demonstrate that acquisition is feasible. Remove unnecessary time under the risk condition, measure actual velocity, and avoid making “smaller requested joint delta” synonymous with slower motion.
4. Preserve false-positive and regression opportunities: same supported condition could be temporary contact settling. Test transient disturbances and ambiguous windows; retain retry for conditions where a new grasp could plausibly change contact.

All six failed history trajectories match their probe-only counterpart in every recorded qpos, qvel and command. All six unguarded baselines also fail, with maximum lift only 0.108–0.301 mm. This corroborates an acquisition limitation rather than a v0.3 carry-history regression.

`analysis.json` contains each event, both probe windows, normal-force/jaw statistics, source checksums, and the native contact-parameter check. `analyze.py` reproduces this analysis from the frozen evidence. This known v0.3 cohort is development data for v0.4; fresh confirmation must be separate.

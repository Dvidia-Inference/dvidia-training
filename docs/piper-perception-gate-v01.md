# PiPER rendered RGB-D perception gate v0.1

**Proposed and unrun.** The native PiPER capture records rendered 160×120 RGB and
metric optical-Z depth, but implements no object tracker. Its four object pose
and uncertainty signals remain missing. This proposal defines a gate for a
frozen estimator before any capture can provide perception inputs to action
learning. It reports no measured estimator accuracy and qualifies no skill or
hardware use.

## Frozen estimator contract

The estimator receives only the current captured RGB image, aligned metric-Z
depth image, the bound camera calibration, and declared rigid-object geometry
and profile. It must not read native object pose, contacts, task phase, outcome,
future frames, or privileged controller diagnostics. Freeze and hash the
implementation, dependencies, configuration, calibration, and object geometry
before confirmation evaluation.

For each decision boundary, emit the existing action-demonstration channels:

| Signal | Contract |
| --- | --- |
| `object_position_estimate_m` | 3-vector in `robot_base`, metres |
| `object_quaternion_estimate_wxyz` | Unit quaternion in `robot_base` |
| `object_position_uncertainty_m` | Nonnegative scalar bound/scale in metres, with its meaning declared |
| `object_orientation_uncertainty_rad` | Nonnegative scalar bound/scale in radians, with its meaning declared |

Each estimate must bind to the exact source frame index and retain capture,
availability, and age times. Capture time is the frame exposure time;
availability cannot precede capture or follow the decision boundary. Age is the
decision time minus exposure time. A result older than the contract's 100 ms
maximum is stale and cannot be used as a fresh learning input. Include measured
processing latency in the evaluation record. Do not forward-fill missing
estimates.

The estimator must abstain when the object is absent, occluded, out of usable
depth, too small or ambiguous, calibration is invalid, or pose uncertainty is
unbounded. Preserve an explicit missing/stale status and reason; do not replace
abstention with native truth or a fabricated zero. For symmetric geometry,
orientation is identifiable only modulo the object's symmetry group. Declare
that equivalence in scoring and report ambiguous orientation as an abstention
when the symmetry leaves task-relevant orientation unresolved.

## Gates

**Contract and leakage gate.** Test image/depth dimensions, types, alignment,
units, calibration and transform rejection; invalid and far-plane depth;
finite normalized quaternions; nonnegative finite uncertainty; source-frame
binding; timestamp ordering and the 100 ms boundary; missing, stale and
occluded-object abstention; and rejection of noncausal or malformed outputs.
Use a spy/input-boundary test to show native truth and diagnostics are never
passed into estimator code. Native truth may be joined only by a separate
offline scorer after estimates and their provenance are sealed.

**Development gate.** Use a separately assigned set of rendered poses to
debug the estimator and choose candidate acceptance thresholds. Include XY
position, height, yaw, partial occlusion, image-edge cases, and depth-invalid
regions. Register the pose groups before running the estimator. Freeze the
candidate thresholds and estimator artifacts before confirmation. Any thresholds
selected during development must be registered before held-out evaluation; no
measured millimetre accuracy is claimed by this proposal.

**Confirmation gate.** Evaluate once on held-out pose groups, keeping every
render or variant of a pose group in one split. Score position and
symmetry-aware orientation error against native truth offline, alongside
coverage/abstention, uncertainty coverage, stale/missing rate, and processing
latency. Report distributions and denominators, including failures. Apply the
frozen development thresholds without tuning. A single fixed-start run can
check data plumbing only; it cannot establish pose accuracy or generalization.
Passing this gate would qualify only this estimator on this rendered profile
and evaluated pose range, not corrective demonstrations or a learned policy.

## Known limits

The capture defaults to 160×120, where a small or partly hidden object may
occupy too few pixels for stable position or orientation. The current classic
renderer can report a CGL `ARB_clip_control` limitation; finite metric depth
does not establish depth precision. Existing metadata explicitly leaves
`depth_accuracy_verified` false. Measure pose and depth error on the held-out
rendered cases and preserve abstention; do not infer accuracy from the file
format or calibration hash. Native segmentation in
`src/dvidia_training/image_goal_scenes.py` is privileged geom-ID information and
may serve as an offline evaluator fixture only, never as estimator input.

This gate does not recover actions from Skillspace video, collect operator
corrections, train a policy, or validate a physical camera or PiPER robot.

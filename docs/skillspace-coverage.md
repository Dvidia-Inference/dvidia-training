# Collect evidence that covers a Skillspace

“Exhausting” a skill means covering a declared operating envelope and checking
its remaining gaps. It cannot mean enumerating every future object or situation.
This guide separates the implemented local planning tool from proposed public
contribution and incentive features.

## Use the local planner

Install from the current checkout, then start `dvidia-training-studio` as described
in [installation](installation.md). Open
[http://127.0.0.1:8270/coverage](http://127.0.0.1:8270/coverage). The page offers an
explicitly fabricated six-case example, a fresh-plan button, editable goals and
conditions, next-recording briefs, JSON import/export and a benchmark description.
It uses no account, GPU, external model or network after installation.

The page keeps its plan in memory. Export it to preserve work; a refresh loads the
example again. It does not upload, inspect or approve footage, run training or
execute benchmarks. The existing footage checker remains a separate step.

For the command line, the planner itself needs only Python's standard library:

```sh
.venv/bin/dvidia-skillspace-coverage example --output skillspace.coverage.json
.venv/bin/dvidia-skillspace-coverage inspect skillspace.coverage.json
```

The companion plan is separate from `skillspace.training.json`. Its exact v1
schema is illustrated in [examples/coverage-plan.json](../examples/coverage-plan.json)
and validated by `dvidia_training.coverage.validate_plan`. The command line and
the coverage endpoint accept at most 1 MiB. Training requests retain their 32 KiB limit.
No paths or executable code are resolved from this plan.

Each plan has a stable ID, revision, goal, operating envelope, selected cases and
benchmark description. Each case has a stable ID, specific capture brief,
requested outcome, group quota and priority. Unknown outcomes can be recorded,
but a planning cell requesting an unknown outcome remains unresolved.

## What counts toward a collection target

The tool counts declared evidence only when it is complete, has intended-use
rights approved, has an accepted human review for the current revision, matches
a known requested outcome and belongs to training or development evidence.
Recording, session and physical object IDs must all be present. Evidence also
binds the media SHA-256 and case ID. Proposed/AI-only reviews, stale reviews,
partial attempts, missing lineage and permission gaps remain visible as exclusions.

Media hashes and recording/session/object IDs connect source groups transitively.
Multiple cuts from one source contribute one group to a cell. A group touching
confirmation evidence or mixing training and development is excluded as a whole.
Conflicting accepted outcomes for identical media cannot be cherry-picked.
Different IDs cannot prove independence; reviewers still need original footage
and near-duplicate checks. These declarations are not authenticated by this tool.

Changing the plan creates a new revision. Earlier records retain their history;
they need review against the new requirements before they satisfy the new targets.
The planner always reports benchmark execution/qualification, acquired robot
skill and physical readiness as false. A descriptor cannot serve as a result.

## Proposed contribution experience

Use a specific mission rather than a generic upload request. For example:
“Record a complete alignment attempt with a different physical pair, starting
sideways.” Explain which coverage gap it addresses and show a short example,
capture checklist and optional practice step. Allow a contributor to choose a
different feasible mission. Do not force a five-second limit that hides an outcome.

After intake, show received, checked, reviewed and accepted/correction-needed
states. A review receipt should name the plan revision, recording hash, outcome,
permission and group decision, with a useful reason for corrections. Acceptance
and verified diversity require the future review service; they are not supplied
by this prototype. Protect private originals and make public credit opt-in.

The next mission should prioritize an unfilled critical condition or a diagnosed
development failure. Avoid a full Cartesian product of all dimensions. Choose
task-relevant interactions among object instances, start/goal configurations,
views, lighting, clutter, outcomes and robot/gripper capabilities.

This draws on DROID's task prompts, preferred-task tracking, practice mode and
periodic scene-change requests. Camera changes also require recalibration in its
workflow. [DROID collection documentation](https://droid-dataset.github.io/droid/example-workflows/data-collection.html).
LIBERO separates spatial, object and goal variations and supplies RGB,
proprioception and scene/task descriptions; those distinctions inform our case
design, not a claim of equivalent task coverage.
[LIBERO datasets](https://libero-project.github.io/datasets).

## Incentives to test

Prefer team coverage milestones and opt-in credit for accepted independent
evidence, difficult conditions, honest failures and useful review corrections.
Avoid leaderboards based on raw upload count, clip length or successes alone:
they would encourage duplicate cuts and concealed failures.

For a future funded campaign, publish the budget, eligible missions, acceptance
terms and review process before recording. A fair base payment for compliant
complete attempts plus a bounded bonus for underrepresented conditions is a
design hypothesis. An unsuccessful task attempt can still be valuable evidence.
This build creates no funded campaign, points, payment or public attribution.

RoboCrowd studied material rewards, intrinsic interest and social comparison in
in-person robot teleoperation. Its engagement and policy results make incentive
design worth testing here; they do not establish the best incentive for remote
Skillspace contributors, and mixing crowd data can affect policy quality.
[RoboCrowd paper](https://arxiv.org/html/2411.01915v2).

Test a mission-based flow against a generic upload flow with equal budgets and
participant-level assignment. Measure gaps closed, independently reviewed
groups per contributor-hour, audited label errors, complete attempts, satisfaction
and return participation. Include assigned people who never submit in engagement
denominators. No effectiveness result has been measured for DVIDIA's proposed UX.

## Enough to train, then enough to qualify

Initial quotas guide collection effort. They do not determine data sufficiency.
Evaluate learning curves across independent groups and collection orders; use
development evidence to diagnose weak conditions. Preserve frozen confirmation
groups and report success, false completion and critical errors over all attempted
trials. Every skill needs its own simulator task and compatibility envelope.

Synthetic augmentation may increase useful variation once its task adapter is
validated, but it remains related to its parent recordings. MimicGen's custom
environment interface requires object poses, subtask boundaries and action/pose
conversions; it is not an arbitrary human-video-to-action converter.
[MimicGen custom environment documentation](https://mimicgen.github.io/docs/tutorials/datagen_custom.html).

The [execution map](roadmap.md) lists the separate collection, review, learning,
benchmark and installation gates. The existing placement adapter cannot establish
shoe organization or deformable lace-tying competence.

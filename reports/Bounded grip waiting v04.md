# Bounded Grip Waiting: Progress Without an Indefinite Stall

DVIDIA's v0.4 simulation study replaces indefinite learned acquisition waiting with a bounded, explicit supervisor. Across 512 scheduled confirmation attempts, including four interrupted outcomes conservatively counted as failures, strict successes were **42 for bounded waiting, 32 for the frozen learned clock, 44 for fixed progression, and 41 for the teacher**. Neither paired comparison establishes a reliable advantage. The result is a more explicit failure contract, not a qualified robot skill or a newly trained model.

## What the fixed controller tells us

The preceding v0.3 study exposed an uncomfortable result: fixed progression achieved more strict placements than the learned timing model. One plausible explanation is that early waiting blocks the approach and centering motion needed to establish a grasp. Contact evidence and the task trajectory interact; stopping at an awkward pose can prevent the evidence a waiting policy requires. This is a mechanism to investigate, not a causal conclusion established by the earlier aggregate scores.

V0.4 tests a narrow response. Preserve the frozen arm and jaw references, permit centering before acquisition eligibility, and wait only when supported bilateral contact and tool proximity are observed. The previous learned clock still runs, but its predictions are advisory. The authored supervisor decides whether progress advances. No demonstration labels, motor model, trajectory geometry, or confirmation thresholds were refitted.

## A finite acquisition contract

The simulator observes control state every 20 ms while native MuJoCo advances at 1 ms. Acquisition requires bilateral pad force of at least 0.10 N, observed table support, a closed reference jaw, and object proximity within 0.12 m of the tool. Secure release requires bilateral force of at least 0.20 N for 100 ms and tool-relative drift no greater than 2 mm. Progress then latches forward along the original coordinated reference.

Waiting has a total budget. Exhausting it, or losing support before readiness, produces an explicit abort. The arm then holds its measured pose using the existing native command caps for the remaining recorded interval. This does not force an insecure lift or count an abort as recovery. The latch is an acquisition decision; it does not continuously certify retention after lift.

Three budgets—200, 360, and 500 ms—were declared before development. Each achieved 3/16 strict development successes with identical recorded wait totals and contact counts. The frozen tie-break selected 200 ms before confirmation. All candidates and failures remain retained.

## Retaining an interrupted experiment

Confirmation scheduled 32 fresh XY groups, two known destinations, two conditions, and four controllers: 512 attempts. The first session ended with 72 complete attempts and four started but incomplete attempts, all for the same nominal case across controllers. Recovery ran only the 436 never-started requests. It preserved the original policy, sources, selection, 72 receipts, and four partial folders byte for byte.

The resulting report contains **508 valid, scored attempts and four explicitly unevaluable interruptions**. It does not describe all 512 as valid or fabricate missing episode evidence. The primary analysis retains 128 scheduled attempts per controller and imputes the unknown outcome as failure. A sensitivity analysis excludes the entire affected XY group, leaving 31 complete groups. This recovery is a documented deviation from an uninterrupted confirmation run.

## Results and what remains unresolved

Bounded waiting produced 105 secure acquisition latches among 127 completed attempts. Its maximum recorded acquisition wait was 200 ms. There were 22 aborts: eight exhausted budgets and fourteen losses of support before readiness.

The paired bounded-minus-learned difference was +7.81 percentage points, with a Bonferroni-adjusted interval of −7.25 to +22.87 points. Bounded-minus-fixed was −1.56 points, with an interval of −12.44 to +9.32 points. Both intervals include zero; the 31-group sensitivity analysis gives the same qualitative conclusion.

The learned clock had 55 acquisition failures, compared with zero recorded for bounded or fixed progression. Bounded waiting still had 31 retention failures and 52 attempts with unintended critical contact. Fixed progression had 48 and 63 respectively. These categories overlap and are not interchangeable with final task failures. Moreover, the assigned post-grip pulse became effective in 53/64 bounded attempts, 22/64 learned-clock attempts, 64/64 fixed attempts, and 58/64 teacher attempts. Different exposure prevents interpreting the retention counts as a controlled slip-recovery advantage.

## Evidence and the next experiment

Independent read-only reconstruction audited all 48 development and 508 complete confirmation attempts: **556 audits, zero mismatches**. It checked frozen commands, sampled decisions, evidence identities, and native strict outcomes. Compact public summaries bind the full report by SHA256; they do not replace its native arrays and complete traces.

Timing remains descriptive. The two sessions ran on a shared CPU host on different dates, with four simulator workers and matching software versions. Bounded full-step p95 was 227.9 ms in the original session and 26.3 ms after resumption. That difference cannot establish an algorithmic speedup. Camera processing, rendering, communications, and physical actuation are excluded; there is no hard real-time claim.

The next experiment should isolate grasp geometry and retention after acquisition using matched intervention exposure. This study concerns one simulated rigid 40 g block, two known goals, and a narrow starting envelope. Human-video learning, shoe manipulation, other arms, and physical qualification remain unproven.

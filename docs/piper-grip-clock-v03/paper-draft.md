# Learning a grip-aware progress clock for a simulated PiPER pick-and-place task

**DVIDIA Research**

Correspondence: [hello@dvidia.org](mailto:hello@dvidia.org) · [@iammrriver](https://x.com/iammrriver)

## Abstract

A fixed reference may need to wait for grasp acquisition. This study trains a binary hold/advance clock from causal synthetic labels and compares it with a matched clock whose live and historical grip inputs are masked. Both coordinate seven frozen PiPER arm and jaw references through shared virtual progress. After development-only selection, confirmation on 32 fresh groups yielded 44/128 strict successes for full clock, 26/128 blind, 72/128 fixed timing, and 67/128 teacher. The full-versus-blind interval included zero; full clock scored below fixed timing. An independent audit reconstructed all 768 attempts with zero mismatches. This simulation does not qualify hardware.

## 1. Introduction

A fixed trajectory encodes where a robot should move, but progress need not advance at a fixed rate. During grasp acquisition, advancing too early can move the arm before the object is held reliably. A timing controller can pause a frozen reference while preserving its raw reference geometry. Retiming changes which joint targets are requested at each physical instant, and the unchanged native caps still limit issued commands.

We study this idea in a simulated PiPER pick-and-place task. At each 20 ms boundary, the policy holds virtual progress or advances it by 20 ms. The arm joint targets and original jaw reference are evaluated from their unchanged raw references at the shared virtual progress. A pause therefore retimes those targets together; it does not learn replacement geometry. Physical elapsed time and the nominal jaw reference at that time are also clock inputs.

We ask whether grip and motion history can inform timing for a known primitive in simulation. This is not camera-based grasping, new grasp planning, or a physical evaluation.

## 2. Methods

### 2.1 Task and controllers

The task moves a 40 g rigid block to one of two known destinations using seven original v0.1 references; no grasp geometry is trained. The 58-feature full clock includes physical-time context, grip state at current, 40 ms, and 120 ms lags, virtual progress, and accumulated clock debt. Blind uses the same rows and fitting recipe with live and historical grip inputs masked.

At each 20 ms boundary, a weighted binned classification tree holds or advances; ties hold. Selected full4 has depth four, nine nodes, and a 13,520-byte model; blind4 has 13 nodes and a 14,292-byte model.

The authored teacher requires bilateral contact, at least 0.20 N per pad, and at most 2 mm tool-relative drift over six 20 ms observations spanning 100 ms. Labels record its decision before command issue. Its acquisition window starts 160 ms after the first raw jaw reference below 0.01 m and ends when that reference exceeds 0.03 m. The latch then stays released; sampled readiness neither certifies continuous contact nor trains slip recovery.

### 2.2 Training and selection

Training used 16 start groups, both goals, and nominal and post-grip conditions: 64 episodes and 64,000 direct labels. Thirty-four episodes passed the strict endpoint. Every finite teacher prefix, including failures, was retained.

Four candidates crossed full or blind inputs with depths four and six. On eight development groups (32 attempts each), full4 and full6 tied at 12/32; the smaller full4 was selected before blind results. Blind4 was fixed as its matched comparator. Fixed timing scored 15/32 and the teacher 14/32.

Post-grip eligibility requires bilateral loaded pad contact at at least 0.10 N, no table support and at least 6.5 cm lift for 100 ms. An eligible case schedules a jaw-target pulse after 100 ms, lasting 40 ms with a 0.020 m floor target; native rate limits remain in force. Eligibility, activation, target effect, sustained contact loss and recovered placement are distinct measurements.

### 2.3 Frozen confirmation and endpoint

Confirmation crossed 32 fresh XY groups with two goals and two conditions: 128 attempts per controller, 512 total. All scheduled attempts remain in the denominator.

Strict success requires the v0.1 lift of 6 cm for 100 ms, carry with no contact gap reaching 40 ms, supported placement, release, settling within 25 mm for 250 ms, retreat by 10 cm, and no unintended contact. Two paired contrasts use approximate t intervals with Bonferroni simultaneous coverage. This is exploratory, not a hardware qualification study.

## 3. Results


All 512 scheduled confirmation attempts were valid. Strict success was 44/128 (34.4%) for full4, 26/128 (20.3%) for blind4, 72/128 (56.3%) for fixed timing, and 67/128 (52.3%) for the teacher. Full4 had 22/64 successes in each condition; blind4 had 13/64 in each. Fixed timing had 37/64 nominal and 35/64 post-grip successes; the teacher had 34/64 and 33/64, respectively. Every scheduled attempt remains in these denominators.

| Controller | Nominal | Assigned post-grip | Strict total | Terminal-only |
|---|---:|---:|---:|---:|
| Learned clock | 22/64 | 22/64 | 44/128 | 60/128 |
| Matched blind clock | 13/64 | 13/64 | 26/128 | 44/128 |
| Fixed timing | 37/64 | 35/64 | 72/128 | 128/128 |
| Sampled acquisition teacher | 34/64 | 33/64 | 67/128 | 120/128 |

A terminal pose that looks finished does not erase an earlier failed lift, lost grip or unintended contact.

Across 32 paired XY groups, the full-minus-blind difference was +14.1 percentage points (approximate simultaneous 95% interval −8.5 to +36.6), which does not establish a reliable advantage. Full-minus-fixed was −21.9 points (−36.8 to −7.0), favoring fixed timing under this endpoint and task distribution. The predeclared paired-benefit, nominal, critical-contact, and post-grip-exposure gates failed; validity and compute gates passed. Forty full-clock attempts failed acquisition versus none for fixed timing. The full clock made 33,418 holds at which the reconstructed sampled teacher would advance, while premature advances at teacher-hold decisions numbered four, versus 300 for blind4. This pattern is consistent with over-waiting as a concern, but the sampled-teacher comparison does not prove that any individual wait was physically unnecessary.

In post-grip cases, pulse eligibility, scheduled activation, command effect, and actual contact-loss events are separate quantities. The clock had 30 eligible, activated, and effective pulses among 64 scheduled post-grip trials, alongside 10 airborne loss events across its 128 attempts. A pulse taking effect is not itself evidence that the block slipped, and these data do not establish recovery from a real slip.

On the shared Apple M5 host, policy-call p50 was 0.937 ms, p95 1.916 ms, p99 3.947 ms and maximum 256.394 ms; full-step computation p95 was 9.594 ms (p99 15.839 ms). The maximum full-step time was 265.909 ms, with 728 of 128,000 steps above 20 ms. The compute gate passed under the declared aggregate criterion, but these tails and the shared four-worker host do not demonstrate hard real-time behavior. Timings are native simulator CPU measurements and exclude camera, transport, and physical actuation.


## 4. Audit and implementation record


The independent audit covered all 768 scheduled attempts: 64 training, 192 development, and 512 confirmation. It found zero mismatches in reconstructed labels, decisions, commands, scores, or numeric evidence. This is a reconstruction audit, not simulator authentication, a physics rerun, or hardware qualification. The sampled teacher comparison counted 33,418 full-clock holds where it would advance and four premature advances where it would hold; blind counts were 40,728 and 300, respectively. These comparisons do not prove any wait was physically unnecessary. In full-clock confirmation, 168 native-insecure 100 ms windows occur among 8,922 sampled-ready observations across 34 attempts. These overlapping observations are not independent trials: sampled readiness can conceal brief native contact loss. This reinforces the scope of six observations spanning 100 ms, rather than continuous grasp certification.

Eighty-seven earlier pilot attempts are retained as a separate lineage and are not pooled with the main study.


The Apple M5 host had ten logical CPUs and 24 GiB memory; four workers ran confirmation concurrently. Fitting full4 took 0.505 seconds after verified labels had been read; this is not end-to-end video training. No GPU was used for native simulation or fitting; the recorded viewer uses WebGL. Timing excludes camera, transport, rendering, and real-actuator latency.

## 5. Discussion

The full clock scored numerically above blind, but the simultaneous interval included no difference; it scored below fixed timing. These results do not support a general benefit claim. The false-wait diagnostic motivates testing ways to distinguish acquisition holds from delays after the sampled teacher would proceed, using a newly frozen confirmation set.

Selected full-clock inputs include current left-jaw position, 120 ms left-pad force, target-minus-object X error, and clock debt. The tree has no hard bilateral-contact guarantee and relies on privileged, uncalibrated simulator state. The task covers one block, one arm, two goals, and nominal or post-grip conditions; no transfer to arbitrary video, other arms, objects, surfaces, or hardware is demonstrated.

An installable skill should bind the model to its frozen primitive, jaw reference, simulator input contract, and strict verifier, while preserving failed attempts. Vision may provide an object or goal estimate; this study does not show that vision replaces privileged clock inputs or that arbitrary video yields a safe skill. Next work should test over-waiting and post-grip feedback in a separate simulation study. Other arms and hardware require transfer evaluations.

The artifacts accompany the [research report](https://research.dvidia.org/papers/learned-piper-grip-clock-v03/), [installable release](https://github.com/Dvidia-Inference/dvidia-training/releases/tag/piper-grip-clock-v0.3), [full-clock model](https://huggingface.co/Dvidia/piper-grip-clock-v03), [demonstration and attempt dataset](https://huggingface.co/datasets/Dvidia/piper-grip-clock-demonstrations-v03), and [static replay viewer](https://dvidia-robot-simulation-lab.static.hf.space/learned-placement/v03/viewer/index.html). These links support inspection and reproduction of this bounded simulation result; they do not expand its task scope.

## 6. Conclusion

This paper evaluates a learned binary timing controller for frozen arm and jaw references in a bounded PiPER simulation. Direct causal labels, a matched blind controller, development-only selection, and grouped fresh confirmation make its outcome auditable. The full clock did not show a reliable advantage over the blind clock and had lower strict success than fixed timing. The study provides an installable simulation artifact and evidence for further experiments; it does not establish physical grasp reliability, general video-to-skill learning, or transfer to another robot.

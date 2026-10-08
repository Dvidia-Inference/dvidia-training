---
title: DVIDIA Training
colorFrom: gray
colorTo: blue
sdk: static
app_file: index.html
license: mit
tags:
  - robotics
  - simulation
  - offline
  - cpu
  - research
---

# DVIDIA Training

A research alpha toward installable robot skills. This static Space presents published evidence from the local DVIDIA Training CPU pipeline: a synthetic visual baseline, an optional learned joint-delta candidate trained from aligned native simulation labels, and one held-out scene's recorded contact/lift/place qualification with an open-jaw control.

**Training runs on your own computer or server.** This Space provides no hosted training or uploads. The replay shows saved native physics states; it does not execute a new physics experiment.

## Published evidence

- [Example dataset](https://huggingface.co/datasets/{{HF_ORG}}/dvidia-training-examples): explicitly synthetic fixtures and native simulation evidence.
- [Pilot artifacts](https://huggingface.co/{{HF_ORG}}/dvidia-training-pilot): learned weights, evaluation reports, and qualification/control evidence.
- [Local training source and installation](https://github.com/Dvidia-Inference/dvidia-training).
- [Methods and measured limitations](https://research.dvidia.org/papers/skillspace-footage-training/).

The visual benchmark uses ten six-second synthetic moving-disc clips, with seven training, two development, and one test group. Three pipeline repeats on one Apple M5 macOS host have a median wall time of 0.5106 seconds. The same held-out recording is re-scored; these are throughput and visual prediction measurements, not independent robotics success trials or minimum hardware requirements.

The movement head is trained from 4,029 aligned numerical labels across seven native simulation recordings and evaluated on one held-out recording. The installed candidate passes native qualification in that recording's exact scene; identical joint targets with open jaws fail. Its authored phase, gripper/contact, and recovery supervisor remains in use, with privileged simulator observations and no camera-based control.

## Scope

Visual learning is not an acquired robot skill. No human-video action bridge, shoe organization policy, general scene capability, demonstration sufficiency threshold, or qualified hardware skill is established. Hashes check artifact consistency, not capture authenticity or independently measured truth.

This page is self-contained apart from its staged local banner at `assets/branding/github-header.png` and standalone recorded replay at `replay.html`. It loads no third-party scripts, fonts, or CDNs, and contains no server or credential. Source code and published synthetic artifacts use the MIT license; see the source repository and artifact cards for scope and provenance.

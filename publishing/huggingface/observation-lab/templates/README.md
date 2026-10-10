---
title: DVIDIA Observation Lab
emoji: 🔎
colorFrom: gray
colorTo: blue
sdk: static
app_file: index.html
license: mit
tags:
  - robotics
  - annotation
  - offline
  - research
  - synthetic
---

# DVIDIA Observation Lab

An interactive **synthetic review-workflow demo** of the open-source local
[DVIDIA Observation Lab](https://github.com/Dvidia-Inference/dvidia-training/blob/observation-lab-v0.1.0/docs/observations.md).
Play a schematic clip, inspect timestamped image boxes, edit labels/boundaries,
approve or reject proposals, add a missed event, save a browser-local revision,
and download its JSON.

**The two videos and annotations are authored geometric fixtures. No object
detector, VLM, trained DVIDIA model, live tracking or robot policy runs here.**
The occlusion example deliberately omits blue-shape annotations behind a panel.
These clips demonstrate the interface, not annotation accuracy or task success.

## Your reviews stay in your browser

This static Space has no upload endpoint, backend review store, account or model
download. Review revisions stay in this browser’s storage and never go to DVIDIA
or Hugging Face as application requests. If browser storage or cross-tab locking is unavailable,
the page explicitly uses temporary memory; download your saved review before
leaving. Reset removes only this demo’s local revisions. The host still receives
ordinary page/media requests under its own policies.

The synthetic originals are read-only. Source/annotation hashes and revision
parents detect inconsistent local history; they do not authenticate a reviewer
or independently establish annotation correctness. An accepted event does not
qualify a robot skill.

## Install the actual local detector tool

The actual tool uses locally provisioned, hash-checked **OpenCV Zoo YOLOX and
MediaPipe palm ONNX detectors**. It runs locally after setup, samples at two
frames/second, associates image boxes, proposes image-overlap relations and
records append-only review revisions. It has no VLM or calibrated 3D/force output.

With Python 3.12+ and FFmpeg/ffprobe installed, check out the versioned release:

```sh
git clone --branch observation-lab-v0.1.0 https://github.com/Dvidia-Inference/dvidia-training.git
cd dvidia-training
python3.12 -m venv .venv
.venv/bin/python -m pip install '.[observations]'
.venv/bin/python scripts/download_observation_models.py --download --output runs/observation-models
.venv/bin/dvidia-observe analyze /path/to/source.json --models runs/observation-models --output runs/observations-001
.venv/bin/dvidia-observation-studio runs/observations-001 --port 8280
```

The explicit provisioning step needs network access and downloads about 40 MB of
upstream weights. Inference makes no model downloads. Keep permitted footage
local and provide the documented provenance/permission manifest. The local
server binds only to 127.0.0.1; its local filesystem interface is not a public
upload service. No GPU is required for this CPU detector path, but measured
minimum hardware and other-platform throughput have not been established.

- [Setup and input contract](https://github.com/Dvidia-Inference/dvidia-training/blob/observation-lab-v0.1.0/docs/observations.md)
- [Actual feasibility report](https://github.com/Dvidia-Inference/dvidia-training/blob/observation-lab-v0.1.0/benchmarks/observation-20261010/README.md)
- [Model artifacts and licenses](https://github.com/Dvidia-Inference/dvidia-training/blob/observation-lab-v0.1.0/docs/observation-models.md)
- [Planned independent accuracy/correction-effort study](https://github.com/Dvidia-Inference/dvidia-training/blob/observation-lab-v0.1.0/docs/observation-pilot-plan.md)
- [DVIDIA research](https://research.dvidia.org/)
- [DVIDIA Hugging Face organization](https://huggingface.co/{{HF_ORG}})

## Reproduce and inspect this demo

The Space contains the original MIT review page at `source/observation_studio.html`,
its clearly labeled demo derivative at `index.html`, a browser-only fixture API,
and generated synthetic videos/JSON. `provenance.json` records explicit source
hashes; `demo/provenance.json` records the drawing recipe, encoder version and
media hashes. The public repository's separate preparation script stages only
these allowlisted files and creates the complete publication inventory.

`python3 generate_demo.py --output new-demo` regenerates the schematic clips
with local FFmpeg/libx264. The geometry is deterministic; encoded file hashes
depend on the FFmpeg build. Serving this directory over HTTP is sufficient to
run the demo. No runtime Python dependencies, Docker or GPU are needed.

Own demo source and authored fixtures are MIT. The separate actual detector
adapter/weights retain upstream Apache 2.0 licenses and attribution, included here
under `licenses/` for reference; no upstream code or weights are bundled in this
Space. See `LICENSE` and `THIRD_PARTY.md`.

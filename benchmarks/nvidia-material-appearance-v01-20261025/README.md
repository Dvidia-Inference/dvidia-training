# NVIDIA fabric appearance pilot — 2026-10-25

Five real diffuse maps from NVIDIA PhysicalAI-SimReady-Materials v0.2.0 are
adapted offline to native MuJoCo RGB textures. This is separate from the newer
[SimReady Central collection](https://huggingface.co/blog/nvidia/simready-central-assets-robotics-ai-factories).
It does not import arbitrary USD geometry, train a motor model, or measure fabric contact.

[Complete report](evidence/report.html) · [Results](evidence/report.json) ·
[Frozen protocol](evidence/protocol.json) · [40-file inventory](evidence/integrity.json) ·
[Integration guide](../../docs/nvidia-simready-integration-v01.md)

## Results

All **6/6** scheduled cases completed: the original two-shoe PiPER scene plus
Denim, Cotton White, Honeycomb Green, Canvas and Safety Pink. All **6/6** compiled
mechanics comparisons matched; all **6/6** sampled native settling trajectories
matched exactly over two simulated seconds. All five imported appearances changed
rendered pixels. Each adds 196,608 native texture bytes at 256×256 RGB. The six
cases took **2.843 seconds** total wall time on the recorded host; this includes
preparation and rendering but excludes the archive download and initial archive
verification. It is a small local timing sample, not a hardware minimum.

The mechanics check explicitly covers mass/inertia, transforms, collision masks,
geometry, friction, solver/contact parameters, joint settings, actuation arrays,
gravity, timestep and mechanical dimensions. The exact list is in the protocol.
Trajectory samples contain qpos/qvel/contact counts at 0, .5, 1, 1.5 and 2 seconds.
This is a settling check of authored rigid shoe proxies, **not a manipulation test**.

A separate single-image native cube diagnostic kept physical pose identical.
The existing green-only observer saw the original green cube and Honeycomb Green,
but returned `unknown/no_green_candidate` for the other four fabrics. Thus appearance
variation immediately reveals a known perception restriction. It does not show a
trained detector improved, or establish identity, pose accuracy or temporal robustness.

## Source and reproduction

Only the 45,574,867-byte Fabric archive was downloaded, not the full USD collection:

- [v0.2.0 release](https://github.com/NVIDIA-Omniverse/PhysicalAI-SimReady-Materials/releases/tag/v0.2.0)
- Archive SHA-256: `a68ca7376ed4abd6160ff7fed234cd1012bded73fd17f1588adcd34a91fe7d42`
- Thirteen fabric choices in that archive; selected source USDA/JPEG hashes and derivative RGB/PNG hashes are recorded.
- NVIDIA derivatives use [MIT-0](LICENSE-NVIDIA-MIT-0.md). DVIDIA code uses the repository license; that does not relicense other Central assets.

From the project root, with Python 3.12+, NumPy 2.5.3, MuJoCo 3.15.0 and FFmpeg:

```sh
PYTHONPATH=src python -m dvidia_training.material_appearance_benchmark \
  --archive PATH/Fabric.zip --assets PATH/PINNED_ROBOT_ASSETS \
  --output NEW_EVIDENCE_DIRECTORY
PYTHONPATH=src python -c 'from dvidia_training.material_appearance_benchmark import inspect_run; print(inspect_run("benchmarks/nvidia-material-appearance-v01-20261025/evidence"))'
```

The inspector verifies retained bytes, source bindings and denominators, without
rerunning physics. Rendering needs a supported native graphics context. The
standard CPU simulator path has no Isaac Sim, CUDA, USD-parser or cloud requirement.
This adapter uses diffuse images only; full OpenPBR/MaterialX shading, sensor
response, calibrated friction/compliance and browser texture transport are future work.

## SimReady Central metadata attribution

The separate [Central metadata receipts](central-metadata/FETCH-RECEIPT-RETRY.md)
retain eight public JSON/USDA responses (44,471 bytes total) from NVIDIA's
`bottle_laboratory_a01` and `box_cardboard_a01` packages at dataset revision
`653d91e182011f357cf1644013cc94ddf194d349`. Exact original URLs and SHA-256 values
are in the receipt. These records are reproduced verbatim from
[NVIDIA SimReady Assets](https://huggingface.co/datasets/nvidia/simready-assets),
credited to NVIDIA under the package-declared
[Creative Commons Attribution 4.0 license](https://creativecommons.org/licenses/by/4.0/).
The fetch commentary is by DVIDIA. This attribution is separate from the MIT-0
fabric derivatives. No box/bottle mesh or texture package was downloaded, imported,
or validated locally; their stored mixed validation results remain visible.

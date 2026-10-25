# NVIDIA SimReady assets in the MuJoCo asset library v0.1

This integration adds a small, offline appearance path for NVIDIA’s PhysicalAI-SimReady-Materials library while retaining the project’s MuJoCo CPU runtime and browser workstation. It is separate from the newer SimReady Central aggregation of simulation assets, and from the video-to-skill bridge and motor-learning work.

## Why start with appearance

NVIDIA’s material repository contains 1,705 USD materials authored with `UsdShade`, MaterialX, and OpenPBR 1.1, with JPG texture maps and SimReady sensor metadata. The v0.2.0 release, dated June 12, 2026, added Fabric and Skin categories. These materials can supply visual variation for shoe uppers, floors, and held-out appearance splits. This is distinct from [SimReady Central](https://huggingface.co/blog/nvidia/simready-central-assets-robotics-ai-factories), announced October 21, 2026: an aggregation of more than 2,000 OpenUSD assets from multiple suppliers, including arms, grippers, warehouse equipment, and factory infrastructure. Central is an asset-discovery and distribution source; the material library is the focused texture source used by the current adapter.

Appearance variety can test whether visual observations and image-based task diagnostics are brittle to color and texture. It cannot establish the real contact behavior of denim, leather, rubber, or another substrate. In particular, texture, semantic labels, or a shader’s roughness do not measure shoe–gripper friction, sole compliance, damping, mass, or grasp reliability. Keep appearance randomization separate from physical randomization and evidence, as required by the [asset-library contract](robot-simulation-library.md).

## Implemented material slice

[`material_appearance.py`](../src/dvidia_training/material_appearance.py) currently supports only the verified Fabric category archive from release v0.2.0. Its pinned archive is 45,574,867 bytes with SHA-256 `a68ca7376ed4abd6160ff7fed234cd1012bded73fd17f1588adcd34a91fe7d42`. The adapter checks the local archive hash, bounds ZIP paths and expansion, reads one declared base-color JPEG from a selected USDA material, and uses FFmpeg to make a small RGB texture and PNG. It records source, material, texture, and output hashes in `appearance.json`.

The adapter does not evaluate or execute USDA, MaterialX, or OpenPBR shader graphs. It intentionally uses only the diffuse/base-color image; normal, roughness, fuzz, weight, sensor, and other channels are not imported. `apply_to_spec` binds that image as an RGB texture to explicitly named MuJoCo geoms. It changes no body mass, inertia, collision geometry, friction, restitution, stiffness, or damping.

Use the local verified archive and a fresh output directory:

```sh
python -m dvidia_training.material_appearance catalog PATH/PhysicalAI-SimReady-Materials-Fabric.zip
python -m dvidia_training.material_appearance prepare PATH/PhysicalAI-SimReady-Materials-Fabric.zip \
  --material Fabric_Denim --output ./prepared-denim --size 256
python -m dvidia_training.material_appearance inspect ./prepared-denim
```

The intended scene API is:

```python
model = asset_library.build_scene(
    "piper", assets,
    shoe_ids=["trainer-left", "trainer-right"],
    appearance="./prepared-denim",
)
```

Appearance is opt-in; no appearance package means the existing catalog scene. `build_scene` applies it to upper geoms only, leaving the sole and its collision/contact settings under the original catalog material/physics. The native compiled scene remains the source for browser data; the browser does not load USD or run USD shader graphs. Carrying the generated texture and provenance through the asset-library preview and live workstation manifests is a follow-up integration, not part of this adapter.

## USD and future geometry imports

MuJoCo 3.15 documents USD import through the `<model file="…" content_type="text/usd"/>` decoder path, and marks OpenUSD support experimental. Enabling it requires a MuJoCo build with OpenUSD support; it is not a requirement for this appearance adapter or the default project runtime. MuJoCo’s documented USD exporter is for MuJoCo-to-USD workflows. [MuJoCo USD importing](https://mujoco.readthedocs.io/en/stable/OpenUSD/importing.html), [build requirements](https://mujoco.readthedocs.io/en/stable/OpenUSD/building.html), [exporting](https://mujoco.readthedocs.io/en/3.15.0/OpenUSD/exporting.html)

A later geometry adapter can assess these props without an Isaac Sim runtime. Validate referenced files, units/axis, transforms, visual and collision meshes, rigid-body membership, mass/inertia, and contact settings against the compiled MuJoCo model. Preserve USD provenance and conversion settings; successful parsing does not prove semantics were retained.

## Central discovery and candidate assets

Central aggregates multiple suppliers and package licenses. The NVIDIA submission list pins assets to dataset revision `653d91e182011f357cf1644013cc94ddf194d349`; profile and validation are package-specific. The indexing workflow can create `library.json` with source, owner, version, license, validation, benchmarks, and missing files, but requires downloading packages. Do not mirror the 472-GB [dataset](https://huggingface.co/datasets/nvidia/simready-assets).

USD Search documents `POST /search_hybrid` for keyword/vector search and filters including USD properties, file type, and bounding-box dimensions; check hosted authentication and indexed fields at use time. Search ranking is discovery assistance, not validation evidence. [Central submission catalog](https://github.com/NVIDIA-Omniverse/simready-central/blob/main/submissions/nvidia/simready-assets), [USD Search API](https://nvidia-omniverse.github.io/usd-search/)

Two package definitions were checked from the pinned list; [receipts](../benchmarks/nvidia-material-appearance-v01-20261025/central-metadata/FETCH-RECEIPT-RETRY.md) record exact URLs and content hashes. Both declare CC-BY-4.0, `metersPerUnit=1`, `upAxis="Z"`, validation profile `Robotics-Prop` 3.1.0, and asset profile `Prop-Robotics-Neutral`. Their Physics layers include rigid-body APIs and MuJoCo-enabled variants, but this does not establish that the current MuJoCo build imports them correctly. The BOM enumerates package file dependencies; validation metadata records feature dependencies, including FET003 PhysX on FET003 neutral and FET004 PhysX on FET003 PhysX plus FET004 neutral.

The [cardboard box Physics layer](https://huggingface.co/datasets/nvidia/simready-assets/resolve/653d91e182011f357cf1644013cc94ddf194d349/box_cardboard_a01/simready_usd/payloads/Physics/physics.usda) authors density 689, friction 0.7, restitution 0.4. Its 30-file BOM totals 21.5 MB; recorded validation is mixed: FET000/`HI.009`, FET001/`AA.001`, FET004 neutral/PhysX/`RB.MB.001`, and FET006 MDL/`VM.MDL.001` fail in the Jan 13 snapshot; FET003 neutral/PhysX pass; FET005 status differs across Jan 13 and Jan 16. The [laboratory bottle Physics layer](https://huggingface.co/datasets/nvidia/simready-assets/resolve/653d91e182011f357cf1644013cc94ddf194d349/bottle_laboratory_a01/simready_usd/payloads/Physics/physics.usda) has three rigid bodies, fixed joints, plastic density 950 and glass density 2500, friction 0.4, restitution 0.4/0.05; its 39-file BOM totals 272.8 MB. Bottle validation passes FET000, FET003, FET004, FET005, but fails FET001/`AA.001` and FET006/`VM.MDL.001`. BOM totals include textures and payloads, not fetched bytes. Start with the smaller box for offline import inspection; authored contact values and profile checks are not calibrated grip measurements.

## Validation status

The [frozen six-case pilot](../benchmarks/nvidia-material-appearance-v01-20261025/README.md) completed in 2.843 seconds on the recorded host, excluding archive download and initial verification. It compares original appearance plus five fabrics. All six compiled mechanics comparisons and sampled two-second native settling trajectories matched exactly. The separate static cube diagnostic exposed the green detector's restriction: it observed the original and green fabric, but returned unknown for four other fabrics. No model was retrained; no manipulation improvement is claimed.

| Check | Status |
| --- | --- |
| Pinned archive verification and bounded material preparation | Verified; 13 source choices, 5 prepared variants |
| Selected upper appearance changes while compiled physics remains invariant | 6/6 mechanics and trajectory matches |
| Green detector sensitivity to Fabric appearance variant | 4/5 imported variants unobservable |
| Browser package carries local texture and matching provenance | Pending |

## Licensing and distribution

The material repository declares MIT-0 for its materials and textures; retain the license and exact release/source hashes in derived packages. That license does not cover unrelated Central packages. Central’s submission manifests pin package versions and identify per-package licensing; inspect the specific license terms before downloading, modifying, or redistributing an asset. [Material library and license](https://github.com/NVIDIA-Omniverse/PhysicalAI-SimReady-Materials)

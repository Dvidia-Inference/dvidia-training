# USD workstation v0.1

A bounded, local OpenUSD-to-MuJoCo bridge for original rigid test geometry and a pinned PiPER arm. The USD SDK parses the scene; native simulation checks that its geometry, mass, centred inertia, gravity and friction reach the compiled model. The arm keeps its separately verified MJCF mechanics. This does not import USD robot articulations or learn a manipulation policy.

## Install and run

From this source revision, with Python 3.12 or newer:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[usd]'
dvidia-scene validate examples/usd-workstation-v01/recipe.json
```

The optional USD dependency is `usd-core==26.11`; native simulation uses `mujoco==3.15.0` and `numpy==2.5.3`. No dedicated GPU is used for these headless physics checks. The recorded browser view needs WebGL2. Minimum RAM, other operating systems and hard real-time performance have not been qualified.

Fetch the exact allowlisted robot files through the existing robot tool, or reuse verified local assets:

```sh
dvidia-robot-sim fetch piper --output ./robot-assets
dvidia-scene build examples/usd-workstation-v01/recipe.json --assets ./robot-assets --output ./scene-check
python -m dvidia_training.usd_scene_benchmark prepare --example examples/usd-workstation-v01/recipe.json --output runs/my-usd-check
python -m dvidia_training.usd_scene_benchmark run --output runs/my-usd-check --assets ./robot-assets
```

`build` emits a compiled-property receipt. The benchmark retains each authored USD input, recipe, trajectory, result and recorded native body poses. It refuses to overwrite or automatically retry started attempts. The read-only visual replay is exported separately; running it does not train a policy or connect to hardware.

The maintainer tested an isolated OpenUSD wheel together with an existing pinned native Python environment through a local `.pth` file. A fresh full dependency install initially stopped on a package checksum mismatch; verification was not disabled. A separate fresh OpenUSD-only download verified and installed. This is a tested runtime record, not a clean-install or minimum-hardware certification. OpenUSD emits an Apple cache-line warning on the tested host; native simulator warnings are recorded separately.

## Supported scene contract

Version `dvidia.usd-workstation-recipe/v1` binds:

- One adjacent self-contained `workstation.usda` by SHA-256.
- Explicit metres, kilograms, seconds and right-handed Z-up coordinates.
- A static horizontal tabletop and one dynamic rigid cube.
- Explicit positive mass, centred diagonal inertia, pose and collision geometry.
- Equal authored static/dynamic friction, mapped to native isotropic sliding friction.
- Authored zero restitution, with declared MuJoCo contact solver assumptions.
- The exact PiPER asset catalog and robot profile lock.
- Seeded XY/yaw reset variation and independent position, velocity, contact and settling predicates.

External layers, payloads, references, other shapes, animation, custom schemas, deformable materials and USD articulation import are unsupported. Unsupported content fails before scene composition. This narrow contract is a first interoperability test; it cannot load arbitrary SimReady packages.

## What the benchmark measures

The predeclared roster contains 45 passive physics cases: 27 drops across three masses, three friction values and three heights; nine sliding cases; and nine negative controls that remove support, gravity or friction. It records every case without outcome filtering. Sliding requires the prescribed initial velocity, nonzero travel, stopping and agreement with a Coulomb stopping-distance calculation. A stationary object cannot pass as a successful slide.

Independent task-goal checks are separate. A slide may correctly stop outside the recipe's starting XY goal. Passing the passive physics checks does not establish successful manipulation, calibrated shoe mechanics or transfer to a physical arm. Physics timings include shared-host load and exclude camera, transport, asset download and real actuators.

## Role in DVIDIA

The intended chain is video evidence → scene and goal estimate → versioned environment recipe → varied simulation rollouts → trained policy → held-out skill benchmark → robot-specific installation. This release establishes a bounded scene conversion and validation step. Video-to-action estimation, feedback-aware policy learning, cross-arm retargeting and real-robot validation remain separate work.

NVIDIA's [OpenUSD curriculum](https://docs.nvidia.com/learn-openusd/latest/index.html) describes reusable scene composition and asset parameterization. [USD Physics](https://openusd.org/dev/api/usd_physics_page_front.html) specifies physical properties. MuJoCo's native [USD import](https://mujoco.readthedocs.io/en/stable/OpenUSD/importing.html) is experimental and requires a USD-enabled build; this bridge uses the Python USD SDK and explicit supported-property conversion instead.

Original scene and bridge: MIT. Pinned PiPER assets: MIT, with upstream notices preserved in visual exports. OpenUSD and other installed dependencies retain their own licenses. DVIDIA research · [@iammrriver](https://x.com/iammrriver) · hello@dvidia.org.

## Separate clean-install follow-up, 30 October

After the frozen kit was packaged, an isolated Python 3.12.9 environment completed a full install using pip 26.2.2 with bounded download resumption. Truncated downloads were rejected by hash checks; checksum verification stayed enabled. Installed modules matched the frozen bridge and benchmark source hashes. From outside the checkout, CLI validation/build passed and a separate 1,200-tick passive smoke reached its settling goal with zero native warnings. No existing native environment was injected. The smoke reused verified local PiPER assets; fresh asset fetching, other operating systems, hardware minima and learned skill transfer remain unqualified. [Machine-readable smoke record](../benchmarks/usd-workstation-v01/clean-install-smoke.json).

# SimReady Central metadata fetch receipt — escalated retry

Date: 2026-10-25
Transport: public HTTPS with `curl --fail --location --max-filesize 524288`; no credentials. Every response below was JSON or USDA text and <=512 KiB. No mesh USD, texture, or full package was downloaded. Attempt 1 remains recorded separately in `FETCH-RECEIPT.md` (DNS failure).

## Retrieved metadata

| Asset / file | Requested pinned URL | HTTP | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| Bottle package definition | https://huggingface.co/datasets/nvidia/simready-assets/resolve/653d91e182011f357cf1644013cc94ddf194d349/bottle_laboratory_a01/com.nvidia.simready.packaging.json | 200 | 566 | `b0e91df9ff7f2c2e9437d7796eccaff26ca1f7f1128bb21308d242d27c5f1546` |
| Bottle BOM (definition-referenced) | https://huggingface.co/datasets/nvidia/simready-assets/resolve/653d91e182011f357cf1644013cc94ddf194d349/bottle_laboratory_a01/.metadata/com.nvidia.simready.packaging.bom.json | 200 | 10,150 | `3db069ce2015a54cc886fffc08a47962452468a8a3374a84600b1f0d83e5223c` |
| Bottle Physics layer (listed in BOM) | https://huggingface.co/datasets/nvidia/simready-assets/resolve/653d91e182011f357cf1644013cc94ddf194d349/bottle_laboratory_a01/simready_usd/payloads/Physics/physics.usda | 200 | 7,099 | `9b72c8fda2cf51ac53ca3a62e688c207c7d200d316e28798bec8e2808be7e0c9` |
| Bottle MuJoCo variant (listed in BOM) | https://huggingface.co/datasets/nvidia/simready-assets/resolve/653d91e182011f357cf1644013cc94ddf194d349/bottle_laboratory_a01/simready_usd/payloads/MuJoCo/enabled.usda | 200 | 5,678 | `ac74cc2348f1fdba10f2d9931fe357b14fdb1016326ea0b9931098d3caf70070` |
| Box package definition | https://huggingface.co/datasets/nvidia/simready-assets/resolve/653d91e182011f357cf1644013cc94ddf194d349/box_cardboard_a01/com.nvidia.simready.packaging.json | 200 | 562 | `04564849ff24749775961a514a895909af6bc32113d185d359a20000d8c341d4` |
| Box BOM (definition-referenced) | https://huggingface.co/datasets/nvidia/simready-assets/resolve/653d91e182011f357cf1644013cc94ddf194d349/box_cardboard_a01/.metadata/com.nvidia.simready.packaging.bom.json | 200 | 7,116 | `2e84f1414884ec776cb8fa67021688b87c1ab79a3b0083ce5eba3abb3679fecc` |
| Box Physics layer (listed in BOM) | https://huggingface.co/datasets/nvidia/simready-assets/resolve/653d91e182011f357cf1644013cc94ddf194d349/box_cardboard_a01/simready_usd/payloads/Physics/physics.usda | 200 | 6,602 | `1347b8787d442807223561abe360ba4b2e4fe79340178070791e347e3b09c558` |
| Box MuJoCo variant (listed in BOM) | https://huggingface.co/datasets/nvidia/simready-assets/resolve/653d91e182011f357cf1644013cc94ddf194d349/box_cardboard_a01/simready_usd/payloads/MuJoCo/enabled.usda | 200 | 6,698 | `ed04adf5dd898c2af0ab523857a88ec16715879880186770e95827e4304b1815` |

## Findings from fetched text

Both package definitions declare `CC-BY-4.0`, pinned `package_id` versions, and a hash reference to their BOM. Bottle package ID is `simready.hf.nvidia.simready-assets.bottle_laboratory_a01.2026.10.10_005704182302`; box is `simready.hf.nvidia.simready-assets.box_cardboard_a01.2026.10.08_223649139513`.

Both Physics USDA layers identify asset type `prop`, `Robotics-Prop` validation profile version `3.1.0`, and root asset profile `Prop-Robotics-Neutral`; both declare `metersPerUnit = 1`, `upAxis = "Z"`, and an enabled MuJoCo physics variant. Bottle validation stamped 2026-04-01 passes FET000_CORE, FET003 neutral/PhysX, FET004 neutral/PhysX, and FET005 neutral; FET001_BASE_NEUTRAL fails AA.001 and FET006_BASE_MDL fails VM.MDL.001. Its layer marks three geometry prims rigid-body and has fixed joints for the bottle part and cap. Plastic PhysicsMaterials author density 950, static/dynamic friction 0.4, restitution 0.4; clear glass authors density 2500, static/dynamic friction 0.4, restitution 0.05.

Box validation records dated 2026-01-13 and 2026-01-16. The Jan 13 record passes FET003 neutral/PhysX and FET005 neutral; it fails FET000_CORE (HI.009), FET001_BASE_NEUTRAL (AA.001), FET004 neutral/PhysX (RB.MB.001), and FET006_BASE_MDL (VM.MDL.001). The Jan 16 record again passes FET003 neutral/PhysX and fails FET004 neutral/PhysX; FET005 neutral is marked failed without a listed requirement. Its Physics layer marks the geometry prim rigid-body and authors cardboard density 689, friction 0.7 static/dynamic, restitution 0.4. Both package BOMs list `instances_mujoco.usda` and MuJoCo enabled/disabled layers, but this is authored data, not proof of correct import into this project.

The BOM inventory totals 30 listed files / 21,502,613 declared bytes for box and 39 / 272,829,972 for bottle. These sums include textures and payloads; neither full package was fetched. Box is the smaller next-prototype candidate, subject to local profile import/collision review. Neither authored density/friction nor validation establishes empirical contact calibration.

All content files in this receipt are the exact bytes received from their canonical pinned URLs. Package JSON `metadata[0].hash.sha256` matches the locally fetched BOM digest for both assets. These records are provenance evidence only; they do not grant rights beyond each package's CC-BY-4.0 terms.

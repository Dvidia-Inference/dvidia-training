# SimReady Central metadata fetch receipt

Date: 2026-10-25
Transport: `curl --fail --location --max-filesize 524288` (public HTTPS, no credentials)
Limit: 512 KiB per response; JSON/metadata text only. No asset geometry or full dataset requested.

The pinned package-definition URLs below were attempted. Both requests failed before receiving an HTTP response because this runtime could not resolve `huggingface.co` (`curl: (6) Could not resolve host`). Therefore no response bytes, server-side hash, package manifest, profile, license, dependency record, or metadata sidecar was obtained; failed output files were not written.

| Candidate | Exact pinned URL | HTTP | Downloaded bytes | Result |
| --- | --- | --- | ---: | --- |
| `bottle_laboratory_a01` | https://huggingface.co/datasets/nvidia/simready-assets/resolve/653d91e182011f357cf1644013cc94ddf194d349/bottle_laboratory_a01/com.nvidia.simready.packaging.json | 000 | 0 | DNS resolution failed |
| `box_cardboard_a01` | https://huggingface.co/datasets/nvidia/simready-assets/resolve/653d91e182011f357cf1644013cc94ddf194d349/box_cardboard_a01/com.nvidia.simready.packaging.json | 000 | 0 | DNS resolution failed |

No substitute metadata was inferred. The exact per-package profiles, capabilities, license terms, dependency/BOM records, and simulation suitability remain unverified.

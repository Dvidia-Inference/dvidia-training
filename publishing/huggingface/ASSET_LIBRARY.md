# Add the first shoe asset library to Robot Simulation Lab

This workflow uses the existing fixed-destination publisher for the public static
`Dvidia/robot-simulation-lab` Space. It preserves every previous benchmark file,
adds `asset-library/v01/`, and adds library links to the lab landing page/card.
It retains the existing 512 MiB stored/32 MiB per-file limits, with at most 700
files in the complete Space and 100 files/96 MiB in the new library.

1. Freeze the native source/protocol, run `dvidia-assets benchmark`, and inspect
   the complete result. Export and inspect one preview per arm with all six shoes.
2. Run `prepare_asset_library.py bundle --benchmark BENCH --ur5e UR_PREVIEW
   --piper PIPER_PREVIEW --so101 SO_PREVIEW --source-commit FULL_SHA --output LIBRARY`.
   The bundle includes exact native receipts, previews, original model notices,
   catalog, local installation guide and a complete file seal.
3. Run `prepare_asset_library.py stage --base-plan EXISTING_PLAN --library LIBRARY
   --source-commit FULL_SHA --expected-remote-commit EXACT_SPACE_SHA --output STAGING`.
   Both the base plan and new library must pass their inspectors. This first-release
   operation refuses to replace an already present library namespace.
4. Review the staged landing page on desktop/mobile, run publication tests, then
   `publish_robot_sim.py check STAGING/publication-plan.json` and
   `publish_robot_sim.py publish STAGING/publication-plan.json` using the existing
   credential adapter. Never put a token in a command or public artifact.
5. Verify the immutable public commit and live links. A network interruption after
   a remote commit requires read-only recovery/verification before another write;
   do not blindly publish again.

The bundle inspector establishes portable consistency and bounded contents, not
physical accuracy or publisher authenticity. Native numerical evidence comes from
the original benchmark run. Real shoe contact, deformation, policy learning and
hardware qualification remain separate work.

When a presentation-only fix follows the frozen native run, pass the corrected
source as `--source-commit` and the original runtime as `--benchmark-source-commit`
to `bundle`. Both revisions remain explicit. The native implementation hashes
must still match; a physics change requires a new benchmark.

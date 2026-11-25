#!/usr/bin/env python3
"""Survey the PiPER camera before any task run, using a separate fixture scene."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import time

import numpy as np

from dvidia_training import piper_workspace_calibration as survey
from dvidia_training import reference_sim as sim

PROFILES = {
    "nominal": ((.30, 0., .12), 1., 135., -45.),
    "relocated": ((.32, -.035, .11), .97, 145., -41.),
    "left": ((.28, .016, .11), .95, 120., -47.),
    "right": ((.29, -.027, .14), 1.04, 151., -51.),
    "lower": ((.32, .01, .11), 1.02, 130., -37.),
    "higher": ((.31, -.018, .13), .97, 142., -56.),
}


def fixture_xml(shifted_markers=()):
    geoms = []
    for name, colour, (x, y, top) in survey.MARKERS:
        if name in shifted_markers:
            x += .012
        rgb = " ".join(str(channel) for channel in colour)
        geoms.append(f'<geom name="{name}" type="box" pos="{x} {y} {top-.0002}" '
                     f'size=".022 .022 .0002" rgba="{rgb} 1"/>')
    return ('<mujoco><visual><global offwidth="320" offheight="240" fovy="45" ipd="0"/>'
            '<headlight ambient="1 1 1" diffuse="0 0 0" specular="0 0 0"/></visual>'
            '<worldbody><geom type="plane" size=".8 .8 .01" rgba=".2 .2 .2 1"/>'
            + "".join(geoms) + '</worldbody></mujoco>')


def capture(profile, output, *, shifted_markers=(), label=None):
    start = time.perf_counter()
    mj, _ = sim._native_imports()
    from mujoco.rendering.classic.renderer import Renderer
    xml = fixture_xml(shifted_markers)
    capture_id = label or profile
    model = mj.MjModel.from_xml_string(xml)
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    camera = mj.MjvCamera()
    camera.lookat[:], camera.distance, camera.azimuth, camera.elevation = PROFILES[profile]
    warnings = []
    with sim.native_warnings(warnings):
        with Renderer(model, survey.HEIGHT, survey.WIDTH) as renderer:
            if getattr(renderer, "_gl_context", None) is None:
                raise RuntimeError("Missing classic rendering context.")
            renderer.update_scene(data, camera=camera)
            rgb = renderer.render()
            renderer.enable_depth_rendering()
            depth = renderer.render()
    try:
        result = survey.make_survey(rgb, depth, camera_profile=profile)
    except ValueError as error:
        with (output/f"{capture_id}-frame.npz").open("xb") as frame:
            np.savez_compressed(frame, rgb=rgb, depth=depth)
        refusal = dict(camera_profile=profile, error=str(error), renderer_warnings=warnings,
                       fixture_xml_sha256=sha256(xml.encode()).hexdigest())
        if isinstance(error, survey.SurveyMismatch):
            refusal.update(fit_error_mm=error.fit_mm, check_error_mm=error.check_mm)
        write(output/f"{capture_id}-refused.json", refusal)
        raise
    result.update(acquire_and_fit_wall_seconds=time.perf_counter()-start,
                  renderer_warnings=warnings,
                  runtime=dict(python=platform.python_version(), numpy=np.__version__, mujoco=mj.__version__),
                  fixture_xml_sha256=sha256(xml.encode()).hexdigest(),
                  setup_scope="One fixed 320×240 simulated RGB-D pair. Includes fixture scene creation, native rendering, marker detection and rigid fitting. Excludes physical fixture provisioning and camera mounting.")
    with (output/f"{capture_id}-frame.npz").open("xb") as frame:
        np.savez_compressed(frame, rgb=rgb, depth=depth)
    return result


def write(path, data):
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, allow_nan=False, indent=2)
        f.write("\n")


def mounting_diagnostics(output):
    reference = capture("nominal", output, label="reference")
    write(output/"reference.json", reference)
    results = {}
    cases = {"anchor_offset": survey.FIT, "check_offset": survey.CHECK,
             "shared_offset": survey.FIT + survey.CHECK}
    for label, markers in cases.items():
        try:
            receipt = capture("nominal", output, shifted_markers=markers, label=label)
        except survey.SurveyMismatch as error:
            results[label] = dict(status="refused", fit_error_mm=error.fit_mm,
                                  check_error_mm=error.check_mm)
        else:
            transform = np.asarray(receipt["base_from_camera"]).reshape(4, 4)
            external_mm = survey.errors_mm(transform, reference["observations"], survey.CHECK)
            # Wrap the diagnostic separately from deployable survey receipts.
            write(output/f"{label}.json", dict(moved_markers=markers, fixture_offset_m=[.012, 0., 0.],
                                               diagnostic_survey=receipt))
            results[label] = dict(status="accepted", fit_error_mm=receipt["fit_error_mm"],
                                  check_error_mm=receipt["check_error_mm"],
                                  independent_base_reference_error_mm=external_mm)
    write(output/"comparison.json", results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="Fresh survey directory")
    p.add_argument("--mounting-diagnostics", action="store_true", help="Run the three preset installation faults.")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.mounting_diagnostics:
        mounting_diagnostics(args.output)
        return
    nominal = capture("nominal", args.output)
    write(args.output/"nominal.json", nominal)
    stale = np.asarray(nominal["base_from_camera"]).reshape(4, 4)
    shifts = {}
    for profile in PROFILES:
        if profile == "nominal":
            continue
        try:
            relocated = capture(profile, args.output)
        except ValueError as error:
            shifts[profile] = dict(status="refused", reason=str(error))
            continue
        write(args.output/f"{profile}.json", relocated)
        shifts[profile] = dict(status="accepted", held_out_mm=relocated["check_error_mm"],
              stale_held_out_mm=survey.errors_mm(stale, relocated["observations"], survey.CHECK),
              acquire_and_fit_wall_seconds=relocated["acquire_and_fit_wall_seconds"])
    write(args.output/"comparison.json", dict(
        nominal_check_mm=nominal["check_error_mm"],
        camera_profiles=dict(PROFILES), shifted_cameras=shifts,
        fit_count=len(survey.FIT), independent_check_count=len(survey.CHECK),
        scope="The same six surveyed fixture tops at fixed camera poses. Four fit each transform. Two withheld points measure registration error. No task arrangement or placement outcome was used."))


if __name__ == "__main__":
    main()

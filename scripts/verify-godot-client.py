"""Independently replay a complete Godot smoke trace in the native workstation.

This checks received-state parity, not whether pixels or physical hardware are
accurate. Godot JSON output rounds doubles, so numerical tolerance is explicit.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

from dvidia_training import reference_sim as sim, workstation as ws


TOLERANCE = 1e-9


def run(assets, receipt, output):
    receipt, output = Path(receipt), Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Verification output must be fresh")
    if receipt.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Smoke receipt exceeds bound")
    raw = receipt.read_bytes()
    result = json.loads(raw)
    if (result.get("schema") != "dvidia.godot-client-smoke/v1" or result.get("ok") is not True
            or result.get("errors") or not 1 <= result.get("step_responses", 0) <= 256
            or result.get("step_responses") != len(result.get("trace", []))
            or result.get("requested_steps") != result["step_responses"]
            or result.get("reset_responses") != 0):
        raise ValueError("Require a complete successful smoke receipt with no reset")
    model = ws.build_model(assets)
    if sim.model_digest(model) != result["expected_model_sha256"]:
        raise ValueError("Native model mismatch")
    data, control = ws.initialize(model), {}
    errors = {}
    scalars = 0

    def compare(actual, reference, path):
        nonlocal scalars
        if type(reference) in (int, float):
            if type(actual) not in (int, float) or not math.isfinite(actual):
                raise AssertionError("Nonfinite/invalid numeric value: " + path)
            error = abs(actual - reference)
            errors[path.split(".")[0]] = max(errors.get(path.split(".")[0], 0.), error)
            scalars += 1
            if error > TOLERANCE:
                raise AssertionError(f"{path}: {error} exceeds tolerance")
        elif isinstance(reference, dict):
            if type(actual) is not dict or set(actual) != set(reference):
                raise AssertionError("State fields differ at " + path)
            for key in reference:
                compare(actual[key], reference[key], path + "." + key if path else key)
        elif isinstance(reference, list):
            if type(actual) is not list or len(actual) != len(reference):
                raise AssertionError("State shape differs at " + path)
            for a, b in zip(actual, reference):
                compare(a, b, path)
        elif actual != reference:
            raise AssertionError("State value differs at " + path)

    compare(result["initial_state"], ws.state(model, data, ws.HOME_TARGET), "")
    for index, row in enumerate(result["trace"]):
        command = row["request"]
        if set(command) != {"type", "sequence", "target", "jaw"} or command["type"] != "step" or command["sequence"] != index + 1:
            raise ValueError("Trace contains incomplete or reordered commands")
        reference = ws.run_control(model, data, control, command["target"], command["jaw"], 10)
        compare(row["state"], reference, "")
    if abs(result["applied_sim_time_seconds"] - data.time) > TOLERANCE:
        raise AssertionError("Final clock mismatch")
    report = {"schema": "dvidia.godot-client-parity/v1", "passed": True,
              "scope": __doc__.strip(), "receipt_sha256": sha256(raw).hexdigest(),
              "verifier_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "workstation_sha256": sha256(Path(ws.__file__).read_bytes()).hexdigest(),
              "model_sha256": result["expected_model_sha256"], "tolerance_absolute": TOLERANCE,
              "received_states_checked": len(result["trace"]) + 1,
              "numeric_scalars_checked": scalars, "maximum_absolute_error_by_field": errors,
              "native_ticks_replayed": len(result["trace"]) * 10,
              "physical_accuracy_validated": False, "pixels_validated": False}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": True, "states": report["received_states_checked"],
                      "scalars": scalars, "maximum_absolute_error": max(errors.values())}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.assets, args.receipt, args.output)

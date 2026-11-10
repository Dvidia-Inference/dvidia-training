"""Compare the unchanged workstation with the local JSON bridge, without a viewer.

Fixed workload: three paired repetitions of the six authored tracking segments.
This tests integration parity and CPU request overhead, not a learned task,
physical fidelity, socket round-trip latency, or desktop rendering performance.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np

from dvidia_training import godot_bridge as bridge, reference_sim as sim, workstation as ws


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def timing(values):
    return {"p50_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
            "maximum_ms": float(max(values))}


def run(assets, output):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Benchmark output must be fresh")
    start = time.perf_counter()
    model = ws.build_model(assets)
    model_sha = sim.model_digest(model)
    compile_seconds = time.perf_counter() - start
    mj, _ = sim._native_imports()
    protocol = ws.protocol()
    results, all_equal = [], True
    for repeat in range(3):
        traces = {}
        for mode in (("direct", "json_bridge") if repeat % 2 == 0 else ("json_bridge", "direct")):
            reset_start = time.perf_counter()
            if mode == "direct":
                data, controller = ws.initialize(model), {}
            else:
                session = bridge.Session(model, model_sha)
                ready = session.process(encoded({"type": "hello", "schema": bridge.SCHEMA,
                                                  "model_sha256": model_sha}))
                if ready["type"] != "ready":
                    raise AssertionError(ready)
                data = session.data
            reset_seconds = time.perf_counter() - reset_start
            durations, trace, segment_errors = [], [], []
            sequence = 0
            wall = time.perf_counter()
            for action in protocol["actions"]:
                for _ in range(action["control_steps"]):
                    sequence += 1
                    begin = time.perf_counter_ns()
                    if mode == "direct":
                        state = ws.run_control(model, data, controller, action["target"], action["jaw"], 10)
                    else:
                        response = session.process(encoded({"type": "step", "sequence": sequence,
                                                            "target": action["target"], "jaw": action["jaw"]}))
                        response = json.loads(encoded(response))
                        if response["type"] != "state" or response["sequence"] != sequence:
                            raise AssertionError(response)
                        state = response["state"]
                    durations.append((time.perf_counter_ns() - begin) / 1e6)
                    trace.append(state)
                segment_errors.append({"segment": action["name"], "position_error_m": state["position_error_m"]})
            seconds = time.perf_counter() - wall
            if any(w.number for w in data.warning) or abs(data.time - 12.) > 1e-9:
                raise AssertionError("Native warnings or clock drift")
            traces[mode] = trace
            results.append({"repeat": repeat, "mode": mode, "intervals": sequence,
                            "native_ticks": sequence * 10, "simulation_seconds": float(data.time),
                            "wall_seconds_including_trace_collection": seconds,
                            "real_time_factor": float(data.time) / seconds,
                            "initialize_seconds_excluded_from_loop": reset_seconds,
                            "request_timing": timing(durations), "solver_warnings": 0,
                            "trace_sha256": sha256(encoded(trace)).hexdigest(),
                            "segment_final_errors": segment_errors})
        exact = traces["direct"] == traces["json_bridge"]
        all_equal = all_equal and exact
        results[-1]["paired_full_state_exactly_equal"] = exact
    if sim.model_digest(model) != model_sha:
        raise AssertionError("Compiled model changed during benchmark")
    source = Path(bridge.__file__).parent
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {"schema": "dvidia.godot-bridge-benchmark/v1", "scope": __doc__.strip(),
              "protocol": protocol, "compiled_model_sha256": model_sha,
              "source_sha256": {name: sha256((source / name).read_bytes()).hexdigest()
                                for name in ("godot_bridge.py", "workstation.py", "reference_sim.py")},
              "benchmark_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "host": {"platform": platform.platform(), "machine": platform.machine(),
                       "python": platform.python_version(), "numpy": np.__version__, "mujoco": mj.__version__,
                       "whole_process_peak_rss_bytes": int(peak_rss if sys.platform == "darwin" else peak_rss * 1024)},
              "compile_seconds": compile_seconds, "runs": results,
              "paired_state_comparisons": 1800, "total_native_ticks": 36000,
              "all_paired_states_exactly_equal": all_equal, "physical_accuracy_validated": False,
              "learned_skill_evaluated": False, "rendering_evaluated": False,
              "socket_latency_evaluated": False, "physical_robot_ready": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "parity": all_equal, "native_ticks": 36000,
                      "real_time_factors": [round(row["real_time_factor"], 2) for row in results]}))
    if not all_equal:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.assets, args.output)

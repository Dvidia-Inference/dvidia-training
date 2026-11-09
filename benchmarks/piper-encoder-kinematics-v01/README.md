# PiPER encoder geometry: static CPU timing

This component translates measured joint angles and a desired tool pose into an authored bounded geometric update. It does not learn, simulate a task or execute an actuator. See [the API and frame contract](../../docs/piper-encoder-kinematics-v01.md).

Two executions on November 9, 2026 are retained: the initial internal prototype probe and the subsequent portable-script verification. The module stayed unchanged; no parameters were tuned from timings. Both used 64 seeded configurations, 64 warmups per API and 1,024 timed calls per API. Targets were prepared outside the timer.

| Probe | FK median / p95 (ms) | Joint-target calculation median / p95 (ms) | Initialization (ms) |
|---|---:|---:|---:|
| Internal prototype | 0.007875 / 0.008292 | 0.030750 / 0.032410 | 220.360 |
| Portable script | 0.007750 / 0.008208 | 0.030500 / 0.032410 | 221.497 |

[Prototype receipt](prototype-timing.json) and [portable receipt](portable-timing.json) bind source and probe hashes. Only the portable probe is distributed here. Its whole-process peak RSS was 564,772,864 bytes, including Python, NumPy, asset loading and native compilation; this is not a minimum hardware requirement. The shared host was macOS 26.5.2 arm64, Python 3.12.9, NumPy 2.5.3, MuJoCo 3.15.0, with BLAS thread settings of 1.

These are static wall-clock measurements, not an observation-to-actuator loop, real-time guarantee, collision-safety result or placement benchmark. Native time remained zero; dynamics entry points were patched to refuse calls. All learned placement results and qualification statuses remain unchanged.

Run from the source checkout, using an existing pinned asset directory and a fresh receipt path:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  PYTHONPATH=src python scripts/benchmark-piper-encoder-kinematics.py \
  --assets ./robot-assets --output ./encoder-kinematics-timing.json
```

The script accepts no model or episode input and needs no renderer or network. It pins the measured source module and refuses an existing output. Platform peak RSS is reported only where its units are known (Darwin/Linux); other platforms report null. Windows and Linux execution have not been checked in this local measurement.

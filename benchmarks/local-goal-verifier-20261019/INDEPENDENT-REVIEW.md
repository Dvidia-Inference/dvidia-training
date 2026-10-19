# Independent review: local goal verifier v0.3

The completed 200-case run passed the independent artifact, decision and metric review. All 12 source hashes match both the working files and the pinned Git commit; original Jev benchmark sources and fixtures remain unchanged.

| Cohort | Cases | Result |
| --- | ---: | --- |
| Native physical scenes | 64 | 16/16 completed states recognized; 48/48 incomplete states rejected; 0 false completions. |
| Perturbed evidence | 128 | 120 unknown and 8 incomplete, matching the declared evidence contracts. The underlying scenes are still complete. |
| Internally consistent false evidence | 8 | 8/8 false completions against moving physical states. This limitation is retained. |

Overall physical false completion is **8/56**, despite **200/200 packet-contract agreement**. The latter must not be presented as perception accuracy or universal physical correctness.

The review independently recomputed native history predicates, checked every raw-to-packet mapping, restored all 64 final native states and reconstructed their pose, speed and contact measurements. All 200 full verifier decisions reproduce. All 128 evidence interventions and eight matching-image moving controls were checked. This did not rerun full trajectories or test hardware.

Across 20,000 repeated calls, compute latency was **0.266583 ms median**, **0.31816905 ms p95**, and **0.962875 ms maximum**. Acquisition and the required 200 ms observation history are excluded, as are parsing, rendering, networking and actuation. Repetitions do not increase the number of physical cases.

The verifier rejects the two pre-freeze issues reproduced during review: huge JSON integers and accumulated timestamp drift. Its current tests pass, along with 9 report tests and 11 publication tests.

These are privileged native measurements with asserted zero uncertainty and eight correlated authored layouts. The protocol discloses development exposure. Forged measurements remain a deliberate blind spot; no image inference, new Jev calls, learned policy or real-robot transfer is demonstrated.

[Structured checks and source hashes](independent-review.json).

# Authored action-demonstration fixture

`authored-example.json` illustrates two 20 ms PiPER control transitions. Every
reading, command, calibration, scene/model identity and frame byte identity was
authored for a format test. This is **not collected video, a camera dataset,
a corrective demonstration, or evidence of a learned skill**. Its task outcome
is `unverified`.

The referenced `.fixture.json` frame paths identify placeholder JSON bytes used
in unit tests. These blobs are intentionally not shipped here, and they are
not RGB or depth pixels. Passing validation checks the declarations only.
The normal learning-target gate refuses the `authored_fixture` source kind.

```sh
python -m dvidia_training.action_demonstration inspect examples/action-demonstration-v01/authored-example.json
```

The inspection must report `physical_robot_ready: false`,
`source_authenticated: false`, `frame_artifacts_verified: false` and
`causal_learning_pairs_declared_available: false`. No simulator, model
training, frame download or robot command runs.

See [the format and CLI guide](../../docs/action-demonstrations.md) for the
requested/issued/applied/measurement distinction and the still-missing capture
producer, actual RGB-D observations and reviewed corrective demonstrations.

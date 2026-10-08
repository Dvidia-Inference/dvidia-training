# Contributing

Use a dedicated Python 3.12+ virtual environment and install `.[arm]` for the
movement and native simulation tests. Install FFmpeg and ffprobe separately.

```sh
python -m pip install -e '.[arm]'
python -m unittest discover -s tests -v
```

Keep source footage, credentials, machine paths and unreviewed third-party data
out of commits. Use explicitly synthetic fixtures for tests. Every claimed skill
needs its own action/sensor/robot contract and independent closed-loop evidence.
Visual prediction metrics must never become a robot success score.

New source adapters must retain original hashes, connected-source grouping,
bounded decoding, data-only imports and offline operation. Numeric preprocessing
learns only from training groups; development data select parameters and test
data measure the frozen result. Preserve failed jobs and honest baselines.

The optional native placement controller is an initial reference adapter, with
authored phases and privileged simulator state. Changes to its behavior need
contact, grasp, lift, release and negative-control verification, plus new runtime
bindings for affected capsules. Do not reuse an old qualification receipt.

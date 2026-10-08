# Skillspace footage format

The current pipeline accepts a local directory of videos, a data-only ZIP, a supported JSON manifest, or a prepared `dataset.json`. Explicit online intake also accepts a public DVIDIA HTTPS store/JSON export that exposes actual media paths. A skill title, thumbnail, demonstration count or metadata-only store pack is not training footage.

This page documents the **implemented footage format**. It is not a claim that all existing DVIDIA Skillspaces already export it, or that a folder supplies robot competence.

Commands below run from the checkout and use the virtual environment directly. In Windows PowerShell, replace `.venv/bin/COMMAND` with `.\.venv\Scripts\COMMAND.exe` and put continued commands on one line. See [installation](installation.md) for complete platform setup.

## A local source

```text
my-skillspace/
  skillspace.training.json
  videos/
    attempt-001.mp4
    attempt-002.mp4
    attempt-003.mp4
```

`skillspace.training.json`:

```json
{
  "kind": "dvidia.skillspace-training-source",
  "schema_version": 1,
  "task": "Move one shoe into a marked slot",
  "media": [
    {
      "id": "attempt-001",
      "path": "videos/attempt-001.mp4",
      "source_recording_id": "recording-001",
      "session_id": "session-001",
      "shoe_pair_id": "pair-001",
      "complete": true
    },
    {
      "id": "attempt-002",
      "path": "videos/attempt-002.mp4",
      "source_recording_id": "recording-002",
      "session_id": "session-002",
      "shoe_pair_id": "pair-002",
      "complete": true
    },
    {
      "id": "attempt-003",
      "path": "videos/attempt-003.mp4",
      "source_recording_id": "recording-003",
      "session_id": "session-003",
      "shoe_pair_id": "pair-003",
      "complete": true
    }
  ]
}
```

This is a format example; the referenced files must exist. The labels declare provenance and completeness, rather than proving them. Three connected groups are the smallest accepted train/development/test split, **not a scientifically adequate shoe-training dataset**. Prefer many reviewed, distinct sessions and physical shoe pairs with documented task coverage.

Supported per-video fields:

| Field | Meaning |
| --- | --- |
| `id` | Required unique clip identifier; bounded letters/numbers and `_.:-` |
| `path` | Required relative video path inside the export |
| `source_recording_id` | Original recording identity; retain it for every cut from that recording |
| `session_id` | Capture-session identity |
| `shoe_pair_id` | Physical shoe-pair identity when known; left/right shoes share this value |
| `complete` | Explicit boolean for a reviewed complete task attempt; missing defaults to false |
| `task` | Optional clip-specific task label |
| `actions_path` | Optional relative path to a compatible movement-supervision JSON sidecar |

Do not invent new recording/session/pair IDs to manufacture independent groups. Unknown lineage is reported as a warning. A plain video directory is discovered automatically, but filenames and byte hashes alone cannot establish independent capture history. `export.json` and `skill.json` can also expose the supported `media` list.

## Grouping and complete attempts

Clips sharing a recording ID, session ID, shoe-pair ID or identical media SHA-256 join one connected group. The entire group stays in one split, including indirect connections through other clips. Adjacent frames are never independently scattered across train/test.

Provide original IDs **before** cutting recordings into 5–6-second clips. Five-second fragments can support visual learning but may omit acquisition, release or final stability; do not mark them complete unless the full attempt is actually visible. Multiple complete attempts from one recording can count as attempts, while still belonging to one dependent source group.

The current source format has no separate `shoe_instance_id` or clip-interval field. Use stable recording/session/pair grouping and retain richer originals locally. A future sample-size protocol may add those fields through an explicitly versioned schema; adding unsupported media-entry keys to this version stops intake.

Ten, 25, 50 and 100 complete demonstrations are useful proposed study checkpoints. None is a proven threshold, and repeating or augmenting one recording does not create new independent demonstrations. New shoe instances, occlusions, materials or grasp conditions need their own coverage and evaluation.

## Bounds and data handling

- Video types: MP4, MOV, WebM, M4V, AVI and MKV.
- Maximum video size: 128 MiB. Maximum source-media budget: 500 MiB.
- Maximum episode count: 1,200; filesystem/archive traversal is also bounded.
- Studio upload: one ZIP of at most 24 MiB. Select a local folder for larger data.
- Media paths must stay within the export. Unsafe traversal, symlinks, encrypted archives and executable/pickle artifacts are rejected by intake.
- An image-only folder is not a video training dataset. Actual bytes are copied, hashed and probed with `ffprobe`; a malformed file stops the job.

Public URL intake is off by default. Start the studio with `--allow-online` and enable its per-source checkbox, or pass `--allow-online` to a CLI prepare/run command. The supported origin is a public DVIDIA store/JSON export, not an arbitrary private page or authenticated clip. A metadata-only page fails with an actionable request for actual footage.

## What is learned and exported

The visual learner samples small RGB frames, fits a PCA basis from training data, and learns a ridge next-frame predictor. Development groups choose regularization; test groups measure prediction error against last-frame and training-mean baselines. Reported group counts reflect declared lineage and hashes, not independently verified novelty.

The output is visual learning. Pixel/latent prediction error does not establish object understanding, calibrated shoe geometry, a grasp policy or successful organization. Original footage remains in the local dataset. The downloadable ZIP includes learned parameters, held-out reports and a trimmed receipt without the videos; `.venv/bin/dvidia-train inspect` verifies its integrity manifest and ZIP contents.

## Movement supervision

Human hand motion or image landmarks are not robot action labels. An optional `actions_path` must point to compatible timestamped state/action supervision with the exact feature contract, video/sample hashes and declared alignment evidence expected by the movement validator.

Accepted origins are robot teleoperation, a validated video/action bridge, or native simulation telemetry. Retain the measurement/alignment evidence: matching hash metadata cannot independently prove the actions are correct. The implemented movement head uses six arm-joint positions and relative object/target context to predict six incremental joint targets; authored task phases, gripper/contact logic and recovery remain separate.

Movement training requires sufficient validated data across the frozen splits, including at least 32 meaningful distinct training samples. Intake reports `movement_training_ready` separately from `movement_ready`; the latter does not become robot qualification merely because fitting is possible. Training exports a simulation-only candidate and reports joint-increment error, rather than closed-loop task success.

With the `arm` extra installed, generate the exact implemented sidecar format through native telemetry and synchronized schematic videos:

```sh
.venv/bin/dvidia-train example --native-arm --clips 10 --output examples/native-demo
.venv/bin/dvidia-train run examples/native-demo --output runs/native-demo
```

These are simulator recordings, not human demonstrations. To export an installable simulation candidate, also provide an explicit compatible task source:

```sh
.venv/bin/dvidia-train run examples/native-demo --output runs/native-candidate \
  --task-source examples/native-demo/task.skill.json
.venv/bin/dvidia-skill-capsule install runs/native-candidate/candidate.skill-capsule.json
```

Use the returned installation ID with `.venv/bin/dvidia-skill-capsule qualify ID --output runs/qualification`. Inspect the independent native success predicate and open-jaw control before running the candidate. This placement adapter supports a bounded rigid-box scene with privileged simulator observations. It does not support arbitrary shoes, cup handles, physical arms or raw-video motor learning.

Physical deployment still needs compatible sensing and actuation, measured calibration/contact behavior and independent hardware qualification. No output of this footage pipeline alone declares a physical robot ready.

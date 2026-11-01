# Camera-to-action evidence publication

This workflow publishes a stopped, sealed simulation study to
`Dvidia/piper-camera-action-v01`. It does not train a model, run physics, alter a
Space, or turn a failed study into a qualified skill.

Build the wheel and a `git archive` tar.gz from the reviewed source commit.
The archive must retain Git's PAX commit comment. Both artifacts must contain
the exact registered runtime bytes, including unchanged native scorers.
The source-artifact limit is 1 GiB compressed, matching the uploader's per-file
limit. The complete reviewed repository at `937d3e3` measured 653,980,983 bytes
compressed and 779,972,592 regular-file bytes expanded; historical checked-in
benchmarks account for most of it. No source files are omitted to fit a smaller
archive. Independent expanded limits remain 2 GiB total, 256 MiB per member and
50,000 members, with path, secret and runtime-identity checks unchanged.
Provide the reviewed dataset card and the independently generated diagnostic
audit. The audit's scope is explicitly the four diagnostic episodes.
The optional `--phase-coverage` argument includes the diagnostic-only phase
coverage proof after verifying its protocol and audit bindings.

After the study stops, independently audit every sealed stage with the numeric
scorer and saved evidence. This command does not import or execute the native
environment, tracker, teacher or learned policy. It verifies frame identities,
50 Hz command and 25 Hz camera clocks, 1 ms native traces, command authority,
teacher-query separation, model identities and split membership. It reconstructs
data-quality gates and, if confirmation exists, grouped bootstrap comparisons
and the final qualification gates. Missing stages remain unrun.

```sh
PYTHONPATH=src:benchmarks/camera-action-v01 python \
  benchmarks/camera-action-v01/study_audit.py /absolute/stopped-study \
  > /absolute/full-study-audit.json
```

Pass `--study-audit /absolute/full-study-audit.json` to preparation. It must
pass, cover every currently sealed stage and match the protocol, model, stage
and episode identities. This proof supplements the diagnostic audit. An audit
passing means reported outcomes agree with saved evidence, including failures;
it does not mean the skill passed its qualification gates.
The reviewed source archive must include the exact full-study auditor and its
diagnostic byte-reader helper; use the final tooling commit while preserving
all registered runtime bytes.

Add `--hardware /absolute/hardware-native.json` to include the measured host,
software versions, native worker count and graphics/timing scope as
`hardware.json`. The field whitelist permits public CPU/model/memory details
and rejects hostname, serial, account or other arbitrary fields. The entire
JSON must fit within 16 KiB. These details describe the measured run; they do
not establish minimum hardware requirements or physical sensor latency.

```sh
PYTHONPATH=src python publishing/prepare_camera_action_v01.py \
  --study /absolute/stopped-study \
  --output /absolute/new-publication-directory \
  --audit /absolute/diagnostic-audit.json \
  --card /absolute/README.md \
  --wheel /absolute/dvidia_training.whl \
  --source-archive /absolute/reviewed-source.tar.gz \
  --source-commit EXACT_40_CHARACTER_COMMIT

python publishing/huggingface/publish_camera_action_v01.py stage \
  /absolute/new-publication-directory --plan /absolute/reviewed-plan.json
```

Review the public roster and returned plan digest before publication. The
explicit `publish` command reads only the existing macOS Keychain entry
`dvidia-huggingface`, account `dvidia-publisher`. No credential argument,
environment variable, credential file, or login cache is required.

```sh
python publishing/huggingface/publish_camera_action_v01.py publish \
  --plan /absolute/reviewed-plan.json --sha256 REVIEWED_PLAN_DIGEST \
  --cache /absolute/publication-cache

python publishing/huggingface/publish_camera_action_v01.py verify \
  --plan /absolute/reviewed-plan.json --sha256 REVIEWED_PLAN_DIGEST \
  --cache /absolute/anonymous-readback-cache
```

Preparation streams one deterministic ZIP64 archive per sealed episode,
checks each file against the original integrity manifest, and decompresses
every member to verify its bytes. DEFLATE level 1 is the publication default.
On the same sealed teacher episode, one sequential shared-host comparison
measured 40.36 seconds / 81,058,092 bytes at level 1 versus 54.02 seconds /
75,301,793 bytes at level 6. That is a 25.3% shorter archive-and-verification
time for 7.64% more bytes in this example, not a general throughput guarantee.
It refuses a worker lock, unsealed episode,
unexpected study file, symbolic link, missing source binding, or detected
credential-shaped text. The 30 GiB disk reserve is preserved. Existing output
directories and files are never overwritten.

The publisher freezes reviewed bytes into a private temporary directory before
credential lookup. On a supporting macOS filesystem, it uses the documented
`fclonefileat(2)` copy-on-write operation from a regular-file descriptor. The
copy has its own inode and attributes, is marked read-only, and is checked
against the reviewed bytes. Later writes to either file are private to that
file. Hardlinks are never used. Unsupported filesystems use bounded streaming
copies with free-space checks before each file and each chunk, preserving the
30 GiB reserve. Every path rejects symlinks and existing destinations.
The uploader uses a single commit bound to the empty dataset's exact
parent revision, and refuses existing nonempty or private repositories.
Anonymous verification downloads every published file at the exact resulting
commit and checks its byte identity. Each file uses a new owned temporary cache
under the explicit `--cache` directory, with its full byte budget plus metadata
headroom checked against the 30 GiB reserve before download. That temporary is
removed after verification, including on failure. Existing shared cache files
are never removed. The retained receipt records the pinned commit, reviewed
plan digest, inventory digest, byte count and largest single readback file.
The pinned remote file-size metadata must match every reviewed byte count
before any file download; missing or changed sizes fail closed.
Verification sets `HF_HUB_DISABLE_XET=1` before SDK import to use HTTP downloads
into the owned Hub cache, and refuses an already-imported SDK whose Xet
transport is still enabled. This avoids a separate shared Xet data cache.
Large files are streamed, not loaded into one large in-memory payload.

To reconstruct the recorded study, verify `SHA256SUMS`, then extract
`study-metadata.zip` and every `episodes/**/*.zip` into one new directory.
The resulting layout contains the original protocol, stage receipts, camera
frames, encoders, issued controls, perception and teacher-query sidecars,
native traces, and each episode's original `integrity.json`. Raw files inside
an episode must not be edited or supplemented before inspection.

The dataset includes separate model JSON files only when sealed training
receipts bind those exact files. A blocked data-quality gate therefore
publishes evidence and source without inventing a learned policy. The package
retains the study's qualification result; installation itself does not award
task qualification or physical-hardware readiness.

## Optional precompression while the study runs

One separate publication worker may precompress already sealed, complete
episodes while the native worker continues. It snapshots the registered stage
rosters, reads only fully sealed stage receipts, and includes completed task
failures as well as successes. It never reads a running stage's episode files,
retries a simulation, queries a policy, or changes study qualification.

```sh
PYTHONPATH=publishing:publishing/huggingface python \
  publishing/episode_archive_cache.py \
  --study /absolute/running-study --cache /absolute/separate-episode-cache \
  --limit 84
```

The limit is a finite maximum of new archives (1–452), with one sequential
worker guarded by its own lock. Before each archive, it reserves the full
remaining registered raw-data capacity, a conservative ZIP expansion allowance
and the existing 30 GiB disk reserve. It can refuse compression even when the
final expected compressed corpus would fit. Existing cache entries are never
overwritten; a partial entry or a lock left after a hard stop requires operator
inspection. The study and native worker remain untouched.

After the study and cache worker stop, add
`--episode-cache /absolute/separate-episode-cache` to the normal preparation
command. Cache receipts provide no authority: final preparation independently
reads every original source file, checks the final inventory, decompresses and
hashes every ZIP member, and checks a private copy again. It uses the same
reviewed copy-on-write helper as publication, with independent inodes and a
bounded streaming fallback. Missing entries use ordinary fresh compression;
malformed or changed entries fail closed. Small refused or incomplete episodes
are retained by normal final packaging, rather than precompressed. No extra
files or relaxed acceptance rules are introduced into the public payload.
Archive and member comments, unknown extra fields, unsorted members, prefix or
trailing payloads, and unused compressed suffixes are refused. Only the exact
local ZIP64 size fields required by the original writer are accepted. DEFLATE
does not encode its compression level, so cached archives truthfully record
`compression_level: null`; final verification establishes original member
bytes and the declared ZIP layout, not an unverifiable producer setting.

## Stopping, interruption and publication scope

Studio and Terminal use the same immutable stage records. Studio owns one local
worker, checks the protocol and assets between stages, and stops at a blocked
or completed study. It does not retry a stage after an exception. A browser
disconnect does not cancel the server's worker. Closing or killing the server
is not a supported pause mechanism.

A native attempt that can write its final receipt retains its completed prefix
as `interrupted` or `refused`; it is not counted as a strict success. If a process
ends before sealing an episode, a subsequent collection/evaluation inspection
retains that existing directory as `unknown` and never reruns its case. Remaining
registered cases are not replaced with easier starts. A sealed stage with an
unknown or unattempted case blocks downstream work.

Fitting has a stricter interruption boundary. Once a model variant's directory
has been created, the frozen trainer refuses to fit that variant again. A fit
exception can therefore leave a model prefix without a sealed training stage.
In this state, the small `inspect` response may still list training as the next
stage; that is not permission or evidence that an automatic retry is possible.
Preserve the directory, fitting log and original stage-start record. Do not
delete them, fabricate a completed receipt or award a model qualification.

An ordinary exception releases the study worker lock. A hard process stop may
leave `worker.lock`; Studio deliberately neither guesses whether it is stale
nor removes it. Any operator investigation must first establish that the
original worker has stopped and preserve the interrupted evidence. Do not
remove a lock to start a second worker against a live study.

The raw-evidence publisher accepts only stopped, sealed `blocked` or `complete`
studies. It refuses an interrupted fitting prefix or an unsealed episode; those
cannot silently disappear from this dataset. If such an interruption happens,
publish its separate, explicitly incomplete evidence report without changing
the registered study or retrying the fit. This is an operational blocker, not a
negative task-success measurement or a completed benchmark.

## Reviewed source snapshot

This change is stacked on `feat/piper-rgbd-tracker-v01`, whose exact local base
for this study is `769d666ecd684406ae228daf722e41f0bf9ac17d`. The camera/action
branch is `feat/piper-camera-action-v01`; use the tracker branch as its pull
request base, not the older stable `main`. Recheck remote ancestry before
publication rather than merging unrelated changes into the frozen study.

Build release artifacts only after the final tooling and report changes are
reviewed and committed. Use that exact commit for both the source archive and
wheel. The archive's PAX comment must identify the commit; the packager checks
that the registered runtime and independent auditor bytes agree. Keep the
existing development version; do not replace a stable release.

```sh
git status --short
git merge-base HEAD origin/feat/piper-rgbd-tracker-v01
git rev-list --left-right --count origin/feat/piper-rgbd-tracker-v01...HEAD
git rev-parse HEAD
git archive --format=tar.gz --prefix=dvidia-training/ \
  --output=/absolute/new-release/reviewed-source.tar.gz EXACT_REVIEWED_COMMIT
```

Extract that reviewed Git archive into a fresh build directory, then build the
wheel from its `dvidia-training` directory with the existing local Python build
environment (`python -m build --wheel --no-isolation --outdir
/absolute/new-release`). This keeps uncommitted files and generated run data out
of the source release. Re-run the fresh-environment installation checks against
the final wheel, and retain its digest with the publication plan. Neither the
source snapshot nor a successful install proves task qualification.

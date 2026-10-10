# Local observation review

The dependency-free page served by the observation lab combines the original
video, sampled boxes, timeline and review queue. It makes no requests beyond its
local origin. There are no fallback detections or automatic action descriptions.

## Grouped review

The main queue groups observations with the same original label, evidence type,
current review label and decision when their intervals overlap or have gaps no
larger than 0.25 seconds. This is a presentation aid, not physical object identity,
tracking, an action segment or proof of continuity through an occlusion.

Each card shows its exact interval and number of included observations. Confirm
or reject applies only to the members shown in the selected group, then advances
to the next pending group. Confirming a group never approves an unrelated group.
Edited descriptions apply to those members; their individual timings remain
unchanged. Differently corrected or decided observations stay separate. Manual
annotations always remain independent.

Expand a group's individual observations to edit one label or interval. The
original raw list remains under **Inspect original evidence**. Nothing is dropped,
suppressed or removed from the stored model record. The immutable source IDs,
raw labels, sample timestamps, scores and boxes remain inspectable.

The six initial pilot clips present 72 original proposals as 33 initial groups:
shirt 7 → 4, stir 25 → 8, pack 7 → 4, onion 10 → 5, melon 14 → 7 and jeans 9 → 5.
These are counts of UI items, not measured reductions in review time or improved
model accuracy. Corrections can split groups. All original proposals still require
an explicit decision before a review can be finished.

## Video and timeline

The timeline is attached to the video, highlights the selected interval, and
shows confirmed intervals separately. Selecting a card seeks to its start.
**Replay moment** plays just that interval. Individual edits can set boundaries
from the playhead or explicit numeric values.

Boxes use normalized `[x1,y1,x2,y2]` image coordinates and respect letterboxing.
Samples are indexed once per episode and looked up with binary search. DOM
nodes are reused by track and updated only when the displayed sample changes;
identical samples do not rebuild the overlay. No new tracking or interpolation
was added. Boxes appear only within 0.25 seconds of a real sampled timestamp,
retain raw model labels and disappear through unsupported gaps. Palm rectangles
remain palm detections; overlap does not establish contact or grasp.

## Automatic saving and completion

Valid drafts save after a 650ms pause. Only one request is in flight at a time.
The response advances the base revision without replacing newer edits; any
changes made during the request are saved in the next revision. Invalid input
remains visibly unsaved and cannot be overwritten by an older save response.

Errors preserve the draft and expose an explicit retry. Conflicting tabs never
silently overwrite one another. A conflict offers export of the unsaved draft
(including current editor input) or explicit discard/reload. Episode navigation
and browser closing protect unsaved changes. No shared browser cache of private
annotations is added: a reload before successful persistence can lose unsaved
input, so heed the leave warning or export the draft.

Draft autosave uses `in_review`. **Finish review** is a separate deliberate
operation and is available only after every proposal has a decision. Failed
completion stays visibly unsaved. Outcomes default to `unknown`; a person can
select visible completion or incomplete when finishing. None implies robot skill
qualification. Export of the saved review is disabled while a draft is unsaved.

The local server stores append-only revisions tied to the original source and
annotation hashes. The public synthetic demo instead uses its existing browser
storage adapter; its UI explicitly describes that storage boundary.

## Mobile and accessibility

Below 821px, selecting a card opens a native dialog styled as a bottom sheet,
with a large Confirm & next action, a reject action and a replay control. The
page brings the video into view above it where viewport height allows. Closing
returns to the queue, or to the finish panel once every item is decided. The dialog
traps focus and makes background content inert. Detailed timing remains collapsed.
On desktop, the same editor stays beside the player. Resizing handles the dialog
state; landscape uses the layout appropriate to its available width.

Controls have at least 44px touch targets; main decision buttons are 50px. Inputs
use 16px text on phones. Semantic buttons, native dialog behavior, labeled inputs,
visible focus, text status labels and polite save feedback accompany the color
cues. Playback is never automatically started by opening a group. Reduced-motion
preferences are honored.

## HTTP and data contract

- `GET /api/episodes`: `{ "episodes": [{ "id": "..." }], "failures": [] }`.
- `GET /api/episodes/{id}`: the episode, immutable observations and latest review.
- `POST /api/episodes/{id}/review`: `{ "base_revision": 0, "events": [],
  "outcome": "unknown", "status": "in_review" }`; responds with `{ "review": ... }`.
- Same-origin video URLs support byte ranges. Writes require the local origin.

Each review event retains `id`, `observation_id`, `label`, `start_seconds`,
`end_seconds` and `decision` (`pending`, `approved`, `rejected`). New manual events
use a `manual-` UUID and `observation_id: null`. No group ID or automatic approval
is added to the backend contract; grouping is derived from the full event list.

## Repeatable checks

`tests/test_observation_review_core.py` runs the pure JavaScript grouping and
sample-boundary invariants when Node.js is installed. The existing Python tests
cover the real decoder, bounded HTTP API, media integrity and review revisions.

For isolated browser tests, use a new synthetic fixture directory in one terminal:

```sh
PYTHONPATH=src .venv/bin/python tests/browser/serve_observation_fixture.py --directory runs/review-browser-fixture --port 8283
```

With Playwright and its Chromium browser already installed, in another terminal:

```sh
node tests/browser/observation_review.cjs
```

`PLAYWRIGHT_MODULE` can select an existing package location. `OBSERVATION_URL`
selects another loopback fixture URL; `OBSERVATION_OUTPUT` selects an ignored
receipt/screenshot directory. The tests require a fresh unreviewed fixture.
They simulate overlapping saves, invalid edits, conflicts, failed writes and
failed completion, then exercise persistence, export, manual events, phone and
landscape layouts, repeated sheet opening, rotation and scroll-lock recovery.
They never modify the original pilot run or production data.

The overlay mutation check makes ten seeks within a single sample and observes
zero DOM mutations. This is a regression invariant, not a phone FPS benchmark.
Headless Chromium checks do not establish Safari or physical-device performance.
Model accuracy and human annotation speed remain unmeasured.

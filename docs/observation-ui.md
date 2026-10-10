# Local observation review

`src/dvidia_training/observation_studio.html` is the dependency-free review page
served by the observation pilot. It makes no network requests outside its local
origin and does not include sample observations or fallback fixture data.

The video and timeline share one clock. Bounding boxes use normalized
`[x1, y1, x2, y2]` coordinates and follow the displayed video area, including
letterboxing. A box appears only within 0.25 seconds of its sampled timestamp;
there is no tracking or interpolation claim. Box labels always come from raw
model output, even after an event is reviewed. The separate green timeline and
accepted list represent the review draft.

Reviewers can select an observation, correct its label and boundaries, approve
or reject it, and add a missed event. Original observations remain visible and
unchanged. Newly added unsaved events can be removed; persisted events retain
their review history and can be rejected in a later revision. Observed outcome
defaults to `unknown`. The other choices are `visible_completion` and
`incomplete`, describing a person's visual assessment rather than robot success.

Saving posts the complete draft with the current base revision and fetches the
episode again to confirm persistence. A reviewer must explicitly mark completion,
and every event must have a decision before a review is marked complete.
Conflicts or other save errors preserve the local draft. Previous/next navigation
offers to save, discard, or keep reviewing when changes exist. The browser also
warns before closing a page with unsaved edits. Downloads contain the last saved
review, episode ID, and original source hash; downloading is disabled while a
draft is unsaved.

## HTTP contract

- `GET /api/episodes`: `{ "episodes": [{ "id": "..." }], "failures": [] }`.
  Optional failures remain visible above the review workspace.
- `GET /api/episodes/{id}`: the episode document below.
- `POST /api/episodes/{id}/review`: `{ "base_revision": 0, "events": [],
  "outcome": "unknown", "status": "in_review" }`; returns the saved review.
- The episode's `video_url` must resolve to the same origin. The server must
  support browser video byte ranges.

An episode has `id`, optional `title`, `video_url`, `duration_seconds`,
`observations`, and `review`. Each observation has `id`, `label`, `start_seconds`,
`end_seconds`, optional numeric `confidence`, and a `boxes` array. Each box has
`time_seconds`, normalized `xyxy`, and an optional `label`.

The review has `revision`, `status`, `outcome`, and `events`. Review events carry
`id`, `observation_id`, `label`, `start_seconds`, `end_seconds`, and `decision`
(`pending`, `approved`, or `rejected`). New manual events use a `manual-` UUID
and `observation_id: null`. Missing review entries are shown as pending.

Optional provenance: `source.{sha256,kind,credit,license}`,
`recipe.{id,model,notes}`, and
`metrics.{elapsed_seconds,frames,detected_frames}`. Missing values are labeled
as not supplied. Model metadata accepts a name or an object with `kind` and a
`models` list of `role`, `name`, and `license` entries. Source kind distinguishes
public, synthetic, and authorized user footage. No accuracy or robot competence
metric is inferred.

## Accessibility and responsive behavior

Controls support keyboard input, visible focus, explicit names, and a minimum
44px target. Checkboxes use their full labeled area as the target. The video
never autoplays. Review/save feedback uses a polite status region; server errors
use an alert. Raw event chips expose label and exact timing to screen readers.
The layout stacks at tablet and phone widths, supports 320px-wide screens, and
honors reduced-motion preferences. Selecting an event on a stacked layout brings
the editor into view; accepted event timing retains decimal seconds. Provenance
follows the review panels in both the visual and document order. There are no
third-party assets or scripts.

## Interface verification

An isolated copy of the six-episode pilot was checked in a headless Chromium
browser on October 10, 2026. Checks exercised real playback, sampled overlays,
label correction and approval with save/reload, adding/removing a new event,
manual-event persistence, saved-review download including the original source
hash, and unsaved navigation. Desktop, 390px, and 320px layouts were captured;
phone layouts had no horizontal overflow and their visible controls met the
44px target. Browser console/page errors were absent. Local receipts and
screenshots are in the ignored `runs/observation-ui-qa-v1/` directory. This
verifies interface behavior only, not model accuracy or human review quality.

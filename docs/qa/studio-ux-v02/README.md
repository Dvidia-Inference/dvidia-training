# Skill Studio alpha 4: camera mapping and proposal review

Checked 26 October 2026. This is a local UX and packaging check with the
synthetic planar green-block example, not a learned-policy benchmark.

## Changes and observations

The earlier screen required a calibration JSON file. The new screen accepts
four measured geometry/error values directly, fills image dimensions from
available video metadata, and keeps JSON and detection settings in disclosures.
Example geometry never fills an uploaded source. The additional form takes
more vertical space while measuring; it collapses after a proposal is created.
This improves the setup task rather than claiming every state is shorter.

The accepted paper/green styling and whole-video preview remain. On the recorded
desktop view, the video occupies 657.6 × 180 CSS pixels. The reviewed source
card is shorter than the open measurement form; start coordinates and seed no
longer compete with the main simulation action. Before/after screenshots are
retained, but viewport reporting varied during the browser session, so they
are visual evidence rather than a controlled conversion or scroll study.

- Entered camera measurements in the real browser; created a proposal and
  inspected its labelled marker at the clip ending.
- Changed the mapping: execution disabled, old result/marker/download hidden,
  and a recheck instruction appeared. Clearing a field stayed cleared; 51 mm
  declared error was rejected. Restoring the exact accepted values reenabled it.
- Tested the installed wheel outside the source checkout. Opened a source ZIP,
  reopened the same file, and imported its saved camera mapping. Both source
  changes cleared the previous proposal/result and left geometry blank.
- The synthetic example used its supplied, locked mapping. After a deliberately
  stopped local service, the loaded UI disabled actions and offered Reconnect;
  restarting the service and reconnecting restored the current source.
- The observed narrow viewport was 444 CSS pixels wide: one column, no horizontal
  overflow, 180 px high complete video preview. Temporary viewport override was
  reset. This is browser layout evidence, not a native mobile test.

## Verification

Thirty focused Python tests passed, including source-package integrity, HTTP
origin/path restrictions, the two explicit packaged script routes, candidate
compatibility, and failure handling. Four Node tests passed for measurement
units/bounds, mapping equality/staleness, and world-to-image projection. Scripts
passed syntax checks. No front-end dependencies or network services were added.

One source-checkout attempt and one installed-wheel attempt completed the same
synthetic placement, each with valid native physics and independently scored
destination error of about 5.0 mm and terminal dwell of 5.38 s. Both simulated
20 seconds. Wall compute time was 1.80 s and 7.49 s respectively on the shared
machine; these isolated timings are not latency guarantees or a minimum hardware
certification. The controller and physics implementation are unchanged.

The installed attempt's [receipt](receipt.json), [synthetic episode](episode.json.gz),
and exact [checks](checks.json) are retained. The wheel contains the expected
packaged HTML/scripts, matches the tested source, and includes no local cache or
run directory. Installer bytes and digest are recorded in `checks.json` and the
release's `SHA256SUMS`.

## Evidence and remaining limits

[Before](before.png) · [Measurement form](mapping.png) ·
[Mapping changed](stale.png) · [Reviewed result](after.png) ·
[Installed result](installed-result.png) · [Narrow layout](mobile.png) ·
[Disconnected UI](offline.png)

The overlay is a destination projection under caller-supplied orthographic
calibration. It does not verify the calibration, track object identity or prove
the human's intended task. “Placement checks passed” compares the final simulator
state with the proposed destination. The controller uses privileged simulator
state. No motor policy, physical transfer, shoes, or other-arm performance is
established. See the [research assessment](../../../reports/Readable%20video%20skill%20readiness.md)
for the next action-learning and independent-verification gates.

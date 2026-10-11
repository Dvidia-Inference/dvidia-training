/* MIT · DVIDIA contributors. Synthetic-only API. No writes leave this browser. */
(() => {
  'use strict';
  const prefix = 'dvidia-observation-demo-v1:';
  const memory = new Map();
  let storage = null;
  try {
    const probe = prefix + 'probe';
    localStorage.setItem(probe,'1'); localStorage.removeItem(probe);
    // Origin-scoped Web Locks serialize commits across tabs. Without them use
    // per-page memory rather than a shared writable history with weak locking.
    if (navigator.locks?.request) storage = localStorage;
  } catch { /* Some embedding/browser privacy settings block storage. */ }
  const copy = value => JSON.parse(JSON.stringify(value));
  const status = () => {
    const el = document.getElementById('demo-storage-status');
    if (el) el.textContent = storage
      ? 'Saved demo revisions stay in this browser. No footage or edits are uploaded. Download reviews before resetting or clearing browser storage.'
      : 'Browser storage or cross-tab locking is unavailable. Reviews last only while this page stays open; download a saved review before leaving. No footage or edits are uploaded.';
  };
  const canonical = value => JSON.stringify(value, (_, item) => item && typeof item === 'object' && !Array.isArray(item)
    ? Object.fromEntries(Object.keys(item).sort().map(key => [key,item[key]])) : item);
  async function digest(value) {
    const raw = await crypto.subtle.digest('SHA-256',new TextEncoder().encode(canonical(value)));
    return Array.from(new Uint8Array(raw), b => b.toString(16).padStart(2,'0')).join('');
  }
  async function fetchJSON(path) {
    const response = await fetch(path,{cache:'no-store'});
    if (!response.ok) throw new Error('The included synthetic sample could not be loaded.');
    return response.json();
  }
  const fixtures = new Map();
  async function fixture(id) {
    if (!['visible','occluded'].includes(id)) throw new Error('Only the included synthetic examples are available.');
    if (!fixtures.has(id)) fixtures.set(id,await fetchJSON(`demo/${id}.json`));
    return copy(fixtures.get(id));
  }
  function revisions(episode) {
    const key = prefix + episode.id + ':' + episode.source.sha256;
    const raw = storage ? storage.getItem(key) : memory.get(key);
    if (raw === null || raw === undefined) return [];
    if (raw.length > 4 * 1024 * 1024) throw new Error('This browser’s demo history is too large. Download and reset it.');
    let rows;
    try { rows = JSON.parse(raw); } catch { throw new Error('This browser’s demo history is unreadable. Reset the demo to start again.'); }
    if (!Array.isArray(rows) || rows.length > 100) throw new Error('Invalid demo history. Reset it to start again.');
    return rows;
  }
  function validateEvents(body,episode) {
    if (!body || !Number.isInteger(body.base_revision) || body.base_revision < 0
      || !['unknown','visible_completion','incomplete'].includes(body.outcome)
      || !['in_review','reviewed'].includes(body.status)
      || !Array.isArray(body.events) || body.events.length > 1500) throw new Error('Invalid demo review.');
    const originals = new Set(episode.observations.map(event => event.id)), seen = new Set(), retained = new Set();
    const clean = body.events.map(event => {
      if (!event || typeof event.id !== 'string' || !/^[A-Za-z0-9_-]{1,100}$/.test(event.id) || seen.has(event.id)
        || typeof event.label !== 'string' || !event.label.trim() || event.label.length > 500
        || !Number.isFinite(event.start_seconds) || !Number.isFinite(event.end_seconds)
        || event.start_seconds < 0 || event.end_seconds <= event.start_seconds || event.end_seconds > episode.duration_seconds
        || !['pending','approved','rejected'].includes(event.decision)
        || (body.status === 'reviewed' && event.decision === 'pending')) throw new Error('Check every event’s label, decision and timing.');
      seen.add(event.id);
      if (event.observation_id !== null) {
        if (!originals.has(event.observation_id) || event.id !== event.observation_id || retained.has(event.observation_id)) throw new Error('Keep original demo proposals; reject them rather than deleting their history.');
        retained.add(event.observation_id);
      } else if (!event.id.startsWith('manual-')) throw new Error('Added events need a manual identifier.');
      return {id:event.id, observation_id:event.observation_id, label:event.label.trim(),
        start_seconds:event.start_seconds,end_seconds:event.end_seconds,decision:event.decision};
    });
    if (retained.size !== originals.size) throw new Error('Keep every original demo proposal.');
    return clean;
  }
  async function current(episode,rows) {
    let review = copy(episode.review);
    for (const row of rows) {
      if (!row || row.revision !== review.revision + 1 || row.parent_sha256 !== await digest(review)
        || row.source_sha256 !== episode.source.sha256 || row.episode_id !== episode.id
        || row.annotation_sha256 !== await digest(episode.observations)) throw new Error('This browser’s demo history changed. Download or reset it before continuing.');
      validateEvents({base_revision:review.revision,events:row.events,outcome:row.outcome,status:row.status},episode);
      review = row;
    }
    return review;
  }
  window.DvidiaObservationDemo = {
    async list() { status(); return fetchJSON('demo/episodes.json'); },
    async episode(id) { const e = await fixture(id); e.review = await current(e,revisions(e)); status(); return e; },
    async save(id,body) {
      if (storage) return navigator.locks.request(prefix + id, () => this.saveLocked(id,body));
      return this.saveLocked(id,body);
    },
    async saveLocked(id,body) {
      const e = await fixture(id), rows = revisions(e), review = await current(e,rows);
      if (body.base_revision !== review.revision) throw new Error('This demo review changed in another browser tab. Refresh before saving.');
      if (rows.length >= 100) throw new Error('Download your review and reset the demo before adding more revisions.');
      const events = validateEvents(body,e);
      if (events.length < e.observations.length) throw new Error('Keep every original demo proposal.');
      const row = {revision:review.revision+1,status:body.status,outcome:body.outcome,events,
        source_sha256:e.source.sha256,episode_id:e.id,annotation_sha256:await digest(e.observations),
        parent_sha256:await digest(review),review_origin:'browser_demo_operator',
        created_at:new Date().toISOString(),scope:'synthetic_workflow_demo',live_inference:false};
      const key = prefix + e.id + ':' + e.source.sha256, raw = JSON.stringify([...rows,row]);
      if (raw.length > 4 * 1024 * 1024) throw new Error('Demo storage is full. Download your saved review and reset.');
      if (storage) {
        try { storage.setItem(key,raw); }
        catch { throw new Error('Browser storage is full or blocked. Your unsaved draft is still here. Download prior saved reviews before resetting.'); }
      } else memory.set(key,raw);
      status(); return {review:copy(row)};
    },
    async reset() {
      if (storage) return navigator.locks.request(prefix + 'visible',
        () => navigator.locks.request(prefix + 'occluded', () => this.resetLocked()));
      this.resetLocked();
    },
    resetLocked() {
      if (storage) for (const key of Object.keys(storage)) if (key.startsWith(prefix)) storage.removeItem(key);
      memory.clear(); status();
    }
  };
})();

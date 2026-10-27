/* DVIDIA Skill Studio. Local proposal review; no robot policy is learned here. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id), mapping = window.DvidiaStudioMapping;
  const fields = Object.fromEntries(['width','height','x','y','span','bound','dwell','spread'].map(name => [name, $('map-'+name)]));
  let state = null, selectedZip = null, busy = false, connected = false, epoch = 0, pollTimer = 0;
  let previewUrl = null, calLoadEpoch = 0, sourceId = null, mode = 'fields', clip = null, mappingNote = '', lastCandidate = null, previewFailed = false;
  const api = async (path, options = {}) => {
    const response = await fetch(path, options);
    let data; try { data = await response.json(); } catch { data = {}; }
    if (!response.ok) throw new Error(data.error || data.message || `Request failed (${response.status}).`);
    return data;
  };
  const post = (path, value) => api(path, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(value)});
  const bytes = n => !Number.isFinite(n) ? '' : n < 1048576 ? `${(n/1024).toFixed(1)} KB` : `${(n/1048576).toFixed(1)} MB`;
  const seconds = n => Number.isFinite(n) ? `${n.toFixed(2)} s` : '—';
  function notice(message, kind = 'info') {
    $('notice-text').textContent = message; $('notice').dataset.kind = kind; $('notice').hidden = !message;
  }
  function clearPoll() { clearTimeout(pollTimer); pollTimer = 0; }
  function activeMapping() { return mode === 'json' ? mapping.parse($('calibration-json').value) : mapping.fromFields(Object.fromEntries(Object.entries(fields).map(([k, el]) => [k, el.value]))); }
  function fillFields(value) {
    const values = {width:value.width, height:value.height, x:value.center_xy_m[0], y:value.center_xy_m[1], span:value.vertical_span_m, bound:value.position_bound_m*1000, dwell:value.terminal_dwell_s, spread:value.maximum_terminal_spread_m*1000};
    for (const [name, n] of Object.entries(values)) fields[name].value = n;
  }
  function resetMapping() {
    mode = 'fields'; mappingNote = ''; calLoadEpoch++;
    for (const el of Object.values(fields)) el.value = '';
    fields.dwell.value = '.5'; fields.spread.value = '5';
    $('calibration-json').value = ''; $('calibration-file').value = '';
    $('calibration-error').hidden = true; $('calibration-error').textContent = '';
  }
  function setMapping(value) {
    // Preserve exact imported numbers until the user edits a measurement field.
    fillFields(value); $('calibration-json').value = JSON.stringify(value, null, 2); mode = 'json';
  }
  function clearSource() {
    clearPoll(); resetMapping(); sourceId = null; clip = null; lastCandidate = null; previewFailed = false;
    if (previewUrl) URL.revokeObjectURL(previewUrl); previewUrl = null;
    $('video-player').pause(); $('video-player').removeAttribute('src'); $('video-player').dataset.url = ''; $('video-player').load();
    $('source-file').value = '';
    $('goal-marker').hidden = true; $('simulation-result').hidden = true; notice('');
  }
  function preflight() {
    if (!clip) return '';
    if (clip.width < 64 || clip.width > 640 || clip.height < 64 || clip.height > 480) return 'Use a 64–640 × 64–480 px copy, and supply the mapping for that copy.';
    if (!Number.isFinite(clip.duration) || clip.duration < .6 || clip.duration > 30) return 'Use a complete clip from 0.6 to 30 seconds.';
    const parsed = activeMapping();
    if (!parsed.error && (parsed.value.width !== clip.width || parsed.value.height !== clip.height)) return 'Mapping image size must match this clip. Check image size under detection settings.';
    return '';
  }
  function stale() {
    if (!state?.source || !state?.candidate) return false;
    const parsed = activeMapping();
    return !!parsed.error || !!preflight() || !mapping.same(parsed.value, state.source.calibration);
  }
  function renderSource() {
    const source = state?.source;
    if (source && source.id !== sourceId) {
      resetMapping(); sourceId = source.id; clip = null;
      if (source.calibration) { setMapping(source.calibration); mappingNote = source.sample ? 'Example camera mapping supplied.' : 'Restored the mapping used for this clip.'; }
      $('camera-mapping').open = !source.calibration;
    }
    const video = $('video-player'), hasVideo = !!source || !!previewUrl;
    video.hidden = !hasVideo; $('video-empty').hidden = hasVideo; $('video-label').hidden = !hasVideo;
    $('video-label').textContent = source?.sample ? 'Synthetic example' : source?.skillspace ? 'Skillspace video' : source ? 'Source clip' : 'Local preview';
    const url = source?.media_url || previewUrl || '';
    if (url && video.dataset.url !== url) { clip = null; previewFailed = false; video.src = url; video.dataset.url = url; video.load(); }
    $('camera-section').hidden = !source;
    $('source-meta').hidden = !source;
    if (source) {
      const tags = [source.name, bytes(source.size_bytes), source.sample ? 'Synthetic example' : source.skillspace ? 'Skillspace · '+source.skillspace.task_title : 'Uploaded locally'];
      $('source-meta').replaceChildren(...tags.map(text => { const el = document.createElement('span'); el.className = 'tag'; el.textContent = text; return el; }));
    }
    $('choose-source').textContent = source ? 'Change video' : 'Open video or package';
    $('choose-source').classList.toggle('primary', !source);
    const locked = !!source?.sample;
    $('calibration-picker').hidden = locked;
    for (const el of [...Object.values(fields), $('calibration-json'), $('calibration-file')]) el.disabled = busy || !connected || state?.job?.status === 'running' || locked;
    $('clip-status').textContent = preflight() || (clip ? `${clip.width} × ${clip.height} px · ${seconds(clip.duration)} · native decoding checks follow.` : previewFailed ? 'Browser preview unavailable. FFmpeg will check the clip when you continue.' : 'Supported clips: 0.6–30 s, up to 640 × 480 px.');
    $('clip-status').classList.toggle('field-error', !!preflight());
  }
  function renderMarker() {
    const el = $('goal-marker'), source = state?.source, video = $('video-player');
    const point = source?.calibration && !stale() && state?.candidate ? mapping.imageGoal(state.candidate.goal_xy_m, source.calibration) : null;
    el.hidden = !point || !video.videoWidth || video.hidden;
    if (el.hidden) return;
    const frame = $('video-frame'), scale = Math.min(frame.clientWidth/video.videoWidth, frame.clientHeight/video.videoHeight);
    const width = video.videoWidth*scale, height = video.videoHeight*scale;
    el.style.left = `${(frame.clientWidth-width)/2+point.u*width}px`;
    el.style.top = `${(frame.clientHeight-height)/2+point.v*height}px`;
  }
  function renderCandidate() {
    const c = state?.candidate, changed = stale();
    $('candidate-empty').hidden = !!c; $('candidate-content').hidden = !c;
    if (c) {
      if (c.id !== lastCandidate && state.source && !changed) $('camera-mapping').open = false;
      lastCandidate = c.id;
      $('candidate-goal').textContent = `(${c.goal_xy_m.map(n => Number(n).toFixed(3)).join(', ')})`;
      $('candidate-kind').textContent = changed ? 'Needs recheck' : state.source ? 'Video proposal' : 'Imported proposal';
      $('candidate-warning').hidden = !changed;
      $('candidate-scope').textContent = c.scope || 'Calibrated planar destination';
      $('candidate-hash').textContent = c.video_sha256 || 'Source digest unavailable';
      $('download-candidate').href = c.download_url || '#'; $('download-candidate').hidden = !c.download_url || changed;
      $('view-ending').hidden = !state.source || !Number.isFinite(clip?.duration) || changed;
    }
    renderMarker();
  }
  function renderResult() {
    const job = state?.job;
    $('job-running').hidden = job?.status !== 'running'; $('job-error').hidden = job?.status !== 'failed';
    $('simulation-result').hidden = !(job?.status === 'complete' && job.kind === 'run' && job.result) || stale();
    if (job?.status === 'running') $('job-message').textContent = (job.kind === 'create' ? 'Checking video destination' : 'Running PiPER simulation') + (Number.isFinite(job.elapsed_s) ? ` · ${job.elapsed_s.toFixed(1)} s` : '');
    if (job?.status === 'failed') $('job-error').textContent = job.error || 'This attempt failed. Review the input and retry.';
    if ($('simulation-result').hidden) return;
    const r = job.result, score = r.goal_score || {};
    $('result-status').textContent = score.success ? 'Placement checks passed' : 'Placement checks not met';
    $('result-error').textContent = Number.isFinite(score.terminal_xy_error_m) ? `${(score.terminal_xy_error_m*1000).toFixed(1)} mm` : '—';
    $('result-valid').textContent = score.valid_native_physics === true ? 'Valid' : 'Not valid';
    $('result-retained').textContent = score.retained_transfer === true ? 'Yes' : 'No';
    $('result-dwell').textContent = seconds(score.terminal_dwell_s); $('result-sim-time').textContent = seconds(r.simulated_seconds); $('result-wall-time').textContent = seconds(r.wall_seconds);
    $('result-goal').textContent = Array.isArray(r.goal_xy_m) ? `Goal (${r.goal_xy_m.map(n => Number(n).toFixed(3)).join(', ')}) m` : '';
    $('receipt-link').hidden = !r.receipt_url; if (r.receipt_url) $('receipt-link').href = r.receipt_url;
  }
  function render() {
    const running = state?.job?.status === 'running', wait = busy || running || !connected;
    $('session-label').textContent = connected ? 'Local service connected' : 'Local service unavailable';
    $('session-label').parentElement.dataset.offline = String(!connected);
    $('retry-connection').hidden = connected;
    renderSource(); renderCandidate(); renderResult();
    const parsed = activeMapping(), source = !!state?.source, changed = stale(), ready = source && !parsed.error && !preflight();
    $('calibration-status').textContent = mappingNote || (!source ? 'Your camera mapping stays with the proposal.' : parsed.error ? 'Camera mapping needed.' : 'Camera measurements entered.');
    $('create-help').textContent = !source ? 'Open a video to begin.' : preflight() || parsed.error || (changed ? 'Recheck the updated mapping.' : 'Ready to check this clip.');
    $('create-candidate').disabled = wait || !ready;
    $('create-candidate').textContent = changed ? 'Recheck destination' : 'Check destination';
    $('create-candidate').classList.toggle('primary', !state?.candidate || changed);
    $('choose-source').disabled = wait; $('source-file').disabled = wait;
    $('try-example').disabled = wait || !state?.example?.available;
    $('import-file').disabled = wait; $('import-candidate').disabled = wait || !selectedZip;
    for (const id of ['object-x','object-y','run-seed']) $(id).disabled = wait;
    $('run-piper').disabled = wait || !state?.candidate || changed || !state?.assets?.ready;
    $('run-readiness').textContent = !state?.candidate ? 'Check a destination first.' : changed ? 'Recheck the camera mapping before running.' : !state?.assets?.ready ? 'Set up the PiPER assets to run.' : 'Test this proposal with the authored PiPER controller.';
    $('assets-gate').hidden = !state || !!state.assets?.ready;
    $('assets-message').textContent = 'Install or select local PiPER assets, then restart the service.';
    $('setup-command').textContent = state?.assets?.setup_command || 'Start the local service to check setup.';
    $('copy-setup').disabled = !state?.assets?.setup_command;
    $('workspace').setAttribute('aria-busy', String(busy || running));
    syncPoll();
  }
  function offline(message) { connected = false; clearPoll(); notice(message, 'error'); render(); }
  function syncPoll() {
    clearPoll(); if (!connected || document.hidden || state?.job?.status !== 'running') return;
    const token = epoch;
    pollTimer = setTimeout(async () => {
      try { const next = await api('/api/state'); if (token !== epoch) return; state = next; render(); }
      catch (error) { if (token === epoch) offline(`Connection lost. Reconnect to recover the current job. ${error.message}`); }
    }, 700);
  }
  async function loadState() {
    if (busy) return;
    busy = true; render(); const token = ++epoch;
    try { const next = await api('/api/state'); if (token !== epoch) return; state = next; connected = true; notice(''); }
    catch (error) { if (token === epoch) offline(`Start the local service, then reconnect. ${error.message}`); }
    finally { if (token === epoch) { busy = false; render(); } }
  }
  async function action(fn) {
    if (busy || !connected || state?.job?.status === 'running') return;
    busy = true; clearPoll(); const token = ++epoch; notice(''); render();
    try { const next = await fn(); if (token === epoch) state = next; }
    catch (error) {
      if (token !== epoch) return;
      notice(error.message, 'error');
      try { state = await api('/api/state'); } catch { offline('Connection lost. Reconnect to recover the current workspace.'); }
    } finally { if (token === epoch) { busy = false; render(); } }
  }
  async function sendFile(path, file, type) { return api(path, {method:'POST', headers:{'Content-Type':type}, body:file}); }
  async function upload(file) {
    if (!file || busy || !connected || state?.job?.status === 'running') return;
    const isZip = file.name.toLowerCase().endsWith('.zip'), limit = isZip ? state?.limits?.source_package_bytes : state?.limits?.video_bytes;
    if (limit && file.size > limit) { notice(`This file is ${bytes(file.size)}; the limit is ${bytes(limit)}.`, 'error'); return; }
    clearSource(); state = {...state, source:null, candidate:null, job:null};
    if (!isZip) previewUrl = URL.createObjectURL(file);
    render();
    await action(async () => {
      try { return await sendFile(isZip ? '/api/import-video' : `/api/upload?name=${encodeURIComponent(file.name)}`, file, isZip ? 'application/zip' : 'application/octet-stream'); }
      finally { if (previewUrl) URL.revokeObjectURL(previewUrl); previewUrl = null; }
    });
  }
  $('choose-source').addEventListener('click', () => $('source-file').click());
  $('source-file').addEventListener('change', e => upload(e.target.files?.[0]));
  $('try-example').addEventListener('click', () => { clearSource(); state = {...state, source:null, candidate:null, job:null}; action(() => post('/api/example', {})); });
  $('source-drop').addEventListener('dragover', e => { e.preventDefault(); $('source-drop').classList.add('drag'); });
  $('source-drop').addEventListener('dragleave', e => { if (!$('source-drop').contains(e.relatedTarget)) $('source-drop').classList.remove('drag'); });
  $('source-drop').addEventListener('drop', e => { e.preventDefault(); $('source-drop').classList.remove('drag'); upload(e.dataTransfer.files?.[0]); });
  for (const el of Object.values(fields)) el.addEventListener('input', () => {
    mode = 'fields'; calLoadEpoch++; mappingNote = '';
    const parsed = activeMapping(); $('calibration-json').value = parsed.value ? JSON.stringify(parsed.value, null, 2) : '';
    $('calibration-error').hidden = true; render();
  });
  $('calibration-json').addEventListener('input', () => {
    mode = 'json'; calLoadEpoch++; mappingNote = ''; const parsed = activeMapping();
    if (!parsed.error) fillFields(parsed.value);
    $('calibration-error').textContent = parsed.error; $('calibration-error').hidden = !parsed.error; render();
  });
  $('calibration-file').addEventListener('change', async e => {
    const token = ++calLoadEpoch, file = e.target.files?.[0]; if (!file) return;
    let parsed;
    try { if (file.size > 32768) throw new Error('Mapping file must be 32 KiB or smaller.'); parsed = mapping.parse(await file.text()); if (parsed.error) throw new Error(parsed.error); }
    catch (error) {
      if (token !== calLoadEpoch) return;
      mode = 'json'; $('calibration-json').value = ''; mappingNote = '';
      $('calibration-error').textContent = error.message; $('calibration-error').hidden = false; render(); return;
    }
    if (token !== calLoadEpoch) return;
    setMapping(parsed.value); mappingNote = `Loaded ${file.name}.`; $('calibration-error').hidden = true; render();
  });
  $('video-player').addEventListener('loadedmetadata', () => {
    const v = $('video-player');
    if (v.readyState < 1 || !v.videoWidth || !v.videoHeight || v.currentSrc !== new URL(v.dataset.url, location.href).href) return;
    previewFailed = false; clip = {width:v.videoWidth, height:v.videoHeight, duration:v.duration};
    if (mode === 'fields') { if (!fields.width.value) fields.width.value = v.videoWidth; if (!fields.height.value) fields.height.value = v.videoHeight; }
    render();
  });
  $('video-player').addEventListener('error', () => { if (!$('video-player').error) return; clip = null; previewFailed = true; render(); });
  new ResizeObserver(renderMarker).observe($('video-frame'));
  $('view-ending').addEventListener('click', () => { const v = $('video-player'); if (Number.isFinite(v.duration)) { v.pause(); v.currentTime = Math.max(0, v.duration - .1); } });
  $('create-candidate').addEventListener('click', () => {
    const parsed = activeMapping(); if (!state?.source || parsed.error || preflight()) return;
    action(() => post('/api/create', {source_id:state.source.id, calibration:parsed.value}));
  });
  $('import-file').addEventListener('change', e => { selectedZip = e.target.files?.[0] || null; render(); });
  $('import-candidate').addEventListener('click', () => {
    if (!selectedZip) return;
    action(async () => { const next = await sendFile('/api/import', selectedZip, 'application/zip'); clearSource(); selectedZip = null; $('import-file').value = ''; return next; });
  });
  $('run-piper').addEventListener('click', () => {
    if (!state?.candidate || stale()) return;
    const [xs,ys,ss] = ['object-x','object-y','run-seed'].map(id => $(id).value), x = Number(xs), y = Number(ys), seed = Number(ss);
    if (![xs,ys,ss].every(s => s.trim()) || !Number.isFinite(x) || !Number.isFinite(y) || x < .25 || x > .48 || y < -.15 || y > .15 || !Number.isInteger(seed) || seed < 0 || seed > 999999) { notice('Use start coordinates within the task workspace and a whole-number seed from 0 to 999999.', 'error'); return; }
    action(() => post('/api/run', {candidate_id:state.candidate.id, object_xy:[x,y], seed}));
  });
  $('retry-connection').addEventListener('click', loadState);
  $('copy-setup').addEventListener('click', async () => { try { await navigator.clipboard.writeText($('setup-command').textContent); $('copy-setup').textContent = 'Copied'; setTimeout(() => $('copy-setup').textContent = 'Copy', 1200); } catch { notice('Select and copy the setup command; clipboard access is unavailable.', 'error'); } });
  document.addEventListener('visibilitychange', () => document.hidden ? clearPoll() : syncPoll());
  loadState();
})();

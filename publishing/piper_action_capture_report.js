/* Local, data-only playback. This page does not simulate, train, or command a robot. */
(() => {
  'use strict';

  const byId = (id) => document.getElementById(id);
  const ui = Object.fromEntries([
    'episode-list', 'repeat-count', 'tracker-count', 'load-state', 'load-message',
    'retry', 'report', 'episode-title', 'episode-status', 'camera-label',
    'show-rgb', 'show-depth', 'frame-image', 'frame-video', 'image-error', 'frame-caption',
    'frame-source', 'depth-source', 'previous', 'play', 'next', 'time-readout', 'timeline',
    'time-start', 'time-end', 'episode-counts', 'native-success', 'recorded-result',
    'wall-seconds', 'joint-rows', 'q7', 'q8', 'qd7', 'qd8', 'initial-note',
    'repeat-detail', 'timing-scope', 'depth-warning', 'episode-warnings',
    'recording-link', 'capture-link', 'audit-link', 'playback-note', 'evidence-description'
  ].map((id) => [id, byId(id)]));
  const integer = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 });
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let study = null;
  let episode = null;
  let index = 0;
  let playing = false;
  let animation = 0;
  let playbackStartedAt = 0;
  let playbackStartedTime = 0;
  let loading = false;
  let playbackGeneration = 0;
  const valueCells = [];

  function text(id, value) { ui[id].textContent = value; }
  function number(value, digits = 4) {
    return typeof value === 'number' && Number.isFinite(value)
      ? (Object.is(value, -0) ? 0 : value).toFixed(digits) : '—';
  }
  function count(value) {
    return Number.isSafeInteger(value) && value >= 0 ? integer.format(value) : '—';
  }
  function localPath(path) {
    if (typeof path !== 'string' || !path || /[\\?#\u0000-\u0020]/.test(path)
      || path.startsWith('/') || path.includes(':')
      || path.split('/').some((part) => part === '..' || part === '')) {
      throw new Error('A camera path is not a safe relative file path.');
    }
    return path;
  }
  function finiteArray(value, length) {
    return Array.isArray(value) && value.length === length
      && value.every((item) => typeof item === 'number' && Number.isFinite(item));
  }
  function videoMode() { return Boolean(episode && episode.video); }
  function archiveLink(data) {
    if (!data || data.download_archive_url == null) return null;
    const value = data.download_archive_url;
    if (typeof value !== 'string') throw new Error('The evidence archive link must be a URL or relative file path.');
    if (value.startsWith('https://')) {
      const url = new URL(value);
      if (!url.hostname || url.username || url.password) throw new Error('The evidence archive link is invalid.');
      return value;
    }
    return localPath(value);
  }
  function validate(data) {
    if (!data || !Array.isArray(data.episodes) || data.episodes.length !== 3) {
      throw new Error('The index must contain the three registered episodes.');
    }
    archiveLink(data);
    const ids = new Set();
    for (const item of data.episodes) {
      if (!item || typeof item.id !== 'string' || !/^[a-z0-9][a-z0-9-]*$/.test(item.id)
        || ids.has(item.id) || !Array.isArray(item.rows) || !item.rows.length
        || item.snapshots !== item.rows.length) {
        throw new Error('An episode identity or snapshot count is inconsistent.');
      }
      ids.add(item.id);
      if (item.video != null) localPath(item.video);
      let lastTime = -Infinity;
      item.rows.forEach((row, rowIndex) => {
        if (!row || typeof row.t !== 'number' || !Number.isFinite(row.t)
          || row.t < 0 || (rowIndex > 0 && row.t <= lastTime)
          || !finiteArray(row.measured, 7) || !finiteArray(row.jaw_pair, 2)
          || !finiteArray(row.velocity, 8) || !finiteArray(row.ramped, 7)
          || !(row.issued === null || finiteArray(row.issued, 7))
          || !(row.native === null || finiteArray(row.native, 7))
          || (rowIndex === 0 && (row.issued !== null || row.native !== null))
          || (rowIndex > 0 && (row.issued === null || row.native === null))) {
          throw new Error('A snapshot has an invalid timestamp or coordinate array.');
        }
        localPath(row.rgb);
        localPath(row.depth);
        lastTime = row.t;
      });
    }
    return data;
  }

  function buildTable() {
    for (let coordinate = 0; coordinate < 7; coordinate += 1) {
      const row = document.createElement('tr');
      const label = document.createElement('th');
      label.scope = 'row';
      label.textContent = coordinate < 6 ? `J${coordinate + 1}` : 'Finger (q7)';
      const unit = document.createElement('span');
      unit.className = 'unit';
      unit.textContent = coordinate < 6 ? 'rad' : 'm';
      label.append(unit);
      row.append(label);
      const cells = [];
      for (let column = 0; column < 4; column += 1) {
        const cell = document.createElement('td');
        cell.textContent = '—';
        row.append(cell);
        cells.push(cell);
      }
      valueCells.push(cells);
      ui['joint-rows'].append(row);
    }
  }

  function setPlaying(nextPlaying) {
    const generation = ++playbackGeneration;
    playing = Boolean(nextPlaying && episode);
    window.cancelAnimationFrame(animation);
    animation = 0;
    if (!playing) ui['frame-video'].pause();
    text('play', playing ? 'Pause' : 'Play');
    ui.play.setAttribute('aria-pressed', String(playing));
    ui.play.setAttribute('aria-label', playing ? 'Pause recorded frames' : 'Play recorded frames');
    if (playing && episode) {
      if (index === episode.rows.length - 1) render(0);
      if (videoMode()) {
        const started = ui['frame-video'].play();
        if (started && typeof started.catch === 'function') {
          started.catch(() => {
            if (generation !== playbackGeneration) return;
            setPlaying(false);
            text('image-error', 'Video playback could not start. Try Play again, or download the original evidence archive.');
            ui['image-error'].hidden = false;
          });
        }
      }
      playbackStartedAt = performance.now();
      playbackStartedTime = episode.rows[index].t;
      animation = window.requestAnimationFrame(tick);
    }
  }

  function tick(now) {
    if (!playing || !episode) return;
    const targetTime = videoMode() ? ui['frame-video'].currentTime
      : playbackStartedTime + (now - playbackStartedAt) / 1000;
    const nextIndex = indexAt(targetTime);
    if (nextIndex !== index) render(nextIndex, false);
    if (targetTime >= episode.rows[episode.rows.length - 1].t) setPlaying(false);
    else animation = window.requestAnimationFrame(tick);
  }

  function indexAt(time) {
    let low = 0;
    let high = episode.rows.length - 1;
    while (low < high) {
      const middle = Math.ceil((low + high) / 2);
      if (episode.rows[middle].t <= time) low = middle;
      else high = middle - 1;
    }
    return low;
  }

  function seekVideo(time) {
    // Seeking is repeated after metadata arrives if a browser has not opened the file yet.
    try { ui['frame-video'].currentTime = time; } catch (_) { /* loadedmetadata retries */ }
  }

  function render(nextIndex, seek = true) {
    if (!episode) return;
    index = Math.max(0, Math.min(episode.rows.length - 1, nextIndex));
    const row = episode.rows[index];
    ui['image-error'].hidden = true;
    if (videoMode()) {
      if (seek) seekVideo(row.t);
      const archive = archiveLink(study);
      ui['frame-source'].href = archive || localPath(episode.video);
      text('frame-source', archive ? 'Download original evidence ZIP ↗' : 'Open derived MP4 ↗');
      text('camera-label', 'RGB camera · lossy derived MP4');
      ui['frame-video'].setAttribute('aria-label', `Lossy derived RGB video, ${episode.label || episode.id}. Timeline telemetry represents recorded snapshots, not decoded video measurements.`);
      ui['depth-source'].hidden = !archive;
      if (archive) ui['depth-source'].href = archive;
      else ui['depth-source'].removeAttribute('href');
      text('depth-source', 'Download raw depth · original ZIP ↗');
    } else {
      const framePath = localPath(row.rgb);
      ui['frame-image'].alt = `RGB camera, ${episode.label || episode.id}, recorded snapshot ${index + 1} of ${episode.rows.length}, simulation time ${number(row.t, 3)} seconds.`;
      if (ui['frame-image'].getAttribute('src') !== framePath) ui['frame-image'].src = framePath;
      ui['frame-source'].href = framePath;
      text('frame-source', 'Open original RGB PNG ↗');
      text('camera-label', 'RGB camera · recorded PNG');
      ui['depth-source'].hidden = false;
      ui['depth-source'].href = localPath(row.depth);
      text('depth-source', 'Download raw depth NPY ↗');
    }
    text('frame-caption', `Snapshot ${count(index + 1)} / ${count(episode.rows.length)} · t = ${number(row.t, 3)} s`);
    text('time-readout', `${number(row.t, 2)} / ${number(episode.rows[episode.rows.length - 1].t, 2)} s`);
    ui.timeline.value = String(index);
    ui.timeline.setAttribute('aria-valuetext', `Snapshot ${index + 1} of ${episode.rows.length}, ${number(row.t, 3)} seconds`);
    ui.previous.disabled = index === 0;
    ui.next.disabled = index === episode.rows.length - 1;
    valueCells.forEach((cells, coordinate) => {
      [row.measured, row.issued, row.ramped, row.native].forEach((values, column) => {
        cells[column].textContent = values === null ? '—' : number(values[coordinate], coordinate < 6 ? 4 : 5);
        cells[column].title = values === null ? 'No preceding interval (null in recording)' : '';
      });
    });
    text('q7', number(row.jaw_pair[0], 5));
    text('q8', number(row.jaw_pair[1], 5));
    text('qd7', number(row.velocity[6], 5));
    text('qd8', number(row.velocity[7], 5));
    ui['initial-note'].hidden = index !== 0;
  }

  function selectEpisode(item) {
    setPlaying(false);
    episode = item;
    const derived = videoMode();
    ui['frame-image'].hidden = derived;
    ui['frame-video'].hidden = !derived;
    if (derived) {
      // The compact mirror deliberately contains no individual RGB images or depth arrays.
      // Never assign any row image path to a media element in this mode.
      ui['frame-image'].removeAttribute('src');
      ui['frame-video'].src = localPath(item.video);
      ui['frame-video'].load();
    } else {
      ui['frame-video'].removeAttribute('src');
      ui['frame-video'].load();
    }
    ui['show-depth'].disabled = true;
    ui['show-depth'].title = 'Raw depth is an NPY array, no image preview';
    ui['show-rgb'].setAttribute('aria-pressed', 'true');
    ui['show-depth'].setAttribute('aria-pressed', 'false');
    text('playback-note', derived
      ? 'Lossy MP4 derived from the recorded RGB PNGs. Telemetry uses saved snapshots; original RGB pixels and raw depth NPY arrays are in the archive. No physics runs here.'
      : 'Playback of saved snapshots at recorded simulation time. No physics runs in this page.');
    text('evidence-description', derived
      ? 'This compact display uses a lossy derived MP4. The original PNGs, depth NPY arrays and complete recording evidence remain in the downloadable ZIP.'
      : 'Local files for the selected episode and the registered study. Camera paths in this viewer point to the preserved PNGs.');
    ui['episode-list'].querySelectorAll('button').forEach((button) => {
      button.setAttribute('aria-pressed', String(button.dataset.episode === item.id));
    });
    text('episode-title', item.label || item.id);
    const interrupted = item.status === 'interrupted';
    text('episode-status', interrupted ? 'Interrupted prefix' : 'Complete recording');
    ui['episode-status'].classList.toggle('interrupted', interrupted);
    text('episode-counts', `${count(item.snapshots)} / ${count(item.steps)}`);
    text('native-success', item.native_success === true ? 'Observed · same fixed start'
      : item.native_success === false ? 'Not observed' : 'Unknown');
    text('recorded-result', typeof item.recorded_task_result === 'string' ? item.recorded_task_result : 'Unknown');
    text('wall-seconds', typeof item.wall_seconds === 'number' ? `${number(item.wall_seconds, 2)} s` : 'Not reported');
    const warnings = Array.isArray(item.warnings) ? item.warnings.map(String) : [];
    text('episode-warnings', warnings.length ? `Capture warnings: ${warnings.join(' · ')}` : 'See the native capture receipt for retained warnings.');
    const archive = derived ? archiveLink(study) : null;
    ui['recording-link'].href = archive || `evidence/registered/${item.id}/recording.json`;
    ui['capture-link'].href = archive || `evidence/registered/${item.id}/capture.json`;
    text('recording-link', archive ? 'Episode recordings · original ZIP ↗' : 'Episode recording · JSON ↗');
    text('capture-link', archive ? 'Native capture receipts · original ZIP ↗' : 'Native capture receipt · JSON ↗');
    ui['audit-link'].href = `audits/${item.id}-audit.json`;
    ui.timeline.max = String(item.rows.length - 1);
    text('time-start', `${number(item.rows[0].t, 2)} s`);
    text('time-end', `${number(item.rows[item.rows.length - 1].t, 2)} s`);
    ui.play.disabled = item.rows.length < 2;
    render(0);
  }

  function showStudy(data) {
    study = data;
    const summary = study.summary || {};
    const repeat = summary.repeat || {};
    text('repeat-count', `${count(repeat.identical_rgb_and_depth_frame_pairs)} / ${count(repeat.snapshot_pairs)}`);
    text('tracker-count', `${count(summary.missing_tracker_channels)} channels missing`);
    text('repeat-detail', `${count(repeat.identical_rgb_and_depth_frame_pairs)} of ${count(repeat.snapshot_pairs)} RGB and depth frame pairs are byte-identical. Maximum recorded difference: ${number(repeat.max_difference, 6)}. Commands: ${repeat.same_commands === true ? 'identical' : 'not established'}. Native controls: ${repeat.same_controls === true ? 'identical' : 'not established'}. This is repeat readback, not a measure of physical accuracy.`);
    if (typeof summary.timing_scope === 'string') text('timing-scope', summary.timing_scope);
    if (typeof summary.depth_warning === 'string') {
      text('depth-warning', `${summary.depth_warning} Matching saved depth readback does not establish metric depth accuracy.`);
    }
    ui['episode-list'].replaceChildren();
    study.episodes.forEach((item) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'episode-button';
      button.dataset.episode = item.id;
      button.setAttribute('aria-pressed', 'false');
      const label = document.createElement('strong');
      label.textContent = item.label || item.id;
      const meta = document.createElement('span');
      meta.textContent = `${count(item.snapshots)} snapshots${item.status === 'interrupted' ? ' · prefix' : ''}`;
      button.append(label, meta);
      button.addEventListener('click', () => selectEpisode(item));
      ui['episode-list'].append(button);
    });
    selectEpisode(study.episodes[0]);
    ui.report.hidden = false;
    ui['load-state'].hidden = true;
  }

  async function load() {
    if (loading) return;
    loading = true;
    ui.retry.hidden = true;
    ui['load-state'].hidden = false;
    text('load-message', 'Loading the local capture index…');
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 15000);
    try {
      const response = await fetch('./viewer-data.json', { signal: controller.signal });
      if (!response.ok) throw new Error(`Capture index returned HTTP ${response.status}.`);
      showStudy(validate(await response.json()));
    } catch (error) {
      const detail = error && error.name === 'AbortError' ? 'The local capture index took too long to respond.' : String(error.message || error);
      text('load-message', `${detail} Serve this folder over local HTTP, or inspect the linked evidence files directly.`);
      ui.retry.hidden = false;
    } finally {
      window.clearTimeout(timeout);
      loading = false;
    }
  }

  buildTable();
  ui.retry.addEventListener('click', load);
  ui.play.addEventListener('click', () => setPlaying(!playing));
  ui.previous.addEventListener('click', () => { setPlaying(false); render(index - 1); });
  ui.next.addEventListener('click', () => { setPlaying(false); render(index + 1); });
  ui.timeline.addEventListener('input', () => { setPlaying(false); render(Number(ui.timeline.value)); });
  ui['show-rgb'].addEventListener('click', () => render(index));
  ui['frame-image'].addEventListener('error', () => {
    if (videoMode()) return;
    text('image-error', 'This frame could not be loaded. Its recorded file is linked below.');
    ui['image-error'].hidden = false;
  });
  ui['frame-image'].addEventListener('load', () => { if (!videoMode()) ui['image-error'].hidden = true; });
  ui['frame-video'].addEventListener('loadedmetadata', () => {
    if (videoMode()) seekVideo(episode.rows[index].t);
  });
  ui['frame-video'].addEventListener('ended', () => {
    if (videoMode()) {
      const time = ui['frame-video'].currentTime;
      render(indexAt(time), false);
      setPlaying(false);
      if (time + 0.001 < episode.rows[episode.rows.length - 1].t) {
        text('image-error', 'The derived video ended before the final recorded snapshot. Inspect the original evidence archive.');
        ui['image-error'].hidden = false;
      }
    }
  });
  ui['frame-video'].addEventListener('error', () => {
    if (!videoMode()) return;
    setPlaying(false);
    text('image-error', 'The derived video could not be loaded. Download the original evidence archive instead.');
    ui['image-error'].hidden = false;
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) setPlaying(false); });
  reducedMotion.addEventListener('change', () => { if (reducedMotion.matches) setPlaying(false); });
  load(); // Playback always starts paused, including when reduced motion is preferred.
})();

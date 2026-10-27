/* Local DOM diagnostics only: relative asset names, phases and byte counts. */
const transportRequests = new Map();
function transportPaint() {
  const target = $('transport-details');
  if (!target) return;
  const rows = [...transportRequests.entries()];
  const active = rows.filter(([, value]) => !['complete', 'failed'].includes(value.phase));
  const recent = rows.filter(([, value]) => ['complete', 'failed'].includes(value.phase)).slice(-9);
  target.textContent = [...active, ...recent].map(([href, value]) =>
    `${href} · ${value.phase}${value.failedPhase ? ` after ${value.failedPhase}` : ''} · HTTP ${value.status ?? 'pending'} · ${value.received}/${value.expected ?? '?'} bytes`).join('\n') || 'No active requests.';
}
function transportBegin(href, expected) {
  localHref(href);
  transportRequests.set(href, {phase: 'awaiting headers', status: null, received: 0, expected: expected ?? null});
  if (transportRequests.size > 256) transportRequests.delete(transportRequests.keys().next().value);
  transportPaint();
}
function transportPhase(href, phase, changes = {}) {
  if (!href) return;
  const value = transportRequests.get(href) || {status: null, received: 0, expected: null};
  transportRequests.set(href, {...value, ...changes, phase}); transportPaint();
}
function transportBytes(href, received) {
  if (href && transportRequests.has(href)) { transportRequests.get(href).received = received; transportPaint(); }
}
function transportSummary(href) {
  let rows = transportRequests.has(href) ? [[href, transportRequests.get(href)]] :
    [...transportRequests.entries()].filter(([, value]) => !['complete', 'failed'].includes(value.phase)).slice(0, 3);
  return rows.length ? ' Transfer detail: ' + rows.map(([name, value]) =>
    `${name}: ${value.phase}, HTTP ${value.status ?? 'pending'}, ${value.received}/${value.expected ?? '?'} bytes`).join('; ') + '.' : '';
}
function transportFailed(href) {
  const value = transportRequests.get(href);
  if (value && value.phase !== 'complete') transportPhase(href, 'failed', {failedPhase: value.phase});
}

/* Hosting-only transport: reconstruct the original geometry bytes exactly. */
async function geometryChunks(meta) {
  if (!meta.chunks) return binary(meta, 'geometry.bin');
  assert(meta.href === 'geometry.bin', 'Unexpected whole geometry file.');
  integer(meta.bytes, 1, MAX_BYTES, 'whole geometry byte count');
  assert(Array.isArray(meta.chunks) && meta.chunks.length > 0 && meta.chunks.length <= 256, 'Invalid geometry chunk count.');
  let total = 0;
  meta.chunks.forEach((chunk, index) => {
    assert(chunk.href === `geometry-part-${String(index).padStart(3, '0')}.bin`, 'Unexpected geometry chunk order.');
    assert(chunk.compressed_href === chunk.href + '.gz', 'Unexpected compressed geometry chunk.');
    integer(chunk.bytes, 1, 262144, 'geometry chunk byte count');
    integer(chunk.compressed_bytes, 1, 263168, 'compressed geometry chunk byte count');
    assert(typeof chunk.sha256 === 'string' && /^[0-9a-f]{64}$/.test(chunk.sha256), 'Invalid geometry chunk digest.');
    total += chunk.bytes;
  });
  assert(total === meta.bytes, 'Geometry chunks do not cover the original byte count.');
  return deadline('chunked geometry', 120000, async signal => {
    let next = 0, completed = 0;
    const pieces = new Array(meta.chunks.length), group = new AbortController();
    const stop = () => group.abort(signal.reason);
    signal.addEventListener('abort', stop, {once: true});
    try {
      const worker = async () => {
        for (;;) {
          if (group.signal.aborted) throw group.signal.reason;
          const index = next++;
          if (index >= meta.chunks.length) return;
          pieces[index] = await binary(meta.chunks[index], meta.chunks[index].href, group.signal);
          if (group.signal.aborted) throw group.signal.reason;
          completed++;
          $('loading').textContent = `Loading saved geometry · ${completed}/${meta.chunks.length} pieces verified…`;
        }
      };
      await Promise.all(Array.from({length: Math.min(3, meta.chunks.length)}, worker));
      const joined = new Uint8Array(meta.bytes);
      let offset = 0;
      for (const piece of pieces) { joined.set(new Uint8Array(piece), offset); offset += piece.byteLength; }
      return await verifiedBinary(joined.buffer, meta);
    } catch (error) {
      group.abort(error);
      throw error;
    } finally {
      signal.removeEventListener('abort', stop);
      group.abort();
    }
  });
}

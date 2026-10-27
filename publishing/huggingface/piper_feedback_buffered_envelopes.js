/* Explicit compatibility experiment: native buffering for known JSON envelopes. */
async function nativeBufferedEnvelope(result, descriptor, signal) {
  envelopeAssert(descriptor && Number.isInteger(descriptor.bytes) && descriptor.bytes > 0 && descriptor.bytes <= 1048576,
    'Native envelope size is outside the 1 MiB bound.');
  const declared = result.headers?.get('content-length');
  if (declared !== null && declared !== undefined) {
    envelopeAssert(typeof declared === 'string' && /^[0-9]+$/.test(declared), 'Invalid envelope Content-Length.');
    const length = Number(declared);
    envelopeAssert(Number.isSafeInteger(length) && length > 0 && length <= descriptor.bytes && length <= 1048576,
      'Envelope Content-Length exceeds its declared byte bound.');
  }
  envelopeAssert(typeof result.arrayBuffer === 'function', 'Native response buffering is unavailable.');
  transportPhase(descriptor.href, 'native body buffering', {received: 0, expected: descriptor.bytes});
  // The browser buffers before this result is exposed. Unlike the streaming
  // reader, it cannot report incremental bytes or cap an undeclared allocation.
  // Exact known file size and SHA checks still reject changed completed bytes.
  const packed = await abortable(result.arrayBuffer(), signal);
  if (signal?.aborted) throw signal.reason;
  transportBytes(descriptor.href, packed?.byteLength ?? 0);
  envelopeAssert(packed && Number.isInteger(packed.byteLength) && packed.byteLength === descriptor.bytes && packed.byteLength <= 1048576,
    'Native buffered envelope byte count differs from its declared bound.');
  return packed;
}

/* Bounded text transport; original decoded bytes and SHA remain authoritative. */
class EnvelopeValidationError extends Error {}
function envelopeAssert(condition, message) { if (!condition) throw new EnvelopeValidationError(message); }
function preferTextTransport() {
  return location.protocol === 'https:' && (location.host === 'research.dvidia.org' ||
    location.host === HF_SPACE_HOST && location.pathname.startsWith(HF_VIEWER_PATH));
}
async function envelopedBinary(meta, expected, outerSignal) {
  const descriptor = meta.json_envelope;
  envelopeAssert(typeof expected === 'string' && /^(?:geometry-part-[0-9]{3}|poses-[0-9]{2})\.bin$/.test(expected) && meta.href === expected, 'Unexpected text-transport binary path.');
  envelopeAssert(descriptor && Object.keys(descriptor).sort().join(',') === 'bytes,href,sha256' && descriptor.href === expected + '.json', 'Unexpected binary-envelope path or fields.');
  envelopeAssert(Number.isInteger(descriptor.bytes) && descriptor.bytes > 0 && descriptor.bytes <= 1048576 && typeof descriptor.sha256 === 'string' && /^[0-9a-f]{64}$/.test(descriptor.sha256), 'Invalid binary-envelope size or digest.');
  envelopeAssert(Number.isInteger(meta.bytes) && meta.bytes > 0 && meta.bytes <= 786432 && typeof meta.sha256 === 'string' && /^[0-9a-f]{64}$/.test(meta.sha256), 'Decoded binary exceeds the envelope bound.');
  return deadline(descriptor.href, 30000, async signal => {
    const packed = await bounded((await response(descriptor.href, signal, descriptor.bytes)).body, descriptor.bytes, signal, descriptor.href);
    let envelope, text;
    try {
      await verifiedBinary(packed, descriptor, descriptor.href);
      text = new TextDecoder('utf-8', {fatal: true}).decode(packed);
      envelope = JSON.parse(text);
    } catch (error) { throw new EnvelopeValidationError('Binary envelope bytes, digest or JSON are invalid.'); }
    transportPhase(descriptor.href, 'decoding text');
    envelopeAssert(envelope && !Array.isArray(envelope) && Object.keys(envelope).sort().join(',') === 'bytes,data,encoding,schema,sha256', 'Unexpected binary-envelope fields.');
    envelopeAssert(envelope.schema === 'dvidia.binary-text-envelope/v1' && envelope.encoding === 'base64' && envelope.bytes === meta.bytes && envelope.sha256 === meta.sha256, 'Envelope identity differs from the original raw binary.');
    envelopeAssert(typeof envelope.data === 'string' && envelope.data.length === 4 * Math.ceil(meta.bytes / 3) && /^[A-Za-z0-9+/]*={0,2}$/.test(envelope.data), 'Malformed or oversized base64 payload.');
    const padding = (3 - meta.bytes % 3) % 3;
    envelopeAssert(envelope.data.length - envelope.data.replace(/=+$/, '').length === padding, 'Invalid base64 padding.');
    envelopeAssert(JSON.stringify({bytes: envelope.bytes, data: envelope.data, encoding: envelope.encoding, schema: envelope.schema, sha256: envelope.sha256}) + '\n' === text, 'Envelope must use its canonical finite JSON encoding.');
    let decoded;
    try { decoded = atob(envelope.data); } catch (error) { throw new EnvelopeValidationError('Base64 payload could not be decoded.'); }
    envelopeAssert(decoded.length === meta.bytes && btoa(decoded) === envelope.data, 'Decoded length or canonical base64 bits changed.');
    const bytes = new Uint8Array(decoded.length);
    for (let i = 0; i < decoded.length; i++) bytes[i] = decoded.charCodeAt(i);
    try { return await verifiedBinary(bytes.buffer, meta, descriptor.href); }
    catch (error) { throw new EnvelopeValidationError('Decoded binary SHA256 differs from the original raw bytes.'); }
  }, outerSignal);
}

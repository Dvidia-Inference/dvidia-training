"""Precompress sealed completed episodes without changing the running study.

Cache receipts are advisory. Final packaging rereads original bytes and every
ZIP member against its own final inventory, then checks the private copy again.
No simulator, policy, credentials or network client is executed.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import sys
import zipfile
import zlib

import prepare_camera_action_v01 as evidence

SCHEMA = 'dvidia.camera-action-archive-cache/v1'
STAGES = ('diagnostic', 'demonstrations', 'development-initial', 'corrections',
          'development-corrected', 'confirmation', 'negative')
MAX_ATTEMPTS = 452


def key(inventory, prefix):
    evidence.relative(prefix)
    for name, record in inventory.items():
        evidence.relative(name)
        evidence.checked_identity(record)
    return sha256(evidence.canonical(dict(schema=SCHEMA, prefix=prefix, files=inventory,
        zip_format='ZIP64-DEFLATE', level=evidence.ZIP_LEVEL, timestamp=[1980, 1, 1, 0, 0, 0],
        file_mode=stat.S_IFREG | 0o644))).hexdigest()


def archive_headroom(inventory, prefix):
    """Conservative DEFLATE/ZIP64 expansion, headers and metadata allowance."""
    key(inventory, prefix)
    raw = sum(row['bytes'] for row in inventory.values())
    headers = sum(512 + 2 * len((prefix + '/' + name).encode('utf-8')) for name in inventory)
    return raw + raw // 100 + headers + 1024 * 1024


def _layout(path, members, central_offset):
    """Account for every archive byte; permit only this writer's ZIP64 layout."""
    positions = {}; cursor = 0; central_parts = []
    with path.open('rb') as stream:
        for item in members:
            encoded = item.filename.encode('utf-8' if item.flag_bits == 0x800 else 'cp437')
            extra = struct.pack('<HHQQ', 1, 16, item.file_size, item.compress_size)
            header = struct.pack('<4s5H3L2H', b'PK\x03\x04', 45, item.flag_bits,
                8, 0, 33, item.CRC, 0xffffffff, 0xffffffff, len(encoded), len(extra)) + encoded + extra
            if item.header_offset != cursor:
                raise ValueError('Cached ZIP contains a prefix, gap or reordered local record.')
            stream.seek(cursor)
            if stream.read(len(header)) != header:
                raise ValueError('Cached ZIP local metadata differs from the reviewed ZIP64 writer.')
            positions[item.filename] = cursor + len(header)
            central_parts.append(struct.pack('<4s6H3L5H2L', b'PK\x01\x02', (3 << 8) | 45,
                45, item.flag_bits, 8, 0, 33, item.CRC, item.compress_size, item.file_size,
                len(encoded), 0, 0, 0, 0, (stat.S_IFREG | 0o644) << 16, cursor) + encoded)
            cursor += len(header) + item.compress_size
        if cursor != central_offset:
            raise ValueError('Cached ZIP contains unreviewed bytes before its central directory.')
        stream.seek(cursor)
        central = b''.join(central_parts)
        if stream.read(len(central)) != central:
            raise ValueError('Cached ZIP central metadata differs from the reviewed writer.')
        end = struct.pack('<4s4H2LH', b'PK\x05\x06', 0, 0, len(members), len(members),
            len(central), cursor, 0)
        if stream.read(len(end)) != end or stream.read(1):
            raise ValueError('Cached ZIP has an archive comment, trailing data or nonstandard end record.')
    return positions


def _member_identity(stream, offset, item, expected):
    stream.seek(offset)
    decoder = zlib.decompressobj(-15)
    digest, size, crc, remaining = sha256(), 0, 0, item.compress_size
    while remaining:
        chunk = stream.read(min(1024 * 1024, remaining))
        if not chunk:
            raise ValueError('Cached ZIP compressed member is truncated.')
        remaining -= len(chunk)
        while True:
            raw = decoder.decompress(chunk, 1024 * 1024)
            size += len(raw)
            if size > expected['bytes'] or decoder.unused_data:
                raise ValueError('Cached ZIP member expands beyond its budget or contains trailing compressed data.')
            digest.update(raw); crc = zlib.crc32(raw, crc)
            chunk = decoder.unconsumed_tail
            if not chunk and len(raw) < 1024 * 1024:
                break
    if (not decoder.eof or crc != item.CRC
            or dict(bytes=size, sha256=digest.hexdigest()) != expected):
        raise ValueError('Cached ZIP member bytes differ from final evidence.')


def validate_archive(path, inventory, prefix):
    """Verify every member against final source identities, never a cache receipt."""
    path = evidence.plain(path)
    expected = {prefix + '/' + name: record for name, record in inventory.items()}
    key(inventory, prefix)
    raw_total = sum(row['bytes'] for row in expected.values())
    if not 2 <= len(expected) <= 9001 or raw_total > evidence.MAX_EPISODE:
        raise ValueError('Cached episode exceeds the bounded file or byte budget.')
    before = evidence.identity(path, maximum=evidence.MAX_EPISODE)
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if (len(members) != len(expected) or [item.filename for item in members] != sorted(expected)):
            raise ValueError('Cached ZIP member roster differs from final evidence.')
        if archive.comment:
            raise ValueError('Cached ZIP archive comments are not reviewed evidence.')
        for item in members:
            evidence.relative(item.filename)
            record = expected[item.filename]
            if (item.file_size != record['bytes'] or item.compress_type != zipfile.ZIP_DEFLATED
                    or item.flag_bits not in (0, 0x800) or item.is_dir() or item.create_system != 3
                    or item.create_version != 45 or item.extract_version != 45 or item.reserved != 0
                    or item.comment or item.extra or item.internal_attr != 0 or item.volume != 0
                    or item.external_attr != (stat.S_IFREG | 0o644) << 16
                    or item.date_time != (1980, 1, 1, 0, 0, 0)):
                raise ValueError('Cached ZIP member type, size or format differs.')
        positions = _layout(path, members, archive.start_dir)
        with path.open('rb') as stream:
            for item in members:
                _member_identity(stream, positions[item.filename], item, expected[item.filename])
    if evidence.identity(path, maximum=evidence.MAX_EPISODE) != before:
        raise ValueError('Cached ZIP changed during verification.')
    return dict(**before, files=len(expected), unpacked_bytes=raw_total,
        compression_level=None, compression_method='DEFLATE',
        compression_level_scope='Not inferable from a verified DEFLATE stream; cache receipt is not trusted.',
        inventory_sha256=sha256(evidence.canonical(expected)).hexdigest())


def validate_sources(root, inventory):
    """Recheck all original bytes even if cached archive members match a manifest."""
    if evidence.episode_inventory(root) != inventory:
        raise ValueError('Final source inventory differs from the cache lookup inventory.')
    for name, record in inventory.items():
        path = evidence.plain(Path(root) / name)
        if evidence.identity(path, secret_scan=path.suffix in evidence.TEXT_SUFFIXES) != record:
            raise ValueError('Original episode bytes differ from final evidence.')


def _private_copy(source, destination, record):
    # Isolate the existing publication helper's historical flat imports from
    # this process. Calling freeze_file performs no credential lookup or upload.
    helper = Path(__file__).resolve().parent / 'huggingface'
    script = """import json, sys
sys.path.insert(0, sys.argv[1])
from publish_camera_action_v01 import freeze_file
method = freeze_file(sys.argv[2], sys.argv[3], {'bytes': int(sys.argv[4]), 'sha256': sys.argv[5]})
print(json.dumps({'method': method}))
"""
    result = subprocess.run([sys.executable, '-I', '-c', script, str(helper), str(source),
        str(destination), str(record['bytes']), record['sha256']], capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise ValueError('Private cached-archive copy failed; source evidence remains unchanged.')
    answer = json.loads(result.stdout)
    if answer.get('method') not in ('apfs_clone', 'stream_copy'):
        raise ValueError('Unknown private cached-archive copy method.')


def reuse(root, inventory, destination, *, prefix, cache):
    """Return normal archive metadata on a fully checked hit, or None on a miss."""
    root, destination, cache = (evidence.plain(p).resolve() for p in (root, destination, cache))
    if (destination.exists() or destination.is_relative_to(root) or cache.is_relative_to(root)
            or root.is_relative_to(cache) or destination.is_relative_to(cache)):
        raise ValueError('Cache and fresh output must be outside the original episode.')
    directory = evidence.plain(cache / key(inventory, prefix))
    if not directory.exists():
        return None
    archive = evidence.plain(directory / 'episode.zip')
    validate_sources(root, inventory)
    checked = validate_archive(archive, inventory, prefix)
    _private_copy(archive, destination, {k: checked[k] for k in ('bytes', 'sha256')})
    copied = validate_archive(destination, inventory, prefix)
    if copied != checked:
        raise ValueError('Private cached archive differs from the verified cache bytes.')
    return copied


def candidates(study):
    """Take a finite snapshot of registered, sealed stage records only."""
    study = evidence.plain(study)
    protocol = evidence.read(study / 'protocol.json')
    protocol_hash = evidence.identity(study / 'protocol.json')['sha256']
    if evidence.read(study / 'registration.json')['protocol_sha256'] != protocol_hash:
        raise ValueError('Study protocol differs from registration.')
    rosters = protocol['rosters']
    if set(rosters) != set(STAGES) or sum(len(rows) for rows in rosters.values()) > MAX_ATTEMPTS:
        raise ValueError('Only the bounded registered camera/action stage roster is accepted.')
    result = []; sealed_raw_bytes = 0
    for stage in STAGES:
        path = study / 'stages' / (stage + '.json')
        seal = path.with_name(stage + '.integrity.json')
        if not path.exists() or not seal.exists():
            continue  # an active stage has not sealed; do not inspect its episodes
        receipt = evidence.read(path)
        if evidence.read(seal) != dict(sha256=evidence.identity(path)['sha256'], protocol_sha256=protocol_hash):
            raise ValueError('Sealed stage identity changed.')
        if (receipt.get('schema') != 'dvidia.camera-action-study/v1/stage'
                or receipt.get('stage') != stage or receipt.get('protocol_sha256') != protocol_hash
                or receipt.get('status') not in ('complete', 'blocked')
                or receipt.get('scheduled') != len(rosters[stage])):
            raise ValueError('Sealed stage registration differs from the protocol.')
        roster = {row['id']: row for row in rosters[stage]}
        rows = receipt['rows']
        if len(roster) != len(rosters[stage]) or len(rows) != len(roster) or {r['id'] for r in rows} != set(roster):
            raise ValueError('Sealed stage differs from its complete registered roster.')
        for row in rows:
            registered = roster[row['id']]
            if any(row.get(k) != registered.get(k) for k in ('case', 'controller', 'negative_control')):
                raise ValueError('Episode identity differs from its registered scene/controller.')
            if row.get('status') in ('unknown', 'unattempted'):
                continue
            prefix = f'episodes/{stage}/{row["id"]}'
            evidence.relative(prefix)
            if row.get('path') != prefix:
                raise ValueError('Episode path differs from the registered attempt.')
            episode = evidence.plain(study / prefix)
            if (evidence.identity(episode / 'integrity.json')['sha256'] != row['integrity_sha256']
                    or evidence.identity(episode / 'rollout.json')['sha256'] != row['receipt_sha256']):
                raise ValueError('Episode receipt differs from its sealed stage.')
            inventory = evidence.episode_inventory(episode)
            sealed_raw_bytes += sum(item['bytes'] for item in inventory.values())
            rollout = evidence.read(episode / 'rollout.json')
            if rollout['status'] != row['status']:
                raise ValueError('Episode completion differs from its sealed stage.')
            if (rollout['status'] == 'complete' and rollout.get('recorded_complete_intervals') == 1000
                    and row.get('complete') is True and row.get('recorded_complete_intervals') == 1000):
                result.append(dict(root=episode, prefix=prefix, inventory=inventory))
    maximum = protocol.get('storage', {}).get('maximum_study_bytes')
    if type(maximum) is not int or maximum != evidence.MAX_TOTAL or sealed_raw_bytes > maximum:
        raise ValueError('Study storage cap differs from the registered 120 GiB budget.')
    return result, maximum - sealed_raw_bytes


def precompress(study, cache, *, limit=1):
    study, cache = (evidence.plain(p).resolve() for p in (study, cache))
    if type(limit) is not int or not 1 <= limit <= MAX_ATTEMPTS:
        raise ValueError('Choose a finite episode limit between 1 and 452.')
    if cache == study or cache.is_relative_to(study) or study.is_relative_to(cache):
        raise ValueError('Use a separate cache outside the study tree.')
    cache.mkdir(parents=True, exist_ok=True)
    lock = cache / 'worker.lock'
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    created = []; existing = 0
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(dict(pid=os.getpid(), operation='publication_archive_cache'), stream)
        roster, remaining = candidates(study)
        for item in roster:
            if len(created) >= limit:
                break
            directory = evidence.plain(cache / key(item['inventory'], item['prefix']))
            if directory.exists():
                existing += 1
                continue  # no trust claim; final packaging validates the ZIP itself
            archive_bytes_reserved = archive_headroom(item['inventory'], item['prefix'])
            # The original native worker may continue writing. Reserving its
            # entire remaining registered cap is intentionally conservative.
            if shutil.disk_usage(cache).free - remaining - archive_bytes_reserved < evidence.DISK_RESERVE:
                raise ValueError('Cache would consume future registered recording space or the 30 GiB reserve.')
            directory.mkdir(exist_ok=False)
            receipt = evidence.archive(item['root'], item['inventory'], directory / 'episode.zip', prefix=item['prefix'])
            evidence.write(directory / 'receipt.json', dict(schema=SCHEMA, prefix=item['prefix'],
                cache_key=directory.name, archive=receipt, advisory_only=True))
            created.append(dict(prefix=item['prefix'], cache_key=directory.name, archive_bytes=receipt['bytes']))
        return dict(schema=SCHEMA, status='complete', created=created, existing_entries_skipped=existing,
            eligible_episodes=len(roster), remaining_registered_raw_bytes_reserved=remaining,
            physics_executed=False, qualification_changed=False)
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True); parser.add_argument('--cache', required=True)
    parser.add_argument('--limit', type=int, default=1)
    args = parser.parse_args(argv)
    print(json.dumps(precompress(args.study, args.cache, limit=args.limit), indent=2))


if __name__ == '__main__':
    main()

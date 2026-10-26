"""Strict offline intake for platform-owned Skillspace video packages.

The importer copies one hash-bound video into a fresh local directory. It does
not fetch remote data, execute package content, or infer movement readiness.
"""
from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import stat
import argparse
import struct
import tempfile
import uuid
import unicodedata
import zipfile

from .footage_data import FootageDataError, _json

SOURCE_SCHEMA = 'dvidia.skillspace-video.v1'
SOURCE_PACKAGE_LIMIT = 100 * 1024 * 1024 + 64 * 1024
MAX_VIDEO_BYTES = 100 * 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024
_MEDIA = {
    '.mp4': ('video/mp4',),
    '.webm': ('video/webm',),
    '.mov': ('video/quicktime',),
}
_HEX = re.compile(r'^[0-9a-f]{64}$')
_SKILLSPACE_SLUG = re.compile(r'^[a-z0-9]+(?:-[a-z0-9]+)*$')
_EOCD = struct.Struct('<4s4H2LH')
_MAX_CENTRAL_DIRECTORY_BYTES = 4 * 1024


def _fail(message: str) -> None:
    raise FootageDataError(message)


def _text(value, name, maximum):
    if type(value) is not str or not 1 <= len(value) <= maximum or not value.strip():
        _fail(f'{name} must be bounded nonempty text.')
    if any(unicodedata.category(char) == 'Cc' or 0xD800 <= ord(char) <= 0xDFFF for char in value):
        _fail(f'{name} contains unsupported characters.')
    return value


def _datetime(value, name):
    _text(value, name, 80)
    if 'T' not in value:
        _fail(f'{name} must be an ISO datetime with a timezone.')
    try:
        parsed = datetime.fromisoformat(value[:-1] + '+00:00' if value.endswith('Z') else value)
    except ValueError as exc:
        raise FootageDataError(f'{name} must be an ISO datetime with a timezone.') from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail(f'{name} must include a timezone.')
    return value


def _uuid(value, name):
    _text(value, name, 36)
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise FootageDataError(f'{name} must be a UUID.') from exc
    if str(parsed) != value:
        _fail(f'{name} must use canonical lowercase UUID form.')
    return value


def _canonical_uuid(value):
    try:
        return str(uuid.UUID(value)) == value
    except (ValueError, AttributeError):
        return False


def _manifest(value):
    if type(value) is not dict or set(value) != {
            'schema', 'platform_origin', 'exported_at', 'contribution', 'video',
            'processing', 'calibration'}:
        _fail('Manifest fields do not match the Skillspace video contract.')
    if value['schema'] != SOURCE_SCHEMA:
        _fail('Unsupported Skillspace video schema.')
    if value['platform_origin'] != 'https://dvidia.org':
        _fail('Skillspace video origin is not the platform origin.')
    _datetime(value['exported_at'], 'exported_at')

    contribution = value['contribution']
    if type(contribution) is not dict or set(contribution) != {
            'id', 'revision', 'skillspace_id', 'task_title'}:
        _fail('Contribution fields do not match the Skillspace video contract.')
    _uuid(contribution['id'], 'contribution.id')
    _datetime(contribution['revision'], 'contribution.revision')
    skillspace_id = _text(contribution['skillspace_id'], 'contribution.skillspace_id', 100)
    if (skillspace_id != 'unsorted' and not _SKILLSPACE_SLUG.fullmatch(skillspace_id)
            and not (skillspace_id.startswith('personal-') and
                     _canonical_uuid(skillspace_id[len('personal-'):]))):
        _fail('contribution.skillspace_id must be a starter slug, personal UUID, or unsorted.')
    _text(contribution['task_title'], 'contribution.task_title', 200)

    video = value['video']
    if type(video) is not dict or set(video) != {
            'filename', 'original_name', 'content_type', 'bytes', 'sha256'}:
        _fail('Video fields do not match the Skillspace video contract.')
    filename = video['filename']
    if type(filename) is not str or filename not in ('video.mp4', 'video.mov', 'video.webm'):
        _fail('Video filename must be video.mp4, video.mov or video.webm.')
    ext = Path(filename).suffix
    _text(video['original_name'], 'video.original_name', 200)
    if (video['original_name'] in ('.', '..') or '/' in video['original_name']
            or '\\' in video['original_name']):
        _fail('Original video name must be a simple basename.')
    if video['content_type'] not in _MEDIA[ext]:
        _fail('Video content type does not match its filename.')
    if type(video['bytes']) is not int or not 1 <= video['bytes'] <= MAX_VIDEO_BYTES:
        _fail('Video byte count is outside the supported limit.')
    if type(video['sha256']) is not str or not _HEX.fullmatch(video['sha256']):
        _fail('Video SHA-256 must be lowercase hexadecimal.')

    processing = value['processing']
    if type(processing) is not dict or set(processing) != {'scope', 'accepted'}:
        _fail('Processing fields do not match the Skillspace video contract.')
    if processing['scope'] != 'local-only' or processing['accepted'] is not True:
        _fail('Skillspace source must be accepted for local-only processing.')
    if value['calibration'] is not None:
        _fail('Calibration is not accepted in Skillspace video v1 exports.')
    return value


def _path_without_symlinks(path: Path):
    """Reject a path if any existing component is a symlink."""
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            _fail('Symlink paths are not accepted.')


def _read_manifest(raw):
    try:
        return _manifest(_json(raw, maximum=MAX_MANIFEST_BYTES))
    except FootageDataError:
        raise


def _hash_file(path: Path, maximum=MAX_VIDEO_BYTES):
    digest, size = sha256(), 0
    with path.open('rb') as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            if size > maximum:
                _fail('Video exceeds its size limit.')
            digest.update(block)
    return size, digest.hexdigest()


def _preflight_zip(path: Path, file_size: int):
    """Bound central-directory metadata before zipfile builds its entry list."""
    if file_size < _EOCD.size:
        _fail('Skillspace package is shorter than its ZIP end record.')
    try:
        with path.open('rb') as stream:
            stream.seek(-_EOCD.size, 2)
            raw = stream.read(_EOCD.size)
    except OSError as exc:
        raise FootageDataError('Skillspace source package could not be read.') from exc
    if len(raw) != _EOCD.size:
        _fail('Skillspace package has an incomplete ZIP end record.')
    signature, disk, cd_disk, disk_entries, total_entries, cd_size, cd_offset, comment_size = _EOCD.unpack(raw)
    if signature != b'PK\x05\x06' or comment_size != 0:
        _fail('Skillspace package must use the unextended ZIP end record.')
    if disk != 0 or cd_disk != 0 or disk_entries != 2 or total_entries != 2:
        _fail('Skillspace package must be a two-entry single-disk ZIP archive.')
    if cd_size > _MAX_CENTRAL_DIRECTORY_BYTES:
        _fail('Skillspace package central directory exceeds 4 KiB.')
    if cd_offset + cd_size != file_size - _EOCD.size:
        _fail('Skillspace package central directory bounds are invalid.')


def import_skillspace_video(package, output):
    """Import a platform ZIP package into a new, atomically installed folder."""
    package, output = Path(package), Path(output)
    _path_without_symlinks(package)
    try:
        package_stat = package.stat()
    except OSError as exc:
        raise FootageDataError('Skillspace source package is unavailable.') from exc
    if not stat.S_ISREG(package_stat.st_mode):
        _fail('Skillspace source package must be a regular file.')
    if package_stat.st_size > SOURCE_PACKAGE_LIMIT:
        _fail('Skillspace source package exceeds its size limit.')
    _preflight_zip(package, package_stat.st_size)
    parent = output.absolute().parent
    _path_without_symlinks(parent)
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise FootageDataError('Output parent directory is unavailable.') from exc
    _path_without_symlinks(parent)
    _path_without_symlinks(output.absolute())
    if output.exists() or output.is_symlink():
        _fail('Output directory already exists; imports require a fresh target.')

    staging = None
    try:
        staging = Path(tempfile.mkdtemp(prefix='.skillspace-video-', dir=parent))
        with zipfile.ZipFile(package, 'r') as archive:
            infos = archive.infolist()
            if len(infos) != 2:
                _fail('Package must contain exactly manifest.json and one video file.')
            if len({info.filename for info in infos}) != len(infos):
                _fail('Package contains duplicate ZIP entries.')
            by_name = {info.filename: info for info in infos}
            if 'manifest.json' not in by_name:
                _fail('Package is missing manifest.json.')
            for info in infos:
                if info.filename not in ('manifest.json', 'video.mp4', 'video.webm', 'video.mov'):
                    _fail('Package contains an unexpected path or file.')
                mode = info.external_attr >> 16
                file_type = stat.S_IFMT(mode)
                if (info.is_dir() or (file_type and file_type != stat.S_IFREG)
                        or info.flag_bits & 1):
                    _fail('ZIP entries must be regular unencrypted files.')
                if info.compress_type != zipfile.ZIP_STORED:
                    _fail('ZIP entries must use ZIP_STORED without compression.')
                if info.file_size < 0 or info.compress_size != info.file_size:
                    _fail('ZIP entry size metadata is invalid.')
            manifest_info = by_name['manifest.json']
            if manifest_info.file_size > MAX_MANIFEST_BYTES:
                _fail('Manifest exceeds its 16 KiB limit.')
            with archive.open(manifest_info, 'r') as stream:
                manifest_raw = stream.read(MAX_MANIFEST_BYTES + 1)
            if len(manifest_raw) != manifest_info.file_size or len(manifest_raw) > MAX_MANIFEST_BYTES:
                _fail('Manifest size does not match its ZIP metadata.')
            manifest = _read_manifest(manifest_raw)
            expected_video = manifest['video']['filename']
            if set(by_name) != {'manifest.json', expected_video}:
                _fail('Package video entry does not match its manifest filename.')
            video_info = by_name[expected_video]
            declared = manifest['video']['bytes']
            if video_info.file_size != declared or video_info.file_size > MAX_VIDEO_BYTES:
                _fail('Video ZIP size does not match its declared byte count.')
            video_path = staging / expected_video
            digest, copied = sha256(), 0
            with archive.open(video_info, 'r') as source, video_path.open('xb') as target:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    copied += len(block)
                    if copied > declared or copied > MAX_VIDEO_BYTES:
                        _fail('Video expanded data exceeds its declared size.')
                    digest.update(block)
                    target.write(block)
            if copied != declared or digest.hexdigest() != manifest['video']['sha256']:
                _fail('Video size or SHA-256 does not match the manifest.')
            (staging / 'manifest.json').write_bytes(manifest_raw)
        # Recheck immediately before install so a newly created target is never replaced.
        _path_without_symlinks(output.absolute())
        if output.exists() or output.is_symlink():
            _fail('Output directory already exists; imports require a fresh target.')
        staging.rename(output)
        staging = None
        return manifest
    except FootageDataError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError, EOFError, ValueError) as exc:
        raise FootageDataError('Skillspace source package is malformed or could not be read.') from exc
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def inspect_skillspace_video(directory):
    """Validate a saved source directory and return its strict manifest."""
    root = Path(directory)
    _path_without_symlinks(root)
    if not root.is_dir():
        _fail('Skillspace source directory is unavailable.')
    try:
        children = {child.name for child in root.iterdir()}
    except OSError as exc:
        raise FootageDataError('Skillspace source directory is unavailable.') from exc
    if 'manifest.json' not in children:
        _fail('Skillspace source directory is missing manifest.json.')
    manifest_path = root / 'manifest.json'
    _path_without_symlinks(manifest_path)
    try:
        manifest_stat = manifest_path.stat()
        if not stat.S_ISREG(manifest_stat.st_mode) or manifest_stat.st_size > MAX_MANIFEST_BYTES:
            _fail('Saved manifest must be a bounded regular file.')
        with manifest_path.open('rb') as stream:
            raw = stream.read(MAX_MANIFEST_BYTES + 1)
    except OSError as exc:
        raise FootageDataError('Saved manifest could not be read.') from exc
    manifest = _read_manifest(raw)
    filename = manifest['video']['filename']
    if children != {'manifest.json', filename}:
        _fail('Skillspace source directory must contain exactly its manifest and video.')
    video_path = root / filename
    _path_without_symlinks(video_path)
    try:
        mode = video_path.stat().st_mode
        if not stat.S_ISREG(mode):
            _fail('Saved video must be a regular file.')
        size, digest = _hash_file(video_path)
    except OSError as exc:
        raise FootageDataError('Saved video could not be read.') from exc
    if size != manifest['video']['bytes'] or digest != manifest['video']['sha256']:
        _fail('Saved video size or SHA-256 does not match the manifest.')
    return manifest


def _summary(manifest, directory):
    video_path = (Path(directory).absolute() / manifest['video']['filename']).resolve()
    contribution, video = manifest['contribution'], manifest['video']
    return {
        'schema': SOURCE_SCHEMA,
        'contribution_id': contribution['id'],
        'revision': contribution['revision'],
        'skillspace_id': contribution['skillspace_id'],
        'task_title': contribution['task_title'],
        'video_filename': video['filename'],
        'video_bytes': video['bytes'],
        'video_sha256': video['sha256'],
        'video_path': str(video_path),
        'calibration_required': True,
        'noexec': True,
        'processing_scope': 'local-only',
    }


def main(argv=None):
    """CLI entry point for strict offline import and verification."""
    parser = argparse.ArgumentParser(prog='dvidia-skillspace')
    commands = parser.add_subparsers(dest='command', required=True)
    importer = commands.add_parser('import-video', help='import a local platform-owned ZIP package')
    importer.add_argument('--package', required=True)
    importer.add_argument('--output', required=True)
    inspector = commands.add_parser('inspect-video', help='verify an imported local source folder')
    inspector.add_argument('--directory', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'import-video':
            manifest = import_skillspace_video(args.package, args.output)
            result = _summary(manifest, args.output)
        else:
            manifest = inspect_skillspace_video(args.directory)
            result = _summary(manifest, args.directory)
        print(json.dumps(result, sort_keys=True, separators=(',', ':')))
        return 0
    except (FootageDataError, OSError) as exc:
        parser.exit(2, f'{parser.prog}: error: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())

"""Read-only dependency and estimated CPU resource checks, using the standard library.

Package versions are read from installation metadata. NumPy and MuJoCo are never
imported, and no footage, URLs, model weights or simulator tasks are opened.
"""
from __future__ import annotations

import argparse
import ctypes
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

NUMPY_VERSION = '2.5.3'
MUJOCO_VERSION = '3.15.0'
REQUIRED_DEMUXERS = ('mov', 'matroska', 'webm', 'avi')
GIB = 1024 ** 3
QUERY_TIMEOUT_SECONDS = 5
QUERY_OUTPUT_LIMIT = 256 * 1024


def _query(executable, *arguments):
    """Bound system queries in time and reject unexpectedly large output."""
    try:
        result = subprocess.run([executable, *arguments], stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=QUERY_TIMEOUT_SECONDS, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if (result.returncode != 0 or len(result.stdout) + len(result.stderr) > QUERY_OUTPUT_LIMIT):
        return None
    return (result.stdout + b'\n' + result.stderr).decode('utf8', errors='replace')


def _installed_version(name):
    try:
        value = metadata.version(name)
    except (metadata.PackageNotFoundError, OSError, ValueError):
        return None
    return value if isinstance(value, str) and 0 < len(value) <= 100 else None


def _package(name, expected, required):
    version = _installed_version(name)
    return {'name': name, 'required': required, 'ok': version == expected,
            'installed_version': version, 'supported_version': expected,
            'method': 'installation metadata; package not imported'}


def _binary(name):
    path = shutil.which(name)
    check = {'name': name, 'required': True, 'ok': False, 'present': path is not None,
             'version': None, 'required_demuxers': list(REQUIRED_DEMUXERS),
             'supported_demuxers': [], 'missing_demuxers': list(REQUIRED_DEMUXERS)}
    if path is None:
        check['error'] = 'Executable is missing from PATH.'
        return check, None
    version, demuxers = _query(path, '-version'), _query(path, '-hide_banner', '-demuxers')
    if version is None or demuxers is None:
        check['error'] = 'A version or demuxer query failed or exceeded its bound.'
        return check, path
    lines = [line.strip() for line in version.splitlines() if line.strip()]
    if not lines or not lines[0].lower().startswith(name + ' version '):
        check['error'] = 'The executable did not report the expected version header.'
        return check, path
    check['version'] = lines[0][:240]
    enabled = set()
    for match in re.finditer(r'^\s*D\s+([a-zA-Z0-9_,]+)\s', demuxers, flags=re.MULTILINE):
        enabled.update(match.group(1).split(','))
    check['supported_demuxers'] = sorted(set(REQUIRED_DEMUXERS) & enabled)
    check['missing_demuxers'] = sorted(set(REQUIRED_DEMUXERS) - enabled)
    check['ok'] = not check['missing_demuxers']
    if not check['ok']:
        check['error'] = 'Required ordinary video containers are unavailable.'
    return check, path


def _fixture_encoder(ffmpeg):
    listing = _query(ffmpeg, '-hide_banner', '-encoders') if ffmpeg else None
    present = listing is not None and any(
        match.group(1) == 'libx264' for match in re.finditer(
            r'^\s*[VAS][A-Z.]{5}\s+([a-zA-Z0-9_]+)\s', listing, flags=re.MULTILINE))
    return {'name': 'libx264 fixture encoder', 'required': False, 'ok': present,
            'scope': 'Synthetic MP4 example creation only; existing-footage training does not encode video.'}


def _positive_bytes(value):
    if type(value) is int and 0 < value <= 1024 * 1024 * GIB:
        return value
    return None


def _ram_bytes(system):
    """Physical capacity, not an assertion about currently available job memory."""
    try:
        if system == 'Darwin':
            value = _query('/usr/sbin/sysctl', '-n', 'hw.memsize')
            return _positive_bytes(int(value.strip())) if value is not None else None
        if system == 'Windows':
            class MemoryStatus(ctypes.Structure):
                _fields_ = [('dwLength', ctypes.c_uint32), ('dwMemoryLoad', ctypes.c_uint32),
                            ('ullTotalPhys', ctypes.c_uint64), ('ullAvailPhys', ctypes.c_uint64),
                            ('ullTotalPageFile', ctypes.c_uint64), ('ullAvailPageFile', ctypes.c_uint64),
                            ('ullTotalVirtual', ctypes.c_uint64), ('ullAvailVirtual', ctypes.c_uint64),
                            ('ullAvailExtendedVirtual', ctypes.c_uint64)]
            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(status)
            function = ctypes.windll.kernel32.GlobalMemoryStatusEx
            function.argtypes = [ctypes.POINTER(MemoryStatus)]
            function.restype = ctypes.c_int
            return _positive_bytes(status.ullTotalPhys) if function(ctypes.byref(status)) else None
        if system == 'Linux':
            pages, page_size = os.sysconf('SC_PHYS_PAGES'), os.sysconf('SC_PAGE_SIZE')
            if type(pages) is int and type(page_size) is int and pages > 0 and page_size > 0:
                return _positive_bytes(pages * page_size)
    except (AttributeError, OSError, ValueError, OverflowError):
        pass
    return None


def _disk(directory):
    """Inspect the nearest existing directory without creating output files."""
    try:
        candidate = Path(directory).expanduser().absolute()
        while not candidate.exists() and candidate != candidate.parent:
            candidate = candidate.parent
        if not candidate.is_dir():
            candidate = candidate.parent
        value = shutil.disk_usage(candidate).free
        return {'checked_directory': str(candidate), 'free_bytes': value if type(value) is int and value >= 0 else None}
    except (OSError, ValueError):
        return {'checked_directory': None, 'free_bytes': None}


def diagnose(directory=None, *, arm=False):
    """Return checks and estimates. Unknown resource capacity never becomes a gate."""
    if type(arm) is not bool:
        raise ValueError('arm must be boolean.')
    python = {'name': 'python', 'required': True, 'ok': sys.version_info >= (3, 12),
              'installed_version': platform.python_version(), 'supported_version': '>=3.12'}
    ffmpeg, ffmpeg_path = _binary('ffmpeg')
    ffprobe, _ = _binary('ffprobe')
    checks = [python, _package('numpy', NUMPY_VERSION, True), ffmpeg, ffprobe,
              _package('mujoco', MUJOCO_VERSION, arm), _fixture_encoder(ffmpeg_path)]
    system = platform.system()
    cpu_count = os.cpu_count()
    memory = _ram_bytes(system)
    disk = _disk(Path.cwd() if directory is None else directory)
    warnings = []
    if memory is None:
        warnings.append('Physical RAM capacity is unknown; no RAM requirement was inferred.')
    elif memory < 2 * GIB:
        warnings.append('Physical RAM is below the unverified 2 GiB starter estimate; try fewer frames and smaller resolution.')
    if disk['free_bytes'] is None:
        warnings.append('Scratch disk capacity is unknown; no disk requirement was inferred.')
    elif disk['free_bytes'] < GIB:
        warnings.append('Free disk is below the unverified 1 GiB starter scratch estimate.')
    if not checks[-1]['ok']:
        warnings.append('libx264 is unavailable or could not be queried; synthetic MP4 example creation may fail.')
    failed = [check['name'] for check in checks if check['required'] and not check['ok']]
    return {'schema_version': 1, 'kind': 'dvidia.training-doctor',
            'status': 'missing-requirements' if failed else 'dependencies-present',
            'mode': 'arm' if arm else 'visual', 'ok': not failed, 'failed_requirements': failed,
            'checks': checks, 'hardware': {'system': system, 'architecture': platform.machine(),
                'logical_cpu_count': cpu_count if type(cpu_count) is int and cpu_count > 0 else None,
                'physical_ram_bytes': memory, 'disk': disk, 'gpu_required': False, 'max_active_jobs': 1},
            'sizing_estimates': {'verified_minimum': False, 'scope': 'Small example at default training settings.',
                'starter': {'logical_cores': 1, 'ram_bytes': 2 * GIB, 'scratch_bytes': GIB},
                'recommended': {'logical_cores': [2, 4], 'ram_bytes': [4 * GIB, 8 * GIB], 'scratch_bytes': 2 * GIB}},
            'warnings': warnings,
            'limitations': ['Package metadata and FFmpeg listings do not prove that each runtime import or media codec works.',
                'Physical RAM may exceed memory available to a container or job; this check does not measure peak working memory.',
                'Sizing estimates are not measured hardware minimums or guarantees for the largest allowed dataset.',
                'This tool does not run training, change files, use the network, or qualify a robot.',
                'GPU acceleration, multiple nodes and increased worker concurrency are not qualified.']}


def _human(report):
    lines = ['Dependency check: ' + ('passed' if report['ok'] else 'missing requirements') + ' (' + report['mode'] + ').']
    for check in report['checks']:
        label = 'PASS' if check['ok'] else ('FAIL' if check['required'] else 'OPTIONAL')
        version = check.get('installed_version', check.get('version'))
        lines.append(f"{label}: {check['name']}" + (f' — {version}' if version else ''))
    hardware = report['hardware']
    ram = hardware['physical_ram_bytes']
    free = hardware['disk']['free_bytes']
    lines.append(f"CPU: {hardware['architecture']} / {hardware['logical_cpu_count'] or 'unknown'} logical cores. GPU required: no.")
    lines.append('Physical RAM: ' + (f'{ram/GIB:.2f} GiB.' if ram is not None else 'unknown.'))
    lines.append('Free scratch disk: ' + (f'{free/GIB:.2f} GiB.' if free is not None else 'unknown.'))
    lines.append('Small-example estimates: 1 core / 2 GiB RAM / 1 GiB scratch; prefer 2–4 cores / 4–8 GiB RAM.')
    lines.append('A measured hardware minimum has not been established. One active CPU job; runtime/media validation remains separate.')
    lines.extend('Note: ' + warning for warning in report['warnings'])
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true', help='Print structured checks and hardware estimates.')
    parser.add_argument('--directory', type=Path, default=None, help='Inspect free scratch space here or at its nearest existing parent.')
    parser.add_argument('--arm', action='store_true', help='Also require the pinned MuJoCo package metadata for simulation training.')
    arguments = parser.parse_args(argv)
    report = diagnose(arguments.directory, arm=arguments.arm)
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) if arguments.json else _human(report))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

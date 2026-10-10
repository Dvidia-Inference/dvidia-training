"""Generate only DVIDIA-authored schematic video and annotation fixtures."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess

WIDTH, HEIGHT, FPS, SECONDS = 640, 360, 20, 8


def rectangle(image, x, y, width, height, color):
    row = bytes(color) * width
    for yy in range(y, y + height):
        image[(yy * WIDTH + x) * 3:(yy * WIDTH + x + width) * 3] = row


def object_box(time):
    x = int(96 + 344 * min(time / 5.5, 1))
    return [x, 172, x + 64, 236]


def frame(time, occluded):
    image = bytearray(bytes([245, 244, 239]) * WIDTH * HEIGHT)
    rectangle(image, 32, 80, 576, 228, [224, 226, 220])
    for x, y, w, h in ((430, 153, 100, 3), (430, 253, 100, 3),
                       (430, 153, 3, 103), (527, 153, 3, 103)):
        rectangle(image, x, y, w, h, [76, 124, 91])
    x1, y1, x2, y2 = object_box(time)
    rectangle(image, x1, y1, x2 - x1, y2 - y1, [61, 106, 191])
    if occluded:
        rectangle(image, 245, 108, 142, 173, [195, 194, 183])
        for x in range(248, 385, 12):
            rectangle(image, x, 110, 3, 169, [214, 214, 205])
    return image


def generate(output):
    output = Path(output).absolute()
    if any(p.is_symlink() for p in (output, *output.parents)) or output.exists():
        raise ValueError('Use a fresh plain fixture directory.')
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        raise ValueError('Preparing schematic videos requires installed FFmpeg with libx264.')
    output.mkdir(parents=True)
    version = subprocess.run([ffmpeg, '-version'], check=True, capture_output=True, text=True).stdout.splitlines()[0]
    fixtures = []
    for name, occluded in [('visible', False), ('occluded', True)]:
        video = output / (name + '.mp4')
        args = [ffmpeg, '-v', 'error', '-nostdin', '-threads', '1', '-f', 'rawvideo',
                '-pix_fmt', 'rgb24', '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS),
                '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '1',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-map_metadata', '-1', str(video)]
        process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            for index in range(FPS * SECONDS):
                process.stdin.write(frame(index / FPS, occluded))
            process.stdin.close()
            error = process.stderr.read()
            if process.wait(timeout=30) != 0:
                raise ValueError('Synthetic video encoding failed.')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        source_hash = sha256(video.read_bytes()).hexdigest()
        observations = []
        intervals = [(0, 8)] if not occluded else [(0, 1.5), (5, 8)]
        for number, (start, end) in enumerate(intervals, 1):
            boxes = []
            for index in range(SECONDS * 2):
                t = index / 2
                if not start <= t < end:
                    continue
                bounds = object_box(t)
                boxes.append({'time_seconds': t, 'xyxy': [bounds[0] / WIDTH, bounds[1] / HEIGHT,
                             bounds[2] / WIDTH, bounds[3] / HEIGHT], 'label': 'blue shape · authored'})
            observations.append({'id': f'sample-{number}', 'label': 'Blue shape visible · authored example',
                'start_seconds': start, 'end_seconds': end, 'boxes': boxes,
                'evidence': 'authored_schematic_example', 'review_state': 'proposed'})
        observations.append({'id': 'region', 'label': 'Marked region visible · authored example',
            'start_seconds': 0, 'end_seconds': 8, 'boxes': [
                {'time_seconds': i / 2, 'xyxy': [430 / WIDTH, 153 / HEIGHT, 530 / WIDTH, 256 / HEIGHT],
                 'label': 'marked region · authored'} for i in range(SECONDS * 2)],
            'evidence': 'authored_schematic_example', 'review_state': 'proposed'})
        episode = {'kind': 'dvidia.observation-demo-episode.v1', 'id': name,
            'title': 'Blue shape crosses an occluder' if occluded else 'Blue shape moves into a marked region',
            'video_url': 'demo/' + name + '.mp4', 'duration_seconds': SECONDS,
            'source': {'kind': 'synthetic', 'sha256': source_hash, 'credit': 'DVIDIA contributors · authored geometric fixture',
                'license': 'MIT', 'session_id': 'synthetic-demo-one-dependent-group',
                'source_recording_id': 'authored-' + name, 'source_url': 'generate_demo.py',
                'notes': 'No person, robot, camera capture, detector or learned action is represented.'},
            'recipe': {'id': 'dvidia-authored-schematic-demo-v1',
                'model': {'kind': 'Authored fixture · No model ran', 'models': []},
                'notes': 'Deterministic example annotations derived from the drawing recipe. No live inference, accuracy score, task segmentation or robot skill is demonstrated.'},
            'observations': observations,
            'review': {'revision': 0, 'status': 'unreviewed', 'outcome': 'unknown', 'events': [
                {'id': o['id'], 'observation_id': o['id'], 'label': o['label'],
                 'start_seconds': o['start_seconds'], 'end_seconds': o['end_seconds'],
                 'decision': 'pending'} for o in observations]}}
        (output / (name + '.json')).write_text(json.dumps(episode, indent=2) + '\n')
        fixtures.append({'id': name, 'title': episode['title'], 'duration_seconds': SECONDS,
                         'video_url': episode['video_url'], 'status': 'unreviewed'})
    (output / 'episodes.json').write_text(json.dumps({'episodes': fixtures, 'failures': []}, indent=2) + '\n')
    files = [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha256(p.read_bytes()).hexdigest()}
             for p in sorted(output.iterdir()) if p.is_file()]
    (output / 'provenance.json').write_text(json.dumps({'kind': 'dvidia.authored-demo-fixtures.v1',
        'generation': 'stdlib RGB rectangle drawing piped to local FFmpeg/libx264; no source footage input',
        'generator_sha256': sha256(Path(__file__).read_bytes()).hexdigest(), 'ffmpeg': version,
        'dimensions': [WIDTH, HEIGHT], 'fps': FPS, 'seconds_per_clip': SECONDS,
        'annotations': 'authored image rectangles every 0.5 seconds; no model inference',
        'occlusion': 'box annotations deliberately absent when the blue shape is behind the drawn panel',
        'independent_source_groups': 1, 'files': files}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    generate(parser.parse_args().output)

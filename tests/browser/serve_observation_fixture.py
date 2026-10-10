"""Synthetic-only fixture server for browser acceptance tests; never model evidence."""
import argparse
import json
from pathlib import Path
import subprocess

from dvidia_training.observation_pipeline import run
from dvidia_training.observation_studio import ObservationServer
from dvidia_training.observations import file_hash


class SyntheticDetector:
    metadata = {'kind': 'authored-browser-test-only', 'device': 'none'}

    def detect(self, _):
        return [
            {'label': 'hand', 'score': .7, 'xyxy': [.1,.4,.25,.7]},
            {'label': 'hand', 'score': .7, 'xyxy': [.7,.4,.85,.7]},
            {'label': 'cup', 'score': .6, 'xyxy': [.4,.2,.6,.5]},
        ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8283)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    video = args.directory/'synthetic.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=s=320x240:r=20:d=3',
                    '-c:v','libx264','-pix_fmt','yuv420p',str(video)],check=True)
    rows = [dict(id=eid, path=video.name, title='Synthetic browser fixture', session_id='test-session',
        source_recording_id='synthetic-test', kind='synthetic', credit='DVIDIA test fixture',
        license='MIT', source_url='local-synthetic-fixture', permission='local_analysis', sha256=file_hash(video))
        for eid in ('first','second')]
    manifest = args.directory/'source.json'
    manifest.write_text(json.dumps(dict(kind='dvidia.observation-source.v1',task='Browser workflow only',episodes=rows)))
    run(manifest,args.directory/'run',SyntheticDetector())
    print(f'Synthetic fixture: http://127.0.0.1:{args.port}',flush=True)
    ObservationServer(('127.0.0.1',args.port),args.directory/'run').serve_forever()


if __name__ == '__main__':
    main()

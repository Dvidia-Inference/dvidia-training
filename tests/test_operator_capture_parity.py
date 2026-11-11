"""Data-only regression tests for the explicit lossless-PNG benchmark mode."""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import importlib.util
import io
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zlib

import numpy as np

from dvidia_training import operator_capture as capture
from dvidia_training import camera_action_capture as camera
from dvidia_training import piper_action_capture as files
from tests.test_camera_action_capture import FakeSession, fake_estimate, CASE, STUDENT


SPEC = importlib.util.spec_from_file_location('operator_parity_script',
    Path(__file__).parents[1]/'scripts/compare-operator-captures.py')
parity = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(parity)


def png(rgb, level):
    def chunk(kind, data):
        return struct.pack('>I', len(data))+kind+data+struct.pack('>I', zlib.crc32(kind+data)&0xffffffff)
    h, w, _ = rgb.shape
    rows = b''.join(b'\0'+row.tobytes() for row in rgb)
    return (b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR', struct.pack('>IIBBBBB',w,h,8,2,0,0,0))
            +chunk(b'IDAT',zlib.compress(rows,level))+chunk(b'IEND',b''))


class PixelSession(FakeSession):
    level = 6
    change = None

    def open(self):
        header, targets, metadata = super().open()
        calibration = header['calibration']
        calibration['intrinsics'].update(width=320,height=240,cx=160.,cy=120.)
        if self.change == 'calibration':
            calibration['intrinsics']['fx'] += 1.
        header['binding']['calibration_sha256'] = capture.tape.digest(calibration)
        config = dict(capture=asdict(self.config), task={'physics_dt_seconds':.001},
            camera={'fixture':'synthetic RGB pixels, no physics'},
            implementation_sha256={name:sha256((Path(files.__file__).parent/name).read_bytes()).hexdigest()
                                   for name in files.SOURCES})
        if self.change == 'configuration':
            config['task']['physics_dt_seconds'] = .002
        header['binding']['configuration_sha256'] = capture.tape.digest(config)
        return header, targets, {**metadata,'configuration':config}

    def sample(self, index):
        snapshot, _, diagnostic, timing = super().sample(index)
        y, x = np.indices((240,320))
        rgb = np.stack((x%256,y%256,(x+y+index)%256),axis=-1).astype(np.uint8)
        if self.change == 'pixel':
            rgb[7,9,1] ^= 1
        depth = np.full((240,320),.5,dtype='<f4')
        if self.change == 'depth':
            depth[7,9] = np.nextafter(depth[7,9],np.float32(1.))
        depth_bytes = io.BytesIO(); np.save(depth_bytes,depth,allow_pickle=False)
        frames = {}
        for channel, suffix, body in [('rgb','png',png(rgb,self.level)),
                                      ('depth','npy',depth_bytes.getvalue())]:
            name = f'frames/{channel}-{index:04}.{suffix}'
            snapshot['camera']['value'][channel] = dict(path=name,bytes=len(body),sha256=sha256(body).hexdigest())
            frames[name] = body
        self.last_camera = deepcopy(snapshot['camera'])
        return snapshot,frames,diagnostic,timing


class OperatorCaptureParityTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        FakeSession.fail_at = None
        self.enterContext(patch.object(camera,'_Session',PixelSession))
        self.enterContext(patch.object(camera,'_fresh_estimate',fake_estimate))
        self.enterContext(patch.object(camera,'_native_episode',lambda *a:{'test_only':True}))
        self.enterContext(patch.object(camera,'_score',lambda *a:dict(valid=False,success=False,reason='no_physics_fixture')))

    def recording(self, name, *, level=6, change=None):
        with patch.object(PixelSession,'level',level), patch.object(PixelSession,'change',change):
            recorder = capture.OperatorCapture('unused',Path(self.temp.name)/name,
                case=CASE,duration_s=.1,source_kind='simulation_scripted_input')
            recorder.open()
            for index in range(4 if change == 'lifecycle' else 5):
                action = deepcopy(STUDENT)
                if change == 'action': action['joint_targets_rad'][0] += .001
                recorder.step(action,input_event={'fixture_index':index})
            recorder.finish('fixture_stopped')
        return recorder.root

    def test_default_keeps_exact_bytes_and_opt_in_requires_exact_decoded_pixels(self):
        baseline = self.recording('baseline')
        identical = self.recording('identical')
        self.assertEqual(parity.compare(baseline,identical)['comparison_mode'],'exact-artifact-bytes')
        compressed = self.recording('compressed',level=1)
        with self.assertRaises(ValueError): parity.compare(baseline,compressed)
        result = parity.compare(baseline,compressed,lossless_png=True)
        self.assertTrue(result['exact'])
        self.assertEqual(result['comparison_mode'],'decoded-lossless-png')
        self.assertEqual(result['png_files_with_different_compressed_bytes'],3)
        self.assertNotEqual(*result['frame_bytes'])
        self.assertEqual(result['perception_rows_compared'],6)
        self.assertEqual(result['features_shape'],[5,269])

    def test_pixel_depth_calibration_action_configuration_and_lifecycle_changes_refused(self):
        baseline = self.recording('baseline')
        for change in ('pixel','depth','calibration','action','configuration','lifecycle'):
            with self.subTest(change=change):
                changed = self.recording(change,level=1,change=change)
                with self.assertRaises(ValueError):
                    parity.compare(baseline,changed,lossless_png=True)

    def test_final_perception_snapshot_digest_is_validated_before_normalization(self):
        baseline = self.recording('baseline'); changed = self.recording('changed',level=1)
        path = changed/'perception/0005.json'
        item = capture._read(changed,'perception/0005.json')
        item['snapshot_sha256'] = '0'*64
        body = capture.tape.canonical(item)+b'\n'; path.write_bytes(body)
        # Re-sealing only file integrity cannot legitimize a wrong snapshot binding.
        manifest = capture._read(changed,'integrity.json')
        manifest['files']['perception/0005.json'] = dict(bytes=len(body),sha256=sha256(body).hexdigest())
        (changed/'integrity.json').write_bytes(capture.tape.canonical(manifest)+b'\n')
        with self.assertRaisesRegex(ValueError,'snapshot/frame/calibration binding'):
            parity.compare(baseline,changed,lossless_png=True)

    def test_configuration_source_exclusion_is_narrow_and_original_digest_required(self):
        root = self.recording('baseline')
        recording = capture.load_capture(root)
        context = capture._read(root,'context.json')
        original = parity._configuration(root,recording,normalize_sources=True)
        # A non-overlapping declared native implementation identity can change;
        # physical configuration must still compare exactly afterward.
        config = context['metadata']['configuration']
        config['implementation_sha256']['reference_sim.py'] = 'a'*64
        (root/'context.json').write_bytes(capture.tape.canonical(context)+b'\n')
        with self.assertRaisesRegex(ValueError,'original binding'):
            parity._configuration(root,recording,normalize_sources=True)
        recording['binding']['configuration_sha256'] = capture.tape.digest(config)
        self.assertEqual(original,parity._configuration(root,recording,normalize_sources=True))
        config['task']['physics_dt_seconds'] = .002
        (root/'context.json').write_bytes(capture.tape.canonical(context)+b'\n')
        recording['binding']['configuration_sha256'] = capture.tape.digest(config)
        self.assertNotEqual(original,parity._configuration(root,recording,normalize_sources=True))
        config['implementation_sha256']['new_unrecognized_source.py'] = 'b'*64
        (root/'context.json').write_bytes(capture.tape.canonical(context)+b'\n')
        recording['binding']['configuration_sha256'] = capture.tape.digest(config)
        with self.assertRaisesRegex(ValueError,'source inventory'):
            parity._configuration(root,recording,normalize_sources=True)

    def test_png_crc_dimensions_filter_bounds_and_trailing_payload_rejected(self):
        pixels = np.zeros((24,32,3),dtype=np.uint8)
        good = png(pixels,1)
        np.testing.assert_array_equal(parity._png_pixels(good,32,24),pixels)
        corrupt = bytearray(good); corrupt[-1] ^= 1
        for body,width,height in ((bytes(corrupt),32,24),(good+b'x',32,24),
                                  (good,33,24),(good[:-3],32,24),(good,641,24)):
            with self.subTest(width=width,bytes=len(body)), self.assertRaises(ValueError):
                parity._png_pixels(body,width,height)
        def replace_pixels(raw, *, suffix=b''):
            data = zlib.compress(raw,1)+suffix
            kind = b'IDAT'
            chunk = struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
            return good[:33]+chunk+good[-12:]
        rows = bytearray(24*(1+32*3)); rows[0] = 1
        with self.assertRaisesRegex(ValueError,'filter-0'):
            parity._png_pixels(replace_pixels(rows),32,24)
        for body in (replace_pixels(bytes(24*(1+32*3)+1)),
                     replace_pixels(bytes(24*(1+32*3)),suffix=b'trailing compressed payload')):
            with self.assertRaisesRegex(ValueError,'pixel length'):
                parity._png_pixels(body,32,24)


if __name__ == '__main__': unittest.main()

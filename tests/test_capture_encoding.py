"""Lossless capture/ownership regressions; no physical or native task claims."""
from contextlib import nullcontext
import io
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

import numpy as np

from dvidia_training import piper_action_capture as capture
from tests.test_piper_action_capture import camera


def decode_png(payload):
    """Independent bounded decoder for this encoder's RGB8/filter-zero contract."""
    assert payload[:8] == b'\x89PNG\r\n\x1a\n'
    offset = 8; chunks = []
    while offset < len(payload):
        size = struct.unpack('>I', payload[offset:offset+4])[0]
        kind = payload[offset+4:offset+8]
        body = payload[offset+8:offset+8+size]
        crc = struct.unpack('>I', payload[offset+8+size:offset+12+size])[0]
        assert crc == zlib.crc32(kind+body) & 0xffffffff
        chunks.append((kind, body)); offset += size+12
    assert offset == len(payload)
    assert [kind for kind, _ in chunks] == [b'IHDR', b'IDAT', b'IEND']
    width, height, bits, colour, compression, filtering, interlace = struct.unpack('>IIBBBBB', chunks[0][1])
    assert (bits, colour, compression, filtering, interlace) == (8, 2, 0, 0, 0)
    assert 0 < width <= 320 and 0 < height <= 240
    raw = zlib.decompress(chunks[1][1])
    assert len(raw) == height*(width*3+1)
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, width*3+1)
    assert not rows[:, 0].any()
    return rows[:, 1:].reshape(height, width, 3), chunks


def make_session():
    """Owned-buffer renderer double following classic render(out=None)."""
    class Renderer:
        scene = SimpleNamespace(camera=[camera()])
        depth = False
        fail_depth = False
        move_clock = False

        def __init__(self):
            self.calls = []; self.outputs = []

        def disable_depth_rendering(self): self.depth = False
        def enable_depth_rendering(self): self.depth = True
        def update_scene(self, *args, **kwargs): pass

        def render(self):
            self.calls.append('depth' if self.depth else 'rgb')
            if self.depth and self.fail_depth:
                raise RuntimeError('Authored depth readback error.')
            if self.depth and self.move_clock:
                session.env.elapsed += .001; session.env.data.time += .001
            value = (np.full((64, 64), .75+len(self.calls)/100, dtype=np.float32)
                     if self.depth else np.full((64, 64, 3), 40+len(self.calls), dtype=np.uint8))
            self.outputs.append(value)
            return value

    session = capture._NativeSession('unused', capture.CaptureConfig(width=64, height=64))
    session.np = np; session.renderer = Renderer(); session.camera = 'test-camera'
    session.native_origin = .5
    session.sim = SimpleNamespace(native_warnings=lambda _: nullcontext())
    session.base = 0; session.tool = 1
    data = SimpleNamespace(time=.5, qpos=np.zeros(15), qvel=np.zeros(14),
        xpos=np.array([[0., 0., 0.], [.1, .2, .3]]),
        xmat=np.array([np.eye(3).ravel(), np.eye(3).ravel()]))
    session.env = SimpleNamespace(data=data, elapsed=0.)
    session.mj = SimpleNamespace(mju_mat2Quat=lambda target, _: target.__setitem__(slice(None), [1., 0., 0., 0.]))
    session.calibration = session._calibration(); session.info = {'valid': True}
    session.observation = {'object_position_m': [9., 9., 9.], 'scope': 'privileged test truth'}
    return session


class CaptureEncodingTests(unittest.TestCase):
    def test_fast_png_is_lossless_with_valid_crc_and_unchanged_pixel_order(self):
        rng = np.random.default_rng(31209)
        full = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        patterns = [np.zeros((64, 64, 3), np.uint8),
                    np.full((64, 64, 3), 255, np.uint8),
                    np.arange(64*64*3, dtype=np.uint8).reshape(64, 64, 3),
                    full, full[::2, ::2, ::-1]]
        for rgb in patterns:
            with self.subTest(shape=rgb.shape, contiguous=rgb.flags.c_contiguous):
                decoded, chunks = decode_png(capture._png(rgb))
                np.testing.assert_array_equal(decoded, rgb)
                # The frozen previous encoder compressed these same filtered
                # scanlines at level 6. The compressed artifact changes; no
                # tolerance or approximate image comparison is permitted.
                old_raw = b''.join(b'\0'+row.tobytes() for row in rgb)
                self.assertEqual(zlib.decompress(chunks[1][1]), zlib.decompress(zlib.compress(old_raw, 6)))
                self.assertNotEqual(chunks[1][1], zlib.compress(old_raw, 6))

    def test_later_sample_does_not_mutate_previous_owned_rgb_or_depth(self):
        session = make_session()
        first, first_frames, _, _ = session.sample(0)
        old_rgb, old_depth = session.latest_rgbd
        expected_rgb = old_rgb.copy(); expected_depth = old_depth.copy()
        session.env.elapsed = .04; session.env.data.time = .54
        second, _, _, _ = session.sample(1)
        new_rgb, new_depth = session.latest_rgbd
        self.assertEqual(session.renderer.calls, ['rgb', 'depth', 'rgb', 'depth'])
        self.assertFalse(session.renderer.depth)
        self.assertFalse(np.shares_memory(old_rgb, new_rgb))
        self.assertFalse(np.shares_memory(old_depth, new_depth))
        np.testing.assert_array_equal(old_rgb, expected_rgb)
        np.testing.assert_array_equal(old_depth, expected_depth)
        decoded, _ = decode_png(first_frames[first['camera']['value']['rgb']['path']])
        np.testing.assert_array_equal(decoded, expected_rgb)
        np.testing.assert_array_equal(np.load(io.BytesIO(first_frames[first['camera']['value']['depth']['path']]), allow_pickle=False), expected_depth)
        self.assertEqual(first['timestamp_seconds'], 0.)
        self.assertEqual(second['timestamp_seconds'], .04)

    def test_depth_failure_restores_colour_mode_and_moving_physics_is_refused(self):
        for failure in ('fail_depth', 'move_clock'):
            with self.subTest(failure=failure):
                session = make_session(); setattr(session.renderer, failure, True)
                with self.assertRaises(RuntimeError): session.sample(0)
                self.assertFalse(session.renderer.depth)
                self.assertFalse(hasattr(session, 'latest_rgbd'))

    def test_pinned_classic_renderer_allocates_fresh_arrays_without_output_buffer(self):
        try:
            import mujoco
            from mujoco.rendering.classic.renderer import Renderer
        except ImportError:
            self.skipTest('Optional native dependency absent; renderer allocation checked in the arm environment.')
        # Exercise the installed public Renderer.render implementation without
        # a native scene or graphics context. Only its low-level render/readback
        # calls are replaced; no simulated physics or visual accuracy claimed.
        renderer = Renderer.__new__(Renderer)
        renderer._height = 64; renderer._width = 64
        renderer._depth_rendering = False; renderer._segmentation_rendering = False
        renderer._scene = SimpleNamespace(flags=np.zeros(32, dtype=np.uint8), camera=[camera()])
        context_calls = []
        renderer._mjr_context = object(); renderer._rect = object()
        renderer._gl_context = SimpleNamespace(make_current=lambda: context_calls.append(True))
        renderer._model = SimpleNamespace(stat=SimpleNamespace(extent=1.),
            vis=SimpleNamespace(map=SimpleNamespace(znear=.01, zfar=10.)))
        def pixels(rgb, depth, *_):
            if rgb is not None: rgb[:] = np.arange(64, dtype=np.uint8)[:, None, None]
            if depth is not None: depth[:] = .25
        try:
            with patch.object(mujoco, 'mjr_render'), patch.object(mujoco, 'mjr_readPixels', pixels):
                rgb_a = renderer.render(); rgb_b = renderer.render()
                renderer.enable_depth_rendering()
                depth_a = renderer.render(); depth_b = renderer.render()
            self.assertTrue(all(a.flags.owndata for a in (rgb_a, rgb_b, depth_a, depth_b)))
            self.assertFalse(np.shares_memory(rgb_a, rgb_b))
            self.assertFalse(np.shares_memory(depth_a, depth_b))
            np.testing.assert_array_equal(rgb_a[:, 0, 0], np.arange(63, -1, -1, dtype=np.uint8))
            self.assertEqual(len(context_calls), 4)
            self.assertFalse(renderer._scene.flags.any())
        finally:
            renderer._mjr_context = None; renderer._gl_context = None


if __name__ == '__main__': unittest.main()

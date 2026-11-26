import os
import unittest

import numpy as np

from dvidia_training.tidy_table_native import NativeTable, HEIGHT, WIDTH, _slot, detect


class TableCameraTests(unittest.TestCase):
    def test_primary_colour_only_and_small_patch_is_unknown(self):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[106:130, 146:170, 0] = 200
        frame[30:34, 90:94, 1] = 200
        result = detect(frame)
        self.assertEqual(list(result), ["A"])
        self.assertAlmostEqual(result["A"][0], .347, places=3)
        self.assertIsNone(_slot(result["A"]))
        frame[10:20, 30:40, 0] = 200
        self.assertEqual(detect(frame), {})

    def test_incomplete_layout_and_overlarge_offset_rejected_before_native_load(self):
        with self.assertRaisesRegex(ValueError, "different table slot"):
            NativeTable(None, {"A": "left", "B": "left", "C": "buffer"})
        with self.assertRaisesRegex(ValueError, "Offsets"):
            NativeTable(None, {"A": "left", "B": "center", "C": "right"}, offsets_xy={"C": [99., 0.]})


@unittest.skipUnless(os.environ.get("DVIDIA_TIDY_NATIVE_ASSETS"), "requires pinned PiPER assets and graphics")
class NativeDevelopmentTests(unittest.TestCase):
    def test_real_swap_uses_three_free_blocks_and_leaves_correct_blue_alone(self):
        start = {"A": "center", "B": "left", "C": "right"}
        goal = {"A": ["left"], "B": ["center"], "C": ["right"]}
        with NativeTable(os.environ["DVIDIA_TIDY_NATIVE_ASSETS"], start) as scene:
            self.assertEqual(scene.observe(), start)
            self.assertFalse(scene.score(goal)["success"])
            with self.assertRaisesRegex(ValueError, "vacant"):
                scene.move("A", "left")
            self.assertEqual(scene.control_steps, 0)
            for obj, slot in [("A", "buffer"), ("B", "center"), ("A", "left")]:
                scene.observe()
                receipt = scene.move(obj, slot)
                self.assertTrue(receipt["verified"], receipt)
                self.assertGreater(receipt["first_motion_ms"], 0)
                self.assertGreaterEqual(receipt["bilateral_lift_dwell_s"], .1)
            self.assertEqual(scene.observe(), {"A":"left","B":"center","C":"right"})
            score = scene.score(goal)
            self.assertTrue(score["success"], score)
            self.assertLess(score["initially_correct_max_displacement_m"]["C"], .001)

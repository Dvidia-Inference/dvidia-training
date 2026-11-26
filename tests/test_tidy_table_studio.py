"""Remembered table receipts bind independent Terminal and Studio runs to the same reference."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import ANY, patch

from dvidia_training import tidy_table_studio as studio
from dvidia_training.tidy_table import HOME, remember_table


class TableReceipts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.assets = self.root / "assets"
        self.assets.mkdir()

    def capture(self, layout, **kwargs):
        return {"observation": dict(layout), "reference": remember_table(layout, kwargs["interchangeable"])}

    def test_terminal_and_studio_use_the_same_reference_and_keep_history(self):
        with patch("dvidia_training.tidy_table.capture_table", self.capture):
            with redirect_stdout(StringIO()) as output:
                self.assertEqual(studio.main(["remember", "--assets", str(self.assets), "--directory", str(self.root),
                                              "--exchange-ab"]), 0)
            first = json.loads(output.getvalue())
            self.assertEqual(first, studio.table_state(self.root)["reference"])
            result = {"success": True, "status": "tidy", "moves": [], "time_to_first_move_ms": None}
            with patch("dvidia_training.tidy_table.run_table_case", return_value=result) as simulate:
                run = studio.table_worker("tidy", self.root, self.assets, HOME, False, first["id"])
                simulate.assert_called_once_with(first["capture"]["reference"], HOME, assets=self.assets, frame_sink=ANY)
            self.assertEqual(first["id"], run["reference_id"])
            self.assertEqual(studio.table_state(self.root)["run"], run)
            second = studio.table_worker("remember", self.root, self.assets, HOME, False)
            self.assertNotEqual(first["id"], second["id"])
            self.assertIsNone(studio.table_state(self.root)["run"])
            self.assertTrue((self.root / "tables" / "runs" / (run["id"] + ".json")).is_file())
            with self.assertRaisesRegex(ValueError, "current remembered table"):
                studio.table_worker("tidy", self.root, self.assets, HOME, False, first["id"])

    def test_rejected_capture_keeps_previous_reference(self):
        with patch("dvidia_training.tidy_table.capture_table", self.capture):
            original = studio.table_worker("remember", self.root, self.assets, HOME, False)
        with patch("dvidia_training.tidy_table.capture_table", side_effect=ValueError("missing camera reading")):
            with self.assertRaisesRegex(ValueError, "missing camera"):
                studio.table_worker("remember", self.root, self.assets, HOME, False)
        self.assertEqual(studio.table_state(self.root)["reference"], original)

    def test_conflicting_zones_rejected_before_native_setup(self):
        with self.assertRaisesRegex(ValueError, "different zone"):
            studio.table_worker("remember", self.root, self.assets, {**HOME, "B": "left"}, False)
        with self.assertRaisesRegex(ValueError, "buffer free"):
            studio.table_worker("remember", self.root, self.assets, {**HOME, "A": "buffer"}, False)

    def test_unsuccessful_simulation_is_retained_and_cli_returns_failure(self):
        with patch("dvidia_training.tidy_table.capture_table", self.capture):
            studio.table_worker("remember", self.root, self.assets, HOME, False)
        result = {"success": False, "status": "failed", "reason": "fixture move failed"}
        with patch("dvidia_training.tidy_table.run_table_case", return_value=result), redirect_stdout(StringIO()):
            self.assertEqual(studio.main(["tidy", "--assets", str(self.assets), "--directory", str(self.root)]), 1)
        self.assertEqual(studio.table_state(self.root)["run"]["result"], result)
